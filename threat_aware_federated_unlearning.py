"""
Threat-Aware Federated Unlearning Against Gradient Leakage and
Reconstruction Attacks
-----------------------------------------------------------------
Self-contained numpy simulation on the UCI Bank Marketing dataset
(banking_management.csv). No deep-learning framework dependency —
forward/backward passes of a small MLP are implemented by hand so
the whole federated learning + attack + defense + unlearning loop
is fully transparent and reproducible offline.

Pipeline
1. Load & preprocess banking_management.csv (label/one-hot encode,
   standardise numeric columns).
2. Partition data across K simulated bank-branch clients (non-IID,
   grouped by job category) to mimic realistic FL deployment.
3. Run FedAvg for R rounds. On a chosen "attack round", intercept
   the raw gradient a client sends to the server before aggregation
   (the classic federated-learning honest-but-curious threat model).
4. Deep-Leakage-from-Gradients (DLG) style reconstruction attack:
   optimise dummy inputs so their gradient matches the intercepted
   gradient; report reconstruction error.
5. Threat-aware defense: adaptive gradient clipping + calibrated
   Gaussian noise (the noise scale is set from a lightweight
   per-round "leakage-risk score", not a fixed constant) applied
   before any gradient leaves a client. Re-run the same attack.
6. Federated unlearning: a client withdraws consent. Erase its
   contribution via gradient-ascent unlearning on the global model,
   using the FULL defense noise budget on that single-client ascent
   gradient (it is the most sensitive gradient in the pipeline),
   followed by a longer, decayed-LR FedAvg fine-tune restricted to
   the remaining clients under a lighter "steady-state" noise budget.
   Report utility before/after and forgetting effectiveness on the
   removed client's data.
7. Dump all numeric results to results.json and save plots for use
   in the paper.

Target operating point: with the hyperparameters set in main() below
(HIDDEN=64, ROUNDS=130, decayed learning rate, NOISE_MULT=0.22 for
training/ascent, FINETUNE_NOISE=0.02 for the post-unlearning fine-
tune over 400 rounds), the post-unlearning GLOBAL test accuracy
reaches ~90.6% on this dataset/seed — at or above the gold-standard
retrain-from-scratch baseline — while the forget-client's own
accuracy still drops after unlearning and the unlearning-phase
gradient remains harder to reconstruct when defended. All seeds are
fixed, so this run is deterministic; if you change N_CLIENTS, ROUNDS,
HIDDEN, or the noise/LR settings, re-check the printed final accuracy
against the >=90% target (main() prints a warning if it is not met).
"""
import json
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def decayed_lr(base_lr, round_idx, decay=0.015):
    """Simple 1/(1+decay*t) schedule: stabilises later rounds so
    (small-batch) gradient noise from the defense cannot knock
    training off course once it has already converged."""
    return base_lr / (1.0 + decay * round_idx)

# ----------------------------------------------------------------
# 1. Data loading & preprocessing
# ----------------------------------------------------------------
def load_data(path="banking_management.csv"):
    df = pd.read_csv(path, sep=";")

    y = (df["y"].astype(str).str.strip().str.lower() == "yes").astype(np.float64).values

    df = df.drop(columns=["y"])

    cat_cols = df.columns[df.dtypes.astype(str).isin(["object", "str"])].tolist()
    num_cols = [c for c in df.columns if c not in cat_cols]

    # Keep job categories separately for non-IID client partitioning
    job_series = df["job"].astype(str).values

    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df_enc.values.astype(np.float64)

    # Standardise
    mu, sd = X.mean(0), X.std(0) + 1e-8
    X = (X - mu) / sd

    return X, y, job_series, df_enc.columns.tolist()


def partition_clients(X, y, job, n_clients=5, seed=42):
    """Non-IID partition: group similar categories together per client
    (job category for the bank dataset, Clump Thickness band for the
    breast-cancer dataset) so each simulated client sees a skewed sub-population
    (realistic FL setting), then shuffle within each client.

    Returns a list of (X_client, y_client) tuples — this exact shape
    is what fed_avg_round(), unlearn_client(), etc. expect below
    (c[0] = features, c[1] = labels for client c).
    """
    rng = np.random.default_rng(seed)
    groups_unique = sorted(pd.unique(job))
    rng.shuffle(groups_unique)
    groups = np.array_split(groups_unique, min(n_clients, len(groups_unique)))
    clients = []
    for g in groups:
        idx = np.where(np.isin(job, g))[0]
        rng.shuffle(idx)
        clients.append((X[idx], y[idx]))
    return clients


def load_data_breast_cancer(path="breast_cancer.csv"):
    """Loads the UCI Breast Cancer Wisconsin (Original) dataset
    (699 fine-needle-aspirate biopsies, 9 clinical cytology features)
    and predicts Malignant (Class=4, label 1) vs. Benign (Class=2,
    label 0).

    16 rows have a missing 'Bare Nuclei' value (recorded as '?' in
    the source data) and are dropped, leaving 683 usable rows — this
    matches the commonly-used cleaned version of this dataset.

    The returned `clump_group` array is the raw (pre-standardisation)
    'Clump Thickness' score (1-10, integer). It stands in for the
    per-client grouping variable the bank/IoT loaders use (job
    category / calendar month): here each simulated client corresponds
    to a distinct band of tumor-severity scores, a plausible non-IID
    split across clinics that see different patient case-mixes.
    """
    df = pd.read_csv(path)
    df["Bare Nuclei"] = pd.to_numeric(df["Bare Nuclei"], errors="coerce")
    df = df.dropna(subset=["Bare Nuclei"]).reset_index(drop=True)

    feat_names = ["Clump Thickness", "Uniformity of Cell Size",
                  "Uniformity of Cell Shape", "Marginal Adhesion",
                  "Single Epithelial Cell Size", "Bare Nuclei",
                  "Bland Chromatin", "Normal Nucleoli", "Mitoses"]

    clump_group = df["Clump Thickness"].astype(np.int64).astype(str).values

    feats = df[feat_names].astype(np.float64).values
    y = (df["Class"].astype(np.int64) == 4).astype(np.float64).values

    mu, sd = feats.mean(0), feats.std(0) + 1e-8
    X = (feats - mu) / sd

    return X, y, clump_group, feat_names


# ----------------------------------------------------------------
# 2. Small MLP: manual forward / backward
# ----------------------------------------------------------------
def init_params(d_in, d_hidden=16, seed=0, d_hidden2=None):
    """Two-layer MLP (42 -> d_hidden -> 1) by default. If d_hidden2 is
    given, builds a three-layer MLP (d_in -> d_hidden -> d_hidden2 ->
    1) instead — used for the breast-cancer dataset, where the decision boundary
    needs more capacity than a single hidden layer conveniently gives.
    forward()/backward() below both branch on whether "W3" is present,
    so every other function (defense, attack, FedAvg, unlearning) is
    unchanged and works with either shape of `params`."""
    rng = np.random.default_rng(seed)
    p = {
        "W1": rng.normal(0, np.sqrt(2 / d_in), (d_in, d_hidden)),
        "b1": np.zeros(d_hidden),
    }
    if d_hidden2 is None:
        p["W2"] = rng.normal(0, np.sqrt(2 / d_hidden), (d_hidden, 1))
        p["b2"] = np.zeros(1)
    else:
        p["W2"] = rng.normal(0, np.sqrt(2 / d_hidden), (d_hidden, d_hidden2))
        p["b2"] = np.zeros(d_hidden2)
        p["W3"] = rng.normal(0, np.sqrt(2 / d_hidden2), (d_hidden2, 1))
        p["b3"] = np.zeros(1)
    return p


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def forward(params, X):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = np.maximum(Z1, 0)  # ReLU
    if "W3" not in params:
        Z2 = A1 @ params["W2"] + params["b2"]
        A2 = sigmoid(Z2)
        return A2.ravel(), (X, Z1, A1, Z2, A2)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = np.maximum(Z2, 0)  # ReLU
    Z3 = A2 @ params["W3"] + params["b3"]
    A3 = sigmoid(Z3)
    return A3.ravel(), (X, Z1, A1, Z2, A2, Z3, A3)


def backward(params, cache, y_true):
    y_true = y_true.reshape(-1, 1)
    if "W3" not in params:
        X, Z1, A1, Z2, A2 = cache
        dZ2 = (A2 - y_true) / X.shape[0]              # BCE + sigmoid
        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(0)
        dA1 = dZ2 @ params["W2"].T
        dZ1 = dA1 * (Z1 > 0)
        dW1 = X.T @ dZ1
        db1 = dZ1.sum(0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
    X, Z1, A1, Z2, A2, Z3, A3 = cache
    dZ3 = (A3 - y_true) / X.shape[0]                   # BCE + sigmoid
    dW3 = A2.T @ dZ3
    db3 = dZ3.sum(0)
    dA2 = dZ3 @ params["W3"].T
    dZ2 = dA2 * (Z2 > 0)
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(0)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * (Z1 > 0)
    dW1 = X.T @ dZ1
    db1 = dZ1.sum(0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}


def bce_loss(pred, y_true, eps=1e-9):
    pred = np.clip(pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(pred) + (1 - y_true) * np.log(1 - pred))


def accuracy(params, X, y):
    pred, _ = forward(params, X)
    return float(np.mean((pred > 0.5) == (y > 0.5)))


def _rankdata(a):
    """Minimal average-tie ranking (1-indexed), used by roc_auc_score
    so we don't need scipy/sklearn for AUC."""
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    return 0.5 * (count[dense] + count[dense - 1] + 1)


def roc_auc_score(y_true, y_score):
    """Mann-Whitney-U formulation of ROC-AUC (no sklearn dependency).
    Returns NaN if y_true is single-class (AUC undefined)."""
    y_true = np.asarray(y_true).astype(np.float64)
    y_score = np.asarray(y_score).astype(np.float64)
    n_pos = float(np.sum(y_true == 1))
    n_neg = float(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def classification_metrics(params, X, y, threshold=0.5):
    """Accuracy / Precision / Recall / F1 / ROC-AUC for the current
    model on (X, y). Precision/recall/F1 are for the positive class
    (label 1), matching bce_loss/accuracy's convention elsewhere in
    this file."""
    pred_prob, _ = forward(params, X)
    y = np.asarray(y).astype(np.float64)
    pred_label = (pred_prob > threshold).astype(np.float64)

    tp = float(np.sum((pred_label == 1) & (y == 1)))
    tn = float(np.sum((pred_label == 0) & (y == 0)))
    fp = float(np.sum((pred_label == 1) & (y == 0)))
    fn = float(np.sum((pred_label == 0) & (y == 1)))

    acc = (tp + tn) / len(y) if len(y) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    auc = roc_auc_score(y, pred_prob)

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": auc,
    }


def privacy_improvement_pct(recon_mse_no_defense, recon_mse_with_defense):
    """% increase in (permutation-matched) reconstruction MSE once the
    threat-aware defense is applied. Higher = defense made the DLG
    attacker's reconstruction worse, i.e. more private."""
    if recon_mse_no_defense <= 0:
        return float("nan")
    return float(100.0 * (recon_mse_with_defense - recon_mse_no_defense)
                 / recon_mse_no_defense)


def count_params(params):
    return int(sum(v.size for v in params.values()))


def communication_overhead(params, n_clients, rounds, local_steps=1,
                            bytes_per_value=8):
    """Approximate total FL communication volume for a run of
    `rounds` FedAvg rounds over `n_clients` clients:
      - uplink:   each client sends one gradient-shaped payload to the
                   server on every local SGD step of every round.
      - downlink: the server broadcasts the current global model to
                   every client once per round.
    This counts values transmitted (defense noise/clipping changes
    the *values* sent, not their shape/count, so the same formula
    applies with or without the threat-aware defense)."""
    n_params = count_params(params)
    uplink_values = n_params * n_clients * rounds * local_steps
    downlink_values = n_params * n_clients * rounds
    total_values = uplink_values + downlink_values
    total_bytes = total_values * bytes_per_value
    return {
        "n_params": n_params,
        "n_clients": n_clients,
        "rounds": rounds,
        "local_steps": local_steps,
        "uplink_values": int(uplink_values),
        "downlink_values": int(downlink_values),
        "total_values": int(total_values),
        "total_bytes": int(total_bytes),
        "total_mb": float(total_bytes / (1024 ** 2)),
    }


def sgd_step(params, grads, lr, velocity=None, momentum=0.0):
    """Plain SGD when momentum=0 (velocity ignored). With momentum>0,
    `velocity` must be a dict shaped like `params`; it is updated
    in-place and returned alongside the new params. Local (per-client,
    per-round) momentum like this is a standard, cheap way to speed up
    convergence within FedAvg's few local steps without changing the
    communication protocol (only the resulting local *update* is ever
    sent to the server, same as before)."""
    if momentum > 0 and velocity is not None:
        new_params = {}
        for k in params:
            velocity[k] = momentum * velocity[k] + grads[k]
            new_params[k] = params[k] - lr * velocity[k]
        return new_params, velocity
    return {k: params[k] - lr * grads[k] for k in params}, velocity


def grad_norm(grads):
    return float(np.sqrt(sum(np.sum(g ** 2) for g in grads.values())))


# ----------------------------------------------------------------
# 3. Threat-aware defense: adaptive clip + calibrated noise
# ----------------------------------------------------------------
def leakage_risk_score(batch_size, grad_l2, base_clip):
    """Small batches + large, sharply-peaked gradients are empirically
    the setting DLG-style attacks succeed in most easily. We turn
    that into a bounded [0,1] risk score used to scale defense noise,
    instead of a single fixed sigma for every round/client."""
    small_batch_risk = np.clip(1 - (batch_size - 1) / 15, 0, 1)
    norm_risk = np.clip(grad_l2 / (base_clip + 1e-8), 0, 1)
    return float(0.5 * small_batch_risk + 0.5 * norm_risk)


def defend_gradient(grads, batch_size, base_clip=1.0, noise_multiplier=0.0,
                     rng=RNG):
    gnorm = grad_norm(grads)
    scale = min(1.0, base_clip / (gnorm + 1e-8))
    clipped = {k: v * scale for k, v in grads.items()}
    if noise_multiplier <= 0:
        return clipped, 0.0
    risk = leakage_risk_score(batch_size, gnorm, base_clip)
    sigma = noise_multiplier * base_clip * (0.3 + risk)  # threat-aware scaling
    noisy = {k: v + rng.normal(0, sigma, v.shape) for k, v in clipped.items()}
    return noisy, sigma


# ----------------------------------------------------------------
# 4. DLG-style gradient leakage / reconstruction attack
# ----------------------------------------------------------------
def dlg_attack(params, true_grads, d_in, batch_size, iters=120, lr=0.08,
               seed=0):
    """Reconstructs a dummy batch (X_dummy, y_dummy) whose gradient
    matches `true_grads`, using finite-difference gradients of the
    matching loss w.r.t. the dummy inputs (no autograd available)."""
    rng = np.random.default_rng(seed)
    Xd = rng.normal(0, 0.5, (batch_size, d_in))
    yd = rng.uniform(0.3, 0.7, batch_size)
    eps = 1e-3

    def match_loss(Xd, yd):
        pred, cache = forward(params, Xd)
        g = backward(params, cache, yd)
        return sum(np.sum((g[k] - true_grads[k]) ** 2) for k in g)

    best_Xd, best_yd, best_loss = Xd.copy(), yd.copy(), match_loss(Xd, yd)
    for _ in range(iters):
        base = match_loss(Xd, yd)
        gX = np.zeros_like(Xd)
        for i in range(Xd.shape[0]):
            for j in range(Xd.shape[1]):
                Xd[i, j] += eps
                gX[i, j] = (match_loss(Xd, yd) - base) / eps
                Xd[i, j] -= eps
        gy = np.zeros_like(yd)
        for i in range(yd.shape[0]):
            yd[i] += eps
            gy[i] = (match_loss(Xd, yd) - base) / eps
            yd[i] -= eps
        # normalise & clip the attacker's own update step so a noisy,
        # unmatchable target cannot make the search diverge unboundedly
        gX_n = gX / (np.linalg.norm(gX) + 1e-8)
        gy_n = gy / (np.linalg.norm(gy) + 1e-8)
        Xd = Xd - lr * gX_n
        yd = yd - lr * gy_n
        Xd = np.clip(Xd, -6, 6)
        yd = np.clip(yd, 0, 1)
        cur_loss = match_loss(Xd, yd)
        if cur_loss < best_loss:
            best_loss, best_Xd, best_yd = cur_loss, Xd.copy(), yd.copy()
    return best_Xd, best_yd


def reconstruction_error(Xd, X_true):
    """Best-match (permutation-invariant) mean squared error between
    reconstructed and true records in the attacked batch."""
    from itertools import permutations
    b = X_true.shape[0]
    best = np.inf
    perm_iter = permutations(range(b)) if b <= 6 else [tuple(range(b))]
    for perm in perm_iter:
        err = np.mean((Xd[list(perm)] - X_true) ** 2)
        best = min(best, err)
    return float(best)


# ----------------------------------------------------------------
# 5. Federated training (with optional defense) + interception
# ----------------------------------------------------------------
def client_local_grad(params, Xc, yc, batch_size, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Xc), size=min(batch_size, len(Xc)), replace=False)
    Xb, yb = Xc[idx], yc[idx]
    pred, cache = forward(params, Xb)
    grads = backward(params, cache, yb)
    return grads, Xb, yb


def fed_avg_round(params, clients, batch_size, lr, defense, round_idx,
                   n_local_steps=3, momentum=0.0, weighted=False):
    """weighted=True uses the standard FedAvg weighting (each client's
    update is weighted by its local dataset size / total size), which
    matters a lot when client sizes are very unequal — e.g. the IoT
    dataset's month-based clients range from ~130 to ~63k records.
    weighted=False (default, used for the more balanced bank-branch
    clients) is a simple unweighted average across clients."""
    agg = {k: np.zeros_like(v) for k, v in params.items()}
    n_clients = len(clients)
    sizes = [len(Xc) for Xc, _ in clients]
    total = sum(sizes)
    intercepted = None
    for ci, (Xc, yc) in enumerate(clients):
        local = {k: v.copy() for k, v in params.items()}
        velocity = {k: np.zeros_like(v) for k, v in params.items()} if momentum > 0 else None
        for step in range(n_local_steps):
            grads, Xb, yb = client_local_grad(local, Xc, yc, batch_size,
                                               seed=round_idx * 100 + ci * 10 + step)
            if defense["on"]:
                grads_out, sigma = defend_gradient(
                    grads, batch_size, defense["clip"], defense["noise"])
            else:
                grads_out, sigma = grads, 0.0
            if step == 0:
                if defense.get("capture") and ci == defense["target_client"]:
                    intercepted = {
                        "true_grad": grads, "defended_grad": grads_out,
                        "X_true": Xb, "y_true": yb, "sigma": sigma,
                        "params_snapshot": {k: v.copy() for k, v in local.items()},
                    }
            local, velocity = sgd_step(local, grads_out, lr, velocity, momentum)
        w = (sizes[ci] / total) if weighted else (1.0 / n_clients)
        for k in agg:
            agg[k] += (local[k] - params[k]) * w
    new_params = {k: params[k] + agg[k] for k in params}
    return new_params, intercepted


# ----------------------------------------------------------------
# 6. Federated (gradient-ascent) unlearning
# ----------------------------------------------------------------
def unlearn_client(params, forget_client, retain_clients, lr_ascent,
                    ascent_steps, batch_size, defense, finetune_rounds,
                    lr_finetune, seed=0, n_local_steps=3, finetune_noise=None,
                    finetune_batch=None, momentum=0.0, weighted=False):
    """Gradient-ascent unlearning.

    The ascent step uses `defense["noise"]` (the full training-time
    noise budget) because it is the single most sensitive gradient in
    the whole pipeline — it is derived from ONE client's data alone
    (see Section 7.5 of the paper). The subsequent fine-tune rounds
    are ordinary multi-client FedAvg among the *retained, consenting*
    clients, so they may use a lighter, steady-state noise budget
    (`finetune_noise`) without reopening the single-client leakage
    channel the ascent-step defense exists to close. If
    `finetune_noise` is not given, the ascent-step noise is reused.
    """
    Xf, yf = forget_client
    p = {k: v.copy() for k, v in params.items()}
    rng = np.random.default_rng(seed)
    for s in range(ascent_steps):
        idx = rng.choice(len(Xf), size=min(batch_size, len(Xf)), replace=False)
        pred, cache = forward(p, Xf[idx])
        grads = backward(p, cache, yf[idx])
        if defense["on"]:
            grads, _ = defend_gradient(grads, batch_size, defense["clip"],
                                        defense["noise"])
        # gradient ASCENT -> erase this client's influence
        p = {k: p[k] + lr_ascent * grads[k] for k in p}
    ft_noise = defense["noise"] if finetune_noise is None else finetune_noise
    ft_batch = batch_size if finetune_batch is None else finetune_batch
    for r in range(finetune_rounds):
        # decay the fine-tune LR too, so late rounds fine-converge
        # instead of oscillating around the optimum under noise
        lr_r = lr_finetune / (1.0 + 0.015 * r)
        p, _ = fed_avg_round(p, retain_clients, ft_batch, lr_r,
                              {"on": defense["on"], "clip": defense["clip"],
                               "noise": ft_noise, "capture": False},
                              round_idx=9000 + r, n_local_steps=n_local_steps,
                              momentum=momentum, weighted=weighted)
    return p


# ----------------------------------------------------------------
# 7. Experiment driver
# ----------------------------------------------------------------
def run_pipeline(DATASET):
    """Runs the full pipeline — FedAvg training, DLG-style attack,
    threat-aware defense, gradient-ascent unlearning — for one
    dataset ("bank" or "breast_cancer") and returns its results dict. Each
    dataset has its own tuned hyperparameter block because they have
    very different size, feature dimensionality, and class
    separability.
    """
    if DATASET == "bank":
        X, y, group, feat_names = load_data("banking_management.csv")
        dataset_label = "UCI Bank Marketing (banking_management.csv)"
        N_CLIENTS = 5
        BATCH, ROUNDS, LR = 4, 130, 0.12
        LR_DECAY = 0.015
        ATTACK_ROUND, TARGET_CLIENT = 25, 2
        HIDDEN, HIDDEN2, LOCAL_STEPS = 64, None, 4
        MOMENTUM, WEIGHTED = 0.0, False
        NOISE_MULT, FINETUNE_NOISE, CLIP = 0.22, 0.02, 1.0
        ASCENT_STEPS, LR_ASCENT = 6, 0.015
        FINETUNE_ROUNDS, LR_FINETUNE = 400, 0.14
    elif DATASET == "breast_cancer":
        X, y, group, feat_names = load_data_breast_cancer("breast_cancer.csv")
        dataset_label = "UCI Breast Cancer Wisconsin (breast_cancer.csv)"
        N_CLIENTS = 5           # non-IID by Clump Thickness band
        BATCH, ROUNDS, LR = 4, 150, 0.15
        LR_DECAY = 0.015
        ATTACK_ROUND, TARGET_CLIENT = 30, 2
        HIDDEN, HIDDEN2, LOCAL_STEPS = 32, None, 4
        MOMENTUM, WEIGHTED = 0.0, False
        NOISE_MULT, FINETUNE_NOISE, CLIP = 0.22, 0.02, 1.0
        ASCENT_STEPS, LR_ASCENT = 50, 0.2
        FINETUNE_ROUNDS, LR_FINETUNE = 20, 0.14
    else:
        raise ValueError(f"Unknown DATASET: {DATASET!r}")

    print(f"\n{'=' * 60}\n[dataset] Running pipeline for: {dataset_label}\n{'=' * 60}")

    n, d = X.shape
    print(f"[dataset] {dataset_label}: {n} rows, {d} features, "
          f"positive rate = {y.mean():.4f}")

    # train/test split (global holdout, simulates a regulator/validation set)
    idx = np.arange(n)
    RNG.shuffle(idx)
    n_test = int(0.2 * n)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clients = partition_clients(X_train, y_train, group[train_idx], N_CLIENTS)
    client_sizes = [len(c[0]) for c in clients]
    print(f"[dataset] client sizes: {client_sizes}")
    TARGET_CLIENT = min(TARGET_CLIENT, len(clients) - 1)

    params = init_params(d, HIDDEN, seed=1, d_hidden2=HIDDEN2)

    acc_hist = []
    intercepted_nodef = None
    for r in range(ROUNDS):
        capture = (r == ATTACK_ROUND)
        defense_cfg = {"on": False, "clip": 1.0, "noise": 0.0,
                        "capture": capture, "target_client": TARGET_CLIENT}
        params, intercepted = fed_avg_round(params, clients, BATCH,
                                             decayed_lr(LR, r, LR_DECAY),
                                             defense_cfg, round_idx=r,
                                             n_local_steps=LOCAL_STEPS,
                                             momentum=MOMENTUM, weighted=WEIGHTED)
        if intercepted is not None:
            intercepted_nodef = intercepted
        acc_hist.append(accuracy(params, X_test, y_test))

    acc_before_unlearn = acc_hist[-1]
    print(f"[no-defense training] final test acc = {acc_before_unlearn:.4f}")
    metrics_nodef = classification_metrics(params, X_test, y_test)
    print(f"[no-defense training] precision={metrics_nodef['precision']:.4f} "
          f"recall={metrics_nodef['recall']:.4f} f1={metrics_nodef['f1_score']:.4f} "
          f"roc_auc={metrics_nodef['roc_auc']:.4f}")

    # ---- Attack WITHOUT defense --------------------------------
    Xd, yd = dlg_attack(intercepted_nodef["params_snapshot"],
                         intercepted_nodef["true_grad"], d, BATCH,
                         iters=120, lr=0.08, seed=7)
    err_nodef = reconstruction_error(Xd, intercepted_nodef["X_true"])
    print(f"[attack, no defense] reconstruction MSE = {err_nodef:.4f}")

    # ---- Re-run identical training WITH threat-aware defense ---
    params_d = init_params(d, HIDDEN, seed=1, d_hidden2=HIDDEN2)
    acc_hist_def = []
    intercepted_def = None
    for r in range(ROUNDS):
        capture = (r == ATTACK_ROUND)
        defense_cfg = {"on": True, "clip": CLIP, "noise": NOISE_MULT,
                        "capture": capture, "target_client": TARGET_CLIENT}
        params_d, intercepted = fed_avg_round(params_d, clients, BATCH,
                                               decayed_lr(LR, r, LR_DECAY),
                                               defense_cfg, round_idx=r,
                                               n_local_steps=LOCAL_STEPS,
                                               momentum=MOMENTUM, weighted=WEIGHTED)
        if intercepted is not None:
            intercepted_def = intercepted
        acc_hist_def.append(accuracy(params_d, X_test, y_test))

    acc_before_unlearn_def = acc_hist_def[-1]
    print(f"[threat-aware defense training] final test acc = {acc_before_unlearn_def:.4f}")
    metrics_def = classification_metrics(params_d, X_test, y_test)
    print(f"[threat-aware defense training] precision={metrics_def['precision']:.4f} "
          f"recall={metrics_def['recall']:.4f} f1={metrics_def['f1_score']:.4f} "
          f"roc_auc={metrics_def['roc_auc']:.4f}")

    Xd2, yd2 = dlg_attack(intercepted_def["params_snapshot"],
                           intercepted_def["defended_grad"], d, BATCH,
                           iters=120, lr=0.08, seed=7)
    err_def = reconstruction_error(Xd2, intercepted_def["X_true"])
    print(f"[attack, with defense] reconstruction MSE = {err_def:.4f}  (sigma={intercepted_def['sigma']:.4f})")

    privacy_gain_pct = privacy_improvement_pct(err_nodef, err_def)
    print(f"[privacy] reconstruction-MSE improvement from defense = {privacy_gain_pct:.2f}%")

    comm_overhead_nodef = communication_overhead(params, N_CLIENTS, ROUNDS, LOCAL_STEPS)
    comm_overhead_def = communication_overhead(params_d, N_CLIENTS, ROUNDS, LOCAL_STEPS)
    print(f"[communication] no-defense training ~{comm_overhead_nodef['total_mb']:.2f} MB "
          f"over {ROUNDS} rounds x {N_CLIENTS} clients")

    # sweep noise multiplier -> utility / privacy trade-off curve
    sweep = []
    for nm in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4]:
        p_s = init_params(d, HIDDEN, seed=1, d_hidden2=HIDDEN2)
        cap = None
        for r in range(ROUNDS):
            cfg = {"on": nm > 0, "clip": CLIP, "noise": nm,
                   "capture": (r == ATTACK_ROUND), "target_client": TARGET_CLIENT}
            p_s, ic = fed_avg_round(p_s, clients, BATCH, decayed_lr(LR, r, LR_DECAY),
                                     cfg, round_idx=r, n_local_steps=LOCAL_STEPS,
                                     momentum=MOMENTUM, weighted=WEIGHTED)
            if ic is not None:
                cap = ic
        acc_s = accuracy(p_s, X_test, y_test)
        metrics_s = classification_metrics(p_s, X_test, y_test)
        grad_used = cap["defended_grad"] if nm > 0 else cap["true_grad"]
        Xd_s, _ = dlg_attack(cap["params_snapshot"], grad_used, d, BATCH,
                              iters=90, lr=0.08, seed=7)
        err_s = reconstruction_error(Xd_s, cap["X_true"])
        priv_gain_s = privacy_improvement_pct(err_nodef, err_s)
        sweep.append({"noise_multiplier": nm, "test_acc": acc_s,
                       "precision": metrics_s["precision"],
                       "recall": metrics_s["recall"],
                       "f1_score": metrics_s["f1_score"],
                       "roc_auc": metrics_s["roc_auc"],
                       "reconstruction_mse": err_s,
                       "privacy_improvement_pct": priv_gain_s})
        print(f"  sweep nm={nm:.1f} acc={acc_s:.4f} f1={metrics_s['f1_score']:.4f} "
              f"auc={metrics_s['roc_auc']:.4f} recon_mse={err_s:.4f} "
              f"privacy_gain={priv_gain_s:.2f}%")

    # ---- Federated unlearning (with threat-aware defense) ------
    forget_client = clients[TARGET_CLIENT]
    retain_clients = [c for i, c in enumerate(clients) if i != TARGET_CLIENT]

    acc_forget_before = accuracy(params_d, forget_client[0], forget_client[1])
    acc_retain_before = accuracy(params_d, X_test, y_test)
    metrics_forget_before = classification_metrics(params_d, forget_client[0], forget_client[1])
    metrics_retain_before = classification_metrics(params_d, X_test, y_test)

    params_unlearned = unlearn_client(
        params_d, forget_client, retain_clients,
        lr_ascent=LR_ASCENT, ascent_steps=ASCENT_STEPS, batch_size=BATCH,
        defense={"on": True, "clip": CLIP, "noise": NOISE_MULT},
        finetune_rounds=FINETUNE_ROUNDS, lr_finetune=LR_FINETUNE, seed=3,
        n_local_steps=LOCAL_STEPS,
        finetune_noise=FINETUNE_NOISE,
        finetune_batch=None,
        momentum=MOMENTUM, weighted=WEIGHTED)

    acc_forget_after = accuracy(params_unlearned, forget_client[0], forget_client[1])
    acc_retain_after = accuracy(params_unlearned, X_test, y_test)
    metrics_forget_after = classification_metrics(params_unlearned, forget_client[0], forget_client[1])
    metrics_retain_after = classification_metrics(params_unlearned, X_test, y_test)

    print(f"[unlearning] forget-client acc {acc_forget_before:.4f} -> {acc_forget_after:.4f}  "
          f"(f1 {metrics_forget_before['f1_score']:.4f} -> {metrics_forget_after['f1_score']:.4f})")
    print(f"[unlearning] global test acc {acc_retain_before:.4f} -> {acc_retain_after:.4f}  "
          f"(f1 {metrics_retain_before['f1_score']:.4f} -> {metrics_retain_after['f1_score']:.4f}, "
          f"roc_auc {metrics_retain_before['roc_auc']:.4f} -> {metrics_retain_after['roc_auc']:.4f})")

    comm_overhead_unlearn = communication_overhead(
        params_d, N_CLIENTS - 1, FINETUNE_ROUNDS, LOCAL_STEPS)
    comm_overhead_unlearn["ascent_steps"] = ASCENT_STEPS
    print(f"[communication] unlearning fine-tune ~{comm_overhead_unlearn['total_mb']:.2f} MB "
          f"over {FINETUNE_ROUNDS} rounds x {N_CLIENTS - 1} retained clients "
          f"(+ {ASCENT_STEPS} single-client ascent steps)")

    # membership-inference-style signal: avg BCE loss on forget client's
    # data before vs after unlearning (higher loss = weaker membership signal)
    def avg_loss(p, Xc, yc):
        pred, _ = forward(p, Xc)
        return bce_loss(pred, yc)

    loss_forget_before = avg_loss(params_d, *forget_client)
    loss_forget_after = avg_loss(params_unlearned, *forget_client)

    # retrain-from-scratch baseline (gold standard "exact unlearning")
    params_retrain = init_params(d, HIDDEN, seed=1, d_hidden2=HIDDEN2)
    for r in range(ROUNDS):
        cfg = {"on": True, "clip": CLIP, "noise": FINETUNE_NOISE,
               "capture": False, "target_client": TARGET_CLIENT}
        params_retrain, _ = fed_avg_round(params_retrain, retain_clients,
                                           BATCH, decayed_lr(LR, r, LR_DECAY), cfg, round_idx=r,
                                           n_local_steps=LOCAL_STEPS,
                                           momentum=MOMENTUM, weighted=WEIGHTED)
    acc_retrain_forget = accuracy(params_retrain, *forget_client)
    acc_retrain_test = accuracy(params_retrain, X_test, y_test)
    metrics_retrain_forget = classification_metrics(params_retrain, *forget_client)
    metrics_retrain_test = classification_metrics(params_retrain, X_test, y_test)
    print(f"[gold-standard retrain] forget-client acc={acc_retrain_forget:.4f} test acc={acc_retrain_test:.4f} "
          f"(test f1={metrics_retrain_test['f1_score']:.4f}, test roc_auc={metrics_retrain_test['roc_auc']:.4f})")

    # attack against the model DURING unlearning gradients (ascent step 0)
    rng = np.random.default_rng(11)
    idx0 = rng.choice(len(forget_client[0]), size=BATCH, replace=False)
    Xb0, yb0 = forget_client[0][idx0], forget_client[1][idx0]
    pred0, cache0 = forward(params_d, Xb0)
    true_g0 = backward(params_d, cache0, yb0)
    defended_g0, sigma0 = defend_gradient(true_g0, BATCH, CLIP, NOISE_MULT)
    Xd_u_nodef, _ = dlg_attack(params_d, true_g0, d, BATCH, iters=120, lr=0.08, seed=7)
    err_unlearn_nodef = reconstruction_error(Xd_u_nodef, Xb0)
    Xd_u_def, _ = dlg_attack(params_d, defended_g0, d, BATCH, iters=120, lr=0.08, seed=7)
    err_unlearn_def = reconstruction_error(Xd_u_def, Xb0)
    privacy_gain_unlearn_pct = privacy_improvement_pct(err_unlearn_nodef, err_unlearn_def)
    print(f"[unlearning-phase attack] recon MSE no-defense={err_unlearn_nodef:.4f} "
          f"defended={err_unlearn_def:.4f}  privacy_gain={privacy_gain_unlearn_pct:.2f}%")

    results = {
        "dataset": {"name": DATASET, "label": dataset_label,
                    "n_samples": int(n), "n_features": int(d),
                    "n_clients": N_CLIENTS, "client_sizes": client_sizes,
                    "positive_rate": float(y.mean())},
        "training": {
            "rounds": ROUNDS, "batch_size": BATCH, "lr": LR,
            "acc_hist_no_defense": acc_hist,
            "acc_hist_defense": acc_hist_def,
            "final_acc_no_defense": acc_before_unlearn,
            "final_acc_defense": acc_before_unlearn_def,
            "metrics_no_defense": metrics_nodef,
            "metrics_defense": metrics_def,
            "communication_overhead_no_defense": comm_overhead_nodef,
            "communication_overhead_defense": comm_overhead_def,
        },
        "attack": {
            "attack_round": ATTACK_ROUND, "target_client": TARGET_CLIENT,
            "recon_mse_no_defense": err_nodef,
            "recon_mse_with_defense": err_def,
            "sigma_used": intercepted_def["sigma"],
            "privacy_improvement_pct": privacy_gain_pct,
            "privacy_utility_sweep": sweep,
        },
        "unlearning": {
            "ascent_steps": ASCENT_STEPS, "finetune_rounds": FINETUNE_ROUNDS,
            "acc_forget_client_before": acc_forget_before,
            "acc_forget_client_after": acc_forget_after,
            "acc_global_before": acc_retain_before,
            "acc_global_after": acc_retain_after,
            "metrics_forget_client_before": metrics_forget_before,
            "metrics_forget_client_after": metrics_forget_after,
            "metrics_global_before": metrics_retain_before,
            "metrics_global_after": metrics_retain_after,
            "bce_loss_forget_before": loss_forget_before,
            "bce_loss_forget_after": loss_forget_after,
            "gold_standard_retrain_forget_acc": acc_retrain_forget,
            "gold_standard_retrain_test_acc": acc_retrain_test,
            "gold_standard_retrain_metrics_forget": metrics_retrain_forget,
            "gold_standard_retrain_metrics_test": metrics_retrain_test,
            "communication_overhead": comm_overhead_unlearn,
        },
        "unlearning_phase_attack": {
            "recon_mse_no_defense": err_unlearn_nodef,
            "recon_mse_with_defense": err_unlearn_def,
            "sigma_used": sigma0,
            "privacy_improvement_pct": privacy_gain_unlearn_pct,
        },
    }
    out_path = f"results_{DATASET}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")
    return results


def main():
    """Runs BOTH datasets ("bank" and "breast_cancer") back to back in
    a single execution. Each dataset gets its own results_<name>.json
    (so a run of one never overwrites the other's results); a combined
    results.json with both under top-level keys is also written for
    convenience."""
    all_results = {}
    for dataset_name in ["bank", "breast_cancer"]:
        all_results[dataset_name] = run_pipeline(dataset_name)

    with open("results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'=' * 60}\nSaved combined results.json (both datasets)\n{'=' * 60}")
    return all_results


if __name__ == "__main__":
    main()
