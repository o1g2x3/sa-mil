"""P0-3b: 43-fold LOOCV baseline with mean-pooled frozen features.

Protocol (per README §4):
- per-sample bag = mean of per-image features
- per fold: standardize features and target with TRAIN-only stats,
  select ridge alpha by exact inner leave-one-out on the train fold
  (closed form over the Gram matrix), predict the held-out sample
- report pooled Spearman rho / R2 / MAE + bootstrap 95% CI over subjects
  + permutation test of rho (labels shuffled, same pipeline re-run)

Usage:
    python scripts/loocv_baseline.py --features outputs/features/resnet50_warped
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "outputs" / "sample_manifest.csv"
ALPHAS = np.logspace(-4, 4, 17)
PERMUTATIONS = 200
BOOTSTRAP = 2000
SEED = 42


# ---------- metrics ----------

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x), kind="mergesort")
    ry = np.argsort(np.argsort(y), kind="mergesort")
    n = len(x)
    r = 1 - 6 * ((rx - ry) ** 2).sum() / (n * (n * n - 1))
    return float(r)


def r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(1 - ((y - y_hat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def mae(y: np.ndarray, y_hat: np.ndarray) -> float:
    return float(np.abs(y - y_hat).mean())


# ---------- ridge with exact inner-LOOCV alpha selection ----------

def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray,
                y_tr_stats: tuple[float, float]) -> float:
    """Train-label-standardized ridge; returns prediction in RAW label units."""
    # input X already feature-standardized with train stats; y_tr in z units
    K = X_tr @ X_tr.T
    n = len(y_tr)
    best_alpha, best_score, M_alpha, h_alpha = None, np.inf, None, None
    for alpha in ALPHAS:
        gram = K + alpha * np.eye(n)
        M = np.linalg.solve(gram, K)          # (n,n), label-independent
        y_hat_full = M @ y_tr
        h = np.clip(np.diag(M), 0.0, 1.0 - 1e-6)
        r = (y_tr - y_hat_full) / (1.0 - h)
        score = float(np.mean(r ** 2))
        if score < best_score:
            best_score, best_alpha, M_alpha, h_alpha = score, alpha, M, h
    beta = np.linalg.solve(K + best_alpha * np.eye(n), y_tr)   # (n,)
    k_te = X_te @ X_tr.T                                        # (1,n)
    y_hat_std = float(np.asarray(k_te @ beta).item())
    mu, sd = y_tr_stats
    return y_hat_std * sd + mu                                  # back to raw units


def loocv_predictions(features: list[np.ndarray], labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One 43-fold LOOCV pass. Returns (y_true, y_pred) in raw label units."""
    n = len(features)
    y_true, y_pred = np.zeros(n), np.zeros(n)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        X = np.stack([features[j] for j in tr])
        X_te = features[i][None, :]
        mu, sd = labels[tr].mean(), labels[tr].std(ddof=0) or 1.0
        mu_f, sd_f = X.mean(axis=0), X.std(axis=0) + 1e-8
        X = (X - mu_f) / sd_f
        X_te = (X_te - mu_f) / sd_f                      # train-only feature stats
        y_tr = (labels[tr] - mu) / sd
        y_true[i] = labels[i]
        y_pred[i] = fit_predict(X, y_tr, X_te, (mu, sd))
    return y_true, y_pred


# ---------- bootstrap CI + permutation test ----------

def bootstrap_ci(y: np.ndarray, y_hat: np.ndarray, n_boot: int = BOOTSTRAP) -> dict[str, tuple[float, float, float]]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    out: dict[str, tuple[float, float, float]] = {}
    for name, f in [("rho", spearman), ("r2", r2), ("mae", mae)]:
        vals = np.array([f(y[i], y_hat[i]) for i in idx])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        out[name] = (float(f(y, y_hat)), float(lo), float(hi))
    return out


def permutation_p(features: list[np.ndarray], labels: np.ndarray, n_perm: int = PERMUTATIONS) -> tuple[float, np.ndarray]:
    """Null distribution of rho under label permutation (full pipeline re-run)."""
    rng = np.random.default_rng(SEED)
    obs_rho = spearman(*loocv_predictions(features, labels))
    null = np.empty(n_perm)
    for k in range(n_perm):
        y_perm = rng.permutation(labels)
        _, yp = loocv_predictions(features, y_perm)
        null[k] = spearman(y_perm, yp)
    p = float((1 + (null >= obs_rho).sum()) / (1 + n_perm))
    return p, null


# ---------- self-check ----------

def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 5))
    y = x[:, 0] - 2 * x[:, 1] + 0.01 * rng.normal(size=12)
    assert abs(spearman(y, y) - 1.0) < 1e-9
    assert abs(r2(y, y) - 1.0) < 1e-9
    assert mae(y, y) < 1e-9
    yt, yp = loocv_predictions([x[i] for i in range(12)], y)
    assert spearman(yt, yp) > 0.95, f"selfcheck rho={spearman(yt, yp):.3f}"
    logger.info("selfcheck passed (metrics + ridge LOOCV solver)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _selfcheck()

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True, help="dir of per-sample .npy bags")
    ap.add_argument("--pool", choices=["mean", "max"], default="mean", help="bag aggregation")
    ap.add_argument("--n-perm", type=int, default=PERMUTATIONS)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    feat_dir = Path(args.features)
    feat_dir = feat_dir if feat_dir.is_dir() else PROJECT / "outputs" / "features" / str(feat_dir)
    tag = args.out or f"{feat_dir.parent.name}_{feat_dir.name}_{args.pool}"
    tag = tag.replace("_warped", "")

    labels = pd.read_csv(MANIFEST).set_index(["dataset", "sample"])["senescence_score"]

    names, features = [], []
    for bag in sorted(feat_dir.glob("*.npy")):
        dataset, sample = bag.stem.split("__")
        key = (dataset, sample)
        if key not in labels.index:
            logger.warning("bag %s not in manifest labels; skipped", bag.stem)
            continue
        feats = np.load(bag)
        names.append(key)
        features.append(feats.max(axis=0) if args.pool == "max" else feats.mean(axis=0))
    y_long = labels.reindex(pd.MultiIndex.from_tuples(names)).to_numpy(dtype=float)

    y_true, y_pred = loocv_predictions(features, y_long)
    ci = bootstrap_ci(y_true, y_pred)
    p_val, _null = permutation_p(features, y_long, args.n_perm)

    pd.DataFrame({"dataset": [k[0] for k in names], "sample": [k[1] for k in names],
                  "y_true": y_true, "y_pred": y_pred}).to_csv(
        PROJECT / "outputs" / f"p0_3_{tag}.csv", index=False)

    print("\n=== LOOCV baseline (mean-pooled frozen features) ===")
    for name, (val, lo, hi) in ci.items():
        print(f"  {name:>4}: {val:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")
    print(f"  permutation p (n={args.n_perm}): {p_val:.4f}")
    print(f"  output: outputs/p0_3_{tag}.csv")

    summary = {"backbone": tag, "n_samples": len(y_true), "ci": ci, "perm_p": p_val}
    with open(PROJECT / "outputs" / f"p0_3_{tag}_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()