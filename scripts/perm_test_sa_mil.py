"""Permutation test for neural SA-MIL (label permutation, full LOOCV re-run).

Protocol identical to loocv_baseline.permutation_p, but the null distribution
re-trains the SAMIL network per permutation instead of ridge:
  - observed: one full 43-fold LOOCV run (seed 42) -> pooled rho
  - null: n_perm label permutations; each re-runs the same LOOCV with the SAME
    torch seed (fixed model init + token sampling), so obs vs null differ only
    in the label assignment (paired design; seed noise is covered separately
    by the 5-seed ablation matrix)
  - p = (1 + #(null >= obs)) / (1 + n_perm)

Heavy: each permutation = 43 folds x 50 epochs x 42 bags. Designed for the lab
RTX 5090. Checkpoint/resume: the null distribution is appended to
outputs/{stem}_perm_null.csv after every permutation and reused on restart,
so an interrupted run continues instead of restarting from zero.

Usage (5090):
    python scripts/perm_test_sa_mil.py                          # Ab-1, 200 perms
    python scripts/perm_test_sa_mil.py --n-perm 1000            # final run
    python scripts/perm_test_sa_mil.py --no-stable              # Ab-2
    python scripts/perm_test_sa_mil.py --smoke                  # 3 perms, sanity
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from train_sa_mil_loocv import SEED, loocv_samil, bootstrap_ci, spearman

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"


def build_tag(cfg: dict, out: str) -> str:
    if out:
        return out
    parts = [f"sa_mil_a{cfg['n_attr']}",
             "stable" if cfg["use_stable"] else "nostab",
             "res" if cfg["use_residual"] else "nores"]
    return "_".join(parts)


def load_bags_and_labels(feat_dir: Path):
    labels_df = pd.read_csv(PROJECT / "outputs" / "sample_manifest.csv") \
        .set_index(["dataset", "sample"])["senescence_score"]
    keys, bags = [], []
    for bag in sorted(feat_dir.glob("*.npy")):
        dataset, sample = bag.stem.split("__")
        key = (dataset, sample)
        assert key in labels_df.index, bag.stem
        keys.append(key)
        bags.append(np.load(bag).astype(np.float32))
    labels = labels_df.reindex(pd.MultiIndex.from_tuples(keys)).to_numpy(dtype=float)
    return bags, labels


def run() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=str, default="dinov2_vitb14_warped")
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-attr", type=int, default=2, choices=(1, 2))
    ap.add_argument("--no-stable", action="store_true")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--smoke", action="store_true", help="3 permutations, local sanity check")
    args = ap.parse_args()

    cfg = {"n_attr": args.n_attr, "use_stable": not args.no_stable,
           "use_residual": not args.no_residual, "tau": None}
    n_perm = 3 if args.smoke else args.n_perm
    tag = build_tag(cfg, args.out)
    null_csv = OUT / f"{tag}_perm_null.csv"
    null_csv.parent.mkdir(exist_ok=True)

    feat_dir = PROJECT / "outputs" / "features" / args.features
    bags, labels = load_bags_and_labels(feat_dir)
    logger.info("bags=%d feat_dim=%d cfg=%s n_perm=%d", len(bags), bags[0].shape[1], cfg, n_perm)

    # observed rho (fixed seed; identical protocol to the ablation matrix seed runs)
    t0 = time.time()
    obs_folds = loocv_samil(bags, labels, cfg, seed=SEED)
    obs_rho = float(spearman(np.array([f["y_true"] for f in obs_folds]),
                             np.array([f["y_pred"] for f in obs_folds])))
    logger.info("observed rho = %.4f  (%.1f min per full LOOCV run)",
                obs_rho, (time.time() - t0) / 60)

    # resume: reuse completed permutations from a previous interrupted run.
    # null_csv is headerless (one float per line) so it must be read with
    # header=None, not by column name.
    done = []
    if null_csv.exists():
        done = pd.read_csv(null_csv, header=None, names=["rho"])["rho"].tolist()
        logger.info("resume: %d/%d permutations already done", len(done), n_perm)

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(labels) for _ in range(n_perm)]
    null = list(done[:n_perm])
    with open(null_csv, "a") as fh:
        for k in range(len(null), n_perm):
            tk = time.time()
            folds = loocv_samil(bags, perms[k], cfg, seed=SEED)
            rho_k = float(spearman(perms[k], np.array([f["y_pred"] for f in folds])))
            null.append(rho_k)
            fh.write(f"{rho_k}\n")
            fh.flush()
            elapsed, eta = (time.time() - t0) / 60, (time.time() - t0) / 60 / (k + 1) * (n_perm - k - 1)
            logger.info("perm %d/%d: rho=%.4f  elapsed %.0fmin  eta %.0fmin",
                        k + 1, n_perm, rho_k, elapsed, eta)

    null_arr = np.asarray(null)
    p_val = float((1 + (null_arr >= obs_rho).sum()) / (1 + len(null_arr)))
    summary = {"tag": tag, "cfg": cfg, "seed": SEED, "n_perm": len(null_arr),
               "obs_rho": obs_rho, "perm_p": p_val,
               "null_mean": float(null_arr.mean()), "null_max": float(null_arr.max()),
               "null_p95": float(np.percentile(null_arr, 95))}
    with open(OUT / f"{tag}_perm_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\n=== SA-MIL permutation test (neural, full LOOCV re-run) ===")
    print(f"  cfg: {tag}   torch seed {SEED} (fixed), {len(null_arr)} label permutations")
    print(f"  observed rho: {obs_rho:+.4f}")
    print(f"  null: mean {null_arr.mean():+.4f}  p95 {np.percentile(null_arr, 95):+.4f}  "
          f"max {null_arr.max():+.4f}")
    print(f"  permutation p = {p_val:.4f}")
    print(f"  outputs: {null_csv.name}, {tag}_perm_summary.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
