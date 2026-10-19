"""Diagnostics: is the LOOCV rho driven by genuine signal or a dataset shortcut?

Checks (all use the same ridge protocol as loocv_baseline):
1. dataset-oracle ceiling: predict each sample by the mean score of its OWN
   dataset's other samples (zero images) -> how much a pure shortcut earns.
2. within-dataset rho: pooled predictions evaluated per dataset (removes the
   inter-dataset score offset) -> residual per-dataset signal.
3. leave-one-dataset-out: train on 2 datasets, test on the 3rd.
4. feature de-meaned by dataset (leave-one-out per sample) -> same LOOCV.

Usage:
    python scripts/leak_check.py --features outputs/features/dinov2_vitb14_warped
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from loocv_baseline import fit_predict, loocv_predictions, spearman

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "outputs" / "sample_manifest.csv"


def load_bags(feat_dir: Path) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    labels_df = pd.read_csv(MANIFEST)
    lookup = labels_df.set_index(["dataset", "sample"])["senescence_score"]
    ds_arr, keys, feat_list = [], [], []
    for bag in sorted(feat_dir.glob("*.npy")):
        dataset, sample = bag.stem.split("__")
        key = (dataset, sample)
        assert key in lookup.index, bag.stem
        keys.append(key)
        ds_arr.append(dataset)
        feat_list.append(np.load(bag).mean(axis=0))
    labels = lookup.reindex(pd.MultiIndex.from_tuples(keys)).to_numpy(dtype=float)
    return feat_list, np.asarray(ds_arr), labels


def run_split(feat: np.ndarray, labels: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray) -> np.ndarray:
    Xtr = np.stack([feat[j] for j in tr_idx])
    mu_f, sd_f = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu_f) / sd_f
    mu, sd = labels[tr_idx].mean(), labels[tr_idx].std(ddof=0) or 1.0
    ytr = (labels[tr_idx] - mu) / sd
    pred = np.empty(len(te_idx))
    for k, j in enumerate(te_idx):
        xt = ((feat[j] - mu_f) / sd_f)[None, :]
        pred[k] = fit_predict(Xtr, ytr, xt, (mu, sd))
    return pred


def dataset_oracle_rho(labels: np.ndarray, ds: np.ndarray) -> float:
    oracle = np.empty(len(labels))
    for d in np.unique(ds):
        idx = np.where(ds == d)[0]
        for j in idx:
            others = [i for i in idx if i != j]
            oracle[j] = labels[others].mean()
    return spearman(labels, oracle)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, required=True)
    args = ap.parse_args()

    feat_dir = Path(args.features)
    feat_dir = feat_dir if feat_dir.is_dir() else PROJECT / "outputs" / "features" / str(feat_dir)
    feats, ds, labels = load_bags(feat_dir)
    n = len(labels)

    # 0) reference: full LOOCV rho
    yt, yp = loocv_predictions(feats, labels)
    rho_full = spearman(yt, yp)

    # 1) dataset-oracle ceiling (zero images)
    rho_oracle = dataset_oracle_rho(labels, ds)

    # 2) within-dataset rho from the LOOCV predictions
    print("\n=== dataset shortcut diagnostics ===")
    print(f"full LOOCV rho (reference)          : {rho_full:+.4f}")
    print(f"dataset-oracle rho (pure shortcut)  : {rho_oracle:+.4f}")
    print("  (if the two are close, most signal comes from dataset identity)")
    for d in np.unique(ds):
        idx = np.where(ds == d)[0]
        rho_d = spearman(yt[idx], yp[idx]) if len(idx) > 1 else float("nan")
        print(f"  within {d} (n={len(idx):2d}) rho: {rho_d:+.4f}")

    # 3) leave-one-dataset-out
    preds_all = np.empty(n)
    for te in np.unique(ds):
        tr_idx = np.where(ds != te)[0]
        te_idx = np.where(ds == te)[0]
        preds_all[te_idx] = run_split(feats, labels, tr_idx, te_idx)
    rho_lodo = spearman(labels, preds_all)
    print(f"leave-one-dataset-out rho           : {rho_lodo:+.4f}")

    # 4) feature de-meaned by dataset (leave-out per sample)
    feat_dm = np.empty_like(np.stack(feats))
    for j in range(n):
        same = np.where(ds == ds[j])[0]
        others = [i for i in same if i != j]
        feat_dm[j] = np.stack(feats)[j] - np.mean([feats[i] for i in others], axis=0)
    yt2, yp2 = loocv_predictions([feat_dm[j] for j in range(n)], labels)
    rho_dm = spearman(yt2, yp2)
    print(f"dataset-mean-removed features rho   : {rho_dm:+.4f}")
    print(f"  (drop from {rho_full:+.4f} -> {rho_dm:+.4f} is the dataset-identity contribution)")


if __name__ == "__main__":
    main()