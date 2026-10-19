"""P1-2: gene-oracle ceiling + within-dataset sanity, same protocol as image baseline.

A per-sample 24,138-dim gene vector (log1p TPM) replaces the image bag.
Ridge LOOCV is identical to loocv_baseline (per-fold standardization,
inner-LOOCV alpha selection). Reports the gene clock ceiling for contrast
with the image-only numbers.

Usage:
    python scripts/gene_oracle.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from loocv_baseline import bootstrap_ci, loocv_predictions, permutation_p, spearman

logger = logging.getLogger(__name__)
PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "outputs" / "sample_manifest.csv"
DATA_ROOT = Path(
    os.environ.get("MIL_DATA_ROOT", "data")
)


def load_gene_matrix() -> tuple[list[np.ndarray], list[tuple[str, str]], np.ndarray]:
    manifest = pd.read_csv(MANIFEST).sort_values(["dataset", "sample"]).reset_index(drop=True)
    vectors, keys = [], []
    for ds, sub in manifest.groupby("dataset", sort=True):
        gene_df = pd.read_csv(DATA_ROOT / ds / "genesymbol_lg1tpm_filtered.csv", index_col=0)
        for _, row in sub.iterrows():
            vec = gene_df[row["sample"]].to_numpy(dtype=float)
            assert vec.shape == (24138,) and np.isfinite(vec).all()
            vectors.append(vec)
            keys.append((row["dataset"], row["sample"]))
    labels = manifest["senescence_score"].to_numpy(dtype=float)
    return vectors, keys, labels


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()

    vecs, keys, labels = load_gene_matrix()
    n = len(vecs)
    logger.info("gene matrix: %d samples x %d genes", n, vecs[0].shape[0])

    y_true, y_pred = loocv_predictions(vecs, labels)
    ci = bootstrap_ci(y_true, y_pred)
    p_val, _null = permutation_p(vecs, labels, args.n_perm)

    pd.DataFrame({"dataset": [k[0] for k in keys], "sample": [k[1] for k in keys],
                  "y_true": y_true, "y_pred": y_pred}).to_csv(
        PROJECT / "outputs" / "p1_2_gene_oracle.csv", index=False)

    print("\n=== gene oracle (24,138 genes, same LOOCV protocol) ===")
    for name, (val, lo, hi) in ci.items():
        print(f"  {name:>4}: {val:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")
    print(f"  permutation p (n={args.n_perm}): {p_val:.4f}")
    print("\n  within-dataset gene rho (y_true vs y_pred, held-out):")
    per = pd.DataFrame({"dataset": [k[0] for k in keys], "y": y_true, "p": y_pred})
    for d, g in per.groupby("dataset"):
        print(f"    {d} (n={len(g)}): {spearman(g['y'].to_numpy(), g['p'].to_numpy()):+.4f}")

    print("\n  contrast (image baseline, WARPED mean-pool):")
    for tag, rho in [("DINOv2 @518", 0.9452), ("ResNet50 @224", 0.8534)]:
        print(f"    {tag}: rho {rho:+.4f}")

    summary = {"n_samples": n, "ci": ci, "perm_p": p_val}
    with open(PROJECT / "outputs" / "p1_2_gene_oracle_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)


if __name__ == "__main__":
    main()