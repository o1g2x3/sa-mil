"""Attention top-5 vs random FOV comparison figures (P2 visualization input).

For representative samples (lowest / median / highest |residual| in one seed's
predictions csv), render two rows of WARPED FOVs:
  row 1: top-5 attention-weight FOVs (weights annotated, from *_attn_topk.csv)
  row 2: 5 random FOVs from the same bag (fixed seed, same-count baseline)
Even with near-uniform attention (see notes 2026-09-08), this shows the model's
relative emphasis against a random baseline rather than claiming spikiness.

Usage:
    python scripts/visualize_attention.py --seed 42
    python scripts/visualize_attention.py --seed 42 --tag sa_mil_a2_stable_res --n-samples 3
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(
    os.environ.get("MIL_DATA_ROOT", "data")
)
VIEW = "WARPED"
SEED = 42


def pick_samples(pred_csv: Path, n: int) -> pd.DataFrame:
    """Lowest / median / highest |residual| samples from one seed's predictions."""
    df = pd.read_csv(pred_csv)
    df["abs_resid"] = (df["y_true"] - df["y_pred"]).abs()
    df = df.sort_values("abs_resid").reset_index(drop=True)
    idx = sorted({0, len(df) // 2, len(df) - 1}) if n >= 3 else [0]
    return df.loc[idx[:n]]


def figure_for(row: pd.Series, topk_csv: Path, out_png: Path, k: int = 5) -> None:
    dset, smp = row["dataset"], row["sample"]
    all_files = sorted((DATA_ROOT / dset / smp / VIEW).glob("*.tif"))
    top = pd.read_csv(topk_csv)
    top = top[(top["dataset"] == dset) & (top["sample"] == smp)].sort_values("rank")

    rng = np.random.default_rng(SEED)
    rand_idx = rng.choice(len(all_files), size=k, replace=False)
    rand_files = [all_files[i] for i in rand_idx]

    fig, axes = plt.subplots(2, k, figsize=(2.4 * k, 5.0))
    for j, (_, r) in enumerate(top.iterrows()):
        img = Image.open(DATA_ROOT / dset / smp / VIEW / r["filename"])
        axes[0, j].imshow(img)
        axes[0, j].set_title(f"w={r['weight']:.3f}", fontsize=9)
        axes[0, j].set_ylabel("top-5" if j == 0 else "", fontsize=10)
    for j, f in enumerate(rand_files):
        axes[1, j].imshow(Image.open(f))
        axes[1, j].set_title("random", fontsize=9)
        axes[1, j].set_ylabel("random-5" if j == 0 else "", fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(f"{dset}/{smp}  true={row['y_true']:.4f}  pred={row['y_pred']:.4f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("saved %s", out_png.name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", type=str, default="sa_mil_a2_stable_res")
    ap.add_argument("--n-samples", type=int, default=3)
    args = ap.parse_args()

    stem = f"sa_mil_{args.tag}_seed{args.seed}"
    pred_csv = PROJECT / "outputs" / f"{stem}_predictions.csv"
    topk_csv = PROJECT / "outputs" / f"{stem}_attn_topk.csv"
    out_dir = PROJECT / "outputs" / "figures"
    out_dir.mkdir(exist_ok=True)

    for _, row in pick_samples(pred_csv, args.n_samples).iterrows():
        png = out_dir / f"{stem}_attn_{row['dataset']}_{row['sample']}.png"
        figure_for(row, topk_csv, png)


if __name__ == "__main__":
    main()