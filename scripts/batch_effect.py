"""P0-2 batch effect check: image domain and score distribution across MSC1/2/3.

Read-only. Samples a bounded set of images per (sample, view), computes
brightness/pixel statistics per dataset, and checks whether the senescence
score labels are inter-comparable across the three datasets (per-dataset
vs global normalization). Outputs two CSVs + a verdict report.

Usage:
    python scripts/batch_effect.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

DATA_ROOT = Path(
    os.environ.get("MIL_DATA_ROOT", "data")
)
MANIFEST = Path(__file__).resolve().parent.parent / "outputs" / "sample_manifest.csv"
OUT_IMG = Path(__file__).resolve().parent.parent / "outputs" / "batch_effect_images.csv"
OUT_SCORE = Path(__file__).resolve().parent.parent / "outputs" / "batch_effect_scores.csv"
VIEWS = ("WARPED", "BGAL", "DAPI")
N_SAMPLE = 10  # images per (sample, view), selected by even stride


def image_stats(path: Path) -> dict[str, float]:
    """Per-image brightness/pixel statistics (HxWxC uint8 TIFF)."""
    arr = np.asarray(Image.open(path)).astype(np.float32)
    return {
        "brightness": float(arr.mean()),
        "std": float(arr.std()),
        "ch0": float(arr[..., 0].mean()),
        "ch1": float(arr[..., 1].mean()),
        "ch2": float(arr[..., 2].mean()),
    }


def sample_image_paths(view_dir: Path, n: int = N_SAMPLE) -> list[Path]:
    """Even-stride deterministic subsample of images in a view dir."""
    if not view_dir.is_dir():
        return []
    files = sorted(p for p in view_dir.iterdir() if p.suffix.lower() == ".tif")
    if not files:
        return []
    idx = np.linspace(0, len(files) - 1, min(n, len(files))).round().astype(int)
    return [files[i] for i in idx]


def image_domain_stats() -> pd.DataFrame:
    """Per (sample, view) mean image stats, aggregated from a subsample."""
    manifest = pd.read_csv(MANIFEST)
    rows: list[dict] = []
    for _, m in manifest.iterrows():
        sample_dir = DATA_ROOT / m["dataset"] / m["sample"]
        for view in VIEWS:
            paths = sample_image_paths(sample_dir / view)
            if not paths:
                continue
            per = pd.DataFrame([image_stats(p) for p in paths])
            rows.append({
                "dataset": m["dataset"],
                "sample": m["sample"],
                "view": view,
                "n_seen": len(paths),
                "brightness": per["brightness"].mean(),
                "std": per["std"].mean(),
                "ch0": per["ch0"].mean(),
                "ch1": per["ch1"].mean(),
                "ch2": per["ch2"].mean(),
            })
    return pd.DataFrame(rows).sort_values(["dataset", "sample", "view"]).reset_index(drop=True)


def score_stats() -> pd.DataFrame:
    """Per-dataset distribution of raw and normalized scores + norm sanity check."""
    manifest = pd.read_csv(MANIFEST)
    rows: list[dict] = []
    for ds, g in manifest.groupby("dataset"):
        rows.append({
            "dataset": ds,
            "n": len(g),
            "raw_mean": g["senescence_score"].mean(),
            "raw_std": g["senescence_score"].std(ddof=1),
            "raw_min": g["senescence_score"].min(),
            "raw_max": g["senescence_score"].max(),
            "norm_mean": g["normalized_score"].mean(),
            "norm_max": g["normalized_score"].max(),
            "norm_min": g["normalized_score"].min(),
        })
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    images = image_domain_stats()
    scores = score_stats()

    images.to_csv(OUT_IMG, index=False)
    scores.to_csv(OUT_SCORE, index=False)

    print("=== image domain (per-dataset across samples) ===")
    pivot = images.groupby(["dataset", "view"])["brightness"].agg(["mean", "std"]).reset_index()
    for _, r in pivot.iterrows():
        print(f"  {r['dataset']:>10} {r['view']:<6} brightness {r['mean']:6.1f} +/- {r['std']:5.1f}")

    print("\n=== score distribution ===")
    for _, r in scores.iterrows():
        dr = r["raw_max"] - r["raw_min"]
        print(f"  {r['dataset']:>10} raw {r['raw_mean']:+.4f} +/- {r['raw_std']:.4f} "
              f"[{r['raw_min']:+.4f}, {r['raw_max']:+.4f}]  span={dr:.4f}  "
              f"normalized {r['norm_min']:.3f}-{r['norm_max']:.3f}")

    # ---- verdict ----
    norm_pd = (scores["norm_min"] <= 0.01) & (scores["norm_max"] >= 0.99)
    print("\n=== verdict ===")
    print(f"  normalized_score is per-dataset min-max (each spans [0,1]): {bool(norm_pd.all())} "
          f"-> {'NOT comparable across datasets' if norm_pd.all() else 'comparable'}")
    raw_means, raw_stds = scores["raw_mean"].to_numpy(), scores["raw_std"].to_numpy()
    max_gap = raw_means.max() - raw_means.min()
    pooled_std = np.sqrt((raw_stds ** 2).mean())
    print(f"  raw-score dataset means:{np.round(raw_means, 4)}  max gap={max_gap:.4f} "
          f"vs pooled std={pooled_std:.4f}")
    if max_gap > pooled_std:
        print("  -> strong label batch offset: standardize labels (per-dataset or global) before pooling.")
    else:
        print("  -> label batch offset within one pooled std; pooling likely OK, but standardize to be safe.")

    raw_means_img = pivot[pivot["view"] == "WARPED"].set_index("dataset")["mean"]
    if raw_means_img.max() - raw_means_img.min() > 10:
        print(f"  -> image domain shift detected in WARPED "
              f"(brightness span {raw_means_img.min():.1f}-{raw_means_img.max():.1f}).")
    else:
        print("  -> no strong WARPED brightness shift across datasets.")
    print(f"\noutputs: {OUT_IMG.name}, {OUT_SCORE.name}")


if __name__ == "__main__":
    main()