"""P0-1 read-only data loader for the MIL sample-level manifest.

Verifies the image/label correspondence for 43 samples across
huc_MSC_1/2/3 (read-only). Outputs a sample-level manifest CSV
used by all downstream steps.

Usage:
    python scripts/load_data.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_ROOT = Path(
    os.environ.get("MIL_DATA_ROOT", "data")
)
DATASETS = ["huc_MSC_1", "huc_MSC_2", "huc_MSC_3"]
VIEWS = ("WARPED", "BGAL", "DAPI")
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
OUTPUT = Path(__file__).resolve().parent.parent / "outputs" / "sample_manifest.csv"


def load_scores(dataset_dir: Path) -> pd.DataFrame:
    """Read senescence_scores.csv; row index == sample id."""
    path = dataset_dir / "senescence_scores.csv"
    df = pd.read_csv(path, index_col=0)
    return df.astype(float)


def count_images(view_dir: Path) -> int:
    """Count image files under a view dir (0 if the dir is absent)."""
    if not view_dir.is_dir():
        return 0
    return sum(
        1 for p in view_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def build_manifest(data_root: Path = DATA_ROOT, datasets: list[str] = DATASETS) -> pd.DataFrame:
    """Join per-sample image counts with their scores across all datasets."""
    rows: list[dict] = []
    for ds in datasets:
        ds_dir = data_root / ds
        if not ds_dir.is_dir():
            logger.warning("dataset dir missing: %s", ds_dir)
            continue
        scores = load_scores(ds_dir)
        for sample_id in scores.index.map(str):
            sample_dir = ds_dir / sample_id
            counts = {view: count_images(sample_dir / view) for view in VIEWS}
            row = {
                "dataset": ds,
                "sample": sample_id,
                "senescence_score": float(scores.loc[sample_id, "senescence_score"]),
                "normalized_score": float(scores.loc[sample_id, "normalized_score"]),
            }
            row.update({f"n_{view.lower()}": n for view, n in counts.items()})
            row["n_total"] = sum(counts.values())
            row["has_label"] = 1 if sample_dir.is_dir() else 0
            rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["dataset", "sample", "senescence_score", "normalized_score"]
    cols += [f"n_{v.lower()}" for v in VIEWS] + ["n_total", "has_label"]
    return df[cols].sort_values(["dataset", "sample"]).reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df = build_manifest()
    df.to_csv(OUTPUT, index=False)

    # ---- verification block ----
    per_ds = df.groupby("dataset").size()
    total_images = int(df["n_total"].sum())
    print(f"samples: {len(df)}  " + " ".join(f"{ds}={n}" for ds, n in per_ds.items()))
    print(f"images : total={total_images}  per-sample "
          f"min={df['n_total'].min()} median={df['n_total'].median():.0f} max={df['n_total'].max()}")
    print(f"missing label dirs: {int((df['has_label'] == 0).sum())}")
    print(f"samples w/o images: {len(df[df['n_total'] == 0])}")
    print(f"output : {OUTPUT}")

    expected = {ds: n for ds, n in zip(DATASETS, [17, 10, 16])}
    ok = len(df) == sum(expected.values()) and all(
        int(df[df["dataset"] == ds].shape[0]) == n for ds, n in expected.items()
    )
    ok = ok and total_images >= 8500
    if not ok:
        raise SystemExit("FAIL: sample/image counts deviate from the checked 43/~8900 facts")
    logger.info("P0-1 checks passed: 43 samples, all with images and labels.")


if __name__ == "__main__":
    main()