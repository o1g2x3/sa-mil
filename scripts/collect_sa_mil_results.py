"""Collect SA-MIL summary JSONs into the paper-ready results table.

Scans outputs/sa_mil_*_summary.json, emits per-seed rows and per-config
mean ± seed-std for rho / R2 / MAE (+ pooled bootstrap CI), plus the
diagnostics (uncertainty Spearman, attention entropy / top1, alpha / tau).
Writes outputs/sa_mil_results_table.csv and prints a markdown table.

Usage:
    python scripts/collect_sa_mil_results.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"


def main() -> None:
    rows = []
    for f in sorted(OUT.glob("sa_mil_*_summary.json")):
        with open(f) as fh:
            s = json.load(fh)
        ci = s["ci"]
        rows.append({
            "config": s["tag"], "seed": s["seed"],
            "n_attr": s["cfg"]["n_attr"], "stable": s["cfg"]["use_stable"],
            "residual": s["cfg"]["use_residual"], "tau": s["cfg"].get("tau"),
            "rho": ci["rho"][0], "rho_ci_lo": ci["rho"][1], "rho_ci_hi": ci["rho"][2],
            "r2": ci["r2"][0], "mae": ci["mae"][0],
            "unc_spearman": s.get("uncertainty_spearman"),
            "entropy": s.get("entropy_mean"), "top1": s.get("top1_mean"),
            "alpha": s.get("alpha_mean"), "tau_mean": s.get("tau_mean"),
        })
    if not rows:
        logger.warning("no sa_mil_*_summary.json under outputs/ yet")
        return
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "sa_mil_results_table.csv", index=False)

    print("\n=== SA-MIL results (per config: mean ± seed std over LOOCV runs) ===")
    for (cfg), g in df.groupby("config"):
        print(f"\n[{cfg}]  n_seeds={len(g)}")
        for m in ("rho", "r2", "mae"):
            print(f"  {m:>4}: {g[m].mean():+.4f} ± {g[m].std(ddof=1) if len(g) > 1 else 0:.4f}  "
                  f"(seeds: {np.round(g[m].to_numpy(), 4)})")
        lo = g["rho_ci_lo"].mean()
        hi = g["rho_ci_hi"].mean()
        print(f"  rho mean-CI: [{lo:+.4f}, {hi:+.4f}]")
        if g["unc_spearman"].notna().any():
            print(f"  unc_spearman: {g['unc_spearman'].mean():+.3f}  "
                  f"entropy {g['entropy'].mean():.2f}/4.16  top1 {g['top1'].mean():.3f}  "
                  f"alpha {g['alpha'].mean():.2f}  tau {g['tau_mean'].mean():.2f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()