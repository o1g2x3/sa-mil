"""Paper-ready figures + tables (P2-1).

Inputs (all existing outputs, nothing recomputed):
  outputs/sa_mil_results_table.csv          per-config per-seed SA-MIL metrics
  outputs/p0_3x_*_attnmil_summary.json      Ilse baseline per-seed rhos + mean CI
  outputs/p0_3_features_dinov2_vitb14{,_max}_summary.json   ridge baselines CI
  outputs/p1_2_gene_oracle_summary.json     gene anchor CI
  outputs/sa_mil_sa_mil_a2_stable_res_seed*_predictions.csv  Ab-1 per-sample preds

Outputs:
  outputs/paper_figures/fig1_stability.{pdf,png}   seed-level rho scatter + bars
  outputs/paper_figures/fig2_mechanism.{pdf,png}   attention entropy / top1 vs uniform
  outputs/paper_figures/fig3_predictions.{pdf,png} y_true vs y_pred (Ab-1, pooled seeds)
  outputs/paper_tables/table1_main.{xlsx,tex,png}  main results
  outputs/paper_tables/table2_ablation.{xlsx,tex,png}

Note: seed counts are ragged (Ilse 3, Ab-1/2 5, Ab-3/4 3) and pubfig's
bar_scatter rejects inhomogeneous arrays, so fig1 is drawn with matplotlib
directly but exported through pubfig's publication pipeline (spec/width/trim).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pubfig as pf
import pubtab

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "outputs"
FIG_DIR = OUT / "paper_figures"
TAB_DIR = OUT / "paper_tables"

ENTROPY_UNIFORM = float(np.log(64))          # 4.1589
TOP1_UNIFORM = 1.0 / 64                      # 0.015625

# category label, results-table config tag, expected seed count
SA_CONFIGS = [
    ("Ab-1 Full SA-MIL", "sa_mil_a2_stable_res", 5),
    ("Ab-2 No stabilization", "sa_mil_a2_nostab_res", 5),
    ("Ab-3 Single branch", "sa_mil_a1_stable_res", 3),
    ("Ab-4 No residual", "sa_mil_a2_stable_nores", 3),
]
PALETTE = ["#7f7f7f", "#4477AA", "#EE6677", "#228833", "#CCBB44"]

RC = {"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
      "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
      "axes.spines.top": False, "axes.spines.right": False}


def load_json(path: Path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def seed_rhos(df: pd.DataFrame, tag: str) -> np.ndarray:
    g = df[df["config"] == tag].sort_values("seed")
    assert len(g) > 0, f"no rows for {tag}"
    return g["rho"].to_numpy(dtype=float)


def config_stats(df: pd.DataFrame, tag: str) -> dict:
    g = df[df["config"] == tag]
    return {"n": len(g), "rho": g["rho"].mean(), "rho_std": g["rho"].std(ddof=1),
            "r2": g["r2"].mean(), "r2_std": g["r2"].std(ddof=1),
            "ci_lo": g["rho_ci_lo"].mean(), "ci_hi": g["rho_ci_hi"].mean(),
            "mae": g["mae"].mean(), "unc": g["unc_spearman"].mean(),
            "entropy": g["entropy"].mean(), "top1": g["top1"].mean(),
            "alpha": g["alpha"].mean(), "tau": g["tau_mean"].mean(),
            "stable": bool(g["stable"].all())}


# ---------- figure 1: seed stability ----------

def fig1_stability(df: pd.DataFrame, ilse_rhos: np.ndarray) -> None:
    rng = np.random.default_rng(42)
    groups = ([("Ilse Attention-MIL", ilse_rhos)]
              + [(label, seed_rhos(df, tag)) for label, tag, _ in SA_CONFIGS])
    means = np.array([v.mean() for _, v in groups])
    stds = np.array([v.std(ddof=1) for _, v in groups])

    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(4.8, 3.1))
        x = np.arange(len(groups))
        for xi, (lbl, vals), ci_color in zip(x, groups, PALETTE):
            ax.errorbar(xi, vals.mean(), yerr=vals.std(ddof=1), fmt="none",
                        ecolor="0.25", elinewidth=1.0, capsize=4, capthick=1.0, zorder=3)
            ax.hlines(vals.mean(), xi - 0.22, xi + 0.22, lw=3.0, color=ci_color, zorder=4)
        for xi, (_, vals) in zip(x, groups):
            jit = rng.uniform(-0.14, 0.14, size=len(vals))
            ax.scatter(np.full(len(vals), xi) + jit, vals, s=14, zorder=5,
                       color="0.15", alpha=0.85, linewidths=0)
        ax.axhline(0.945, ls="--", lw=0.9, color="0.35")
        ax.text(len(groups) - 0.45, 0.949, "MeanPool ridge 0.945", ha="right", fontsize=6.5, color="0.35")
        ax.axhline(0.990, ls=":", lw=0.9, color="0.55")
        ax.text(len(groups) - 0.45, 0.994, "gene-oracle 0.990", ha="right", fontsize=6.5, color="0.55")
        ax.set_xticks(x, [lbl.replace(" ", "\n", 1) for lbl, _ in groups])
        ax.set_ylabel("Spearman $\\rho$ (LOOCV, 43 samples)")
        ax.set_ylim(0.60, 1.03)
        fig.tight_layout()
    pf.batch_export(fig, str(FIG_DIR / "fig1_stability"), formats=("pdf", "png"),
                    spec="nature", width="double", dpi=300)

    # verification against notes.md numbers
    ab1 = config_stats(df, "sa_mil_a2_stable_res")
    assert abs(ab1["rho"] - 0.893) < 5e-3 and abs(ab1["rho_std"] - 0.026) < 5e-3, ab1
    assert abs(means[0] - 0.749) < 5e-3 and abs(stds[0] - 0.047) < 5e-3
    logger.info("fig1 verified: Ilse %.3f±%.3f, Ab-1 %.3f±%.3f",
                means[0], stds[0], ab1["rho"], ab1["rho_std"])


# ---------- figure 2: attention mechanism ----------

def fig2_mechanism(df: pd.DataFrame) -> None:
    tags = [t for _, t, _ in SA_CONFIGS]
    labels = [lbl.split(" ")[0] for lbl, _, _ in SA_CONFIGS]
    ent = np.array([df[df["config"] == t]["entropy"].to_numpy() for t in tags], dtype=object)
    top1 = np.array([df[df["config"] == t]["top1"].to_numpy() for t in tags], dtype=object)
    ent_mean = np.array([v.mean() for v in ent])
    top1_mean = np.array([v.mean() for v in top1])
    rng = np.random.default_rng(42)

    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.8))
        for ax, vals, _means, ref, ref_txt, ylab in (
                (axes[0], ent, ent_mean, ENTROPY_UNIFORM, "uniform ln 64 = 4.159", "Attention entropy (nats)"),
                (axes[1], top1, top1_mean, TOP1_UNIFORM, "uniform $1/64$ = 0.0156", "Top-1 attention weight")):
            for xi, v in enumerate(vals):
                ax.errorbar(xi, v.mean(), yerr=v.std(ddof=1), fmt="none",
                            ecolor="0.25", elinewidth=0.9, capsize=3, capthick=0.9, zorder=3)
                ax.hlines(v.mean(), xi - 0.2, xi + 0.2, lw=2.6, color=PALETTE[1 + xi], zorder=4)
                jit = rng.uniform(-0.12, 0.12, size=len(v))
                ax.scatter(np.full(len(v), xi) + jit, v, s=12, zorder=5,
                           color="0.15", alpha=0.85, linewidths=0)
            ax.axhline(ref, ls="--", lw=0.9, color="0.4")
            ax.text(len(tags) - 0.45, ref, ref_txt, ha="right", va="bottom",
                    fontsize=6.5, color="0.4")
            ax.set_xticks(np.arange(len(tags)), labels)
            ax.set_ylabel(ylab)
        axes[0].set_ylim(3.9, 4.22)
        fig.tight_layout()
    pf.batch_export(fig, str(FIG_DIR / "fig2_mechanism"), formats=("pdf", "png"),
                    spec="nature", width="double", dpi=300)
    logger.info("fig2 verified: entropy %s (uniform %.3f), top1 %s",
                np.round(ent_mean, 3), ENTROPY_UNIFORM, np.round(top1_mean, 4))


# ---------- figure 3: predictions ----------

def fig3_predictions() -> None:
    frames = [pd.read_csv(p) for p in sorted(OUT.glob("sa_mil_sa_mil_a2_stable_res_seed*_predictions.csv"))]
    assert len(frames) == 5, f"expected 5 Ab-1 seed files, got {len(frames)}"
    d = pd.concat(frames, ignore_index=True)
    # display-only z-scoring (monotonic; rho unchanged), avoids raw-unit tick clutter
    z = lambda v: (v - v.mean()) / v.std(ddof=0)
    labels = d["dataset"].str.replace("huc_", "").to_numpy()
    fig = pf.scatter(z(d["y_true"]).to_numpy(), z(d["y_pred"]).to_numpy(),
                     labels=labels, x_label="Expert visual score (z)",
                     y_label="SA-MIL prediction (z)", show_y_equal_x=True)
    # remove the auto-annotated p-value (test unspecified by pubfig; our claim
    # metric is Spearman rho, reported in the caption/table, not here)
    for txt in list(fig.axes[0].texts) + list(fig.axes[0].get_children()):
        if isinstance(txt, matplotlib.text.Text) and "p =" in txt.get_text():
            txt.set_visible(False)
    pf.batch_export(fig, str(FIG_DIR / "fig3_predictions"), formats=("pdf", "png"),
                    spec="nature", width="single", dpi=300)
    logger.info("fig3 done: %d pooled points (5 seeds x 43 samples)", len(d))


# ---------- tables ----------

def fmt(v: float, nd: int = 3) -> str:
    return f"{v:.{nd}f}"


def table1_main(df: pd.DataFrame, ilse: dict, dino: dict, dino_max: dict, oracle: dict) -> pd.DataFrame:
    rows = []
    for name, agg, seeds, rho, std, ci, r2, mae in [
        ("Gene-oracle (RNA-seq, anchor)", "ridge on 24,138 genes", 1,
         oracle["ci"]["rho"][0], None, oracle["ci"]["rho"][1:], oracle["ci"]["r2"][0], oracle["ci"]["mae"][0]),
        ("MeanPool (ridge)", "mean of FOV features", None,
         dino["ci"]["rho"][0], None, dino["ci"]["rho"][1:], dino["ci"]["r2"][0], dino["ci"]["mae"][0]),
        ("MaxPool (ridge)", "max of FOV features", None,
         dino_max["ci"]["rho"][0], None, dino_max["ci"]["rho"][1:], dino_max["ci"]["r2"][0], dino_max["ci"]["mae"][0]),
        ("Ilse Attention-MIL", "gated attention", len(ilse["per_seed"]),
         ilse["mean_ci"]["rho"][0], np.std([s["rho"] for s in ilse["per_seed"]], ddof=1),
         ilse["mean_ci"]["rho"][1:], ilse["mean_ci"]["r2"][0], ilse["mean_ci"]["mae"][0]),
    ] + [
        (f"{label}" + (" (main)" if label.startswith("Ab-1") else
                       " (recommended simplified)" if label.startswith("Ab-3") else ""),
         "SA-MIL", cfg["n"], cfg["rho"], cfg["rho_std"], (cfg["ci_lo"], cfg["ci_hi"]),
         cfg["r2"], cfg["mae"])
        for (label, tag, _n) in SA_CONFIGS
        for cfg in [config_stats(df, tag)]
    ]:
        rho_s = fmt(rho) + (f" ± {std:.3f}" if std is not None else "")
        rows.append({"Method": name, "Aggregation": agg, "Seeds": seeds or "—",
                     "Spearman rho": rho_s, "95% CI": f"[{ci[0]:.3f}, {ci[1]:.3f}]",
                     "R2": fmt(r2), "MAE": fmt(mae, 4)})
    return pd.DataFrame(rows)


def table2_ablation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, tag, _n in SA_CONFIGS:
        c = config_stats(df, tag)
        n_attr = int(df[df["config"] == tag]["n_attr"].iloc[0])
        rows.append({
            "Config": label, "Seeds": c["n"], "Spearman rho": f"{c['rho']:.3f} ± {c['rho_std']:.3f}",
            "R2": f"{c['r2']:.3f} ± {c['r2_std']:.3f}",
            "unc vs |resid| (rho)": "—" if n_attr == 1 or np.isnan(c["unc"]) else fmt(c["unc"], 3),
            "Entropy": fmt(c["entropy"], 3), "Top-1": fmt(c["top1"], 4),
            "alpha": fmt(c["alpha"], 3) if c["stable"] else "—",
            "tau": fmt(c["tau"], 3) if c["stable"] else "—",
        })
    return pd.DataFrame(rows)


def export_table(tab: pd.DataFrame, name: str, caption: str, label: str) -> None:
    xlsx = TAB_DIR / f"{name}.xlsx"
    tab.to_excel(xlsx, index=False)
    tex = TAB_DIR / f"{name}.tex"
    pubtab.xlsx2tex(str(xlsx), output=str(tex), theme="three_line",
                    caption=caption, label=label)
    try:
        pubtab.preview(str(tex), output=str(TAB_DIR / f"{name}.png"), dpi=300)
    except Exception as e:  # LaTeX preview optional; tex/xlsx remain the deliverable
        logger.warning("preview skipped for %s (%s)", name, type(e).__name__)


# ---------- main ----------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    FIG_DIR.mkdir(exist_ok=True)
    TAB_DIR.mkdir(exist_ok=True)
    plt.rcParams.update(RC)

    df = pd.read_csv(OUT / "sa_mil_results_table.csv")
    ilse = load_json(OUT / "p0_3x_features_dinov2_vitb14_attnmil_summary.json")
    dino = load_json(OUT / "p0_3_features_dinov2_vitb14_summary.json")
    dino_max = load_json(OUT / "p0_3_features_dinov2_vitb14_max_summary.json")
    oracle = load_json(OUT / "p1_2_gene_oracle_summary.json")

    fig1_stability(df, np.array([s["rho"] for s in ilse["per_seed"]]))
    fig2_mechanism(df)
    fig3_predictions()

    t1 = table1_main(df, ilse, dino, dino_max, oracle)
    export_table(t1, "table1_main",
                 "Main results: 43-sample LOOCV, frozen DINOv2 ViT-B/14 features. "
                 "Neural rows: mean ± seed std; CI is the mean of per-seed bootstrap intervals. "
                 "Ridge/oracle rows: single deterministic run with subject-level bootstrap CI.",
                 "tab:main")
    t2 = table2_ablation(df)
    export_table(t2, "table2_ablation",
                 "SA-MIL ablations (DINOv2). Uniform references: entropy ln 64 = 4.159, "
                 "top-1 = 0.0156. alpha/tau only apply to stabilized configs; the single-branch "
                 "config (Ab-3) has no branch variance, hence no uncertainty column.",
                 "tab:ablation")

    print("\n=== Table 1 (main) ===")
    print(t1.to_string(index=False))
    print("\n=== Table 2 (ablation) ===")
    print(t2.to_string(index=False))
    logger.info("figures -> %s, tables -> %s", FIG_DIR, TAB_DIR)


if __name__ == "__main__":
    main()
