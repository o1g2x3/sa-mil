"""SA-MIL per-fold LOOCV training + report (frozen features).

Protocol identical to loocv_baseline.py (replacement pooler):
  per fold i: standardize features & target with TRAIN-only stats, train SAMIL
  on the 42 train bags (fixed-count sampling 64, token random dropping,
  two-stage warm-up 5 + joint 45, Adam lr 1e-3, Huber delta=1.0), predict the
  held-out bag, record the raw-unit prediction + branch variance.

Reports:
  - pooled rho / R2 / MAE + bootstrap 95% CI (same evaluators as baselines)
  - uncertainty self-check (outline §5-2): Spearman(branch variance, |residual|)
  - attention sanity: mean per-sample entropy + top-1 frac (not uniform/one-hot)
  - top-k attention instances per sample -> csv (P2 visualization input)

Usage:
    python scripts/train_sa_mil_loocv.py --features dinov2_vitb14_warped
    python scripts/train_sa_mil_loocv.py --no-stable     # Ablation-2
    python scripts/train_sa_mil_loocv.py --n-attr 1      # Ablation-3
    python scripts/train_sa_mil_loocv.py --no-residual   # Ablation-4
    python scripts/train_sa_mil_loocv.py --selfcheck
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.sa_mil import SAMIL, get_top_instances
from loocv_baseline import bootstrap_ci, mae, r2, spearman

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "outputs" / "sample_manifest.csv"
DATA_ROOT = Path(
    os.environ.get("MIL_DATA_ROOT", "data")
)

# fixed hyperparameters (outline §5-1; churn log 2026-09-02: lr=1e-3 + two-stage)
LR = 1e-3
N_TOKEN = 64
DROP_PROB = 0.15
EPOCHS_WARM = 5
EPOCHS_TOTAL = 50
SEED = 42
TOPK = 5


# ---------- token sampling / dropping ----------

def sample_indices(n: int, k: int, g: torch.Generator) -> torch.Tensor:
    """Fixed-count random sampling (nnMIL idea): without replacement, resample if tiny."""
    if n >= k:
        return torch.randperm(n, generator=g)[:k]
    return torch.randint(0, n, (k,), generator=g)   # tiny-bag guard: replacement


def token_keep(k: int, drop_p: float, g: torch.Generator) -> torch.Tensor:
    """ASMIL token random dropping (training-time stochasticity)."""
    if drop_p <= 0.0:
        return torch.ones(k, dtype=torch.bool)
    keep = torch.rand(k, generator=g) >= drop_p
    if not keep.any():
        keep[0] = True
    return keep


# ---------- training ----------

def make_model(feat_dim: int, cfg: dict) -> SAMIL:
    return SAMIL(feat_dim=feat_dim, hidden=128, dropout=0.3,
                 n_attr=cfg["n_attr"], residual=0.3, tau=cfg["tau"],
                 use_stable=cfg["use_stable"], use_residual=cfg["use_residual"])


def train_fold(bags: list[np.ndarray], labels: np.ndarray, i: int, cfg: dict,
               device: torch.device, seed: int) -> dict:
    """One held-out fold. Returns raw pred + branch variance + attention."""
    n = len(bags)
    tr = [j for j in range(n) if j != i]
    Xtr = np.concatenate([bags[j] for j in tr], axis=0)          # train-only stats
    mu_f, sd_f = Xtr.mean(0), Xtr.std(0) + 1e-8
    tr_bags = [(bags[j] - mu_f) / sd_f for j in tr]
    te_std = (bags[i] - mu_f) / sd_f
    mu, sd = labels[tr].mean(), labels[tr].std(ddof=0) or 1.0
    y_tr = (labels[tr] - mu) / sd

    g = torch.Generator().manual_seed(seed * 1000 + i)
    torch.manual_seed(seed * 1000 + i)          # seed model init too: same seed -> same fold result
    model = make_model(bags[i].shape[1], cfg).to(device)
    loss_f = nn.SmoothL1Loss(beta=1.0)                            # Huber delta=1
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    for epoch in range(EPOCHS_TOTAL):
        freeze = epoch < EPOCHS_WARM                              # two-stage warm-up
        model.train()
        for j, bag in enumerate(tr_bags):
            idx = sample_indices(len(bag), N_TOKEN, g)
            keep = token_keep(N_TOKEN, DROP_PROB if cfg["use_stable"] else 0.0, g)
            x = torch.tensor(bag[idx], dtype=torch.float32, device=device).unsqueeze(0)
            kee = keep.to(device).unsqueeze(0)
            y = torch.tensor(y_tr[j], dtype=torch.float32, device=device).unsqueeze(0)
            opt.zero_grad()
            loss_f(model(x, keep=kee, freeze_attn=freeze), y).backward()
            opt.step()

    model.eval()
    idx = sample_indices(len(te_std), N_TOKEN, g)
    x = torch.tensor(te_std[idx], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        out = model(x, freeze_attn=False, return_attn=True)
    attn_avg = out["attn"].mean(dim=0)[0]                         # average over branches
    y_hat = float(out["pred"].item()) * sd + mu                   # back to raw units
    top_idx, top_w = get_top_instances(attn_avg, idx.numpy(), TOPK)
    diag = {"alpha_mean": float(np.mean([torch.sigmoid(b.alpha_logit).item()
                                         for b in model.branches])),
            "tau_mean": float(np.mean([torch.clamp(b.tau, 0.1, 10.0).item()
                                       for b in model.branches]))}
    return {"y_true": float(labels[i]), "y_pred": y_hat,
            "uncertainty": float(out["uncertainty"].item()),
            "entropy": float(-(attn_avg * torch.log(attn_avg + 1e-12)).sum().item()),
            "top1": float(attn_avg.max().item()),
            "top_idx": top_idx.astype(int), "top_w": top_w, **diag}


def loocv_samil(bags: list[np.ndarray], labels: np.ndarray, cfg: dict,
                seed: int = SEED) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return [train_fold(bags, labels, i, cfg, device, seed) for i in range(len(bags))]


# ---------- selfcheck ----------

def _selfcheck() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 4))                                   # latent -> label
    y = x[:, 0] - 2 * x[:, 1] + 0.01 * rng.normal(size=12)
    bags = [np.tile(xi, (3, 1)).astype(np.float32) for xi in x]    # 3 instances/sample
    for cfg in ({"n_attr": 2, "use_stable": True, "use_residual": True, "tau": None},
                {"n_attr": 1, "use_stable": False, "use_residual": False, "tau": None}):
        folds = loocv_samil(bags, y, cfg, seed=0)
        rho = spearman(np.array([r["y_true"] for r in folds]),
                       np.array([r["y_pred"] for r in folds]))
        logger.info("selfcheck cfg=%s rho=%.3f", cfg, rho)
        assert np.isfinite([r["y_pred"] for r in folds]).all() and rho > 0.3, f"selfcheck rho={rho:.3f}"
    logger.info("selfcheck passed (SA-MIL LOOCV train loop, both ablation configs)")


# ---------- main ----------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--seeds", type=int, default=1, help="independent seed runs")
    ap.add_argument("--n-attr", type=int, default=2, choices=(1, 2))
    ap.add_argument("--no-stable", action="store_true", help="Ablation-2: plain softmax, no anchor/drop")
    ap.add_argument("--no-residual", action="store_true", help="Ablation-4: drop mean-pool residual")
    ap.add_argument("--tau", type=float, default=None,
                    help="fix normalized-sigmoid temperature (default: learnable)")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        _selfcheck()
        return

    feat_dir = Path(args.features) if args.features else None
    feat_dir = (feat_dir if feat_dir is not None and feat_dir.is_dir()
                else PROJECT / "outputs" / "features" / str(feat_dir or "dinov2_vitb14_warped"))

    cfg = {"n_attr": args.n_attr, "use_stable": not args.no_stable,
           "use_residual": not args.no_residual, "tau": args.tau}
    parts = [f"sa_mil_a{cfg['n_attr']}"]
    parts += ["stable" if cfg["use_stable"] else "nostab"]
    parts += ["res" if cfg["use_residual"] else "nores"]
    tag = args.out or "_".join(parts)

    labels_df = pd.read_csv(MANIFEST).set_index(["dataset", "sample"])["senescence_score"]
    keys, bags, bag_files = [], [], []
    for bag in sorted(feat_dir.glob("*.npy")):
        dataset, sample = bag.stem.split("__")
        key = (dataset, sample)
        assert key in labels_df.index, bag.stem
        keys.append(key)
        bags.append(np.load(bag).astype(np.float32))
        img_dir = DATA_ROOT / dataset / sample / "WARPED"
        bag_files.append(sorted(img_dir.glob("*.tif")))
    labels = labels_df.reindex(pd.MultiIndex.from_tuples(keys)).to_numpy(dtype=float)

    # multi-seed: reuse cached fold results; each seed -> its own prediction table
    for sidx in range(args.seeds):
        seed = SEED + sidx
        folds = loocv_samil(bags, labels, cfg, seed=seed)
        y_true = np.array([r["y_true"] for r in folds])
        y_pred = np.array([r["y_pred"] for r in folds])
        unc = np.array([r["uncertainty"] for r in folds])
        ci = bootstrap_ci(y_true, y_pred)

        unc_spearman = (float(spearman(unc, np.abs(y_true - y_pred)))
                        if cfg["n_attr"] > 1 else np.nan)
        ent_mean = float(np.mean([r["entropy"] for r in folds]))
        top1_mean = float(np.mean([r["top1"] for r in folds]))
        alpha_mean = float(np.mean([r["alpha_mean"] for r in folds]))
        tau_mean = float(np.mean([r["tau_mean"] for r in folds]))
        ent_uniform = float(np.log(N_TOKEN))
        logger.info("seed %d: rho=%.4f unc_spearman=%.3f ent=%.2f (uniform %.2f) top1=%.3f",
                    seed, ci["rho"][0], unc_spearman, ent_mean, ent_uniform, top1_mean)

        stem = f"sa_mil_{tag}_seed{seed}"
        pd.DataFrame({"dataset": [k[0] for k in keys], "sample": [k[1] for k in keys],
                      "y_true": y_true, "y_pred": y_pred, "branch_var": unc}).to_csv(
            PROJECT / "outputs" / f"{stem}_predictions.csv", index=False)

        # top-k attention instances per sample (P2 visualization input)
        rows = []
        for ri, fold in enumerate(folds):
            dset, smp = keys[ri]
            for rank, (oidx, w) in enumerate(zip(fold["top_idx"], fold["top_w"]), start=1):
                fname = bag_files[ri][oidx].name if oidx < len(bag_files[ri]) else ""
                rows.append({"dataset": dset, "sample": smp, "rank": rank,
                             "orig_index": int(oidx), "filename": fname, "weight": float(w)})
        pd.DataFrame(rows).to_csv(PROJECT / "outputs" / f"{stem}_attn_topk.csv", index=False)

        with open(PROJECT / "outputs" / f"{stem}_summary.json", "w") as f:
            json.dump({"seed": seed, "cfg": cfg, "tag": tag,
                       "recipe": {"lr": LR, "n_token": N_TOKEN, "drop_prob": DROP_PROB,
                                  "epochs_warm": EPOCHS_WARM, "epochs_total": EPOCHS_TOTAL,
                                  "huber_delta": 1.0, "hidden": 128, "dropout": 0.3},
                       "ci": ci, "uncertainty_spearman": unc_spearman,
                       "entropy_mean": ent_mean, "entropy_uniform": ent_uniform,
                       "top1_mean": top1_mean, "alpha_mean": alpha_mean,
                       "tau_mean": tau_mean}, f, indent=2, default=float)

    # console table (last seed's CI shown; multi-seed spread on next line)
    print("\n=== SA-MIL LOOCV (frozen features) ===")
    print(f"  cfg: n_attr={cfg['n_attr']} stable={cfg['use_stable']} residual={cfg['use_residual']}")
    for name, (val, lo, hi) in ci.items():
        print(f"  {name:>4}: {val:+.4f}  [95% CI {lo:+.4f}, {hi:+.4f}]")
    print(f"  uncertainty |resid| Spearman: {unc_spearman:+.4f} (self-check §5-2, "
          f"{'>0 & sig' if unc_spearman > 0.15 else 'needs scrutiny'})")
    print(f"  attention entropy: {ent_mean:.2f} mean vs {ent_uniform:.2f} uniform "
          f"(not-spiky = entropy not tiny), top1 frac {top1_mean:.3f}")
    print(f"  outputs: outputs/sa_mil_{tag}_seed*/{{predictions,attn_topk,summary}}.*")


if __name__ == "__main__":
    main()