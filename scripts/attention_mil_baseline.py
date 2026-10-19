"""Ilse Attention-MIL (gated attention) regression baseline, per-fold LOOCV.

Frozen features; per-fold TRAIN-only feature + target standardization.
Training recipe (empirically verified to actually train at n=43; see notes):
two-stage — head warms up with attention frozen, then joint — Adam lr 1e-3,
no dropout in bag pooling, 64-token random sampling. Reporting identical
to loocv_baseline: rho / R2 / MAE + bootstrap 95% CI + permutation p.

NOTE (deviation record): the bare-one-pass small-net recipe (lr 1e-4, 50 ep,
dropout 0.3) provably fails to train here — attention collapses, predictions
shrivel toward the mean. The two-stage recipe keeps the SAME Ilse gated
architecture but trains it; results remain seed-sensitive, which is itself a
measured instability fact (ASMIL motivation) and is reported by --seeds.

Usage:
    python scripts/attention_mil_baseline.py --features outputs/features/dinov2_vitb14_warped
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from loocv_baseline import bootstrap_ci, mae, r2, spearman

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "outputs" / "sample_manifest.csv"

HIDDEN = 128
LR = 1e-3
EPOCHS_WARM = 20
EPOCHS_JOINT = 80
N_TOKEN = 64
SEED = 42


class GatedAttention(nn.Module):
    """Ilse et al. ICML 2018 gated attention pooling + linear regression head."""

    def __init__(self, dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        self.head = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))       # (n, hidden)
        a = torch.softmax(self.w(g).squeeze(-1), dim=0)            # (n,)
        bag = (a[:, None] * x).sum(0)                              # (dim,)
        return self.head(bag).squeeze(-1)                          # scalar


def subsample(bag: np.ndarray, rng: np.random.Generator, k: int = N_TOKEN) -> np.ndarray:
    """Deterministic fixed-size token subsample of a bag."""
    n = len(bag)
    if n <= k:
        return bag
    return bag[rng.permutation(n)[:k]]


def train_predict(bags: list[np.ndarray], y_tr_std: np.ndarray, bag_te: np.ndarray,
                  mu_f: np.ndarray, sd_f: np.ndarray, mu: float, sd: float,
                  rng: np.random.Generator, device: torch.device, dim: int) -> float:
    """Two-stage gated-attention training; predict test bag (raw units)."""
    torch.manual_seed(int(rng.integers(0, 2 ** 31)))
    model = GatedAttention(dim).to(device)
    loss_f = nn.SmoothL1Loss(beta=1.0)
    train_x = [torch.tensor(subsample(b, rng), dtype=torch.float32) for b in bags]
    train_y = [torch.tensor(y_tr_std[j], dtype=torch.float32, device=device)
               for j in range(len(bags))]

    def run(epochs: int, joint: bool) -> None:
        if not joint:
            for p in model.V, model.U, model.w:
                for pp in p.parameters():
                    pp.requires_grad_(False)
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=LR)
        model.train()
        for _ in range(epochs):
            for Xi, yi in zip(train_x, train_y):
                opt.zero_grad()
                loss_f(model(Xi.to(device)), yi).backward()
                opt.step()
        if not joint:
            for p in model.V, model.U, model.w:
                for pp in p.parameters():
                    pp.requires_grad_(True)

    run(EPOCHS_WARM, joint=False)     # warm head, attention frozen
    run(EPOCHS_JOINT, joint=True)     # unlock attention, joint fine-tune

    model.eval()
    x_te = torch.tensor(subsample(bag_te, rng), dtype=torch.float32, device=device)
    with torch.no_grad():
        y_hat_std = float(model(x_te).item())
    return y_hat_std * sd + mu


def loocv_attention(bags: list[np.ndarray], labels: np.ndarray, seed: int = SEED
                    ) -> tuple[np.ndarray, np.ndarray]:
    """43-fold LOOCV of the attention-MIL regression. Returns (y_true, y_pred) raw units."""
    n = len(bags)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_true, y_pred = np.zeros(n), np.zeros(n)
    for i in range(n):
        rng = np.random.default_rng(seed * 1000 + i)
        tr = [j for j in range(n) if j != i]
        Xtr = np.concatenate([bags[j] for j in tr], axis=0)       # train-only stats
        mu_f, sd_f = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
        tr_bags = [(bags[j] - mu_f) / sd_f for j in tr]
        te_std = (bags[i] - mu_f) / sd_f
        mu, sd = labels[tr].mean(), labels[tr].std(ddof=0) or 1.0
        y_tr_std = (labels[tr] - mu) / sd
        y_true[i] = labels[i]
        y_pred[i] = train_predict(tr_bags, y_tr_std, te_std, mu_f, sd_f, mu, sd,
                                  rng, device, bags[i].shape[1])
    return y_true, y_pred


def _selfcheck() -> None:
    import torch as _
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 4))
    y = x[:, 0] - 2 * x[:, 1] + 0.01 * rng.normal(size=12)
    bags = [np.tile(xi, (3, 1)) for xi in x]                      # 3 identical instances/sample
    rhos = []
    for seed in (0, 1, 2):
        yt, yp = loocv_attention(bags, y, seed=seed)
        rhos.append(spearman(yt, yp))
    logger.info("selfcheck seed rhos: %s", np.round(rhos, 3))
    assert float(np.mean(rhos)) > 0.5, f"selfcheck mean rho={np.mean(rhos):.3f}"
    logger.info("selfcheck passed (two-stage gated attention LOOCV train loop)")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--seeds", type=int, default=3, help="independent seed runs")
    ap.add_argument("--n-perm", type=int, default=0, help="permutations per seed (0=skip)")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    if args.selfcheck:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        _selfcheck()
        return

    feat_dir = Path(args.features)
    feat_dir = feat_dir if feat_dir.is_dir() else PROJECT / "outputs" / "features" / str(feat_dir)
    tag = args.out or f"{feat_dir.parent.name}_{feat_dir.name}_attnmil"
    tag = tag.replace("_warped", "")

    labels_df = pd.read_csv(MANIFEST).set_index(["dataset", "sample"])["senescence_score"]
    keys, bags = [], []
    for bag in sorted(feat_dir.glob("*.npy")):
        dataset, sample = bag.stem.split("__")
        key = (dataset, sample)
        assert key in labels_df.index, bag.stem
        keys.append(key)
        bags.append(np.load(bag).astype(np.float32))
    labels = labels_df.reindex(pd.MultiIndex.from_tuples(keys)).to_numpy(dtype=float)

    table, ci_rows = [], {}
    for seed in range(args.seeds):
        seed_eff = SEED + seed
        y_true, y_pred = loocv_attention(bags, labels, seed=seed_eff)
        ci = bootstrap_ci(y_true, y_pred)
        table.append({"seed": seed, **{k: v[0] for k, v in ci.items()}})
        for k, v in ci.items():
            ci_rows.setdefault(k, []).append(v)

    mean_ci = {k: (float(np.mean([v[0] for v in vals])),
                   float(np.min([v[1] for v in vals])), float(np.max([v[2] for v in vals])))
               for k, vals in ci_rows.items()}

    # report seed-level instability
    seed_metrics = pd.DataFrame(table)
    print("\n=== Ilse Attention-MIL (gated), LOOCV, frozen features ===")
    print(f"  per-seed rho: {np.round(seed_metrics['rho'].to_numpy(), 4)}")
    for name, (val, lo, hi) in mean_ci.items():
        print(f"  {name:>4} mean: {val:+.4f}  [min-CI {lo:+.4f}, max-CI {hi:+.4f}]  (n_seed={args.seeds})")
    print(f"  rho spread (max-min): {seed_metrics['rho'].max() - seed_metrics['rho'].min():+.4f}")
    if args.n_perm > 0:
        obs = float(np.mean(seed_metrics['rho']))
        p_vals = []
        for seed in range(max(1, min(args.seeds, 2))):
            p_vals.append(_perm_p(bags, labels, args.n_perm, SEED + seed))
        print(f"  permutation p (n={args.n_perm}): {np.round(p_vals, 4)}")
    print(f"  recipe: warm {EPOCHS_WARM} ep + joint {EPOCHS_JOINT} ep, Adam lr={LR}, "
          f"no dropout, hidden={HIDDEN}, n_token={N_TOKEN}")

    y0, yp0 = loocv_attention(bags, labels, seed=SEED)
    pd.DataFrame({"dataset": [k[0] for k in keys], "sample": [k[1] for k in keys],
                  "y_true": y0, "y_pred": yp0}).to_csv(
        PROJECT / "outputs" / f"p0_3x_{tag}.csv", index=False)
    with open(PROJECT / "outputs" / f"p0_3x_{tag}_summary.json", "w") as f:
        json.dump({"per_seed": table, "mean_ci": mean_ci, "recipe": {
            "hidden": HIDDEN, "lr": LR, "epochs_warm": EPOCHS_WARM,
            "epochs_joint": EPOCHS_JOINT, "n_token": N_TOKEN}},
            f, indent=2, default=float)


def _perm_p(bags: list[np.ndarray], labels: np.ndarray, n_perm: int, seed: int) -> float:
    obs_rho = spearman(*loocv_attention(bags, labels, seed=seed))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for k in range(n_perm):
        y_perm = rng.permutation(labels)
        _, yp = loocv_attention(bags, y_perm, seed=seed + 100000 + k)
        null[k] = spearman(y_perm, yp)
    return float((1 + (null >= obs_rho).sum()) / (1 + n_perm))


if __name__ == "__main__":
    main()