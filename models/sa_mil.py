"""SA-MIL: Stable-Attribute MIL for continuous senescence scoring (regression).

One frozen-feature bag in, a continuous regression out:

    X in R^{B x N x D}   (N <= n_token sampled instances, D frozen feature dim)
    -> pred, uncertainty, per-branch attention weights (eval mode)

Design & compliance (docs/Outline.md §3; churn log 2026-09-02):
- skeleton concept from AttriMIL multi-branch attribute scoring (Apache-2.0).
  This file is a fully self-contained re-implementation: the idea plus the
  Apache-2.0 license grant (see NOTICE/AttriMIL entry in README §10), no
  AttriMIL source is copied. All region/slide constraints removed.
- stabilization by the principle of ASMIL (arXiv:2603.06658), self-written:
  normalized-sigmoid attention, a detached anchor path, token random dropping.
- fixed-count random sampling per the nnMIL idea (paper referenced only).

Ablation switches (outline §4, must be effective):
  use_stable   False -> plain gated softmax, no anchor, no token dropping
  n_attr       1     -> single attribute branch (uncertainty claim dropped)
  use_residual False -> no mean-pool residual connection
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def normalized_sigmoid(scores: torch.Tensor, keep: torch.Tensor | None = None,
                       temperature: float | torch.Tensor = 1.0, eps: float = 1e-8
                       ) -> torch.Tensor:
    """ASMIL-style normalized-sigmoid attention (self-written, paper cited).

    Softmax over-concentrates (spiking attention); normalizing sigmoid by its
    own sum keeps weights smooth and row-normalized. The learnable temperature
    (ASMIL mechanism, added churn-log 2026-09-07: v1 runs collapsed to near
    uniform attention, entropy 4.14 vs 4.16; tau gives the branches sharpness
    without spiking). Dropped tokens (keep=False) contribute a zero numerator
    and drop out of the denominator, so no NaN even when every score is very
    negative.

    scores: (B, N) logits -> (B, N) weights in [0,1], rows sum to 1.
    """
    sig = torch.sigmoid(scores / temperature)
    if keep is not None:
        sig = sig * keep.float().to(sig.device)
    return sig / (sig.sum(dim=-1, keepdim=True) + eps)


class GatedScores(nn.Module):
    """Ilse et al. ICML 2018 gated attention score function (per-token logits)."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, N, dim) -> (B, N)
        g = torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))
        return self.w(g).squeeze(-1)


class AttributeBranch(nn.Module):
    """One symmetric attribute branch: anchor attention + gated main attention.

    NOTE (simplified anchor): full ASMIL keeps a separately-updated anchor
    (EMA / parameter copy). Here the anchor is a fixed random linear map whose
    scores are detached -- it never trains, so it is a data-dependent but
    training-invariant reference that steadies the attention dynamics. Adopted
    for v1 per review; Ablation-2 (use_stable=False) measures its contribution.
    """

    def __init__(self, dim: int, hidden: int, tau: float | None = None):
        super().__init__()
        self.anchor = nn.Linear(dim, 1)
        self.main = GatedScores(dim, hidden)
        self.alpha_logit = nn.Parameter(torch.zeros(1))   # sigmoid() -> 0.5 blend init
        # normalized-sigmoid sharpness; None -> learnable (init 0.7), float -> fixed
        self.tau = nn.Parameter(torch.tensor(tau if tau is not None else 0.7))
        if tau is not None:
            self.tau.requires_grad_(False)

    def forward(self, x: torch.Tensor, keep: torch.Tensor, use_stable: bool,
                freeze_attn: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (attention-weighted bag repr (B, dim), weights (B, N))."""
        tau = torch.clamp(self.tau, min=0.1, max=10.0)
        if use_stable:
            anchor_s = self.anchor(x).squeeze(-1).detach()   # (B, N), fixed reference
            if freeze_attn:                               # two-stage warm-up: head only
                w = normalized_sigmoid(anchor_s, keep, tau)
            else:
                alpha = torch.sigmoid(self.alpha_logit)
                blend = alpha * self.main(x) + (1.0 - alpha) * anchor_s
                w = normalized_sigmoid(blend, keep, tau)
        else:                                             # Ablation-2: plain gated softmax
            main_s = self.main(x)
            if freeze_attn:
                keep_f = keep.float()
                w = keep_f / keep_f.sum(dim=-1, keepdim=True).clamp(min=1.0)
            else:
                w = F.softmax(main_s.masked_fill(~keep, -1e9), dim=-1) * keep.float()
        return (w.unsqueeze(-1) * x).sum(dim=1), w.detach()


class SAMIL(nn.Module):
    """Stable-Attribute MIL: attribute branches + stabilization + regression.

    Args:
        feat_dim: frozen feature dim (768 for DINOv2).
        hidden: attention/head width.
        dropout: projection/head dropout (not applied inside bag pooling).
        n_attr: number of symmetric attribute branches (1 or 2).
        residual: initial MeanPool residual weight (learnable scalar).
        use_stable / use_residual: ablation switches (outline §4).
    """

    def __init__(self, feat_dim: int, hidden: int = 128, dropout: float = 0.3,
                 n_attr: int = 2, residual: float = 0.3, tau: float | None = None,
                 use_stable: bool = True, use_residual: bool = True):
        super().__init__()
        assert n_attr in (1, 2), f"n_attr must be 1 or 2, got {n_attr}"
        self.n_attr = n_attr
        self.use_stable = use_stable
        self.use_residual = use_residual
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.branches = nn.ModuleList([AttributeBranch(hidden, hidden, tau) for _ in range(n_attr)])
        self.res_lambda = nn.Parameter(torch.tensor(float(residual)))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, keep: torch.Tensor | None = None,
                freeze_attn: bool = False, return_attn: bool = False):
        """x: (B, N, D). Returns pred (B,) by default; a dict when return_attn."""
        B, N, _ = x.shape
        if keep is None:
            keep = torch.ones(B, N, dtype=torch.bool, device=x.device)
        keep_f = keep.float()
        h = self.proj(x)
        mean_pool = (h * keep_f.unsqueeze(-1)).sum(1) / keep_f.sum(1).clamp(min=1.0).unsqueeze(-1)

        reprs, attn_w = [], []
        for branch in self.branches:
            r, w = branch(h, keep, self.use_stable, freeze_attn)
            if self.use_residual:
                r = r + self.res_lambda * mean_pool
            reprs.append(r)
            attn_w.append(w)

        scores = torch.cat([self.head(r) for r in reprs], dim=-1)   # (B, n_attr)
        pred = scores.mean(dim=-1)                                   # (B,)
        uncertainty = (scores.var(dim=-1, unbiased=False) if self.n_attr > 1
                       else torch.zeros(B, device=x.device))         # sample-level σ²(§5-2)

        if return_attn:
            return {"pred": pred, "uncertainty": uncertainty, "attn": torch.stack(attn_w)}
        return pred


def get_top_instances(weights: torch.Tensor, indices: np.ndarray, k: int = 5,
                      ) -> tuple[np.ndarray, np.ndarray]:
    """Top-k sampled instances of one bag -> (original index array, weights).

    weights: (N,) attention over the *sampled* tokens; indices: (N,) array
    mapping each sampled token back to its original index in the full bag
    (which is the sorted-glob order the features were extracted in).
    """
    vals, top = torch.topk(weights, k=min(k, weights.numel()))
    return indices[top.cpu().numpy()], vals.detach().cpu().numpy()


def _selfcheck() -> None:
    torch.manual_seed(0)
    m = SAMIL(feat_dim=8, hidden=16, n_attr=2).eval()
    x = torch.randn(2, 10, 8)
    keep = torch.ones(2, 10, dtype=torch.bool)

    out = m(x, keep, freeze_attn=False)
    assert out.shape == (2,) and torch.isfinite(out).all(), "pred must be finite"

    # two-stage warm-up path must also be finite (anchor only)
    out_w = m(x, keep, freeze_attn=True)
    assert torch.isfinite(out_w).all(), "warm-up (anchor) forward must be finite"

    d = m(x, keep=keep, freeze_attn=False, return_attn=True)
    assert d["attn"].shape == (2, 2, 10), d["attn"].shape           # (n_attr, B, N)
    row = d["attn"][0, 0]
    assert torch.allclose(row.sum(), torch.ones(()), atol=1e-5), "attn rows must sum to 1"
    assert d["uncertainty"].shape == (2,) and (d["uncertainty"] >= 0).all()

    # single-branch (Ablation-3) must run and report zero uncertainty
    m1 = SAMIL(feat_dim=8, hidden=16, n_attr=1, use_stable=False)
    d1 = m1(x, freeze_attn=False, return_attn=True)
    assert d1["attn"].shape == (1, 2, 10) and torch.allclose(d1["uncertainty"], torch.zeros(2))

    idx, w = get_top_instances(d["attn"][0, 0], np.arange(10), k=3)
    assert len(idx) == len(w) == 3 and w[0] >= w[-1], "top-k must be ordered desc"
    logger.info("selfcheck passed (SA-MIL forward, branches, warm-up, top-k, ablation paths)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _selfcheck()