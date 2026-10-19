# SA-MIL — reference implementation and evaluation protocol

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23471845.svg)](https://doi.org/10.5281/zenodo.23471845)

Code accompanying the ISBI 2027 submission:

> **Automated Visual Senescence Scoring of Stem-Cell Cultures from Brightfield
> Microscopy: Stabilized Attention MIL at Small Sample Size**
> Guanxi Ou, Zhanquan Wang (corresponding author)
> East China University of Science and Technology, Shanghai, China

This repository contains the scripts, the frozen-feature evaluation protocol,
and the per-run results behind the paper. It is a self-contained
re-implementation: no third-party model code is copied (see
[Provenance](#provenance)).

**Run first, read later:** point `MIL_DATA_ROOT` at the data root and run
`scripts/load_data.py`, then `scripts/extract_features.py` — everything else
consumes their outputs.

---

## 1. Scope of this release

| Included | Not included |
|---|---|
| Feature-extraction, aggregation, LOOCV, permutation-test and figure scripts | **The microscopy images** (restricted) |
| `models/sa_mil.py` — SA-MIL and all four ablations | **Extracted feature tensors** (derived from restricted images) |
| Frozen-backbone baselines (ResNet50 / DINOv2 ViT-B/14 / ConvNeXt-B) and ridge / mean-pool / max-pool / gated-attention references | Model checkpoints (aggregator heads are ~10⁵ parameters and retrain in minutes) |
| `outputs/sa_mil_results_table.csv` — the per-run table behind Table 1 | — |

## 2. Data

43 human umbilical-cord mesenchymal stem cell (hUC-MSC) samples, ~200
brightfield fields of view (FOV) per sample, each sample carrying one
continuous visual senescence score assigned by expert inspection.

**The images are not redistributed here.** They are subject to institutional
approval and are available from the authors on request. Accordingly, this
release lets a reader inspect and re-run the *protocol* end to end, but
reproducing the exact numbers requires the private dataset. We state this
plainly rather than implying a one-command reproduction.

Expected layout under `MIL_DATA_ROOT`:

```
<MIL_DATA_ROOT>/
├── huc_MSC_1/
│   ├── senescence_scores.csv          # index = sample id; senescence_score, normalized_score
│   ├── genesymbol_lg1tpm_filtered.csv # gene oracle only; never a model input
│   └── <sample_id>/{WARPED,BGAL,DAPI}/*.tif
├── huc_MSC_2/  (same layout)
└── huc_MSC_3/  (same layout)
```

## 3. Environment

```
python >= 3.10
pip install -r requirements.txt
```

Exact version pinning for the runs reported in the paper is **not yet
recorded** — see [Known gaps](#8-known-gaps). Until it is, treat the numbers
here as reproducible in protocol, not bit-for-bit.

## 4. The evaluation protocol

The protocol is the contribution; it is short enough to state completely.

1. **Bag definition.** One sample = one bag; each FOV = one instance. FOVs are
   never split across folds.
2. **Frozen features.** Backbones are frozen and used as fixed extractors;
   only the aggregation head trains. At n = 43, anything else overfits.
3. **Leave-one-sample-out CV.** 43 folds, one sample held out per fold. This
   is the only split that keeps a sample's ~200 FOVs together on both sides.
4. **Fixed-count instance sampling.** 64 FOVs sampled per bag per step,
   seeded deterministically (see 6).
5. **Aggregation head.** SA-MIL: attribute branches → normalized-sigmoid
   attention → detached anchor path → token random dropping → mean-pool
   residual. See `models/sa_mil.py`.
6. **Determinism.** Fold *i*, seed *s*: both the sampling generator and the
   model initialization are seeded with `s*1000 + i`, so any
   (config, seed) pair reproduces exactly.
7. **Reporting.** Each configuration is re-run over independent seeds;
   reported mean ± std over seeds, with a 95% bootstrap CI over folds.
8. **Permutation inference.** Labels are permuted and the *entire* pipeline,
   43-fold retraining included, is re-run to build the null distribution of
   ρ; p = (1 + #{null ≥ obs}) / (1 + n_perm). The observed and null runs share
   model init and sampling seeds, so they differ only in label assignment —
   a paired design.

## 5. Reproducing the pipeline

```bash
export MIL_DATA_ROOT=/path/to/data          # Windows: set MIL_DATA_ROOT=F:\path\to\data

python scripts/load_data.py                                  # -> outputs/sample_manifest.csv
python scripts/extract_features.py --backbone dinov2_vitb14 --view WARPED
python scripts/train_sa_mil_loocv.py --config <config> --seed 42
python scripts/collect_sa_mil_results.py                     # -> outputs/sa_mil_results_table.csv
python scripts/perm_test_sa_mil.py                           # checkpoint-resumable
python scripts/make_paper_figures.py                         # -> outputs/paper_figures/
```

Reference baselines and diagnostics:

```bash
python scripts/loocv_baseline.py            # ridge / mean-pool / max-pool
python scripts/attention_mil_baseline.py    # gated attention (Ilse et al.)
python scripts/batch_effect.py              # cross-dataset domain-shift check
python scripts/gene_oracle.py               # gene-only ceiling reference
python scripts/leak_check.py                # leakage diagnostics
python scripts/visualize_attention.py       # per-FOV attention overlays
```

## 6. Configuration names ↔ paper

The `config` column in `outputs/sa_mil_results_table.csv` uses internal names.
Mapping to the paper's ablation labels:

| Paper label | Config | Meaning | Result (mean ± std over seeds) |
|---|---|---|---|
| **Ab-1** (full SA-MIL, main) | `sa_mil_a2_stable_res` | 2 branches, stabilized, residual | 0.893 ± 0.026 (5 seeds) |
| **Ab-2** | `sa_mil_a2_nostab_res` | stabilization removed (softmax, no anchor, no dropping) | 0.892 ± 0.032 (5 seeds) |
| **Ab-3** | `sa_mil_a1_stable_res` | single branch, stabilized | 0.900 ± 0.004 (3 seeds) |
| **Ab-4** | `sa_mil_a2_stable_nores` | residual removed | 0.896 ± 0.005 (3 seeds) |
| (sensitivity) | `sa_mil_tau_0_4` | attention-temperature sweep | seed 42 only |

ρ is Spearman rank correlation between predicted and reference visual
senescence score, computed out-of-fold.

## 7. What this release does **not** claim

These are the paper's own boundaries, repeated here because released code is
often read more loosely than the manuscript.

- **No aggregation method here exceeds mean pooling.** DINOv2 mean-pool
  reaches ρ = 0.945; the best SA-MIL configuration reaches 0.900. The claim is
  about *stability and admissibility at small sample size*, not accuracy.
- **Nothing here predicts molecular or gene expression from images.** The
  gene-oracle reference (ρ = 0.990) is a ridge model over RNA-seq computed on
  the same reference labels. It is a ceiling marker for how much biological
  signal the visual score carries — it is not an image model, and it is not a
  target that these models approach.
- Attention weights are diagnostic, not instance-level explanations; at this
  redundancy the learned attention is close to uniform, and the paper reports
  that rather than a localization claim.

## 8. Known gaps

- **Unpinned dependency versions.** The versions used for the reported runs are
  not recorded in this release. Until they are, exact numeric reproduction is
  not guaranteed.
- **Feature tensors withheld** pending confirmation of the data provider's
  position on derived artifacts.
- **Backbone-resolution alignment** (running the CNN backbones at 518 px to
  match DINOv2) is not included.

## 9. Provenance

SA-MIL is assembled from three published components. The compliance path is
deliberate and is stated in the paper:

| Component | What is used | Code status |
|---|---|---|
| **AttriMIL** | multi-branch attribute scoring | **Concept only.** Apache-2.0; no source copied. All region- and slide-level constraints removed. |
| **ASMIL** | normalized-sigmoid attention, detached anchor path, token random dropping | **Re-implemented from the published principles**, self-written. |
| **nnMIL** | fixed-count random instance sampling | **Idea only**, referenced. |

Frozen feature extractors: DINOv2 ViT-B/14 (`vit_base_patch14_dinov2.lvd142m`
via `timm`), ResNet50 (`torchvision`, IMAGENET1K_V1), ConvNeXt-B
(`convnext_base.fb_in22k_ft_in1k` via `timm`).

## 10. License

MIT for the authors' code — see `LICENSE`. The attribution notices in §9 apply
to the respective upstream projects.

## 11. Citation

```bibtex
@inproceedings{ou2027samilsenescence,
  title     = {Automated Visual Senescence Scoring of Stem-Cell Cultures from
               Brightfield Microscopy: Stabilized Attention MIL at Small Sample Size},
  author    = {Ou, Guanxi and Wang, Zhanquan},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027}
}
```

Please update this entry with the final page numbers and DOI once the
proceedings are published.

If you cite the **software** itself:

```bibtex
@software{ou2026samilsenescence,
  title     = {SA-MIL: Stabilized Attention MIL for Visual Senescence
               Scoring at Small Sample Size},
  author    = {Ou, Guanxi and Wang, Zhanquan},
  year      = {2026},
  version   = {1.0.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23471845},
  url       = {https://doi.org/10.5281/zenodo.23471845}
}
```
