"""P0-3a: frozen-backbone feature extraction for MIL bags (read-only images).

Per sample: extract one feature vector per WARPED image (or another view),
save the bag as {backbone}_{view}/{dataset}__{sample}.npy under outputs/features/.
Deterministic, resumable (skips finished samples).

Usage:
    python scripts/extract_features.py --backbone dinov2_vitb14 --view WARPED
    python scripts/extract_features.py --backbone resnet50 --view WARPED --size 224
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT / "outputs" / "sample_manifest.csv"
DATA_ROOT = Path(
    os.environ.get("MIL_DATA_ROOT", "data")
)

BACKBONES = {
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",   # 768-d CLS (timm)
    "resnet50": "tv/resnet50_IMAGENET1K_V1",              # 2048-d pooled (torchvision, cached weights)
    "convnext_base": "convnext_base.fb_in22k_ft_in1k",    # 1024-d pooled (timm)
}


def build_model(name: str, weights: Path | None, device: torch.device) -> torch.nn.Module:
    """Create a frozen backbone; optionally load explicit weights file."""
    if name.startswith("tv/"):
        import torchvision.models as tvm
        arch, tag = name[3:].split("_", 1)
        model = getattr(tvm, arch)(weights=tag)
        model.fc = torch.nn.Identity()          # expose 2048-d penultimate features
    else:
        import timm
        model = timm.create_model(name, pretrained=(weights is None))
    if weights is not None:
        if weights.suffix == ".safetensors":
            from safetensors.torch import load_file
            state = load_file(str(weights))
        else:
            state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        logger.info("loaded local weights: %s", weights)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def make_transform(input_size: int) -> T.Compose:
    import timm
    return timm.data.create_transform(input_size=input_size, is_training=False)


def extract_sample(model, transform, device, bag_paths: list[Path], batch_size: int) -> np.ndarray:
    """Return (n_images, feat_dim) features for one sample's bag."""
    feats: list[np.ndarray] = []
    for i in range(0, len(bag_paths), batch_size):
        batch_paths = bag_paths[i:i + batch_size]
        tensors = torch.stack([transform(Image.open(p).convert("RGB")) for p in batch_paths])
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            x = tensors.to(device)
            out = model.forward_features(x) if hasattr(model, "forward_features") else model(x)
        if out.ndim == 4:                                   # CNN spatial -> pooled
            out = out.mean(dim=(-2, -1))
        cls = out[:, 0] if out.ndim == 3 else out           # ViT CLS / CNN pooled
        feats.append(cls.float().cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", default="dinov2_vitb14", choices=sorted(BACKBONES))
    ap.add_argument("--view", default="WARPED")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--weights", type=Path, default=None, help="local checkpoint override")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap on samples")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(BACKBONES[args.backbone], args.weights, device)
    transform = make_transform(args.size)

    manifest = pd.read_csv(MANIFEST)
    out_dir = PROJECT / "outputs" / "features" / f"{args.backbone}_{args.view.lower()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for _, m in manifest.iterrows():
        sample_dir = DATA_ROOT / m["dataset"] / m["sample"]
        bag_paths = sorted((sample_dir / args.view).glob("*.tif"))
        if not bag_paths:
            logger.warning("no %s images for %s/%s", args.view, m["dataset"], m["sample"])
            continue
        out_path = out_dir / f"{m['dataset']}__{m['sample']}.npy"
        if out_path.exists():
            logger.info("skip (exists): %s", out_path.name)
            n = int(np.load(out_path).shape[0])
        else:
            feats = extract_sample(model, transform, device, bag_paths, args.batch_size)
            np.save(out_path, feats)
            n = int(feats.shape[0])
            logger.info("extracted %s/%s: %d imgs, feat %d", m["dataset"], m["sample"], n, feats.shape[1])
        rows.append({"dataset": m["dataset"], "sample": m["sample"], "n_images": n})
        if args.limit and len(rows) >= args.limit:
            break

    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    logger.info("done: %d bags -> %s", len(rows), out_dir)


if __name__ == "__main__":
    main()
