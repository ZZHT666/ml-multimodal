"""Assemble a deployable checkpoint: pretrained frozen backbones + trained projection heads.

The output has exactly the format src/train.py writes ({"model_state", "config"}),
so the serving code (ClipEncoder.from_checkpoint) loads it unchanged.

    python -m src.assemble --heads outputs/flickr30k/heads.pt --out outputs/flickr30k/fusion_model.pt
"""
import argparse
from pathlib import Path

import torch

from .config import Config
from .model import FusionCLIPModel

HEAD_PREFIXES = ("vision_proj.", "text_proj.", "logit_scale")


def assemble(heads_path, vision_config=None, text_config=None) -> dict:
    blob = torch.load(heads_path, map_location="cpu", weights_only=False)
    c = blob["config"]
    model = FusionCLIPModel(c["vision_model_name"], c["text_model_name"], c["embed_dim"],
                            vision_config=vision_config, text_config=text_config)
    for head in (model.vision_proj, model.text_proj):
        head.net[-1].p = c.get("dropout", 0.1)
    missing, unexpected = model.load_state_dict(blob["heads_state"], strict=False)
    if unexpected:
        raise ValueError(f"unexpected keys in heads checkpoint: {unexpected}")
    not_loaded = [k for k in missing if k.startswith(HEAD_PREFIXES)]
    if not_loaded:
        raise ValueError(f"heads checkpoint is missing: {not_loaded}")
    # Everything else that is "missing" must be the frozen backbone, which came from pretrained.
    assert all(k.startswith(("vision_encoder.backbone.", "text_encoder.backbone.")) for k in missing)

    cfg = Config(vision_model_name=c["vision_model_name"], text_model_name=c["text_model_name"],
                 embed_dim=c["embed_dim"], max_length=c.get("max_length", 32))
    config = {k: v for k, v in vars(cfg).items() if k != "device"}
    return {"model_state": model.state_dict(), "config": config,
            "best_epoch": blob.get("best_epoch"), "features_manifest": blob.get("features_manifest")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heads", default="outputs/flickr30k/heads.pt")
    p.add_argument("--out", default="outputs/flickr30k/fusion_model.pt")
    args = p.parse_args(argv)
    ckpt = assemble(args.heads)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    n = sum(v.numel() for v in ckpt["model_state"].values())
    print(f"assembled {args.out}  ({n:,} parameters, best_epoch={ckpt['best_epoch']})")


if __name__ == "__main__":
    main()
