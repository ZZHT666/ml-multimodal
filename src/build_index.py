"""Offline step: encode every gallery image once and save the embeddings.

Serving then only has to encode the *query* (one short text) per request,
which is what keeps online latency low -- the expensive image encoding is
paid once here, not per request.

Two sources for the gallery images:

  # from cached backbone features (Flickr30k): only the projection head runs, ~seconds
  python -m src.build_index --checkpoint outputs/flickr30k/fusion_model.pt \
      --features_dir features/flickr30k-v1 --snapshot data/flickr30k/pairs.parquet \
      --out outputs/flickr30k/gallery_index.pt

  # from image files (synthetic data): runs the full vision backbone per image
  python -m src.build_index --checkpoint outputs/fusion_model.pt \
      --manifests data/synthetic/train.jsonl data/synthetic/val.jsonl \
      --image_root data/synthetic/images --out outputs/gallery_index.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .serving.index import RetrievalIndex
from .serving.service import ClipEncoder


def from_features(encoder: ClipEncoder, features_dir, snapshot=None, splits=None):
    d = Path(features_dir)
    feats = torch.from_numpy(np.load(d / "images.npy").astype(np.float32))
    idx = pd.read_parquet(d / "images.parquet")
    if splits:
        keep = idx["split"].isin(splits).values
        feats, idx = feats[keep], idx[keep].reset_index(drop=True)
    if snapshot:   # attach the first caption of each image, purely for readable results
        first = (pd.read_parquet(snapshot, columns=["image_id", "caption_idx", "caption"])
                 .query("caption_idx == 0")[["image_id", "caption"]])
        idx = idx.merge(first, on="image_id", how="left")
    with torch.inference_mode():
        emb = F.normalize(encoder.model.vision_proj(feats.to(encoder.device)), dim=-1).cpu()
    records = idx.drop(columns=["md5"], errors="ignore").to_dict("records")
    return emb, records, {"features_dir": str(features_dir), "splits": splits,
                          "features_manifest": json.loads((d / "manifest.json").read_text())}


def from_images(encoder: ClipEncoder, manifests, image_root, uri_prefix=None, batch_size=64):
    records = []
    for manifest in manifests:
        with open(manifest) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    if uri_prefix:
        for r in records:
            r["uri"] = uri_prefix.rstrip("/") + "/" + r["image"]
    root = Path(image_root)
    chunks = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        chunks.append(encoder.encode_images([Image.open(root / r["image"]).convert("RGB")
                                             for r in batch]).cpu())
    return torch.cat(chunks), records, {"manifests": manifests}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default="outputs/fusion_model.pt")
    parser.add_argument("--out", default="outputs/gallery_index.pt")
    parser.add_argument("--features_dir", default=None, help="use cached backbone features")
    parser.add_argument("--snapshot", default=None, help="pairs.parquet, to show a caption per image")
    parser.add_argument("--splits", nargs="*", default=None, help="e.g. test (default: all images)")
    parser.add_argument("--manifests", nargs="+",
                        default=["data/synthetic/train.jsonl", "data/synthetic/val.jsonl"])
    parser.add_argument("--image_root", default="data/synthetic/images")
    parser.add_argument("--uri_prefix", default=None,
                        help="Optional, e.g. gs://<bucket>/images/ -- returned as 'uri' in results.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    encoder = ClipEncoder.from_checkpoint(args.checkpoint, device=args.device)
    if args.features_dir:
        embeddings, records, meta = from_features(encoder, args.features_dir, args.snapshot, args.splits)
    else:
        embeddings, records, meta = from_images(encoder, args.manifests, args.image_root,
                                                args.uri_prefix, args.batch_size)

    index = RetrievalIndex(embeddings, records, meta={
        "model_fingerprint": encoder.fingerprint,   # == DVC md5 of the checkpoint file
        "embed_dim": embeddings.size(1), **meta,
    })
    index.save(args.out)
    print(f"Indexed {len(index)} images (dim={index.dim}) -> {args.out}")


if __name__ == "__main__":
    main()
