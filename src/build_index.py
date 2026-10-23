"""Offline step: encode every gallery image once and save the embeddings.

Serving then only has to encode the *query* (one short text) per request,
which is what keeps online latency low -- the expensive image encoding is
paid once here, not per request.

    python -m src.build_index --checkpoint outputs/fusion_model.pt \
        --manifests data/synthetic/train.jsonl data/synthetic/val.jsonl \
        --image_root data/synthetic/images --out outputs/gallery_index.pt
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .serving.index import RetrievalIndex
from .serving.service import ClipEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/fusion_model.pt")
    parser.add_argument("--manifests", nargs="+",
                        default=["data/synthetic/train.jsonl", "data/synthetic/val.jsonl"])
    parser.add_argument("--image_root", default="data/synthetic/images")
    parser.add_argument("--out", default="outputs/gallery_index.pt")
    parser.add_argument("--uri_prefix", default=None,
                        help="Optional, e.g. gs://<bucket>/images/ -- returned as 'uri' in results.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    encoder = ClipEncoder.from_checkpoint(args.checkpoint, device=args.device)

    records = []
    for manifest in args.manifests:
        with open(manifest) as f:
            records.extend(json.loads(line) for line in f if line.strip())
    if args.uri_prefix:
        for r in records:
            r["uri"] = args.uri_prefix.rstrip("/") + "/" + r["image"]

    root = Path(args.image_root)
    chunks = []
    for i in range(0, len(records), args.batch_size):
        batch = records[i:i + args.batch_size]
        images = [Image.open(root / r["image"]).convert("RGB") for r in batch]
        chunks.append(encoder.encode_images(images).cpu())
    embeddings = torch.cat(chunks)

    index = RetrievalIndex(embeddings, records, meta={
        "model_fingerprint": encoder.fingerprint,   # == DVC md5 of the checkpoint file
        "manifests": args.manifests,
        "embed_dim": embeddings.size(1),
    })
    index.save(args.out)
    print(f"Indexed {len(index)} images (dim={index.dim}) -> {args.out}")


if __name__ == "__main__":
    main()
