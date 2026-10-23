"""Pre-compute frozen-backbone features for a dataset snapshot (run once, on the VM).

Both backbones are frozen, so DINOv2(image) and MiniLM(caption) give the same output
every epoch. Computing them once turns training into "fit two small MLPs on cached
vectors" (seconds per epoch on CPU) instead of re-running ~45M-param backbones on
31k images every epoch.

Train/serve consistency: features are produced by the *same* VisionEncoder /
TextEncoder classes (same pooling, same max_length) that FusionCLIPModel uses at
serving time, so the projection heads see identical inputs in both places.

    python -m src.features --snapshot data/flickr30k/pairs.parquet \
        --image_root ~/flickr30k_work/extract/flickr30k-images \
        --out_dir features/flickr30k-v1 \
        --upload gs://zht-mle-demo-2026-a1b7/features/flickr30k/v1/dinov2-small__minilm-l6

Outputs (row i of each .npy matches row i of its .parquet):
    images.npy  (N_img, D_vision) float16     images.parquet  image_id, split, gcs_uri, md5
    texts.npy   (N_cap, D_text)   float16     texts.parquet   sent_id, image_id, caption_idx, split
    manifest.json   backbones, pooling, max_length, snapshot md5, counts, timings
"""
import argparse
import hashlib
import io
import json
import os
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import Config
from .model import TextEncoder, VisionEncoder


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_snapshot(path):
    """pairs.parquet (one row per caption) -> (unique images, captions), deterministically ordered."""
    pairs = pd.read_parquet(path)
    pairs = pairs.sort_values(["image_id", "caption_idx"], kind="stable").reset_index(drop=True)
    images = (pairs.drop_duplicates("image_id")[["image_id", "filename", "gcs_uri", "md5", "split"]]
              .reset_index(drop=True))
    captions = pairs[["sent_id", "image_id", "caption_idx", "caption", "split"]].reset_index(drop=True)
    return images, captions


# ------------------------------------------------------------------ image loading
_gcs_client = None


def _read_gcs(uri: str) -> bytes:
    global _gcs_client
    if _gcs_client is None:                 # one client per DataLoader worker process
        from google.cloud import storage
        _gcs_client = storage.Client()
    bucket, _, blob = uri[len("gs://"):].partition("/")
    return _gcs_client.bucket(bucket).blob(blob).download_as_bytes()


class ImageFeatureDataset(Dataset):
    """Reads each image from a local mirror (fast) or straight from its gcs_uri."""

    def __init__(self, images: pd.DataFrame, image_processor, image_root: Optional[str] = None,
                 verify_md5: bool = False):
        self.records = images.to_dict("records")
        self.processor = image_processor
        self.root = Path(image_root).expanduser() if image_root else None
        self.verify_md5 = verify_md5

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rec = self.records[i]
        data = ((self.root / rec["filename"]).read_bytes() if self.root
                else _read_gcs(rec["gcs_uri"]))
        if self.verify_md5 and rec.get("md5") and hashlib.md5(data).hexdigest() != rec["md5"]:
            raise ValueError(f"md5 mismatch for {rec['filename']}: the image is not the "
                             "version pinned in the snapshot")
        image = Image.open(io.BytesIO(data)).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


# ------------------------------------------------------------------ encoding
@torch.inference_mode()
def encode_images(images, encoder: VisionEncoder, image_processor, device, batch_size=64,
                  num_workers=4, image_root=None, verify_md5=False, log_every=50) -> np.ndarray:
    ds = ImageFeatureDataset(images, image_processor, image_root, verify_md5)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    out, start = [], time.perf_counter()
    for step, pixel_values in enumerate(loader, start=1):
        out.append(encoder(pixel_values.to(device), pooled=True).float().cpu().numpy())
        if step % log_every == 0:
            done = min(step * batch_size, len(ds))
            rate = done / (time.perf_counter() - start)
            print(f"  images {done}/{len(ds)}  {rate:.1f} img/s  eta {(len(ds) - done) / rate:.0f}s",
                  flush=True)
    return np.concatenate(out).astype(np.float16)


@torch.inference_mode()
def encode_texts(captions, encoder: TextEncoder, tokenizer, device, max_length=32,
                 batch_size=512, log_every=50) -> np.ndarray:
    texts = captions["caption"].tolist()
    out, start = [], time.perf_counter()
    for step, i in enumerate(range(0, len(texts), batch_size), start=1):
        tok = tokenizer(texts[i:i + batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors="pt")
        out.append(encoder(tok["input_ids"].to(device), tok["attention_mask"].to(device),
                           pooled=True).float().cpu().numpy())
        if step % log_every == 0:
            done = min(i + batch_size, len(texts))
            print(f"  texts {done}/{len(texts)}  {done / (time.perf_counter() - start):.0f} txt/s",
                  flush=True)
    return np.concatenate(out).astype(np.float16)


def upload_dir(local_dir: Path, gcs_prefix: str) -> None:
    from google.cloud import storage

    bucket_name, _, prefix = gcs_prefix[len("gs://"):].partition("/")
    bucket = storage.Client().bucket(bucket_name)
    for f in sorted(local_dir.iterdir()):
        bucket.blob(f"{prefix.rstrip('/')}/{f.name}").upload_from_filename(str(f))
        print(f"  uploaded {gcs_prefix.rstrip('/')}/{f.name}")


def main(argv=None):
    from transformers import AutoImageProcessor, AutoTokenizer

    cfg = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--snapshot", default="data/flickr30k/pairs.parquet")
    p.add_argument("--image_root", default=None,
                   help="local mirror of the raw images; omit to read each image from its gcs_uri")
    p.add_argument("--out_dir", default="features/flickr30k-v1")
    p.add_argument("--vision_model_name", default=cfg.vision_model_name)
    p.add_argument("--text_model_name", default=cfg.text_model_name)
    p.add_argument("--max_length", type=int, default=cfg.max_length)
    p.add_argument("--image_batch_size", type=int, default=64)
    p.add_argument("--text_batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    p.add_argument("--device", default=cfg.device)
    p.add_argument("--limit", type=int, default=None, help="only the first N images (smoke test)")
    p.add_argument("--verify_md5", action="store_true", help="check every image against the snapshot")
    p.add_argument("--upload", default=None, help="gs:// prefix to copy the outputs to")
    args = p.parse_args(argv)

    torch.set_num_threads(os.cpu_count() or 1)
    device = torch.device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    images, captions = load_snapshot(args.snapshot)
    if args.limit:
        images = images.head(args.limit)
        captions = captions[captions["image_id"].isin(images["image_id"])].reset_index(drop=True)
    print(f"snapshot: {len(images)} images, {len(captions)} captions  device={device}")

    timings = {}
    t0 = time.perf_counter()
    vision = VisionEncoder(args.vision_model_name, freeze=True).to(device).eval()
    img_feats = encode_images(images, vision, AutoImageProcessor.from_pretrained(args.vision_model_name),
                              device, args.image_batch_size, args.num_workers, args.image_root,
                              args.verify_md5)
    timings["images_s"] = round(time.perf_counter() - t0, 1)
    del vision

    t0 = time.perf_counter()
    text = TextEncoder(args.text_model_name, freeze=True).to(device).eval()
    txt_feats = encode_texts(captions, text, AutoTokenizer.from_pretrained(args.text_model_name),
                             device, args.max_length, args.text_batch_size)
    timings["texts_s"] = round(time.perf_counter() - t0, 1)

    np.save(out / "images.npy", img_feats)
    images[["image_id", "split", "gcs_uri", "md5"]].to_parquet(out / "images.parquet", index=False)
    np.save(out / "texts.npy", txt_feats)
    captions[["sent_id", "image_id", "caption_idx", "split"]].to_parquet(out / "texts.parquet", index=False)

    manifest = {
        "snapshot": str(args.snapshot),
        "snapshot_md5": file_md5(args.snapshot),   # which data version these features came from
        "vision_model_name": args.vision_model_name, "vision_pooling": "cls",
        "text_model_name": args.text_model_name, "text_pooling": "mean", "max_length": args.max_length,
        "dtype": "float16",
        "num_images": int(img_feats.shape[0]), "vision_dim": int(img_feats.shape[1]),
        "num_texts": int(txt_feats.shape[0]), "text_dim": int(txt_feats.shape[1]),
        "limit": args.limit, "timings_s": timings, "device": str(device),
        "torch": torch.__version__, "python": platform.python_version(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

    if args.upload:
        upload_dir(out, args.upload)


if __name__ == "__main__":
    main()
