"""Shared fixtures.

Everything here is built fully offline: tiny randomly-initialized DINOv2 / BERT
backbones are saved to a temp dir and then loaded by *path*, which exercises the
exact same AutoModel / AutoTokenizer / AutoImageProcessor code paths as the real
`facebook/dinov2-small` + `all-MiniLM-L6-v2` models -- without network access, a
GPU, or a 180 MB checkpoint. That is what lets these tests run in CI in seconds.
"""
import json
import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

EMBED_DIM = 16
VOCAB = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "an", "on", "background",
         "red", "green", "blue", "yellow", "purple", "orange", "white", "black", "gray",
         "circle", "square", "triangle", "warmup", "something", "small", "shape", "dark"]


@pytest.fixture(scope="session")
def tiny_backbones(tmp_path_factory):
    from transformers import (
        BertConfig,
        BertModel,
        BertTokenizerFast,
        BitImageProcessor,
        Dinov2Config,
        Dinov2Model,
    )

    torch.manual_seed(0)
    root = tmp_path_factory.mktemp("hf")

    vision_dir = root / "tiny-dinov2"
    Dinov2Model(Dinov2Config(image_size=28, patch_size=14, hidden_size=32, num_hidden_layers=1,
                             num_attention_heads=2, intermediate_size=64)).save_pretrained(vision_dir)
    BitImageProcessor(size={"shortest_edge": 28}, crop_size={"height": 28, "width": 28},
                      do_convert_rgb=True).save_pretrained(vision_dir)

    text_dir = root / "tiny-bert"
    BertModel(BertConfig(vocab_size=len(VOCAB), hidden_size=32, num_hidden_layers=1,
                         num_attention_heads=2, intermediate_size=64,
                         max_position_embeddings=64)).save_pretrained(text_dir)
    vocab_file = root / "vocab.txt"
    vocab_file.write_text("\n".join(VOCAB) + "\n")
    BertTokenizerFast(vocab_file=str(vocab_file)).save_pretrained(text_dir)
    return str(vision_dir), str(text_dir)


@pytest.fixture(scope="session")
def tiny_model(tiny_backbones):
    from src.model import FusionCLIPModel

    torch.manual_seed(0)
    return FusionCLIPModel(*tiny_backbones, embed_dim=EMBED_DIM).eval()


@pytest.fixture(scope="session")
def tiny_ckpt(tiny_backbones, tiny_model, tmp_path_factory):
    """A checkpoint in exactly the format src/train.py writes."""
    from src.config import Config

    cfg = Config(vision_model_name=tiny_backbones[0], text_model_name=tiny_backbones[1],
                 embed_dim=EMBED_DIM, device="mps")  # deliberately a device CI doesn't have
    path = tmp_path_factory.mktemp("ckpt") / "fusion_model.pt"
    torch.save({"model_state": tiny_model.state_dict(), "config": vars(cfg)}, path)
    return path


@pytest.fixture(scope="session")
def tiny_dataset(tmp_path_factory):
    """A few synthetic image/caption pairs, written like data/generate_synthetic.py does."""
    from generate_synthetic import build_split

    out = tmp_path_factory.mktemp("synthetic")
    images = out / "images"
    images.mkdir()
    rng = random.Random(0)
    for name, n, start in (("train", 12, 0), ("val", 6, 12)):
        records = build_split(images, n=n, rng=rng, start_idx=start)
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    return out


@pytest.fixture(scope="session")
def tiny_index(tiny_ckpt, tiny_dataset, tmp_path_factory):
    from PIL import Image

    from src.serving.index import RetrievalIndex
    from src.serving.service import ClipEncoder

    encoder = ClipEncoder.from_checkpoint(tiny_ckpt, device="cpu")
    records = [json.loads(line) for line in (tiny_dataset / "val.jsonl").read_text().splitlines()]
    images = [Image.open(tiny_dataset / "images" / r["image"]) for r in records]
    index = RetrievalIndex(encoder.encode_images(images), records,
                           meta={"model_fingerprint": encoder.fingerprint})
    path = tmp_path_factory.mktemp("index") / "gallery_index.pt"
    index.save(path)
    return path
