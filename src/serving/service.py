"""Model loading + query encoding + search, with per-stage latency timing."""
import base64
import io
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import torch
from PIL import Image

from ..config import config_from_checkpoint
from ..model import FusionCLIPModel
from .artifacts import file_md5
from .index import RetrievalIndex


class ClipEncoder:
    """Wraps FusionCLIPModel + its tokenizer / image processor for single-request inference."""

    def __init__(self, model: FusionCLIPModel, tokenizer, image_processor,
                 device: str = "cpu", max_length: int = 32, fingerprint: Optional[str] = None):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = torch.device(device)
        self.max_length = max_length
        self.fingerprint = fingerprint

    @classmethod
    def from_checkpoint(cls, ckpt_path, device: Optional[str] = None) -> "ClipEncoder":
        from transformers import AutoImageProcessor, AutoTokenizer

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = config_from_checkpoint(ckpt, device=device)
        # pretrained=False: build the backbones from their configs only -- every weight
        # (frozen backbones included) comes from our checkpoint, so nothing is downloaded
        # twice and the container only needs the HF config/tokenizer files baked in.
        model = FusionCLIPModel(
            cfg.vision_model_name, cfg.text_model_name, cfg.embed_dim,
            freeze_vision=cfg.freeze_vision, freeze_text=cfg.freeze_text, pretrained=False,
        )
        model.load_state_dict(ckpt["model_state"])
        return cls(model,
                   AutoTokenizer.from_pretrained(cfg.text_model_name),
                   AutoImageProcessor.from_pretrained(cfg.vision_model_name),
                   device=cfg.device, max_length=cfg.max_length,
                   fingerprint=file_md5(ckpt_path))

    @property
    def embed_dim(self) -> int:
        return self.model.text_proj.ln.normalized_shape[0]

    @torch.inference_mode()
    def encode_text(self, texts: Sequence[str]) -> torch.Tensor:
        tok = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=self.max_length, return_tensors="pt")
        return self.model.encode_text(tok["input_ids"].to(self.device),
                                      tok["attention_mask"].to(self.device))

    @torch.inference_mode()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        pixel_values = self.image_processor(images=list(images), return_tensors="pt")["pixel_values"]
        return self.model.encode_image(pixel_values.to(self.device))

    def sync(self) -> None:
        # CUDA kernels are async: without a sync, the timer measures kernel *launch*, not compute.
        if self.device.type == "cuda":
            torch.cuda.synchronize()


def decode_image(image_b64: str) -> Image.Image:
    try:
        return Image.open(io.BytesIO(base64.b64decode(image_b64, validate=True))).convert("RGB")
    except Exception as e:  # bad base64 / not an image
        raise ValueError(f"could not decode image_b64: {e}") from e


class RetrievalService:
    def __init__(self, encoder, index: RetrievalIndex):
        if getattr(encoder, "embed_dim", index.dim) != index.dim:
            raise ValueError(f"encoder dim {encoder.embed_dim} != index dim {index.dim}")
        index.check_model(getattr(encoder, "fingerprint", None))
        self.encoder = encoder
        self.index = index.to(getattr(encoder, "device", "cpu"))

    @property
    def model_version(self) -> Optional[str]:
        return getattr(self.encoder, "fingerprint", None)

    @contextmanager
    def _timed(self, timings: Dict[str, float], name: str):
        sync = getattr(self.encoder, "sync", lambda: None)
        start = time.perf_counter()
        yield
        sync()
        timings[name] = round((time.perf_counter() - start) * 1000, 3)

    def search_text(self, query: str, top_k: int = 5) -> Dict:
        return self._search(lambda: self.encoder.encode_text([query]), top_k)

    def search_image(self, image_b64: str, top_k: int = 5) -> Dict:
        timings: Dict[str, float] = {}
        with self._timed(timings, "decode_ms"):
            image = decode_image(image_b64)
        return self._search(lambda: self.encoder.encode_images([image]), top_k, timings)

    def _search(self, encode_fn, top_k: int, timings: Optional[Dict] = None) -> Dict:
        timings = timings if timings is not None else {}
        start = time.perf_counter()
        with self._timed(timings, "encode_ms"):
            query_embed = encode_fn()
        with self._timed(timings, "search_ms"):
            hits: List[Dict] = self.index.search(query_embed, top_k)[0]
        timings["inference_ms"] = round((time.perf_counter() - start) * 1000, 3) + timings.get("decode_ms", 0)
        return {"results": hits, "timing_ms": timings}


def load_service(model_path: str, index_path: str, device: Optional[str] = None) -> RetrievalService:
    from .artifacts import resolve

    encoder = ClipEncoder.from_checkpoint(resolve(model_path), device=device)
    return RetrievalService(encoder, RetrievalIndex.load(resolve(index_path)))
