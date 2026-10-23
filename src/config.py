import os
from dataclasses import dataclass, field

import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # Foundation models pulled from the Hugging Face Hub.
    # Env vars allow swapping backbones without code edits (e.g. local model dirs).
    vision_model_name: str = field(default_factory=lambda: os.environ.get(
        "VISION_MODEL_NAME", "facebook/dinov2-small"))
    text_model_name: str = field(default_factory=lambda: os.environ.get(
        "TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"))

    # Fusion / projection settings.
    embed_dim: int = 256
    freeze_vision: bool = True
    freeze_text: bool = True

    # Data & training.
    max_length: int = 32
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-4
    seed: int = 42
    data_dir: str = "data/synthetic"
    output_dir: str = "outputs"
    device: str = _default_device()


def config_from_checkpoint(ckpt: dict, device: str = None) -> Config:
    """Rebuild the Config stored in a checkpoint.

    Older checkpoints saved the *training* machine's device (e.g. "mps" or
    "cuda"); loading that on a CPU-only server would crash, so the device is
    always re-resolved for the current machine unless explicitly given.
    """
    known = Config.__dataclass_fields__.keys()
    values = {k: v for k, v in ckpt.get("config", {}).items() if k in known and k != "device"}
    return Config(**values, device=device or _default_device())
