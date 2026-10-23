import argparse
import json
import random
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from .config import Config
from .dataset import ImageTextDataset, make_collate_fn
from .metrics import retrieval_recall
from .model import FusionCLIPModel, clip_contrastive_loss


@torch.no_grad()
def evaluate(model, loader, device, captions):
    model.eval()
    image_embeds, text_embeds = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image_embeds.append(model.encode_image(pixel_values))
        text_embeds.append(model.encode_text(input_ids, attention_mask))
    return retrieval_recall(torch.cat(image_embeds), torch.cat(text_embeds), captions)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def dvc_data_md5(data_dir: str) -> str:
    """md5 of the dataset version from its .dvc pointer (or dvc.lock), if any."""
    pointer = Path(str(data_dir).rstrip("/") + ".dvc")
    if pointer.exists():
        for line in pointer.read_text().splitlines():
            line = line.strip().lstrip("- ")
            if line.startswith("md5:"):
                return line.split(":", 1)[1].strip()
    return "untracked"


class _NullTracker:
    """Stand-in when MLflow is disabled, so the training loop has one code path."""

    def log_params(self, *_a, **_k): pass
    def log_metrics(self, *_a, **_k): pass
    def set_tags(self, *_a, **_k): pass
    def log_artifact(self, *_a, **_k): pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--vision_model_name", default=None)
    parser.add_argument("--text_model_name", default=None)
    parser.add_argument("--no_mlflow", action="store_true", help="Disable MLflow tracking.")
    parser.add_argument("--experiment", default="clip-retrieval", help="MLflow experiment name.")
    parser.add_argument("--register_model", default=None,
                        help="If set, also log the model with mlflow.pytorch and register it "
                             "under this name in the MLflow Model Registry.")
    args = parser.parse_args()

    cfg = Config()
    for field in ("data_dir", "output_dir", "epochs", "batch_size", "lr", "embed_dim",
                  "vision_model_name", "text_model_name"):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, field, value)

    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)

    data_dir = Path(cfg.data_dir)
    collate_fn = make_collate_fn(tokenizer, cfg.max_length)

    train_set = ImageTextDataset(
        data_dir / "train.jsonl", data_dir / "images", tokenizer, image_processor, cfg.max_length,
    )
    val_set = ImageTextDataset(
        data_dir / "val.jsonl", data_dir / "images", tokenizer, image_processor, cfg.max_length,
    )
    val_captions = [r["text"] for r in val_set.records]
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = FusionCLIPModel(
        cfg.vision_model_name, cfg.text_model_name, cfg.embed_dim,
        freeze_vision=cfg.freeze_vision, freeze_text=cfg.freeze_text,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    if args.no_mlflow:
        tracker = _NullTracker()
    else:
        import mlflow  # tracking URI from MLFLOW_TRACKING_URI (Makefile: sqlite:///mlflow.db)
        mlflow.set_experiment(args.experiment)
        mlflow.start_run()
        tracker = mlflow

    try:
        params = {k: v for k, v in vars(cfg).items() if k != "device"}
        tracker.log_params(params)
        tracker.set_tags({
            "git_commit": git_commit(),
            "data_version_md5": dvc_data_md5(cfg.data_dir),
            "device": cfg.device,
            "n_train": len(train_set),
            "n_val": len(val_set),
            "trainable_params": n_trainable,
        })

        metrics = {}
        for epoch in range(cfg.epochs):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                logits_per_image, logits_per_text = model(pixel_values, input_ids, attention_mask)
                loss = clip_contrastive_loss(logits_per_image, logits_per_text)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Same clamp as OpenAI CLIP: keeps the learned temperature <= 100.
                with torch.no_grad():
                    model.logit_scale.clamp_(0, np.log(100))
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            metrics = {"train_loss": avg_loss, **evaluate(model, val_loader, device, val_captions),
                       "temperature": float(1 / model.logit_scale.detach().exp())}
            tracker.log_metrics(metrics, step=epoch + 1)
            print(f"epoch {epoch + 1}/{cfg.epochs}  " +
                  "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        ckpt_path = out_dir / "fusion_model.pt"
        torch.save({"model_state": model.state_dict(), "config": params}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

        # DVC reads this file as the stage's metrics (`dvc metrics show / diff`).
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        tracker.log_artifact(str(ckpt_path), artifact_path="checkpoint")

        if args.register_model and not args.no_mlflow:
            import mlflow.pytorch
            mlflow.pytorch.log_model(model.cpu(), artifact_path="model",
                                     registered_model_name=args.register_model)
            print(f"Registered model '{args.register_model}' in the MLflow Model Registry")
    except BaseException:
        if not args.no_mlflow:
            mlflow.end_run(status="FAILED")
        raise
    if not args.no_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
