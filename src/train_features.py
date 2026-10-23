"""Train the projection heads on cached backbone features (output of src/features.py).

Same model as FusionCLIPModel minus the frozen backbones: two ProjectionHeads + a
learned temperature, trained with symmetric InfoNCE. Parameter names match
FusionCLIPModel ("vision_proj.*", "text_proj.*", "logit_scale"), so src/assemble.py
can drop these weights straight into the full serving model.

Sampling: each epoch pairs every training image with ONE of its 5 captions (random
per epoch). Feeding all 5 captions as separate pairs would put the same image in a
batch several times, and InfoNCE would then treat its own other captions as negatives.

Model selection: the epoch with the best val rsum is kept; test is evaluated once,
at the end, on that checkpoint (never used to pick anything).

    python -m src.train_features --features_dir features/flickr30k-v1 --output_dir outputs/flickr30k
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import grouped_recall
from .model import ProjectionHead, clip_contrastive_loss
from .train import _NullTracker, git_commit


class ProjectionHeads(nn.Module):
    def __init__(self, vision_dim: int, text_dim: int, embed_dim: int = 256, p_drop: float = 0.1):
        super().__init__()
        self.vision_proj = ProjectionHead(vision_dim, embed_dim, p_drop)
        self.text_proj = ProjectionHead(text_dim, embed_dim, p_drop)
        self.logit_scale = nn.Parameter(torch.tensor(float(np.log(1 / 0.07))))

    def encode_image(self, feats):
        return F.normalize(self.vision_proj(feats), dim=-1)

    def encode_text(self, feats):
        return F.normalize(self.text_proj(feats), dim=-1)


class FeatureStore:
    """Cached features + their index tables, split-aware."""

    def __init__(self, features_dir):
        d = Path(features_dir)
        self.manifest = json.loads((d / "manifest.json").read_text())
        self.img = torch.from_numpy(np.load(d / "images.npy").astype(np.float32))
        self.txt = torch.from_numpy(np.load(d / "texts.npy").astype(np.float32))
        self.img_idx = pd.read_parquet(d / "images.parquet")
        self.txt_idx = pd.read_parquet(d / "texts.parquet")
        if len(self.img) != len(self.img_idx) or len(self.txt) != len(self.txt_idx):
            raise ValueError("features and their index parquet files are misaligned")

    def split(self, name: str):
        """(image rows, text rows, image_ids, text_image_ids) for one split."""
        img_rows = np.flatnonzero(self.img_idx["split"].values == name)
        txt_rows = np.flatnonzero(self.txt_idx["split"].values == name)
        return (img_rows, txt_rows, self.img_idx["image_id"].values[img_rows].tolist(),
                self.txt_idx["image_id"].values[txt_rows].tolist())

    def captions_per_image(self, img_rows):
        """For each image row: array of its caption rows (for per-epoch caption sampling)."""
        by_image = self.txt_idx.reset_index().groupby("image_id")["index"].apply(np.array).to_dict()
        ids = self.img_idx["image_id"].values[img_rows]
        missing = [i for i in ids if i not in by_image]
        if missing:
            raise ValueError(f"{len(missing)} training images have no captions")
        return [by_image[i] for i in ids]


@torch.no_grad()
def evaluate(model, store: FeatureStore, split: str, device) -> dict:
    model.eval()
    img_rows, txt_rows, img_ids, txt_img_ids = store.split(split)
    img = model.encode_image(store.img[img_rows].to(device))
    txt = model.encode_text(store.txt[txt_rows].to(device))
    return grouped_recall(img, img_ids, txt, txt_img_ids)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features_dir", default="features/flickr30k-v1")
    p.add_argument("--output_dir", default="outputs/flickr30k")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--embed_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--no_mlflow", action="store_true")
    p.add_argument("--experiment", default="flickr30k-heads")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    store = FeatureStore(args.features_dir)
    train_rows, _, _, _ = store.split("train")
    caps = store.captions_per_image(train_rows)
    rng = np.random.default_rng(args.seed)
    print(f"train images={len(train_rows)}  val images={len(store.split('val')[0])}  "
          f"test images={len(store.split('test')[0])}")

    model = ProjectionHeads(store.img.size(1), store.txt.size(1), args.embed_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = (len(train_rows) + args.batch_size - 1) // args.batch_size
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.1)

    if args.no_mlflow:
        tracker = _NullTracker()
    else:
        import mlflow

        from .tracking import set_experiment
        set_experiment(args.experiment)
        mlflow.start_run()
        tracker = mlflow

    try:
        tracker.log_params({k: v for k, v in vars(args).items() if k not in ("no_mlflow", "experiment")})
        m = store.manifest
        tracker.set_tags({"git_commit": git_commit(), "snapshot_md5": m["snapshot_md5"],
                          "vision_model_name": m["vision_model_name"],
                          "text_model_name": m["text_model_name"],
                          "n_train_images": len(train_rows)})

        best = {"rsum": -1.0}
        for epoch in range(1, args.epochs + 1):
            model.train()
            t0 = time.perf_counter()
            # one random caption per image this epoch
            pick = np.array([c[rng.integers(len(c))] for c in caps])
            order = rng.permutation(len(train_rows))
            total = 0.0
            for b in range(steps_per_epoch):
                idx = order[b * args.batch_size:(b + 1) * args.batch_size]
                img = model.encode_image(store.img[train_rows[idx]].to(device))
                txt = model.encode_text(store.txt[pick[idx]].to(device))
                logits = model.logit_scale.exp() * img @ txt.t()
                loss = clip_contrastive_loss(logits, logits.t())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    model.logit_scale.clamp_(0, np.log(100))
                total += loss.item()

            val = evaluate(model, store, "val", device)
            metrics = {"train_loss": total / steps_per_epoch, "lr": scheduler.get_last_lr()[0],
                       "temperature": float(1 / model.logit_scale.detach().exp()),
                       **{f"val_{k}": v for k, v in val.items()}}
            tracker.log_metrics(metrics, step=epoch)
            print(f"epoch {epoch:>3}/{args.epochs}  loss={metrics['train_loss']:.4f}  "
                  f"val R@1 t2i={val['t2i_recall_at_1']:.3f} i2t={val['i2t_recall_at_1']:.3f}  "
                  f"rsum={val['rsum']:.1f}  ({time.perf_counter() - t0:.1f}s)")
            if val["rsum"] > best["rsum"]:
                best = {"rsum": val["rsum"], "epoch": epoch,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

        model.load_state_dict(best["state"])
        test = evaluate(model, store, "test", device)
        val = evaluate(model, store, "val", device)
        print(f"best epoch {best['epoch']}  TEST: " +
              "  ".join(f"{k}={v:.3f}" for k, v in test.items()))

        heads_path = out / "heads.pt"
        torch.save({"heads_state": best["state"],
                    "config": {"embed_dim": args.embed_dim, "dropout": args.dropout,
                               "vision_model_name": m["vision_model_name"],
                               "text_model_name": m["text_model_name"],
                               "max_length": m["max_length"]},
                    "features_manifest": m, "best_epoch": best["epoch"]}, heads_path)
        final = {"best_epoch": best["epoch"], **{f"test_{k}": v for k, v in test.items()},
                 **{f"val_{k}": v for k, v in val.items()}}
        (out / "metrics.json").write_text(json.dumps(final, indent=2))
        tracker.log_metrics({k: v for k, v in final.items() if k != "best_epoch"})
        tracker.set_tags({"best_epoch": best["epoch"]})
        tracker.log_artifact(str(heads_path), artifact_path="heads")
        print(f"saved {heads_path} and {out / 'metrics.json'}")
    except BaseException:
        if not args.no_mlflow:
            mlflow.end_run(status="FAILED")
        raise
    if not args.no_mlflow:
        mlflow.end_run()


if __name__ == "__main__":
    main()
