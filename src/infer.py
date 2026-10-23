import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from .config import config_from_checkpoint
from .dataset import ImageTextDataset, make_collate_fn
from .model import FusionCLIPModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/fusion_model.pt")
    parser.add_argument("--query", required=True, help="Text query to retrieve images for.")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = config_from_checkpoint(ckpt)
    device = torch.device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)

    model = FusionCLIPModel(
        cfg.vision_model_name, cfg.text_model_name, cfg.embed_dim,
        freeze_vision=cfg.freeze_vision, freeze_text=cfg.freeze_text,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    data_dir = Path(cfg.data_dir)
    val_set = ImageTextDataset(
        data_dir / "val.jsonl", data_dir / "images", tokenizer, image_processor, cfg.max_length,
    )
    records = val_set.records
    collate_fn = make_collate_fn(tokenizer, cfg.max_length)
    loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    with torch.no_grad():
        image_embeds = []
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            image_embeds.append(model.encode_image(pixel_values))
        image_embeds = torch.cat(image_embeds)

        tokenized = tokenizer(
            [args.query], padding=True, truncation=True,
            max_length=cfg.max_length, return_tensors="pt",
        )
        query_embed = model.encode_text(
            tokenized["input_ids"].to(device), tokenized["attention_mask"].to(device),
        )

        sims = (query_embed @ image_embeds.t()).squeeze(0)
        top_k = min(args.top_k, len(records))
        top_vals, top_idx = sims.topk(top_k)

    print(f"Query: {args.query!r}")
    for rank, (score, idx) in enumerate(zip(top_vals.tolist(), top_idx.tolist(), strict=False), start=1):
        rec = records[idx]
        print(f"  #{rank}  score={score:.3f}  image={rec['image']}  caption={rec['text']!r}")


if __name__ == "__main__":
    main()
