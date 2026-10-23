import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer

from src import features
from src.model import TextEncoder, VisionEncoder


@pytest.fixture(scope="module")
def snapshot(tiny_dataset, tmp_path_factory):
    """A pairs.parquet shaped like the BigQuery export: 2 captions per image."""
    rows = []
    recs = [json.loads(line) for line in (tiny_dataset / "train.jsonl").read_text().splitlines()]
    for i, r in enumerate(recs[:6]):
        data = (tiny_dataset / "images" / r["image"]).read_bytes()
        for k in range(2):
            rows.append({"sent_id": f"{i}_{k}", "image_id": f"{i:03d}", "caption_idx": k,
                         "caption": r["text"] if k == 0 else f"another view of {r['text']}",
                         "split": "test" if i >= 4 else "train", "filename": r["image"],
                         "gcs_uri": f"gs://bucket/raw/{r['image']}", "md5": hashlib.md5(data).hexdigest()})
    path = tmp_path_factory.mktemp("snap") / "pairs.parquet"
    pd.DataFrame(rows).sample(frac=1, random_state=0).to_parquet(path, index=False)  # shuffled on disk
    return path


def _run(snapshot, tiny_backbones, tiny_dataset, out, *extra):
    features.main(["--snapshot", str(snapshot), "--image_root", str(tiny_dataset / "images"),
                   "--out_dir", str(out), "--vision_model_name", tiny_backbones[0],
                   "--text_model_name", tiny_backbones[1], "--num_workers", "0", "--device", "cpu",
                   "--image_batch_size", "4", "--text_batch_size", "5", *extra])


def test_outputs_are_aligned_and_ordered(snapshot, tiny_backbones, tiny_dataset, tmp_path):
    out = tmp_path / "feat"
    _run(snapshot, tiny_backbones, tiny_dataset, out)
    img, txt = np.load(out / "images.npy"), np.load(out / "texts.npy")
    img_idx, txt_idx = pd.read_parquet(out / "images.parquet"), pd.read_parquet(out / "texts.parquet")
    assert img.dtype == txt.dtype == np.float16
    assert img.shape == (6, 32) and txt.shape == (12, 32)
    assert len(img_idx) == 6 and len(txt_idx) == 12
    assert img_idx["image_id"].tolist() == sorted(img_idx["image_id"])        # deterministic order
    assert txt_idx[["image_id", "caption_idx"]].values.tolist() == sorted(
        txt_idx[["image_id", "caption_idx"]].values.tolist())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["num_images"] == 6 and manifest["num_texts"] == 12
    assert manifest["snapshot_md5"] == features.file_md5(snapshot)
    assert manifest["vision_pooling"] == "cls" and manifest["text_pooling"] == "mean"


def test_features_match_the_serving_encoders(snapshot, tiny_backbones, tiny_dataset, tmp_path):
    """Cached features must equal what FusionCLIPModel's encoders compute at serving time."""
    out = tmp_path / "feat"
    _run(snapshot, tiny_backbones, tiny_dataset, out)
    images, captions = features.load_snapshot(snapshot)

    proc = AutoImageProcessor.from_pretrained(tiny_backbones[0])
    vision = VisionEncoder(tiny_backbones[0]).eval()
    first = Image.open(tiny_dataset / "images" / images.loc[0, "filename"]).convert("RGB")
    with torch.no_grad():
        expected = vision(proc(images=first, return_tensors="pt")["pixel_values"]).numpy()[0]
    np.testing.assert_allclose(np.load(out / "images.npy")[0], expected, rtol=1e-2, atol=1e-2)

    tok = AutoTokenizer.from_pretrained(tiny_backbones[1])
    text = TextEncoder(tiny_backbones[1]).eval()
    t = tok([captions.loc[3, "caption"]], return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        expected = text(t["input_ids"], t["attention_mask"]).numpy()[0]
    np.testing.assert_allclose(np.load(out / "texts.npy")[3], expected, rtol=1e-2, atol=1e-2)


def test_limit_keeps_captions_of_selected_images_only(snapshot, tiny_backbones, tiny_dataset, tmp_path):
    out = tmp_path / "feat"
    _run(snapshot, tiny_backbones, tiny_dataset, out, "--limit", "2")
    assert np.load(out / "images.npy").shape[0] == 2
    txt_idx = pd.read_parquet(out / "texts.parquet")
    assert len(txt_idx) == 4 and set(txt_idx["image_id"]) == set(
        pd.read_parquet(out / "images.parquet")["image_id"])


def test_md5_mismatch_is_detected(snapshot, tiny_dataset, tiny_backbones):
    images, _ = features.load_snapshot(snapshot)
    images.loc[0, "md5"] = "0" * 32
    ds = features.ImageFeatureDataset(images, AutoImageProcessor.from_pretrained(tiny_backbones[0]),
                                      image_root=str(tiny_dataset / "images"), verify_md5=True)
    with pytest.raises(ValueError, match="md5 mismatch"):
        ds[0]
    assert ds[1].shape[0] == 3   # the others still load
