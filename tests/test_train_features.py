"""Feature-based training -> assemble -> index-from-features -> serve, end to end (offline)."""
import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src import assemble, build_index, features, train_features
from src.metrics import grouped_recall
from src.serving.service import load_service


# ---------------- metric ----------------
def test_grouped_recall_protocol():
    img = torch.eye(3)
    # 2 captions per image; caption rows ordered by image; caption 1 of image 2 points at image 0
    txt = torch.tensor([[1., 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]])
    r = grouped_recall(img, ["a", "b", "c"], txt, ["a", "a", "b", "b", "c", "c"], ks=(1, 3))
    assert r["i2t_recall_at_1"] == 1.0                  # every image's top text is one of its own
    assert r["t2i_recall_at_1"] == pytest.approx(5 / 6)  # one caption retrieves the wrong image
    assert r["t2i_recall_at_3"] == 1.0
    assert r["rsum"] == pytest.approx(100 * (1 + 5 / 6 + 1 + 1))


def test_grouped_recall_validates_ids():
    with pytest.raises(ValueError):
        grouped_recall(torch.eye(2), ["a", "a"], torch.eye(2), ["a", "a"])
    with pytest.raises(ValueError):
        grouped_recall(torch.eye(2), ["a", "b"], torch.eye(2), ["a", "zzz"])


# ---------------- training on a learnable synthetic feature store ----------------
def _fake_store(path, n_img=120, dim_v=24, dim_t=20, caps=3, seed=0):
    """Text features are a fixed linear map of their image's features + noise, so there
    is real signal for the heads to learn (random features would make recall meaningless)."""
    g = np.random.default_rng(seed)
    img = g.normal(size=(n_img, dim_v))
    W = g.normal(size=(dim_v, dim_t))
    txt = np.concatenate([img @ W + 0.3 * g.normal(size=(n_img, dim_t)) for _ in range(caps)])
    image_ids = [f"{i:04d}" for i in range(n_img)]
    split = ["train"] * (n_img - 40) + ["val"] * 20 + ["test"] * 20
    path.mkdir(parents=True)
    np.save(path / "images.npy", img.astype(np.float16))
    np.save(path / "texts.npy", txt.astype(np.float16))
    pd.DataFrame({"image_id": image_ids, "split": split, "gcs_uri": [f"gs://b/{i}.jpg" for i in image_ids],
                  "md5": "x"}).to_parquet(path / "images.parquet")
    pd.DataFrame({"sent_id": [f"{i}_{k}" for k in range(caps) for i in image_ids],
                  "image_id": image_ids * caps, "caption_idx": [k for k in range(caps) for _ in image_ids],
                  "split": split * caps}).to_parquet(path / "texts.parquet")
    (path / "manifest.json").write_text(json.dumps({
        "snapshot_md5": "abc", "vision_model_name": "v", "text_model_name": "t", "max_length": 32}))
    return path


def test_training_learns_and_writes_outputs(tmp_path):
    feats = _fake_store(tmp_path / "feat")
    out = tmp_path / "out"
    store = train_features.FeatureStore(feats)
    untrained = train_features.evaluate(train_features.ProjectionHeads(24, 20, 16), store, "val", "cpu")

    train_features.main(["--features_dir", str(feats), "--output_dir", str(out), "--epochs", "25",
                         "--batch_size", "32", "--lr", "3e-3", "--embed_dim", "16", "--no_mlflow"])
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["val_rsum"] > untrained["rsum"] + 50          # it actually learned
    assert metrics["val_t2i_recall_at_5"] > 0.5
    assert {"test_rsum", "test_i2t_recall_at_10", "best_epoch"} <= set(metrics)
    heads = torch.load(out / "heads.pt", weights_only=False)
    assert set(heads["heads_state"]) >= {"logit_scale", "vision_proj.ln.weight", "text_proj.ln.weight"}
    assert heads["config"]["embed_dim"] == 16 and 1 <= heads["best_epoch"] <= 25


def test_captions_per_image_rejects_uncaptioned_images(tmp_path):
    feats = _fake_store(tmp_path / "feat")
    t = pd.read_parquet(feats / "texts.parquet")
    t[t["image_id"] != "0000"].to_parquet(feats / "texts.parquet")
    np.save(feats / "texts.npy", np.load(feats / "texts.npy")[(t["image_id"] != "0000").values])
    store = train_features.FeatureStore(feats)
    with pytest.raises(ValueError, match="no captions"):
        store.captions_per_image(store.split("train")[0])


def test_training_with_mlflow(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
    feats = _fake_store(tmp_path / "feat")
    train_features.main(["--features_dir", str(feats), "--output_dir", str(tmp_path / "out"),
                         "--epochs", "2", "--embed_dim", "16", "--experiment", "t"])
    run = mlflow.search_runs(experiment_names=["t"]).iloc[0]
    assert run["status"] == "FINISHED" and run["tags.snapshot_md5"] == "abc"
    assert "metrics.test_rsum" in run and "metrics.val_rsum" in run


# ---------------- real (tiny) backbones: features -> train -> assemble -> index -> serve ----------
def test_full_flow_and_cached_vs_raw_parity(tiny_backbones, tiny_dataset, tmp_path):
    rows = []
    recs = [json.loads(line) for line in (tiny_dataset / "train.jsonl").read_text().splitlines()]
    for i, r in enumerate(recs):
        for k in range(2):
            rows.append({"sent_id": f"{i}_{k}", "image_id": f"{i:03d}", "caption_idx": k,
                         "caption": r["text"], "split": ["train", "train", "val", "test"][i % 4],
                         "filename": r["image"], "gcs_uri": f"gs://b/{r['image']}", "md5": None})
    snap = tmp_path / "pairs.parquet"
    pd.DataFrame(rows).to_parquet(snap)

    feat = tmp_path / "feat"
    features.main(["--snapshot", str(snap), "--image_root", str(tiny_dataset / "images"),
                   "--out_dir", str(feat), "--vision_model_name", tiny_backbones[0],
                   "--text_model_name", tiny_backbones[1], "--num_workers", "0", "--device", "cpu"])
    out = tmp_path / "out"
    train_features.main(["--features_dir", str(feat), "--output_dir", str(out), "--epochs", "2",
                         "--batch_size", "4", "--embed_dim", "16", "--no_mlflow"])
    assemble.main(["--heads", str(out / "heads.pt"), "--out", str(out / "fusion_model.pt")])
    build_index.main(["--checkpoint", str(out / "fusion_model.pt"), "--features_dir", str(feat),
                      "--snapshot", str(snap), "--out", str(out / "gallery_index.pt"), "--device", "cpu"])

    service = load_service(str(out / "fusion_model.pt"), str(out / "gallery_index.pt"), device="cpu")
    hits = service.search_text("a red circle", top_k=3)["results"]
    assert len(hits) == 3 and {"image_id", "gcs_uri", "caption", "split"} <= set(hits[0])

    # Index built from cached fp16 features == embedding the raw image with the full model.
    from PIL import Image
    first = sorted(rows, key=lambda r: r["image_id"])[0]
    raw = service.encoder.encode_images([Image.open(tiny_dataset / "images" / first["filename"])])[0]
    cached = service.index.embeddings[0]
    assert service.index.records[0]["image_id"] == first["image_id"]
    assert F.cosine_similarity(raw, cached, dim=0) > 0.999


def test_mlflow_artifacts_go_to_artifact_root(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("MLFLOW_ARTIFACT_ROOT", (tmp_path / "remote-artifacts").as_uri())  # gs:// on the VM
    monkeypatch.chdir(tmp_path)
    feats = _fake_store(tmp_path / "feat")
    train_features.main(["--features_dir", str(feats), "--output_dir", str(tmp_path / "out"),
                         "--epochs", "1", "--embed_dim", "16", "--experiment", "remote"])
    exp = mlflow.get_experiment_by_name("remote")
    assert exp.artifact_location.endswith("remote-artifacts/remote")
    assert list((tmp_path / "remote-artifacts" / "remote").rglob("heads.pt"))
