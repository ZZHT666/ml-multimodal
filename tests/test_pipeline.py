"""Smoke test of the offline pipeline: train (1 epoch) -> checkpoint -> build_index -> serve."""
import json
import sys

import pytest

from src import build_index, train
from src.serving.service import load_service


def _run(module, monkeypatch, argv):
    monkeypatch.setattr(sys, "argv", ["prog", *argv])
    module.main()


@pytest.mark.parametrize("use_mlflow", [False, True])
def test_train_then_index_then_serve(tiny_backbones, tiny_dataset, tmp_path, monkeypatch, use_mlflow):
    out = tmp_path / "outputs"
    args = ["--data_dir", str(tiny_dataset), "--output_dir", str(out), "--epochs", "1",
            "--batch_size", "4", "--embed_dim", "16",
            "--vision_model_name", tiny_backbones[0], "--text_model_name", tiny_backbones[1]]
    if use_mlflow:
        pytest.importorskip("mlflow")
        monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
        monkeypatch.chdir(tmp_path)  # keep artifact store inside tmp
        args += ["--experiment", "test"]
    else:
        args += ["--no_mlflow"]
    _run(train, monkeypatch, args)

    ckpt = out / "fusion_model.pt"
    metrics = json.loads((out / "metrics.json").read_text())
    assert ckpt.exists()
    assert {"train_loss", "t2i_recall_at_1", "i2t_recall_at_5"} <= set(metrics)

    if use_mlflow:
        import mlflow
        runs = mlflow.search_runs(experiment_names=["test"])
        assert len(runs) == 1
        assert runs.iloc[0]["params.epochs"] == "1"
        assert "metrics.t2i_recall_at_1" in runs.columns

    index_path = out / "gallery_index.pt"
    _run(build_index, monkeypatch, [
        "--checkpoint", str(ckpt),
        "--manifests", str(tiny_dataset / "train.jsonl"), str(tiny_dataset / "val.jsonl"),
        "--image_root", str(tiny_dataset / "images"), "--out", str(index_path),
        "--uri_prefix", "gs://bucket/images", "--device", "cpu",
    ])
    service = load_service(str(ckpt), str(index_path), device="cpu")
    assert len(service.index) == 18
    hits = service.search_text("a red circle", top_k=5)["results"]
    assert len(hits) == 5 and hits[0]["uri"].startswith("gs://bucket/images/img_")
