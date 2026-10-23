import base64
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serving.app import create_app
from src.serving.index import IndexModelMismatch, RetrievalIndex
from src.serving.service import ClipEncoder, RetrievalService, load_service


@pytest.fixture(scope="module")
def service(tiny_ckpt, tiny_index):
    return load_service(str(tiny_ckpt), str(tiny_index), device="cpu")


@pytest.fixture(scope="module")
def client(service):
    with TestClient(create_app(service)) as c:   # `with` runs the lifespan (warmup)
        yield c


def _png_b64(color=(220, 60, 60)):
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), color).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_health(client, service):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["index_size"] == len(service.index)
    assert body["model_version"] == service.model_version


def test_search_contract(client):
    r = client.post("/search", json={"query": "a red circle on a white background", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    assert {"rank", "score", "image", "text"} <= set(body["results"][0])
    scores = [h["score"] for h in body["results"]]
    assert scores == sorted(scores, reverse=True)
    assert {"encode_ms", "search_ms", "inference_ms"} <= set(body["timing_ms"])
    assert float(r.headers["x-server-time-ms"]) > 0
    assert r.headers["x-request-id"]


def test_request_id_is_propagated(client):
    r = client.post("/search", json={"query": "a blue square"}, headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"


@pytest.mark.parametrize("payload", [
    {"query": ""}, {"query": "x", "top_k": 0}, {"query": "x", "top_k": 10_000}, {"top_k": 3},
    {"query": "x" * 1000},
])
def test_search_validation(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_image_search(client):
    r = client.post("/search/image", json={"image_b64": _png_b64(), "top_k": 2})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2
    assert "decode_ms" in r.json()["timing_ms"]


def test_bad_image_is_400_not_500(client):
    r = client.post("/search/image", json={"image_b64": base64.b64encode(b"not an image").decode()})
    assert r.status_code == 400


def test_vertex_predict_format(client):
    r = client.post("/predict", json={
        "instances": [{"text": "a green triangle"}, {"image_b64": _png_b64((60, 100, 220))}],
        "parameters": {"top_k": 2},
    })
    assert r.status_code == 200
    preds = r.json()["predictions"]
    assert len(preds) == 2 and all(len(p["results"]) == 2 for p in preds)
    assert all("timing_ms" in p for p in preds)


@pytest.mark.parametrize("payload", [
    {"instances": []},
    {"instances": [{}]},
    {"instances": [{"text": "a", "image_b64": "b"}]},
])
def test_predict_validation(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("top_k", [0, "abc", 9999])
def test_predict_bad_top_k(client, top_k):
    r = client.post("/predict", json={"instances": [{"text": "a"}], "parameters": {"top_k": top_k}})
    assert r.status_code == 400


def test_metrics_endpoint(client):
    client.post("/search", json={"query": "a circle"})
    text = client.get("/metrics").text
    assert "retrieval_requests_total" in text
    assert 'route="/search"' in text


def test_service_refuses_index_from_another_model(tiny_ckpt, tiny_index):
    encoder = ClipEncoder.from_checkpoint(tiny_ckpt, device="cpu")
    index = RetrievalIndex.load(tiny_index)
    index.meta["model_fingerprint"] = "0" * 32
    with pytest.raises(IndexModelMismatch):
        RetrievalService(encoder, index)


def test_service_refuses_dim_mismatch(tiny_ckpt):
    import torch
    encoder = ClipEncoder.from_checkpoint(tiny_ckpt, device="cpu")
    index = RetrievalIndex(torch.randn(3, encoder.embed_dim + 1), [{"image": str(i)} for i in range(3)])
    with pytest.raises(ValueError):
        RetrievalService(encoder, index)
