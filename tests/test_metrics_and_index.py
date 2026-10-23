import pytest
import torch
import torch.nn.functional as F

from src.metrics import retrieval_recall
from src.serving.index import IndexModelMismatch, RetrievalIndex


# ---------------- metrics ----------------
def test_perfect_embeddings_give_recall_one():
    emb = F.normalize(torch.randn(6, 8), dim=-1)
    r = retrieval_recall(emb, emb, [f"cap {i}" for i in range(6)])
    assert r["i2t_recall_at_1"] == r["t2i_recall_at_1"] == 1.0


def test_duplicate_captions_count_as_hits():
    # Items 0 and 1 share a caption; image 0 retrieving text 1 is still correct.
    img = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    txt = torch.tensor([[0.9, 0.1], [1.0, 0.0], [0.0, 1.0]])
    r = retrieval_recall(F.normalize(img, dim=-1), F.normalize(txt, dim=-1), ["red", "red", "blue"])
    assert r["i2t_recall_at_1"] == 1.0


def test_recall_at_k_is_monotonic():
    torch.manual_seed(0)
    img, txt = F.normalize(torch.randn(20, 8), dim=-1), F.normalize(torch.randn(20, 8), dim=-1)
    r = retrieval_recall(img, txt, [str(i) for i in range(20)], ks=(1, 5, 10))
    assert r["t2i_recall_at_1"] <= r["t2i_recall_at_5"] <= r["t2i_recall_at_10"]


def test_recall_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        retrieval_recall(torch.randn(3, 4), torch.randn(2, 4), ["a", "b", "c"])


# ---------------- index ----------------
@pytest.fixture
def index():
    emb = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [-1.0, 0.0]])
    records = [{"image": f"img_{i}.png", "text": f"t{i}"} for i in range(4)]
    return RetrievalIndex(emb, records, meta={"model_fingerprint": "abc"})


def test_search_returns_sorted_top_k(index):
    hits = index.search(torch.tensor([1.0, 0.0]), top_k=3)[0]
    assert [h["image"] for h in hits] == ["img_0.png", "img_2.png", "img_1.png"]
    assert [h["rank"] for h in hits] == [1, 2, 3]
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_index_normalizes_embeddings(index):
    torch.testing.assert_close(index.embeddings.norm(dim=-1), torch.ones(4))


def test_top_k_larger_than_index_is_clamped(index):
    assert len(index.search(torch.tensor([0.0, 1.0]), top_k=100)[0]) == 4


def test_batch_queries(index):
    out = index.search(torch.eye(2), top_k=1)
    assert [o[0]["image"] for o in out] == ["img_0.png", "img_1.png"]


@pytest.mark.parametrize("bad", [torch.randn(3), torch.randn(2, 5)])
def test_dim_mismatch_raises(index, bad):
    with pytest.raises(ValueError):
        index.search(bad)


def test_invalid_top_k_raises(index):
    with pytest.raises(ValueError):
        index.search(torch.tensor([1.0, 0.0]), top_k=0)


def test_save_load_roundtrip(index, tmp_path):
    index.save(tmp_path / "idx.pt")
    loaded = RetrievalIndex.load(tmp_path / "idx.pt")
    torch.testing.assert_close(loaded.embeddings, index.embeddings)
    assert loaded.records == index.records and loaded.meta == index.meta


def test_model_mismatch_is_detected(index):
    index.check_model("abc")          # same model: fine
    index.check_model(None)           # unknown: allowed
    with pytest.raises(IndexModelMismatch):
        index.check_model("different")
