"""In-memory retrieval index over pre-computed, L2-normalized gallery embeddings.

Brute-force cosine similarity (one matrix-vector product) is exact and, for a
gallery of a few thousand to ~1M vectors x 256 dims, fast enough on CPU. Past
that point, swap this class for an ANN index (FAISS IVF/HNSW, or Vertex AI
Vector Search) behind the same `search()` interface.
"""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F


class IndexModelMismatch(RuntimeError):
    """The gallery index was built with a different checkpoint than the one being served."""


class RetrievalIndex:
    def __init__(self, embeddings: torch.Tensor, records: Sequence[Dict],
                 meta: Optional[Dict] = None):
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be (N, D), got {tuple(embeddings.shape)}")
        if len(records) != embeddings.size(0):
            raise ValueError("need exactly one record per embedding row")
        # Re-normalize defensively: cosine similarity == dot product only on unit vectors.
        self.embeddings = F.normalize(embeddings.float(), dim=-1)
        self.records = list(records)
        self.meta = dict(meta or {})

    def __len__(self) -> int:
        return len(self.records)

    @property
    def dim(self) -> int:
        return self.embeddings.size(1)

    def to(self, device) -> "RetrievalIndex":
        self.embeddings = self.embeddings.to(device)
        return self

    @torch.no_grad()
    def search(self, queries: torch.Tensor, top_k: int = 5) -> List[List[Dict]]:
        """queries: (D,) or (Q, D) L2-normalized. Returns one ranked hit list per query."""
        if queries.ndim == 1:
            queries = queries.unsqueeze(0)
        if queries.size(1) != self.dim:
            raise ValueError(f"query dim {queries.size(1)} != index dim {self.dim}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        k = min(top_k, len(self))
        sims = queries.to(self.embeddings.device, self.embeddings.dtype) @ self.embeddings.t()
        scores, idx = sims.topk(k, dim=-1)          # sorted descending
        results = []
        for row_scores, row_idx in zip(scores.tolist(), idx.tolist(), strict=False):
            results.append([
                {"rank": rank, "score": round(score, 6), **self.records[i]}
                for rank, (score, i) in enumerate(zip(row_scores, row_idx, strict=False), start=1)
            ])
        return results

    def check_model(self, model_fingerprint: Optional[str]) -> None:
        """Fail fast on index/model version skew: embeddings from checkpoint A are
        meaningless when queries are encoded by checkpoint B, but nothing would crash --
        results would just silently be garbage."""
        built_with = self.meta.get("model_fingerprint")
        if built_with and model_fingerprint and built_with != model_fingerprint:
            raise IndexModelMismatch(
                f"index built with model {built_with[:12]}, serving model {model_fingerprint[:12]}; "
                "rebuild the index (python -m src.build_index) after retraining")

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"embeddings": self.embeddings.cpu(), "records": self.records,
                    "meta": self.meta}, path)

    @classmethod
    def load(cls, path) -> "RetrievalIndex":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return cls(blob["embeddings"], blob["records"], blob.get("meta"))
