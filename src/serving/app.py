"""FastAPI app for text->image and image->image retrieval.

Routes
  GET  /health        liveness/readiness (Vertex AI health route)
  POST /search        {"query": "...", "top_k": 5}
  POST /search/image  {"image_b64": "...", "top_k": 5}
  POST /predict       Vertex AI format: {"instances": [{"text": ...} | {"image_b64": ...}],
                                         "parameters": {"top_k": 5}} -> {"predictions": [...]}
  GET  /metrics       Prometheus metrics (request count + latency histogram)

Run locally:  uvicorn src.serving.app:app --port 8080
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, model_validator

from .artifacts import CKPT_NAME, INDEX_NAME, default_location
from .service import RetrievalService, load_service

MAX_TOP_K = int(os.environ.get("MAX_TOP_K", "50"))
MAX_INSTANCES = int(os.environ.get("MAX_INSTANCES", "16"))

log = logging.getLogger("retrieval")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")

REQUESTS = Counter("retrieval_requests_total", "Requests", ["route", "status"])
LATENCY = Histogram("retrieval_request_latency_seconds", "End-to-end request latency in the server",
                    ["route"], buckets=(.005, .01, .02, .03, .05, .075, .1, .2, .5, 1, 2, 5))


# ---------- request / response schemas ----------
class TextQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)


class ImageQuery(BaseModel):
    image_b64: str = Field(..., min_length=1)
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)


class Instance(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=512)
    image_b64: Optional[str] = None

    @model_validator(mode="after")
    def exactly_one(self):
        if (self.text is None) == (self.image_b64 is None):
            raise ValueError("each instance needs exactly one of 'text' or 'image_b64'")
        return self


class PredictRequest(BaseModel):
    instances: List[Instance] = Field(..., min_length=1, max_length=MAX_INSTANCES)
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ---------- app ----------
def _load_from_env() -> RetrievalService:
    torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", os.cpu_count() or 1)))
    return load_service(
        model_path=default_location("MODEL_PATH", CKPT_NAME),
        index_path=default_location("INDEX_PATH", INDEX_NAME),
        device=os.environ.get("DEVICE"),
    )


def create_app(service: Optional[RetrievalService] = None, warmup: bool = True) -> FastAPI:
    """`service` is injected by the unit tests; in production it is loaded from env vars."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = service or _load_from_env()
        if warmup:
            # First inference pays one-off costs (lazy init, allocator, kernel selection);
            # do it before /health turns green so no user request eats that latency.
            for _ in range(3):
                svc.search_text("warmup", top_k=1)
        app.state.service = svc
        log.info(json.dumps({"event": "ready", "model_version": svc.model_version,
                             "index_size": len(svc.index)}))
        yield

    app = FastAPI(title="CLIP retrieval", lifespan=lifespan)

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
        start = time.perf_counter()

        def record(status: int) -> float:
            elapsed = time.perf_counter() - start
            route = request.url.path
            if route != "/metrics":
                REQUESTS.labels(route, str(status)).inc()
                LATENCY.labels(route).observe(elapsed)
                # One JSON line per request: Cloud Logging parses it into structured fields.
                log.info(json.dumps({"request_id": request_id, "route": route, "status": status,
                                     "latency_ms": round(elapsed * 1000, 3)}))
            return elapsed

        try:
            response = await call_next(request)
        except Exception:
            record(500)
            raise
        elapsed = record(response.status_code)
        response.headers["x-request-id"] = request_id
        response.headers["x-server-time-ms"] = f"{elapsed * 1000:.3f}"
        return response

    def svc(request: Request) -> RetrievalService:
        service_ = getattr(request.app.state, "service", None)
        if service_ is None:
            raise HTTPException(503, "model not loaded")
        return service_

    @app.get("/health")
    def health(request: Request):
        s = svc(request)
        return {"status": "ok", "model_version": s.model_version, "index_size": len(s.index)}

    @app.post("/search")
    def search(body: TextQuery, request: Request):
        s = svc(request)
        return {**s.search_text(body.query, body.top_k), "model_version": s.model_version}

    @app.post("/search/image")
    def search_image(body: ImageQuery, request: Request):
        s = svc(request)
        try:
            out = s.search_image(body.image_b64, body.top_k)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        return {**out, "model_version": s.model_version}

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request):
        s = svc(request)
        try:
            top_k = int(body.parameters.get("top_k", 5))
        except (TypeError, ValueError):
            raise HTTPException(400, "parameters.top_k must be an integer") from None
        if not 1 <= top_k <= MAX_TOP_K:
            raise HTTPException(400, f"parameters.top_k must be in [1, {MAX_TOP_K}]")
        predictions = []
        for inst in body.instances:
            try:
                out = (s.search_text(inst.text, top_k) if inst.text is not None
                       else s.search_image(inst.image_b64, top_k))
            except ValueError as e:
                raise HTTPException(400, str(e)) from e
            # Vertex AI only forwards the "predictions" list, so per-request server
            # timing has to live inside each prediction to reach the client.
            predictions.append({**out, "model_version": s.model_version})
        return {"predictions": predictions}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


# Module-level app for `uvicorn src.serving.app:app`; the model loads in lifespan, not at import.
app = create_app()
