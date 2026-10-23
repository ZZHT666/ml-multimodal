# Common commands. MLflow logs to a local SQLite store (needed for the Model Registry).
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
PY ?= python

FEATURES_URI ?= gs://zht-mle-demo-2026-a1b7/features/flickr30k/v1/dinov2-small__minilm-l6
MODEL_PATH ?= outputs/flickr30k/fusion_model.pt
INDEX_PATH ?= outputs/flickr30k/gallery_index.pt
export MODEL_PATH INDEX_PATH

.PHONY: install features repro train-synthetic data serve test lint bench bench-http docker-build docker-run mlflow-ui

install:            ## dev environment
	pip install -r requirements.txt -r requirements-dev.txt

features:           ## import the cached Flickr30k features from GCS (tracked by DVC)
	mkdir -p features && dvc import-url $(FEATURES_URI) features/flickr30k-v1

repro:              ## DVC pipeline: train_flickr -> assemble -> build_index
	dvc repro && dvc metrics show

data:               ## regenerate the synthetic demo dataset
	$(PY) data/generate_synthetic.py

train-synthetic:    ## original synthetic-shapes demo (end-to-end backbone training)
	$(PY) -m src.train && $(PY) -m src.build_index --checkpoint outputs/fusion_model.pt

serve:              ## local API on :8080
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --no-access-log

test:
	pytest -q

lint:
	ruff check src tests scripts

bench:              ## in-process latency (model only)
	$(PY) scripts/benchmark_latency.py inprocess --checkpoint $(MODEL_PATH) --index $(INDEX_PATH)

bench-http:         ## latency through HTTP (run `make serve` or `make docker-run` first)
	$(PY) scripts/benchmark_latency.py http --url http://localhost:8080

docker-build:       ## on Apple silicon: docker buildx build --platform linux/amd64 ...
	docker build -t clip-retrieval .

docker-run:
	docker run --rm -p 8080:8080 -e TORCH_NUM_THREADS=2 --cpus=2 \
	  -e MODEL_PATH=/app/$(MODEL_PATH) -e INDEX_PATH=/app/$(INDEX_PATH) \
	  -v $(PWD)/outputs:/app/outputs:ro clip-retrieval

mlflow-ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000
