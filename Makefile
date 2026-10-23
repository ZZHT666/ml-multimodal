# Common commands. MLflow logs to a local SQLite store (needed for the Model Registry).
export MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
PY ?= python

.PHONY: install data train index repro serve test lint bench bench-http docker-build docker-run mlflow-ui

install:            ## dev environment
	pip install -r requirements.txt -r requirements-dev.txt

data:               ## regenerate the synthetic dataset
	$(PY) data/generate_synthetic.py

train:              ## train + log to MLflow (params from params.yaml are used by `make repro`)
	$(PY) -m src.train

index:              ## pre-compute gallery embeddings
	$(PY) -m src.build_index

repro:              ## DVC pipeline: train -> build_index (only stale stages re-run)
	dvc repro && dvc metrics show

serve:              ## local API on :8080
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --no-access-log

test:
	pytest -q

lint:
	ruff check src tests scripts

bench:              ## in-process latency (model only)
	$(PY) scripts/benchmark_latency.py inprocess

bench-http:         ## latency through HTTP (run `make serve` or `make docker-run` first)
	$(PY) scripts/benchmark_latency.py http --url http://localhost:8080

docker-build:       ## on Apple silicon: docker buildx build --platform linux/amd64 ...
	docker build -t clip-retrieval .

docker-run:
	docker run --rm -p 8080:8080 -e TORCH_NUM_THREADS=2 --cpus=2 \
	  -v $(PWD)/outputs:/app/outputs:ro clip-retrieval

mlflow-ui:
	mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port 5000
