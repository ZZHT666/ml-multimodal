# CPU serving image for the CLIP retrieval API.
#   docker build -t clip-retrieval .
#   docker run -p 8080:8080 -v $PWD/outputs:/app/outputs clip-retrieval
#
# Model weights are NOT baked in: the image is the code + runtime, the checkpoint
# and gallery index are versioned separately (DVC / GCS / Vertex --artifact-uri),
# so a retrained model does not require a new image.
FROM python:3.11-slim

ARG TORCH_VERSION=2.5.1
# CPU-only wheels: ~200 MB instead of the ~2.5 GB CUDA build. For a GPU image,
# start FROM an nvidia/cuda runtime image and drop the --index-url.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG VISION_MODEL=facebook/dinov2-small
ARG TEXT_MODEL=sentence-transformers/all-MiniLM-L6-v2

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# Dependencies first, in their own layers: source edits then don't re-run pip.
RUN pip install torch==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL}
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Bake only the backbones' config + tokenizer + image-processor files (a few MB);
# all weights come from our checkpoint (FusionCLIPModel(pretrained=False)).
RUN python -c "from transformers import AutoConfig, AutoTokenizer, AutoImageProcessor; \
AutoConfig.from_pretrained('${VISION_MODEL}'); AutoImageProcessor.from_pretrained('${VISION_MODEL}'); \
AutoConfig.from_pretrained('${TEXT_MODEL}'); AutoTokenizer.from_pretrained('${TEXT_MODEL}')"
# From here on, never touch the network for HF files (no surprise downloads at startup).
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY src/ src/

RUN useradd --create-home --uid 1000 app && chown -R app /app /opt/hf
USER app

EXPOSE 8080
# Vertex AI sets AIP_HTTP_PORT, Cloud Run sets PORT; default 8080 locally.
# One worker per container: scale out with replicas, not processes fighting over CPU threads.
# --no-access-log: the app already writes one structured JSON line per request.
CMD ["sh", "-c", "exec uvicorn src.serving.app:app --host 0.0.0.0 --port ${AIP_HTTP_PORT:-${PORT:-8080}} --workers 1 --no-access-log"]
