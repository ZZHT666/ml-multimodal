#!/usr/bin/env bash
# Flickr30k -> GCS (raw, immutable) -> validated metadata parquet -> BigQuery.
# Runs ON the VM from the repo root, inside the venv:
#
#   source ~/venv/bin/activate
#   export BUCKET=zht-mle-demo-2026-a1b7      # optional, this is the default
#   bash scripts/flickr30k/ingest.sh
#
# Re-runnable: downloads resume, GCS upload is --no-clobber (raw data is never
# overwritten in place -- a changed dataset becomes v2), BigQuery loads are WRITE_TRUNCATE.
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-$(gcloud config get-value project 2>/dev/null)}"
BUCKET="${BUCKET:-zht-mle-demo-2026-a1b7}"
VERSION="${VERSION:-v1}"
WORK="${WORK:-${HOME}/flickr30k_work}"
HF_BASE="https://huggingface.co/datasets/nlphuji/flickr30k/resolve/main"

RAW="gs://${BUCKET}/raw/flickr30k/${VERSION}"
PROCESSED="gs://${BUCKET}/processed/flickr30k/${VERSION}"
BQ_LOCATION="${BQ_LOCATION:-$(gcloud storage buckets describe "gs://${BUCKET}" --format='value(location)')}"
mkdir -p "${WORK}"

step() { echo; echo "==== $* ($(date +%H:%M:%S))"; }

step "1/6 download from Hugging Face -> ${WORK}"
dl() {  # resumable; skips files that already finished downloading
  local name="$1"
  [[ -f "${WORK}/${name}" ]] && { echo "have ${name}"; return; }
  curl -fL --retry 5 -C - -o "${WORK}/${name}.part" "${HF_BASE}/${name}"
  mv "${WORK}/${name}.part" "${WORK}/${name}"
}
dl flickr_annotations_30k.csv
dl flickr30k-images.zip
# Provenance: record exactly which bytes this dataset version came from.
( cd "${WORK}" && sha256sum flickr_annotations_30k.csv flickr30k-images.zip > SHA256SUMS ) && cat "${WORK}/SHA256SUMS"

step "2/6 unzip"
[[ -d "${WORK}/extract/flickr30k-images" ]] || unzip -q -o "${WORK}/flickr30k-images.zip" -d "${WORK}/extract"
IMG_DIR="${WORK}/extract/flickr30k-images"
echo "$(find "${IMG_DIR}" -maxdepth 1 -type f | wc -l) files in ${IMG_DIR}"

step "3/6 build + validate metadata (md5, size, dims, 5 captions/image, splits)"
python -m src.data.flickr30k build \
  --csv "${WORK}/flickr_annotations_30k.csv" --image_dir "${IMG_DIR}" \
  --gcs_image_prefix "${RAW}/images" --version "${VERSION}" \
  --out_dir "${WORK}/build" --verify

step "4/6 upload raw images + annotations to ${RAW}"
gcloud storage rsync --recursive --no-clobber "${IMG_DIR}" "${RAW}/images"
gcloud storage cp --no-clobber "${WORK}/flickr_annotations_30k.csv" "${WORK}/SHA256SUMS" "${RAW}/annotations/"
EXPECTED=$(python -c "import pandas as pd; print(len(pd.read_parquet('${WORK}/build/images.parquet')))")
ACTUAL=$(gcloud storage ls "${RAW}/images/" | wc -l)
echo "images in metadata=${EXPECTED}  objects in GCS=${ACTUAL}"
[[ "${ACTUAL}" -ge "${EXPECTED}" ]] || { echo "ERROR: GCS is missing images"; exit 1; }

step "5/6 upload processed parquet to ${PROCESSED}"
gcloud storage cp "${WORK}/build/images.parquet" "${WORK}/build/captions.parquet" \
  "${WORK}/build/validation_report.json" "${PROCESSED}/"

step "6/6 load BigQuery (${PROJECT_ID}.flickr30k, location ${BQ_LOCATION})"
python -m src.data.flickr30k load-bq --project "${PROJECT_ID}" --location "${BQ_LOCATION}" \
  --parquet_prefix "${PROCESSED}"

echo
echo "done. local copy kept in ${WORK} (reused by feature extraction; rm -rf it to free disk)."
echo "next: export the DVC snapshot -- see scripts/flickr30k/README.md step 4"
