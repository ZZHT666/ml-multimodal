#!/usr/bin/env bash
# Build -> push -> register -> deploy the retrieval service on Vertex AI.
#
#   export PROJECT_ID=<your-project>          # required
#   ./scripts/deploy_vertex.sh all            # build + upload + deploy + test
#   ./scripts/deploy_vertex.sh test           # one prediction against the endpoint
#   ./scripts/deploy_vertex.sh undeploy       # STOP BILLING: a deployed model bills per node-hour, even idle
#
# Steps can also be run one by one: build | artifacts | upload | deploy | test | undeploy | cloudrun
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-central1}"
BUCKET="${BUCKET:-zht-mle-demo-2026-a1b7}"            # same bucket as the DVC remote
AR_REPO="${AR_REPO:-ml-serving}"
IMAGE_NAME="${IMAGE_NAME:-clip-retrieval}"
VERSION="${VERSION:-$(git rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M)}"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${VERSION}"
MODEL_DIR="gs://${BUCKET}/models/${IMAGE_NAME}/${VERSION}"
MODEL_NAME="${MODEL_NAME:-clip-retrieval}"
ENDPOINT_NAME="${ENDPOINT_NAME:-clip-retrieval-endpoint}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-2}"

gc() { gcloud --project="${PROJECT_ID}" "$@"; }

endpoint_id() {
  local name
  name=$(gc ai endpoints list --region="${REGION}" --filter="display_name=${ENDPOINT_NAME}" \
           --format="value(name)" | head -n1)
  echo "${name##*/}"
}

build() {
  gc services enable aiplatform.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com
  gc artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1 ||
    gc artifacts repositories create "${AR_REPO}" --repository-format=docker --location="${REGION}"
  # Cloud Build builds on linux/amd64 in GCP. (A plain `docker build` on an Apple-silicon
  # Mac produces an arm64 image that Vertex AI cannot run.)
  gc builds submit --tag "${IMAGE_URI}" .
  echo "image: ${IMAGE_URI}"
}

artifacts() {
  # Model artifacts live next to (not inside) the image; Vertex copies this folder and
  # passes it to the container as AIP_STORAGE_URI.
  gcloud storage cp outputs/fusion_model.pt outputs/gallery_index.pt "${MODEL_DIR}/"
  echo "artifacts: ${MODEL_DIR}"
}

upload() {
  # If the model already exists, upload as a new *version* of it (Vertex Model Registry),
  # so every deployed version stays traceable and rollback is "deploy version N-1".
  local parent parent_flag=()
  parent=$(gc ai models list --region="${REGION}" --filter="display_name=${MODEL_NAME}" \
             --sort-by="createTime" --limit=1 --format="value(name)")
  [[ -n "${parent}" ]] && parent_flag=(--parent-model="${parent}")
  gc ai models upload --region="${REGION}" ${parent_flag[@]+"${parent_flag[@]}"} \
    --display-name="${MODEL_NAME}" \
    --container-image-uri="${IMAGE_URI}" \
    --artifact-uri="${MODEL_DIR}" \
    --container-ports=8080 \
    --container-health-route=/health \
    --container-predict-route=/predict \
    --container-env-vars="TORCH_NUM_THREADS=2,LOG_LEVEL=INFO" \
    --version-aliases="v-${VERSION}"
}

deploy() {
  local ep model
  ep=$(endpoint_id)
  if [[ -z "${ep}" ]]; then
    gc ai endpoints create --region="${REGION}" --display-name="${ENDPOINT_NAME}"
    ep=$(endpoint_id)
  fi
  model=$(gc ai models list --region="${REGION}" --filter="display_name=${MODEL_NAME}" \
            --sort-by="~createTime" --limit=1 --format="value(name)")
  model="${model##*/}"
  # traffic-split 0=100: the model being deployed ("0") gets all traffic.
  # Canary instead: --traffic-split=0=10,<OLD_DEPLOYED_MODEL_ID>=90, watch metrics, then shift.
  gc ai endpoints deploy-model "${ep}" --region="${REGION}" \
    --model="${model}" --display-name="${MODEL_NAME}-${VERSION}" \
    --machine-type="${MACHINE_TYPE}" \
    --min-replica-count=1 --max-replica-count=2 \
    --autoscaling-metric-specs=cpu-usage=70 \
    --traffic-split=0=100
  echo "endpoint id: ${ep}"
}

test_endpoint() {
  local ep
  ep=$(endpoint_id)
  curl -sS -X POST \
    -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Content-Type: application/json" \
    "https://${REGION}-aiplatform.googleapis.com/v1/projects/${PROJECT_ID}/locations/${REGION}/endpoints/${ep}:predict" \
    -d '{"instances": [{"text": "a red circle on a white background"}], "parameters": {"top_k": 3}}'
  echo
  echo "benchmark: python scripts/benchmark_latency.py vertex --project ${PROJECT_ID} --region ${REGION} --endpoint_id ${ep}"
}

undeploy() {
  local ep id
  ep=$(endpoint_id)
  for id in $(gc ai endpoints describe "${ep}" --region="${REGION}" --format="value(deployedModels[].id)" | tr ';' ' '); do
    gc ai endpoints undeploy-model "${ep}" --region="${REGION}" --deployed-model-id="${id}"
  done
  echo "undeployed all models from endpoint ${ep} (endpoint itself is free to keep)"
}

cloudrun() {
  # Cheaper alternative: scales to zero, billed per request. Needs a service account
  # with storage.objectViewer on the bucket to read the artifacts.
  gc run deploy "${IMAGE_NAME}" --image="${IMAGE_URI}" --region="${REGION}" \
    --cpu=2 --memory=4Gi --concurrency=8 --min-instances=0 --max-instances=2 \
    --set-env-vars="MODEL_PATH=${MODEL_DIR}/fusion_model.pt,INDEX_PATH=${MODEL_DIR}/gallery_index.pt,TORCH_NUM_THREADS=2" \
    --no-allow-unauthenticated
}

case "${1:-}" in
  build) build ;;
  artifacts) artifacts ;;
  upload) upload ;;
  deploy) deploy ;;
  test) test_endpoint ;;
  undeploy) undeploy ;;
  cloudrun) cloudrun ;;
  all) build; artifacts; upload; deploy; test_endpoint ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
