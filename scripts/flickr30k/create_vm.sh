#!/usr/bin/env bash
# Create a CPU worker VM in the same region as the data bucket.
# Run from Cloud Shell (or any machine with gcloud).
#
#   export PROJECT_ID=<project>
#   bash scripts/flickr30k/create_vm.sh            # create
#   bash scripts/flickr30k/create_vm.sh stop       # stop when idle (disk still billed, ~$10/month for 100GB)
#   bash scripts/flickr30k/create_vm.sh delete     # delete VM + disk
set -euo pipefail

PROJECT_ID="${PROJECT_ID:?set PROJECT_ID}"
BUCKET="${BUCKET:-zht-mle-demo-2026-a1b7}"
VM_NAME="${VM_NAME:-ml-worker}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-8}"   # 8 vCPU / 32 GB: enough for ingest + CPU feature extraction

# Put the VM where the data lives: same-region GCS <-> VM traffic is free and fast.
LOC=$(gcloud storage buckets describe "gs://${BUCKET}" --project="${PROJECT_ID}" --format="value(location)")
LOC=$(echo "${LOC}" | tr '[:upper:]' '[:lower:]')
case "${LOC}" in
  us|nam4) REGION=us-central1 ;;
  eu|eur4) REGION=europe-west4 ;;
  asia)    REGION=asia-east1 ;;
  *)       REGION="${LOC}" ;;
esac
ZONE="${ZONE:-${REGION}-a}"
echo "bucket location=${LOC} -> zone=${ZONE}"

case "${1:-create}" in
  create)
    gcloud compute instances create "${VM_NAME}" --project="${PROJECT_ID}" --zone="${ZONE}" \
      --machine-type="${MACHINE_TYPE}" \
      --image-family=debian-12 --image-project=debian-cloud \
      --boot-disk-size=100GB --boot-disk-type=pd-balanced \
      --scopes=cloud-platform
    echo
    echo "ssh in with:  gcloud compute ssh ${VM_NAME} --zone=${ZONE} --project=${PROJECT_ID}"
    ;;
  stop)   gcloud compute instances stop "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" ;;
  start)  gcloud compute instances start "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" ;;
  delete) gcloud compute instances delete "${VM_NAME}" --zone="${ZONE}" --project="${PROJECT_ID}" ;;
  *) echo "usage: $0 [create|stop|start|delete]"; exit 1 ;;
esac
