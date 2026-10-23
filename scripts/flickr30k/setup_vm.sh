#!/usr/bin/env bash
# One-time setup ON the VM (Debian 12): system packages + Python env for the whole repo
# (ingest, feature extraction, training, MLflow, DVC, serving, tests).
#   bash scripts/flickr30k/setup_vm.sh
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq git unzip tmux make python3-venv python3-pip
[[ -d "${HOME}/venv" ]] || python3 -m venv "${HOME}/venv"
"${HOME}/venv/bin/pip" install -q --upgrade pip
# CPU-only torch first (~200 MB instead of ~2.5 GB of CUDA wheels; this VM has no GPU) ...
"${HOME}/venv/bin/pip" install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
# ... then everything else, pinned (serving + data + mlflow + dvc + pytest/ruff).
"${HOME}/venv/bin/pip" install -q -r requirements-dev.txt
echo "done. activate with:  source ~/venv/bin/activate"
