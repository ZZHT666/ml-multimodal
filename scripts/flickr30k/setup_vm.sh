#!/usr/bin/env bash
# One-time setup ON the VM (Debian 12): system packages + Python env for the repo.
#   bash scripts/flickr30k/setup_vm.sh
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq git unzip tmux python3-venv python3-pip
[[ -d "${HOME}/venv" ]] || python3 -m venv "${HOME}/venv"
"${HOME}/venv/bin/pip" install -q --upgrade pip
"${HOME}/venv/bin/pip" install -q -r requirements-data.txt
# Feature extraction: CPU-only torch (~200 MB instead of ~2.5 GB of CUDA wheels) + transformers.
"${HOME}/venv/bin/pip" install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
"${HOME}/venv/bin/pip" install -q transformers==4.46.3
echo "done. activate with:  source ~/venv/bin/activate"
