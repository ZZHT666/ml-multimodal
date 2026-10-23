#!/usr/bin/env bash
# One-time setup ON the VM (Debian 12): system packages + Python env for the repo.
#   bash scripts/flickr30k/setup_vm.sh
set -euo pipefail
sudo apt-get update -qq
sudo apt-get install -y -qq git unzip python3-venv python3-pip
python3 -m venv "${HOME}/venv"
"${HOME}/venv/bin/pip" install -q --upgrade pip
"${HOME}/venv/bin/pip" install -q -r requirements-data.txt
echo "done. activate with:  source ~/venv/bin/activate"
