#!/usr/bin/env bash
set -euo pipefail

# Compared to $PWD, this command improves the portability of scripts
# Not where you ran this script,PROJECT_DIR and SCRIPTS_DIRS will be evaluated based on the absoluate paths of scripts.
export PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export SCRIPTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

#export LD_LIBRARY_PATH="${SCRIPTS_DIR}/third_party/lib":$LD_LIBRARY_PATH
#echo "[INFO] LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
#export LD_LIBRARY_PATH=~/CLionProjects/X-Blossom-GPU/third_party/lib:$LD_LIBRARY_PATH

# downloader
download() {
  local url="$1" out="$2"
  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$out"
  elif command -v wget >/dev/null 2>&1; then
    wget "$url" -O "$out"
  else
    echo "ERROR: need curl or wget" >&2
    exit 1
  fi
}
export -f download
