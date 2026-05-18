#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

echo "[BUILD] PROJECT_DIR is ${PROJECT_DIR}"

cmake -S "${PROJECT_DIR}" -B "${PROJECT_DIR}/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "${PROJECT_DIR}/build" -j
