#!/usr/bin/env bash
set -euo pipefail

COMMON_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common/common.sh"
source "${COMMON_FILE}"

echo "[RUNME] PROJECT_DIR is ${PROJECT_DIR}"
echo "[RUNME] SCRIPTS_DIR is ${SCRIPTS_DIR}"

# 1. Install third-party dependencies
bash "${SCRIPTS_DIR}/install/install_all.sh"

# 2. Build executables
bash "${SCRIPTS_DIR}/build/build_all.sh"