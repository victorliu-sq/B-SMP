#!/usr/bin/env bash
set -euo pipefail

# Compared to $PWD, this command improves the portability of scripts
# Not where you ran this script,PROJECT_DIR and SCRIPTS_DIRS will be evaluated based on the absoluate paths of scripts.
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCRIPTS_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "[RUN_BSMP] PROJECT_DIR is ${PROJECT_DIR}"
echo "[RUN_BSMP] SCRIPTS_DIR is ${SCRIPTS_DIR}"

WORKLOAD_SIZE="${1:-10}"
WORKLOAD_TYPE="${2:-CONGESTED}"

case "${WORKLOAD_TYPE}" in
  CONGESTED|SOLO|RANDOM|PERFECT)
    ;;
  *)
    echo "Usage: $0 [size] [CONGESTED|SOLO|RANDOM|PERFECT]" >&2
    exit 1
    ;;
esac

# Run executable
mkdir -p "${PROJECT_DIR}/tmp/logs"
"${PROJECT_DIR}/build/bin/bsmp_exe" "${WORKLOAD_SIZE}" "${WORKLOAD_TYPE}"
