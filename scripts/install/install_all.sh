#!/usr/bin/env bash
set -euo pipefail

# --- Installation Config
export DEPS_DIR="${PROJECT_DIR}/third_party"
export DEPS_TMP_DIR="${DEPS_DIR}/tmp"
INSTALL_DIR="${SCRIPTS_DIR}/install"

mkdir -p \
  "${DEPS_DIR}" \
  "${DEPS_TMP_DIR}"

# --- Install gflags (once) ----------------------------------------------------
bash "${INSTALL_DIR}/install_gflags.sh"

# --- Install glog (once) ----------------------------------------------------
bash "${INSTALL_DIR}/install_glog.sh"

# --- Install gtest (once) ----------------------------------------------------
bash "${INSTALL_DIR}/install_gtest.sh"

# --- Install google benchmark (once) ----------------------------------------------------
bash "${INSTALL_DIR}/install_gbenchmark.sh"

#export LD_LIBRARY_PATH="${DEPS_DIR}/lib:${PROJECT_DIR}/bin:${LD_LIBRARY_PATH:-}"
#echo "[INFO] LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# remove the tmp directory
rm -rf ${DEPS_TMP_DIR}