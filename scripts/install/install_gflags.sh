#!/usr/bin/env bash
set -euo pipefail

GFLAGS_VER="2.2.2"
GFLAGS_STAMP="${DEPS_DIR}/.stamp-gflags"

if [[ ! -f "${GFLAGS_STAMP}" ]]; then
  echo "[gflags] Installing gflags v${GFLAGS_VER} to ${DEPS_DIR} ..."
  pushd "${DEPS_TMP_DIR}" >/dev/null

  TARBALL="gflags-${GFLAGS_VER}.tar.gz"
  URL="https://github.com/gflags/gflags/archive/refs/tags/v${GFLAGS_VER}.tar.gz"
  download "${URL}" "${TARBALL}"

  rm -rf "gflags-${GFLAGS_VER}"
  tar zxf "${TARBALL}"

  pushd "gflags-${GFLAGS_VER}" >/dev/null

  # --- install gflags following its own step
  rm -rf build
  mkdir -p build && cd build

  cmake -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="${DEPS_DIR}" \
        -DCMAKE_BUILD_TYPE=Release \
        ..
  cmake --build . --target install -j

  popd >/dev/null # gflags-$GFLAGS_VER
  popd >/dev/null # tmp

  # remove tmp files
  rm "${DEPS_TMP_DIR}/${TARBALL}"
  rm -rf "${DEPS_TMP_DIR}/gflags-${GFLAGS_VER}"

  touch "${GFLAGS_STAMP}"
  echo "[gflags] Installed."
else
  echo "[gflags] Found stamp ${GFLAGS_STAMP}, skipping install."
fi
