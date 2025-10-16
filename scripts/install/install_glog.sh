#!/usr/bin/env bash
set -euo pipefail

GLOG_VER="0.6.0"
GLOG_STAMP="${DEPS_DIR}/.stamp-glog"

if [[ ! -f "${GLOG_STAMP}" ]]; then
  echo "[glog] Installing glog v${GLOG_VER} to ${DEPS_DIR} ..."
  pushd "${DEPS_TMP_DIR}" >/dev/null

  TARBALL="glog-$GLOG_VER.tar.gz"
  URL="https://github.com/google/glog/archive/refs/tags/v$GLOG_VER.tar.gz"
  download "$URL" "$TARBALL"

  rm -rf "glog-$GLOG_VER"
  tar zxf "$TARBALL"
  pushd "glog-$GLOG_VER" >/dev/null

    rm -rf build
    mkdir -p build && cd build

    cmake \
      -DBUILD_SHARED_LIBS=ON \
      -DWITH_GTEST=OFF \
      -DCMAKE_INSTALL_PREFIX="${DEPS_DIR}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="${DEPS_DIR}" \
      ..

    cmake --build . --target install -j

  popd >/dev/null # glog-$GLOG_VER
  popd >/dev/null # tmp

  # remove tmp files
  rm "${DEPS_TMP_DIR}/${TARBALL}"
  rm -rf "${DEPS_TMP_DIR}/glog-${GLOG_VER}"

  touch "${GLOG_STAMP}"
  echo "[glog] Installed."
else
  echo "[glog] Found stamp ${GLOG_STAMP}, skipping install."
fi
