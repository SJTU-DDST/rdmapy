#!/usr/bin/env bash
set -euo pipefail

GCC_VER="11.4.0"
PREFIX="${HOME}/local/gcc-${GCC_VER}"
SRC_DIR="${HOME}/src"
BUILD_DIR="${HOME}/build/gcc-${GCC_VER}"
TARBALL="gcc-${GCC_VER}.tar.xz"
URL="https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VER}/${TARBALL}"

echo "[1/8] check base tools"
for x in gcc g++ make tar xz; do
  command -v "${x}" >/dev/null 2>&1 || {
    echo "missing command: ${x}"
    exit 1
  }
done

mkdir -p "${SRC_DIR}" "${BUILD_DIR}" "$(dirname "${PREFIX}")"
cd "${SRC_DIR}"

echo "[2/8] download gcc ${GCC_VER}"
if [ ! -f "${TARBALL}" ]; then
  curl -L -o "${TARBALL}" "${URL}" || wget -O "${TARBALL}" "${URL}"
fi

echo "[3/8] extract source"
if [ ! -d "gcc-${GCC_VER}" ]; then
  tar -xf "${TARBALL}"
fi

cd "gcc-${GCC_VER}"

echo "[4/8] download prerequisites"
./contrib/download_prerequisites

echo "[5/8] configure"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

"${SRC_DIR}/gcc-${GCC_VER}/configure" \
  --prefix="${PREFIX}" \
  --disable-multilib \
  --enable-languages=c,c++ \
  --disable-bootstrap

echo "[6/8] build"
CPU_N="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
if [ "${CPU_N}" -gt 8 ]; then
  CPU_N=8
fi
make -j"${CPU_N}"

echo "[7/8] install"
make install

echo "[8/8] done"
cat <<EOF

GCC ${GCC_VER} installed to:
  ${PREFIX}

Temporarily enable it in current shell:
  export PATH=${PREFIX}/bin:\$PATH
  export LD_LIBRARY_PATH=${PREFIX}/lib64:\$LD_LIBRARY_PATH
  hash -r

Check version:
  gcc --version
  g++ --version

Use in cmake:
  rm -rf build
  cmake -B build \\
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \\
    -DCMAKE_INSTALL_PREFIX=\$INSTALL_DIR \\
    -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \\
    -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \\
    -DCMAKE_CXX_STANDARD=20 \\
    -DCMAKE_CXX_STANDARD_REQUIRED=ON \\
    -DCMAKE_CXX_EXTENSIONS=OFF \\
    -DRDMAPP_BUILD_EXAMPLES_PYBIND=ON \\
    -Dpybind11_DIR=\$(python3 -m pybind11 --cmakedir)

  cmake --build build -j
EOF