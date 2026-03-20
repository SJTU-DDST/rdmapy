#!/usr/bin/env bash
set -euo pipefail

GCC_VER="11.4.0"
PREFIX="${HOME}/local/gcc-${GCC_VER}"
SRC_ROOT="${HOME}/src"
GCC_SRC="${SRC_ROOT}/gcc-${GCC_VER}"
BUILD_DIR="${HOME}/build/gcc-${GCC_VER}"

mkdir -p "${SRC_ROOT}" "${BUILD_DIR}" "$(dirname "${PREFIX}")"
cd "${SRC_ROOT}"

download() {
  local url="$1"
  local file="$2"
  if [ ! -f "$file" ]; then
    wget --no-check-certificate -O "$file" "$url"
  fi
}

echo "[1/7] 下载 GCC 源码"
download "https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VER}/gcc-${GCC_VER}.tar.xz" "gcc-${GCC_VER}.tar.xz"

if [ ! -d "${GCC_SRC}" ]; then
  tar -xf "gcc-${GCC_VER}.tar.xz"
fi

cd "${GCC_SRC}"

echo "[2/7] 下载依赖源码"
download "https://gcc.gnu.org/pub/gcc/infrastructure/gmp-6.1.0.tar.bz2" "gmp-6.1.0.tar.bz2"
download "https://gcc.gnu.org/pub/gcc/infrastructure/mpfr-3.1.4.tar.bz2" "mpfr-3.1.4.tar.bz2"
download "https://gcc.gnu.org/pub/gcc/infrastructure/mpc-1.0.3.tar.gz" "mpc-1.0.3.tar.gz"
download "https://gcc.gnu.org/pub/gcc/infrastructure/isl-0.18.tar.bz2" "isl-0.18.tar.bz2"

echo "[3/7] 解压依赖"
[ -d gmp-6.1.0 ] || tar -xf gmp-6.1.0.tar.bz2
[ -d mpfr-3.1.4 ] || tar -xf mpfr-3.1.4.tar.bz2
[ -d mpc-1.0.3 ] || tar -xf mpc-1.0.3.tar.gz
[ -d isl-0.18 ] || tar -xf isl-0.18.tar.bz2

ln -sfn gmp-6.1.0 gmp
ln -sfn mpfr-3.1.4 mpfr
ln -sfn mpc-1.0.3 mpc
ln -sfn isl-0.18 isl

echo "[4/7] 配置"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

"${GCC_SRC}/configure" \
  --prefix="${PREFIX}" \
  --disable-multilib \
  --enable-languages=c,c++ \
  --disable-bootstrap

echo "[5/7] 编译"
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
if [ "${JOBS}" -gt 8 ]; then
  JOBS=8
fi
make -j"${JOBS}"

echo "[6/7] 安装"
make install

echo "[7/7] 完成"
echo
echo "执行下面命令启用新 GCC:"
echo "export PATH=${PREFIX}/bin:\$PATH"
echo "export LD_LIBRARY_PATH=${PREFIX}/lib64:\$LD_LIBRARY_PATH"
echo "hash -r"
echo
echo "检查版本:"
echo "gcc --version"
echo "g++ --version"