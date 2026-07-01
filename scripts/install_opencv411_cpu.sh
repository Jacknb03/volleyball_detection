#!/bin/bash
# 工控机无显卡：源码安装 OpenCV 4.11 到 /usr/local，供 station_detector_cpp 链接。
# 用法: bash scripts/install_opencv411_cpu.sh   （不要用 sudo 跑整个脚本）
set -eo pipefail

run_sudo() {
  if sudo -n true 2>/dev/null; then
    sudo "$@"
  elif [[ -n "${SUDO_PASSWORD:-}" ]]; then
    echo "$SUDO_PASSWORD" | sudo -S "$@"
  else
    sudo "$@"
  fi
}

OPENCV_VER="${OPENCV_VER:-4.11.0}"
# 始终用当前用户目录，避免 sudo 时落到 /root
OPENCV_SRC="${OPENCV_SRC:-$HOME/opencv411/opencv}"
OPENCV_BUILD="${OPENCV_BUILD:-$HOME/opencv411/build}"
OPENCV_TAR="${OPENCV_TAR:-$HOME/opencv411/opencv-${OPENCV_VER}.tar.gz}"
LOG="${LOG:-/tmp/opencv411_build.log}"

if [[ -f /usr/local/lib/libopencv_dnn.so ]]; then
  ver=$(pkg-config --modversion opencv4 2>/dev/null || true)
  if [[ "$ver" == 4.11* ]]; then
    echo "OpenCV $ver 已在 /usr/local，跳过编译。"
    exit 0
  fi
fi

echo "=== 安装编译依赖 (需 sudo) ==="
run_sudo apt-get update -qq
run_sudo apt-get install -y \
  build-essential cmake git wget unzip \
  libgtk-3-dev libcanberra-gtk3-module \
  libavcodec-dev libavformat-dev libswscale-dev \
  libv4l-dev libxvidcore-dev libx264-dev \
  libjpeg-dev libpng-dev libtiff-dev \
  libopenblas-dev liblapack-dev

mkdir -p "$(dirname "$OPENCV_SRC")"

if [[ ! -f "$OPENCV_SRC/CMakeLists.txt" ]]; then
  if [[ -f "$OPENCV_TAR" ]]; then
    echo "=== 解压本地包 $OPENCV_TAR ==="
    tar -xzf "$OPENCV_TAR" -C "$(dirname "$OPENCV_SRC")"
    extracted="$(dirname "$OPENCV_SRC")/opencv-${OPENCV_VER}"
    if [[ -d "$extracted" ]]; then
      rm -rf "$OPENCV_SRC"
      mv "$extracted" "$OPENCV_SRC"
    fi
  else
    echo "=== 克隆 OpenCV $OPENCV_VER (需能访问 GitHub) ==="
    git clone --depth 1 -b "$OPENCV_VER" https://github.com/opencv/opencv.git "$OPENCV_SRC"
  fi
fi

if [[ ! -f "$OPENCV_SRC/CMakeLists.txt" ]]; then
  echo "ERROR: 源码不存在。请把 opencv-${OPENCV_VER}.tar.gz 放到 $HOME/opencv411/ 后重试。"
  exit 1
fi

mkdir -p "$OPENCV_BUILD"
cd "$OPENCV_BUILD"

echo "=== CMake 配置 (CPU only) ==="
cmake "$OPENCV_SRC" \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_GENERATE_PKGCONFIG=ON \
  -D WITH_CUDA=OFF \
  -D WITH_OPENCL=OFF \
  -D BUILD_opencv_dnn=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_python3=OFF

echo "=== 编译 (日志: $LOG) ==="
make -j"$(nproc)" 2>&1 | tee "$LOG"

echo "=== 安装到 /usr/local (需 sudo) ==="
run_sudo make install
run_sudo ldconfig

echo "=== 完成 ==="
pkg-config --modversion opencv4
ls -la /usr/local/lib/libopencv_dnn.so
