#!/bin/bash
# Rebuild OpenCV with CUDA + cuDNN so station_detector_cpp can use yolo.device:=cuda
#
# Prerequisites: NVIDIA driver already installed (nvidia-smi works).
# Run:  bash scripts/setup_opencv_cuda.sh
# Then: cd ~/volleyball_detection && colcon build --symlink-install --packages-select station_detector_cpp
#       YOLO_DEVICE=cuda ./start_all.sh
#
set -eo pipefail

OPENCV_SRC="${OPENCV_SRC:-$HOME/opencv_build/opencv}"
OPENCV_CONTRIB="${OPENCV_CONTRIB:-$HOME/opencv_build/opencv_contrib}"
OPENCV_BUILD="${OPENCV_BUILD:-$HOME/opencv_build/opencv/build_cuda}"
JOBS="${JOBS:-$(nproc)}"

echo "=== Step 0: checks ==="
if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi failed. Install NVIDIA driver first."
  exit 1
fi

if [[ ! -d "$OPENCV_SRC" ]]; then
  echo "ERROR: OpenCV source not found at $OPENCV_SRC"
  echo "Clone first (see docs/DEPLOYMENT.md §1.3)."
  exit 1
fi

echo "=== Step 1: install CUDA toolkit + cuDNN (requires sudo) ==="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "Installing nvidia-cuda-toolkit ..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    nvidia-cuda-toolkit nvidia-cudnn
else
  echo "nvcc already present: $(nvcc --version | tail -1)"
fi

# Ubuntu meta-package may put nvcc here
export PATH="/usr/local/cuda/bin:/usr/lib/nvidia-cuda-toolkit/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc still not found after install. Check CUDA toolkit."
  exit 1
fi

echo "=== Step 1b: ensure cuDNN libraries (apt nvidia-cudnn is only a downloader) ==="
CUDNN_PREFIX="${CUDNN_PREFIX:-$HOME/opencv_build/cudnn}"
if [[ ! -f "$CUDNN_PREFIX/lib/libcudnn.so" ]]; then
  if command -v update-nvidia-cudnn >/dev/null 2>&1; then
    echo "Installing cuDNN via update-nvidia-cudnn (needs sudo once) ..."
    sudo update-nvidia-cudnn --update --prefix /usr || true
  fi
  if [[ ! -f /usr/lib/x86_64-linux-gnu/libcudnn.so ]] && [[ -f /tmp/nvidia-cudnn/cudnn.tgz ]]; then
    echo "Fallback: extract cuDNN to $CUDNN_PREFIX"
    mkdir -p "$CUDNN_PREFIX/lib" "$CUDNN_PREFIX/include"
    tar xf /tmp/nvidia-cudnn/cudnn.tgz -C /tmp/nvidia-cudnn/ 2>/dev/null || true
    cp -a /tmp/nvidia-cudnn/cuda/lib64/libcudnn* "$CUDNN_PREFIX/lib/" 2>/dev/null || true
    cp -a /tmp/nvidia-cudnn/cuda/include/cudnn*.h "$CUDNN_PREFIX/include/" 2>/dev/null || true
  fi
fi

CUDNN_CMAKE=()
if [[ -f "$CUDNN_PREFIX/lib/libcudnn.so" ]]; then
  CUDNN_CMAKE=(
    -D CUDNN_INCLUDE_DIR="$CUDNN_PREFIX/include"
    -D CUDNN_LIBRARY="$CUDNN_PREFIX/lib/libcudnn.so"
  )
  export LD_LIBRARY_PATH="$CUDNN_PREFIX/lib:${LD_LIBRARY_PATH:-}"
elif [[ -f /usr/lib/x86_64-linux-gnu/libcudnn.so ]]; then
  echo "Using system cuDNN in /usr"
else
  echo "ERROR: libcudnn not found. Run: sudo update-nvidia-cudnn --update"
  exit 1
fi

echo "=== Step 2: configure OpenCV with CUDA ==="
mkdir -p "$OPENCV_BUILD"
cd "$OPENCV_BUILD"

EXTRA_MODULES=""
if [[ -d "$OPENCV_CONTRIB/modules" ]]; then
  EXTRA_MODULES="-D OPENCV_EXTRA_MODULES_PATH=$OPENCV_CONTRIB/modules"
fi

cmake -D CMAKE_BUILD_TYPE=Release \
  $EXTRA_MODULES \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D CUDA_ARCH_BIN=8.6 \
  -D CUDA_ARCH_PTX=8.6 \
  "${CUDNN_CMAKE[@]}" \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_python3=OFF \
  "$OPENCV_SRC"

echo "=== Step 3: build & install OpenCV (may take 30–60 min) ==="
make -j"$JOBS"
sudo make install
sudo ldconfig

echo "=== Step 4: rebuild volleyball_detection C++ node ==="
WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u 2>/dev/null || true
cd "$WS_PATH"
colcon build --symlink-install --packages-select station_detector_cpp 2>&1 | tee /tmp/colcon_cuda_build.log

if grep -q "OpenCV CUDA detected" /tmp/colcon_cuda_build.log; then
  echo ""
  echo "SUCCESS: OpenCV CUDA enabled. Start with:"
  echo "  cd $WS_PATH && YOLO_DEVICE=cuda ./start_all.sh"
else
  echo ""
  echo "WARNING: CMake did not report 'OpenCV CUDA detected'."
  echo "Check /tmp/colcon_cuda_build.log — you may still be on CPU."
fi
