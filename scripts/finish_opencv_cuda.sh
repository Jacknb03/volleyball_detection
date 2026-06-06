#!/bin/bash
# Finish OpenCV CUDA build after setup_opencv_cuda.sh (needs g++-10 once).
set -eo pipefail

if ! command -v g++-10 >/dev/null 2>&1; then
  echo "Installing gcc-10 / g++-10 (sudo required) ..."
  sudo apt-get update
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y gcc-10 g++-10
fi

CUDNN_PREFIX="${CUDNN_PREFIX:-$HOME/opencv_build/cudnn}"
OPENCV_BUILD="${OPENCV_BUILD:-$HOME/opencv_build/opencv/build_cuda}"
OPENCV_SRC="${OPENCV_SRC:-$HOME/opencv_build/opencv}"
OPENCV_CONTRIB="${OPENCV_CONTRIB:-$HOME/opencv_build/opencv_contrib}"
JOBS="${JOBS:-$(nproc)}"

export PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"
export LD_LIBRARY_PATH="$CUDNN_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "=== Reconfigure (gcc-10 host compiler, sm_86 for RTX 4060 + CUDA 11.5) ==="
cd "$OPENCV_BUILD"
cmake -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 \
  -D CUDA_HOST_COMPILER=/usr/bin/g++-10 \
  -D CMAKE_CXX_COMPILER=/usr/bin/g++-10 \
  -D CMAKE_C_COMPILER=/usr/bin/gcc-10 \
  -D OPENCV_EXTRA_MODULES_PATH="$OPENCV_CONTRIB/modules" \
  -D WITH_CUDA=ON -D WITH_CUDNN=ON -D OPENCV_DNN_CUDA=ON \
  -D CUDNN_INCLUDE_DIR="$CUDNN_PREFIX/include" \
  -D CUDNN_LIBRARY="$CUDNN_PREFIX/lib/libcudnn.so" \
  -D CUDA_ARCH_BIN=8.6 -D CUDA_ARCH_PTX=8.6 \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D BUILD_LIST=cudev,core,imgproc,imgcodecs,dnn,highgui,videoio \
  -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF \
  -D BUILD_opencv_python3=OFF \
  "$OPENCV_SRC"

echo "=== Build OpenCV (~15–30 min, minimal modules) ==="
make -j"$JOBS" 2>&1 | tee /tmp/opencv_cuda_build.log

echo "=== Install to /usr/local (sudo) ==="
sudo make install
sudo ldconfig

echo "=== Rebuild station_detector_cpp ==="
WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
set -u 2>/dev/null || true
cd "$WS_PATH"
colcon build --symlink-install --packages-select station_detector_cpp 2>&1 | tee /tmp/colcon_cuda_build.log

if grep -q "OpenCV CUDA detected" /tmp/colcon_cuda_build.log; then
  echo ""
  echo "SUCCESS. Run: YOLO_DEVICE=cuda ./start_all.sh"
else
  echo ""
  echo "WARNING: check /tmp/colcon_cuda_build.log"
fi
