#!/bin/bash
# 用 /usr/local OpenCV 4.11 重编 cv_bridge overlay（解决与 ball_detector 混链崩溃）
set -eo pipefail

OVERLAY="${OVERLAY:-$HOME/ros_cv_bridge_overlay}"
SRC="$OVERLAY/src/vision_opencv"
OPENCV_DIR="${OPENCV_DIR:-/usr/local/lib/cmake/opencv4}"

if [[ ! -f /usr/local/lib/libopencv_dnn.so ]]; then
  echo "请先安装 OpenCV 4.11: bash scripts/install_opencv411_cpu.sh"
  exit 1
fi

mkdir -p "$OVERLAY/src"
if [[ ! -f "$SRC/cv_bridge/CMakeLists.txt" ]]; then
  echo "=== 克隆 vision_opencv (humble) ==="
  git clone --depth 1 -b humble https://github.com/ros-perception/vision_opencv.git "$SRC"
fi

set +u
source /opt/ros/humble/setup.bash
set -u 2>/dev/null || true

cd "$OVERLAY"
colcon build --packages-select cv_bridge \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$OPENCV_DIR"

echo "=== cv_bridge 链接 ==="
ldd "$OVERLAY/install/cv_bridge/lib/libcv_bridge.so" | grep opencv

echo "overlay: source $OVERLAY/install/setup.bash"
