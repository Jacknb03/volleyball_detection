#!/bin/bash
# 工控机本地编译 station_detector_cpp（勿用 symlink-install，避免坏链接）
set -eo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"

set +u
source /opt/ros/humble/setup.bash
CV_BRIDGE_OVERLAY="${CV_BRIDGE_OVERLAY:-$HOME/ros_cv_bridge_overlay}"
if [[ -f "$CV_BRIDGE_OVERLAY/install/setup.bash" ]]; then
  source "$CV_BRIDGE_OVERLAY/install/setup.bash"
fi
set -u 2>/dev/null || true

if [[ ! -f /usr/local/lib/libopencv_dnn.so ]]; then
  echo "ERROR: OpenCV 4.11 未安装，先跑 install_opencv411_cpu.sh"
  exit 1
fi

rm -rf build/station_detector_cpp install/station_detector_cpp
OpenCV_DIR=/usr/local/lib/cmake/opencv4 colcon build --packages-select station_detector_cpp

BIN="$WS/install/station_detector_cpp/lib/station_detector_cpp/ball_detector_node"
if [[ ! -f "$BIN" ]] || [[ -L "$BIN" ]]; then
  echo "ERROR: 二进制无效: $BIN"
  ls -la "$BIN" 2>&1 || true
  exit 1
fi

echo "OK: $BIN"
file "$BIN"
ldd "$BIN" | grep opencv | head -4
