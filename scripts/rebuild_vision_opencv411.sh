#!/bin/bash
# OpenCV 4.11 装好后，重编 cv_bridge overlay + station_detector_cpp
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
bash "$WS/scripts/rebuild_cv_bridge_opencv411.sh"
cd "$WS"

set +u
source /opt/ros/humble/setup.bash
CV_BRIDGE_OVERLAY="${CV_BRIDGE_OVERLAY:-$HOME/ros_cv_bridge_overlay}"
source "$CV_BRIDGE_OVERLAY/install/setup.bash"
set -u 2>/dev/null || true

rm -rf build/station_detector_cpp install/station_detector_cpp
OpenCV_DIR=/usr/local/lib/cmake/opencv4 colcon build --packages-select station_detector_cpp
set +u
source install/setup.bash
set -u 2>/dev/null || true

echo "=== ball_detector 链接 ==="
ldd install/station_detector_cpp/lib/station_detector_cpp/ball_detector_node | grep opencv

echo "=== cv_bridge 链接 ==="
ldd "$CV_BRIDGE_OVERLAY/install/cv_bridge/lib/libcv_bridge.so" | grep opencv

echo "完成。运行: ./stop_all.sh && ./start_all.sh"
