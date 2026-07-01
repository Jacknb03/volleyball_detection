#!/bin/bash
# 工控机诊断：启动 vision 并检查 /debug_image
set -eo pipefail
WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"
set +u
source /opt/ros/humble/setup.bash
CV_BRIDGE_OVERLAY="${CV_BRIDGE_OVERLAY:-$HOME/ros_cv_bridge_overlay}"
if [[ -f "$CV_BRIDGE_OVERLAY/install/setup.bash" ]]; then
  source "$CV_BRIDGE_OVERLAY/install/setup.bash"
fi
source install/setup.bash
set -u 2>/dev/null || true
export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

pkill -f 'yolo.launch.py|ball_detector_node|realsense2_camera' 2>/dev/null || true
sleep 1

LOG=/tmp/volleyball_launch.log
ros2 launch station_detector_cpp yolo.launch.py \
  pipeline_mode:=realsense \
  yolo_device:=cpu \
  model_path:="$WS/src/station_detector_cpp/model/best.onnx" \
  > "$LOG" 2>&1 &
LPID=$!
echo "launch_pid=$LPID log=$LOG"
sleep 25

echo "=== nodes ==="
ros2 node list || true
echo "=== debug_image ==="
timeout 5 ros2 topic hz /debug_image 2>&1 | head -4 || true
echo "=== log ==="
tail -40 "$LOG"
