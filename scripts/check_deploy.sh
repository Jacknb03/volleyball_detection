#!/bin/bash
# 工控机部署自检 — bash scripts/check_deploy.sh  或  ./run.sh bash scripts/check_deploy.sh
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$WS/scripts/ros_env.sh"

MODEL="$WS/src/station_detector_cpp/model/best.onnx"
INSTALL_MODEL="$(ros2 pkg prefix station_detector_cpp)/share/station_detector_cpp/model/best.onnx"

echo "=== 模型 ==="
for p in "$MODEL" "$INSTALL_MODEL"; do
  if [[ -f "$p" ]]; then
    echo "OK  $p ($(du -h "$p" | cut -f1))"
  else
    echo "MISS $p"
  fi
done

echo ""
echo "=== 节点（需先 ./start_all.sh）==="
pgrep -af 'ball_detector|realsense2_camera|static_transform.*base_link' | head -5 || echo "无相关进程"

echo ""
echo "=== 话题 ==="
timeout 3 ros2 topic hz /camera/camera/color/image_raw 2>&1 | head -2 || echo "无彩色图"
timeout 3 ros2 topic hz /ball_intercept 2>&1 | head -2 || echo "无 ball_intercept（KF 未激活或无检测）"

echo ""
echo "=== TF base_link -> camera_color_optical_frame ==="
timeout 4 ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame 2>&1 | head -6 || true

echo ""
echo "提示: 用 ./start_all.sh 启动（已自动 source）；调试请 ./run.sh ros2 ..."
