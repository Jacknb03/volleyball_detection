#!/bin/bash
# 工控机自检 — ./run.sh bash scripts/check_deploy.sh
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
set +u
source /opt/ros/humble/setup.bash
cd "$WS"
source install/setup.bash
set -u 2>/dev/null || true

MODEL="$WS/src/station_detector_cpp/model/best.onnx"
echo "=== 模型 ==="
ls -lh "$MODEL" 2>/dev/null || echo "MISSING $MODEL"

echo ""
echo "=== 节点（需 ./start_all.sh）==="
pgrep -af 'ball_detector|realsense2_camera|static_transform.*base_link' | head -5 || echo "无"

echo ""
echo "=== 话题 ==="
timeout 3 ros2 topic hz /camera/camera/color/image_raw 2>&1 | head -2 || true
timeout 3 ros2 topic hz /debug_image 2>&1 | head -2 || true
