#!/bin/bash
# 关闭 ./start_all.sh 拉起的视觉 / 桥接 / RViz

set -u

echo "正在关闭排球检测相关节点..."

PATTERNS=(
  'ros2 launch station_detector_cpp yolo\.launch\.py'
  'ros2 launch volleyball_executor executor\.launch\.py'
  'ball_detector_node'
  'video_publisher'
  'realsense2_camera_node'
  'intercept_bridge_node'
  'static_transform_publisher.*--frame-id base_link --child-frame-id camera_optical_frame'
  'static_transform_publisher.*--frame-id base_link --child-frame-id camera_color_optical_frame'
  'static_transform_publisher.*--frame-id base_link --child-frame-id camera_link'
  'static_transform_publisher.*--frame-id odom --child-frame-id camera'
  'rviz2 -d.*volleyball_debug\.rviz'
)

stop_patterns() {
  local sig=$1
  for pat in "${PATTERNS[@]}"; do
    if pgrep -f "$pat" >/dev/null 2>&1; then
      pkill "$sig" -f "$pat" 2>/dev/null || true
    fi
  done
}

stop_patterns ''
sleep 1
stop_patterns -9

REMAINING=$(
  pgrep -af \
    'yolo\.launch|ball_detector|realsense2_camera|intercept_bridge|rviz2.*volleyball_debug' \
    2>/dev/null | grep -v pgrep || true
)

if [[ -n "$REMAINING" ]]; then
  echo ""
  echo "警告：以下进程可能仍在运行："
  echo "$REMAINING"
  exit 1
fi

echo "已全部关闭。"
