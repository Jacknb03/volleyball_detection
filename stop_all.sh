#!/bin/bash
# 关闭 ./start_all.sh 或 ros2 launch station_detector_cpp yolo*.launch.py 拉起的进程

set -u

echo "正在关闭排球检测相关节点..."

# 按优先级：先停 launch 监督进程，再停各节点（避免 launch 还在时子进程被拉起）
PATTERNS=(
  'ros2 launch station_detector_cpp yolo\.launch\.py'
  'ball_detector_node'
  'video_publisher'
  'realsense2_camera_node'
  'static_transform_publisher.*--frame-id base_link --child-frame-id camera_optical_frame'
  'static_transform_publisher.*--frame-id base_link --child-frame-id camera_color_optical_frame'
  'static_transform_publisher.*--frame-id base_link --child-frame-id camera_link'
  'static_transform_publisher.*--frame-id odom --child-frame-id camera'
  'intercept_bridge_node'
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
    'ros2 launch station_detector_cpp yolo|ball_detector_node|video_publisher|realsense2_camera_node|static_transform_publisher.*--frame-id (base_link|odom)|intercept_bridge_node|rviz2 -d.*volleyball_debug' \
    2>/dev/null | grep -v pgrep || true
)

if [[ -n "$REMAINING" ]]; then
  echo ""
  echo "警告：以下进程可能仍在运行（请手动检查）："
  echo "$REMAINING"
  exit 1
fi

echo "已全部关闭。"
