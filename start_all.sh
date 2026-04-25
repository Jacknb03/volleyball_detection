#!/bin/bash
set -euo pipefail

# This repo is a ROS2 workspace. We resolve paths relative to this file,
# so the script no longer depends on $HOME/volleyball_detection.
WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source /opt/ros/humble/setup.bash
cd "$WS_PATH"
source install/setup.bash

echo "正在启动系统（C++ YOLO / 视频模式）..."

# Defaults (can be overridden by env vars)
VIDEO_PATH="${VIDEO_PATH:-$WS_PATH/src/station_detector/videos/test.mp4}"
MODEL_PATH="${MODEL_PATH:-$WS_PATH/src/station_detector_cpp/model/best.onnx}"
PARAMS_FILE="${PARAMS_FILE:-$WS_PATH/src/station_detector_cpp/config/ball_detector_params.yaml}"

LAUNCH_CMD="ros2 launch station_detector_cpp yolo_cpp_video.launch.py video_path:=$VIDEO_PATH model_path:=$MODEL_PATH params_file:=$PARAMS_FILE"

if command -v gnome-terminal >/dev/null 2>&1; then
  gnome-terminal --tab --title="YOLO_CPP_System" -- bash -c "$LAUNCH_CMD; exec bash"
  gnome-terminal --tab --title="RViz2" -- bash -c "rviz2; exec bash"
else
  echo "未检测到 gnome-terminal，改为前台启动（需要多终端请用 tmux/另开终端）。"
  echo "Launch: $LAUNCH_CMD"
  bash -c "$LAUNCH_CMD"
fi