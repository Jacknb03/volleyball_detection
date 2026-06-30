#!/bin/bash
# 实际启动视觉 launch（由 start_all.sh / gnome-terminal 调用，已设好环境变量）
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$WS_PATH/scripts/ros_env.sh"

PIPELINE_MODE="${PIPELINE_MODE:-realsense}"
MODEL_PATH="${MODEL_PATH:-$WS_PATH/src/station_detector_cpp/model/best.onnx}"
YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
VIDEO_PATH="${VIDEO_PATH:-$WS_PATH/src/station_detector_cpp/videos/test.mp4}"
FRAME_RATE="${FRAME_RATE:-15.0}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "错误: YOLO 模型不存在: $MODEL_PATH" >&2
  exit 1
fi

ARGS=(
  ros2 launch station_detector_cpp yolo.launch.py
  "pipeline_mode:=${PIPELINE_MODE}"
  "model_path:=${MODEL_PATH}"
  "yolo_device:=${YOLO_DEVICE}"
)

if [[ "$PIPELINE_MODE" == "video" ]]; then
  ARGS+=("video_path:=${VIDEO_PATH}" "frame_rate:=${FRAME_RATE}")
fi

echo "执行: ${ARGS[*]}"
exec "${ARGS[@]}"
