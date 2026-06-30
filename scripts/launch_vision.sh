#!/bin/bash
# 实际启动视觉 launch（由 start_all.sh 调用，可单独运行）
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPELINE_CONF="${PIPELINE_CONF:-$WS_PATH/config/pipeline.conf}"

if [[ -f "$PIPELINE_CONF" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$PIPELINE_CONF"
  set -u 2>/dev/null || true
fi

# shellcheck disable=SC1091
source "$WS_PATH/scripts/ros_env.sh"

if [[ -z "${PIPELINE_MODE:-}" ]]; then
  if [[ "${USE_REALSENSE:-false}" == "true" ]]; then
    PIPELINE_MODE=realsense
  else
    PIPELINE_MODE=video
  fi
fi

MODEL_PATH="${MODEL_PATH:-$WS_PATH/src/station_detector_cpp/model/best.onnx}"
VIDEO_PATH="${VIDEO_PATH:-$WS_PATH/src/station_detector_cpp/videos/test.mp4}"
FRAME_RATE="${FRAME_RATE:-15.0}"
YOLO_DEVICE="${YOLO_DEVICE:-cpu}"

# 工控机无独显时 cuda 无意义，auto/cuda 仍可回退 CPU，但显式 cpu 更稳
if [[ "$YOLO_DEVICE" == "cuda" ]] && ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "警告: 无 NVIDIA GPU，YOLO_DEVICE 由 cuda 改为 cpu"
  YOLO_DEVICE=cpu
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "错误: YOLO 模型不存在: $MODEL_PATH" >&2
  exit 1
fi

cd "$WS_PATH"

ARGS=(
  ros2 launch station_detector_cpp yolo.launch.py
  "pipeline_mode:=${PIPELINE_MODE}"
  "model_path:=${MODEL_PATH}"
  "yolo_device:=${YOLO_DEVICE}"
)

if [[ "$PIPELINE_MODE" == "video" ]]; then
  ARGS+=("video_path:=${VIDEO_PATH}" "frame_rate:=${FRAME_RATE}")
fi

echo "工作区: $WS_PATH"
echo "执行: ${ARGS[*]}"
exec "${ARGS[@]}"
