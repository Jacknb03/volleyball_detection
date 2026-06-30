#!/bin/bash
set -eo pipefail

ros_source() {
  set +u
  source /opt/ros/humble/setup.bash
  cd "$WS_PATH"
  source install/setup.bash
  set -u 2>/dev/null || true
}

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="${RVIZ_CONFIG:-$WS_PATH/config/volleyball_debug.rviz}"
PIPELINE_CONF="${PIPELINE_CONF:-$WS_PATH/config/pipeline.conf}"

# 读取 config/pipeline.conf（USE_REALSENSE / YOLO_DEVICE 等）
if [[ -f "$PIPELINE_CONF" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$PIPELINE_CONF"
  set -u 2>/dev/null || true
fi

ros_source

CUDNN_LIB="${CUDNN_LIB:-$HOME/opencv_build/cudnn/lib}"
if [[ -f /usr/local/lib/libopencv_dnn.so ]]; then
  export LD_LIBRARY_PATH="/usr/local/lib:$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
else
  OPENCV_CUDA_LIB="${OPENCV_CUDA_LIB:-$HOME/opencv_build/opencv/build_cuda/lib}"
  if [[ -d "$OPENCV_CUDA_LIB" ]]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$OPENCV_CUDA_LIB:${LD_LIBRARY_PATH:-}"
  fi
fi

# USE_REALSENSE=true → realsense；否则 video（命令行 PIPELINE_MODE 仍可覆盖）
if [[ -z "${PIPELINE_MODE:-}" ]]; then
  if [[ "${USE_REALSENSE:-false}" == "true" ]]; then
    PIPELINE_MODE=realsense
  else
    PIPELINE_MODE=video
  fi
fi

DEFAULT_VIDEO="$WS_PATH/src/station_detector_cpp/videos/test.mp4"
DEFAULT_MODEL="$WS_PATH/src/station_detector_cpp/model/best.onnx"

VIDEO_PATH="${VIDEO_PATH:-$DEFAULT_VIDEO}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
FRAME_RATE="${FRAME_RATE:-15.0}"
YOLO_DEVICE="${YOLO_DEVICE:-auto}"

LAUNCH_CMD="ros2 launch station_detector_cpp yolo.launch.py \
  pipeline_mode:=$PIPELINE_MODE \
  model_path:=$MODEL_PATH \
  yolo_device:=$YOLO_DEVICE"

if [[ "$PIPELINE_MODE" == "video" ]]; then
  LAUNCH_CMD="$LAUNCH_CMD video_path:=$VIDEO_PATH frame_rate:=$FRAME_RATE"
fi

RVIZ_CMD="rviz2"
if [[ -f "$RVIZ_CONFIG" ]]; then
  RVIZ_CMD="rviz2 -d $RVIZ_CONFIG"
fi

MODE_LABEL="视频 + bbox 估深"
if [[ "$PIPELINE_MODE" == "realsense" ]]; then
  MODE_LABEL="RealSense D455i + RGB-D 深度"
fi

echo "正在启动：$MODE_LABEL"
echo "配置: $PIPELINE_CONF"
echo ""
echo "Launch: $LAUNCH_CMD"
echo "RViz:   $RVIZ_CMD"
echo ""
echo "改模式: 编辑 config/pipeline.conf → USE_REALSENSE=true/false"
echo "改 GPU:  编辑 config/pipeline.conf → YOLO_DEVICE=cuda"
echo ""

if command -v gnome-terminal >/dev/null 2>&1; then
  RUN_PREFIX="set +u; source /opt/ros/humble/setup.bash; cd '$WS_PATH'; source install/setup.bash; set -u 2>/dev/null || true; export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}'; "
  gnome-terminal --tab --title="Volleyball_${PIPELINE_MODE}" -- bash -c "${RUN_PREFIX} ${LAUNCH_CMD}; exec bash"
  gnome-terminal --tab --title="RViz2" -- bash -c "${RUN_PREFIX} ${RVIZ_CMD}; exec bash"
else
  echo "未检测到 gnome-terminal，改为前台启动。"
  bash -c "$LAUNCH_CMD"
fi
