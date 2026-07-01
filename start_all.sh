#!/bin/bash
set -eo pipefail

ros_source() {
  set +u
  source /opt/ros/humble/setup.bash
  cd "$WS_PATH"
  if [[ -f install/setup.bash ]]; then
    source install/setup.bash
  fi
  set -u 2>/dev/null || true
}

load_conf() {
  local f="$1"
  if [[ -f "$f" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "$f"
    set -u 2>/dev/null || true
  fi
}

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_CONF="${PIPELINE_CONF:-$WS_PATH/config/pipeline.conf}"
PIPELINE_LOCAL="${PIPELINE_LOCAL:-$WS_PATH/config/pipeline.local.conf}"
RVIZ_CONFIG="${RVIZ_CONFIG:-$WS_PATH/config/volleyball_debug.rviz}"

load_conf "$PIPELINE_CONF"
load_conf "$PIPELINE_LOCAL"

# 缺模型时尝试 GitHub Release 下载
MODEL_DIR="$WS_PATH/src/station_detector_cpp/model"
need_model=false
if [[ "${YOLO_INPUT_SIZE:-640}" == "416" ]]; then
  [[ -f "$MODEL_DIR/best_416.onnx" ]] || need_model=true
else
  [[ -f "$MODEL_DIR/best.onnx" ]] || need_model=true
fi
if $need_model; then
  echo "未找到 ONNX，尝试从 GitHub Release 下载..."
  bash "$WS_PATH/scripts/download_models.sh" || {
    echo "WARN: 模型下载失败，请手动: bash scripts/download_models.sh"
    echo "      或: bash scripts/sync_to_ipc.sh USER@IPC --deploy"
  }
fi

ros_source

CV_BRIDGE_OVERLAY="${CV_BRIDGE_OVERLAY:-$HOME/ros_cv_bridge_overlay}"
if [[ -f "$CV_BRIDGE_OVERLAY/install/setup.bash" ]]; then
  set +u
  # shellcheck disable=SC1090
  source "$CV_BRIDGE_OVERLAY/install/setup.bash"
  set -u 2>/dev/null || true
fi

if [[ -f /usr/local/lib/libopencv_dnn.so ]]; then
  export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"
else
  CUDNN_LIB="${CUDNN_LIB:-$HOME/opencv_build/cudnn/lib}"
  OPENCV_CUDA_LIB="${OPENCV_CUDA_LIB:-$HOME/opencv_build/opencv/build_cuda/lib}"
  if [[ -d "$OPENCV_CUDA_LIB" ]]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$OPENCV_CUDA_LIB:${LD_LIBRARY_PATH:-}"
  fi
fi

if [[ -z "${PIPELINE_MODE:-}" ]]; then
  if [[ "${USE_REALSENSE:-false}" == "true" ]]; then
    PIPELINE_MODE=realsense
  else
    PIPELINE_MODE=video
  fi
fi

DEFAULT_VIDEO="$WS_PATH/src/station_detector_cpp/videos/test.mp4"
DEFAULT_MODEL="$WS_PATH/src/station_detector_cpp/model/best.onnx"
YOLO_INPUT_SIZE="${YOLO_INPUT_SIZE:-640}"
if [[ "$YOLO_INPUT_SIZE" == "416" ]]; then
  DEFAULT_MODEL="$WS_PATH/src/station_detector_cpp/model/best_416.onnx"
fi

VIDEO_PATH="${VIDEO_PATH:-$DEFAULT_VIDEO}"
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL}"
FRAME_RATE="${FRAME_RATE:-15.0}"
YOLO_DEVICE="${YOLO_DEVICE:-auto}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  YOLO_DEVICE=cpu
fi

START_RVIZ="${START_RVIZ:-true}"
START_EXECUTOR="${START_EXECUTOR:-true}"

LAUNCH_CMD="ros2 launch station_detector_cpp yolo.launch.py \
  pipeline_mode:=$PIPELINE_MODE \
  model_path:=$MODEL_PATH \
  yolo_device:=$YOLO_DEVICE"

if [[ "$PIPELINE_MODE" == "realsense" && "$YOLO_INPUT_SIZE" == "416" ]]; then
  PARAMS_416="$WS_PATH/src/station_detector_cpp/config/ball_detector_params_realsense_416.yaml"
  if [[ -f "$PARAMS_416" ]]; then
    LAUNCH_CMD="$LAUNCH_CMD params_file:=$PARAMS_416"
  fi
fi

if [[ "$PIPELINE_MODE" == "video" ]]; then
  LAUNCH_CMD="$LAUNCH_CMD video_path:=$VIDEO_PATH frame_rate:=$FRAME_RATE"
fi

EXECUTOR_CMD="ros2 launch volleyball_executor executor.launch.py"

RVIZ_CMD=""
if [[ "$START_RVIZ" == "true" ]]; then
  RVIZ_CMD="rviz2"
  if [[ -f "$RVIZ_CONFIG" ]]; then
    RVIZ_CMD="rviz2 -d $RVIZ_CONFIG"
  fi
fi

MODE_LABEL="视频 + bbox 估深"
if [[ "$PIPELINE_MODE" == "realsense" ]]; then
  MODE_LABEL="RealSense D455i + RGB-D 深度"
fi

echo "正在启动：$MODE_LABEL"
echo "配置: $PIPELINE_CONF"
[[ -f "$PIPELINE_LOCAL" ]] && echo "本地: $PIPELINE_LOCAL"
echo ""
echo "视觉:   $LAUNCH_CMD"
echo "桥接:   $([ "$START_EXECUTOR" = true ] && echo ON || echo OFF)  $EXECUTOR_CMD"
echo "RViz:   $([ "$START_RVIZ" = true ] && echo ON || echo OFF)  ${RVIZ_CMD:-—}"
echo ""
echo "执行端 launch 请单独启动（问队友要路径）。"
echo "开关: config/pipeline.conf 或 pipeline.local.conf → START_RVIZ / START_EXECUTOR"
echo ""

RUN_PREFIX="set +u; source /opt/ros/humble/setup.bash; cd '$WS_PATH'; "
RUN_PREFIX+="[[ -f install/setup.bash ]] && source install/setup.bash; "
if [[ -f "$CV_BRIDGE_OVERLAY/install/setup.bash" ]]; then
  RUN_PREFIX+="source '$CV_BRIDGE_OVERLAY/install/setup.bash'; "
fi
RUN_PREFIX+="set -u 2>/dev/null || true; export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}'; "

launch_tab() {
  local title="$1"
  local cmd="$2"
  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --tab --title="$title" -- bash -c "${RUN_PREFIX} ${cmd}; exec bash"
  else
    echo ">>> $title"
    bash -c "${RUN_PREFIX} ${cmd}" &
  fi
}

if command -v gnome-terminal >/dev/null 2>&1; then
  launch_tab "Vision_${PIPELINE_MODE}" "$LAUNCH_CMD"
  if [[ "$START_EXECUTOR" == "true" ]]; then
    launch_tab "Executor" "$EXECUTOR_CMD"
  fi
  if [[ "$START_RVIZ" == "true" && -n "$RVIZ_CMD" ]]; then
    launch_tab "RViz2" "$RVIZ_CMD"
  fi
else
  echo "未检测到 gnome-terminal，后台启动各组件。"
  launch_tab "Vision" "$LAUNCH_CMD"
  [[ "$START_EXECUTOR" == "true" ]] && launch_tab "Executor" "$EXECUTOR_CMD"
  [[ "$START_RVIZ" == "true" && -n "$RVIZ_CMD" ]] && launch_tab "RViz" "$RVIZ_CMD"
  wait
fi
