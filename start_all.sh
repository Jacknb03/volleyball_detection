#!/bin/bash
set -eo pipefail

# ROS setup.bash references unset vars; must not use 'set -u' before sourcing.
ros_source() {
  set +u
  source /opt/ros/humble/setup.bash
  cd "$WS_PATH"
  source install/setup.bash
  set -u 2>/dev/null || true
}

# ROS2 workspace root (portable — not tied to $HOME path).
WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="${RVIZ_CONFIG:-$WS_PATH/config/volleyball_debug.rviz}"

ros_source

# Prefer sudo-installed OpenCV CUDA (/usr/local); fallback to build tree.
CUDNN_LIB="${CUDNN_LIB:-$HOME/opencv_build/cudnn/lib}"
if [[ -f /usr/local/lib/libopencv_dnn.so ]]; then
  export LD_LIBRARY_PATH="/usr/local/lib:$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
else
  OPENCV_CUDA_LIB="${OPENCV_CUDA_LIB:-$HOME/opencv_build/opencv/build_cuda/lib}"
  if [[ -d "$OPENCV_CUDA_LIB" ]]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$OPENCV_CUDA_LIB:${LD_LIBRARY_PATH:-}"
  fi
fi

echo "正在启动系统（C++ YOLO / 视频模式）..."

# Defaults (override via env vars)
VIDEO_PATH="${VIDEO_PATH:-$WS_PATH/src/station_detector/videos/test.mp4}"
MODEL_PATH="${MODEL_PATH:-$WS_PATH/src/station_detector_cpp/model/best.onnx}"
PARAMS_FILE="${PARAMS_FILE:-$WS_PATH/src/station_detector_cpp/config/ball_detector_params.yaml}"
FRAME_RATE="${FRAME_RATE:-15.0}"   # must be float, not integer
YOLO_DEVICE="${YOLO_DEVICE:-auto}" # auto | cpu | cuda (cuda: run scripts/setup_opencv_cuda.sh first)

LAUNCH_CMD="ros2 launch station_detector_cpp yolo_cpp_video.launch.py \
  video_path:=$VIDEO_PATH \
  model_path:=$MODEL_PATH \
  params_file:=$PARAMS_FILE \
  frame_rate:=$FRAME_RATE \
  yolo_device:=$YOLO_DEVICE"

RVIZ_CMD="rviz2"
if [[ -f "$RVIZ_CONFIG" ]]; then
  RVIZ_CMD="rviz2 -d $RVIZ_CONFIG"
fi

RQT_CMD="ros2 run rqt_image_view rqt_image_view"

echo ""
echo "Launch: $LAUNCH_CMD"
echo "RViz:   $RVIZ_CMD"
echo ""
echo "RViz 已预配置: Fixed Frame=odom, /debug_image, /volleyball_trajectory"
echo "rqt 请选 /debug_image（不是 /image_raw）"
echo "GPU 加速: bash scripts/setup_opencv_cuda.sh 完成后 YOLO_DEVICE=cuda ./start_all.sh"
echo ""

if command -v gnome-terminal >/dev/null 2>&1; then
  RUN_PREFIX="set +u; source /opt/ros/humble/setup.bash; cd '$WS_PATH'; source install/setup.bash; set -u 2>/dev/null || true; export PYTHONPATH='/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages'; export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}'; "
  gnome-terminal --tab --title="YOLO_CPP_System" -- bash -c "${RUN_PREFIX} ${LAUNCH_CMD}; exec bash"
  gnome-terminal --tab --title="RViz2" -- bash -c "${RUN_PREFIX} ${RVIZ_CMD}; exec bash"
  gnome-terminal --tab --title="rqt_image_view" -- bash -c "${RUN_PREFIX} ${RQT_CMD}; exec bash"
else
  echo "未检测到 gnome-terminal，改为前台启动（需要多终端请用 tmux/另开终端）。"
  bash -c "$LAUNCH_CMD"
fi
