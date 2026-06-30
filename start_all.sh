#!/bin/bash
# 启动视觉 + 可选 RViz（工控机有桌面时用 gnome-terminal；否则前台运行）
set -euo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="${RVIZ_CONFIG:-$WS_PATH/config/volleyball_debug.rviz}"
PIPELINE_CONF="${PIPELINE_CONF:-$WS_PATH/config/pipeline.conf}"
USE_RVIZ="${USE_RVIZ:-true}"

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

export PIPELINE_MODE
export MODEL_PATH="${MODEL_PATH:-$WS_PATH/src/station_detector_cpp/model/best.onnx}"
export VIDEO_PATH="${VIDEO_PATH:-$WS_PATH/src/station_detector_cpp/videos/test.mp4}"
export FRAME_RATE="${FRAME_RATE:-15.0}"
export YOLO_DEVICE="${YOLO_DEVICE:-cpu}"
export RVIZ_CONFIG

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "错误: YOLO 模型不存在: $MODEL_PATH"
  echo "请把 best.onnx 放到 src/station_detector_cpp/model/ 或设置 MODEL_PATH"
  exit 1
fi

echo "YOLO 模型: $MODEL_PATH ($(du -h "$MODEL_PATH" | cut -f1))"
echo "模式: $PIPELINE_MODE | 设备: $YOLO_DEVICE | RViz: $USE_RVIZ"
echo "配置: $PIPELINE_CONF"
echo ""
echo "调试（无需 source）:"
echo "  ./run.sh ros2 topic echo /ball_intercept --once"
echo "  ./start_executor.sh"
echo ""

VISION_SCRIPT="$WS_PATH/scripts/launch_vision.sh"
RVIZ_SCRIPT="$WS_PATH/scripts/launch_rviz.sh"

use_gnome=false
if command -v gnome-terminal >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
  use_gnome=true
fi

if $use_gnome; then
  echo "使用 gnome-terminal 启动..."
  if ! gnome-terminal --tab --title="Volleyball_${PIPELINE_MODE}" -- \
      bash -lc "'$VISION_SCRIPT'; exec bash"; then
    echo "警告: gnome-terminal 失败，改前台启动。"
    use_gnome=false
  elif [[ "${USE_RVIZ}" == "true" ]]; then
    gnome-terminal --tab --title="RViz2" -- \
      bash -lc "'$RVIZ_SCRIPT'; exec bash" || echo "警告: RViz 标签页未打开（可忽略）。"
  fi
  if $use_gnome; then
    echo "已在 gnome-terminal 中启动。"
    exit 0
  fi
fi

if [[ -z "${DISPLAY:-}" ]]; then
  echo "未检测到 DISPLAY（SSH 远程登录？）。改为本终端前台启动视觉节点。"
  echo "若要在工控机本机桌面启动，请在本机图形终端里运行 ./start_all.sh"
  echo ""
fi

echo "前台启动视觉 launch（Ctrl+C 停止）..."
exec "$VISION_SCRIPT"
