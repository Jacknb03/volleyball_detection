#!/bin/bash
# 启动视觉 + 可选 RViz
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RVIZ_CONFIG="${RVIZ_CONFIG:-$WS_PATH/config/volleyball_debug.rviz}"
PIPELINE_CONF="${PIPELINE_CONF:-$WS_PATH/config/pipeline.conf}"
USE_RVIZ="${USE_RVIZ:-true}"

if [[ ! -f "$WS_PATH/scripts/launch_vision.sh" ]]; then
  echo "错误: 缺少 scripts/launch_vision.sh"
  exit 1
fi

if [[ ! -f "$WS_PATH/src/station_detector_cpp/model/best.onnx" ]]; then
  echo "错误: 缺少 best.onnx → src/station_detector_cpp/model/"
  exit 1
fi

echo "=== 排球视觉启动 ==="
echo "配置: $PIPELINE_CONF"
echo "RViz: $USE_RVIZ（画面来自 ball_detector 的 /debug_image，不是原始相机话题）"
echo ""

VISION_SCRIPT="$WS_PATH/scripts/launch_vision.sh"
RVIZ_SCRIPT="$WS_PATH/scripts/launch_rviz.sh"

# 与桌面/SSH 无关：有 DISPLAY 且 gnome-terminal 可用 → 开新 tab；否则当前终端前台跑
if command -v gnome-terminal >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
  echo "→ gnome-terminal 启动视觉 tab..."
  gnome-terminal --tab --title="Volleyball" -- \
    env PIPELINE_CONF="$PIPELINE_CONF" WS_PATH="$WS_PATH" bash "$VISION_SCRIPT" || {
    echo "gnome-terminal 失败，改前台启动"
    exec env PIPELINE_CONF="$PIPELINE_CONF" WS_PATH="$WS_PATH" bash "$VISION_SCRIPT"
  }
  if [[ "${USE_RVIZ}" == "true" ]]; then
    echo "→ gnome-terminal 启动 RViz tab..."
    gnome-terminal --tab --title="RViz2" -- \
      env RVIZ_CONFIG="$RVIZ_CONFIG" WS_PATH="$WS_PATH" bash "$RVIZ_SCRIPT" || true
  fi
  echo "已提交到 gnome-terminal。请查看 Volleyball tab 是否有 launch 输出。"
  echo "验证: ./run.sh ros2 topic hz /camera/camera/color/image_raw"
  echo "      ./run.sh ros2 topic hz /debug_image"
  exit 0
fi

echo "→ 当前终端前台启动（SSH 或无 gnome-terminal 时正常）..."
exec env PIPELINE_CONF="$PIPELINE_CONF" WS_PATH="$WS_PATH" bash "$VISION_SCRIPT"
