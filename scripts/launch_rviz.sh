#!/bin/bash
# 可选 RViz（由 start_all.sh 调用）
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$WS_PATH/scripts/ros_env.sh"

RVIZ_CONFIG="${RVIZ_CONFIG:-$WS_PATH/config/volleyball_debug.rviz}"
if [[ -f "$RVIZ_CONFIG" ]]; then
  exec rviz2 -d "$RVIZ_CONFIG"
else
  exec rviz2
fi
