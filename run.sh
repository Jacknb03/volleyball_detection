#!/bin/bash
# 已 source 工作区的 ros2 命令包装 — 无需手动 source
# 例: ./run.sh ros2 topic echo /ball_intercept --once
#     ./run.sh ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
#     ./run.sh bash scripts/check_deploy.sh
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$WS_PATH/scripts/ros_env.sh"

if [[ $# -eq 0 ]]; then
  echo "用法: $0 <命令> [参数...]"
  echo "示例: $0 ros2 topic list"
  exit 1
fi

exec "$@"
