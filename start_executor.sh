#!/bin/bash
# 启动 intercept_bridge（/ball_intercept → /vision/stewart_target）
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$WS_PATH/scripts/ros_env.sh"

echo "启动 volleyball_executor（intercept_bridge）..."
exec ros2 launch volleyball_executor executor.launch.py
