#!/bin/bash
# 可选：免手敲 source 跑 ros2 命令 — ./run.sh ros2 topic list
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /opt/ros/humble/setup.bash
cd "$WS_PATH"
source install/setup.bash
set -u 2>/dev/null || true

exec "$@"
