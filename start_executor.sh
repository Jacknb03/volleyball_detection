#!/bin/bash
set -eo pipefail

WS_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /opt/ros/humble/setup.bash
cd "$WS_PATH"
source install/setup.bash
set -u 2>/dev/null || true

exec ros2 launch volleyball_executor executor.launch.py
