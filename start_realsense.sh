#!/bin/bash
# 兼容旧入口 → 请在 config/pipeline.conf 里设 USE_REALSENSE=true，然后 ./start_all.sh
export USE_REALSENSE=true
exec "$(dirname "$0")/start_all.sh" "$@"
