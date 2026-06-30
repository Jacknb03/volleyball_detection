# shellcheck shell=bash
# 工作区 ROS 环境 — 被 start_all.sh / run.sh / check_deploy.sh 等 source
# 用法: source "$(dirname "$0")/scripts/ros_env.sh"   或   ./run.sh ros2 topic list

if [[ -n "${VOLLEYBALL_ROS_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

: "${WS_PATH:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
cd "$WS_PATH" || exit 1
if [[ ! -f install/setup.bash ]]; then
  echo "错误: 未找到 install/setup.bash，请先在 $WS_PATH 执行: colcon build --symlink-install" >&2
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source install/setup.bash
set -u 2>/dev/null || true

# 可选：CUDA OpenCV（开发机 4060；工控机无显卡可忽略）
CUDNN_LIB="${CUDNN_LIB:-$HOME/opencv_build/cudnn/lib}"
if [[ -f /usr/local/lib/libopencv_dnn.so ]]; then
  export LD_LIBRARY_PATH="/usr/local/lib:$CUDNN_LIB:${LD_LIBRARY_PATH:-}"
else
  OPENCV_CUDA_LIB="${OPENCV_CUDA_LIB:-$HOME/opencv_build/opencv/build_cuda/lib}"
  if [[ -d "$OPENCV_CUDA_LIB" ]]; then
    export LD_LIBRARY_PATH="$CUDNN_LIB:$OPENCV_CUDA_LIB:${LD_LIBRARY_PATH:-}"
  fi
fi

export VOLLEYBALL_ROS_ENV_LOADED=1
export WS_PATH
