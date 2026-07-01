#!/bin/bash
# 工控机（无 NVIDIA / x86_64）快速部署
#
# 用法:
#   bash scripts/deploy_ipc.sh              # 常规：假定 ROS + OpenCV 4.11 已有，拉代码后编译
#   bash scripts/deploy_ipc.sh --check      # 只检查环境，不编译
#   bash scripts/deploy_ipc.sh --pull       # git pull 后再编译
#   bash scripts/deploy_ipc.sh --full       # 新机器从零：RealSense apt + OpenCV 4.11 源码编译 + 全量编译（数小时）
#   bash scripts/deploy_ipc.sh --skip-opencv  # 跳过 cv_bridge overlay（系统 OpenCV 4.5 会崩，不推荐）
#
# 模型文件 *.onnx 不进 git，需从开发机拷:
#   rsync -avz src/station_detector_cpp/model/best_416.onnx USER@IPC:~/volleyball_detection/src/station_detector_cpp/model/
#
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"

DO_CHECK=false
DO_PULL=false
DO_FULL=false
SKIP_OPENCV_OVERLAY=false

for arg in "$@"; do
  case "$arg" in
    --check) DO_CHECK=true ;;
    --pull) DO_PULL=true ;;
    --full) DO_FULL=true ;;
    --skip-opencv) SKIP_OPENCV_OVERLAY=true ;;
    -h|--help)
      sed -n '2,14p' "$0"
      exit 0
      ;;
    *)
      echo "未知参数: $arg （用 --help）"
      exit 1
      ;;
  esac
done

log() { echo ""; echo "=== $* ==="; }

check_ros() {
  if [[ ! -f /opt/ros/humble/setup.bash ]]; then
    echo "ERROR: 未找到 ROS2 Humble。请先安装: https://docs.ros.org/en/humble/Installation.html"
    exit 1
  fi
  log "ROS2 Humble OK"
}

check_opencv411() {
  if [[ -f /usr/local/lib/libopencv_dnn.so ]]; then
    local ver
    ver=$(pkg-config --modversion opencv4 2>/dev/null || true)
    echo "OpenCV: ${ver:-unknown} @ /usr/local"
    return 0
  fi
  echo "WARN: /usr/local 无 OpenCV 4.11（工控机 YOLO ONNX 需 4.11，4.5 会 forward 失败）"
  echo "      运行: bash scripts/install_opencv411_cpu.sh"
  echo "      或整包: bash scripts/deploy_ipc.sh --full"
  return 1
}

check_models() {
  log "ONNX 模型"
  local ok=true
  for f in best.onnx best_416.onnx; do
    local p="$WS/src/station_detector_cpp/model/$f"
    if [[ -f "$p" ]]; then
      ls -lh "$p"
    else
      echo "MISSING: $p"
      ok=false
    fi
  done
  if [[ "$ok" != true ]]; then
    echo ""
    echo "从开发机拷贝示例:"
    echo "  rsync -avz src/station_detector_cpp/model/best_416.onnx USER@本机IP:$WS/src/station_detector_cpp/model/"
    return 1
  fi
  return 0
}

check_binary() {
  local bin="$WS/install/station_detector_cpp/lib/station_detector_cpp/ball_detector_node"
  if [[ ! -f "$bin" ]] || [[ -L "$bin" ]]; then
    echo "ERROR: ball_detector_node 无效（勿用 symlink-install 拷 install/）: $bin"
    ls -la "$bin" 2>&1 || true
    return 1
  fi
  file "$bin"
  ldd "$bin" | grep opencv | head -3
  return 0
}

ros_source() {
  set +u
  source /opt/ros/humble/setup.bash
  CV_BRIDGE_OVERLAY="${CV_BRIDGE_OVERLAY:-$HOME/ros_cv_bridge_overlay}"
  if [[ -f "$CV_BRIDGE_OVERLAY/install/setup.bash" ]]; then
    source "$CV_BRIDGE_OVERLAY/install/setup.bash"
  fi
  cd "$WS"
  if [[ -f install/setup.bash ]]; then
    source install/setup.bash
  fi
  set -u 2>/dev/null || true
}

install_ros_build_deps() {
  log "ROS 编译依赖"
  set +u
  source /opt/ros/humble/setup.bash
  set -u 2>/dev/null || true
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    libeigen3-dev \
    2>/dev/null || true
  if ! rosdep --version &>/dev/null; then
    sudo apt-get install -y python3-rosdep
  fi
  if [[ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]]; then
    sudo rosdep init 2>/dev/null || true
  fi
  rosdep update 2>/dev/null || true
  rosdep install --from-paths src --ignore-src -r -y 2>/dev/null || true
}

build_workspace() {
  log "colcon build（无 symlink-install）"
  ros_source
  rm -rf build install log
  local cmake_extra=()
  if [[ -f /usr/local/lib/cmake/opencv4/OpenCVConfig.cmake ]]; then
    cmake_extra+=(--cmake-args "-DOpenCV_DIR=/usr/local/lib/cmake/opencv4")
  fi
  colcon build --packages-select volleyball_msgs station_detector_cpp volleyball_executor \
    "${cmake_extra[@]}"
  check_binary
}

# --- main ---

log "工控机部署 @ $WS"
check_ros
OPENCV_OK=false
check_opencv411 && OPENCV_OK=true || true
check_models || true

if $DO_CHECK; then
  echo ""
  echo "检查完成（--check 模式，未编译）"
  $OPENCV_OK && [[ -f "$WS/install/station_detector_cpp/lib/station_detector_cpp/ball_detector_node" ]] && check_binary || true
  exit 0
fi

if $DO_FULL; then
  log "完整安装：RealSense + OpenCV 4.11"
  bash "$WS/scripts/install_realsense_deps.sh"
  bash "$WS/scripts/install_opencv411_cpu.sh"
  OPENCV_OK=true
fi

if $DO_PULL; then
  log "git pull"
  git pull --ff-only
fi

if ! $SKIP_OPENCV_OVERLAY && $OPENCV_OK; then
  bash "$WS/scripts/rebuild_cv_bridge_opencv411.sh"
elif ! $OPENCV_OK; then
  echo "ERROR: 无 OpenCV 4.11，无法安全编译。加 --full 或先 install_opencv411_cpu.sh"
  exit 1
fi

install_ros_build_deps
build_workspace

log "部署完成"
echo ""
echo "下一步:"
echo "  1. 确认 config/pipeline.conf（USE_REALSENSE=true, YOLO_INPUT_SIZE=416, YOLO_DEVICE=cpu）"
echo "  2. ./stop_all.sh && ./start_all.sh"
echo "  3. bash scripts/check_deploy.sh"
echo "  4. 联调: ros2 launch volleyball_executor executor.launch.py"
echo "  5. 详见: src/volleyball_executor/docs/INTEGRATION_CHECKLIST.md"
