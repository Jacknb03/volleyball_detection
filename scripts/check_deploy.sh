#!/bin/bash
# 工控机跑通自检 — bash scripts/check_deploy.sh
set -eo pipefail

WS="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WS"

set +u
source /opt/ros/humble/setup.bash
CV_BRIDGE_OVERLAY="${CV_BRIDGE_OVERLAY:-$HOME/ros_cv_bridge_overlay}"
if [[ -f "$CV_BRIDGE_OVERLAY/install/setup.bash" ]]; then
  source "$CV_BRIDGE_OVERLAY/install/setup.bash"
fi
[[ -f install/setup.bash ]] && source install/setup.bash
set -u 2>/dev/null || true

export LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH:-}"

pass() { echo "[OK] $*"; }
warn() { echo "[WARN] $*"; }
fail() { echo "[FAIL] $*"; }

echo "=== 环境 ==="
if [[ -f /usr/local/lib/libopencv_dnn.so ]]; then
  pass "OpenCV 4.11 @ /usr/local ($(pkg-config --modversion opencv4 2>/dev/null || echo '?'))"
else
  warn "无 /usr/local OpenCV 4.11"
fi

if [[ -f "$CV_BRIDGE_OVERLAY/install/cv_bridge/lib/libcv_bridge.so" ]]; then
  pass "cv_bridge overlay: $CV_BRIDGE_OVERLAY"
else
  warn "无 cv_bridge overlay（混链可能崩溃）"
fi

echo ""
echo "=== 模型 ==="
for f in best_416.onnx best.onnx; do
  p="$WS/src/station_detector_cpp/model/$f"
  if [[ -f "$p" ]]; then
    pass "$f ($(du -h "$p" | cut -f1))"
  else
    fail "MISSING $p"
  fi
done

echo ""
echo "=== 二进制 ==="
BIN="$WS/install/station_detector_cpp/lib/station_detector_cpp/ball_detector_node"
if [[ -f "$BIN" ]] && [[ ! -L "$BIN" ]]; then
  pass "ball_detector_node 为真实 ELF"
  ldd "$BIN" 2>/dev/null | grep opencv | head -2 || true
else
  fail "ball_detector_node 缺失或为符号链接 → bash scripts/deploy_ipc.sh"
fi

echo ""
echo "=== 节点（需 ./start_all.sh 运行中）==="
if pgrep -f ball_detector_node >/dev/null; then
  pass "ball_detector_node 在跑"
else
  warn "ball_detector_node 未运行"
fi
if pgrep -f realsense2_camera >/dev/null; then
  pass "realsense 在跑"
else
  warn "realsense 未运行（未插相机或未 start_all）"
fi

echo ""
echo "=== 话题 hz（3s 采样，无输出=未发布）==="
for t in \
  /camera/camera/color/image_raw \
  /debug_image \
  /volleyball_pose \
  /ball_intercept \
  /volleyball_ball_marker; do
  echo -n "  $t: "
  timeout 4 ros2 topic hz "$t" 2>&1 | head -1 || echo "(无)"
done

echo ""
echo "=== pipeline.conf ==="
grep -E '^(USE_REALSENSE|YOLO_DEVICE|YOLO_INPUT_SIZE)=' "$WS/config/pipeline.conf" 2>/dev/null || true
