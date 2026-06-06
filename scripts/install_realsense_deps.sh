#!/bin/bash
# RealSense 依赖安装（Ubuntu 22.04 + ROS2 Humble）
#
# 用法: bash scripts/install_realsense_deps.sh
#
# 说明:
#   - librealsense2-dev / realsense-viewer 不在 Ubuntu 默认 apt 源里
#   - 本脚本先装 ROS 包（跑 ./start_all.sh 够用）
#   - 再可选添加 Intel 官方源（装 realsense-viewer 调试工具）
set -eo pipefail

echo "=== Step 1: ROS2 RealSense 驱动（ROS 官方源，推荐） ==="
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  ros-humble-realsense2-camera \
  ros-humble-diagnostic-updater

echo ""
echo "=== Step 2: Intel librealsense 官方源（realsense-viewer + 内核驱动） ==="
echo "需要 sudo，添加 Intel apt 仓库..."

sudo mkdir -p /etc/apt/keyrings
if [[ ! -f /etc/apt/keyrings/librealsense.gpg ]]; then
  curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
    | sudo gpg --dearmor -o /etc/apt/keyrings/librealsense.gpg
fi

if [[ ! -f /etc/apt/sources.list.d/librealsense.list ]]; then
  echo "deb [signed-by=/etc/apt/keyrings/librealsense.gpg] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
    | sudo tee /etc/apt/sources.list.d/librealsense.list
fi

sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  librealsense2-dkms \
  librealsense2-utils \
  librealsense2-dev \
  || echo "WARNING: Intel 包安装失败，ROS 驱动可能仍可用，见下方说明"

echo ""
echo "=== Step 3: udev 规则（免 root 访问相机） ==="
if [[ -f /lib/udev/rules.d/99-realsense-libusb.rules ]]; then
  echo "udev 规则已存在"
else
  echo "若相机权限有问题，插相机后运行: sudo udevadm control --reload-rules && sudo udevadm trigger"
fi

echo ""
echo "=== 验证 ==="
if command -v realsense-viewer >/dev/null 2>&1; then
  echo "realsense-viewer: OK — 插上 D455i 后运行: realsense-viewer"
else
  echo "realsense-viewer 未安装（Step 2 可能失败），可用 ROS 话题验证："
  echo "  ros2 launch realsense2_camera rs_launch.py"
fi

echo ""
echo "=== 下一步 ==="
echo "1. 编辑 config/pipeline.conf → USE_REALSENSE=true"
echo "2. ./start_all.sh"
echo ""
echo "若用 apt 版驱动、不编译工作区源码，可在 src/realsense2_camera 下 touch COLCON_IGNORE"
