#!/bin/bash
# One-time deps for RealSense (apt path — recommended).
# Run: bash scripts/install_realsense_deps.sh
set -eo pipefail

echo "=== Installing RealSense system + ROS2 packages (sudo) ==="
sudo apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  librealsense2-dev \
  ros-humble-realsense2-camera \
  ros-humble-diagnostic-updater

echo ""
echo "Done. Start with:"
echo "  cd ~/volleyball_detection && ./start_realsense.sh"
