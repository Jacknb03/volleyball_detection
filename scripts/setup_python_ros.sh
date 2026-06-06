#!/bin/bash
# Fix ROS Python tooling (cv_bridge / rqt_image_view) on Ubuntu 22.04 + Humble.
# NumPy 2.x breaks prebuilt cv_bridge → rqt segfault.
#
# Run once:  bash scripts/setup_python_ros.sh
set -euo pipefail

echo "=== Pin NumPy 1.x for ROS cv_bridge ==="
pip install "numpy>=1.21,<2"

echo "=== Verify cv_bridge ==="
# Prefer system cv_bridge over a stale cv_bridge_ws overlay
export PYTHONPATH="/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages"
python3 - <<'PY'
from cv_bridge import CvBridge
import numpy
print("numpy", numpy.__version__)
CvBridge()
print("cv_bridge OK")
PY

echo ""
echo "Done. In new terminals, avoid prepending cv_bridge_ws to PYTHONPATH."
echo "If ~/.bashrc sets PYTHONPATH to cv_bridge_ws, comment it out for this project."
echo ""
echo "Test rqt (after ./start_all.sh is running):"
echo "  ros2 run rqt_image_view rqt_image_view"
echo "  Select topic: /debug_image"
