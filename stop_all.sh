#!/bin/bash
echo "正在关闭所有排球检测相关节点..."

# 杀掉主要的进程名
pkill -f ball_detector_node
pkill -f video_publisher
pkill -f realsense2_camera_node
pkill -f static_transform_publisher
pkill -f rviz2

echo "已全部关闭。"