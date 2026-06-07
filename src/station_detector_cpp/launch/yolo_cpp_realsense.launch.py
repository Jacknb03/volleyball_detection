#!/usr/bin/env python3
"""
C++ YOLO pipeline for Intel RealSense D455i (RGB-D).

Prerequisites (on Ubuntu 22.04 + Humble):
  sudo apt install ros-humble-realsense2-camera

Usage:
  ros2 launch station_detector_cpp yolo_cpp_realsense.launch.py \\
    model_path:=/abs/path/to/best.onnx

Notes:
  - YOLO still runs on the color stream.
  - 3D position uses aligned depth at the detection center (position.mode=depth).
  - Static TF below is a placeholder; replace with robot URDF / calibrated extrinsics.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cpp_share = FindPackageShare("station_detector_cpp").find("station_detector_cpp")
    default_params = os.path.join(cpp_share, "config", "ball_detector_params_realsense.yaml")
    default_model = os.path.join(cpp_share, "model", "best.onnx")

    params_file_arg = DeclareLaunchArgument("params_file", default_value=default_params)
    model_path_arg = DeclareLaunchArgument("model_path", default_value=default_model)
    yolo_device_arg = DeclareLaunchArgument("yolo_device", default_value="auto")

    # Placeholder: camera 1 m above odom origin, facing forward.
    # Replace with your robot's camera extrinsics when integrated.
    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tf_static_camera",
        output="screen",
        arguments=[
            "--x", "0", "--y", "0", "--z", "1.0",
            "--yaw", "0", "--pitch", "0", "--roll", "0",
            "--frame-id", "odom",
            "--child-frame-id", "camera_color_optical_frame",
        ],
    )

    realsense = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        namespace="camera",
        name="camera",
        output="screen",
        parameters=[
            {
                "camera_name": "camera",
                "camera_namespace": "camera",
                "enable_depth": True,
                "enable_color": True,
                "enable_infra1": False,
                "enable_infra2": False,
                "enable_gyro": False,
                "enable_accel": False,
                "pointcloud.enable": False,
                "align_depth.enable": True,
                "depth_module.depth_profile": "640,480,30",
                "rgb_camera.color_profile": "640,480,30",
            }
        ],
    )

    ball_detector = Node(
        package="station_detector_cpp",
        executable="ball_detector_node",
        name="ball_detector_node",
        output="screen",
        parameters=[LaunchConfiguration("params_file")],
        arguments=[
            "--ros-args",
            "-p", ["yolo.model_path:=", LaunchConfiguration("model_path")],
            "-p", ["yolo.device:=", LaunchConfiguration("yolo_device")],
        ],
    )

    return LaunchDescription(
        [
            params_file_arg,
            model_path_arg,
            yolo_device_arg,
            LogInfo(msg=["Starting C++ YOLO pipeline (RealSense D455i RGB-D mode)..."]),
            static_tf,
            realsense,
            ball_detector,
        ]
    )
