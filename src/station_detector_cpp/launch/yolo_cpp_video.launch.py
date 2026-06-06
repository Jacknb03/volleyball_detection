#!/usr/bin/env python3
"""
One-command entry for the C++ YOLO (ONNX) pipeline:
- static TF (odom -> camera_optical_frame)
- video publisher (station_detector_cpp/video_publisher.py)
- C++ detector node (station_detector_cpp/ball_detector_node)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    cpp_share = FindPackageShare("station_detector_cpp").find("station_detector_cpp")

    default_params = os.path.join(cpp_share, "config", "ball_detector_params_video.yaml")
    default_video = os.path.join(cpp_share, "videos", "test.mp4")
    default_model = os.path.join(cpp_share, "model", "best.onnx")

    params_file_arg = DeclareLaunchArgument("params_file", default_value=default_params)
    video_path_arg = DeclareLaunchArgument("video_path", default_value=default_video)
    loop_arg = DeclareLaunchArgument("loop", default_value="true")
    frame_rate_arg = DeclareLaunchArgument("frame_rate", default_value="15.0")
    model_path_arg = DeclareLaunchArgument("model_path", default_value=default_model)
    yolo_device_arg = DeclareLaunchArgument("yolo_device", default_value="auto")

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tf_static_camera",
        output="screen",
        arguments=[
            "--x",
            "0",
            "--y",
            "0",
            "--z",
            "1",
            "--yaw",
            "0",
            "--pitch",
            "0",
            "--roll",
            "0",
            "--frame-id",
            "odom",
            "--child-frame-id",
            "camera_optical_frame",
        ],
    )

    video_pub = Node(
        package="station_detector_cpp",
        executable="video_publisher.py",
        name="video_publisher",
        output="screen",
        parameters=[
            {
                "video_path": LaunchConfiguration("video_path"),
                "loop": LaunchConfiguration("loop"),
                "frame_rate": LaunchConfiguration("frame_rate"),
            }
        ],
    )

    ball_detector = Node(
        package="station_detector_cpp",
        executable="ball_detector_node",
        name="ball_detector_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "yolo.model_path": ParameterValue(
                    LaunchConfiguration("model_path"), value_type=str
                ),
                "yolo.device": ParameterValue(
                    LaunchConfiguration("yolo_device"), value_type=str
                ),
                "position.mode": "bbox",
            },
        ],
    )

    return LaunchDescription(
        [
            params_file_arg,
            video_path_arg,
            loop_arg,
            frame_rate_arg,
            model_path_arg,
            yolo_device_arg,
            LogInfo(msg=["Starting C++ YOLO pipeline (video mode)..."]),
            static_tf,
            video_pub,
            ball_detector,
        ]
    )

