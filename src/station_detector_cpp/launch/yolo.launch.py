#!/usr/bin/env python3
"""
Unified volleyball detection launch.

pipeline_mode:
  video      — 本地视频 + bbox 估深（默认，开发调试）
  realsense  — RealSense D455i RGB-D + 对齐深度采样

Usage:
  ros2 launch station_detector_cpp yolo.launch.py
  ros2 launch station_detector_cpp yolo.launch.py pipeline_mode:=realsense
  ros2 launch station_detector_cpp yolo.launch.py pipeline_mode:=video video_path:=/path/to.mp4
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _setup(context, *args, **kwargs):
    cpp_share = FindPackageShare("station_detector_cpp").find("station_detector_cpp")
    mode = LaunchConfiguration("pipeline_mode").perform(context).strip().lower()
    if mode not in ("video", "realsense"):
        raise RuntimeError(f"pipeline_mode must be 'video' or 'realsense', got: {mode!r}")

    use_static_tf = LaunchConfiguration("use_static_camera_tf").perform(context).strip().lower() in (
        "true",
        "1",
        "yes",
    )

    default_model = os.path.join(cpp_share, "model", "best.onnx")
    override = LaunchConfiguration("model_path").perform(context).strip()
    model_path = override if override else default_model
    yolo_device = LaunchConfiguration("yolo_device").perform(context)

    params_override = LaunchConfiguration("params_file").perform(context)
    if params_override:
        params_file = params_override
    elif mode == "realsense":
        params_file = os.path.join(cpp_share, "config", "ball_detector_params_realsense.yaml")
    else:
        params_file = os.path.join(cpp_share, "config", "ball_detector_params_video.yaml")

    nodes = []

    static_tf = None
    if use_static_tf:
        if mode == "video":
            static_tf = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="tf_static_camera",
                output="screen",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "1",
                    "--yaw", "0", "--pitch", "0", "--roll", "0",
                    "--frame-id", "base_link",
                    "--child-frame-id", "camera_optical_frame",
                ],
            )
        else:
            # 开发占位：base_link→camera_link。实机由 URDF 发布 base_link→camera_link。
            static_tf = Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="tf_static_camera",
                output="screen",
                arguments=[
                    "--x", "0", "--y", "0", "--z", "1.0",
                    "--yaw", "0", "--pitch", "0", "--roll", "0",
                    "--frame-id", "base_link",
                    "--child-frame-id", "camera_link",
                ],
            )

    if mode == "video":
        video_pub = Node(
            package="station_detector_cpp",
            executable="video_publisher.py",
            name="video_publisher",
            output="screen",
            parameters=[
                {
                    "video_path": LaunchConfiguration("video_path"),
                    "loop": LaunchConfiguration("loop"),
                    "frame_rate": ParameterValue(
                        LaunchConfiguration("frame_rate"), value_type=float
                    ),
                }
            ],
        )
        ball_detector = Node(
            package="station_detector_cpp",
            executable="ball_detector_node",
            name="ball_detector_node",
            output="screen",
            parameters=[
                params_file,
                {
                    "yolo.model_path": ParameterValue(model_path, value_type=str),
                    "yolo.device": ParameterValue(yolo_device, value_type=str),
                },
            ],
        )
        nodes = []
        if static_tf is not None:
            nodes.append(static_tf)
        nodes.extend([video_pub, ball_detector])
    else:
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
            parameters=[
                params_file,
                {
                    "yolo.model_path": ParameterValue(model_path, value_type=str),
                    "yolo.device": ParameterValue(yolo_device, value_type=str),
                },
            ],
        )
        nodes = [realsense, ball_detector]
        if static_tf is not None:
            nodes.insert(0, static_tf)

    if not use_static_tf:
        nodes.insert(
            0,
            LogInfo(
                msg=[
                    "use_static_camera_tf=false: 请由 URDF 发布 ",
                    "base_link→camera_link→…→camera_color_optical_frame",
                ]
            ),
        )

    return nodes


def generate_launch_description():
    cpp_share = FindPackageShare("station_detector_cpp").find("station_detector_cpp")
    default_video = os.path.join(cpp_share, "videos", "test.mp4")
    default_params = os.path.join(cpp_share, "config", "ball_detector_params_video.yaml")
    default_model = os.path.join(cpp_share, "model", "best.onnx")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pipeline_mode",
                default_value="video",
                description="video | realsense",
            ),
            DeclareLaunchArgument("params_file", default_value="", description="Override params yaml (optional)"),
            DeclareLaunchArgument("video_path", default_value=default_video),
            DeclareLaunchArgument("loop", default_value="true"),
            DeclareLaunchArgument("frame_rate", default_value="15.0"),
            DeclareLaunchArgument("model_path", default_value=default_model),
            DeclareLaunchArgument("yolo_device", default_value="auto"),
            DeclareLaunchArgument(
                "use_static_camera_tf",
                default_value="true",
                description="true=开发占位 static TF；false=接机器人底盘/URDF 的 TF 树",
            ),
            LogInfo(msg=["pipeline_mode=", LaunchConfiguration("pipeline_mode")]),
            OpaqueFunction(function=_setup),
        ]
    )
