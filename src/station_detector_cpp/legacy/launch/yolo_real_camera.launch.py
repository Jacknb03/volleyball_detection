#!/usr/bin/env python3
"""
YOLO真实摄像头启动文件
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    mindvision_dir = FindPackageShare('mindvision_camera').find('mindvision_camera')
    station_dir = FindPackageShare('station_detector').find('station_detector')
    
    yolo_params_file = os.path.join(station_dir, 'config', 'yolo_params.yaml')
    
    # 相机节点
    camera_node = Node(
        package='mindvision_camera',
        executable='mindvision_camera_node',
        name='mv_camera',
        parameters=[
            {'frame_rate': 30.0},
            os.path.join(mindvision_dir, 'config', 'camera_params.yaml')
        ],
        output='screen'
    )
    
    # YOLO检测节点
    yolo_node = Node(
        package='station_detector',
        executable='yolo_volleyball_node.py',
        name='yolo_volleyball_detector',
        output='screen',
        parameters=[yolo_params_file],
        remappings=[
            ('/image_raw', '/image_raw'),
            ('/camera_info', '/camera_info'),
        ]
    )
    
    # 调试图像查看器
    debug_viewer = Node(
        package='image_tools',
        executable='showimage',
        name='debug_image_viewer',
        output='screen',
        remappings=[('image', '/debug_image')]
    )
    
    # 原始图像查看器
    raw_viewer = Node(
        package='image_tools',
        executable='showimage',
        name='raw_image_viewer',
        output='screen',
        remappings=[('image', '/image_raw')]
    )
    
    return LaunchDescription([
        LogInfo(msg=['启动YOLO真实摄像头检测...']),
        camera_node,
        yolo_node,
        debug_viewer,
        raw_viewer,
    ])

