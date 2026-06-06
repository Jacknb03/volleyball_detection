#!/usr/bin/env python3
"""
YOLO视频测试启动文件
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os


def generate_launch_description():
    pkg_share = FindPackageShare('station_detector').find('station_detector')
    
    # 默认视频路径
    default_video = os.path.join(pkg_share, 'videos', 'test.mp4')
    yolo_params_file = os.path.join(pkg_share, 'config', 'yolo_params.yaml')
    
    # 启动参数
    video_arg = DeclareLaunchArgument('video_path', default_value=default_video)
    frame_rate_arg = DeclareLaunchArgument('frame_rate', default_value='30.0')
    loop_arg = DeclareLaunchArgument('loop', default_value='true')
    
    # 视频发布节点
    video_pub_node = Node(
        package='station_detector',
        executable='video_publisher.py',
        name='video_publisher',
        output='screen',
        parameters=[{
            'video_path': LaunchConfiguration('video_path'),
            'frame_rate': LaunchConfiguration('frame_rate'),
            'loop': LaunchConfiguration('loop'),
            'rotate_vertical': False,
        }]
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
    
    return LaunchDescription([
        video_arg,
        frame_rate_arg,
        loop_arg,
        LogInfo(msg=['启动YOLO视频测试...']),
        video_pub_node,
        yolo_node,
        debug_viewer,
    ])

