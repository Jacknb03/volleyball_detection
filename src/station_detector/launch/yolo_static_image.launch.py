#!/usr/bin/env python3
"""
YOLO静态图片测试启动文件
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition
import os


def generate_launch_description():
    pkg_share = FindPackageShare('station_detector').find('station_detector')
    
    # 默认图片路径
    default_image = PathJoinSubstitution(
        [FindPackageShare('station_detector'), 'test_images', 'image1.png']
    )
    default_cam_yaml = PathJoinSubstitution(
        [FindPackageShare('mindvision_camera'), 'config', 'camera_info.yaml']
    )
    yolo_params_file = os.path.join(pkg_share, 'config', 'yolo_params.yaml')
    
    # 启动参数
    image_arg = DeclareLaunchArgument('image', default_value=default_image)
    cam_yaml_arg = DeclareLaunchArgument('camera_yaml', default_value=default_cam_yaml)
    frame_arg = DeclareLaunchArgument('frame_id', default_value='camera_optical_frame')
    hz_arg = DeclareLaunchArgument('hz', default_value='5.0')
    use_viewer_arg = DeclareLaunchArgument('use_viewer', default_value='true')
    
    # 静态图片发布节点
    static_img_node = Node(
        package='station_detector',
        executable='static_image_pub.py',
        name='static_image_pub',
        output='screen',
        parameters=[{
            'image': LaunchConfiguration('image'),
            'camera_yaml': LaunchConfiguration('camera_yaml'),
            'frame_id': LaunchConfiguration('frame_id'),
            'hz': LaunchConfiguration('hz'),
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
        condition=IfCondition(LaunchConfiguration('use_viewer')),
        package='image_tools',
        executable='showimage',
        name='debug_image_viewer',
        output='screen',
        remappings=[('image', '/debug_image')]
    )
    
    return LaunchDescription([
        image_arg,
        cam_yaml_arg,
        frame_arg,
        hz_arg,
        use_viewer_arg,
        LogInfo(msg=['启动YOLO静态图片测试...']),
        static_img_node,
        yolo_node,
        debug_viewer,
    ])

