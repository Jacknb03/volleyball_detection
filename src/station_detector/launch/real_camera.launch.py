# real_camera.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    mindvision_dir = get_package_share_directory('mindvision_camera')
    station_dir = get_package_share_directory('station_detector')
    
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
    
    # 检测节点 - 确保正确重映射话题
    detector_node = Node(
        package='station_detector',
        executable='station_pose_estimator',  # 统一可执行文件名称
        name='station_detector',
        parameters=[
            {'debug': True},
            os.path.join(station_dir, 'config', 'station_params.yaml')
        ],
        output='screen',
        remappings=[
            ('image_raw', '/image_raw'),  # 订阅相机原始图像
            ('camera_info', '/camera_info')  # 订阅相机信息
        ]
    )
    
    # 调试图像查看器 - 订阅检测节点发布的调试图像
    debug_view_node = Node(
        package='image_tools',
        executable='showimage',
        name='debug_image_viewer',
        output='screen',
        remappings=[('image', '/station_detector/debug_image')]
    )
    
    # 二值图像查看器 - 订阅检测节点发布的二值图像
    binary_view_node = Node(
        package='image_tools',
        executable='showimage',
        name='binary_image_viewer',
        output='screen',
        remappings=[('image', '/station_detector/binary_image')]
    )
    
    # 原始图像查看器 - 订阅相机原始图像
    raw_view_node = Node(
        package='image_tools',
        executable='showimage',
        name='raw_image_viewer',
        output='screen',
        remappings=[('image', '/image_raw')]
    )
    
    return LaunchDescription([
        camera_node,
        detector_node,
        debug_view_node,
        binary_view_node,
        raw_view_node
    ])


