import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    pkg_dir = get_package_share_directory('station_detector')
    video_path = os.path.join(pkg_dir, 'videos', 'test.mp4')
    
    return LaunchDescription([
        # 视频发布节点
        Node(
            package='station_detector',
            executable='video_publisher',
            name='video_publisher',
            parameters=[{'video_path': video_path,
                        'frame_rate': 30.0,
                        'loop': True,
                        'rotate_vertical': True  }]#翻转视频
        ),
        
        # 虚拟相机信息发布节点
        Node(
            package='station_detector',
            executable='dummy_camera_info_publisher',
            name='dummy_camera_info_publisher',
            parameters=[{
                'camera_matrix': [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0],
                'distortion_coeffs': [0.0, 0.0, 0.0, 0.0, 0.0]
            }]
        ),
        
        # 站点检测节点
        Node(
            package='station_detector',
            executable='station_pose_estimator',  # 统一可执行文件名称
            name='station_detector',
            parameters=[{'debug': True}]
        ),
        
        # 图像查看器 - 使用稳定的image_tools
        Node(
            package='image_tools',
            executable='showimage',
            name='image_viewer',
            remappings=[('image', '/station_detector/debug_image')]
        )
    ])