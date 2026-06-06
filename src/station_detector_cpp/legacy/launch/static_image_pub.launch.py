# portable static_image_pub.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch_ros.actions import Node
from launch.conditions import IfCondition

def generate_launch_description():
    # 包内默认资源
    default_image = PathJoinSubstitution(
        [FindPackageShare('station_detector'), 'test_images', 'image.png'])
    default_cam_yaml = PathJoinSubstitution(
        [FindPackageShare('mindvision_camera'), 'config', 'camera_info.yaml'])

    # 可覆盖参数
    image_arg = DeclareLaunchArgument('image', default_value=default_image)
    cam_arg   = DeclareLaunchArgument('camera_yaml', default_value=default_cam_yaml)
    frame_arg = DeclareLaunchArgument('frame_id', default_value='camera_optical_frame')
    hz_arg    = DeclareLaunchArgument('hz', default_value='5.0')
    rqt_arg   = DeclareLaunchArgument('use_rqt', default_value='true')

    static_img = Node(
        package='station_detector',
        executable='static_image_pub.py',   # ✔ 脚本作为可执行
        name='static_image_pub',
        output='screen',
        parameters=[{
            'image':       LaunchConfiguration('image'),
            'camera_yaml': LaunchConfiguration('camera_yaml'),
            'frame_id':    LaunchConfiguration('frame_id'),
            'hz':          LaunchConfiguration('hz'),
        }]
    )

    detector = Node(
        package='station_detector',
        executable='station_pose_estimator',  # ✔ 与 CMake 可执行一致
        name='station_pose_estimator',
        output='screen',
        parameters=[{
            'debug': True,
            # 放宽静态图的门槛，先跑通
            'min_contour_area': 50.0,
            'max_contour_area': 400000.0,
            'max_reprojection_error': 5.0,
        }]
    )

    # 使用更稳定的image_view替代rqt_image_view
    debug_viewer = Node(
        condition=IfCondition(LaunchConfiguration('use_rqt')),
        package='image_tools',
        executable='showimage',
        name='debug_image_viewer',
        output='screen',
        remappings=[('image', '/station_pose_estimator/debug_image')]
    )
    
    binary_viewer = Node(
        condition=IfCondition(LaunchConfiguration('use_rqt')),
        package='image_tools',
        executable='showimage',
        name='binary_image_viewer',
        output='screen',
        remappings=[('image', '/station_pose_estimator/binary_image')]
    )

    return LaunchDescription([
        image_arg, cam_arg, frame_arg, hz_arg, rqt_arg,
        static_img,
        detector,
        debug_viewer,
        binary_viewer
    ])
