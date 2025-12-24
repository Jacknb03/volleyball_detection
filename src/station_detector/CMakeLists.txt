cmake_minimum_required(VERSION 3.8)
project(station_detector)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# set(OpenCV_DIR "/home/song/opencv_build/opencv/build")

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d highgui imgcodecs)

# 添加包含目录
include_directories(include)

# ✅ 统一可执行名：station_pose_estimator
add_executable(station_pose_estimator
  src/main.cpp
  src/station_pose_estimator.cpp
  src/fusion_station_detector.cpp
  src/kalman_filter.cpp
  src/debug_manager.cpp
  src/station_params.cpp
)

# 先链接 OpenCV 和其他库，再链接 ROS2 依赖
target_link_libraries(station_pose_estimator
  ${OpenCV_LIBS}
  yaml-cpp
)

# 然后添加 ROS2 依赖
ament_target_dependencies(station_pose_estimator
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  cv_bridge
  tf2 
  tf2_ros 
  tf2_geometry_msgs
  image_transport
  camera_info_manager
  ament_index_cpp
)

# 安装 C++ 可执行到 lib/${PROJECT_NAME}
install(TARGETS station_pose_estimator
  DESTINATION lib/${PROJECT_NAME})

# 安装脚本（作为可执行，便于 ros2 run）
install(PROGRAMS 
  scripts/static_image_pub.py
  scripts/video_publisher.py
  scripts/yolo_volleyball_node.py
  scripts/yolo_detector.py
  scripts/kalman_tracker.py
  scripts/detection_validator.py
  DESTINATION lib/${PROJECT_NAME})

# 安装资源（launch/config/test_images），换设备也能被 FindPackageShare 找到
install(DIRECTORY launch config test_images
  DESTINATION share/${PROJECT_NAME}
  PATTERN ".gitkeep" EXCLUDE)

ament_package()