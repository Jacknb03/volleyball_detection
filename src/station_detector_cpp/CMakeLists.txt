cmake_minimum_required(VERSION 3.8)
project(station_detector_cpp)

# C++ 标准
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# ---- 依赖 ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ---- 头文件目录 ----
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/VolleyballIntercept.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

# ---- 可执行文件 ----
add_executable(ball_detector_node
  src/ball_detector_node.cpp
  src/yolo_inference.cpp
  src/ball_tracker.cpp
  src/detection_filter.cpp
  src/trajectory_predictor.cpp
  src/ball_position_estimator.cpp
)

ament_target_dependencies(ball_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
  tf2
  tf2_ros
  tf2_geometry_msgs
)

target_link_libraries(ball_detector_node
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(ball_detector_node "${cpp_typesupport_target}")
add_dependencies(ball_detector_node ${PROJECT_NAME}__rosidl_generator_cpp)

# YOLO device selection is runtime via cv::dnn::getAvailableBackends().
set(_opencv_cuda_ok FALSE)
if(OpenCV_CUDA_VERSION)
  set(_opencv_cuda_ok TRUE)
  message(STATUS "OpenCV CUDA detected (${OpenCV_CUDA_VERSION})")
elseif(OpenCV_DIR AND EXISTS "${OpenCV_DIR}/cvconfig.h")
  file(READ "${OpenCV_DIR}/cvconfig.h" _cvconfig)
  if(_cvconfig MATCHES "#define HAVE_CUDA")
    set(_opencv_cuda_ok TRUE)
    message(STATUS "OpenCV CUDA detected (cvconfig.h HAVE_CUDA)")
  endif()
endif()
if(NOT _opencv_cuda_ok)
  message(STATUS "OpenCV DNN CUDA not available; yolo.device:=cuda/auto will use CPU")
endif()

# ---- 安装 ----
install(TARGETS ball_detector_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Launch files (ROS2 launch)
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY model/
  DESTINATION share/${PROJECT_NAME}/model
  FILES_MATCHING PATTERN "*.onnx"
)

install(PROGRAMS
  scripts/video_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY videos/
  DESTINATION share/${PROJECT_NAME}
)

# ---- ament 包 ----
ament_export_dependencies(rosidl_default_runtime)
ament_package()