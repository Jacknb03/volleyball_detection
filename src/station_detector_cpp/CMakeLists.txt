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

# ---- 头文件目录 ----
include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
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
)

target_link_libraries(ball_detector_node
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

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
  DESTINATION share/${PROJECT_NAME}
)

# ---- ament 包 ----
ament_package()