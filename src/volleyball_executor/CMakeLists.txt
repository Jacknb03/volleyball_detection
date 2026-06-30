cmake_minimum_required(VERSION 3.8)
project(volleyball_executor)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(station_detector_cpp REQUIRED)
find_package(volleyball_msgs REQUIRED)

add_executable(intercept_bridge_node src/intercept_bridge_node.cpp)

ament_target_dependencies(intercept_bridge_node
  rclcpp
  station_detector_cpp
  volleyball_msgs
)

install(TARGETS intercept_bridge_node
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config docs
  DESTINATION share/${PROJECT_NAME})

ament_package()
