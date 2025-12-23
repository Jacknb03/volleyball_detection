#include "station_params.h"
#include <rclcpp/rclcpp.hpp>

void StationParamsLoader::declareDetectionParams(std::shared_ptr<rclcpp::Node> node) {
    // 相机参数
    node->declare_parameter("use_auto_calibration", true);
    node->declare_parameter("manual_camera_matrix", std::vector<double>{600, 0, 320, 0, 600, 240, 0, 0, 1});
    node->declare_parameter("manual_distortion_coeffs", std::vector<double>{0, 0, 0, 0, 0});
    
    // 检测参数
    node->declare_parameter("min_contour_area", 400.0);
    node->declare_parameter("max_contour_area", 12000.0);
    node->declare_parameter("station_width", 0.262);
    node->declare_parameter("station_height", 0.262);
    node->declare_parameter("outer_size", 0.288);
    node->declare_parameter("inner_size", 0.240);
    
    // 调试参数
    node->declare_parameter("debug", false);
    node->declare_parameter("debug_file", false);
    node->declare_parameter("debug_window", false);
}

void StationParamsLoader::declareEstimationParams(std::shared_ptr<rclcpp::Node> node) {
    // 位姿估计参数
    node->declare_parameter("max_reprojection_error", 2.0);
    node->declare_parameter("corner_filter_gain", 0.1);
    node->declare_parameter("publish_tf", true);
    
    // 卡尔曼滤波参数
    node->declare_parameter("process_noise", 0.01);
    node->declare_parameter("measurement_noise", 0.05);
    node->declare_parameter("dt", 0.05);
}

DetectionParams StationParamsLoader::loadDetectionParams(std::shared_ptr<rclcpp::Node> node) {
    declareDetectionParams(node);
    
    DetectionParams params;
    params.use_auto_calibration = node->get_parameter("use_auto_calibration").as_bool();
    params.manual_camera_matrix = node->get_parameter("manual_camera_matrix").as_double_array();
    params.manual_distortion_coeffs = node->get_parameter("manual_distortion_coeffs").as_double_array();
    params.min_contour_area = node->get_parameter("min_contour_area").as_double();
    params.max_contour_area = node->get_parameter("max_contour_area").as_double();
    params.station_width = node->get_parameter("station_width").as_double();
    params.station_height = node->get_parameter("station_height").as_double();
    params.outer_size = node->get_parameter("outer_size").as_double();
    params.inner_size = node->get_parameter("inner_size").as_double();
    params.debug = node->get_parameter("debug").as_bool();
    params.debug_file = node->get_parameter("debug_file").as_bool();
    params.debug_window = node->get_parameter("debug_window").as_bool();
    
    return params;
}

EstimationParams StationParamsLoader::loadEstimationParams(std::shared_ptr<rclcpp::Node> node) {
    declareEstimationParams(node);
    
    EstimationParams params;
    params.max_reprojection_error = node->get_parameter("max_reprojection_error").as_double();
    params.corner_filter_gain = node->get_parameter("corner_filter_gain").as_double();
    params.publish_tf = node->get_parameter("publish_tf").as_bool();
    params.process_noise = node->get_parameter("process_noise").as_double();
    params.measurement_noise = node->get_parameter("measurement_noise").as_double();
    params.dt = node->get_parameter("dt").as_double();
    
    return params;
}

