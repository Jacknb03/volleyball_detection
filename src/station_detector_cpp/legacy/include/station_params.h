#pragma once
#include <vector>
#include <memory>
#include <rclcpp/rclcpp.hpp>

// 统一的参数结构体
struct DetectionParams {
    // 相机参数
    bool use_auto_calibration = true;
    std::vector<double> manual_camera_matrix = {600, 0, 320, 0, 600, 240, 0, 0, 1};
    std::vector<double> manual_distortion_coeffs = {0, 0, 0, 0, 0};
    
    // 检测参数
    double min_contour_area = 400.0;
    double max_contour_area = 12000.0;
    double station_width = 0.262;
    double station_height = 0.262;
    double outer_size = 0.288;  // 外框尺寸288mm
    double inner_size = 0.240;  // 内框尺寸240mm
    
    // 调试参数
    bool debug = false;
    bool debug_file = false;
    bool debug_window = false;
};

struct EstimationParams {
    // 位姿估计参数
    double max_reprojection_error = 2.0;
    double corner_filter_gain = 0.1;
    bool publish_tf = true;
    
    // 卡尔曼滤波参数
    double process_noise = 0.01;
    double measurement_noise = 0.05;
    double dt = 0.05;  // 时间步长
};

// 参数加载器
class StationParamsLoader {
public:
    static DetectionParams loadDetectionParams(std::shared_ptr<rclcpp::Node> node);
    static EstimationParams loadEstimationParams(std::shared_ptr<rclcpp::Node> node);
    
private:
    static void declareDetectionParams(std::shared_ptr<rclcpp::Node> node);
    static void declareEstimationParams(std::shared_ptr<rclcpp::Node> node);
};
