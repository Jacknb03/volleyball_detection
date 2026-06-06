#pragma once
#include <rclcpp/rclcpp.hpp>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <vector>
#include <memory>

// 颜色检测参数结构体 - 用于排球检测
struct ColorDetectionParams {
    // 白色/浅色检测参数 (HSV空间)
    cv::Scalar white_lower;  // 白色HSV下限
    cv::Scalar white_upper;  // 白色HSV上限
    
    // 形态学处理参数
    int morphology_kernel_size;
    bool use_adaptive_kernel;
    
    // 默认构造函数
    ColorDetectionParams() {
        // 设置默认值 - 白色检测范围
        // HSV中白色: H任意(0-180), S低(0-30), V高(200-255)
        white_lower = cv::Scalar(0, 0, 200);
        white_upper = cv::Scalar(180, 30, 255);
        morphology_kernel_size = 5;
        use_adaptive_kernel = true;
    }
};

class FusionStationDetector {
public:
    explicit FusionStationDetector();

    void setCameraParameters(const cv::Mat& camera_matrix, const cv::Mat& distortion_coeffs);

    // 统一为 shared_ptr，避免生命周期问题
    void init(std::shared_ptr<rclcpp::Node> node);

    // loadCalibration函数已删除 - 相机参数现在由StationPoseEstimator统一管理

    // 颜色分割（白色/浅色）+ 形态学
    cv::Mat preprocess(const cv::Mat& input_image);

    // 从二值图中检测排球圆形中心 - 返回圆心和半径
    bool detectVolleyball(const cv::Mat& bin_image, const cv::Mat& original_image, 
                         cv::Point2f& center, float& radius);
    
    // 验证检测到的圆形质量
    bool validateDetectedCircle(const cv::Point2f& center, float radius, const cv::Size& image_size);
    
    // 新增：加载颜色检测参数
    void loadColorDetectionParams();
    
    // 新增：调试图像输出函数
    void saveDebugImage(const std::string& filename, const cv::Mat& image);
    void drawContours(const cv::Mat& original, const std::vector<std::vector<cv::Point>>& contours, const std::string& filename);
    void drawCorners(const cv::Mat& original, const std::vector<cv::Point2f>& corners, const std::string& filename);
    void drawFittedRectangle(const cv::Mat& original, const std::vector<cv::Point2f>& corners, const std::string& filename);
    void drawAllDebugInfo(const cv::Mat& original, const std::vector<std::vector<cv::Point>>& contours, 
                         const std::vector<cv::Point2f>& corners, const std::string& filename);

    const cv::Mat& getCameraMatrix() const { return camera_matrix_; }
    const cv::Mat& getDistortionCoeffs() const { return distortion_coeffs_; }
    double getVolleyballRadius() const;  // 获取排球真实半径(米)

    bool isCalibrated() const { return calibration_initialized_; }
    bool isDebugMode() const { return debug_mode_; }

    // 调试：返回最近一次检测到的轮廓
    const std::vector<std::vector<cv::Point>>& getDetectedContours() const { return detected_contours_; }

private:
    // 从轮廓中检测圆形
    bool detectCircleFromContours(const std::vector<std::vector<cv::Point>>& contours, 
                                  cv::Point2f& center, float& radius);
    
    // 过滤轮廓 - 用于排球检测
    std::vector<std::vector<cv::Point>> filterContoursForVolleyball(
        const std::vector<std::vector<cv::Point>>& contours);

    std::shared_ptr<rclcpp::Node> node_{nullptr};
    bool calibration_initialized_{false};
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

    cv::Mat camera_matrix_;
    cv::Mat distortion_coeffs_;

    std::vector<std::vector<cv::Point>> detected_contours_;
    
    bool debug_mode_;
    std::string debug_dir_;  // 调试图片保存目录
    
    // 颜色检测参数
    ColorDetectionParams color_params_;
    
    // 最近检测到的排球信息
    cv::Point2f last_detected_center_;
    float last_detected_radius_;
    bool has_last_detection_;
};