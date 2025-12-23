#pragma once
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <string>
#include <memory>

class DebugManager {
public:
    explicit DebugManager(std::shared_ptr<rclcpp::Node> node);
    
    // 设置调试模式
    void setMode(bool ros_debug, bool file_debug, bool window_debug);
    
    // 发布调试图像到ROS话题
    void publishImage(const cv::Mat& image, const std::string& topic, 
                     const std_msgs::msg::Header& header);
    
    // 保存图像到文件
    void saveImage(const cv::Mat& image, const std::string& filename);
    
    // 显示图像窗口
    void showImage(const cv::Mat& image, const std::string& window_name);
    
    // 创建发布者
    void createPublishers();
    
    // 检查是否启用调试
    bool isDebugEnabled() const { return ros_debug_ || file_debug_ || window_debug_; }
    bool isRosDebugEnabled() const { return ros_debug_; }
    bool isFileDebugEnabled() const { return file_debug_; }
    bool isWindowDebugEnabled() const { return window_debug_; }

private:
    std::shared_ptr<rclcpp::Node> node_;
    bool ros_debug_;
    bool file_debug_;
    bool window_debug_;
    
    // ROS发布者
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr binary_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr contour_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr corner_pub_;
    
    std::string debug_dir_;
    
    // 辅助函数
    void ensureDebugDirectory();
    cv::Mat prepareImageForPublishing(const cv::Mat& image);
};
