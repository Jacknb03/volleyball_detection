#include "debug_manager.h"
#include <cv_bridge/cv_bridge.h>
#include <filesystem>
#include <chrono>

DebugManager::DebugManager(std::shared_ptr<rclcpp::Node> node)
    : node_(node), ros_debug_(false), file_debug_(false), window_debug_(false) {
    
    // 从参数获取调试设置
    node_->declare_parameter("debug", false);
    node_->declare_parameter("debug_file", false);
    node_->declare_parameter("debug_window", false);
    
    ros_debug_ = node_->get_parameter("debug").as_bool();
    file_debug_ = node_->get_parameter("debug_file").as_bool();
    window_debug_ = node_->get_parameter("debug_window").as_bool();
    
    debug_dir_ = "/tmp/station_debug";
    
    if (file_debug_) {
        ensureDebugDirectory();
    }
    
    RCLCPP_INFO(node_->get_logger(), "DebugManager initialized - ROS: %s, File: %s, Window: %s",
                ros_debug_ ? "ON" : "OFF", file_debug_ ? "ON" : "OFF", window_debug_ ? "ON" : "OFF");
}

void DebugManager::setMode(bool ros_debug, bool file_debug, bool window_debug) {
    ros_debug_ = ros_debug;
    file_debug_ = file_debug;
    window_debug_ = window_debug;
    
    if (file_debug_) {
        ensureDebugDirectory();
    }
}

void DebugManager::createPublishers() {
    if (ros_debug_) {
        debug_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("debug_image", 10);
        binary_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("binary_image", 10);
        contour_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("contour_image", 10);
        corner_pub_ = node_->create_publisher<sensor_msgs::msg::Image>("corner_image", 10);
        
        RCLCPP_INFO(node_->get_logger(), "Debug publishers created");
    }
}

void DebugManager::publishImage(const cv::Mat& image, const std::string& topic, 
                               const std_msgs::msg::Header& header) {
    if (!ros_debug_ || image.empty()) {
        return;
    }
    
    try {
        cv::Mat prepared_image = prepareImageForPublishing(image);
        
        auto cv_image = std::make_shared<cv_bridge::CvImage>();
        cv_image->header = header;
        cv_image->encoding = (image.channels() == 1) ? "mono8" : "bgr8";
        cv_image->image = prepared_image;
        
        auto msg = cv_image->toImageMsg();
        
        if (topic == "debug_image" && debug_pub_) {
            debug_pub_->publish(*msg);
        } else if (topic == "binary_image" && binary_pub_) {
            binary_pub_->publish(*msg);
        } else if (topic == "contour_image" && contour_pub_) {
            contour_pub_->publish(*msg);
        } else if (topic == "corner_image" && corner_pub_) {
            corner_pub_->publish(*msg);
        }
        
        RCLCPP_DEBUG(node_->get_logger(), "Published debug image to %s", topic.c_str());
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Error publishing debug image: %s", e.what());
    }
}

void DebugManager::saveImage(const cv::Mat& image, const std::string& filename) {
    if (!file_debug_ || image.empty()) {
        return;
    }
    
    try {
        std::string full_path = debug_dir_ + "/" + filename;
        cv::imwrite(full_path, image);
        RCLCPP_DEBUG(node_->get_logger(), "Saved debug image: %s", full_path.c_str());
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Error saving image: %s", e.what());
    }
}

void DebugManager::showImage(const cv::Mat& image, const std::string& window_name) {
    if (!window_debug_ || image.empty()) {
        return;
    }
    
    try {
        cv::imshow(window_name, image);
        cv::waitKey(1);
    } catch (const cv::Exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Error showing image: %s", e.what());
    }
}

void DebugManager::ensureDebugDirectory() {
    try {
        if (std::filesystem::create_directories(debug_dir_)) {
            RCLCPP_INFO(node_->get_logger(), "Created debug directory: %s", debug_dir_.c_str());
        }
    } catch (const std::filesystem::filesystem_error& e) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to create debug directory: %s", e.what());
    }
}

cv::Mat DebugManager::prepareImageForPublishing(const cv::Mat& image) {
    cv::Mat result = image.clone();
    
    // 确保图像连续存储
    if (!result.isContinuous()) {
        cv::Mat temp;
        result.copyTo(temp);
        result = temp;
    }
    
    // 如果图像太大，进行缩放
    if (result.cols > 1280 || result.rows > 1024) {
        double scale = std::min(1280.0 / result.cols, 1024.0 / result.rows);
        cv::Size new_size(
            static_cast<int>(result.cols * scale),
            static_cast<int>(result.rows * scale)
        );
        cv::resize(result, result, new_size);
    }
    
    return result;
}

