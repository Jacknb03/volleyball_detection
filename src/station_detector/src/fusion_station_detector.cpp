#include "fusion_station_detector.h"
#include <vector>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>  
#include <ament_index_cpp/get_package_share_directory.hpp>  
#include <yaml-cpp/yaml.h>
#include <sensor_msgs/msg/camera_info.hpp>
#include <future>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <unistd.h>  // for getcwd

FusionStationDetector::FusionStationDetector() 
    : node_(nullptr), calibration_initialized_(false), debug_mode_(false),
      has_last_detection_(false), last_detected_radius_(0.0f) {
}

void FusionStationDetector::init(std::shared_ptr<rclcpp::Node> node) {
    if (!node) {
        throw std::invalid_argument("Node pointer cannot be null");
    }

    // 确保只初始化一次
    if (node_) {
        RCLCPP_WARN(node->get_logger(), "Detector already initialized. Skipping re-initialization.");
        return;
    }
    
    node_ = node;
    
    // 只声明检测相关的参数，相机参数由外部设置
    node_->declare_parameter("station_width", 0.262);
    node_->declare_parameter("station_height", 0.262);
    node_->declare_parameter("min_contour_area", 400.0);
    node_->declare_parameter("max_contour_area", 12000.0);
    node_->declare_parameter("outer_size", 0.288);  // 外框尺寸288mm
    node_->declare_parameter("inner_size", 0.240);  // 内框尺寸240mm
    node_->declare_parameter("debug", false);  // 新增调试模式参数
    
    // 声明颜色检测参数 - 排球白色检测
    node_->declare_parameter("color_detection.white_lower_h", 0);
    node_->declare_parameter("color_detection.white_lower_s", 0);
    node_->declare_parameter("color_detection.white_lower_v", 200);
    node_->declare_parameter("color_detection.white_upper_h", 180);
    node_->declare_parameter("color_detection.white_upper_s", 30);
    node_->declare_parameter("color_detection.white_upper_v", 255);
    node_->declare_parameter("color_detection.morphology_kernel_size", 5);
    node_->declare_parameter("color_detection.use_adaptive_kernel", true);
    
    // 排球检测参数
    node_->declare_parameter("volleyball.min_radius", 5.0);  // 最小半径(像素)
    node_->declare_parameter("volleyball.max_radius", 200.0);  // 最大半径(像素)
    node_->declare_parameter("volleyball.min_circularity", 0.7);  // 最小圆形度
    node_->declare_parameter("volleyball.real_radius", 0.105);  // 排球真实半径(米) - 标准排球直径约21cm
    
    // 获取调试模式参数
    debug_mode_ = node_->get_parameter("debug").as_bool();
    if (debug_mode_) {
        // 获取当前工作目录，在其下创建debug_images文件夹
        char* cwd = getcwd(nullptr, 0);
        if (cwd) {
            debug_dir_ = std::string(cwd) + "/debug_images";
            free(cwd);
        } else {
            // 如果获取工作目录失败，使用相对路径
            debug_dir_ = "debug_images";
        }
        
        RCLCPP_INFO(node_->get_logger(), "Debug mode enabled - intermediate images will be saved to %s", debug_dir_.c_str());
        
        // 提前创建调试目录
        if (std::filesystem::create_directories(debug_dir_)) {
            RCLCPP_INFO(node_->get_logger(), "Created debug directory: %s", debug_dir_.c_str());
        }
    }
    
    // 加载颜色检测参数
    loadColorDetectionParams();
}

void FusionStationDetector::setCameraParameters(const cv::Mat& camera_matrix, const cv::Mat& distortion_coeffs) {
    camera_matrix_ = camera_matrix.clone();
    distortion_coeffs_ = distortion_coeffs.clone();
    calibration_initialized_ = true;
    if (node_) {
        RCLCPP_INFO(node_->get_logger(), "Camera parameters set from external source");
    }
}

// loadCalibration函数已删除 - 相机参数现在由StationPoseEstimator统一管理

cv::Mat FusionStationDetector::preprocess(const cv::Mat& input_image) {
    if (camera_matrix_.empty() || distortion_coeffs_.empty()) {
        throw std::runtime_error("Camera parameters not loaded");
    }

    // === Debug: 保存原始图像 ===
    if (debug_mode_) {
        std::filesystem::create_directories(debug_dir_);
        cv::imwrite(debug_dir_ + "/raw.png", input_image);
        RCLCPP_INFO(node_->get_logger(), "Saved raw image to: %s/raw.png", debug_dir_.c_str());
    }

    cv::Mat undistorted;
    cv::undistort(input_image, undistorted, camera_matrix_, distortion_coeffs_);

    cv::Mat hsv, mask_white;
    cv::cvtColor(undistorted, hsv, cv::COLOR_BGR2HSV);
    
    // 使用参数化的白色检测
    cv::inRange(hsv, color_params_.white_lower, color_params_.white_upper, mask_white);
    cv::Mat combined = mask_white;

    // 形态学处理 - 使用参数化设置
    int k;
    if (color_params_.use_adaptive_kernel) {
        // 自适应卷积核：分辨率越大，核略大
        k = std::max(3, int(std::min(combined.cols, combined.rows) / 320.0 * color_params_.morphology_kernel_size));
    } else {
        // 使用固定核大小
        k = color_params_.morphology_kernel_size;
    }
    
    if (k < 3) k = 3;  // 最小核大小
    if ((k & 1) == 0) k += 1; // 确保为奇数
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
    cv::morphologyEx(combined, combined, cv::MORPH_OPEN,  kernel);
    cv::morphologyEx(combined, combined, cv::MORPH_CLOSE, kernel);

    // === Debug: 保存掩膜图像 ===
    if (debug_mode_) {
        cv::imwrite(debug_dir_ + "/mask.png", combined);
        RCLCPP_INFO(node_->get_logger(), "Saved mask image to: %s/mask.png", debug_dir_.c_str());
    }

    return combined;
}

bool FusionStationDetector::detectVolleyball(const cv::Mat& bin_image, const cv::Mat& original_image, 
                                             cv::Point2f& center, float& radius) {
    detected_contours_.clear();
    has_last_detection_ = false;
    
    if (bin_image.empty() || bin_image.cols < 10 || bin_image.rows < 10) {
        if (node_) RCLCPP_WARN(node_->get_logger(), "Binary image is empty or too small");
        return false;
    }
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin_image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        if (node_) RCLCPP_DEBUG(node_->get_logger(), "No contours found");
        return false;
    }

    // === Debug: 保存原始轮廓图像 ===
    if (debug_mode_ && !original_image.empty()) {
        std::filesystem::create_directories(debug_dir_);
        cv::Mat contour_debug = original_image.clone();
        cv::drawContours(contour_debug, contours, -1, cv::Scalar(0, 255, 0), 2);
        cv::imwrite(debug_dir_ + "/all_contours.png", contour_debug);
        RCLCPP_INFO(node_->get_logger(), "Saved all contours image to: %s/all_contours.png", debug_dir_.c_str());
    }

    // 过滤轮廓用于排球检测
    auto filtered = filterContoursForVolleyball(contours);
    detected_contours_ = filtered;
    
    // === Debug: 保存过滤后的轮廓图像 ===
    if (debug_mode_ && !original_image.empty()) {
        drawContours(original_image, filtered, "filtered_contours.png");
    }
    
    if (node_) {
        RCLCPP_INFO(node_->get_logger(), "Found %zu contours, filtered to %zu", 
                   contours.size(), filtered.size());
    }

    // 从过滤后的轮廓中检测圆形
    bool detected = detectCircleFromContours(filtered, center, radius);
    
    if (detected) {
        // 验证检测到的圆形
        if (validateDetectedCircle(center, radius, original_image.size())) {
            last_detected_center_ = center;
            last_detected_radius_ = radius;
            has_last_detection_ = true;
            
            // === Debug: 保存检测结果图像 ===
            if (debug_mode_ && !original_image.empty()) {
                cv::Mat debug_image = original_image.clone();
                cv::circle(debug_image, center, static_cast<int>(radius), cv::Scalar(0, 255, 0), 3);
                cv::circle(debug_image, center, 5, cv::Scalar(0, 0, 255), -1);
                cv::putText(debug_image, "Volleyball", cv::Point(center.x + 10, center.y - 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                saveDebugImage("volleyball_detected.png", debug_image);
            }
            
            return true;
        } else {
            if (node_) RCLCPP_DEBUG(node_->get_logger(), "Circle validation failed");
            return false;
        }
    }
    
    return false;
}

bool FusionStationDetector::validateDetectedCircle(const cv::Point2f& center, float radius, const cv::Size& image_size) {
    if (!node_) {
        return false;
    }
    
    // 1. 半径范围校验
    double min_radius = node_->get_parameter("volleyball.min_radius").as_double();
    double max_radius = node_->get_parameter("volleyball.max_radius").as_double();
    
    if (radius < min_radius || radius > max_radius) {
        if (node_) RCLCPP_DEBUG(node_->get_logger(), "Radius %.1f not in range [%.1f, %.1f]", 
                               radius, min_radius, max_radius);
        return false;
    }
    
    // 2. 圆心位置校验 - 确保圆心在图像范围内
    if (center.x < 0 || center.x >= image_size.width || 
        center.y < 0 || center.y >= image_size.height) {
        if (node_) RCLCPP_DEBUG(node_->get_logger(), "Center (%.1f, %.1f) out of image bounds", 
                               center.x, center.y);
        return false;
    }
    
    // 3. 圆形边界校验 - 确保圆形在图像范围内
    if (center.x - radius < 0 || center.x + radius >= image_size.width ||
        center.y - radius < 0 || center.y + radius >= image_size.height) {
        if (node_) RCLCPP_DEBUG(node_->get_logger(), "Circle extends beyond image bounds");
        return false;
    }
    
    return true;
}

void FusionStationDetector::loadColorDetectionParams() {
    if (!node_) {
        RCLCPP_WARN(node_->get_logger(), "Node pointer is null, using default color parameters");
        return;
    }
    
    try {
        // 加载白色检测参数
        color_params_.white_lower = cv::Scalar(
            node_->get_parameter("color_detection.white_lower_h").as_int(),
            node_->get_parameter("color_detection.white_lower_s").as_int(),
            node_->get_parameter("color_detection.white_lower_v").as_int()
        );
        color_params_.white_upper = cv::Scalar(
            node_->get_parameter("color_detection.white_upper_h").as_int(),
            node_->get_parameter("color_detection.white_upper_s").as_int(),
            node_->get_parameter("color_detection.white_upper_v").as_int()
        );
        
        // 加载形态学处理参数
        color_params_.morphology_kernel_size = node_->get_parameter("color_detection.morphology_kernel_size").as_int();
        color_params_.use_adaptive_kernel = node_->get_parameter("color_detection.use_adaptive_kernel").as_bool();
        
        RCLCPP_INFO(node_->get_logger(), "Color detection parameters loaded successfully");
        RCLCPP_INFO(node_->get_logger(), "White range: [%d,%d,%d] - [%d,%d,%d]",
                   (int)color_params_.white_lower[0], (int)color_params_.white_lower[1], (int)color_params_.white_lower[2],
                   (int)color_params_.white_upper[0], (int)color_params_.white_upper[1], (int)color_params_.white_upper[2]);
        RCLCPP_INFO(node_->get_logger(), "Morphology kernel size: %d, adaptive: %s",
                   color_params_.morphology_kernel_size, color_params_.use_adaptive_kernel ? "true" : "false");
    } catch (const std::exception& e) {
        RCLCPP_ERROR(node_->get_logger(), "Failed to load color detection parameters: %s", e.what());
        RCLCPP_WARN(node_->get_logger(), "Using default color detection parameters");
    }
}

// 新增：保存调试图像
void FusionStationDetector::saveDebugImage(const std::string& filename, const cv::Mat& image) {
    if (debug_mode_ && !image.empty()) {
        std::string full_path = debug_dir_ + "/" + filename;
        cv::imwrite(full_path, image);
        RCLCPP_INFO(node_->get_logger(), "Saved debug image to: %s", full_path.c_str());
    }
}

// 新增：绘制轮廓
void FusionStationDetector::drawContours(const cv::Mat& original, const std::vector<std::vector<cv::Point>>& contours, const std::string& filename) {
    if (!debug_mode_ || original.empty()) return;
    
    cv::Mat contour_image = original.clone();
    
    // 绘制所有轮廓
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Scalar color = cv::Scalar(0, 255, 0); // 绿色
        cv::drawContours(contour_image, contours, i, color, 2);
        
        // 添加轮廓编号
        if (!contours[i].empty()) {
            cv::Point center = cv::Point(0, 0);
            for (const auto& pt : contours[i]) {
                center += pt;
            }
            center.x /= contours[i].size();
            center.y /= contours[i].size();
            cv::putText(contour_image, std::to_string(i), center, cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        }
    }
    
    saveDebugImage(filename, contour_image);
}

// 新增：绘制角点
void FusionStationDetector::drawCorners(const cv::Mat& original, const std::vector<cv::Point2f>& corners, const std::string& filename) {
    if (!debug_mode_ || original.empty() || corners.empty()) return;
    
    cv::Mat corner_image = original.clone();
    
    for (size_t i = 0; i < corners.size(); ++i) {
        cv::Point2f pt = corners[i];
        
        // 绘制角点圆圈
        cv::circle(corner_image, pt, 8, cv::Scalar(0, 0, 255), -1); // 红色实心圆
        
        // 绘制角点编号
        cv::Point text_pt = cv::Point(pt.x + 10, pt.y - 10);
        cv::putText(corner_image, std::to_string(i), text_pt, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 255), 2);
    }
    
    saveDebugImage(filename, corner_image);
}

// 新增：绘制拟合的矩形
void FusionStationDetector::drawFittedRectangle(const cv::Mat& original, const std::vector<cv::Point2f>& corners, const std::string& filename) {
    if (!debug_mode_ || original.empty() || corners.size() != 4) return;
    
    cv::Mat rect_image = original.clone();
    
    // 绘制拟合的矩形
    std::vector<cv::Point> rect_points;
    for (const auto& corner : corners) {
        rect_points.push_back(cv::Point(corner.x, corner.y));
    }
    
    // 绘制矩形边框
    for (size_t i = 0; i < rect_points.size(); ++i) {
        cv::Point pt1 = rect_points[i];
        cv::Point pt2 = rect_points[(i + 1) % rect_points.size()];
        cv::line(rect_image, pt1, pt2, cv::Scalar(255, 0, 0), 3); // 蓝色线条
    }
    
    // 绘制角点
    for (size_t i = 0; i < corners.size(); ++i) {
        cv::Point2f pt = corners[i];
        cv::circle(rect_image, pt, 6, cv::Scalar(0, 255, 0), -1); // 绿色实心圆
        cv::Point text_pt = cv::Point(pt.x + 10, pt.y - 10);
        cv::putText(rect_image, std::to_string(i), text_pt, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
    }
    
    saveDebugImage(filename, rect_image);
}

// 新增：绘制所有调试信息
void FusionStationDetector::drawAllDebugInfo(const cv::Mat& original, const std::vector<std::vector<cv::Point>>& contours, 
                                            const std::vector<cv::Point2f>& corners, const std::string& filename) {
    if (!debug_mode_ || original.empty()) return;
    
    cv::Mat debug_image = original.clone();
    
    // 绘制轮廓
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Scalar color = cv::Scalar(0, 255, 255); // 黄色
        cv::drawContours(debug_image, contours, i, color, 1);
    }
    
    // 绘制角点
    for (size_t i = 0; i < corners.size(); ++i) {
        cv::Point2f pt = corners[i];
        cv::circle(debug_image, pt, 8, cv::Scalar(0, 0, 255), -1); // 红色实心圆
        cv::Point text_pt = cv::Point(pt.x + 10, pt.y - 10);
        cv::putText(debug_image, std::to_string(i), text_pt, cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 2);
    }
    
    // 绘制拟合的矩形
    if (corners.size() == 4) {
        std::vector<cv::Point> rect_points;
        for (const auto& corner : corners) {
            rect_points.push_back(cv::Point(corner.x, corner.y));
        }
        
        for (size_t i = 0; i < rect_points.size(); ++i) {
            cv::Point pt1 = rect_points[i];
            cv::Point pt2 = rect_points[(i + 1) % rect_points.size()];
            cv::line(debug_image, pt1, pt2, cv::Scalar(255, 0, 0), 2); // 蓝色线条
        }
    }
    
    // 添加信息文本
    std::string info_text = "Contours: " + std::to_string(contours.size()) + 
                           ", Corners: " + std::to_string(corners.size());
    cv::putText(debug_image, info_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    
    saveDebugImage(filename, debug_image);
}

// 从轮廓中检测圆形
bool FusionStationDetector::detectCircleFromContours(const std::vector<std::vector<cv::Point>>& contours, 
                                                      cv::Point2f& center, float& radius) {
    if (contours.empty() || !node_) {
        return false;
    }
    
    double min_radius = node_->get_parameter("volleyball.min_radius").as_double();
    double max_radius = node_->get_parameter("volleyball.max_radius").as_double();
    double min_circularity = node_->get_parameter("volleyball.min_circularity").as_double();
    
    cv::Point2f best_center;
    float best_radius = 0;
    double best_score = 0.0;
    bool found = false;
    
    for (const auto& contour : contours) {
        if (contour.size() < 5) continue;  // 至少需要5个点才能拟合圆
        
        // 方法1: 使用最小外接圆
        cv::Point2f temp_center;
        float temp_radius;
        cv::minEnclosingCircle(contour, temp_center, temp_radius);
        
        // 计算圆形度 (circularity) = 4π*面积/周长²
        double area = cv::contourArea(contour);
        double perimeter = cv::arcLength(contour, true);
        double circularity = 0.0;
        if (perimeter > 1e-6) {
            circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        }
        
        // 验证半径和圆形度
        if (temp_radius >= min_radius && temp_radius <= max_radius && 
            circularity >= min_circularity) {
            // 计算得分: 圆形度 * 面积 (优先选择更大更圆的)
            double score = circularity * area;
            if (score > best_score) {
                best_score = score;
                best_center = temp_center;
                best_radius = temp_radius;
                found = true;
            }
        }
        
        // 方法2: 使用Hough圆检测作为备选 (如果轮廓足够大)
        if (contour.size() > 20) {
            // 将轮廓转换为点集用于Hough检测
            std::vector<cv::Point2f> points;
            for (const auto& pt : contour) {
                points.push_back(cv::Point2f(pt.x, pt.y));
            }
            
            // 使用fitEllipse作为圆形检测的替代
            if (points.size() >= 5) {
                cv::RotatedRect ellipse = cv::fitEllipse(points);
                float ellipse_radius = (ellipse.size.width + ellipse.size.height) / 4.0f;
                
                // 检查是否为圆形 (长宽比接近1)
                double aspect_ratio = std::max(ellipse.size.width, ellipse.size.height) / 
                                     std::min(ellipse.size.width, ellipse.size.height);
                if (aspect_ratio < 1.3 && ellipse_radius >= min_radius && 
                    ellipse_radius <= max_radius) {
                    double score = (1.0 / aspect_ratio) * area;  // 更圆得分更高
                    if (score > best_score) {
                        best_score = score;
                        best_center = ellipse.center;
                        best_radius = ellipse_radius;
                        found = true;
                    }
                }
            }
        }
    }
    
    if (found) {
        center = best_center;
        radius = best_radius;
        if (node_) {
            RCLCPP_INFO(node_->get_logger(), "Detected volleyball: center=(%.1f, %.1f), radius=%.1f, score=%.2f",
                       center.x, center.y, radius, best_score);
        }
        return true;
    }
    
    return false;
}

// 过滤轮廓用于排球检测
std::vector<std::vector<cv::Point>> FusionStationDetector::filterContoursForVolleyball(
    const std::vector<std::vector<cv::Point>>& contours) 
{
    if (!node_) {
        throw std::runtime_error("Node pointer is null in filterContoursForVolleyball");
    }
    
    std::vector<std::vector<cv::Point>> filtered;
    double min_area = node_->get_parameter("min_contour_area").as_double();
    double max_area = node_->get_parameter("max_contour_area").as_double();
    double min_circularity = node_->get_parameter("volleyball.min_circularity").as_double();
    
    for(size_t i = 0; i < contours.size(); ++i) {
        const auto& contour = contours[i];
        
        // 检查轮廓点有效性
        bool contour_valid = true;
        for (const auto& pt : contour) {
            if (pt.x < 0 || pt.y < 0) {
                contour_valid = false;
                break;
            }
        }
        if (!contour_valid) {
            if (node_) RCLCPP_DEBUG(node_->get_logger(), "Contour %zu: Invalid contour points detected", i);
            continue;
        }
        
        double area = cv::contourArea(contour);
        
        // 面积筛选 
        if(area < min_area || area > max_area) {
            if (node_) RCLCPP_DEBUG(node_->get_logger(), "Contour %zu: Area %.1f not in range [%.1f, %.1f]", 
                                   i, area, min_area, max_area);
            continue;
        }
        
        // 圆形度筛选 - 排球应该是圆形
        double perimeter = cv::arcLength(contour, true);
        double circularity = 0.0;
        if (perimeter > 1e-6) {
            circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
        }
        
        if (circularity < min_circularity) {
            if (node_) RCLCPP_DEBUG(node_->get_logger(), "Contour %zu: Circularity %.3f < %.3f", 
                                   i, circularity, min_circularity);
            continue;
        }
        
        // 长宽比筛选 - 圆形应该接近1:1
        cv::RotatedRect rect = cv::minAreaRect(contour);
        float width = rect.size.width;
        float height = rect.size.height;
        float aspect_ratio = (width > height) ? width/height : height/width;
        
        if(aspect_ratio > 1.5) {  // 长宽比不应超过1.5:1
            if (node_) RCLCPP_DEBUG(node_->get_logger(), "Contour %zu: Aspect ratio %.2f > 1.5", i, aspect_ratio);
            continue;
        }
        
        // 添加轮廓点数量筛选 - 确保轮廓足够复杂
        if(contour.size() < 5) {
            if (node_) RCLCPP_DEBUG(node_->get_logger(), "Contour %zu: Too few points %zu < 5", i, contour.size());
            continue;
        }
        
        if (node_) RCLCPP_INFO(node_->get_logger(), "Contour %zu: PASSED - Area: %.1f, Circularity: %.3f, Aspect: %.2f, Points: %zu", 
                              i, area, circularity, aspect_ratio, contour.size());
        filtered.push_back(contour);
    }
    
    // 按面积排序，优先选择更大的轮廓
    std::sort(filtered.begin(), filtered.end(), [](const auto& a, const auto& b){
        return cv::contourArea(a) > cv::contourArea(b);
    });
    
    // 只保留最好的几个候选
    if(filtered.size() > 3) {
        filtered.resize(3);
    }
    
    return filtered;
}

// 获取排球的真实半径(米)
double FusionStationDetector::getVolleyballRadius() const {
    if (!node_) {
        return 0.105;  // 默认值: 标准排球半径约10.5cm
    }
    return node_->get_parameter("volleyball.real_radius").as_double();
}