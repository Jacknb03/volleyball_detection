#include "station_pose_estimator.h"
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cv_bridge/cv_bridge.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <chrono>
#include <tf2/exceptions.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <memory>
#include <array>
#include <mutex>
#include <algorithm>
#include <numeric>
#include <limits>

// 辅助函数：确保图像连续存储
void ensureContinuousImage(cv::Mat& image) {
    if (!image.isContinuous()) {
        cv::Mat temp;
        image.copyTo(temp);
        image = temp;
    }
}

StationPoseEstimator::StationPoseEstimator(const rclcpp::NodeOptions& options)
: Node("station_pose_estimator", options),
  detector_initialized_(false),
  camera_info_received_(false),
  debug_mode_(false),
  show_binary_window_(false),
  tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
  tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this)),
  tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(this))
{
    // 参数声明 - 只声明本节点需要的参数
    declare_parameter("publish_tf", true);
    declare_parameter("max_reprojection_error", 2.0);
    declare_parameter("corner_filter_gain", 0.1);
    declare_parameter("show_binary_window", false);
    
    // debug参数由FusionStationDetector管理，这里先设为false
    debug_mode_ = false;
    show_binary_window_ = get_parameter("show_binary_window").as_bool();
    RCLCPP_INFO(get_logger(), "Debug mode: %s", debug_mode_ ? "ON" : "OFF");
    RCLCPP_INFO(get_logger(), "Show binary window: %s", show_binary_window_ ? "ON" : "OFF");

    // 先创建检测器但不初始化
    detector_ = std::make_unique<FusionStationDetector>();
    
    // 创建相机信息订阅 - 使用直接回调链
    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", 10,
        std::bind(&StationPoseEstimator::cameraInfoCallback, this, std::placeholders::_1));

    initKalmanFilter();

    RCLCPP_INFO(this->get_logger(), "StationPoseEstimator initialized");
}

void StationPoseEstimator::cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    if(!camera_info_received_) {
        camera_matrix_ = cv::Mat(3, 3, CV_64F, const_cast<double*>(msg->k.data())).clone();
        distortion_coeffs_ = cv::Mat(1, 5, CV_64F, const_cast<double*>(msg->d.data())).clone();
        camera_info_received_ = true;
        RCLCPP_INFO(get_logger(), "Received camera info");
        
        // 添加相机矩阵诊断信息
        RCLCPP_INFO(get_logger(), "Camera Matrix: [%f, %f, %f; %f, %f, %f; %f, %f, %f]",
                   camera_matrix_.at<double>(0,0), camera_matrix_.at<double>(0,1), camera_matrix_.at<double>(0,2),
                   camera_matrix_.at<double>(1,0), camera_matrix_.at<double>(1,1), camera_matrix_.at<double>(1,2),
                   camera_matrix_.at<double>(2,0), camera_matrix_.at<double>(2,1), camera_matrix_.at<double>(2,2));
        
        // 直接初始化检测器
        initDetector();
    }
}

void StationPoseEstimator::initDetector() {
    if (detector_initialized_) {
        return;
    }
    
    try {
        // 使用接收到的相机参数初始化检测器
        detector_->setCameraParameters(camera_matrix_, distortion_coeffs_);
        
        // 传递节点共享指针给检测器
        detector_->init(shared_from_this());
        
        // 现在才创建图像订阅
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", 10,
            [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {
                RCLCPP_INFO(this->get_logger(), "Received new image frame! Size: %dx%d", 
                           msg->width, msg->height);
                this->imageCallback(msg);
            });
        
        // 创建发布者   
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("volleyball_pose", 10);
        debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("debug_image", 10);
        binary_pub_ = this->create_publisher<sensor_msgs::msg::Image>("binary_image", 10);
        trajectory_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("volleyball_trajectory", 10);
        
        RCLCPP_INFO(this->get_logger(), "FusionStationDetector initialized");
        
        // 从检测器获取debug模式设置
        debug_mode_ = detector_->isDebugMode();
        RCLCPP_INFO(this->get_logger(), "Debug mode from detector: %s", debug_mode_ ? "ON" : "OFF");
        
        // 检查检测器是否已校准
        if (!detector_->isCalibrated()) {
            RCLCPP_WARN(this->get_logger(), "Detector is not calibrated yet");
        } else {
            RCLCPP_INFO(this->get_logger(), "Detector is properly calibrated");
        }
        
        detector_initialized_ = true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Detector initialization failed: %s", e.what());
    }
}

void StationPoseEstimator::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
    // 添加互斥锁防止多线程竞争
    std::lock_guard<std::mutex> lock(image_mutex_);
    
    RCLCPP_INFO(get_logger(), "开始处理图像: %dx%d", msg->width, msg->height);
    
    if (!detector_initialized_) {
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 1000, 
            "Detector not initialized yet. Skipping frame.");
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        publishDebugImage(frame, cv::Point2f(0,0), 0.0f, {}, msg->header, "Waiting for detector init");
        return;
    }

    if (!detector_->isCalibrated()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                            "Detector not calibrated yet. Skipping frame.");
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        publishDebugImage(frame, cv::Point2f(0,0), 0.0f, {}, msg->header, "Not calibrated");
        return;
    }

    if(!camera_info_received_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Waiting for camera info...");
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        publishDebugImage(frame, cv::Point2f(0,0), 0.0f, {}, msg->header, "No camera info");
        return;
    }
    
    if (camera_matrix_.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Camera matrix not initialized");
        cv::Mat frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        publishDebugImage(frame, cv::Point2f(0,0), 0.0f, {}, msg->header, "Invalid camera matrix");
        return;
    }
    
    cv::Mat frame;
    try {
        frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }
    
    cv::Mat processed;
    try {
        processed = detector_->preprocess(frame);
        RCLCPP_DEBUG(get_logger(), "预处理完成");
        
        // 发布二值图像用于调试
        publishBinaryImage(processed, msg->header);
        
        // 可选：显示二值化窗口
        if (show_binary_window_) {
            cv::imshow("Binary Image", processed);
            cv::waitKey(1);
        }
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "预处理失败: %s", e.what());
        publishDebugImage(frame, cv::Point2f(0,0), 0.0f, {}, msg->header, "Preprocessing failed");
        return;
    }
    
    // 检测排球
    cv::Point2f center;
    float radius = 0.0f;
    std::vector<std::vector<cv::Point>> detected_contours;
    bool detected = false;

    try {
        detected = detector_->detectVolleyball(processed, frame, center, radius);
        if (detected) {
            RCLCPP_INFO(get_logger(), "检测到排球: 中心=(%.1f, %.1f), 半径=%.1f", center.x, center.y, radius);
        } else {
            RCLCPP_DEBUG(get_logger(), "未检测到排球");
        }
        // 获取检测到的轮廓用于可视化
        detected_contours = detector_->getDetectedContours();
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "排球检测失败: %s", e.what());
        detected = false;
        detected_contours.clear();
    }
    
    // 发布调试图像
    publishDebugImage(frame, center, radius, detected_contours, msg->header, 
                     detected ? "Volleyball detected" : "No volleyball");
    
    if (detected) {
        // 使用卡尔曼滤波平滑中心位置
        if (has_prev_center_) {
            double gain = get_parameter("corner_filter_gain").as_double();  // 复用参数
            center = (1.0 - gain) * prev_center_ + gain * center;
            radius = (1.0 - gain) * prev_radius_ + gain * radius;
        }
        prev_center_ = center;
        prev_radius_ = radius;
        has_prev_center_ = true;
        
        // 基于圆心和半径估计3D位置
        try {
            geometry_msgs::msg::PoseStamped pose = estimateVolleyballPose(center, radius, msg->header);
            
            if (!pose.header.frame_id.empty()) {
                // 应用卡尔曼滤波
                applyKalmanFilter(pose);
                
                // 预测未来路径
                auto predicted_poses = predictTrajectory(pose, 0.5);  // 预测0.5秒
                
                // 发布当前位姿
                pose_pub_->publish(pose);
                RCLCPP_INFO(get_logger(), "Published volleyball pose: (%.3f, %.3f, %.3f)", 
                           pose.pose.position.x, pose.pose.position.y, pose.pose.position.z);
                
                // 发布预测路径(可选)
                if (trajectory_pub_) {
                    publishTrajectory(predicted_poses, msg->header);
                }
                
                publishTransform(pose);
                publishDebugImage(frame, center, radius, detected_contours, msg->header, "Pose estimated");
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "位姿估计失败: %s", e.what());
            publishDebugImage(frame, center, radius, detected_contours, msg->header, "Pose estimation failed");
        }
    } else {
        has_prev_center_ = false;
        publishDebugImage(frame, center, radius, detected_contours, msg->header, "No detection");
    }
}

// validateCorners函数已删除 - 现在使用FusionStationDetector::validateDetectedCorners

geometry_msgs::msg::PoseStamped StationPoseEstimator::transformToBaseLink(
    const cv::Mat& rvec, const cv::Mat& tvec, 
    const std_msgs::msg::Header& header)
{
    geometry_msgs::msg::PoseStamped pose_cam;
    pose_cam.header = header;
    
    // 旋转向量转换为旋转矩阵
    cv::Mat rotation_matrix;
    cv::Rodrigues(rvec, rotation_matrix);
    
    // 转换为四元数
    tf2::Matrix3x3 tf_rot(
        rotation_matrix.at<double>(0,0), rotation_matrix.at<double>(0,1), rotation_matrix.at<double>(0,2),
        rotation_matrix.at<double>(1,0), rotation_matrix.at<double>(1,1), rotation_matrix.at<double>(1,2),
        rotation_matrix.at<double>(2,0), rotation_matrix.at<double>(2,1), rotation_matrix.at<double>(2,2));
    
    tf2::Quaternion quat;
    tf_rot.getRotation(quat);
    
    pose_cam.pose.position.x = tvec.at<double>(0);
    pose_cam.pose.position.y = tvec.at<double>(1);
    pose_cam.pose.position.z = tvec.at<double>(2);
    pose_cam.pose.orientation = tf2::toMsg(quat);
    
    // 转换到base_link坐标系
    geometry_msgs::msg::PoseStamped pose_base;
    try {
        auto transform = tf_buffer_->lookupTransform(
            "base_link", header.frame_id, header.stamp,
            tf2::durationFromSec(0.1));
            
        tf2::doTransform(pose_cam, pose_base, transform);
        pose_base.header.frame_id = "base_link";
        return pose_base;
    } 
    catch (tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "TF error: %s", ex.what());
        return geometry_msgs::msg::PoseStamped(); // 返回空位姿
    }
}

void StationPoseEstimator::applyKalmanFilter(geometry_msgs::msg::PoseStamped& pose) {
    // 卡尔曼滤波预测
    kalman_filter_.predict();
    
    // 测量向量 [x, y, z]
    Eigen::Vector3d measurement(
        pose.pose.position.x,
        pose.pose.position.y,
        pose.pose.position.z);
    
    // 更新卡尔曼滤波器
    kalman_filter_.update(measurement);
    
    // 获取滤波后状态 [x, y, z, vx, vy, vz, ax, ay, az]
    Eigen::VectorXd state = kalman_filter_.getState();
    
    // 更新位姿
    pose.pose.position.x = state[0];
    pose.pose.position.y = state[1];
    pose.pose.position.z = state[2];
    
    // 注意: 速度信息存储在state[3-5], 加速度在state[6-8], 可用于路径预测
}

void StationPoseEstimator::publishTransform(const geometry_msgs::msg::PoseStamped& pose) {
    if(get_parameter("publish_tf").as_bool()) {
        geometry_msgs::msg::TransformStamped tf;
        tf.header = pose.header;
        tf.child_frame_id = "volleyball";
        tf.transform.translation.x = pose.pose.position.x;
        tf.transform.translation.y = pose.pose.position.y;
        tf.transform.translation.z = pose.pose.position.z;
        tf.transform.rotation = pose.pose.orientation;
        tf_broadcaster_->sendTransform(tf);
        RCLCPP_DEBUG(get_logger(), "Published TF transform");
    }
}

// 基于圆心和半径估计3D位置
geometry_msgs::msg::PoseStamped StationPoseEstimator::estimateVolleyballPose(
    const cv::Point2f& center, float radius, const std_msgs::msg::Header& header) {
    
    geometry_msgs::msg::PoseStamped pose;
    pose.header = header;
    
    // 获取相机内参
    double fx = camera_matrix_.at<double>(0, 0);
    double fy = camera_matrix_.at<double>(1, 1);
    double cx = camera_matrix_.at<double>(0, 2);
    double cy = camera_matrix_.at<double>(1, 2);
    
    // 获取排球真实半径(米)
    double real_radius = detector_->getVolleyballRadius();
    
    // 计算图像中的半径(像素)对应的深度
    // 使用相似三角形: real_radius / depth = pixel_radius / focal_length
    // depth = real_radius * focal_length / pixel_radius
    double avg_focal = (fx + fy) / 2.0;
    double depth = (real_radius * avg_focal) / radius;
    
    // 计算3D位置 (相机坐标系)
    // x = (u - cx) * depth / fx
    // y = (v - cy) * depth / fy
    // z = depth
    double x = (center.x - cx) * depth / fx;
    double y = (center.y - cy) * depth / fy;
    double z = depth;
    
    pose.pose.position.x = x;
    pose.pose.position.y = y;
    pose.pose.position.z = z;
    
    // 朝向: 假设排球朝向相机(四元数单位)
    pose.pose.orientation.w = 1.0;
    pose.pose.orientation.x = 0.0;
    pose.pose.orientation.y = 0.0;
    pose.pose.orientation.z = 0.0;
    
    // 转换到base_link坐标系
    try {
        auto transform = tf_buffer_->lookupTransform(
            "base_link", header.frame_id, header.stamp,
            tf2::durationFromSec(0.1));
        
        geometry_msgs::msg::PoseStamped pose_base;
        tf2::doTransform(pose, pose_base, transform);
        pose_base.header.frame_id = "base_link";
        return pose_base;
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "TF error: %s", ex.what());
        // 返回相机坐标系下的位姿
        return pose;
    }
}

// 预测未来路径
std::vector<geometry_msgs::msg::PoseStamped> StationPoseEstimator::predictTrajectory(
    const geometry_msgs::msg::PoseStamped& current_pose, double time_horizon) {
    
    std::vector<geometry_msgs::msg::PoseStamped> predicted_poses;
    
    // 获取当前状态
    Eigen::VectorXd state = kalman_filter_.getState();
    
    // 当前状态: [x, y, z, vx, vy, vz, ax, ay, az]
    Eigen::Vector3d pos(state[0], state[1], state[2]);
    Eigen::Vector3d vel(state[3], state[4], state[5]);
    Eigen::Vector3d acc(state[6], state[7], state[8]);
    
    // 预测时间步长
    double dt = 0.05;  // 50ms
    int num_steps = static_cast<int>(time_horizon / dt);
    
    for (int i = 1; i <= num_steps; ++i) {
        double t = i * dt;
        
        // 使用运动学方程预测: x = x0 + v0*t + 0.5*a*t²
        Eigen::Vector3d predicted_pos = pos + vel * t + 0.5 * acc * t * t;
        
        geometry_msgs::msg::PoseStamped predicted_pose = current_pose;
        predicted_pose.pose.position.x = predicted_pos[0];
        predicted_pose.pose.position.y = predicted_pos[1];
        predicted_pose.pose.position.z = predicted_pos[2];
        
        // 更新时间戳
        predicted_pose.header.stamp = rclcpp::Time(current_pose.header.stamp) + 
                                      rclcpp::Duration::from_seconds(t);
        
        predicted_poses.push_back(predicted_pose);
    }
    
    return predicted_poses;
}

// 发布预测路径
void StationPoseEstimator::publishTrajectory(
    const std::vector<geometry_msgs::msg::PoseStamped>& poses,
    const std_msgs::msg::Header& header) {
    
    if (!trajectory_pub_ || poses.empty()) {
        return;
    }
    
    visualization_msgs::msg::MarkerArray marker_array;
    
    // 创建路径线条
    visualization_msgs::msg::Marker line_marker;
    line_marker.header = header;
    line_marker.ns = "volleyball_trajectory";
    line_marker.id = 0;
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.scale.x = 0.02;  // 线宽
    line_marker.color.r = 1.0;
    line_marker.color.g = 0.0;
    line_marker.color.b = 0.0;
    line_marker.color.a = 1.0;
    
    for (const auto& pose : poses) {
        geometry_msgs::msg::Point pt;
        pt.x = pose.pose.position.x;
        pt.y = pose.pose.position.y;
        pt.z = pose.pose.position.z;
        line_marker.points.push_back(pt);
    }
    
    marker_array.markers.push_back(line_marker);
    
    // 创建预测点标记
    for (size_t i = 0; i < poses.size(); i += 5) {  // 每5个点显示一个标记
        visualization_msgs::msg::Marker point_marker;
        point_marker.header = header;
        point_marker.ns = "volleyball_trajectory";
        point_marker.id = static_cast<int>(i + 1);
        point_marker.type = visualization_msgs::msg::Marker::SPHERE;
        point_marker.action = visualization_msgs::msg::Marker::ADD;
        point_marker.pose = poses[i].pose;
        point_marker.scale.x = 0.05;
        point_marker.scale.y = 0.05;
        point_marker.scale.z = 0.05;
        point_marker.color.r = 0.0;
        point_marker.color.g = 1.0;
        point_marker.color.b = 0.0;
        point_marker.color.a = 0.5;
        marker_array.markers.push_back(point_marker);
    }
    
    trajectory_pub_->publish(marker_array);
}

void StationPoseEstimator::initKalmanFilter() {
    // 状态向量: [x, y, z, vx, vy, vz, ax, ay, az] - 添加加速度用于路径预测
    double dt = 0.05;  // 时间步长(秒), 假设20Hz
    Eigen::MatrixXd F(9,9);  // 状态转移矩阵
    F << 1,0,0,dt,0,0,0.5*dt*dt,0,0,   // x = x + vx*dt + 0.5*ax*dt²
         0,1,0,0,dt,0,0,0.5*dt*dt,0,
         0,0,1,0,0,dt,0,0,0.5*dt*dt,
         0,0,0,1,0,0,dt,0,0,          // vx = vx + ax*dt
         0,0,0,0,1,0,0,dt,0,
         0,0,0,0,0,1,0,0,dt,
         0,0,0,0,0,0,1,0,0,            // ax = ax (假设加速度变化缓慢)
         0,0,0,0,0,0,0,1,0,
         0,0,0,0,0,0,0,0,1;
    
    Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(9,9);  // 过程噪声
    Q.block<3,3>(0,0) *= 0.01;  // 位置噪声
    Q.block<3,3>(3,3) *= 0.1;   // 速度噪声
    Q.block<3,3>(6,6) *= 0.5;  // 加速度噪声
    
    Eigen::MatrixXd H(3,9);  // 观测矩阵 (只能观测位置)
    H << 1,0,0,0,0,0,0,0,0,
         0,1,0,0,0,0,0,0,0,
         0,0,1,0,0,0,0,0,0;
    
    Eigen::MatrixXd R = 0.1 * Eigen::MatrixXd::Identity(3,3);  // 观测噪声
    
    kalman_filter_.init(F, H, Q, R, Eigen::VectorXd::Zero(9));
    RCLCPP_INFO(get_logger(), "Kalman filter initialized with position, velocity, and acceleration");
}

void StationPoseEstimator::publishBinaryImage(const cv::Mat& binary_image, 
                                             const std_msgs::msg::Header& header) 
{
    // 添加安全检查和详细日志
    if (!binary_pub_) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, 
                            "Binary publisher not initialized");
        return;
    }
    // 检查图像有效性
    if (binary_image.empty() || binary_image.cols < 10 || binary_image.rows < 10) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                            "Binary image is empty or too small, skip publishing");
        return;
    }
    
    // 检查图像类型 (必须为单通道)
    cv::Mat img_to_publish;
    if (binary_image.type() != CV_8UC1) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, 
                            "Converting binary image type: %d to CV_8UC1", 
                            binary_image.type());
        binary_image.convertTo(img_to_publish, CV_8UC1);
    } else {
        img_to_publish = binary_image;
    }

    try {
        // 创建独立的图像副本
        cv::Mat img_copy = img_to_publish.clone();
        // 确保图像连续存储
        ensureContinuousImage(img_copy);
        
        // 使用共享指针确保图像数据在发布期间保持有效
        auto cv_image = std::make_shared<cv_bridge::CvImage>();
        cv_image->header = header;
        cv_image->encoding = "mono8";
        cv_image->image = img_copy;
        
        auto binary_msg = cv_image->toImageMsg();
        binary_pub_->publish(*binary_msg);
        RCLCPP_DEBUG(get_logger(), "Published binary image");
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error publishing binary image: %s", e.what());
    }
}

void StationPoseEstimator::publishDebugImage(
    const cv::Mat& frame, 
    const cv::Point2f& center,
    float radius,
    const std::vector<std::vector<cv::Point>>& current_contours,
    const std_msgs::msg::Header& header,
    const std::string& status_text)
{
    if (!debug_pub_) {
        RCLCPP_WARN(get_logger(), "Debug publisher not initialized, skipping debug image");
        return;
    }
    
    // 检查原始图像有效性
    if (frame.empty() || frame.cols < 10 || frame.rows < 10) {
        RCLCPP_WARN(get_logger(), "Input frame is empty or too small, skip publishing debug image");
        return;
    }
    
    // 创建调试图像（缩放大图像以提高性能）
    cv::Mat debug;
    double scale = 1.0;
    const int min_dimension = 50; // 最小有效尺寸
    
    if (frame.cols > 1280 || frame.rows > 1024) {
        scale = 0.5;
        cv::Size new_size(
            std::max(min_dimension, static_cast<int>(frame.cols * scale)),
            std::max(min_dimension, static_cast<int>(frame.rows * scale))
        );
        cv::resize(frame, debug, new_size);
    } else {
        debug = frame.clone();
    }
    
    // 检查缩放后图像有效性
    if (debug.empty() || debug.cols < min_dimension || debug.rows < min_dimension) {
        RCLCPP_WARN(get_logger(), "Resized debug image is invalid, skip publishing");
        return;
    }
    
    const int width = debug.cols;
    const int height = debug.rows;
    
    // 安全绘制状态文本
    if (!status_text.empty() && height > 40) {
        cv::putText(debug, status_text, cv::Point(20, 40), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
    }
    
    // 安全绘制轮廓（使用缩放因子）
    for (const auto& contour : current_contours) {
        if (contour.empty()) continue;
        
        // 创建缩放后的轮廓
        std::vector<cv::Point> scaled_contour;
        for (const auto& pt : contour) {
            int x = static_cast<int>(pt.x * scale);
            int y = static_cast<int>(pt.y * scale);
            
            // 确保坐标在图像范围内
            if (x >= 0 && x < width && y >= 0 && y < height) {
                scaled_contour.push_back(cv::Point(x, y));
            }
        }
        
        if (!scaled_contour.empty()) {
            std::vector<std::vector<cv::Point>> temp = {scaled_contour};
            cv::drawContours(debug, temp, -1, cv::Scalar(0, 255, 255), 2);
        }
    }
    
    // 绘制检测到的排球圆形
    if (radius > 0) {
        int x = static_cast<int>(center.x * scale);
        int y = static_cast<int>(center.y * scale);
        int r = static_cast<int>(radius * scale);
        
        if (x >= 0 && x < width && y >= 0 && y < height && r > 0) {
            // 绘制圆形
            cv::circle(debug, cv::Point(x, y), r, cv::Scalar(0, 255, 0), 2);
            // 绘制圆心
            cv::circle(debug, cv::Point(x, y), 5, cv::Scalar(0, 0, 255), -1);
            
            // 绘制文本
            if (y + 30 < height) {
                std::string info = "R:" + std::to_string(static_cast<int>(radius));
                cv::putText(debug, info, cv::Point(x + 10, y + 10),
                           cv::FONT_HERSHEY_SIMPLEX, 0.6 * scale, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    
    // 添加时间戳和帧ID
    if (height > 120) {
        cv::putText(debug, "Frame: " + header.frame_id, cv::Point(20, 80), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7 * scale, cv::Scalar(200, 200, 200), 1);
        
        // 添加当前时间
        auto now = this->now();
        cv::putText(debug, "Time: " + std::to_string(now.seconds()), cv::Point(20, 120), 
                   cv::FONT_HERSHEY_SIMPLEX, 0.7 * scale, cv::Scalar(200, 200, 200), 1);
    }
    
    // 添加参数信息 - 避免访问可能不存在的参数
    // 使用try-catch确保不会崩溃
    try {
        // 只尝试获取本节点声明的参数
        double gain = get_parameter("corner_filter_gain").as_double();
        
        if (height > 160) {
            cv::putText(debug, "Filter Gain: " + std::to_string(gain), 
                       cv::Point(20, 160), cv::FONT_HERSHEY_SIMPLEX, 0.5 * scale, cv::Scalar(200, 200, 200), 1);
        }
    } catch (const rclcpp::exceptions::ParameterNotDeclaredException& e) {
        // 忽略未声明参数的错误
    } catch (...) {
        // 忽略其他错误
    }

    try {
        // 创建独立的图像副本
        cv::Mat debug_copy = debug.clone();
        // 确保图像连续存储
        ensureContinuousImage(debug_copy);
        
        // 使用共享指针确保图像数据在发布期间保持有效
        auto cv_image = std::make_shared<cv_bridge::CvImage>();
        cv_image->header = header;
        cv_image->encoding = "bgr8";
        cv_image->image = debug_copy;
        
        auto debug_msg = cv_image->toImageMsg();
        debug_pub_->publish(*debug_msg);
        RCLCPP_DEBUG(get_logger(), "Published debug image");
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Error publishing debug image: %s", e.what());
    }
}
