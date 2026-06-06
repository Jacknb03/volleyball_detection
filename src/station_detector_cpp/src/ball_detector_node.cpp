#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/utils.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <algorithm>
#include <cmath>

#include "yolo_inference.hpp"
#include "ball_tracker.hpp"
#include "detection_filter.hpp"
#include "trajectory_predictor.hpp"
#include "ball_position_estimator.hpp"

using std::placeholders::_1;

class BallDetectorNode : public rclcpp::Node
{
public:
    BallDetectorNode()
        : rclcpp::Node("ball_detector_node")
    {
        // 参数声明（简化版，对齐 Python 节点的含义）
        declare_parameter<std::string>("yolo.model_path", "");
        declare_parameter<std::string>("yolo.model_type", "yolov8");
        declare_parameter<double>("yolo.conf_threshold", 0.5);
        declare_parameter<double>("yolo.iou_threshold", 0.45);
        declare_parameter<std::string>("yolo.device", "auto");
        declare_parameter<std::vector<std::string>>("yolo.volleyball_classes",
                                                    std::vector<std::string>{"sports ball", "ball"});

        declare_parameter<double>("kalman.process_noise", 0.1);
        declare_parameter<double>("kalman.measurement_noise", 5.0);
        declare_parameter<double>("kalman.initial_uncertainty", 10.0);
        declare_parameter<int>("kalman.max_missing_frames", 5);

        declare_parameter<double>("detection.max_jump_distance", 100.0);
        declare_parameter<int>("detection.min_consistent_detections", 1);
        declare_parameter<int>("detection.max_detections", 3);
        declare_parameter<std::string>("detection.selection_method", "confidence");
        declare_parameter<double>("detection.min_confidence", 0.3);
        declare_parameter<bool>("detection.use_tracker", true);
        declare_parameter<double>("detection.h_ema_alpha", 0.3);
        declare_parameter<double>("detection.max_physical_speed", 25.0);
        declare_parameter<bool>("detection.emergency_bypass", false);

        declare_parameter<std::string>("input.image_topic", "/image_raw");
        declare_parameter<std::string>("input.camera_info_topic", "/camera_info");

        declare_parameter<std::string>("position.mode", "bbox");  // bbox | depth
        declare_parameter<std::string>("position.depth_topic",
                                      "/camera/aligned_depth_to_color/image_raw");
        declare_parameter<double>("position.depth_max_stamp_delta_sec", 0.05);
        declare_parameter<double>("position.depth_min_m", 0.3);
        declare_parameter<double>("position.depth_max_m", 8.0);
        declare_parameter<int>("position.depth_patch_radius", 2);

        declare_parameter<double>("trajectory.prediction_time", 0.5);
        declare_parameter<int>("trajectory.num_points", 10);

        // 坐标系与物理参数（用于世界系 TF + 阻力预测）
        declare_parameter<std::string>("world_frame_id", "odom");
        declare_parameter<std::string>("camera_frame_id", "camera_optical_frame");

        declare_parameter<double>("air_density", 1.225);         // kg/m^3
        declare_parameter<double>("drag_coefficient", 0.47);    // 球的 Cd（无量纲）
        declare_parameter<double>("volleyball.mass_kg", 0.27);   // kg，标准排球约 260-280g
        declare_parameter<double>("trajectory.integration_dt", 0.01);
        declare_parameter<double>("trajectory.max_time", 5.0);
        declare_parameter<double>("trajectory.ground_z", 0.0);

        declare_parameter<bool>("debug.enable", true);
        declare_parameter<bool>("debug.show_fps", true);
        declare_parameter<bool>("debug.draw_trajectory", true);

        declare_parameter<double>("volleyball.real_radius", 0.105);
        declare_parameter<double>("volleyball.min_depth", 0.3);
        declare_parameter<double>("volleyball.max_depth", 5.0);
        declare_parameter<double>("volleyball.diameter", 0.225);  // 22-23cm 取中间值

        // 读取参数并初始化 YOLO
        initYolo();

        // 初始化检测过滤和轨迹跟踪
        initTracking();
        world_frame_id_ = get_parameter("world_frame_id").as_string();
        camera_frame_id_ = get_parameter("camera_frame_id").as_string();
        if (world_frame_id_ == camera_frame_id_) {
            RCLCPP_WARN(
                get_logger(),
                "CRITICAL FRAME WARNING: world_frame_id == camera_frame_id (%s). "
                "Gravity is applied on World Z(height), but camera optical Z is depth. "
                "Trajectory may drop vertically unless TF maps camera->world correctly.",
                world_frame_id_.c_str());
        }

        // TF2: 相机坐标系 -> 世界坐标系
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 物理参数初始化（用于欧拉积分预测）
        const double gravity = 9.81;
        const double air_density = get_parameter("air_density").as_double();
        const double drag_coefficient = get_parameter("drag_coefficient").as_double();
        double diameter = get_parameter("volleyball.diameter").as_double();
        if (diameter <= 0.0) {
            // 兼容：如果 YAML 没填 diameter，则用 real_radius*2
            const double r = get_parameter("volleyball.real_radius").as_double();
            if (r > 0.0) {
                diameter = 2.0 * r;
            }
        }
        const double mass_kg = get_parameter("volleyball.mass_kg").as_double();
        const double integration_dt = get_parameter("trajectory.integration_dt").as_double();
        const double max_time = get_parameter("trajectory.max_time").as_double();
        const double ground_z = get_parameter("trajectory.ground_z").as_double();

        trajectory_predictor_ = std::make_unique<TrajectoryPredictor>(
            gravity, air_density, drag_coefficient, diameter, mass_kg,
            integration_dt, max_time, ground_z);

        position_mode_ = toLower(get_parameter("position.mode").as_string());
        if (position_mode_ != "bbox" && position_mode_ != "depth") {
            RCLCPP_WARN(get_logger(),
                        "Unknown position.mode='%s', fallback to 'bbox'",
                        position_mode_.c_str());
            position_mode_ = "bbox";
        }

        const auto image_topic = get_parameter("input.image_topic").as_string();
        const auto camera_info_topic = get_parameter("input.camera_info_topic").as_string();

        // QoS：实时性优先
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            image_topic, qos,
            std::bind(&BallDetectorNode::imageCallback, this, _1));

        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            camera_info_topic, 10,
            std::bind(&BallDetectorNode::cameraInfoCallback, this, _1));

        if (position_mode_ == "depth") {
            const auto depth_topic = get_parameter("position.depth_topic").as_string();
            depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
                depth_topic, qos,
                std::bind(&BallDetectorNode::depthCallback, this, _1));
            RCLCPP_INFO(get_logger(),
                        "Position mode=depth, subscribed to %s", depth_topic.c_str());
        } else {
            RCLCPP_INFO(get_logger(), "Position mode=bbox (2D box height -> depth)");
        }

        pose_pub_ = create_publisher<geometry_msgs::msg::PoseStamped>(
            "/volleyball_pose", 10);

        traj_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/volleyball_trajectory", 10);

        debug_pub_ = create_publisher<sensor_msgs::msg::Image>(
            "/debug_image", 10);

        ball_state_pub_ = create_publisher<geometry_msgs::msg::Point>(
            "/ball_state", 10);
        ball_prediction_pub_ = create_publisher<geometry_msgs::msg::Point>(
            "/ball_prediction", 10);

        last_frame_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
        last_detection_time_valid_ = false;
        RCLCPP_INFO(get_logger(), "BallDetectorNode started.");
    }

private:
    void initYolo()
    {
        auto model_path   = get_parameter("yolo.model_path").as_string();
        auto model_type   = get_parameter("yolo.model_type").as_string();
        auto conf_th      = static_cast<float>(get_parameter("yolo.conf_threshold").as_double());
        auto iou_th       = static_cast<float>(get_parameter("yolo.iou_threshold").as_double());
        auto device       = get_parameter("yolo.device").as_string();

        yolo_detector_ = std::make_unique<YOLODetector>(
            model_path,
            model_type,
            conf_th,
            iou_th,
            device);

        RCLCPP_INFO(get_logger(),
                    "YOLODetector created: type=%s, conf=%.2f, iou=%.2f, device=%s",
                    model_type.c_str(), conf_th, iou_th, device.c_str());
    }

    void initTracking()
    {
        float process_noise =
            static_cast<float>(get_parameter("kalman.process_noise").as_double());
        float meas_noise =
            static_cast<float>(get_parameter("kalman.measurement_noise").as_double());
        float init_uncertainty =
            static_cast<float>(get_parameter("kalman.initial_uncertainty").as_double());
        int max_missing_frames =
            get_parameter("kalman.max_missing_frames").as_int();

        ball_tracker_ = std::make_unique<BallTracker>(
            process_noise, meas_noise, init_uncertainty, max_missing_frames);

        float max_jump =
            static_cast<float>(get_parameter("detection.max_jump_distance").as_double());
        int min_consistent =
            get_parameter("detection.min_consistent_detections").as_int();

        const float min_confidence =
            static_cast<float>(get_parameter("detection.min_confidence").as_double());
        detection_filter_ = std::make_unique<DetectionFilter>(
            max_jump, min_consistent, 5, min_confidence);

        bool use_tracker = get_parameter("detection.use_tracker").as_bool();
        if (use_tracker) {
            multi_tracker_ = std::make_unique<MultiDetectionTracker>(max_jump * 1.5f);
        } else {
            multi_tracker_.reset();
        }
    }

    void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (camera_info_received_) {
            return;
        }

        camera_matrix_ = cv::Mat(3, 3, CV_64F);
        for (int i = 0; i < 9; ++i) {
            camera_matrix_.at<double>(i / 3, i % 3) = msg->k[i];
        }

        distortion_coeffs_ = cv::Mat(1, static_cast<int>(msg->d.size()), CV_64F);
        for (size_t i = 0; i < msg->d.size(); ++i) {
            distortion_coeffs_.at<double>(0, static_cast<int>(i)) = msg->d[i];
        }

        camera_info_received_ = true;
        RCLCPP_INFO(get_logger(), "Received camera info.");

        // 使用相机内参和排球直径初始化 3D 位置估计器
        const double fx = camera_matrix_.at<double>(0, 0);
        const double fy = camera_matrix_.at<double>(1, 1);
        const double cx = camera_matrix_.at<double>(0, 2);
        const double cy = camera_matrix_.at<double>(1, 2);
        double diameter = get_parameter("volleyball.diameter").as_double();
        if (diameter <= 0.0) {
            const double r = get_parameter("volleyball.real_radius").as_double();
            diameter = (r > 0.0) ? (2.0 * r) : 0.225;
        }
        position_estimator_ = std::make_unique<BallPositionEstimator>(
            fx, fy, cx, cy, diameter);
    }

    static std::string toLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
        return s;
    }

    bool isVolleyballClass(int class_id) const
    {
        if (!yolo_detector_) {
            return false;
        }

        // 单类别模型（如 [1,5,8400] -> 1 class）直接信任检测，不做类别名字符串过滤
        if (yolo_detector_->getNumClasses() == 1) {
            return true;
        }

        const std::string cls_name = toLower(yolo_detector_->getClassName(class_id));

        auto allowed = get_parameter("yolo.volleyball_classes").as_string_array();
        for (const auto& a : allowed) {
            const std::string needle = toLower(a);
            if (!needle.empty() && cls_name.find(needle) != std::string::npos) {
                return true;
            }
        }
        return false;
    }

    void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            if (msg->encoding == sensor_msgs::image_encodings::TYPE_16UC1) {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
            } else if (msg->encoding == sensor_msgs::image_encodings::TYPE_32FC1) {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_32FC1);
            } else {
                cv_ptr = cv_bridge::toCvCopy(msg);
            }
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_WARN(get_logger(), "depth cv_bridge exception: %s", e.what());
            return;
        }

        latest_depth_ = cv_ptr->image.clone();
        latest_depth_encoding_ = msg->encoding;
        latest_depth_stamp_ = rclcpp::Time(msg->header.stamp);
        has_latest_depth_ = !latest_depth_.empty();
    }

    std::optional<double> sampleDepthMeters(int u, int v,
                                            const rclcpp::Time& color_stamp)
    {
        if (!has_latest_depth_ || latest_depth_.empty()) {
            return std::nullopt;
        }

        const double max_delta =
            get_parameter("position.depth_max_stamp_delta_sec").as_double();
        const double stamp_delta =
            std::abs((color_stamp - latest_depth_stamp_).seconds());
        if (stamp_delta > max_delta) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Depth/RGB stamp delta %.3fs > %.3fs, skip depth sample",
                stamp_delta, max_delta);
            return std::nullopt;
        }

        const int radius = std::max(
            0, static_cast<int>(get_parameter("position.depth_patch_radius").as_int()));
        const int h = latest_depth_.rows;
        const int w = latest_depth_.cols;
        if (u < 0 || v < 0 || u >= w || v >= h) {
            return std::nullopt;
        }

        std::vector<double> samples;
        samples.reserve(static_cast<size_t>((2 * radius + 1) * (2 * radius + 1)));

        for (int dy = -radius; dy <= radius; ++dy) {
            for (int dx = -radius; dx <= radius; ++dx) {
                const int x = std::clamp(u + dx, 0, w - 1);
                const int y = std::clamp(v + dy, 0, h - 1);
                double depth_m = 0.0;
                if (latest_depth_encoding_ == sensor_msgs::image_encodings::TYPE_16UC1) {
                    const uint16_t raw = latest_depth_.at<uint16_t>(y, x);
                    if (raw == 0) {
                        continue;
                    }
                    depth_m = static_cast<double>(raw) * 0.001;  // mm -> m
                } else if (latest_depth_encoding_ == sensor_msgs::image_encodings::TYPE_32FC1) {
                    const float raw = latest_depth_.at<float>(y, x);
                    if (!(raw > 0.0f) || !std::isfinite(raw)) {
                        continue;
                    }
                    depth_m = static_cast<double>(raw);
                } else {
                    return std::nullopt;
                }
                samples.push_back(depth_m);
            }
        }

        if (samples.empty()) {
            return std::nullopt;
        }

        const size_t mid = samples.size() / 2;
        std::nth_element(samples.begin(), samples.begin() + static_cast<long>(mid), samples.end());
        const double depth_m = samples[mid];

        const double min_depth = get_parameter("position.depth_min_m").as_double();
        const double max_depth = get_parameter("position.depth_max_m").as_double();
        if (depth_m < min_depth || depth_m > max_depth) {
            return std::nullopt;
        }
        return depth_m;
    }

    std::optional<Eigen::Vector3d> estimatePositionCamera(
        float cx, float cy, float /*bbox_height*/,
        const rclcpp::Time& color_stamp)
    {
        if (!camera_info_received_ || !position_estimator_) {
            return std::nullopt;
        }

        Eigen::Vector3d pos_cam;
        if (position_mode_ == "depth") {
            const int u = static_cast<int>(std::lround(cx));
            const int v = static_cast<int>(std::lround(cy));
            const auto depth_m = sampleDepthMeters(u, v, color_stamp);
            if (!depth_m.has_value()) {
                return std::nullopt;
            }
            if (!position_estimator_->estimateFromDepth(cx, cy, depth_m.value(), pos_cam)) {
                return std::nullopt;
            }
        } else {
            const double h_smooth = getSmoothedBboxHeight();
            if (h_smooth <= 0.0) {
                return std::nullopt;
            }
            if (!position_estimator_->estimate(cx, cy, h_smooth, pos_cam)) {
                return std::nullopt;
            }
        }
        return pos_cam;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        // 每帧开始清空“当帧检测状态”，避免复用上一帧检测
        bool frame_has_fresh_detection = false;
        bool ball_found = false;
        double frame_raw_bbox_height = 0.0;
        std::string fresh_det_block_reason = "No YOLO box";

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image;
        if (frame.empty()) {
            RCLCPP_WARN(get_logger(), "Received empty image.");
            return;
        }

        const rclcpp::Time frame_time =
            (msg->header.stamp.sec == 0 && msg->header.stamp.nanosec == 0)
                ? this->now()
                : rclcpp::Time(msg->header.stamp);
        double timestamp_sec = frame_time.seconds();
        double frame_dt = 0.0;
        if (last_frame_time_.nanoseconds() > 0) {
            frame_dt = (frame_time - last_frame_time_).seconds();
        }

        // 1) YOLO 检测
        std::vector<Detection> detections;
        try {
            detections = yolo_detector_->detect(frame);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "YOLO detect failed: %s", e.what());
            detections.clear();
        }
        RCLCPP_DEBUG(get_logger(), "[YOLO RAW] %zu detections after conf filter", detections.size());

        const bool emergency_bypass =
            get_parameter("detection.emergency_bypass").as_bool();
        const double min_confidence =
            get_parameter("detection.min_confidence").as_double();

        // 2) 类别 / 置信度 / 跟踪过滤（默认开启；emergency_bypass=true 时跳过）
        std::vector<Detection> volleyball_detections;
        if (emergency_bypass) {
            volleyball_detections = detections;
        } else {
            volleyball_detections.reserve(detections.size());
            for (const auto& d : detections) {
                if (!isVolleyballClass(d.class_id)) {
                    continue;
                }
                if (d.confidence < static_cast<float>(min_confidence)) {
                    continue;
                }
                volleyball_detections.push_back(d);
            }
        }

        // 3) 选择最佳检测
        bool has_best = false;
        Detection best_det{};
        if (!volleyball_detections.empty()) {
            if (!emergency_bypass && multi_tracker_) {
                std::vector<cv::Point2f> centers;
                centers.reserve(volleyball_detections.size());
                for (const auto& d : volleyball_detections) {
                    centers.emplace_back(d.x + d.width * 0.5f, d.y + d.height * 0.5f);
                }
                const int best_idx = multi_tracker_->update(centers);
                if (best_idx >= 0 &&
                    best_idx < static_cast<int>(volleyball_detections.size())) {
                    best_det = volleyball_detections[static_cast<size_t>(best_idx)];
                    has_best = true;
                    ball_found = true;
                }
            } else {
                has_best = yolo_detector_->selectBestDetection(
                    volleyball_detections, "confidence", best_det);
                ball_found = has_best;
            }
        } else {
            fresh_det_block_reason = "No detection after filter";
        }

        if (has_best && !emergency_bypass && detection_filter_) {
            const float cx = best_det.x + best_det.width * 0.5f;
            const float cy = best_det.y + best_det.height * 0.5f;
            if (!detection_filter_->validate(cx, cy, best_det.confidence,
                                             frame.cols, frame.rows)) {
                has_best = false;
                ball_found = false;
                fresh_det_block_reason = "DetectionFilter rejected";
            }
        }

        // 4) 估计相机 3D -> TF 到 world
        std::optional<Eigen::Vector3d> measurement_world = std::nullopt;
        if (has_best) {
            float cx = best_det.x + best_det.width * 0.5f;
            float cy = best_det.y + best_det.height * 0.5f;
            frame_raw_bbox_height = static_cast<double>(best_det.height);
            const double h_ema_alpha = get_parameter("detection.h_ema_alpha").as_double();
            updateBboxHeightEma(static_cast<double>(best_det.height), h_ema_alpha);

            const auto pos_cam_opt = estimatePositionCamera(cx, cy, best_det.height, frame_time);
            if (pos_cam_opt.has_value()) {
                std::optional<Eigen::Vector3d> pos_world =
                    transformCameraToWorld(pos_cam_opt.value(), msg->header, frame_time);
                if (pos_world.has_value()) {
                    measurement_world = pos_world.value();
                    frame_has_fresh_detection = true;
                    last_detection_time_valid_ = true;
                    last_detection_time_ = frame_time;
                } else {
                    fresh_det_block_reason = "TF transform failed";
                }
            } else {
                fresh_det_block_reason =
                    (position_mode_ == "depth") ? "depth sample/estimate failed"
                                               : "bbox depth estimate failed";
            }
        }

        // 信任当帧 YOLO：不再因“位置变化过小”丢弃 measurement

        // 6) Smart Velocity Gating（物理速度门控）
        if (measurement_world.has_value() && has_last_measurement_world_ && frame_dt > 1e-4) {
            const Eigen::Vector3d v_temp = (measurement_world.value() - last_measurement_world_) / frame_dt;
            const double max_speed = get_parameter("detection.max_physical_speed").as_double();
            if (v_temp.norm() > max_speed) {
                RCLCPP_WARN(
                    get_logger(),
                    "Velocity gate triggered: |v_temp|=%.3f > max_physical_speed=%.3f, reset tracking",
                    v_temp.norm(), max_speed);
                measurement_world = std::nullopt;
                resetTrackingState("velocity gate (scene jump / video loop)");
            }
        }

        // 7) 卡尔曼滤波更新（含丢帧预测）
        const bool kf_was_initialized = ball_tracker_->isInitialized();
        ball_tracker_->updateWithMissing(measurement_world, timestamp_sec);
        if (!ball_tracker_->isInitialized() && kf_was_initialized) {
            resetTrackingState("kalman lost track");
        }
        if (measurement_world.has_value()) {
            last_measurement_world_ = measurement_world.value();
            has_last_measurement_world_ = true;
        }
        if (ball_found && !frame_has_fresh_detection) {
            RCLCPP_ERROR(
                get_logger(),
                "PARADOX: YOLO detected box(es) but fresh_det=false. Reason: %s",
                fresh_det_block_reason.c_str());
        }
        bool valid_track = ball_tracker_->isInitialized();
        RCLCPP_DEBUG(get_logger(), "Frame dt: %.4f, KF dt: %.4f, fresh_det: %s",
                     frame_dt, static_cast<double>(ball_tracker_->getDt()),
                     frame_has_fresh_detection ? "true" : "false");

        // 8) 发布 world pose + KF 状态，并进行世界系轨迹预测/可视化
        if (valid_track) {
            const Eigen::Vector3d pos_world = ball_tracker_->getPosition();
            const Eigen::Vector3d vel_world = ball_tracker_->getVelocity();
            RCLCPP_INFO_THROTTLE(
                get_logger(), *this->get_clock(), 500,
                "Tracking Ball at [%.3f, %.3f, %.3f]",
                pos_world.x(), pos_world.y(), pos_world.z());
            RCLCPP_INFO_THROTTLE(
                get_logger(), *this->get_clock(), 500,
                "Velocity: Vx=%.2f, Vy=%.2f, Vz=%.2f",
                vel_world.x(), vel_world.y(), vel_world.z());

            // /volleyball_pose（world frame）
            geometry_msgs::msg::PoseStamped pose;
            pose.header = msg->header;
            pose.header.frame_id = world_frame_id_;
            pose.pose.position.x = pos_world.x();
            pose.pose.position.y = pos_world.y();
            pose.pose.position.z = pos_world.z();
            pose.pose.orientation.w = 1.0;
            pose.pose.orientation.x = 0.0;
            pose.pose.orientation.y = 0.0;
            pose.pose.orientation.z = 0.0;
            pose_pub_->publish(pose);

            publishBallState(pos_world, vel_world);

            // 预测并发布 marker（world frame）
            publishWorldTrajectory(msg->header, pos_world, vel_world);
        }

        // 9) 生成并发布调试图像（对齐 Python: 用 volleyball_detections + best + kalman_state）
        if (get_parameter("debug.enable").as_bool()) {
            publishDebugImage(msg->header, frame, volleyball_detections,
                              has_best ? &best_det : nullptr,
                              frame_time);
        }

        last_frame_time_ = frame_time;
    }

    std::optional<Eigen::Vector3d> transformCameraToWorld(
        const Eigen::Vector3d& pos_cam,
        const std_msgs::msg::Header& header,
        const rclcpp::Time& now_time) const
    {
        if (!tf_buffer_) {
            return std::nullopt;
        }

        geometry_msgs::msg::PointStamped p_in;
        p_in.header.stamp = header.stamp;
        // 兼容：视频/仿真可能不给 frame_id
        p_in.header.frame_id =
            (!header.frame_id.empty()) ? header.frame_id : camera_frame_id_;

        // 若时间戳是 0（极端情况），使用当前时间
        if (p_in.header.stamp.sec == 0 && p_in.header.stamp.nanosec == 0) {
            p_in.header.stamp = now_time;
        }

        p_in.point.x = pos_cam.x();
        p_in.point.y = pos_cam.y();
        p_in.point.z = pos_cam.z();

        geometry_msgs::msg::PointStamped p_out;
        try {
            // 等待很短，保证实时性
            const tf2::Duration timeout = tf2::durationFromSec(0.05);
            p_out = tf_buffer_->transform(p_in, world_frame_id_, timeout);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(get_logger(), "TF transform camera->world failed: %s", ex.what());
            return std::nullopt;
        }

        return Eigen::Vector3d(p_out.point.x, p_out.point.y, p_out.point.z);
    }

    void updateBboxHeightEma(double bbox_height, double alpha)
    {
        if (bbox_height <= 0.0) {
            return;
        }
        alpha = std::max(0.0, std::min(1.0, alpha));

        if (!has_bbox_height_ema_) {
            bbox_height_ema_ = bbox_height;
            has_bbox_height_ema_ = true;
            return;
        }
        // EMA: h_s = alpha * h_raw + (1-alpha) * h_prev
        bbox_height_ema_ = alpha * bbox_height + (1.0 - alpha) * bbox_height_ema_;
    }

    double getSmoothedBboxHeight() const
    {
        if (!has_bbox_height_ema_) {
            return 0.0;
        }
        return bbox_height_ema_;
    }

    void resetTrackingState(const char* reason)
    {
        RCLCPP_INFO(get_logger(), "Reset tracking state: %s", reason);
        has_last_measurement_world_ = false;
        has_bbox_height_ema_ = false;
        if (detection_filter_) {
            detection_filter_->reset();
        }
        if (multi_tracker_) {
            multi_tracker_->reset();
        }
        if (ball_tracker_) {
            ball_tracker_->reset();
        }
    }

    void publishBallState(const Eigen::Vector3d& pos,
                          const Eigen::Vector3d& vel)
    {
        (void)vel; // 当前话题仅发布位置
        // 简单实现：/ball_state 使用 geometry_msgs/Point，仅携带当前位置 (x,y,z)
        geometry_msgs::msg::Point msg;
        msg.x = pos.x();
        msg.y = pos.y();
        msg.z = pos.z();
        ball_state_pub_->publish(msg);
    }

    void publishBallPrediction(const Eigen::Vector3d& landing_pos,
                               double time_to_land)
    {
        // /ball_prediction: x,y 为落地点，z 为 time_to_land
        geometry_msgs::msg::Point msg;
        msg.x = landing_pos.x();
        msg.y = landing_pos.y();
        msg.z = time_to_land;
        ball_prediction_pub_->publish(msg);
    }

    void publishWorldTrajectory(const std_msgs::msg::Header& header,
                                 const Eigen::Vector3d& pos_world,
                                 const Eigen::Vector3d& vel_world)
    {
        if (!trajectory_predictor_ || !ball_tracker_->isInitialized()) {
            return;
        }

        double t_land = 0.0;
        Eigen::Vector3d landing_pos(0.0, 0.0, 0.0);
        std::vector<Eigen::Vector3d> path_points;

        if (!trajectory_predictor_->predictLanding(pos_world, vel_world, t_land, landing_pos, path_points)) {
            return;
        }

        publishBallPrediction(landing_pos, t_land);

        // world marker
        visualization_msgs::msg::MarkerArray array;

        visualization_msgs::msg::Marker line;
        line.header.stamp = header.stamp;
        line.header.frame_id = world_frame_id_;
        line.ns = "volleyball_trajectory";
        line.id = 0;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.02;

        line.color.r = 1.0f;
        line.color.g = 0.2f;
        line.color.b = 0.0f;
        line.color.a = 1.0f;

        for (const auto& p : path_points) {
            geometry_msgs::msg::Point pt;
            pt.x = p.x();
            pt.y = p.y();
            pt.z = p.z();
            line.points.push_back(pt);
        }

        visualization_msgs::msg::Marker sphere;
        sphere.header.stamp = header.stamp;
        sphere.header.frame_id = world_frame_id_;
        sphere.ns = "volleyball_landing";
        sphere.id = 1;
        sphere.type = visualization_msgs::msg::Marker::SPHERE;
        sphere.action = visualization_msgs::msg::Marker::ADD;

        double diameter = get_parameter("volleyball.diameter").as_double();
        if (diameter <= 0.0) {
            const double r = get_parameter("volleyball.real_radius").as_double();
            diameter = (r > 0.0) ? (2.0 * r) : 0.225;
        }

        sphere.scale.x = diameter;
        sphere.scale.y = diameter;
        sphere.scale.z = diameter;
        sphere.color.r = 0.0f;
        sphere.color.g = 1.0f;
        sphere.color.b = 0.0f;
        sphere.color.a = 1.0f;

        sphere.pose.orientation.w = 1.0;
        sphere.pose.position.x = landing_pos.x();
        sphere.pose.position.y = landing_pos.y();
        sphere.pose.position.z = landing_pos.z();

        array.markers.push_back(line);
        array.markers.push_back(sphere);
        traj_pub_->publish(array);
    }

    void publishDebugImage(const std_msgs::msg::Header& header,
                           const cv::Mat& frame,
                           const std::vector<Detection>& detections,
                           const Detection* best_det,
                           const rclcpp::Time& now_time)
    {
        // Atomic fix: always draw on a deep copy that is explicitly published.
        cv::Mat debug_img = frame.clone();

        // 对齐 Python: 绘制所有检测结果（黄色）
        for (const auto& det : detections) {
            int x1 = (int)det.x;
            int y1 = (int)det.y;
            int x2 = (int)(det.x + det.width);
            int y2 = (int)(det.y + det.height);
            cv::rectangle(debug_img, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 255), 2);
            int cx = (int)(det.x + det.width * 0.5f);
            int cy = (int)(det.y + det.height * 0.5f);
            cv::circle(debug_img, cv::Point(cx, cy), 3, cv::Scalar(0, 255, 255), -1);

            // label: class_name: conf
            std::string cls = yolo_detector_
                                ? yolo_detector_->getClassName(det.class_id)
                                : "class";
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s: %.2f", cls.c_str(), (double)det.confidence);
            cv::putText(debug_img, buf, cv::Point(x1, y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }

        // 对齐 Python: 绘制最佳检测（绿色）
        if (best_det) {
            int x1 = static_cast<int>(best_det->x);
            int y1 = static_cast<int>(best_det->y);
            int x2 = static_cast<int>(best_det->x + best_det->width);
            int y2 = static_cast<int>(best_det->y + best_det->height);
            cv::rectangle(debug_img, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 3);
            int cx = (int)(best_det->x + best_det->width * 0.5f);
            int cy = (int)(best_det->y + best_det->height * 0.5f);
            cv::circle(debug_img, cv::Point(cx, cy), 5, cv::Scalar(0, 255, 0), -1);

            // Big Red Circle around detected ball
            const int radius = std::max(20, static_cast<int>(best_det->height * 0.8f));
            cv::circle(debug_img, cv::Point(cx, cy), radius, cv::Scalar(0, 0, 255), 4);
        }

        // 无目标时也持续显示搜索状态（UI 始终可见）
        if (!best_det) {
            cv::putText(debug_img,
                        "Searching...",
                        cv::Point(10, 110),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 200, 255), 2);
        }

        // Bright green system status
        const char* mode_label =
            (position_mode_ == "depth") ? "MODE: RGB-D" : "MODE: RGB/BBOX";
        cv::putText(debug_img,
                    mode_label,
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);
        cv::putText(debug_img,
                    "SYSTEM ACTIVE",
                    cv::Point(10, 55),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2);

        // 绘制状态信息（对齐 Python）
        int info_y = 80;
        cv::putText(debug_img,
                    "Detections: " + std::to_string(detections.size()),
                    cv::Point(10, info_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
        info_y += 25;

        if (ball_tracker_->isInitialized()) {
            std::string msg_text = "KF: OK, Missing: " + std::to_string(ball_tracker_->getMissingFrames());
            cv::putText(debug_img,
                        msg_text,
                        cv::Point(10, info_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(debug_img,
                        "KF: Not initialized",
                        cv::Point(10, info_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 2);
        }
        info_y += 25;

        // Current Z Depth / Height in world frame (from KF state z)
        double current_z = 0.0;
        if (ball_tracker_->isInitialized()) {
            current_z = ball_tracker_->getPosition().z();
        }
        char zbuf[96];
        std::snprintf(zbuf, sizeof(zbuf), "Current Z Depth: %.3f m", current_z);
        cv::putText(debug_img,
                    zbuf,
                    cv::Point(10, info_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
        info_y += 25;

        // 始终显示 FPS，避免无检测时 UI 消失
        double fps = 0.0;
        if (last_detection_time_valid_) {
            double dt = (now_time - last_detection_time_).seconds();
            if (dt > 1e-6) {
                fps = 1.0 / dt;
            }
        }
        char buf[64];
        std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
        cv::putText(debug_img,
                    buf,
                    cv::Point(10, info_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);

        // FORCE PUBLISH using the exact drawn Mat
        auto out_msg = cv_bridge::CvImage(header, "bgr8", debug_img).toImageMsg();
        debug_pub_->publish(*out_msg);
    }

private:
    std::unique_ptr<YOLODetector>   yolo_detector_;
    std::unique_ptr<BallTracker>    ball_tracker_;
    std::unique_ptr<DetectionFilter> detection_filter_;
    std::unique_ptr<MultiDetectionTracker> multi_tracker_;
    std::unique_ptr<TrajectoryPredictor> trajectory_predictor_;
    std::unique_ptr<BallPositionEstimator> position_estimator_;

    std::string world_frame_id_;
    std::string camera_frame_id_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr traj_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr ball_state_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr ball_prediction_pub_;

    rclcpp::Time last_frame_time_;
    rclcpp::Time last_detection_time_;
    bool last_detection_time_valid_;

    bool camera_info_received_{false};
    cv::Mat camera_matrix_;
    cv::Mat distortion_coeffs_;
    double bbox_height_ema_{0.0};
    bool has_bbox_height_ema_{false};
    Eigen::Vector3d last_measurement_world_{0.0, 0.0, 0.0};
    bool has_last_measurement_world_{false};

    std::string position_mode_{"bbox"};
    cv::Mat latest_depth_;
    std::string latest_depth_encoding_;
    rclcpp::Time latest_depth_stamp_{0, 0, RCL_ROS_TIME};
    bool has_latest_depth_{false};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}