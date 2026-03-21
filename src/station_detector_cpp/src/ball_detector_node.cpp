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

        // QoS：实时性优先
        rclcpp::QoS qos(rclcpp::KeepLast(1));
        qos.best_effort();

        image_sub_ = create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", qos,
            std::bind(&BallDetectorNode::imageCallback, this, _1));

        camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera_info", 10,
            std::bind(&BallDetectorNode::cameraInfoCallback, this, _1));

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

        last_frame_time_ = now();
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

        detection_filter_ = std::make_unique<DetectionFilter>(
            max_jump, min_consistent, 5);

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

        const auto now_time = this->now();
        double timestamp_sec = now_time.seconds();
        const double frame_dt = (now_time - last_frame_time_).seconds();

        // 1) YOLO 检测
        std::vector<Detection> detections;
        try {
            detections = yolo_detector_->detect(frame);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "YOLO detect failed: %s", e.what());
            detections.clear();
        }
        {
            int raw_cnt = 0;
            for (const auto& d : detections) {
                if (d.confidence > 0.1f) {
                    ++raw_cnt;
                }
            }
            RCLCPP_INFO(get_logger(), "[YOLO RAW] Detected %d boxes with conf > 0.1", raw_cnt);
        }

        // 2) EMERGENCY BYPASS: 禁用所有内部过滤链，YOLO 有框即认为找到球
        std::vector<Detection> volleyball_detections = detections;
        std::vector<Detection> filtered = detections;

        // 3) 选择最佳检测（仅按置信度，不做距离/速度/大小等检查）
        bool has_best = false;
        Detection best_det{};
        if (!filtered.empty()) {
            has_best = yolo_detector_->selectBestDetection(filtered, "confidence", best_det);
            ball_found = has_best;
        } else {
            fresh_det_block_reason = "No YOLO box after detect()";
        }

        // 5) 验证检测 -> 估计相机 3D -> TF 到 world（对齐：YOLO -> Estimator -> TF -> KF）
        std::optional<Eigen::Vector3d> measurement_world = std::nullopt;
        if (has_best) {
            float cx = best_det.x + best_det.width * 0.5f;
            float cy = best_det.y + best_det.height * 0.5f;
            frame_raw_bbox_height = static_cast<double>(best_det.height);
            frame_has_fresh_detection = true;
            RCLCPP_INFO(get_logger(), "Raw bbox.height: %.3f", frame_raw_bbox_height);
            const double h_ema_alpha = get_parameter("detection.h_ema_alpha").as_double();
            updateBboxHeightEma(static_cast<double>(best_det.height), h_ema_alpha);

            if (camera_info_received_ && position_estimator_) {
                Eigen::Vector3d pos_cam;
                    if (position_estimator_->estimate(cx, cy, getSmoothedBboxHeight(), pos_cam)) {
                    // EMERGENCY BYPASS: 禁用深度范围过滤，直接走 TF
                    std::optional<Eigen::Vector3d> pos_world =
                        transformCameraToWorld(pos_cam, msg->header, now_time);
                    if (pos_world.has_value()) {
                        measurement_world = pos_world.value();
                    } else {
                        fresh_det_block_reason = "TF transform failed";
                    }
                } else {
                    fresh_det_block_reason = "position_estimator->estimate failed";
                }
            } else {
                fresh_det_block_reason = "camera_info/position_estimator not ready";
            }

            if (measurement_world.has_value()) {
                frame_has_fresh_detection = true;
                last_detection_time_valid_ = true;
                last_detection_time_ = now_time;
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
                    "Velocity gate triggered: |v_temp|=%.3f > max_physical_speed=%.3f, use KF predict only",
                    v_temp.norm(), max_speed);
                measurement_world = std::nullopt;
            }
        }

        // 7) 卡尔曼滤波更新（含丢帧预测）
        ball_tracker_->updateWithMissing(measurement_world, timestamp_sec);
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
        RCLCPP_INFO(get_logger(), "Frame dt: %.4f, KF dt: %.4f, fresh_det: %s",
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
            RCLCPP_INFO(
                get_logger(),
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
                              now_time);
        }

        last_frame_time_ = now_time;
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
        cv::putText(debug_img,
                    "SYSTEM ACTIVE",
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9,
                    cv::Scalar(0, 255, 0), 2);

        // 绘制状态信息（对齐 Python）
        int info_y = 20;
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
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}