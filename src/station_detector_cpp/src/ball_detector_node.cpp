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
#include <deque>
#include <numeric>

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
        declare_parameter<int>("detection.min_consistent_detections", 2);
        declare_parameter<int>("detection.max_detections", 3);
        declare_parameter<std::string>("detection.selection_method", "confidence");
        declare_parameter<double>("detection.min_confidence", 0.3);
        declare_parameter<bool>("detection.use_tracker", true);

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

        // 1) YOLO 检测
        std::vector<Detection> detections;
        try {
            detections = yolo_detector_->detect(frame);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "YOLO detect failed: %s", e.what());
            detections.clear();
        }

        // 2) 过滤出排球（按 COCO 名称匹配 yolo.volleyball_classes）
        std::vector<Detection> volleyball_detections;
        volleyball_detections.reserve(detections.size());
        for (const auto& d : detections) {
            if (isVolleyballClass(d.class_id)) {
                volleyball_detections.push_back(d);
            }
        }

        // 3) 过滤低置信度（对齐 Python detection.min_confidence）
        const float min_confidence =
            static_cast<float>(get_parameter("detection.min_confidence").as_double());
        std::vector<Detection> filtered;
        filtered.reserve(volleyball_detections.size());
        for (const auto& d : volleyball_detections) {
            if (d.confidence >= min_confidence) {
                filtered.push_back(d);
            }
        }

        // 4) 选择最佳检测（对齐 Python: 可选 MultiDetectionTracker，否则按 selection_method 选）
        bool has_best = false;
        Detection best_det{};
        if (!filtered.empty()) {
            const std::string method = get_parameter("detection.selection_method").as_string();

            if (multi_tracker_) {
                // Python tracker.update(filtered) 返回 best detection
                // 我们用中心点列表回传索引
                std::vector<cv::Point2f> centers;
                centers.reserve(filtered.size());
                for (const auto& d : filtered) {
                    centers.emplace_back(d.x + d.width * 0.5f, d.y + d.height * 0.5f);
                }
                int idx = multi_tracker_->update(centers);
                if (idx >= 0 && idx < (int)filtered.size()) {
                    best_det = filtered[idx];
                    has_best = true;
                }
            } else {
                has_best = yolo_detector_->selectBestDetection(filtered, method, best_det);
            }
        }

        // 5) 验证检测 -> 估计相机 3D -> TF 到 world（对齐：YOLO -> Estimator -> TF -> KF）
        std::optional<Eigen::Vector3d> measurement_world = std::nullopt;
        if (has_best) {
            float cx = best_det.x + best_det.width * 0.5f;
            float cy = best_det.y + best_det.height * 0.5f;
            updateBboxHeightHistory(static_cast<double>(best_det.height));

            bool ok = detection_filter_->validate(
                cx, cy,
                best_det.confidence,
                frame.cols, frame.rows);

            if (ok) {
                if (camera_info_received_ && position_estimator_) {
                    Eigen::Vector3d pos_cam;
                    if (position_estimator_->estimate(cx, cy, getSmoothedBboxHeight(), pos_cam)) {
                        const double min_depth = get_parameter("volleyball.min_depth").as_double();
                        const double max_depth = get_parameter("volleyball.max_depth").as_double();
                        if (pos_cam.z() >= min_depth && pos_cam.z() <= max_depth) {
                            // 相机坐标系 -> world 坐标系
                            std::optional<Eigen::Vector3d> pos_world =
                                transformCameraToWorld(pos_cam, msg->header, now_time);
                            if (pos_world.has_value()) {
                                measurement_world = pos_world.value();
                            }
                        }
                    }
                }

                if (measurement_world.has_value()) {
                    last_detection_time_valid_ = true;
                    last_detection_time_ = now_time;
                }
            }
        }

        // 6) 卡尔曼滤波更新（含丢帧预测）
        ball_tracker_->updateWithMissing(measurement_world, timestamp_sec);
        bool valid_track = ball_tracker_->isInitialized();

        // 7) 发布 world pose + KF 状态，并进行世界系轨迹预测/可视化
        if (valid_track) {
            const Eigen::Vector3d pos_world = ball_tracker_->getPosition();
            const Eigen::Vector3d vel_world = ball_tracker_->getVelocity();
            RCLCPP_INFO_THROTTLE(
                get_logger(), *this->get_clock(), 500,
                "Tracking Ball at [%.3f, %.3f, %.3f]",
                pos_world.x(), pos_world.y(), pos_world.z());

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

        // 8) 生成并发布调试图像（对齐 Python: 用 volleyball_detections + best + kalman_state）
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

    void updateBboxHeightHistory(double bbox_height)
    {
        if (bbox_height <= 0.0) {
            return;
        }
        bbox_height_history_.push_back(bbox_height);
        while (bbox_height_history_.size() > kBboxHeightWindowSize) {
            bbox_height_history_.pop_front();
        }
    }

    double getSmoothedBboxHeight() const
    {
        if (bbox_height_history_.empty()) {
            return 0.0;
        }
        const double sum = std::accumulate(bbox_height_history_.begin(),
                                           bbox_height_history_.end(), 0.0);
        return sum / static_cast<double>(bbox_height_history_.size());
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
        cv::Mat debug = frame.clone();

        // 对齐 Python: 绘制所有检测结果（黄色）
        for (const auto& det : detections) {
            int x1 = (int)det.x;
            int y1 = (int)det.y;
            int x2 = (int)(det.x + det.width);
            int y2 = (int)(det.y + det.height);
            cv::rectangle(debug, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 255), 2);
            int cx = (int)(det.x + det.width * 0.5f);
            int cy = (int)(det.y + det.height * 0.5f);
            cv::circle(debug, cv::Point(cx, cy), 3, cv::Scalar(0, 255, 255), -1);

            // label: class_name: conf
            std::string cls = yolo_detector_
                                ? yolo_detector_->getClassName(det.class_id)
                                : "class";
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%s: %.2f", cls.c_str(), (double)det.confidence);
            cv::putText(debug, buf, cv::Point(x1, y1 - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 1);
        }

        // 对齐 Python: 绘制最佳检测（绿色）
        if (best_det) {
            int x1 = static_cast<int>(best_det->x);
            int y1 = static_cast<int>(best_det->y);
            int x2 = static_cast<int>(best_det->x + best_det->width);
            int y2 = static_cast<int>(best_det->y + best_det->height);
            cv::rectangle(debug, cv::Point(x1, y1), cv::Point(x2, y2),
                          cv::Scalar(0, 255, 0), 3);
            int cx = (int)(best_det->x + best_det->width * 0.5f);
            int cy = (int)(best_det->y + best_det->height * 0.5f);
            cv::circle(debug, cv::Point(cx, cy), 5, cv::Scalar(0, 255, 0), -1);
        }

        // 绘制状态信息（对齐 Python）
        int info_y = 20;
        cv::putText(debug,
                    "Detections: " + std::to_string(detections.size()),
                    cv::Point(10, info_y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255, 255, 255), 2);
        info_y += 25;

        if (ball_tracker_->isInitialized()) {
            std::string msg_text = "KF: OK, Missing: " + std::to_string(ball_tracker_->getMissingFrames());
            cv::putText(debug,
                        msg_text,
                        cv::Point(10, info_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2);
        } else {
            cv::putText(debug,
                        "KF: Not initialized",
                        cv::Point(10, info_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 0, 255), 2);
        }
        info_y += 25;

        if (get_parameter("debug.show_fps").as_bool()) {
            // 对齐 Python: fps 使用 last_detection_time（仅在有有效检测时更新）
            double fps = 0.0;
            if (last_detection_time_valid_) {
                double dt = (now_time - last_detection_time_).seconds();
                if (dt > 1e-6) {
                    fps = 1.0 / dt;
                }
            }
            char buf[64];
            std::snprintf(buf, sizeof(buf), "FPS: %.1f", fps);
            cv::putText(debug,
                        buf,
                        cv::Point(10, info_y),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(255, 255, 255), 2);
        }

        cv_bridge::CvImage out_msg;
        out_msg.header = header;
        out_msg.encoding = "bgr8";
        out_msg.image = debug;

        debug_pub_->publish(*out_msg.toImageMsg());
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
    static constexpr size_t kBboxHeightWindowSize = 5;
    std::deque<double> bbox_height_history_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}