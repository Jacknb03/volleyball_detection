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
        declare_parameter<int>("detection.min_consistent_detections", 2);
        declare_parameter<int>("detection.max_detections", 3);
        declare_parameter<std::string>("detection.selection_method", "confidence");
        declare_parameter<double>("detection.min_confidence", 0.3);
        declare_parameter<bool>("detection.use_tracker", true);

        declare_parameter<double>("trajectory.prediction_time", 0.5);
        declare_parameter<int>("trajectory.num_points", 10);

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
        trajectory_predictor_ = std::make_unique<TrajectoryPredictor>();

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
            diameter = 0.225;
        }
        position_estimator_ = std::make_unique<BallPositionEstimator>(
            fx, fy, cx, cy, diameter);
    }

    static std::string toLower(std::string s)
    {
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return (char)std::tolower(c); });
        return s;
    }

    static const std::vector<std::string>& coco80Names()
    {
        static const std::vector<std::string> names = YOLODetector::coco80Names();
        return names;
    }

    bool isVolleyballClass(int class_id) const
    {
        const auto& names = coco80Names();
        if (class_id < 0 || class_id >= (int)names.size()) {
            return false;
        }
        const std::string cls_name = toLower(names[class_id]);

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

        // 5) 验证检测（对齐 Python DetectionValidator.validate）
        std::optional<cv::Point2f> measurement = std::nullopt;
        if (has_best) {
            float cx = best_det.x + best_det.width * 0.5f;
            float cy = best_det.y + best_det.height * 0.5f;
            last_bbox_height_ = static_cast<double>(best_det.height);

            bool ok = detection_filter_->validate(
                cx, cy,
                best_det.confidence,
                frame.cols, frame.rows);

            if (ok) {
                measurement = cv::Point2f(cx, cy);
                last_detection_time_valid_ = true;
                last_detection_time_ = now_time;
            } else {
                measurement = std::nullopt;
            }
        }

        // 6) 卡尔曼滤波更新（含丢帧预测）
        cv::Point2f est_pos = ball_tracker_->updateWithMissing(measurement, timestamp_sec);
        bool valid_track = ball_tracker_->isInitialized();

        // 7) 估计 3D pose（对齐 Python _estimate_3d_pose: 固定 depth=2.0 + 内参反投影 + depth范围限制）
        if (camera_info_received_ && valid_track) {
            std::optional<geometry_msgs::msg::PoseStamped> pose =
                estimate3dPose(est_pos, msg->header);
            if (pose.has_value()) {
                pose_pub_->publish(*pose);

                // 基于 3D 位姿和历史位姿估计 3D 速度，并进行落点预测
                Eigen::Vector3d pos3(
                    pose->pose.position.x,
                    pose->pose.position.y,
                    pose->pose.position.z);

                Eigen::Vector3d vel3(0.0, 0.0, 0.0);
                bool have_velocity = false;
                if (last_pose_valid_) {
                    double dt = timestamp_sec - last_pose_time_;
                    if (dt > 1e-4) {
                        vel3 = (pos3 - last_pose_) / dt;
                        have_velocity = true;
                    }
                }

                last_pose_ = pos3;
                last_pose_time_ = timestamp_sec;
                last_pose_valid_ = true;

                // 发布当前状态（位置）
                publishBallState(pos3, vel3);

                // 轨迹预测（仅在已有速度估计时进行）
                if (have_velocity) {
                    double t_land = 0.0;
                    Eigen::Vector2d landing_xy(0.0, 0.0);
                    if (trajectory_predictor_->predictLanding(pos3, vel3, t_land, landing_xy)) {
                        publishBallPrediction(landing_xy, t_land);
                    }
                }

                if (get_parameter("debug.enable").as_bool()) {
                    publishTrajectory(msg->header);
                }
            }
        }

        // 8) 生成并发布调试图像（对齐 Python: 用 volleyball_detections + best + kalman_state）
        if (get_parameter("debug.enable").as_bool()) {
            publishDebugImage(msg->header, frame, volleyball_detections,
                              has_best ? &best_det : nullptr,
                              valid_track ? &est_pos : nullptr,
                              now_time);
        }

        last_frame_time_ = now_time;
    }

    std::optional<geometry_msgs::msg::PoseStamped> estimate3dPose(
        const cv::Point2f& image_center,
        const std_msgs::msg::Header& header) const
    {
        if (!camera_info_received_ || !position_estimator_) {
            return std::nullopt;
        }

        // 防止 bbox 过小导致深度爆炸
        if (last_bbox_height_ < 10.0) {
            return std::nullopt;
        }

        const double min_depth = get_parameter("volleyball.min_depth").as_double();
        const double max_depth = get_parameter("volleyball.max_depth").as_double();

        Eigen::Vector3d pos_cam;
        if (!position_estimator_->estimate(
                static_cast<double>(image_center.x),
                static_cast<double>(image_center.y),
                last_bbox_height_,
                pos_cam)) {
            return std::nullopt;
        }

        const double X = pos_cam.x();
        const double Y = pos_cam.y();
        const double Z = pos_cam.z();

        if (Z < min_depth || Z > max_depth) {
            return std::nullopt;
        }

        geometry_msgs::msg::PoseStamped pose;
        pose.header = header;
        pose.header.frame_id = header.frame_id;
        pose.pose.position.x = X;
        pose.pose.position.y = Y;
        pose.pose.position.z = Z;
        pose.pose.orientation.w = 1.0;
        pose.pose.orientation.x = 0.0;
        pose.pose.orientation.y = 0.0;
        pose.pose.orientation.z = 0.0;

        return pose;
    }

    void publishBallState(const Eigen::Vector3d& pos,
                          const Eigen::Vector3d& vel)
    {
        // 简单实现：/ball_state 使用 geometry_msgs/Point，仅携带当前位置 (x,y,z)
        geometry_msgs::msg::Point msg;
        msg.x = pos.x();
        msg.y = pos.y();
        msg.z = pos.z();
        ball_state_pub_->publish(msg);
    }

    void publishBallPrediction(const Eigen::Vector2d& landing_xy,
                               double time_to_land)
    {
        // /ball_prediction: x,y 为落地点，z 为 time_to_land
        geometry_msgs::msg::Point msg;
        msg.x = landing_xy.x();
        msg.y = landing_xy.y();
        msg.z = time_to_land;
        ball_prediction_pub_->publish(msg);
    }

    void publishTrajectory(const std_msgs::msg::Header& header)
    {
        if (!ball_tracker_->isInitialized()) {
            return;
        }

        double pred_time =
            get_parameter("trajectory.prediction_time").as_double();
        int num_points =
            get_parameter("trajectory.num_points").as_int();

        auto future_positions = ball_tracker_->predictFuture(
            static_cast<float>(pred_time), num_points);

        if (future_positions.empty()) {
            return;
        }

        visualization_msgs::msg::MarkerArray array;

        visualization_msgs::msg::Marker line;
        line.header = header;
        line.ns = "volleyball_trajectory";
        line.id = 0;
        line.type = visualization_msgs::msg::Marker::LINE_STRIP;
        line.action = visualization_msgs::msg::Marker::ADD;
        line.scale.x = 0.02; // 对齐 Python: 0.02

        line.color.r = 1.0f;
        line.color.g = 0.0f;
        line.color.b = 0.0f;
        line.color.a = 1.0f;

        for (const auto& p : future_positions) {
            geometry_msgs::msg::Point pt;
            // 对齐 Python: 使用图像坐标/100
            pt.x = static_cast<double>(p.x) / 100.0;
            pt.y = static_cast<double>(p.y) / 100.0;
            pt.z = 0.0;
            line.points.push_back(pt);
        }

        array.markers.push_back(line);
        traj_pub_->publish(array);
    }

    void publishDebugImage(const std_msgs::msg::Header& header,
                           const cv::Mat& frame,
                           const std::vector<Detection>& detections,
                           const Detection* best_det,
                           const cv::Point2f* track_pos,
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
            const auto& names = coco80Names();
            std::string cls = (det.class_id >= 0 && det.class_id < (int)names.size())
                                ? names[det.class_id]
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

        // 对齐 Python: 绘制卡尔曼滤波结果（红色） + 速度箭头 + 预测点
        if (track_pos && ball_tracker_->isInitialized()) {
            cv::Point2f vel = ball_tracker_->getVelocity();
            int kf_cx = (int)track_pos->x;
            int kf_cy = (int)track_pos->y;
            cv::circle(debug,
                       cv::Point(kf_cx, kf_cy),
                       8, cv::Scalar(0, 0, 255), 2);

            // velocity arrow (vel_scale=10.0)
            const float vel_scale = 10.0f;
            int end_x = (int)(kf_cx + vel.x * vel_scale);
            int end_y = (int)(kf_cy + vel.y * vel_scale);
            cv::arrowedLine(debug, cv::Point(kf_cx, kf_cy), cv::Point(end_x, end_y),
                            cv::Scalar(0, 0, 255), 2, cv::LINE_AA, 0, 0.3);

            if (get_parameter("debug.draw_trajectory").as_bool()) {
                double pred_time =
                    get_parameter("trajectory.prediction_time").as_double();
                int num_points =
                    get_parameter("trajectory.num_points").as_int();

                auto future_positions = ball_tracker_->predictFuture(
                    static_cast<float>(pred_time), num_points);

                for (size_t i = 0; i < future_positions.size(); ++i) {
                    const auto& p = future_positions[i];
                    int px = static_cast<int>(p.x);
                    int py = static_cast<int>(p.y);
                    if (px < 0 || py < 0 || px >= debug.cols || py >= debug.rows) {
                        continue;
                    }
                    float alpha = static_cast<float>(i) /
                                  static_cast<float>(future_positions.size());
                    cv::Scalar color((int)(255 * alpha), 0, (int)(255 * (1.0f - alpha)));
                    cv::circle(debug, cv::Point(px, py), 2, color, -1);
                }
            }
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
    double last_bbox_height_{0.0};

    // 用于估计 3D 速度的历史位姿
    Eigen::Vector3d last_pose_{0.0, 0.0, 0.0};
    double last_pose_time_{0.0};
    bool last_pose_valid_{false};
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<BallDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}