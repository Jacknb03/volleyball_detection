#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "station_detector_cpp/msg/volleyball_intercept.hpp"
#include "volleyball_msgs/msg/stewart_control.hpp"

namespace {

using station_detector_cpp::msg::VolleyballIntercept;
using volleyball_msgs::msg::StewartControl;

constexpr uint8_t kEmergencyStop = 1;
constexpr uint8_t kNormal = 0;

double radOrDeg(double rad, bool use_degrees)
{
    return use_degrees ? rad * 180.0 / M_PI : rad;
}

void aimFromVector(double vx,
                   double vy,
                   double vz,
                   double default_roll,
                   double& roll,
                   double& pitch,
                   double& yaw)
{
    const double dist_sq = vx * vx + vy * vy + vz * vz;
    if (dist_sq < 1e-8) {
        roll = default_roll;
        pitch = 0.0;
        yaw = 0.0;
        return;
    }

    roll = default_roll;
    yaw = std::atan2(vy, vx);
    const double horiz = std::hypot(vx, vy);
    pitch = std::atan2(-vz, horiz);
}

// 将点投影到以 center 为心、半径 radius 的球面上（沿 center→point 方向）
void projectOntoSphereShell(double& x,
                            double& y,
                            double& z,
                            double cx,
                            double cy,
                            double cz,
                            double radius)
{
    if (radius <= 0.0) {
        return;
    }

    const double dx = x - cx;
    const double dy = y - cy;
    const double dz = z - cz;
    const double dist = std::hypot(dx, std::hypot(dy, dz));
    if (dist < 1e-8) {
        x = cx;
        y = cy;
        z = cz + radius;
        return;
    }

    const double scale = radius / dist;
    x = cx + dx * scale;
    y = cy + dy * scale;
    z = cz + dz * scale;
}

// 限制在球体内部：|p-center|<=R
void clampInsideSphere(double& x,
                       double& y,
                       double& z,
                       double cx,
                       double cy,
                       double cz,
                       double radius)
{
    if (radius <= 0.0) {
        return;
    }

    const double dx = x - cx;
    const double dy = y - cy;
    const double dz = z - cz;
    const double dist_sq = dx * dx + dy * dy + dz * dz;
    if (dist_sq <= radius * radius) {
        return;
    }

    const double dist = std::sqrt(dist_sq);
    const double scale = radius / dist;
    x = cx + dx * scale;
    y = cy + dy * scale;
    z = cz + dz * scale;
}

StewartControl makeCommand(const VolleyballIntercept& intercept,
                           uint8_t emergency_stop,
                           double default_roll,
                           bool use_ball_xyz,
                           double fallback_x,
                           double fallback_y,
                           double fallback_z,
                           double center_x,
                           double center_y,
                           double center_z,
                           double workspace_radius,
                           bool constrain_sphere,
                           const std::string& sphere_mode,
                           double max_z,
                           bool aim_from_ball,
                           bool use_degrees)
{
    StewartControl cmd;
    cmd.emergency_stop = emergency_stop;

    double x = fallback_x;
    double y = fallback_y;
    double z = fallback_z;

    if (use_ball_xyz) {
        x = intercept.position.x;
        y = intercept.position.y;
        z = intercept.position.z;
    }

    if (constrain_sphere && workspace_radius > 0.0) {
        if (sphere_mode == "inside") {
            clampInsideSphere(x, y, z, center_x, center_y, center_z, workspace_radius);
        } else {
            // 默认 on_shell：低轨道球面，沿 center→目标 方向落在半径 R 上
            projectOntoSphereShell(
                x, y, z, center_x, center_y, center_z, workspace_radius);
        }
    }

    if (max_z > -900.0) {
        z = std::min(z, max_z);
    }

    double roll = default_roll;
    double pitch = 0.0;
    double yaw = 0.0;

    if (aim_from_ball) {
        aimFromVector(
            intercept.position.x - center_x,
            intercept.position.y - center_y,
            intercept.position.z - center_z,
            default_roll,
            roll,
            pitch,
            yaw);
    }

    cmd.x = static_cast<float>(x);
    cmd.y = static_cast<float>(y);
    cmd.z = static_cast<float>(z);
    cmd.roll = static_cast<float>(radOrDeg(roll, use_degrees));
    cmd.pitch = static_cast<float>(radOrDeg(pitch, use_degrees));
    cmd.yaw = static_cast<float>(radOrDeg(yaw, use_degrees));
    return cmd;
}

}  // namespace

class InterceptBridgeNode : public rclcpp::Node
{
public:
    InterceptBridgeNode()
        : Node("intercept_bridge_node")
    {
        declare_parameter<std::string>("intercept_topic", "/ball_intercept");
        declare_parameter<std::string>("target_topic", "/vision/stewart_target");
        declare_parameter<std::string>("expected_frame_id", "base_link");
        declare_parameter<bool>("require_expected_frame", true);
        declare_parameter<double>("stale_timeout_sec", 0.6);
        declare_parameter<bool>("publish_on_invalid", true);
        declare_parameter<bool>("hold_on_invalid", true);
        declare_parameter<bool>("use_ball_xyz", true);
        declare_parameter<double>("fallback_x", 0.0);
        declare_parameter<double>("fallback_y", 0.0);
        declare_parameter<double>("fallback_z", 0.27);
        declare_parameter<double>("workspace_center_x", 0.0);
        declare_parameter<double>("workspace_center_y", 0.0);
        declare_parameter<double>("workspace_center_z", 0.27);
        declare_parameter<double>("workspace_radius", 0.12);
        declare_parameter<bool>("constrain_workspace_sphere", true);
        declare_parameter<std::string>("sphere_mode", "on_shell");
        declare_parameter<double>("workspace_max_z", 0.30);
        declare_parameter<double>("default_roll", 0.0);
        declare_parameter<bool>("aim_from_ball", true);
        declare_parameter<bool>("use_degrees", false);

        const auto intercept_topic = get_parameter("intercept_topic").as_string();
        const auto target_topic = get_parameter("target_topic").as_string();
        expected_frame_id_ = get_parameter("expected_frame_id").as_string();
        require_expected_frame_ = get_parameter("require_expected_frame").as_bool();
        stale_timeout_sec_ = get_parameter("stale_timeout_sec").as_double();
        publish_on_invalid_ = get_parameter("publish_on_invalid").as_bool();
        hold_on_invalid_ = get_parameter("hold_on_invalid").as_bool();
        use_ball_xyz_ = get_parameter("use_ball_xyz").as_bool();
        fallback_x_ = get_parameter("fallback_x").as_double();
        fallback_y_ = get_parameter("fallback_y").as_double();
        fallback_z_ = get_parameter("fallback_z").as_double();
        center_x_ = get_parameter("workspace_center_x").as_double();
        center_y_ = get_parameter("workspace_center_y").as_double();
        center_z_ = get_parameter("workspace_center_z").as_double();
        workspace_radius_ = get_parameter("workspace_radius").as_double();
        constrain_sphere_ = get_parameter("constrain_workspace_sphere").as_bool();
        sphere_mode_ = get_parameter("sphere_mode").as_string();
        workspace_max_z_ = get_parameter("workspace_max_z").as_double();
        default_roll_ = get_parameter("default_roll").as_double();
        aim_from_ball_ = get_parameter("aim_from_ball").as_bool();
        use_degrees_ = get_parameter("use_degrees").as_bool();

        target_pub_ = create_publisher<StewartControl>(target_topic, 10);
        intercept_sub_ = create_subscription<VolleyballIntercept>(
            intercept_topic, 10,
            std::bind(&InterceptBridgeNode::onIntercept, this, std::placeholders::_1));

        stale_timer_ = create_wall_timer(
            std::chrono::milliseconds(50),
            std::bind(&InterceptBridgeNode::onStaleCheck, this));

        RCLCPP_INFO(
            get_logger(),
            "Fixed-camera bridge: %s -> %s | frame=%s | center=(%.3f,%.3f,%.3f) R=%.3f mode=%s",
            intercept_topic.c_str(),
            target_topic.c_str(),
            expected_frame_id_.c_str(),
            center_x_, center_y_, center_z_,
            workspace_radius_,
            sphere_mode_.c_str());
    }

private:
    StewartControl buildCommand(const VolleyballIntercept& intercept,
                                  uint8_t emergency_stop) const
    {
        return makeCommand(
            intercept,
            emergency_stop,
            default_roll_,
            use_ball_xyz_,
            fallback_x_,
            fallback_y_,
            fallback_z_,
            center_x_,
            center_y_,
            center_z_,
            workspace_radius_,
            constrain_sphere_,
            sphere_mode_,
            workspace_max_z_,
            aim_from_ball_,
            use_degrees_);
    }

    void publishCommand(const VolleyballIntercept& intercept, uint8_t emergency_stop)
    {
        target_pub_->publish(buildCommand(intercept, emergency_stop));
    }

    void onIntercept(const VolleyballIntercept::SharedPtr msg)
    {
        last_intercept_time_ = now();
        has_intercept_ = true;
        last_intercept_ = *msg;

        if (require_expected_frame_ && msg->header.frame_id != expected_frame_id_) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Frame mismatch: got '%s', expected '%s'",
                msg->header.frame_id.c_str(), expected_frame_id_.c_str());
            if (publish_on_invalid_) {
                publishCommand(*msg, kEmergencyStop);
            }
            return;
        }

        if (!msg->valid) {
            if (hold_on_invalid_ && has_last_good_) {
                publishCommand(last_good_intercept_, kNormal);
                return;
            }
            if (publish_on_invalid_) {
                publishCommand(*msg, kEmergencyStop);
            }
            return;
        }

        last_good_intercept_ = *msg;
        has_last_good_ = true;
        publishCommand(*msg, kNormal);

        const auto cmd = buildCommand(*msg, kNormal);
        RCLCPP_INFO_THROTTLE(
            get_logger(), *get_clock(), 500,
            "Stewart xyz=(%.2f,%.2f,%.2f) rpy=(%.2f,%.2f,%.2f) ball=(%.2f,%.2f,%.2f)",
            cmd.x, cmd.y, cmd.z, cmd.roll, cmd.pitch, cmd.yaw,
            msg->position.x, msg->position.y, msg->position.z);
    }

    void onStaleCheck()
    {
        if (!has_intercept_) {
            return;
        }
        const double age = (now() - last_intercept_time_).seconds();
        if (age <= stale_timeout_sec_) {
            return;
        }
        if (hold_on_invalid_ && has_last_good_) {
            publishCommand(last_good_intercept_, kNormal);
            return;
        }
        if (publish_on_invalid_) {
            publishCommand(last_intercept_, kEmergencyStop);
        }
        has_intercept_ = false;
    }

    rclcpp::Publisher<StewartControl>::SharedPtr target_pub_;
    rclcpp::Subscription<VolleyballIntercept>::SharedPtr intercept_sub_;
    rclcpp::TimerBase::SharedPtr stale_timer_;

    std::string expected_frame_id_;
    bool require_expected_frame_{true};
    double stale_timeout_sec_{0.6};
    bool publish_on_invalid_{true};
    bool hold_on_invalid_{true};
    bool use_ball_xyz_{true};
    double fallback_x_{0.0};
    double fallback_y_{0.0};
    double fallback_z_{0.27};
    double center_x_{0.0};
    double center_y_{0.0};
    double center_z_{0.27};
    double workspace_radius_{0.12};
    bool constrain_sphere_{true};
    std::string sphere_mode_{"on_shell"};
    double workspace_max_z_{0.30};
    double default_roll_{0.0};
    bool aim_from_ball_{true};
    bool use_degrees_{false};

    bool has_intercept_{false};
    bool has_last_good_{false};
    rclcpp::Time last_intercept_time_;
    VolleyballIntercept last_intercept_;
    VolleyballIntercept last_good_intercept_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<InterceptBridgeNode>());
    rclcpp::shutdown();
    return 0;
}
