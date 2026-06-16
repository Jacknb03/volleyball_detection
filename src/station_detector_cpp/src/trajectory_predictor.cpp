#include "trajectory_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>

namespace {

std::string toLower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool crossedZ(double prev_z, double cur_z, double target_z, const std::string& crossing)
{
    if (crossing == "descending") {
        return prev_z > target_z && cur_z <= target_z;
    }
    if (crossing == "ascending") {
        return prev_z < target_z && cur_z >= target_z;
    }
    // next: any crossing
    return (prev_z - target_z) * (cur_z - target_z) <= 0.0 &&
           std::fabs(prev_z - cur_z) > 1e-12;
}

}  // namespace

TrajectoryPredictor::TrajectoryPredictor(double gravity,
                                       double air_density,
                                       double drag_coefficient,
                                       double ball_diameter,
                                       double ball_mass,
                                       double integration_dt,
                                       double max_time,
                                       double ground_z)
    : g_(gravity),
      rho_(air_density),
      cd_(drag_coefficient),
      dt_(integration_dt),
      max_time_(max_time),
      ground_z_(ground_z)
{
    mass_ = ball_mass;
    // Sphere projected area
    const double r = ball_diameter * 0.5;
    const double pi = std::acos(-1.0);
    area_ = pi * r * r;
}

bool TrajectoryPredictor::predictAtZ(const Eigen::Vector3d& pos0,
                                     const Eigen::Vector3d& vel0,
                                     double target_z,
                                     const std::string& crossing_in,
                                     double& time_to_event,
                                     Eigen::Vector3d& event_pos,
                                     std::vector<Eigen::Vector3d>& path_points) const
{
    path_points.clear();

    if (dt_ <= 0.0 || max_time_ <= 0.0 || mass_ <= 0.0 || g_ <= 0.0) {
        return false;
    }

    const std::string crossing = toLower(crossing_in.empty() ? "descending" : crossing_in);

    Eigen::Vector3d pos = pos0;
    Eigen::Vector3d vel = vel0;
    double t = 0.0;

    path_points.push_back(pos);

    const Eigen::Vector3d gravity(0.0, 0.0, -g_);
    const double k = (0.5 * rho_ * cd_ * area_) / mass_;

    if (crossedZ(pos.z() + 1e-9, pos.z(), target_z, crossing)) {
        time_to_event = 0.0;
        event_pos = pos;
        event_pos.z() = target_z;
        return true;
    }

    const int max_steps = static_cast<int>(std::ceil(max_time_ / dt_));
    for (int step = 0; step < max_steps; ++step) {
        const Eigen::Vector3d prev_pos = pos;
        const Eigen::Vector3d prev_vel = vel;

        const double speed = vel.norm();
        Eigen::Vector3d a = gravity;
        if (speed > 1e-6) {
            a += (-k * speed) * vel;
        }

        vel = prev_vel + a * dt_;
        pos = prev_pos + vel * dt_;
        t += dt_;

        path_points.push_back(pos);

        if (crossedZ(prev_pos.z(), pos.z(), target_z, crossing)) {
            const double denom = prev_pos.z() - pos.z();
            double alpha = 1.0;
            if (std::fabs(denom) > 1e-9) {
                alpha = (prev_pos.z() - target_z) / denom;
            }
            alpha = std::clamp(alpha, 0.0, 1.0);

            event_pos = prev_pos + alpha * (pos - prev_pos);
            event_pos.z() = target_z;
            time_to_event = t - dt_ + alpha * dt_;
            return true;
        }
    }

    return false;
}

bool TrajectoryPredictor::predictLanding(const Eigen::Vector3d& pos0,
                                         const Eigen::Vector3d& vel0,
                                         double& time_to_land,
                                         Eigen::Vector3d& landing_pos,
                                         std::vector<Eigen::Vector3d>& path_points) const
{
    if (pos0.z() <= ground_z_) {
        path_points = {pos0};
        time_to_land = 0.0;
        landing_pos = pos0;
        landing_pos.z() = ground_z_;
        return true;
    }
    return predictAtZ(pos0, vel0, ground_z_, "descending",
                      time_to_land, landing_pos, path_points);
}

