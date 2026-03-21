#include "trajectory_predictor.hpp"

#include <cmath>

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

bool TrajectoryPredictor::predictLanding(const Eigen::Vector3d& pos0,
                                         const Eigen::Vector3d& vel0,
                                         double& time_to_land,
                                         Eigen::Vector3d& landing_pos,
                                         std::vector<Eigen::Vector3d>& path_points) const
{
    path_points.clear();

    if (dt_ <= 0.0 || max_time_ <= 0.0 || mass_ <= 0.0 || g_ <= 0.0) {
        return false;
    }

    Eigen::Vector3d pos = pos0;
    Eigen::Vector3d vel = vel0;
    double t = 0.0;

    path_points.push_back(pos);

    // Semi-implicit Euler (a) update velocity then (b) update position
    Eigen::Vector3d gravity(0.0, 0.0, -g_);

    // acceleration coefficient: a_drag = -(0.5*rho*Cd*A / m) * |v| * v
    const double k = (0.5 * rho_ * cd_ * area_) / mass_;

    if (pos.z() <= ground_z_) {
        time_to_land = 0.0;
        landing_pos = pos;
        landing_pos.z() = ground_z_;
        return true;
    }

    const int max_steps = static_cast<int>(std::ceil(max_time_ / dt_));
    for (int step = 0; step < max_steps; ++step) {
        const Eigen::Vector3d prev_pos = pos;
        const Eigen::Vector3d prev_vel = vel;

        const double speed = vel.norm();
        Eigen::Vector3d a = gravity;
        if (speed > 1e-6) {
            // F_drag = -k*m * |v| * v, so a_drag = -k * |v| * v
            a += (-k * speed) * vel;
        }

        // v_{n+1} = v_n + a_n * dt
        vel = prev_vel + a * dt_;

        // x_{n+1} = x_n + v_{n+1} * dt  (semi-implicit)
        pos = prev_pos + vel * dt_;
        t += dt_;

        // sample path
        path_points.push_back(pos);

        if (pos.z() <= ground_z_) {
            // Interpolate landing point more accurately
            // prev_pos.z() > ground_z_ and pos.z() <= ground_z_
            const double denom = (prev_pos.z() - pos.z());
            double alpha = 1.0;
            if (std::fabs(denom) > 1e-9) {
                alpha = (prev_pos.z() - ground_z_) / denom; // in [0,1]
            }

            landing_pos = prev_pos + alpha * (pos - prev_pos);
            landing_pos.z() = ground_z_;

            time_to_land = t - dt_ + alpha * dt_;
            return true;
        }
    }

    return false;
}

