#include "trajectory_predictor.hpp"

#include <cmath>

TrajectoryPredictor::TrajectoryPredictor(double gravity)
    : g_(gravity)
{
}

bool TrajectoryPredictor::predictLanding(const Eigen::Vector3d& pos,
                                         const Eigen::Vector3d& vel,
                                         double& time_to_land,
                                         Eigen::Vector2d& landing_xy) const
{
    // 求解 z(t) = z0 + vz * t - 0.5 * g * t^2 = 0
    // a t^2 + b t + c = 0，其中:
    //   a = -0.5 * g
    //   b = vz
    //   c = z0

    const double z0 = pos.z();
    const double vz = vel.z();

    if (g_ <= 0.0) {
        return false;
    }

    const double a = -0.5 * g_;
    const double b = vz;
    const double c = z0;

    const double discriminant = b * b - 4.0 * a * c;
    if (discriminant < 0.0) {
        // 抛物线不与 z=0 相交
        return false;
    }

    const double sqrt_d = std::sqrt(discriminant);
    const double t1 = (-b + sqrt_d) / (2.0 * a);
    const double t2 = (-b - sqrt_d) / (2.0 * a);

    // 只接受正的时间解，并选择较大的一个（更接近“真正落地时刻”）
    double t_candidate = -1.0;
    if (t1 > 0.0 && t2 > 0.0) {
        t_candidate = std::max(t1, t2);
    } else if (t1 > 0.0) {
        t_candidate = t1;
    } else if (t2 > 0.0) {
        t_candidate = t2;
    } else {
        // 两个解都不在未来
        return false;
    }

    time_to_land = t_candidate;

    // 水平面上做匀速运动：
    // x(t) = x0 + vx * t
    // y(t) = y0 + vy * t
    const double x_land = pos.x() + vel.x() * time_to_land;
    const double y_land = pos.y() + vel.y() * time_to_land;

    landing_xy.x() = x_land;
    landing_xy.y() = y_land;

    return true;
}

