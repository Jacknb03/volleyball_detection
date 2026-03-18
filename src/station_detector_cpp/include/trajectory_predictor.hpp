#pragma once

#include <Eigen/Dense>

/**
 * TrajectoryPredictor
 *
 * 使用简单的抛体运动模型预测落地点：
 *   z(t) = z0 + vz * t - 0.5 * g * t^2
 *
 * 输入:
 *   - 当前 3D 位置 pos = (x, y, z)
 *   - 当前 3D 速度 vel = (vx, vy, vz)
 *
 * 输出:
 *   - 落地时刻 time_to_land
 *   - 落地点平面坐标 landing_xy = (x_land, y_land)
 */
class TrajectoryPredictor
{
public:
    explicit TrajectoryPredictor(double gravity = 9.81);

    /**
     * 预测落地点
     *
     * @param pos            当前 3D 位置 (x, y, z)
     * @param vel            当前 3D 速度 (vx, vy, vz)
     * @param time_to_land   输出：着地时间（秒）
     * @param landing_xy     输出：着地点 (x, y)
     * @return               是否预测成功（若当前抛物线不会落到 z=0，则返回 false）
     */
    bool predictLanding(const Eigen::Vector3d& pos,
                        const Eigen::Vector3d& vel,
                        double& time_to_land,
                        Eigen::Vector2d& landing_xy) const;

private:
    double g_; // 重力加速度
};

