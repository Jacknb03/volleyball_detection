#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

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
    explicit TrajectoryPredictor(double gravity,
                                   double air_density,
                                   double drag_coefficient,
                                   double ball_diameter,
                                   double ball_mass,
                                   double integration_dt = 0.01,
                                   double max_time = 5.0,
                                   double ground_z = 0.0);

    /**
     * 数值积分预测落地点与轨迹
     *
     * 运动模型：
     * - 重力：a_gravity = (0, 0, -g)
     * - 二次阻力：F_drag = -0.5 * rho * Cd * Area * |v| * v
     *   a_drag = F_drag / m
     *
     * 使用 Euler 数值积分，迭代至 World_Z <= ground_z。
     *
     * @param pos             初始世界系位置 (x, y, z)
     * @param vel             初始世界系速度 (vx, vy, vz)
     * @param time_to_land    输出：着地时间（秒）
     * @param landing_pos     输出：着地点 (x, y, z=ground_z)
     * @param path_points     输出：离散轨迹点（包含初末点）
     * @return                是否预测成功（未在 max_time 内落地返回 false）
     */
    bool predictLanding(const Eigen::Vector3d& pos,
                         const Eigen::Vector3d& vel,
                         double& time_to_land,
                         Eigen::Vector3d& landing_pos,
                         std::vector<Eigen::Vector3d>& path_points) const;

    /**
     * 预测球轨迹首次穿过 target_z 平面的时刻与位置。
     * @param crossing  "next" | "descending" | "ascending"
     */
    bool predictAtZ(const Eigen::Vector3d& pos,
                    const Eigen::Vector3d& vel,
                    double target_z,
                    const std::string& crossing,
                    double& time_to_event,
                    Eigen::Vector3d& event_pos,
                    std::vector<Eigen::Vector3d>& path_points) const;

private:
    void eulerStep(Eigen::Vector3d& pos, Eigen::Vector3d& vel) const;

    /// RViz LINE_STRIP 至少需要 2 个点；松手假设下积分出可视化轨迹
    void buildVizPath(const Eigen::Vector3d& pos0,
                      const Eigen::Vector3d& vel0,
                      std::vector<Eigen::Vector3d>& path_points,
                      int min_points = 15) const;

    double g_;             // 重力加速度
    double rho_;           // 空气密度
    double cd_;            // 阻力系数
    double area_;          // 球迎风面积
    double mass_;          // 球质量
    double dt_;            // 积分步长
    double max_time_;      // 最大积分时长
    double ground_z_;     // 地面高度（世界系 Z <= ground_z 表示落地）
};

