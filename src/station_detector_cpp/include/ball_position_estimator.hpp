#pragma once

#include <Eigen/Dense>

/**
 * BallPositionEstimator
 *
 * 根据 2D 边界框和相机内参估计排球在相机坐标系下的 3D 位置。
 *
 * 模型假设：
 *  - 已知排球直径 D（米）
 *  - 使用简单的针孔成像 / 透视投影模型：
 *
 *      Z = fx * D / bbox_height
 *      X = (u - cx) * Z / fx
 *      Y = (v - cy) * Z / fy
 *
 *  其中 (u, v) 是像素坐标系中的边界框中心，bbox_height 是像素高度。
 */
class BallPositionEstimator
{
public:
    BallPositionEstimator() = default;

    BallPositionEstimator(double fx, double fy,
                          double cx, double cy,
                          double ball_diameter)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          ball_diameter_(ball_diameter),
          intrinsics_set_(true)
    {
    }

    /// 设置相机内参
    void setCameraIntrinsics(double fx, double fy, double cx, double cy)
    {
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
        intrinsics_set_ = true;
    }

    /// 设置排球直径（米），例如 0.22 或 0.23
    void setBallDiameter(double diameter_m)
    {
        ball_diameter_ = diameter_m;
    }

    /**
     * 通过 2D 边界框估计 3D 位置（相机坐标系）
     *
     * @param u            边界框中心 u（像素）
     * @param v            边界框中心 v（像素）
     * @param bbox_height  边界框高度（像素）
     * @param out_position 输出：3D 位置 (X, Y, Z)
     * @return             是否估计成功（相机参数或直径未设置、bbox_height <= 0 时返回 false）
     */
    bool estimate(double u, double v,
                  double bbox_height,
                  Eigen::Vector3d& out_position) const
    {
        if (!intrinsics_set_ || ball_diameter_ <= 0.0 || fx_ <= 0.0 || fy_ <= 0.0) {
            return false;
        }
        if (bbox_height <= 0.0) {
            return false;
        }

        // 深度估计（使用像素高度对应竖直焦距 fy）
        // pinhole: bbox_height_pixels / fy = ball_diameter_meters / Z  =>  Z = fy * D / bbox_height
        const double Z = fy_ * ball_diameter_ / bbox_height;

        // 反投影到相机坐标系
        const double X = (u - cx_) * Z / fx_;
        const double Y = (v - cy_) * Z / fy_;

        out_position = Eigen::Vector3d(X, Y, Z);
        return true;
    }

    /**
     * 通过 RGB 像素坐标 + 深度（米）反投影到相机坐标系。
     * 适用于 RealSense 等 RGB-D 相机（depth 已与 color 对齐）。
     */
    bool estimateFromDepth(double u, double v,
                           double depth_m,
                           Eigen::Vector3d& out_position) const
    {
        if (!intrinsics_set_ || depth_m <= 0.0 || fx_ <= 0.0 || fy_ <= 0.0) {
            return false;
        }

        const double Z = depth_m;
        const double X = (u - cx_) * Z / fx_;
        const double Y = (v - cy_) * Z / fy_;
        out_position = Eigen::Vector3d(X, Y, Z);
        return true;
    }

private:
    double fx_{0.0};
    double fy_{0.0};
    double cx_{0.0};
    double cy_{0.0};
    double ball_diameter_{0.0};  // 排球直径（米）
    bool   intrinsics_set_{false};
};

