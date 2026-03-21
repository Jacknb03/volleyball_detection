#pragma once

#include <Eigen/Dense>
#include <optional>

/**
 * BallTracker
 *
 * 世界系 6D 卡尔曼滤波器（常速度 CV）：
 * 状态向量: [x, y, z, vx, vy, vz]（world，单位：m, m/s）
 * 测量输入: [x, y, z]（world）
 */
class BallTracker
{
public:
    /**
     * 构造函数，对应 Python __init__
     *
     * @param process_noise        过程噪声
     * @param measurement_noise    观测噪声
     * @param initial_uncertainty  初始状态协方差对角线值
     * @param max_missing_frames   最大允许丢帧数（保留，但当前接口不显式暴露“缺测”）
     */
    BallTracker(float process_noise = 1e-3f,
                float measurement_noise = 1e-2f,
                float initial_uncertainty = 1.0f,
                int max_missing_frames = 5);

    /**
     * 使用观测更新滤波器（对应 Kalman update）
     *
     * @param measurement 观测位置（world）[x, y, z]
     * @param timestamp   时间戳（秒）
     */
    void update(const Eigen::Vector3d& measurement, double timestamp);

    /**
     * 丢帧处理：measurement == std::nullopt 时只做 predict
     *
     * @param measurement 若有观测则提供位置，否则 std::nullopt 表示丢帧
     * @param timestamp   时间戳（秒）
     * @return            当前估计位置（未初始化返回全 0）
     */
    Eigen::Vector3d updateWithMissing(const std::optional<Eigen::Vector3d>& measurement,
                                        double timestamp);

    /// 当前是否已经初始化（等价 is_initialized）
    bool isInitialized() const { return is_initialized_; }

    /// 获取当前估计位置 (x, y, z)
    Eigen::Vector3d getPosition() const;

    /// 获取当前速度 (vx, vy, vz)
    Eigen::Vector3d getVelocity() const;

    /// 当前丢帧数（等价 missing_frames）
    int getMissingFrames() const { return missing_frames_; }

    /// 最大丢帧数（等价 max_missing_frames）
    int getMaxMissingFrames() const { return max_missing_frames_; }

    /// 重置（等价 Python reset）
    void reset();

private:
    // 内部初始化函数
    void initialize(const Eigen::Vector3d& measurement, double timestamp);

    // 预测一步（对应 Python predict），返回预测位置
    Eigen::Vector3d predict(double timestamp);

private:
    static constexpr int kStateDim       = 6; // [x, y, z, vx, vy, vz]
    static constexpr int kMeasurementDim = 3; // [x, y, z]

    // 状态向量和矩阵（全部使用 float，与 Python np.float32 对齐）
    Eigen::Matrix<float, kStateDim, 1>            state_;   // x
    Eigen::Matrix<float, kStateDim, kStateDim>    P_;       // 协方差
    Eigen::Matrix<float, kStateDim, kStateDim>    Q_;       // 过程噪声
    Eigen::Matrix<float, kMeasurementDim, kMeasurementDim> R_; // 观测噪声
    Eigen::Matrix<float, kMeasurementDim, kStateDim>       H_; // 观测矩阵
    Eigen::Matrix<float, kStateDim, kStateDim>             F_; // 状态转移

    float  dt_;                 // 时间步长
    bool   is_initialized_;     // 是否初始化
    int    missing_frames_;     // 丢帧计数
    int    max_missing_frames_; // 最大丢帧数
    double last_update_time_;   // 上次更新时间（秒）
};