#pragma once

#include <opencv2/core.hpp>
#include <vector>
#include <Eigen/Dense>
#include <optional>

/**
 * BallTracker
 *
 * 等价于 Python 中的 KalmanTracker：
 * 状态向量: [x, y, vx, vy, ax, ay] (图像坐标系)
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
    BallTracker(float  process_noise        = 0.1f,
                float  measurement_noise    = 5.0f,
                float  initial_uncertainty  = 10.0f,
                int    max_missing_frames   = 5);

    /**
     * 使用观测更新滤波器（对应 Python update）
     *
     * @param x          观测位置 x
     * @param y          观测位置 y
     * @param timestamp  时间戳（秒）
     */
    void update(float x, float y, double timestamp);

    /**
     * 等价于 Python update_with_missing(measurement_or_None, timestamp)
     *
     * @param measurement  若有观测则提供中心点，否则 std::nullopt 表示丢帧
     * @param timestamp    时间戳（秒）
     * @return             当前估计位置 (x, y)，未初始化或重置时返回 (0,0)
     */
    cv::Point2f updateWithMissing(const std::optional<cv::Point2f>& measurement,
                                  double timestamp);

    /**
     * 预测未来轨迹（对应 Python predict_future）
     *
     * @param duration    预测时间范围（秒）
     * @param num_points  预测点数（默认 10，与 Python 一致）
     * @return            预测位置列表（按时间递增）
     */
    std::vector<cv::Point2f> predictFuture(float duration,
                                           int num_points = 10) const;

    /// 当前是否已经初始化（等价 is_initialized）
    bool isInitialized() const { return is_initialized_; }

    /// 获取当前估计位置 (x, y)
    cv::Point2f getPosition() const;

    /// 获取当前速度 (vx, vy)
    cv::Point2f getVelocity() const;

    /// 获取当前加速度 (ax, ay)
    cv::Point2f getAcceleration() const;

    /// 当前丢帧数（等价 missing_frames）
    int getMissingFrames() const { return missing_frames_; }

    /// 最大丢帧数（等价 max_missing_frames）
    int getMaxMissingFrames() const { return max_missing_frames_; }

    /// 重置（等价 Python reset）
    void reset();

private:
    // 内部初始化函数（对应 Python initialize）
    void initialize(float x, float y, double timestamp);

    // 预测一步（对应 Python predict），返回预测位置
    cv::Point2f predict(double timestamp);

private:
    static constexpr int kStateDim       = 6; // [x, y, vx, vy, ax, ay]
    static constexpr int kMeasurementDim = 2; // [x, y]

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