#include "ball_tracker.hpp"

#include <algorithm>
#include <cmath>

BallTracker::BallTracker(float process_noise,
                         float measurement_noise,
                         float initial_uncertainty,
                         int   max_missing_frames)
    : dt_(0.033f),
      is_initialized_(false),
      missing_frames_(0),
      max_missing_frames_(max_missing_frames),
      last_update_time_(0.0)
{
    // state_ = [x, y, z, vx, vy, vz]
    state_.setZero();

    // P = eye * initial_uncertainty
    P_.setIdentity();
    P_ *= initial_uncertainty;

    // Q = eye * process_noise，然后对不同块做缩放
    Q_.setIdentity();
    Q_ *= process_noise;

    // 位置噪声更小，速度噪声更大（工程启发式）
    Q_.block<3,3>(0,0) *= 0.1f;
    Q_.block<3,3>(3,3) *= 1.0f;

    // R = eye(3) * measurement_noise
    R_.setIdentity();
    R_ *= measurement_noise;

    // H 只观测位置 x,y,z
    H_.setZero();
    H_(0,0) = 1.0f; // x
    H_(1,1) = 1.0f; // y
    H_(2,2) = 1.0f; // z

    // F 初始为单位阵，具体值在 predict() 时根据 dt 更新
    F_.setIdentity();
}

void BallTracker::initialize(const Eigen::Vector3d& measurement, double timestamp)
{
    // 对应 Kalman initialize(measurement)
    state_.setZero();
    state_(0) = static_cast<float>(measurement.x());
    state_(1) = static_cast<float>(measurement.y());
    state_(2) = static_cast<float>(measurement.z());

    // P = eye * initial_uncertainty * 1.0（用构造函数提供的初值已设置过，这里固定放大）
    P_.setIdentity();
    P_ *= 10.0f;

    is_initialized_ = true;
    missing_frames_ = 0;
    last_update_time_ = timestamp;
    last_measurement_ = Eigen::Vector3f(
        static_cast<float>(measurement.x()),
        static_cast<float>(measurement.y()),
        static_cast<float>(measurement.z()));
    has_last_measurement_ = true;
    velocity_bootstrapped_ = false;
}

Eigen::Vector3d BallTracker::predict(double timestamp)
{
    if (!is_initialized_) {
        return Eigen::Vector3d::Zero();
    }

    if (last_update_time_ > 0.0) {
        float dt = static_cast<float>(timestamp - last_update_time_);
        dt_ = std::max(0.001f, dt);
    }

    const float dt      = dt_;
    F_.setIdentity();
    // CV: p' = p + v*dt, v' = v
    F_(0,3) = dt;
    F_(1,4) = dt;
    F_(2,5) = dt;

    // x = F x
    state_ = F_ * state_;

    // P = F P F^T + Q
    P_ = F_ * P_ * F_.transpose() + Q_;

    // 额外膨胀一点点数值鲁棒性
    P_ += Eigen::Matrix<float,kStateDim,kStateDim>::Identity() * 0.1f;

    return Eigen::Vector3d(state_(0), state_(1), state_(2));
}

void BallTracker::update(const Eigen::Vector3d& measurement, double timestamp)
{
    if (!is_initialized_) {
        initialize(measurement, timestamp);
        return;
    }

    if (last_update_time_ > 0.0) {
        float dt = static_cast<float>(timestamp - last_update_time_);
        dt_ = std::max(0.001f, dt);
    }

    // 第二次检测时用有限差分“注入”初速度，避免轨迹预测从零速起步
    if (!velocity_bootstrapped_ && has_last_measurement_ && dt_ > 1e-4f) {
        const Eigen::Vector3f meas(
            static_cast<float>(measurement.x()),
            static_cast<float>(measurement.y()),
            static_cast<float>(measurement.z()));
        const Eigen::Vector3f v0 = (meas - last_measurement_) / dt_;
        state_(3) = v0.x();
        state_(4) = v0.y();
        state_(5) = v0.z();
        velocity_bootstrapped_ = true;
    }

    // predict 到当前 timestamp
    predict(timestamp);

    // z = [x, y, z]^T
    Eigen::Matrix<float,kMeasurementDim,1> z;
    z(0) = static_cast<float>(measurement.x());
    z(1) = static_cast<float>(measurement.y());
    z(2) = static_cast<float>(measurement.z());

    // y = z - H x
    Eigen::Matrix<float,kMeasurementDim,1> y_residual = z - H_ * state_;

    // S = H P H^T + R
    Eigen::Matrix<float,kMeasurementDim,kMeasurementDim> S =
        H_ * P_ * H_.transpose() + R_;

    // K = P H^T S^{-1}
    Eigen::Matrix<float,kStateDim,kMeasurementDim> K =
        P_ * H_.transpose() * S.inverse();

    // x = x + K y
    state_ = state_ + K * y_residual;

    // P = (I - K H) P
    Eigen::Matrix<float,kStateDim,kStateDim> I =
        Eigen::Matrix<float,kStateDim,kStateDim>::Identity();
    P_ = (I - K * H_) * P_;

    missing_frames_   = 0;
    last_update_time_ = timestamp;
    last_measurement_ = Eigen::Vector3f(
        static_cast<float>(measurement.x()),
        static_cast<float>(measurement.y()),
        static_cast<float>(measurement.z()));
    has_last_measurement_ = true;
}

Eigen::Vector3d BallTracker::updateWithMissing(const std::optional<Eigen::Vector3d>& measurement,
                                           double timestamp)
{
    if (!measurement.has_value()) {
        missing_frames_ += 1;

        if (missing_frames_ > max_missing_frames_) {
            is_initialized_ = false;
            state_.setZero();
            P_.setIdentity();
            P_ *= 10.0f;
            missing_frames_ = 0;
            last_update_time_ = 0.0;
            has_last_measurement_ = false;
            velocity_bootstrapped_ = false;
            return Eigen::Vector3d::Zero();
        }

        return predict(timestamp);
    }

    update(*measurement, timestamp);
    return Eigen::Vector3d(state_(0), state_(1), state_(2));
}

Eigen::Vector3d BallTracker::getPosition() const
{
    return is_initialized_ ? Eigen::Vector3d(state_(0), state_(1), state_(2))
                           : Eigen::Vector3d::Zero();
}

Eigen::Vector3d BallTracker::getVelocity() const
{
    return is_initialized_ ? Eigen::Vector3d(state_(3), state_(4), state_(5))
                           : Eigen::Vector3d::Zero();
}

void BallTracker::reset()
{
    is_initialized_ = false;
    state_.setZero();
    P_.setIdentity();
    P_ *= 10.0f;
    missing_frames_ = 0;
    last_update_time_ = 0.0;
    has_last_measurement_ = false;
    velocity_bootstrapped_ = false;
}