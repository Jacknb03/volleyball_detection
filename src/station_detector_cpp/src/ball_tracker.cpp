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
    // 对应 Python:
    // self.state = np.zeros(6, dtype=np.float32)
    state_.setZero();

    // self.P = eye * initial_uncertainty
    P_.setIdentity();
    P_ *= initial_uncertainty;

    // self.Q = eye * process_noise，随后对不同块缩放
    Q_.setIdentity();
    Q_ *= process_noise;
    // 位置噪声更小
    Q_.block<2,2>(0,0) *= 0.1f;
    // 速度噪声
    Q_.block<2,2>(2,2) *= 1.0f;
    // 加速度噪声
    Q_.block<2,2>(4,4) *= 2.0f;

    // self.R = eye(2) * measurement_noise
    R_.setIdentity();
    R_ *= measurement_noise;

    // H 只观测位置 x,y
    H_.setZero();
    H_(0,0) = 1.0f; // 观测 x
    H_(1,1) = 1.0f; // 观测 y

    // F 初始为单位阵，具体值在 predict() 时根据 dt 更新
    F_.setIdentity();
}

void BallTracker::initialize(float x, float y, double timestamp)
{
    // 对应 Python initialize()
    state_.setZero();
    state_(0) = x; // x
    state_(1) = y; // y

    // P = eye * 10.0
    P_.setIdentity();
    P_ *= 10.0f;

    is_initialized_   = true;
    missing_frames_   = 0;
    last_update_time_ = timestamp;
}

cv::Point2f BallTracker::predict(double timestamp)
{
    // 对应 Python predict(timestamp)
    if (!is_initialized_) {
        return cv::Point2f(0.0f, 0.0f);
    }

    // 更新时间步长
    if (last_update_time_ > 0.0) {
        float dt = static_cast<float>(timestamp - last_update_time_);
        dt_ = std::max(0.001f, dt);
    }
    // dt 太大时使用默认 0.033
    if (dt_ > 0.1f) {
        dt_ = 0.033f;
    }

    // 更新 F（与 Python 完全一致）
    const float dt      = dt_;
    const float dt2_half = 0.5f * dt * dt;

    F_ <<
        1.0f, 0.0f, dt,   0.0f, dt2_half, 0.0f,
        0.0f, 1.0f, 0.0f, dt,   0.0f,     dt2_half,
        0.0f, 0.0f, 1.0f, 0.0f, dt,       0.0f,
        0.0f, 0.0f, 0.0f, 1.0f, 0.0f,     dt,
        0.0f, 0.0f, 0.0f, 0.0f, 1.0f,     0.0f,
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f,     1.0f;

    // 状态预测: x = F x
    state_ = F_ * state_;

    // 协方差预测: P = F P F^T + Q
    P_ = F_ * P_ * F_.transpose() + Q_;

    // P += eye * 0.1
    P_ += Eigen::Matrix<float,kStateDim,kStateDim>::Identity() * 0.1f;

    return cv::Point2f(state_(0), state_(1));
}

void BallTracker::update(float x, float y, double timestamp)
{
    // 对应 Python update(measurement, timestamp)
    if (!is_initialized_) {
        initialize(x, y, timestamp);
        return;
    }

    // 计算 dt
    if (last_update_time_ > 0.0) {
        float dt = static_cast<float>(timestamp - last_update_time_);
        dt_ = std::max(0.001f, dt);
    }
    if (dt_ > 0.1f) {
        dt_ = 0.033f;
    }

    // 先做一次预测（与 Python 相同）
    predict(timestamp);

    // 观测向量 z = [x, y]^T
    Eigen::Matrix<float,kMeasurementDim,1> z;
    z(0) = x;
    z(1) = y;

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
}

cv::Point2f BallTracker::updateWithMissing(const std::optional<cv::Point2f>& measurement,
                                           double timestamp)
{
    // 对齐 Python update_with_missing():
    // - measurement == None: missing_frames += 1, 超过 max_missing_frames 则重置并返回(0,0)
    // - 否则: update() 并返回当前位置
    if (!measurement.has_value()) {
        missing_frames_ += 1;

        if (missing_frames_ > max_missing_frames_) {
            is_initialized_ = false;
            // Python reset behavior: state zeros, P eye*10, missing_frames=0, last_update_time=None
            state_.setZero();
            P_.setIdentity();
            P_ *= 10.0f;
            missing_frames_ = 0;
            last_update_time_ = 0.0;
            return cv::Point2f(0.0f, 0.0f);
        }

        return predict(timestamp);
    }

    update(measurement->x, measurement->y, timestamp);
    return cv::Point2f(state_(0), state_(1));
}

std::vector<cv::Point2f> BallTracker::predictFuture(float duration,
                                                    int num_points) const
{
    // 对应 Python predict_future(time_horizon, num_points)
    std::vector<cv::Point2f> predictions;

    if (!is_initialized_ || duration <= 0.0f || num_points <= 0) {
        return predictions;
    }

    predictions.reserve(num_points);

    float dt_step = duration / static_cast<float>(num_points);

    // 备份当前状态（Python 里还备份 P，但预测只用到 state）
    Eigen::Matrix<float,kStateDim,1> state_backup = state_;

    for (int i = 1; i <= num_points; ++i) {
        float t = dt_step * static_cast<float>(i);

        // 使用运动学方程：
        // x = x0 + vx*t + 0.5*ax*t^2
        // y = y0 + vy*t + 0.5*ay*t^2
        float x = state_backup(0) +
                  state_backup(2) * t +
                  0.5f * state_backup(4) * t * t;
        float y = state_backup(1) +
                  state_backup(3) * t +
                  0.5f * state_backup(5) * t * t;

        predictions.emplace_back(x, y);
    }

    return predictions;
}

cv::Point2f BallTracker::getPosition() const
{
    return is_initialized_ ? cv::Point2f(state_(0), state_(1))
                           : cv::Point2f(0.0f, 0.0f);
}

cv::Point2f BallTracker::getVelocity() const
{
    return is_initialized_ ? cv::Point2f(state_(2), state_(3))
                           : cv::Point2f(0.0f, 0.0f);
}

cv::Point2f BallTracker::getAcceleration() const
{
    return is_initialized_ ? cv::Point2f(state_(4), state_(5))
                           : cv::Point2f(0.0f, 0.0f);
}

void BallTracker::reset()
{
    // 等价 Python reset()
    is_initialized_ = false;
    state_.setZero();
    P_.setIdentity();
    P_ *= 10.0f;
    missing_frames_ = 0;
    last_update_time_ = 0.0;
}