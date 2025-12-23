#include "kalman_filter.h"

void KalmanFilter::init(const Eigen::MatrixXd& F, 
                        const Eigen::MatrixXd& H, 
                        const Eigen::MatrixXd& Q, 
                        const Eigen::MatrixXd& R,
                        const Eigen::VectorXd& x0) {
    F_ = F;
    H_ = H;
    Q_ = Q;
    R_ = R;
    x_ = x0;
    
    int n = x_.size();
    P_ = Eigen::MatrixXd::Identity(n, n);
}

void KalmanFilter::predict() {
    x_ = F_ * x_;
    P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::update(const Eigen::VectorXd& z) {
    Eigen::VectorXd y = z - H_ * x_;
    Eigen::MatrixXd S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    
    x_ = x_ + K * y;
    int n = x_.size();
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
    P_ = (I - K * H_) * P_;
}

// 增强版卡尔曼滤波器实现
void EnhancedKalmanFilter::init(double dt, double process_noise, double measurement_noise) {
    dt_ = dt;
    process_noise_ = process_noise;
    measurement_noise_ = measurement_noise;
    
    // 初始化状态
    position_ = Eigen::Vector3d::Zero();
    velocity_ = Eigen::Vector3d::Zero();
    orientation_ = Eigen::Quaterniond::Identity();
    angular_velocity_ = Eigen::Vector3d::Zero();
    
    // 初始化协方差矩阵
    P_position_ = Eigen::Matrix3d::Identity() * 0.1;
    P_velocity_ = Eigen::Matrix3d::Identity() * 0.1;
    P_orientation_ = Eigen::Matrix3d::Identity() * 0.1;
    P_angular_ = Eigen::Matrix3d::Identity() * 0.1;
}

void EnhancedKalmanFilter::predict() {
    // 位置预测：x = x + v*dt
    position_ += velocity_ * dt_;
    
    // 朝向预测：q = q * exp(ω*dt/2)
    Eigen::Vector3d angle_increment = angular_velocity_ * dt_ * 0.5;
    double angle_norm = angle_increment.norm();
    if (angle_norm > 1e-6) {
        Eigen::Quaterniond delta_q;
        delta_q.w() = std::cos(angle_norm);
        delta_q.vec() = std::sin(angle_norm) * angle_increment / angle_norm;
        orientation_ = orientation_ * delta_q;
    }
    
    // 协方差预测
    P_position_ += P_velocity_ * dt_ + Eigen::Matrix3d::Identity() * process_noise_ * dt_ * dt_;
    P_velocity_ += Eigen::Matrix3d::Identity() * process_noise_ * dt_;
    P_orientation_ += Eigen::Matrix3d::Identity() * process_noise_ * dt_;
    P_angular_ += Eigen::Matrix3d::Identity() * process_noise_ * dt_;
    
    // 归一化四元数
    normalizeQuaternion();
}

void EnhancedKalmanFilter::updatePosition(const Eigen::Vector3d& position) {
    // 位置观测更新
    Eigen::Vector3d innovation = position - position_;
    Eigen::Matrix3d S = P_position_ + Eigen::Matrix3d::Identity() * measurement_noise_;
    Eigen::Matrix3d K = P_position_ * S.inverse();
    
    position_ += K * innovation;
    P_position_ = (Eigen::Matrix3d::Identity() - K) * P_position_;
}

void EnhancedKalmanFilter::updateOrientation(const Eigen::Quaterniond& orientation) {
    // 朝向观测更新
    Eigen::Quaterniond error_q = orientation * orientation_.inverse();
    Eigen::Vector3d error_vec = error_q.vec();
    
    Eigen::Matrix3d S = P_orientation_ + Eigen::Matrix3d::Identity() * measurement_noise_;
    Eigen::Matrix3d K = P_orientation_ * S.inverse();
    
    Eigen::Vector3d correction = K * error_vec;
    Eigen::Quaterniond delta_q;
    delta_q.w() = std::cos(correction.norm() * 0.5);
    delta_q.vec() = std::sin(correction.norm() * 0.5) * correction / correction.norm();
    
    orientation_ = orientation_ * delta_q;
    P_orientation_ = (Eigen::Matrix3d::Identity() - K) * P_orientation_;
    
    normalizeQuaternion();
}

void EnhancedKalmanFilter::updatePose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation) {
    updatePosition(position);
    updateOrientation(orientation);
}

Eigen::VectorXd EnhancedKalmanFilter::getFullState() const {
    Eigen::VectorXd state(13); // [x,y,z, qx,qy,qz,qw, vx,vy,vz, wx,wy,wz]
    state.segment<3>(0) = position_;
    state.segment<4>(3) = Eigen::Vector4d(orientation_.x(), orientation_.y(), orientation_.z(), orientation_.w());
    state.segment<3>(7) = velocity_;
    state.segment<3>(10) = angular_velocity_;
    return state;
}

void EnhancedKalmanFilter::normalizeQuaternion() {
    double norm = std::sqrt(orientation_.w() * orientation_.w() + 
                           orientation_.x() * orientation_.x() + 
                           orientation_.y() * orientation_.y() + 
                           orientation_.z() * orientation_.z());
    if (norm > 1e-6) {
        orientation_.w() /= norm;
        orientation_.x() /= norm;
        orientation_.y() /= norm;
        orientation_.z() /= norm;
    } else {
        orientation_ = Eigen::Quaterniond::Identity();
    }
}

Eigen::Matrix3d EnhancedKalmanFilter::quaternionToRotationMatrix(const Eigen::Quaterniond& q) const {
    return q.toRotationMatrix();
}