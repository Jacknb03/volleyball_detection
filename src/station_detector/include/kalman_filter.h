#pragma once
#include <Eigen/Dense>

class KalmanFilter {
public:
    KalmanFilter() = default;
    
    void init(const Eigen::MatrixXd& F, 
              const Eigen::MatrixXd& H, 
              const Eigen::MatrixXd& Q, 
              const Eigen::MatrixXd& R,
              const Eigen::VectorXd& x0);
    
    void predict();
    void update(const Eigen::VectorXd& z);
    
    const Eigen::VectorXd& getState() const { return x_; }
    
private:
    Eigen::MatrixXd F_; // 状态转移矩阵
    Eigen::MatrixXd H_; // 观测矩阵
    Eigen::MatrixXd Q_; // 过程噪声协方差
    Eigen::MatrixXd R_; // 观测噪声协方差
    Eigen::MatrixXd P_; // 估计误差协方差
    Eigen::VectorXd x_; // 状态向量
};

// 增强版卡尔曼滤波器 - 支持位置和朝向滤波
class EnhancedKalmanFilter {
public:
    EnhancedKalmanFilter() = default;
    
    // 初始化滤波器
    void init(double dt, double process_noise = 0.01, double measurement_noise = 0.05);
    
    // 预测步骤
    void predict();
    
    // 更新步骤 - 位置测量
    void updatePosition(const Eigen::Vector3d& position);
    
    // 更新步骤 - 朝向测量（四元数）
    void updateOrientation(const Eigen::Quaterniond& orientation);
    
    // 更新步骤 - 完整位姿测量
    void updatePose(const Eigen::Vector3d& position, const Eigen::Quaterniond& orientation);
    
    // 获取状态
    const Eigen::Vector3d& getPosition() const { return position_; }
    const Eigen::Vector3d& getVelocity() const { return velocity_; }
    const Eigen::Quaterniond& getOrientation() const { return orientation_; }
    const Eigen::Vector3d& getAngularVelocity() const { return angular_velocity_; }
    
    // 获取完整状态向量
    Eigen::VectorXd getFullState() const;
    
private:
    // 状态变量
    Eigen::Vector3d position_;      // 位置 [x, y, z]
    Eigen::Vector3d velocity_;     // 速度 [vx, vy, vz]
    Eigen::Quaterniond orientation_; // 朝向（四元数）
    Eigen::Vector3d angular_velocity_; // 角速度 [wx, wy, wz]
    
    // 协方差矩阵
    Eigen::Matrix3d P_position_;   // 位置协方差
    Eigen::Matrix3d P_velocity_;   // 速度协方差
    Eigen::Matrix3d P_orientation_; // 朝向协方差
    Eigen::Matrix3d P_angular_;     // 角速度协方差
    
    // 噪声参数
    double process_noise_;
    double measurement_noise_;
    double dt_;
    
    // 辅助函数
    void normalizeQuaternion();
    Eigen::Matrix3d quaternionToRotationMatrix(const Eigen::Quaterniond& q) const;
};