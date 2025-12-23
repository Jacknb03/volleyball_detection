#include "station_pose_estimator.h"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    // 直接创建节点实例
    auto estimator_node = std::make_shared<StationPoseEstimator>();
    
    rclcpp::spin(estimator_node);
    rclcpp::shutdown();
    return 0;
}