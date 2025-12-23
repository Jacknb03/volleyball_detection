#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO排球检测ROS2节点
集成YOLO检测和卡尔曼滤波，输出平滑的排球轨迹
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import cv2
from cv_bridge import CvBridge
import numpy as np
from typing import Optional, Tuple
import time

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Header

from yolo_detector import YOLODetector
from kalman_tracker import KalmanTracker
from detection_validator import DetectionValidator, MultiDetectionTracker


class YOLOVolleyballNode(Node):
    """
    YOLO排球检测节点
    """
    
    def __init__(self):
        super().__init__('yolo_volleyball_detector')
        
        # 初始化CV桥接
        self.bridge = CvBridge()
        
        # 参数声明
        self._declare_parameters()
        
        # 初始化YOLO检测器
        self._init_yolo_detector()
        
        # 初始化卡尔曼滤波器
        self._init_kalman_tracker()
        
        # 初始化检测验证器
        self._init_detection_validator()
        
        # 相机参数
        self.camera_matrix = None
        self.distortion_coeffs = None
        self.camera_info_received = False
        
        # 状态变量
        self.last_detection_time = None
        self.frame_count = 0
        self.detection_count = 0
        
        # QoS配置（实时性优先）
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 订阅者
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos_profile
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info',
            self.camera_info_callback,
            10
        )
        
        # 发布者
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/volleyball_pose',
            10
        )
        
        self.debug_pub = self.create_publisher(
            Image,
            '/debug_image',
            10
        )
        
        self.trajectory_pub = self.create_publisher(
            MarkerArray,
            '/volleyball_trajectory',
            10
        )
        
        self.get_logger().info("YOLO排球检测节点已启动")
    
    def _declare_parameters(self):
        """声明ROS2参数"""
        # YOLO参数
        self.declare_parameter('yolo.model_path', '')
        self.declare_parameter('yolo.model_type', 'yolov8')  # 'yolov5' 或 'yolov8'
        self.declare_parameter('yolo.conf_threshold', 0.5)
        self.declare_parameter('yolo.iou_threshold', 0.45)
        self.declare_parameter('yolo.device', 'auto')  # 'auto', 'cpu', 'cuda'
        self.declare_parameter('yolo.volleyball_classes', ['sports ball', 'ball'])
        
        # 卡尔曼滤波参数
        self.declare_parameter('kalman.process_noise', 0.1)
        self.declare_parameter('kalman.measurement_noise', 5.0)
        self.declare_parameter('kalman.initial_uncertainty', 10.0)
        self.declare_parameter('kalman.max_missing_frames', 5)
        
        # 检测参数
        self.declare_parameter('detection.max_detections', 3)  # 最多保留几个检测结果
        self.declare_parameter('detection.selection_method', 'confidence')  # 'confidence' 或 'center'
        self.declare_parameter('detection.min_confidence', 0.3)  # 最低置信度
        self.declare_parameter('detection.max_jump_distance', 100.0)  # 最大跳跃距离(像素)
        self.declare_parameter('detection.min_consistent_detections', 2)  # 最小连续检测次数
        self.declare_parameter('detection.use_tracker', True)  # 是否使用多目标跟踪
        
        # 3D估计参数
        self.declare_parameter('volleyball.real_radius', 0.105)  # 排球真实半径(米)
        self.declare_parameter('volleyball.min_depth', 0.3)  # 最小深度(米)
        self.declare_parameter('volleyball.max_depth', 5.0)  # 最大深度(米)
        
        # 轨迹预测参数
        self.declare_parameter('trajectory.prediction_time', 0.5)  # 预测时间(秒)
        self.declare_parameter('trajectory.num_points', 10)  # 预测点数
        
        # 调试参数
        self.declare_parameter('debug.enable', True)
        self.declare_parameter('debug.show_fps', True)
        self.declare_parameter('debug.draw_trajectory', True)
        
        # 发布TF参数
        self.declare_parameter('publish_tf', False)  # 暂时不发布TF，简化实现
    
    def _init_yolo_detector(self):
        """初始化YOLO检测器"""
        try:
            model_path = self.get_parameter('yolo.model_path').get_parameter_value().string_value
            if not model_path:
                model_path = None
            
            model_type = self.get_parameter('yolo.model_type').get_parameter_value().string_value
            conf_threshold = self.get_parameter('yolo.conf_threshold').get_parameter_value().double_value
            iou_threshold = self.get_parameter('yolo.iou_threshold').get_parameter_value().double_value
            device = self.get_parameter('yolo.device').get_parameter_value().string_value
            
            self.yolo_detector = YOLODetector(
                model_path=model_path,
                model_type=model_type,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                device=device
            )
            
            self.volleyball_classes = self.get_parameter('yolo.volleyball_classes').get_parameter_value().string_array_value
            
            self.get_logger().info(f"YOLO检测器初始化成功: {model_type}, 设备: {device}")
            
        except Exception as e:
            self.get_logger().error(f"YOLO检测器初始化失败: {e}")
            raise
    
    def _init_kalman_tracker(self):
        """初始化卡尔曼滤波器"""
        process_noise = self.get_parameter('kalman.process_noise').get_parameter_value().double_value
        measurement_noise = self.get_parameter('kalman.measurement_noise').get_parameter_value().double_value
        initial_uncertainty = self.get_parameter('kalman.initial_uncertainty').get_parameter_value().double_value
        max_missing_frames = self.get_parameter('kalman.max_missing_frames').get_parameter_value().integer_value
        
        self.kalman_tracker = KalmanTracker(
            process_noise=process_noise,
            measurement_noise=measurement_noise,
            initial_uncertainty=initial_uncertainty,
            max_missing_frames=max_missing_frames
        )
        
        self.get_logger().info("卡尔曼滤波器初始化成功")
    
    def _init_detection_validator(self):
        """初始化检测验证器"""
        max_jump_distance = self.get_parameter('detection.max_jump_distance').get_parameter_value().double_value
        min_consistent = self.get_parameter('detection.min_consistent_detections').get_parameter_value().integer_value
        use_tracker = self.get_parameter('detection.use_tracker').get_parameter_value().bool_value
        
        self.validator = DetectionValidator(
            max_jump_distance=max_jump_distance,
            min_consistent_detections=min_consistent
        )
        
        if use_tracker:
            self.tracker = MultiDetectionTracker(max_track_distance=max_jump_distance * 1.5)
        else:
            self.tracker = None
        
        self.get_logger().info("检测验证器初始化成功")
    
    def camera_info_callback(self, msg: CameraInfo):
        """相机信息回调"""
        if not self.camera_info_received:
            # 提取相机内参
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.distortion_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info("收到相机标定信息")
    
    def image_callback(self, msg: Image):
        """图像回调函数"""
        try:
            # 转换图像
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return
        
        # 更新帧计数
        self.frame_count += 1
        current_time = time.time()
        
        # YOLO检测
        detections = self.yolo_detector.detect(cv_image)
        
        # 过滤出排球
        volleyball_detections = self.yolo_detector.filter_volleyball(
            detections, 
            self.volleyball_classes
        )
        
        # 选择最佳检测结果
        best_detection = None
        if volleyball_detections:
            selection_method = self.get_parameter('detection.selection_method').get_parameter_value().string_value
            min_confidence = self.get_parameter('detection.min_confidence').get_parameter_value().double_value
            
            # 过滤低置信度检测
            filtered = [d for d in volleyball_detections if d['confidence'] >= min_confidence]
            
            if filtered:
                # 使用多目标跟踪器选择最佳检测
                if self.tracker:
                    best_detection = self.tracker.update(filtered)
                else:
                    best_detection = self.yolo_detector.select_best_detection(
                        filtered, 
                        method=selection_method
                    )
                
                # 验证检测结果
                if best_detection:
                    image_size = (cv_image.shape[1], cv_image.shape[0])
                    if self.validator.validate(best_detection, image_size):
                        self.detection_count += 1
                    else:
                        # 验证失败，认为是误检
                        best_detection = None
        
        # 卡尔曼滤波更新
        if best_detection:
            center = best_detection['center']
            self.kalman_tracker.update_with_missing(center, current_time)
            self.last_detection_time = current_time
        else:
            # 丢帧，只预测
            self.kalman_tracker.update_with_missing(None, current_time)
        
        # 获取滤波后的状态
        kalman_state = self.kalman_tracker.get_state()
        
        # 估计3D位置（如果有相机标定）
        if self.camera_info_received and kalman_state['is_initialized']:
            pose = self._estimate_3d_pose(kalman_state['position'], msg.header)
            if pose:
                self.pose_pub.publish(pose)
                
                # 发布预测轨迹
                if self.get_parameter('debug.enable').get_parameter_value().bool_value:
                    self._publish_trajectory(msg.header)
        
        # 发布调试图像
        if self.get_parameter('debug.enable').get_parameter_value().bool_value:
            debug_image = self._create_debug_image(
                cv_image, 
                volleyball_detections, 
                best_detection,
                kalman_state
            )
            self._publish_debug_image(debug_image, msg.header)
    
    def _estimate_3d_pose(self, image_center: Tuple[float, float], 
                          header: Header) -> Optional[PoseStamped]:
        """
        基于图像中心点估计3D位置
        
        注意：这里简化处理，假设排球在固定深度
        实际应该使用YOLO检测框的尺寸来估计深度
        """
        if not self.camera_info_received:
            return None
        
        # 获取相机内参
        fx = self.camera_matrix[0, 0]
        fy = self.camera_matrix[1, 1]
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        
        # 获取参数
        real_radius = self.get_parameter('volleyball.real_radius').get_parameter_value().double_value
        min_depth = self.get_parameter('volleyball.min_depth').get_parameter_value().double_value
        max_depth = self.get_parameter('volleyball.max_depth').get_parameter_value().double_value
        
        # 简化：使用固定深度或根据历史估计
        # 实际应该根据检测框大小估计深度
        # 这里使用一个合理的默认值
        depth = 2.0  # 默认2米
        
        # 计算3D位置（相机坐标系）
        x = (image_center[0] - cx) * depth / fx
        y = (image_center[1] - cy) * depth / fy
        z = depth
        
        # 限制深度范围
        if z < min_depth or z > max_depth:
            return None
        
        # 创建位姿消息
        pose = PoseStamped()
        pose.header = header
        pose.header.frame_id = header.frame_id  # 保持相机坐标系
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0  # 单位四元数
        
        return pose
    
    def _publish_trajectory(self, header: Header):
        """发布预测轨迹"""
        prediction_time = self.get_parameter('trajectory.prediction_time').get_parameter_value().double_value
        num_points = self.get_parameter('trajectory.num_points').get_parameter_value().integer_value
        
        # 获取预测轨迹
        future_positions = self.kalman_tracker.predict_future(prediction_time, num_points)
        
        if not future_positions:
            return
        
        # 创建MarkerArray
        marker_array = MarkerArray()
        
        # 创建路径线条
        line_marker = Marker()
        line_marker.header = header
        line_marker.ns = "volleyball_trajectory"
        line_marker.id = 0
        line_marker.type = Marker.LINE_STRIP
        line_marker.action = Marker.ADD
        line_marker.scale.x = 0.02
        line_marker.color.r = 1.0
        line_marker.color.g = 0.0
        line_marker.color.b = 0.0
        line_marker.color.a = 1.0
        
        # 注意：这里简化处理，实际应该将图像坐标转换为3D坐标
        # 暂时使用图像坐标（需要后续改进）
        for pos in future_positions:
            point = Point()
            point.x = pos[0] / 100.0  # 简化转换
            point.y = pos[1] / 100.0
            point.z = 0.0
            line_marker.points.append(point)
        
        marker_array.markers.append(line_marker)
        self.trajectory_pub.publish(marker_array)
    
    def _create_debug_image(self, 
                           image: np.ndarray,
                           detections: list,
                           best_detection: Optional[dict],
                           kalman_state: dict) -> np.ndarray:
        """创建调试图像"""
        debug_image = image.copy()
        
        # 绘制所有检测结果
        for det in detections:
            bbox = det['bbox']
            center = det['center']
            conf = det['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = map(int, center)
            
            # 绘制边界框（黄色）
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # 绘制中心点
            cv2.circle(debug_image, (cx, cy), 3, (0, 255, 255), -1)
            
            # 绘制标签
            label = f"{det['class_name']}: {conf:.2f}"
            cv2.putText(debug_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # 绘制最佳检测结果（绿色）
        if best_detection:
            bbox = best_detection['bbox']
            center = best_detection['center']
            x1, y1, x2, y2 = map(int, bbox)
            cx, cy = map(int, center)
            
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.circle(debug_image, (cx, cy), 5, (0, 255, 0), -1)
        
        # 绘制卡尔曼滤波结果（红色）
        if kalman_state['is_initialized']:
            kf_pos = kalman_state['position']
            kf_vel = kalman_state['velocity']
            kf_cx, kf_cy = map(int, kf_pos)
            
            # 绘制滤波后的中心点
            cv2.circle(debug_image, (kf_cx, kf_cy), 8, (0, 0, 255), 2)
            
            # 绘制速度向量
            vel_scale = 10.0
            end_x = int(kf_cx + kf_vel[0] * vel_scale)
            end_y = int(kf_cy + kf_vel[1] * vel_scale)
            cv2.arrowedLine(debug_image, (kf_cx, kf_cy), (end_x, end_y), 
                          (0, 0, 255), 2, tipLength=0.3)
            
            # 绘制预测轨迹
            if self.get_parameter('debug.draw_trajectory').get_parameter_value().bool_value:
                prediction_time = self.get_parameter('trajectory.prediction_time').get_parameter_value().double_value
                num_points = self.get_parameter('trajectory.num_points').get_parameter_value().integer_value
                future_positions = self.kalman_tracker.predict_future(prediction_time, num_points)
                
                for i, pos in enumerate(future_positions):
                    px, py = map(int, pos)
                    if 0 <= px < debug_image.shape[1] and 0 <= py < debug_image.shape[0]:
                        alpha = i / len(future_positions)
                        color = (int(255 * alpha), 0, int(255 * (1 - alpha)))
                        cv2.circle(debug_image, (px, py), 2, color, -1)
        
        # 绘制状态信息
        info_y = 20
        cv2.putText(debug_image, f"Detections: {len(detections)}", 
                   (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        info_y += 25
        
        if kalman_state['is_initialized']:
            cv2.putText(debug_image, f"KF: OK, Missing: {kalman_state['missing_frames']}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(debug_image, "KF: Not initialized", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示FPS
        if self.get_parameter('debug.show_fps').get_parameter_value().bool_value:
            if self.last_detection_time:
                fps = 1.0 / (time.time() - self.last_detection_time) if time.time() != self.last_detection_time else 0
                cv2.putText(debug_image, f"FPS: {fps:.1f}", 
                           (10, info_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return debug_image
    
    def _publish_debug_image(self, image: np.ndarray, header: Header):
        """发布调试图像"""
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            debug_msg.header = header
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"发布调试图像失败: {e}")


def main(args=None):
    rclpy.init(args=args)
    
    node = YOLOVolleyballNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

