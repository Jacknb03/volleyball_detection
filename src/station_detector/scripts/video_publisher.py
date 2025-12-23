#!/usr/bin/env python3
"""
视频发布节点 - 从视频文件发布图像到ROS2话题
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import os


class VideoPublisher(Node):
    def __init__(self):
        super().__init__('video_publisher')
        
        # 参数
        self.declare_parameter('video_path', '')
        self.declare_parameter('frame_rate', 30.0)
        self.declare_parameter('loop', True)
        self.declare_parameter('rotate_vertical', False)
        
        video_path = self.get_parameter('video_path').get_parameter_value().string_value
        frame_rate = self.get_parameter('frame_rate').get_parameter_value().double_value
        self.loop = self.get_parameter('loop').get_parameter_value().bool_value
        self.rotate_vertical = self.get_parameter('rotate_vertical').get_parameter_value().bool_value
        
        if not video_path or not os.path.isfile(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        # 打开视频
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {video_path}")
        
        # 获取视频信息
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.get_logger().info(
            f"视频信息: {video_path}, "
            f"尺寸: {self.width}x{self.height}, "
            f"FPS: {self.fps}, "
            f"总帧数: {self.total_frames}"
        )
        
        self.bridge = CvBridge()
        self.frame_id = 'camera_optical_frame'
        
        # 发布者
        self.image_pub = self.create_publisher(Image, '/image_raw', 10)
        self.info_pub = self.create_publisher(CameraInfo, '/camera_info', 10)
        
        # 创建虚拟相机信息
        self.camera_info = CameraInfo()
        self.camera_info.width = self.width
        self.camera_info.height = self.height
        self.camera_info.distortion_model = 'plumb_bob'
        # 默认相机内参（需要根据实际情况调整）
        fx = fy = self.width * 0.8  # 简化的焦距估计
        cx = self.width / 2.0
        cy = self.height / 2.0
        self.camera_info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        self.camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        self.camera_info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self.camera_info.header.frame_id = self.frame_id
        
        # 定时器
        period = 1.0 / max(frame_rate, 1.0)
        self.timer = self.create_timer(period, self.publish_frame)
        
        self.get_logger().info(f"开始发布视频，帧率: {frame_rate} Hz")
    
    def publish_frame(self):
        ret, frame = self.cap.read()
        
        if not ret:
            if self.loop:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    self.get_logger().info("视频循环播放")
                else:
                    self.get_logger().error("无法重新打开视频")
                    return
            else:
                self.get_logger().warn("视频播放完毕")
                return
        
        # 旋转（如果需要）
        if self.rotate_vertical:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            # 更新相机信息
            self.camera_info.width = frame.shape[1]
            self.camera_info.height = frame.shape[0]
        
        # 发布图像
        now = self.get_clock().now().to_msg()
        self.camera_info.header.stamp = now
        
        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            img_msg.header.stamp = now
            img_msg.header.frame_id = self.frame_id
            
            self.image_pub.publish(img_msg)
            self.info_pub.publish(self.camera_info)
        except Exception as e:
            self.get_logger().error(f"发布图像失败: {e}")
    
    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

