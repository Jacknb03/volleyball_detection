#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2, yaml, os

class StaticImagePub(Node):
    def __init__(self):
        super().__init__('static_image_pub')

        # 声明参数（全部可由 launch 传入）
        self.declare_parameter('image', '')
        self.declare_parameter('camera_yaml', '')
        self.declare_parameter('frame_id', 'camera_optical_frame')
        self.declare_parameter('hz', 5.0)

        image_path = self.get_parameter('image').get_parameter_value().string_value
        cam_yaml   = self.get_parameter('camera_yaml').get_parameter_value().string_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        hz = float(self.get_parameter('hz').get_parameter_value().double_value)

        assert image_path, 'param "image" is empty'
        assert cam_yaml,   'param "camera_yaml" is empty'
        if not os.path.isabs(image_path):
            self.get_logger().warn(f'image is not absolute path: {image_path}')
        if not os.path.isfile(image_path):
            raise FileNotFoundError(image_path)
        if not os.path.isfile(cam_yaml):
            raise FileNotFoundError(cam_yaml)

        self.bridge = CvBridge()
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError(f'cannot read image: {image_path}')

        # 读取 CameraInfo
        with open(cam_yaml, 'r') as f:
            data = yaml.safe_load(f)
        self.info = CameraInfo()
        self.info.width  = data.get('image_width',  self.img.shape[1])
        self.info.height = data.get('image_height', self.img.shape[0])
        self.info.distortion_model = data.get('distortion_model', 'plumb_bob')
        self.info.k = data.get('camera_matrix', {}).get('data', [0]*9)
        self.info.d = data.get('distortion_coefficients', {}).get('data', [0]*5)
        self.info.r = data.get('rectification_matrix', {}).get('data', [1,0,0,0,1,0,0,0,1])
        self.info.p = data.get('projection_matrix', {}).get('data', [0]*12)
        self.info.header.frame_id = self.frame_id

        self.pub_img  = self.create_publisher(Image, '/image_raw', 10)
        self.pub_info = self.create_publisher(CameraInfo, '/camera_info', 10)

        period = 1.0 / max(hz, 0.1)
        self.timer = self.create_timer(period, self.on_timer)
        self.get_logger().info(f'Publishing {image_path} at {hz} Hz, frame_id={self.frame_id}')

    def on_timer(self):
        now = self.get_clock().now().to_msg()
        self.info.header.stamp = now
        msg = self.bridge.cv2_to_imgmsg(self.img, encoding='bgr8')
        msg.header.stamp = now
        msg.header.frame_id = self.frame_id
        self.pub_info.publish(self.info)
        self.pub_img.publish(msg)

def main():
    rclpy.init()
    node = StaticImagePub()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
