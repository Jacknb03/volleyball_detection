# YOLO排球检测快速开始 (Ubuntu)

## 快速安装

```bash
cd src/station_detector

# 安装Python依赖（包括Python版OpenCV）
pip3 install -r requirements.txt

# 验证安装
python3 -c "import cv2; print('OpenCV版本:', cv2.__version__)"
python3 -c "import torch; print('PyTorch版本:', torch.__version__)"
```

**注意**: 即使系统已安装C++版本的OpenCV，Python代码仍需要单独安装Python版本的OpenCV (`opencv-python`)。两者可以共存，互不影响。

## 快速测试

### 1. 静态图片测试
```bash
ros2 launch station_detector yolo_static_image.launch.py
```

### 2. 视频测试
```bash
ros2 launch station_detector yolo_test_video.launch.py
```

### 3. 真实摄像头
```bash
ros2 launch station_detector yolo_real_camera.launch.py
```

## 关键参数调整

编辑 `config/yolo_params.yaml`:

- **检测不到排球**: 降低 `yolo.conf_threshold` 到 0.3
- **误检太多**: 提高 `yolo.conf_threshold` 到 0.6
- **轨迹抖动**: 增加 `kalman.measurement_noise` 到 8.0
- **响应太慢**: 减少 `kalman.measurement_noise` 到 3.0

详细说明见主README。

