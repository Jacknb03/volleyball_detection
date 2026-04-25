# 排球检测与落点预测（ROS2）

一个 ROS2 视觉子系统：**YOLO 2D 检测 → 3D 位置估计 → TF 坐标变换 → 卡尔曼滤波 → 物理落点/轨迹预测**。  
本仓库是一个 **ROS2（colcon）工作区**，默认主线为 **C++ + ONNX（YOLO）** 管线。

## 效果截图

![demo](assets/demo.png)

## 快速开始（默认主线：C++ YOLO / ONNX）

1) 把模型放到（或用启动参数指定）：

- `src/station_detector_cpp/model/best.onnx`（默认路径，**不提交到git**）

2) 编译并 source：

```bash
colcon build --symlink-install
source /opt/ros/humble/setup.bash
source install/setup.bash
```

3) 启动（视频模式，含静态TF + 视频发布 + C++检测节点）：

```bash
./start_all.sh
```

也可以直接用 launch（更清晰、便于写入你的作品集复现实验步骤）：

```bash
ros2 launch station_detector_cpp yolo_cpp_video.launch.py \
  video_path:=/abs/path/to/video.mp4 \
  model_path:=/abs/path/to/best.onnx
```

停止：

```bash
./stop_all.sh
```

## 可选方案（不影响主线启动）

### 方案 B：Python YOLO（`station_detector` 包）

适合快速验证训练出来的模型或做算法对照（依赖 Python 环境、torch/ultralytics）。

```bash
ros2 launch station_detector yolo_test_video.launch.py
ros2 launch station_detector yolo_real_camera.launch.py
```

### 方案 C（Legacy）：传统 OpenCV 圆形/颜色分割（`station_detector` 包）

- **定位**：备用/对照实现  
- **适用性**：在“光照稳定 / 背景简单 / 颜色强先验”的场景可能有价值

## 仓库结构

```
volleyball_detection/
├── src/                      
│   ├── mindvision_camera/      # 工业相机驱动（MindVision）
│   ├── station_detector/       # 传统法 + Python YOLO 工具链/launch
│   └── station_detector_cpp/   # 主线：C++ YOLO(ONNX)+滤波+预测
├── start_all.sh                # 主入口：启动 C++ YOLO（可改 env 覆盖路径）
├── stop_all.sh
├── build/ install/ log/        # colcon 输出（已加 .gitignore）
└── README.md
```

## 配置与路径约定

- **模型路径**：建议始终通过 launch 参数覆盖 `model_path`，而不是在 YAML 里写死绝对路径。
- **主参数文件**：`src/station_detector_cpp/config/ball_detector_params.yaml`
- **视频发布**：由 `station_detector/scripts/video_publisher.py` 提供（launch 已集成）

`start_all.sh` 支持环境变量覆盖：

```bash
VIDEO_PATH=/abs/path/to/video.mp4 MODEL_PATH=/abs/path/to/best.onnx ./start_all.sh
```

## 常用观测

```bash
ros2 node list
ros2 topic list
ros2 topic echo /volleyball_pose
ros2 topic echo /volleyball_trajectory
```

## 系统要求（主线）

- Ubuntu 22.04 + ROS2 Humble
- OpenCV / Eigen3 / tf2（C++包依赖）

