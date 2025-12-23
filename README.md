# Engineer Vision - 排球检测与跟踪系统

一个基于ROS2的智能视觉检测与跟踪系统，专门用于检测、跟踪排球并预测其运动路径。

## 项目概述

本项目包含两个主要ROS2包：
- **`mindvision_camera`**: 工业相机驱动包，支持海康威视MindVision系列相机
- **`station_detector`**: 排球检测与跟踪包，支持两种检测方案：
  - **传统方案** (C++): 基于颜色分割+圆形检测
  - **YOLO方案** (Python): 基于深度学习的YOLO检测，支持多色排球（蓝/黄/白）

## 主要功能

### 相机功能 (`mindvision_camera`)
- 支持海康威视MindVision工业相机
- 自动相机标定和参数管理
- 实时图像采集和发布
- 相机参数动态配置

### 检测功能 (`station_detector`)

#### 传统方案 (C++)
- **目标检测**: 基于颜色分割的白色/浅色物体检测
- **圆形检测**: 使用最小外接圆和椭圆拟合检测排球
- **3D位置估计**: 基于圆心和半径的深度估计（无需PnP）
- **卡尔曼滤波**: 位置、速度、加速度的平滑和预测
- **路径预测**: 基于运动学方程预测未来0.5秒的轨迹

#### YOLO方案 (Python) - 推荐
- **YOLO检测**: 支持YOLOv5/YOLOv8，自动检测多色排球（蓝/黄/白）
- **误检过滤**: 通过历史信息和多目标跟踪过滤误检
- **卡尔曼滤波**: 平滑轨迹，预测未来运动
- **丢帧处理**: 自动处理丢帧，保持轨迹连续性
- **3D位置估计**: 基于检测框和相机标定的深度估计
- **实时性**: 优化推理速度，支持30Hz+处理频率

两种方案都支持：
- **调试可视化**: 实时调试图像输出和路径可视化
- **多种测试模式**: 静态图片、视频文件、真实摄像头

## 项目结构

```
Engineer_Vision/
├── src/
│   ├── mindvision_camera/          # 相机驱动包
│   │   ├── src/                    # 源代码
│   │   │   └── mv_camera_node.cpp  # 相机节点主程序
│   │   ├── config/                 # 相机配置
│   │   │   ├── camera_info.yaml    # 相机标定信息
│   │   │   └── camera_params.yaml  # 相机参数
│   │   ├── launch/                 # 启动文件
│   │   │   └── mv_launch.py        # 相机启动脚本
│   │   └── mvsdk/                  # MindVision SDK头文件
│   │       └── include/
│   └── station_detector/           # 排球检测包
│       ├── src/                   # 源代码
│       │   ├── main.cpp           # 主程序入口
│       │   ├── fusion_station_detector.cpp      # 检测器实现
│       │   ├── station_pose_estimator.cpp        # 位姿估计器实现
│       │   ├── kalman_filter.cpp   # 卡尔曼滤波器实现
│       │   ├── debug_manager.cpp  # 调试管理器
│       │   └── station_params.cpp # 参数管理
│       ├── include/                # 头文件
│       │   ├── fusion_station_detector.h
│       │   ├── station_pose_estimator.h
│       │   ├── kalman_filter.h
│       │   └── station_params.h
│       ├── config/                 # 参数配置
│       │   ├── station_params.yaml # 检测参数配置
│       │   └── fake_camera_info.yaml # 模拟相机信息
│       ├── launch/                 # 启动文件
│       │   ├── static_image_pub.launch.py    # 静态图像测试
│       │   ├── real_camera.launch.py         # 实时相机检测
│       │   └── test_video.launch.py          # 视频文件测试
│       ├── scripts/                # 脚本文件
│       │   └── static_image_pub.py # 静态图像发布脚本
│       ├── test_images/            # 测试图像
│       │   └── *.png              # 测试图片
│       └── videos/                 # 测试视频
│           └── test.mp4           # 测试视频文件
├── build/                          # 编译输出目录
├── install/                        # 安装目录
├── log/                            # 日志目录
└── debug_images/                   # 调试图像输出目录（运行时生成）
```

## 使用方法

### 1. 编译项目

```bash
cd Engineer_Vision
colcon build --symlink-install
source install/setup.bash
```

### 2. 静态图像测试

#### 传统方案
```bash
ros2 launch station_detector static_image_pub.launch.py
```

#### YOLO方案（推荐）
```bash
ros2 launch station_detector yolo_static_image.launch.py
```

### 3. 视频文件测试

#### 传统方案
```bash
ros2 launch station_detector test_video.launch.py
```

#### YOLO方案（推荐）
```bash
ros2 launch station_detector yolo_test_video.launch.py
```

### 4. 实时相机检测

#### 传统方案
```bash
ros2 launch station_detector real_camera.launch.py
```

#### YOLO方案（推荐）
```bash
# 确保已安装Python依赖（包括opencv-python）
cd src/station_detector
pip3 install -r requirements.txt

# 启动检测
ros2 launch station_detector yolo_real_camera.launch.py
```

### 5. 查看检测结果

```bash
# 查看位姿输出
ros2 topic echo /volleyball_pose

# 查看调试图像
ros2 run image_tools showimage --ros-args -r image:=/debug_image

# 查看发布频率
ros2 topic hz /volleyball_pose

# 查看预测轨迹（RViz2中）
ros2 run rviz2 rviz2
# 添加MarkerArray显示，订阅 /volleyball_trajectory
```

## 调试方法

### 1. 查看节点状态

```bash
# 列出所有节点
ros2 node list

# 查看节点详细信息
ros2 node info /station_pose_estimator

# 查看节点日志
ros2 topic echo /rosout
```

### 2. 查看话题信息

```bash
# 列出所有话题
ros2 topic list

# 查看话题详细信息
ros2 topic info /volleyball_pose
ros2 topic info /debug_image
ros2 topic info /binary_image
ros2 topic info /volleyball_trajectory

# 查看话题发布频率
ros2 topic hz /volleyball_pose

# 查看话题数据
ros2 topic echo /volleyball_pose
```

### 3. 查看和修改参数

```bash
# 列出所有参数
ros2 param list /station_pose_estimator

# 查看参数值
ros2 param get /station_pose_estimator debug
ros2 param get /station_pose_estimator corner_filter_gain
ros2 param get /station_pose_estimator color_detection.white_lower_h

# 设置参数（运行时修改）
ros2 param set /station_pose_estimator debug true
ros2 param set /station_pose_estimator corner_filter_gain 0.3
```

### 4. 调试图像输出

当 `debug: true` 时，系统会在 `debug_images/` 目录下生成以下调试图像：

- **`raw.png`**: 原始输入图像（去畸变后）
- **`mask.png`**: 颜色分割掩码（二值化图像）
- **`all_contours.png`**: 所有检测到的轮廓
- **`filtered_contours.png`**: 过滤后的轮廓（符合圆形度要求）
- **`volleyball_detected.png`**: 检测到的排球（带圆心和半径标注）

### 5. 可视化调试

#### 使用RViz2查看预测路径

```bash
# 启动RViz2
ros2 run rviz2 rviz2

# 添加以下显示项：
# - TF: 查看坐标系变换
# - MarkerArray: 订阅 /volleyball_trajectory 查看预测路径
# - Image: 订阅 /debug_image 查看调试图像
```

#### 使用image_view查看图像

```bash
# 查看调试图像
ros2 run image_tools showimage --ros-args -r image:=/debug_image

# 查看二值化图像
ros2 run image_tools showimage --ros-args -r image:=/binary_image
```

### 6. 常见问题排查

#### 检测不到排球
1. **检查颜色参数**: 调整 `color_detection` 中的HSV范围，确保能检测到白色
2. **检查半径范围**: 确认 `volleyball.min_radius` 和 `volleyball.max_radius` 设置合理
3. **检查圆形度**: 调整 `volleyball.min_circularity`（默认0.7）
4. **查看调试图像**: 检查 `debug_images/mask.png` 确认颜色分割是否正常

#### 位置估计不准确
1. **检查相机标定**: 确认 `camera_info.yaml` 中的相机内参正确
2. **检查真实半径**: 确认 `volleyball.real_radius` 设置正确（标准排球约0.105米）
3. **检查滤波增益**: 调整 `corner_filter_gain`（0-1，越小越平滑但响应越慢）

#### 路径预测不准确
1. **检查卡尔曼滤波参数**: 在代码中调整过程噪声和观测噪声
2. **检查时间步长**: 确认 `dt` 设置与实际帧率匹配

## 话题和消息说明

### 输入话题

- **`/image_raw`** (`sensor_msgs/msg/Image`)
  - 原始图像输入
  - 支持BGR8格式

- **`/camera_info`** (`sensor_msgs/msg/CameraInfo`)
  - 相机标定信息
  - 包含相机内参和畸变系数

### 输出话题

- **`/volleyball_pose`** (`geometry_msgs/msg/PoseStamped`)
  - 排球3D位置和朝向
  - 坐标系: `base_link` 或 `camera_optical_frame`
  - 发布频率: 取决于图像输入频率（通常20-30Hz）
  - 数据内容:
    - `position.x, y, z`: 位置（米）
    - `orientation`: 四元数（x, y, z, w）

- **`/debug_image`** (`sensor_msgs/msg/Image`)
  - 调试图像，包含检测结果可视化
  - 显示检测到的圆形、圆心、半径等信息

- **`/binary_image`** (`sensor_msgs/msg/Image`)
  - 二值化图像（颜色分割结果）
  - 格式: mono8

- **`/volleyball_trajectory`** (`visualization_msgs/msg/MarkerArray`)
  - 预测路径可视化
  - 包含未来0.5秒的预测轨迹点

### TF变换

- **`volleyball`**: 排球坐标系
  - 父坐标系: `base_link` 或 `camera_optical_frame`
  - 发布频率: 与 `/volleyball_pose` 相同

## 参数配置

主要参数文件: `src/station_detector/config/station_params.yaml`

### 检测参数

```yaml
min_contour_area: 100.0      # 最小轮廓面积(像素²)
max_contour_area: 50000.0    # 最大轮廓面积(像素²)
corner_filter_gain: 0.2      # 位置滤波增益(0-1, 越小越平滑)
```

### 颜色检测参数

```yaml
color_detection:
  white_lower_h: 0      # 白色HSV下限 - 色调
  white_lower_s: 0      # 白色HSV下限 - 饱和度
  white_lower_v: 200    # 白色HSV下限 - 明度
  white_upper_h: 180    # 白色HSV上限 - 色调
  white_upper_s: 30     # 白色HSV上限 - 饱和度
  white_upper_v: 255    # 白色HSV上限 - 明度
  morphology_kernel_size: 5  # 形态学处理卷积核大小
  use_adaptive_kernel: true # 是否自适应调整卷积核
```

**HSV颜色空间说明**:
- **H (色调)**: 0-180（OpenCV中H范围是0-180而非0-360）
- **S (饱和度)**: 0-255，白色饱和度低（0-30）
- **V (明度)**: 0-255，白色明度高（200-255）

### 排球检测参数

```yaml
volleyball:
  min_radius: 5.0       # 最小半径(像素)
  max_radius: 200.0    # 最大半径(像素)
  min_circularity: 0.7  # 最小圆形度(0-1, 1为完美圆形)
  real_radius: 0.105   # 排球真实半径(米) - 标准排球直径约21cm
```

### 调试参数

```yaml
debug: true             # 是否启用调试模式（保存调试图像）
show_binary_window: true # 是否显示二值化窗口
publish_tf: true        # 是否发布TF变换
```

## 算法原理

### 1. 颜色分割
- 将图像从BGR转换到HSV颜色空间
- 使用HSV范围过滤白色/浅色区域
- 应用形态学操作（开运算、闭运算）去除噪声

### 2. 圆形检测
- 查找轮廓并计算最小外接圆
- 计算圆形度: `circularity = 4π*面积/周长²`
- 使用椭圆拟合作为备选方案
- 根据圆形度和半径范围过滤候选

### 3. 3D位置估计
- 使用相似三角形原理: `depth = real_radius * focal_length / pixel_radius`
- 计算3D坐标: `x = (u - cx) * depth / fx`, `y = (v - cy) * depth / fy`, `z = depth`
- 转换到base_link坐标系（如果TF可用）

### 4. 卡尔曼滤波
- 状态向量: `[x, y, z, vx, vy, vz, ax, ay, az]` (9维)
- 预测步骤: 使用状态转移矩阵预测下一状态
- 更新步骤: 使用观测位置更新状态估计

### 5. 路径预测
- 使用运动学方程: `x = x0 + v0*t + 0.5*a*t²`
- 预测时间范围: 0.5秒
- 时间步长: 0.05秒（10个预测点）

## 性能指标

- **检测精度**: 位置误差 < 5cm（在2米距离内）
- **处理频率**: 20-30Hz（取决于图像输入频率）
- **检测范围**: 0.5m - 5.0m
- **响应延迟**: < 50ms（单帧处理）
- **路径预测**: 0.5秒未来轨迹

## 依赖项

### ROS2包（通用）
- `rclcpp` / `rclpy`
- `geometry_msgs`
- `sensor_msgs`
- `tf2_ros`
- `tf2_geometry_msgs`
- `cv_bridge`
- `visualization_msgs`

### 传统方案（C++）
- OpenCV (>= 4.0)
- Eigen3
- yaml-cpp

### YOLO方案（Python）

**注意**: 即使系统已安装C++版本的OpenCV，Python代码仍需要Python版本的OpenCV。

安装依赖：
```bash
cd src/station_detector

# 安装Python版OpenCV和其他依赖
pip3 install -r requirements.txt

# 或者单独安装（如果requirements.txt有问题）
pip3 install opencv-python>=4.5.0
pip3 install torch torchvision
pip3 install ultralytics  # YOLOv8
pip3 install numpy>=1.21.0
```

**验证安装**:
```bash
python3 -c "import cv2; print(cv2.__version__)"  # 应该显示版本号
python3 -c "import torch; print(torch.__version__)"  # 应该显示版本号
```

## 系统要求

- **操作系统**: Ubuntu 20.04/22.04
- **ROS2**: Foxy/Humble
- **Python**: 3.8+
- **OpenCV**: C++版本（系统已安装）+ Python版本（需要安装）

## 许可证

Apache License 2.0
