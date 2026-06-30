# 部署与硬件加速指南

本文档说明：**开发机（RTX 4060）CUDA**、**RealSense D455i**、**1260P 无显卡工控机**、**EtherCAT 整机联调** 等部署路径。  
代码默认 **`position.mode=bbox`（传统相机）**；深度相机装好驱动后切到 **`depth`** 即可，无需改 C++ 逻辑。

---

## 架构概览

```
                    ┌─────────────────────────────────────┐
  传统 RGB 模式      │  YOLO(color) → bbox 估深 → TF → KF  │
  position.mode=bbox │  视频 / MindVision / 普通 USB 相机   │
                    └─────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
  RGB-D 模式         │  YOLO(color) → 对齐 depth 采样 → TF │
  position.mode=depth│  Intel RealSense D455i 等           │
                    └─────────────────────────────────────┘
                              ↓
                    卡尔曼滤波 → 物理轨迹预测 → ROS 话题
```

两条路径 **共用同一套** `ball_detector_node`；仅 3D 测量来源不同。

> **完整数学模型**（KF 矩阵、阻力积分、端到端时序图）：见仓库根目录 [README.md §方法论](../../../README.md#方法论与数学模型)  
> **参数怎么调**：见 [readme.md](../readme.md)

---

## bbox 与 depth 模式说明

| 已知 | 从哪来 |
|------|--------|
| 框的中心在画面哪 | YOLO 检测框 |
| 框有多高（像素） | YOLO 检测框 |
| 排球真实直径（约 21 cm） | 配置文件里写死 |
| 相机焦距 | `/camera_info` 内参 |

对应代码里的公式（详见根 README）：

$$
Z \approx \frac{f_y \cdot D}{h_\text{px}}
$$

**优点**：任意 RGB 相机都能用，不用深度相机。  
**缺点**：框抖一点，算出来的距离就抖；框不准，距离就不准。

### depth 模式（`position.mode=depth`）
**优点**：3D 通常比 bbox 模式稳。  
**缺点**：要多一路深度相机和驱动；快球、远距时深度可能有空洞。

### 和 debug 画面上的字对应关系

| 画面左上角 | 含义 |
|------------|------|
| `MODE: RGB/BBOX` | 普通相机，用检测框估距离 |
| `MODE: RGB-D` | 深度相机，用深度图测距离 |

---

## 开发环境 vs 实战环境检查表

上机 / 比赛前勾一遍（PC 上测通 ≠ 实战就绪）。

### A. 软件与性能

- [ ] 在 **最终比赛机**（Jetson / 1260P / 视觉专机）上 `colcon build` 通过
- [ ] 实测 **该机器上** 的 pose 间隔（不是 4060 上的 fps）
- [ ] 无显卡机：`YOLO_DEVICE=cpu`，并按 [§六](#六无显卡--cpu-推理与模型优化) 评估是否需小模型 / OpenVINO / 视觉双机
- [ ] 有 GPU 时：确认 `yolo.device` 未 silent 回退 CPU
- [ ] 下游（1260P 控制机）能收到 **`/ball_intercept`**（或兼容 `/volleyball_pose`）

### B. 相机与 3D

- [ ] 确认用的是 **bbox** 还是 **depth** 模式，和实际硬件一致
- [ ] **不是**视频假内参：实战用真实 `/camera_info` 或标定结果
- [ ] depth 模式：`/camera/camera/aligned_depth_to_color/image_raw` 有数据，框中心能采到非 0 深度
- [ ] 球在工作距离内（D455i 大致 0.6–6 m，视配置而定）

### C. 坐标系（最容易漏）

- [ ] 小车模式：`use_static_camera_tf:=false`，TF 链 `odom→base_link→camera_link→…` 由底盘+URDF 提供
- [ ] 不再用占位 static TF（“相机在 odom 上方 1 m”那种）
- [ ] `world_frame_id` 和队伍约定的场地/机器人坐标一致
- [ ] 重力方向对：落点不会总往地下穿或往天上飞

### D. 场景

- [ ] 用 **接近比赛** 的环境测过（顶灯、背景人、球网）
- [ ] 误检、丢检在可接受范围，KF 不会经常 reset
- [ ] 和队友对齐：检测 10–15 Hz 够不够、落点误差容忍多少

---

## 一、开发机 RTX 4060：启用 GPU 推理

### 1.1 现状

- YOLO 使用 **OpenCV DNN + ONNX**（`yolo_inference.cpp`）
- 参数 `yolo.device`：`auto` | `cpu` | `cuda`
- **Ubuntu apt 自带的 OpenCV 通常不带 CUDA**，`auto` 会 silently 回退 CPU

### 1.2 确认当前是否在 CPU 跑

```bash
colcon build --symlink-install --packages-select station_detector_cpp
source install/setup.bash

# 编译时看 CMake 输出：
#   "OpenCV CUDA detected" → 可用 cuda
#   "OpenCV CUDA not found" → 只能 CPU

ros2 launch station_detector_cpp yolo.launch.py \
  pipeline_mode:=video \
  yolo_device:=cuda \
  frame_rate:=15
```

终端若 `Frame dt` 经常 **> 0.1 s**，说明仍在 CPU 或模型过大。

### 1.3 路径 A：OpenCV + CUDA（与现有代码最贴合）

**4060 开发机**（需已装 NVIDIA 驱动 + CUDA Toolkit + cuDNN）：

```bash
# 1) 查 CUDA 版本
nvidia-smi

# 2) 从源码编译 OpenCV（示例，版本按本机 CUDA 调整）
git clone --depth 1 -b 4.x https://github.com/opencv/opencv.git
git clone --depth 1 -b 4.x https://github.com/opencv/opencv_contrib.git

cd opencv && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=Release \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D WITH_CUDA=ON \
  -D WITH_CUDNN=ON \
  -D OPENCV_DNN_CUDA=ON \
  -D ENABLE_FAST_MATH=ON \
  -D CUDA_FAST_MATH=ON \
  -D BUILD_EXAMPLES=OFF \
  ..
make -j$(nproc)
sudo make install
sudo ldconfig

# 3) 重新编译本包（CMake 会自动定义 HAVE_CUDA）
cd ~/volleyball_detection
colcon build --symlink-install --packages-select station_detector_cpp
```

运行：

```bash
ros2 launch station_detector_cpp yolo.launch.py pipeline_mode:=video yolo_device:=cuda
```

### 1.4 路径 B：Jetson 部署（推荐上机方案）

| 步骤 | 说明 |
|------|------|
| 1 | 在 4060 上完成算法验证（bbox / depth 两模式） |
| 2 | 代码 push 到 Jetson，`colcon build` |
| 3 | JetPack 自带 CUDA OpenCV，设 `yolo.device:=cuda` |
| 4 | **TensorRT**：ONNX → `.engine` 需在 **Jetson 本机** 生成，不能从 4060 拷贝 |
| 5 | RealSense：`sudo apt install ros-humble-realsense2-camera` |

Orin Nano 8G 经验目标：**YOLOv8n/s @640 + TensorRT → 20–30 fps**。

### 1.5 4060 vs Jetson 注意

- **TensorRT engine 不能跨设备共用**（SM 架构不同）
- 4060 用于开发调试；**比赛机以 Jetson 实测 fps 为准**
- IMU（D455i 内置）后续可接 `robot_localization`，第一版可忽略

---

## 二、模式切换（video / realsense）

### 2.1 一键启动

编辑 `config/pipeline.conf`：

```bash
USE_REALSENSE=false   # 视频 bbox
USE_REALSENSE=true    # RealSense depth
YOLO_DEVICE=cuda
```

然后：

```bash
./start_all.sh
```

### 2.2 参数文件

| 模式 | YAML | 关键字段 |
|------|------|----------|
| video | `src/station_detector_cpp/config/ball_detector_params_video.yaml` | `position.mode: bbox` |
| realsense | `src/station_detector_cpp/config/ball_detector_params_realsense.yaml` | `position.mode: depth` + RealSense 话题 |

统一 launch：

```bash
ros2 launch station_detector_cpp yolo.launch.py pipeline_mode:=video
ros2 launch station_detector_cpp yolo.launch.py pipeline_mode:=realsense
```

### 2.3 视频验证

```bash
ros2 launch station_detector_cpp yolo.launch.py \
  pipeline_mode:=video \
  video_path:=/abs/path/to/video.mp4 \
  model_path:=/abs/path/to/best.onnx \
  frame_rate:=15.0 \
  yolo_device:=auto
```

关键参数见 `ball_detector_params_video.yaml`：

| 参数 | 默认 | 说明 |
|------|------|------|
| `position.mode` | `bbox` | bbox 高度估深 |
| `detection.min_confidence` | `0.18` | 检测置信度 |
| `yolo.device` | `auto` | CPU / CUDA |

### 2.4 观测

```bash
ros2 topic echo /ball_intercept
ros2 topic echo /volleyball_pose
ros2 topic hz /camera/camera/color/image_raw    # realsense 模式
# RViz 看 /debug_image（MODE: RGB/BBOX 或 RGB-D）
```

### 2.5 时间戳

`/volleyball_pose` 与 `/ball_intercept` 的 `header.stamp` 为**观测时刻**；`event_time = stamp + time_to_event` 为预计到达拦截高度时刻。

---

## 三、RealSense D455i 深度模式（驱动就绪后启用）

### 3.1 安装驱动

**报错 `Unable to locate package librealsense2-dev`？**  
该包在 **Intel 官方 apt 源**，不在 Ubuntu 默认源里。用项目脚本一键装：

```bash
bash scripts/install_realsense_deps.sh
```

脚本会：
1. 装 `ros-humble-realsense2-camera`（跑 `./start_all.sh` 必需）
2. 添加 Intel 源，装 `librealsense2-utils`（含 `realsense-viewer`）和内核驱动

**手动分步（可选）：**

```bash
# A) 仅 ROS 驱动（最小，无 realsense-viewer）
sudo apt update
sudo apt install ros-humble-realsense2-camera ros-humble-diagnostic-updater

# B) Intel 源 + 调试工具（realsense-viewer）
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
  | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt update
sudo apt install librealsense2-dkms librealsense2-utils librealsense2-dev

realsense-viewer   # 插上 D455i，确认 RGB + Depth
```

### 3.2 启动

```bash
# 推荐：改 config/pipeline.conf → USE_REALSENSE=true
./start_all.sh

# 或显式 launch
ros2 launch station_detector_cpp yolo.launch.py pipeline_mode:=realsense
```

自动加载 `ball_detector_params_realsense.yaml`（**注意**：ROS2 驱动默认命名空间为 `/camera/camera/...`）：

| 参数 | RealSense 值 |
|------|----------------|
| `position.mode` | `depth` |
| `input.image_topic` | `/camera/camera/color/image_raw` |
| `input.camera_info_topic` | `/camera/camera/color/camera_info` |
| `position.depth_topic` | `/camera/camera/aligned_depth_to_color/image_raw` |
| `camera_frame_id` | `camera_color_optical_frame` |

### 3.3 手动切换（已有 realsense 节点在跑）

```bash
ros2 run station_detector_cpp ball_detector_node \
  --ros-args \
  --params-file src/station_detector_cpp/config/ball_detector_params_realsense.yaml \
  -p yolo.model_path:=/abs/path/to/best.onnx
```

（YAML 内话题已指向 `/camera/camera/...`；若你的 launch 用了不同 namespace，需相应改 YAML。）

### 3.4 深度采样逻辑

- 在 YOLO 框中心取 **对齐深度图** 像素
- 支持 `16UC1`（RealSense 毫米）和 `32FC1`（米）
- 小 patch **中值** 滤波（`position.depth_patch_radius`）
- RGB/Depth 时间戳差 > `depth_max_stamp_delta_sec` 则丢弃该帧深度

### 3.5 D455i 注意

- 有效深度大约 **0.6–6 m**（视配置），排球高远球可能超量程
- 快速运动深度可能有空洞 → KF 会 predict，属正常现象
- **静态 TF 是占位**（相机 z=1 m）；上机器人后换成 URDF/标定外参
- IMU 第一版未接入，后续可用于相机运动补偿

---

## 四、推荐工作流

```
阶段 1  [现在]  4060 + 视频 + bbox 模式
         → 确认 YOLO、KF、轨迹链路
         → 尽量启用 CUDA，目标 Frame dt < 50 ms

阶段 2  [D455i 已接]  USE_REALSENSE=true + depth 模式
         → 驱动/话题已验证；有球时对比 bbox vs depth 的 Z 抖动与落点

阶段 3  [定比赛主控]  见 §五：Jetson **或** 1260P 工控机 **或** 视觉/控制双机
         → 板上 colcon build + 实测 fps（无显卡见 §六）

阶段 4  [上机器人]  换真实 TF、标定、对接 EtherCAT/颠球机构、调 max_physical_speed
```

---

## 五、机器人整机架构（EtherCAT / 1260P / 颠球）

视觉仓库只负责 **看球 → `/volleyball_pose` / `/ball_intercept`**；接球机构、颠球、EtherCAT 主站由队友运动控制组实现。联调前建议对齐下面分工。

### 5.1 「以太猫」是什么？

队友说的 **以太猫** 一般指 **EtherCAT**（工业实时以太网）：主站（通常是 x86 工控机）以 **1 ms 甚至更短周期** 与伺服驱动器、IO 模块通信。

| 接法 | 有没有 STM32「下位机」 | 说明 |
|------|------------------------|------|
| **EtherCAT 伺服直驱** | 往往**没有**单独 MCU | 1260P 当 **EtherCAT 主站**；驱动器/IO 是从站 |
| **CAN + STM32** | **有** | 上位机发目标 → 下位机跑电机环 + 颠球时序 |
| **混合** | **部分有** | 关节走 EtherCAT；颠球气缸/步进走 **IO 模块或单独 MCU** |

**不是「只剩一台小电脑」**，而是：传统下位机角色可能由 **伺服驱动 + EtherCAT IO** 承担；**颠球机构**仍要单独接 IO 或 MCU。

### 5.2 1260P 工控机 vs Jetson

| | **1260P x86 工控机** | **Jetson Orin 等** |
|--|----------------------|---------------------|
| 典型用途 | EtherCAT 主站 + ROS2 控制 +（可选）视觉 | GPU 视觉、边缘 AI |
| 与 EtherCAT | 队里栈成熟（PREEMPT_RT + 主站软件） | 能做但门槛高，很多队不用 |
| 视觉 YOLO | **无独显则纯 CPU**，偏慢（见 §六） | TensorRT，15–30 fps 可期 |
| 本仓库编译 | `x86_64 colcon build`（与 4060 同架构） | `aarch64` 需重编 |

**1260P 够控、不够爽地跑 YOLO**：运动 + 颠球 + EtherCAT 通常 OK；**同机再跑 640 YOLO** 容易只有 **2–6 Hz**。

### 5.3 推荐整机拓扑

**方案 A：双机（视觉 + 控制，最稳）**

```text
[视觉机] 4060 笔记本 / 带 GPU 的小主机 / Jetson
  RealSense + ball_detector_node
  发布 /volleyball_pose、/ball_intercept（局域网 ROS2）

[控制机] 1260P（EtherCAT 主站）
  订阅视觉话题 → 接球机构 + 颠球 IO
  不跑 YOLO
```

**方案 B：单机 1260P（省设备，fps 要实测）**

```text
1260P：EtherCAT + ROS2 + RealSense + CPU YOLO + 颠球
```

适合：颠球间隔 ≥1 s、检测 **≥5 Hz** 可接受、模型已按 §六 缩小。

**方案 C：1260P 控制 + 独立 MCU 颠球**

颠球硬节拍 / 气路安全 → STM32 管颠球；1260P 管 EtherCAT 与接球逻辑。

### 5.4 颠球与视觉的时序

```text
颠球触发 → 球出手 → 相机跟踪 → KF/轨迹 → 落点 → 接球机构
```

- 视觉节点 **不驱动颠球**；全队需约定颠球时刻与 `header.stamp` 是否对齐。
- 接球准备时间 + 颠球间隔决定 **最低检测 Hz**（见 §六 帧率表）。
- 下游优先订阅 **`/ball_intercept`**（拦截点 + 到达时间 + 速度）；兼容 **`/volleyball_pose`**（当前 3D）。

### 5.5 联调前问队友的三句话

1. 1260P **要不要同机跑 YOLO**，还是只跑 EtherCAT + 接球？  
2. 颠球接 **EtherCAT IO** 还是 **STM32/CAN**？  
3. 颠球间隔、接球准备时间各多少 → 定最低 fps。

---

## 六、无显卡 / CPU 推理与模型优化

目标平台：**1260P 等无 NVIDIA 独显**的工控机。当前代码路径：**OpenCV DNN + ONNX**（`yolo_inference.cpp`），输入 **硬编码 640×640**。

### 6.1 帧率粗估

| 配置 | 大致 `/volleyball_pose` 频率 |
|------|------------------------------|
| 现有模型 @640，OpenCV CPU | ~2–5 Hz |
| **YOLOv8n @416**，OpenCV CPU | ~5–10 Hz |
| **YOLOv8n @416 + OpenVINO** | ~10–18 Hz |
| 跳帧（每 2 帧检 1 次）+ KF 补中间 | 跟踪更密，检测 Hz ×2 体感 |
| 4060 + CUDA（对照） | ~6–8 Hz（已测） |
| Jetson Orin + TensorRT | ~15–30 Hz（模型小） |

颠球 **1–2 s 一颗** 时，**8 Hz 检测 + KF** 往往可用；连续快颠目标 **15 Hz+**。

### 6.2 重新训练：把模型做小（推荐）

| 项 | 建议 |
|----|------|
| 骨干 | **YOLOv8n**（勿用 s/m） |
| 类别 | **单类 `volleyball`**（`single_cls=True`） |
| 输入 | 先 **416**，够再试 **320** |
| 数据 | RealSense 实拍、顶光、球网、运动模糊、远近距离 |
| 数量 | 训练 800+ / 验证 150+（越多越好） |

**训练示例（Ultralytics，单独训练项目即可）：**

```bash
pip install ultralytics

yolo detect train \
  model=yolov8n.pt \
  data=volleyball.yaml \
  imgsz=416 \
  epochs=150 \
  batch=16 \
  patience=30 \
  single_cls=True
```

`volleyball.yaml` 示例：

```yaml
path: /path/to/dataset
train: images/train
val: images/val
nc: 1
names:
  0: volleyball
```

**导出 ONNX（与 C++ 对齐）：**

```bash
yolo export model=runs/detect/train/weights/best.pt format=onnx imgsz=416 opset=12 simplify=True
```

⚠️ 导出 **416** 后须把 `yolo_inference.cpp` 里 `inp_w/inp_h` 改为 **416**（或后续做成 yaml 参数）。  
更完整的数据集说明见 [MODEL_TRAINING_BRIEF.md](MODEL_TRAINING_BRIEF.md)（**可直接转发给训模型同学**）。

**训练注意：**

- 含 **运动模糊、部分出画、误检背景**（灯、人头）。  
- 验证 **远距离 recall**，不只看 mAP。  
- 新 ONNX 先在 **4060 同一视频** 上与旧 `best.onnx` 对比，再上 1260P 测 Hz。

### 6.3 Intel 1260P：OpenVINO（优先尝试）

1260P 无独显时，**OpenVINO 跑 ONNX** 常比 OpenCV 默认 CPU **快 2–4 倍**。

- OpenCV 编译带 OpenVINO 时，可设 `DNN_BACKEND_INFERENCE_ENGINE` + `DNN_TARGET_CPU`。  
- 或 **ONNX Runtime + OpenVINO EP**（需额外集成，当前仓库尚未内置）。

工控机上先验证 OpenVINO 环境，再测同模型 fps；**代码侧 OpenVINO 开关**可作为后续 PR。

### 6.4 不改训练也能做的

| 举措 | 做法 |
|------|------|
| 降相机分辨率 | RealSense 彩色 **640×480@30** |
| 明确 CPU | `config/pipeline.conf` → `YOLO_DEVICE=cpu` |
| 跳帧检测 | `yolo.detect_min_interval_sec`（默认 0.10s）| 中间帧 KF 按真实 dt 预测，**已实现** |
| ROI 跟踪 | 有框后只裁局部 patch 推理（**待实现**） |
| CPU 绑核 | EtherCAT 与视觉分 P 核，避免互相拖死 |

### 6.5 推荐优化顺序

1. 训 **YOLOv8n 单类 @416** → 导出 ONNX → 4060 验效果  
2. 1260P 装 **OpenVINO** → 测 fps  
3. 仍不够 → 跳帧 + ROI（代码）或 **视觉专机**（§5.3 方案 A）

---

## 七、常见问题

**Q: 深度模式 debug 仍显示 Searching，但 YOLO 有框？**  
A: 多为 depth 采样失败（话题路径错、未对齐、时间戳不同步、深度为 0）。查：

```bash
ros2 topic hz /camera/camera/color/image_raw
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
```

若话题名是 `/camera/color/...` 而非 `/camera/camera/...`，改 YAML 或 launch 的 namespace。

**Q: Velocity gate 频繁触发？**  
A: 增大 `detection.max_physical_speed`，或确认 `header.stamp` 正常；检测太稀疏时先提 fps。

**Q: 想临时跳过所有过滤 debug YOLO？**  
A: `-p detection.emergency_bypass:=true`（仅调试，不建议比赛用）。

**Q: 4060 上 cuda 设了还是慢？**  
A: 看编译日志是否有 `OpenCV CUDA detected`；没有则需按 §1.3 重编 OpenCV。

**Q: 队友要用 EtherCAT / 1260P，不用 Jetson？**  
A: 见 [§五](#五机器人整机架构ethercat--1260p--颠球)。视觉可双机；单机无显卡见 §六。

**Q: 1260P 无显卡，fps 太低怎么办？**  
A: YOLOv8n @416 重训 + OpenVINO；仍不够则视觉专机或等跳帧/ROI 功能。

---

## 八、文件索引

| 文件 | 作用 |
|------|------|
| `config/pipeline.conf` | `USE_REALSENSE` / `YOLO_DEVICE` 一键切换 |
| `src/station_detector_cpp/config/ball_detector_params_video.yaml` | 视频 bbox 模式参数 |
| `src/station_detector_cpp/config/ball_detector_params_realsense.yaml` | RealSense depth 模式参数 |
| `launch/yolo.launch.py` | **统一入口**（`pipeline_mode:=video\|realsense` 或 `./start_all.sh`） |
| `src/ball_detector_node.cpp` | 双模式主节点 |
| `src/ball_tracker.cpp` | 6 状态卡尔曼 |
| `src/trajectory_predictor.cpp` | 重力 + 二次阻力积分 |
| `include/ball_position_estimator.hpp` | bbox / depth 两种 3D 反投影 |
| 根目录 `README.md` | 方法论、公式、数据流图 |
| `readme.md` | 参数调优速查 |
