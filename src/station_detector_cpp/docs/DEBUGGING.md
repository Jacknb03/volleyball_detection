# 调试指南（从零到跑通）

面向：**不太熟悉 ROS2，先在 Ubuntu 上用视频把系统调通**。  
参数调优见 [../readme.md](../readme.md)；CUDA / Jetson / RealSense 见 [DEPLOYMENT.md](DEPLOYMENT.md)。

---

## 文档地图

| 文档 | 干什么用 |
|------|----------|
| **本文 DEBUGGING.md** | 一步步：怎么启动、怎么看、出问题查哪 |
| `readme.md` | 调参：轨迹飘、落点不准、反应慢 |
| `DEPLOYMENT.md` | 4060 CUDA、D455i、上 Jetson 前的检查表 |
| 根目录 `README.md` | 编译、launch 命令速查 |

---

## 第 0 步：准备（5 分钟）

### 需要有的文件

- [ ] ONNX 模型：例如 `src/station_detector_cpp/model/best.onnx`
- [ ] 测试视频：例如 `src/station_detector/videos/test.mp4`

### 环境

```bash
# 每次新开终端都要 source
source /opt/ros/humble/setup.bash
cd ~/volleyball_detection   # 换成你的路径
source install/setup.bash
```

### 编译

```bash
colcon build --symlink-install --packages-select station_detector_cpp station_detector
source install/setup.bash
```

编译成功且无报错再继续。

---

## 第 1 步：启动（开 2 个终端）

**终端 1 — 跑系统**

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch station_detector_cpp yolo_cpp_video.launch.py \
  video_path:=/绝对路径/你的视频.mp4 \
  model_path:=/绝对路径/best.onnx \
  frame_rate:=15 \
  yolo_device:=auto
```

**终端 2 — 看图像**

```bash
source install/setup.bash
ros2 run rqt_image_view rqt_image_view
```

在 rqt 里选话题 **`/debug_image`**。

### 启动成功长什么样

`/debug_image` 左上角应看到：

- `MODE: RGB/BBOX`
- `SYSTEM ACTIVE`
- `Detections: ...`
- `KF: OK` 或 `KF: Not initialized`（刚启动时可能还未初始化）

若一片黑或 No image：跳到 [故障 A](#故障-ano-image--话题里没有图)。

---

## 第 2 步：分层排查（最重要）

整条链路像 5 节火车，**一节一节确认**，不要一上来就纠结落点准不准。

```
[1 有图?] → [2 YOLO 有框?] → [3 3D/KF 有数?] → [4 轨迹/落点?] → [5 够不够快?]
```

### [1] 图像有没有进来？

```bash
ros2 topic hz /image_raw
```

- 大约 **15 Hz**（你 launch 里 `frame_rate:=15`）→ 正常
- 没有输出 → 视频路径错、视频节点挂了，看终端 1 报错

### [2] YOLO 有没有框？

看 `/debug_image`：

| 现象 | 说明 |
|------|------|
| **黄色框** | YOLO  raw 检测 |
| **绿色框 + 红圈** | 当前选中的球 |
| 一直 `Searching...` | YOLO 没认出球 |

YOLO 不出框时：

1. 降低置信度：改 `config/ball_detector_params.yaml` 里 `yolo.conf_threshold`（如 `0.15`）
2. 确认模型是排球单类或类别名对：`yolo.volleyball_classes`
3. 临时跳过过滤：launch 加 `-p detection.emergency_bypass:=true`（仅调试）

### [3] KF 有没有跟上？

debug 画面：

- `KF: OK, Missing: 0` → 滤波在跑
- 长期 `KF: Not initialized` → 有框但 3D 没算出来（看终端有没有 `PARADOX` / `TF transform failed`）
- `Missing` 一直涨 → 检测断断续续（多半是 **推理太慢**，见第 5 步）

另开终端：

```bash
ros2 topic echo /volleyball_pose
```

有球时应持续输出 `position.x/y/z`。

### [4] 落点 / 轨迹有没有？

```bash
ros2 topic echo /ball_prediction
# x,y = 预测落地点，z = 落地时间

# RViz2（可选）
rviz2
# Fixed Frame 设 odom，Add → MarkerArray → /volleyball_trajectory
```

| 现象 | 先查什么 |
|------|----------|
| 没有 `/ball_prediction` | KF 未初始化，或速度全 0 |
| 落点总扎地底下 | 静态 TF 相机高度 `z=1.0` 是否靠谱；见 readme 场景 C |
| 抛物线乱指 | 检测太稀疏 / 框抖 → 先解决第 5 步性能 |

### [5] 够不够快？（你之前的主要问题）

```bash
# 图像在发，但处理跟不跟得上，看检测节点 CPU 占用 + 实际体验
ros2 topic hz /volleyball_pose
```

- **`/image_raw` 有 15 Hz，但 `/volleyball_pose` 只有 2–5 Hz** → **检测慢**，不是相机掉帧
- 目标：pose 更新 **≥ 10 Hz**（开发阶段），理想 15+

加速顺序：

1. `frame_rate:=10` 先降视频帧率，减轻压力
2. `yolo_device:=cuda`（需 OpenCV 带 CUDA，见 DEPLOYMENT §1）
3. 换更小模型（yolov8n）、或导出 320×320 ONNX
4. `debug.enable: false` 关调试图，省一点 CPU

---

## 第 3 步：改参数怎么生效

改 `src/station_detector_cpp/config/ball_detector_params.yaml` 后：

1. **Ctrl+C** 停掉 launch
2. 重新运行 launch（不必每次 colcon build，除非改了 C++）

launch 临时覆盖参数示例：

```bash
ros2 launch station_detector_cpp yolo_cpp_video.launch.py \
  video_path:=/path/to/video.mp4 \
  model_path:=/path/to/best.onnx \
  frame_rate:=15 \
  yolo_device:=auto
# 或在 ball_detector 节点参数里加：
#  -p yolo.conf_threshold:=0.2
#  -p debug.enable:=false
```

---

## 第 4 步：常用命令速查

```bash
# 有哪些节点 / 话题
ros2 node list
ros2 topic list

# 看图 / 位姿 / 落点
ros2 run rqt_image_view rqt_image_view          # 选 /debug_image
ros2 topic echo /volleyball_pose
ros2 topic echo /ball_prediction

# 频率
ros2 topic hz /image_raw
ros2 topic hz /volleyball_pose

# 看 TF（可选）
ros2 run tf2_tools view_frames
```

---

## 故障清单

### 故障 A：No image / 话题里没有图

- 检查 `video_path` 是否是 **绝对路径**、文件存在
- 终端 1 有无 `FileNotFoundError` / `无法打开视频`
- `ros2 topic list` 里有没有 `/image_raw`

### 故障 B：有图，永远没有框

- 模型路径错、模型不是排球
- `conf_threshold` 太高
- `-p detection.emergency_bypass:=true` 试一次

### 故障 C：有框，KF 不初始化

- 终端搜 `PARADOX`、`TF transform failed`
- `ros2 topic echo /camera_info` 有没有数据
- 视频模式会用 **假内参**，一般仍能初始化；若 TF 失败查 static_transform 节点是否在跑

### 故障 D：框闪、轨迹飘、速度怪

- **优先**解决检测频率低（第 2 步 [5]）
- 再调 `kalman.measurement_noise`、`detection.h_ema_alpha`（见 readme 场景 A/B）
- 确认已用 **图像时间戳**（当前代码已修复，需重新 colcon build）

### 故障 E：Velocity gate 警告刷屏

- 检测太稀疏导致相邻位置跳太大
- 临时：`detection.max_physical_speed` 调大（如 `30`）
- 根本：提高检测 fps

---

## 推荐接下来做什么（按顺序）

```
✅ 1. 按本文第 0–1 步跑起来，rqt 能看到 /debug_image
✅ 2. 确认 YOLO 能稳定出绿框（不行先调 conf / 模型）
✅ 3. ros2 topic echo /volleyball_pose 有连续输出
✅ 4. 看 /volleyball_pose 频率 — 若 <10Hz，按 DEPLOYMENT 开 CUDA 或降 frame_rate
✅ 5. RViz 看轨迹和落点，按 readme 调 diameter / kalman / drag
⬜ 6. D455i 驱动装好 → DEPLOYMENT 里 realsense launch
⬜ 7. 队友确认 Jetson → 板上重测 fps，换真实 TF
```

---

## 调试心态（给非专业同学）

1. **先确认「有没有框」，再管「准不准」**
2. **先确认「够不够快」，再管「轨迹漂不漂」** — 慢检测会让后面全崩
3. **一次只改一个参数**，否则不知道谁起作用
4. PC 视频测试 = 练链路；Jetson + 真相机 = 实战验收

有问题时，记录这三样发给队友/论坛：

- `/debug_image` 截图
- `ros2 topic hz /image_raw` 和 `/volleyball_pose` 的输出
- 终端里最后 20 行 log
