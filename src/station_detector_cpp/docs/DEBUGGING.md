# 调试指南（从零到跑通）

面向：**先在 Ubuntu 上用视频把链路调通，再接 RealSense**。  
参数调优见 [readme.md](../readme.md)；部署 / 整机 / 无显卡优化见 [DEPLOYMENT.md](DEPLOYMENT.md)。

---

## 文档地图

| 文档 | 干什么用 |
|------|----------|
| **本文 DEBUGGING.md** | 一步步：怎么启动、怎么看、出问题查哪 |
| `readme.md` | 调参：轨迹飘、落点不准、反应慢 |
| `DEPLOYMENT.md` | CUDA、RealSense、**1260P/EtherCAT 整机**、**无显卡模型优化**（§五–§六） |
| 根目录 `README.md` | 架构、模式切换、后续计划 |

---

## 第 0 步：准备

### 需要有的文件

- [ ] ONNX 模型：`src/station_detector_cpp/model/best.onnx`
- [ ] 测试视频：`src/station_detector_cpp/videos/test.mp4`

### 环境

```bash
source /opt/ros/humble/setup.bash
cd ~/volleyball_detection
source install/setup.bash
```

### 编译

```bash
colcon build --symlink-install --packages-select station_detector_cpp
source install/setup.bash
```

---

## 第 1 步：启动

**推荐：编辑 `config/pipeline.conf` 后 `./start_all.sh`**

```bash
# config/pipeline.conf
USE_REALSENSE=false
YOLO_DEVICE=cuda
./start_all.sh
```

**或手动 launch：**

```bash
ros2 launch station_detector_cpp yolo.launch.py \
  pipeline_mode:=video \
  video_path:=/绝对路径/test.mp4 \
  model_path:=/绝对路径/best.onnx \
  frame_rate:=15.0 \
  yolo_device:=auto
```

### 看图像 — 用 RViz（不用 rqt）

`start_all.sh` 会自动加载 `config/volleyball_debug.rviz`：

- **Fixed Frame** = `base_link`
- **Image** → `/debug_image`
- **MarkerArray** → `/volleyball_trajectory`

若手动开 RViz：`rviz2 -d config/volleyball_debug.rviz`

### 启动成功长什么样

`/debug_image` 左上角：

| 显示 | 含义 |
|------|------|
| `MODE: RGB/BBOX` | 视频 bbox 估深模式 |
| `MODE: RGB-D` | RealSense 深度模式 |
| `SYSTEM ACTIVE` | 节点在跑 |
| `KF: OK` | 卡尔曼已初始化 |
| `Searching...` | 还没检测到球 |

---

## 第 2 步：分层排查

```
[1 有图?] → [2 YOLO 有框?] → [3 KF 有数?] → [4 轨迹/落点?] → [5 够不够快?]
```

### [1] 图像有没有进来？

**video 模式：**

```bash
ros2 topic hz /image_raw    # 约 15 Hz
```

**realsense 模式：**

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/aligned_depth_to_color/image_raw
```

### [2] YOLO 有没有框？

看 `/debug_image`：黄色框 = 原始检测，绿色框 + 红圈 = 当前跟踪球。

不出框 → 降低 `yolo.conf_threshold`（当前 0.18），见 `config/ball_detector_params_video.yaml`。

### [3] KF 有没有跟上？

```bash
ros2 topic hz /volleyball_pose    # 有球时应 > 0
ros2 topic echo /volleyball_pose --once
```

- 有框但 `KF: Not initialized` → TF 失败或 3D 估计失败（看终端 `PARADOX` / `TF transform failed`）
- **视频循环后轨迹消失** → 已修复自动 reset；若仍出现，检查 `max_physical_speed`

### [4] 拦截预测 / 轨迹有没有？

```bash
ros2 topic hz /volleyball_trajectory
ros2 topic echo /ball_intercept
```

RViz 中应看到橙色抛物线和绿色拦截点球；debug 画面有 `Intercept: ... in X.XXs`。

### [5] 够不够快？

```bash
ros2 topic hz /volleyball_pose
```

- 视频 15 Hz 输入，pose 约 **6–8 Hz** → YOLO 推理是瓶颈（正常）
- 目标：开发阶段 ≥ 6 Hz 可用；实战见 [DEPLOYMENT §六](DEPLOYMENT.md#六无显卡--cpu-推理与模型优化)（1260P 小模型 / OpenVINO / Jetson TensorRT）

加速：`YOLO_DEVICE=cuda ./start_all.sh` 或 `FRAME_RATE=10.0`

---

## 第 3 步：改参数怎么生效

改 `config/ball_detector_params_video.yaml` 或 `ball_detector_params_realsense.yaml` 后：

```bash
./stop_all.sh
./start_all.sh
```

临时覆盖：

```bash
ros2 launch station_detector_cpp yolo.launch.py \
  pipeline_mode:=video \
  params_file:=/path/to/custom.yaml
```

---

## 第 4 步：常用命令

```bash
ros2 node list
ros2 topic list
ros2 topic hz /volleyball_pose
ros2 topic echo /ball_intercept
```

---

## 故障清单

| 现象 | 排查 |
|------|------|
| 无图 | video_path 是否存在；RealSense 是否 `realsense-viewer` 能看 |
| 有图无框 | conf 太高、模型不对 |
| 有框无 pose | KF 未初始化；TF / 3D 失败 |
| 循环后无轨迹 | 等 1–2 秒 re-acquire；或调 `max_jump_distance` |
| pose 只有 2–3 Hz | 推理慢，非相机问题；见 DEPLOYMENT CUDA |
| rqt 崩溃 | **已弃用**，只用 RViz |

---

## 推荐接下来做什么

```
✅ 1. ./start_all.sh，RViz 看 /debug_image 有框
✅ 2. ros2 topic hz /volleyball_pose 有输出（6–8 Hz 可接受）
✅ 3. 轨迹/落点符合直觉，按 readme.md 微调参数
⬜ 4. bash scripts/install_realsense_deps.sh
⬜ 5. 接 D455i → PIPELINE_MODE=realsense ./start_all.sh
⬜ 6. 对比 bbox vs depth 的 Z 抖动和落点误差
⬜ 7. 上机器人：换真实 TF、标定、对接控制机（1260P/Jetson，见 DEPLOYMENT §五）
```

有问题时记录：`/debug_image` 截图、`ros2 topic hz` 输出、终端最后 20 行 log。
