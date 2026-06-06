# 参数调优手册

> **数学模型与公式**（KF 矩阵、阻力积分、数据流图）：见仓库根目录 [README.md](../../README.md#方法论与数学模型)  
> 硬件部署 / CUDA / RealSense / **1260P·EtherCAT** / **无显卡优化**：见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)（§五–§六）  
> 从零调试：见 [docs/DEBUGGING.md](docs/DEBUGGING.md)

## 核心链路

1. **YOLO** — 2D 像素框  
2. **DetectionFilter** — 像素跳变 / 置信度 / 连续帧门控（见 `detection_filter.cpp`）  
3. **Estimator** — bbox 估深 或 深度图采样 → 3D  
4. **TF** — 相机系 → `odom`  
5. **Kalman** — 6 状态 CV 模型，平滑 + 速度（见 `ball_tracker.cpp`）  
6. **Predictor** — 重力 + 二次阻力 → 落点（见 `trajectory_predictor.cpp`）  

---

## 模式相关参数

| 模式 | YAML | 关键参数 |
|------|------|----------|
| 视频 / RGB | `ball_detector_params_video.yaml` | `position.mode: bbox`, `camera_frame_id: camera_optical_frame` |
| RealSense | `ball_detector_params_realsense.yaml` | `position.mode: depth`，话题 `/camera/camera/...` |

切换方式：编辑根目录 `config/pipeline.conf` → `USE_REALSENSE=true/false`，然后 `./start_all.sh`

---

## 零、公式与参数的对应关系（调参速查）

完整推导见根 [README.md §方法论](../../README.md#方法论与数学模型)。这里只列**调参时该动哪几个旋钮**：

| 现象 | 优先改 | 对应公式/逻辑 |
|------|--------|---------------|
| bbox 距离系统性偏大/偏小 | `volleyball.diameter` | $Z = f_y D / h_\text{px}$ |
| 框高抖动 → Z 抖 | `detection.h_ema_alpha` ↓ | EMA 平滑 $h_\text{px}$ |
| 3D 位置抖、轨迹乱指 | `kalman.measurement_noise` ↑ | $\mathbf{R} = \sigma_r I_3$ |
| 跟不上加速、反应慢 | `kalman.process_noise` ↑ 或 `h_ema_alpha` ↑ | $\mathbf{Q}$ 速度块更大 |
| 测量瞬移、KF reset | `detection.max_physical_speed` ↓ | $\|\Delta z\|/\Delta t > v_\text{max}$ |
| 视频循环后长时间无轨迹 | `detection.max_jump_distance` ↑ | DetectionFilter 跳变 reset |
| 落点偏远、弧太「抛」 | `drag_coefficient` ↑ | $a = -k\|v\|v$，$k \propto C_d$ |
| depth 有框无 pose | 查深度话题 / 量程 | `depth_min_m`–`depth_max_m`，patch 中值 |

**源码**：`ball_tracker.cpp`（KF）、`trajectory_predictor.cpp`（阻力）、`ball_position_estimator.hpp`（3D）。

---

## 一、深度与空间

| 参数 | 默认 | 说明 |
|------|------|------|
| `volleyball.diameter` | 0.21 | **bbox 模式最重要**。Z 偏大→调小；偏小→调大 |
| `volleyball.min_depth` | 0.2 | 过滤过近误检 |
| `position.depth_min_m` / `depth_max_m` | 0.3 / 8.0 | **depth 模式**有效量程 |
| `position.depth_patch_radius` | 2 | 深度中值滤波半径 |
| `world_frame_id` | odom | 与 TF 一致 |
| static TF `z` | 1.0 | 占位相机高度；上机器人后换标定 |

---

## 二、平滑与防抖（卡尔曼 + 2D 门控）

| 参数 | 默认 | 调参 |
|------|------|------|
| `detection.h_ema_alpha` | 0.3 | 越小越稳（0.15–0.2），越大越灵敏（0.4–0.5） |
| `kalman.process_noise` | 0.05 | ↑ 更跟得上加速；↓ 轨迹更平滑 |
| `kalman.measurement_noise` | 50.0 | 越大越不信 YOLO/深度抖动（对应 $\mathbf{R}$） |
| `kalman.max_missing_frames` | 15 | 连续无检测 reset 阈值 |
| `detection.max_physical_speed` | 25.0 | 轨迹瞬移→调小（15–20） |
| `detection.max_jump_distance` | 100.0 | 视频循环 re-acquire；可略增到 120–150 |
| `detection.min_consistent_detections` | 1 | ↑ 可减误检，但启动慢 1–2 帧 |
| `detection.min_confidence` | 0.18 | 与 `yolo.conf_threshold` 保持一致 |

---

## 三、物理预测（二次阻力）

默认 $k \approx 0.037\,\text{s}^{-1}$（$D=0.21$ m, $m=0.27$ kg）。10 m/s 时阻力加速度约 3.7 m/s²，与重力同量级。

| 参数 | 默认 | 调参 |
|------|------|------|
| `drag_coefficient` | 0.47 | 落点偏远→调大（0.5–0.6） |
| `air_density` | 1.225 | 室内一般不动 |
| `volleyball.mass_kg` | 0.27 | 一般不动 |
| `trajectory.integration_dt` | 0.01 | 一般不动 |
| `trajectory.ground_z` | 0.0 | 地面高度 |

---

## 四、常见问题

**A. 轨迹反应慢** → 增大 `h_ema_alpha`，减小 `measurement_noise`

**B. 轨迹抖、乱指** → 减小 `h_ema_alpha`，增大 `measurement_noise`，减小 `max_physical_speed`

**C. 抛物线垂直下落** → 检查 Velocity 日志、static TF pitch、`world_frame_id`

**D. 远距离检不出** → 降低 `yolo.conf_threshold`（如 0.12）

**E. 视频循环后轨迹消失** → 已自动 reset；稍等 1–2 帧 re-acquire

**F. depth 模式有框无 pose** → 查 `ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw`，深度是否为 0；排查时用 `--log-level debug`（PARADOX 已降为 DEBUG）

---

## 五、工作流

1. `./start_all.sh`，RViz 看 `/debug_image`  
2. 确认绿框 + `KF: OK`  
3. 调轨迹：先求稳（噪声↑ alpha↓），再求准（diameter / drag）  
4. depth 模式：接 D455i 后重复，对比 Z 抖动  

改 YAML 后：`./stop_all.sh && ./start_all.sh`

当前 PC（4060+CUDA）pose 约 **6–8 Hz**。1260P 无独显见 [DEPLOYMENT §六](docs/DEPLOYMENT.md#六无显卡--cpu-推理与模型优化)（YOLOv8n @416 + OpenVINO 目标 ~10–18 Hz）。
