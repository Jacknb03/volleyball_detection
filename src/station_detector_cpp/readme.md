# 参数调优手册

> 硬件部署 / CUDA / RealSense：见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)  
> 从零调试：见 [docs/DEBUGGING.md](docs/DEBUGGING.md)

## 核心链路

1. **YOLO** — 2D 像素框  
2. **Estimator** — bbox 估深 或 深度图采样 → 3D  
3. **TF** — 相机系 → `odom`  
4. **Kalman** — 平滑 + 速度  
5. **Predictor** — 重力 + 阻力 → 落点  

---

## 模式相关参数

| 模式 | YAML | 关键参数 |
|------|------|----------|
| 视频 / RGB | `ball_detector_params_video.yaml` | `position.mode: bbox`, `camera_frame_id: camera_optical_frame` |
| RealSense | `ball_detector_params_realsense.yaml` | `position.mode: depth`, 深度话题已配好 |

切换方式：编辑根目录 `config/pipeline.conf` → `USE_REALSENSE=true/false`，然后 `./start_all.sh`

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

## 二、平滑与防抖

| 参数 | 默认 | 调参 |
|------|------|------|
| `detection.h_ema_alpha` | 0.3 | 越小越稳（0.15–0.2），越大越灵敏（0.4–0.5） |
| `kalman.measurement_noise` | 50.0 | 越大越不信 YOLO 抖动 |
| `detection.max_physical_speed` | 25.0 | 轨迹瞬移→调小（15–20） |
| `detection.max_jump_distance` | 100.0 | 视频循环 re-acquire；可略增到 120–150 |
| `detection.min_confidence` | 0.18 | 与 `yolo.conf_threshold` 保持一致 |

---

## 三、物理预测

| 参数 | 默认 | 调参 |
|------|------|------|
| `drag_coefficient` | 0.47 | 落点偏远→调大（0.5–0.6） |
| `volleyball.mass_kg` | 0.27 | 一般不动 |
| `trajectory.ground_z` | 0.0 | 地面高度 |

---

## 四、常见问题

**A. 轨迹反应慢** → 增大 `h_ema_alpha`，减小 `measurement_noise`

**B. 轨迹抖、乱指** → 减小 `h_ema_alpha`，增大 `measurement_noise`，减小 `max_physical_speed`

**C. 抛物线垂直下落** → 检查 Velocity 日志、static TF pitch、`world_frame_id`

**D. 远距离检不出** → 降低 `yolo.conf_threshold`（如 0.12）

**E. 视频循环后轨迹消失** → 已自动 reset；稍等 1–2 帧 re-acquire

**F. depth 模式有框无 pose** → 查 `ros2 topic hz /camera/aligned_depth_to_color/image_raw`，深度是否为 0

---

## 五、工作流

1. `./start_all.sh`，RViz 看 `/debug_image`  
2. 确认绿框 + `KF: OK`  
3. 调轨迹：先求稳（噪声↑ alpha↓），再求准（diameter / drag）  
4. depth 模式：接 D455i 后重复，对比 Z 抖动  

改 YAML 后：`./stop_all.sh && ./start_all.sh`

当前 PC 上 pose 约 **6–8 Hz** 属正常；以落点稳定性为先，fps 上 Jetson 再优化。
