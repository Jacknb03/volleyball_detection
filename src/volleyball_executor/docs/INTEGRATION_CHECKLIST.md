# 视觉 → 执行端 联调检查清单

面向：**RealSense 已跑通、camera_link 暂未实测标定** 的阶段。  
按顺序勾选；每项给出命令、预期结果、异常时怎么查。

相关文档：[CALIBRATION.md](CALIBRATION.md)（标定）、[DEBUGGING.md](../../station_detector_cpp/docs/DEBUGGING.md)（视觉排查）、根 [README.md](../../../README.md)（架构）。

---

## 0. 前置条件

```bash
cd ~/volleyball_detection
source /opt/ros/humble/setup.bash
source install/setup.bash
# 工控机若装了 OpenCV 4.11 overlay，start_all.sh 会自动 source
```

- [ ] `config/pipeline.conf` 已按现场设置（`USE_REALSENSE=true`、`YOLO_INPUT_SIZE=416`、`YOLO_DEVICE=cpu`）
- [ ] 模型存在：`src/station_detector_cpp/model/best_416.onnx`
- [ ] 视觉节点已编译（工控机改 C++ 后：`bash scripts/rebuild_ipc.sh`）

**启动视觉 + RViz（不含桥接/UC）：**

```bash
./stop_all.sh && ./start_all.sh
```

---

## 1. 相机与图像（有图才能往下）

| 检查项 | 命令 | 预期 |
|--------|------|------|
| 彩色图 | `./run.sh ros2 topic hz /camera/camera/color/image_raw` | ~15–30 Hz |
| 对齐深度 | `./run.sh ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw` | ~15–30 Hz |
| Debug 图 | `./run.sh ros2 topic hz /debug_image` | ~12 Hz（限频，正常） |

**RViz**：Fixed Frame = `base_link`；DebugImage 有画面；左上角 `SYSTEM ACTIVE`。

异常：
- `Device or resource busy` → `./stop_all.sh`，必要时 `pkill -9 -f realsense`，再 `./start_all.sh`
- 无 debug 图但相机有 hz → 查 `ball_detector_node` 是否在跑：`./run.sh ros2 node list`

---

## 2. 检测与跟踪（理解「不是满速识别」）

| 环节 | 配置 | 实际表现 |
|------|------|----------|
| YOLO 推理 | `yolo.detect_min_interval_sec: 0.10` | 最高 **~10 Hz**，中间帧不跑推理 |
| KF 预测 | 每相机帧 | 框会「跟一小段」再随新检测跳变 |
| Debug 发布 | `debug.max_publish_hz: 12` | 仅影响 RViz 图像，**不影响** `/ball_intercept` |

| 检查项 | 怎么看 | 预期 |
|--------|--------|------|
| 有检测框 | RViz `/debug_image` | 黄框=原始检测，绿框+红圈=跟踪球 |
| KF 初始化 | debug 左上角 | `KF: OK`（有球时） |
| 3D 位姿 | `./run.sh ros2 topic hz /volleyball_pose` | 有球时 > 0，通常接近相机帧率 |
| 3D 红点 | `./run.sh ros2 topic hz /volleyball_ball_marker` | 有球且 KF OK 时有输出；RViz **BallPosition** 层勾选 |

```bash
./run.sh ros2 topic echo /volleyball_pose --once
```

预期（**数值仅作方向参考，未标定前勿当真**）：
- `header.frame_id: base_link`
- 球在相机前方时：`position.z` 多为正，随手上下移动球时 **z 应同向变化**
- 无球 / 丢检：topic 停更或 KF reset 后无输出

---

## 3. 视觉主输出 `/ball_intercept`（执行端优先订阅）

当前配置 `trajectory.enable: false` → **实时跟踪模式**，不是弹道 intercept。

```bash
# 频率
./run.sh ros2 topic hz /ball_intercept

# 看一条
./run.sh ros2 topic echo /ball_intercept --once

# 持续看（扔球时观察 xyz / valid）
./run.sh ros2 topic echo /ball_intercept
```

### 字段说明与预期

| 字段 | 跟踪模式下含义 | 有球时预期 |
|------|----------------|------------|
| `header.frame_id` | 坐标系 | `base_link` |
| `valid` | 是否有有效跟踪 | `true` |
| `position.x/y/z` | **当前球心**（非未来拦截点） | 随球移动连续变化（KF 预测帧也发） |
| `velocity.x/y/z` | KF 估计球速 | 静止球接近 0；抛球时有合理非零值 |
| `time_to_event` | 到「事件」的时间 | **≈ 0.0**（实时模式） |
| `event_time` | 事件时刻 | ≈ `header.stamp` |
| `intercept_z` | 拦截高度参考 | ≈ 当前 `position.z` |

### 手工验收（对着球做）

- [ ] **静止举球**：`valid: true`，xyz 大致稳定（允许小幅抖动）
- [ ] **上下移动球**：`position.z` 方向与运动一致
- [ ] **左右/前后移动球**：`position.x` 或 `position.y` 有相应变化
- [ ] **球出视野 / 遮挡**：数秒内 `valid` 变 `false` 或 topic 停更（视 `max_missing_frames`）
- [ ] **重新入画**：1–2 秒内恢复 `valid: true`

丢检或未 init 时：可能无消息，或 `valid: false`（若仍发布）。

---

## 4. 桥接层 `/vision/stewart_target`（本仓库内）

`start_all.sh` **不会**自动起桥接；需单独开终端：

```bash
./run.sh ros2 launch volleyball_executor executor.launch.py
```

| 检查项 | 命令 | 预期 |
|--------|------|------|
| 桥接节点 | `./run.sh ros2 node list \| grep intercept` | `/intercept_bridge_node` |
| Stewart 目标 | `./run.sh ros2 topic hz /vision/stewart_target` | 与 `/ball_intercept` 同量级（有球时） |
| 内容 | `./run.sh ros2 topic echo /vision/stewart_target` | 见下表 |

`StewartControl` 字段（`volleyball_msgs/StewartControl`）：

| 字段 | 含义 | 有球时预期 |
|------|------|------------|
| `x, y, z` | 机构目标点（经低轨道球面约束） | 在 `workspace_center ± R` 附近 |
| `pitch, yaw` | 机构中心指向球 | 球移动时角度跟随变化 |
| `roll` | 默认姿态 | ≈ `default_roll`（0） |
| `emergency_stop` | 0=正常，1=停 | 有有效 intercept 且 frame 正确时为 **0** |

约束参数见 `config/executor_params.yaml`：
- 球心 `(workspace_center_x/y/z)`，半径 `workspace_radius`（默认 R=0.12 m）
- `sphere_mode: on_shell` → 投影到球面，不是原样把球坐标发给 Stewart

异常：
- `emergency_stop: 1` 且 frame 报错 → `header.frame_id` 不是 `base_link`
- 有 intercept 但 stewart 不动 → UC 未起或未订阅 `/vision/stewart_target`（见第 5 步）

---

## 5. Universal Controllers / EtherCAT（队友侧）

本仓库**不包含** `volleyball_hub`；控制机在 `universal_controllers_v2-main/`（独立仓库）。

联调前与运动控制组确认：

1. UC 订阅的话题名是否为 **`/vision/stewart_target`**
2. 是否需 CH7 / 视觉模式开关
3. ROS_DOMAIN_ID、网络是否同一网段

控制机侧示例（在 UC 所在机器）：

```bash
ros2 topic hz /vision/stewart_target
ros2 topic echo /vision/stewart_target
# 若 UC 有反馈话题，按队友文档 echo
```

---

## 6. 端到端一次过（推荐顺序）

```
终端 A:  ./start_all.sh
终端 B:  ./run.sh ros2 topic echo /ball_intercept
终端 C:  ./run.sh ros2 launch volleyball_executor executor.launch.py
终端 D:  ./run.sh ros2 topic echo /vision/stewart_target
```

对着球缓慢移动，同时看 B、D：

- [ ] B 中 `valid: true`，xyz 随球变
- [ ] D 中 `emergency_stop: 0`，xyz/pitch/yaw 有响应
- [ ] RViz **BallPosition** 红点与 B 中 `position` 大致一致（同一 `base_link`）

---

## 7. 已知限制（当前阶段可接受）

| 项目 | 状态 | 说明 |
|------|------|------|
| `camera_link` / static TF | **未实测标定** | xyz 绝对值不可信，仅看**相对变化**与方向 |
| 轨迹 `/volleyball_trajectory` | 已关闭 | `trajectory.enable: false`，无抛物线正常 |
| 拦截预测 | 实时跟踪 | `time_to_event ≈ 0`，不是未来落点 |
| debug 框更新 | ~10 Hz 检测 + KF 补帧 | 不是每帧都出新框，属正常 |

标定后重做：[CALIBRATION.md §1–§2](CALIBRATION.md)。

---

## 8. 故障速查

| 现象 | 先查 |
|------|------|
| 无 `/ball_intercept` | KF 是否 OK；`ros2 topic hz /volleyball_pose` |
| 有 pose 无 intercept | 不应出现（同路径发布）；重启 `ball_detector_node` |
| intercept 有，stewart 无 | 是否起了 `executor.launch.py` |
| stewart 全是 emergency_stop=1 | `frame_id` 是否为 `base_link` |
| xyz 乱跳 / 深度离谱 | TF 未标定或深度噪声；看 debug 深度采样日志 |
| RViz 卡 | 正常，`debug.max_publish_hz` 已限频；可再降到 8 或 `debug.enable: false` |
| 无红点 | 工控机是否重编并重启；`ros2 topic list \| grep ball_marker` |

---

## 9. 记录模板（出问题发给队友）

```
日期：
机器：开发机 / 工控机 172.19.33.210
pipeline.conf：USE_REALSENSE= YOLO_INPUT_SIZE= YOLO_DEVICE=

ros2 topic hz /debug_image:
ros2 topic hz /volleyball_pose:
ros2 topic hz /ball_intercept:
ros2 topic hz /vision/stewart_target:

ball_intercept 样例（--once）：
stewart_target 样例（--once）：

现象：
/debug_image 截图：有/无
终端 log 最后 20 行：（粘贴）
```

---

## 10. 下一步（标定前 / 标定后）

**标定前（现在）**
- [ ] 本清单 §1–§4 全部打勾
- [ ] 与 UC 侧确认能收到 `/vision/stewart_target`

**标定后**
- [ ] 实测 static TF → [CALIBRATION.md §1](CALIBRATION.md)
- [ ] 实测 Stewart 球心 + R → [CALIBRATION.md §2](CALIBRATION.md)
- [ ] 再跑一遍 §3、§6，核对绝对距离与机构可达范围
