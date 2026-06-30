# volleyball_executor

固定相机 + **base_link** 球位 → Universal Controllers Stewart。

## 数据流

```text
RealSense（固定于 base_link）
  → static TF 标定 base_link→camera
  → /ball_intercept（frame=base_link，球 xyz）
  → intercept_bridge
       · xyz：球位投影到低轨道球面（中心=执行机构假想点）
       · pitch/yaw：中心→球
  → /vision/stewart_target
  → volleyball_hub → /stewart_command
```

## 低轨道球约束

以执行机构中心 `(x1,y1,z1)` 为球心、半径 `R`：

```text
(x-x1)² + (y-y1)² + (z-z1)² = R²   （sphere_mode: on_shell，默认）
(x-x1)² + (y-y1)² + (z-z1)² ≤ R²   （sphere_mode: inside）
```

球在 base_link 的测量点沿 **center→ball** 方向投影到球面，再发给 Stewart。

参数见 `config/executor_params.yaml`：`workspace_center_*`、`workspace_radius`。

## 相机标定（固定安装后做一次）

1. 量相机相对 `base_link` 的 x,y,z,roll,pitch,yaw  
2. 写入 launch 的 `static_transform_publisher`（`yolo.launch.py`，`use_static_camera_tf:=true`）  
3. 验证：

```bash
ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame
ros2 topic echo /ball_intercept   # frame_id 应为 base_link
```

## 启动

```bash
./start_all.sh
ros2 launch volleyball_executor executor.launch.py
```

遥控 CH7 中位 = 视觉云台；队友 hub 需拷贝 **pitch**。

## 对接

- 发 **`/vision/stewart_target`**（`volleyball_msgs/StewartControl`）
- `universal_controllers_v2-main/` 已 gitignore，仅作对照
