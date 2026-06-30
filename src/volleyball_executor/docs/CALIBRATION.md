# 固定相机 + 低轨道标定

## 1. 相机 → base_link（TF）

RealSense 刚性固定在底盘/立杆上后，用尺或 CAD 测 `camera_link`（或 `camera_color_optical_frame`）相对 `base_link` 的位姿，填入 launch：

```bash
ros2 launch station_detector_cpp yolo.launch.py pipeline_mode:=realsense use_static_camera_tf:=true
```

修改 `yolo.launch.py` 中 static TF 的 `--x --y --z --yaw --pitch --roll`。

验证：扔球时 `/ball_intercept` 的 `z` 应随球高度合理变化，`frame_id=base_link`。

## 2. 执行机构假想中心 + 球半径 R

在 `config/executor_params.yaml`：

```yaml
workspace_center_x: 0.0    # 实测 x1
workspace_center_y: 0.0    # 实测 y1
workspace_center_z: 0.27   # 实测 z1（待机/击球参考中心）
workspace_radius: 0.12       # R：平台在低轨道球面上的工作半径
sphere_mode: "on_shell"
```

含义：Stewart 收到的 `(x,y,z)` 是 **球方向** 在球心 `(x1,y1,z1)`、半径 `R` 的球面上的点，而不是直接把球坐标原样发过去（避免平台伸太远）。

## 3. 联调顺序

1. 仅视觉：`ros2 topic echo /ball_intercept`  
2. 开桥接：`ros2 topic echo /vision/stewart_target`  
3. CH7 视觉模式，观察 xyz 是否在球面附近、pitch/yaw 是否跟球  
4. 微调 `R` 与 center，避免 IK 超限
