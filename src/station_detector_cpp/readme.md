本手册旨在帮助你通过修改 ball_detector_params.yaml 配置文件，优化排球拦截系统的准确性、稳定性和响应速度。

> **硬件部署 / CUDA / RealSense D455i**：见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)  
> **从零调试步骤**：见 [docs/DEBUGGING.md](docs/DEBUGGING.md)

核心逻辑链路回顾

YOLO (眼睛)：识别 2D 像素框。

Estimator (空间感)：利用排球直径和相机焦距，计算 3D 深度 (Z)。

TF Transform (地图)：将相机坐标转为地面世界坐标 (odom/world)。

Kalman Filter (大脑)：平滑位置抖动，并推算 3D 速度 (Vx, Vy, Vz)。

Predictor (预言)：基于当前状态，利用物理公式（重力+阻力）预测未来落点。

一、 核心参数详解
1. 深度与空间校准 (3D Calibration)

如果发现球的 3D 位置与实际不符（例如：明明球在 3 米远，系统显示 5 米）。

参数	建议值	说明
volleyball.diameter	0.21	最重要的杠杆。若 Z 显示偏大，调小直径；若 Z 显示偏小，调大直径。
volleyball.min_depth	0.3	过滤掉离镜头太近的干扰物。
world_frame_id	"odom"	必须与你启动静态 TF 时的 frame-id 一致。
static_transform (启动指令)	z=1.0	相机离地高度。若预测落点总穿模到地下，检查此高度是否真实。
2. 平滑与防抖 (Smoothing & Stability)

解决轨迹“乱跳”、“闪现”或“滑稽感”的关键。

参数	建议值	调参建议
detection.h_ema_alpha	0.2 ~ 0.4	柔性系数。越小越稳（轨迹丝滑），越大越灵敏（追球快）。建议从 0.2 开始。
kalman.measurement_noise	30.0 ~ 80.0	不信任度。数值越大，系统越不迷信 YOLO 的抖动，轨迹越平滑。
detection.max_physical_speed	20.0	安全门禁。若轨迹偶尔“瞬移”，减小此值（限制球的最大物理时速）。
3. 物理预测优化 (Physics Prediction)

解决预测落点（那颗红球）准不准的问题。

参数	建议值	调参建议
drag_coefficient	0.47 ~ 0.6	空气阻力。若预测落点总比实际落点远，调大此系数。
volleyball.mass_kg	0.27	标准排球重量。减轻重量会使阻力效果更明显。
trajectory.ground_z	0.0	预测停止的高度（地面高度）。通常设为 0。
二、 常见问题诊断 (Troubleshooting)
场景 A：轨迹“追着球跑”，反应太慢

原因：平滑过度（EMA 或 Kalman 延迟）。

对策：

增大 h_ema_alpha（如调至 0.5）。

减小 kalman.measurement_noise（如调至 20.0）。

场景 B：轨迹“闪现”、抛物线像尖刺一样乱指

原因：YOLO 框抖动导致虚假速度。

对策：

减小 h_ema_alpha（如调至 0.15）。

减小 max_physical_speed（如调至 15.0）。

增大 kalman.measurement_noise。

场景 C：抛物线始终垂直向下掉

原因：系统没拿到水平速度。

对策：

检查终端 Velocity: Vx, Vy 是否有数值。

检查 static_transform_publisher 的 pitch 角度。

确保 world_frame_id 不是 camera_optical_frame（两者必须不同）。

场景 D：球在远距离识别不到

原因：YOLO 门槛过高。

对策：

降低 yolo.conf_threshold（如 0.1）。

确认 yolo.volleyball_classes 为空 []（针对单类模型）。

三、 最佳调试工作流

准备视频：找一段排球从远处飞向近处且有明显落地过程的视频。

启动监控：运行 ./start_all.sh，并在 RViz 中打开 /debug_image 查看 FPS 和 Z 深度。

定点观察：

观察球在空中时，红色落点球是否能保持相对稳定。

观察球落地那一瞬间，预测落点与实际落点的空间重合度。

修改参数：修改 config/ball_detector_params.yaml 后，运行 ./stop_all.sh 再运行 ./start_all.sh（或在 ros2 run 终端按 Ctrl+C 后重新运行）。

反复迭代：先求“稳”（调大噪声，减小 alpha），再求“准”（微调直径和阻力）。

四、 进阶技巧

保存 RViz 配置：在 RViz 中设置好 Fixed Frame: odom、debug_image、MarkerArray 后，点击 File -> Save Config。下次启动会直接进入该视角。

关于 FPS：由于 CPU 推理限制，FPS 在 5-10 之间是正常的。不要因为卡顿而怀疑算法，要观察落点预判的稳定性。

祝你的机器人早日接到第一个球！ 🏐🤖
