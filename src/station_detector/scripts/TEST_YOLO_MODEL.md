# YOLO模型测试指南

本指南说明如何使用 `test_yolo_video.py` 脚本测试您自己训练的YOLO模型。

## 快速开始

### 基本用法

```bash
# 进入scripts目录
cd src/station_detector/scripts

# 基本测试（使用YOLOv8模型）
python test_yolo_video.py --model /path/to/your/model.pt --video /path/to/your/video.mp4

# 使用YOLOv5模型
python test_yolo_video.py --model /path/to/your/model.pt --video /path/to/your/video.mp4 --model-type yolov5
```

## 参数说明

### 必需参数

- `--model` 或 `-m`: YOLO模型文件路径（.pt文件）
- `--video` 或 `-v`: 测试视频文件路径

### 可选参数

- `--model-type` 或 `-t`: 模型类型，`yolov5` 或 `yolov8`（默认: `yolov8`）
- `--conf`: 置信度阈值，范围0-1（默认: 0.5）
  - 检测不到目标时，降低此值（如0.3）
  - 误检太多时，提高此值（如0.6）
- `--iou`: IoU阈值，用于NMS（默认: 0.45）
- `--device` 或 `-d`: 计算设备，`auto`、`cpu` 或 `cuda`（默认: `auto`）
- `--output` 或 `-o`: 输出视频路径，保存检测结果视频（可选）
- `--classes` 或 `-c`: 要检测的类别名称列表，用空格分隔
  - 例如: `--classes volleyball ball`
  - 如果不指定，会检测所有类别
- `--no-display`: 不显示实时检测窗口（适合批量处理）
- `--save-frames`: 保存所有有检测结果的帧到单独目录

## 使用示例

### 1. 基本测试（显示结果）

```bash
python test_yolo_video.py \
    --model models/volleyball_yolov8.pt \
    --video videos/training_video.mp4
```

### 2. 调整检测阈值

```bash
# 降低置信度阈值，检测更多目标
python test_yolo_video.py \
    --model models/volleyball_yolov8.pt \
    --video videos/training_video.mp4 \
    --conf 0.3

# 提高置信度阈值，减少误检
python test_yolo_video.py \
    --model models/volleyball_yolov8.pt \
    --video videos/training_video.mp4 \
    --conf 0.7
```

### 3. 保存结果视频

```bash
python test_yolo_video.py \
    --model models/volleyball_yolov8.pt \
    --video videos/training_video.mp4 \
    --output results/detection_result.mp4
```

### 4. 只检测特定类别

```bash
# 只检测"volleyball"和"ball"类别
python test_yolo_video.py \
    --model models/volleyball_yolov8.pt \
    --video videos/training_video.mp4 \
    --classes volleyball ball
```

### 5. 批量处理（无显示模式）

```bash
# 适合处理多个视频，不显示窗口
python test_yolo_video.py \
    --model models/volleyball_yolov8.pt \
    --video videos/training_video.mp4 \
    --output results/result.mp4 \
    --no-display
```

### 6. 保存检测帧

```bash
# 保存所有有检测结果的帧
python test_yolo_video.py \
    --model models/volleyball_yolov8.pt \
    --video videos/training_video.mp4 \
    --save-frames
```

## 交互控制

在显示窗口中，可以使用以下按键：

- `q`: 退出程序
- `p`: 暂停/继续播放
- `s`: 保存当前帧为图片

## 输出信息

脚本会显示：

1. **模型加载信息**: 模型路径、类型、设备
2. **视频信息**: 分辨率、帧率、总帧数、时长
3. **实时统计**: 
   - 当前帧数/总帧数
   - 当前帧的检测数量
   - 最佳检测的类别和置信度
4. **最终统计**:
   - 总帧数
   - 有检测的帧数
   - 总检测数
   - 检测率
   - 平均每帧检测数

## 常见问题

### 1. 模型加载失败

**错误**: `模型加载失败: ...`

**解决方案**:
- 检查模型文件路径是否正确
- 确认模型文件格式是 `.pt`
- 确认安装了相应的依赖：
  - YOLOv8: `pip install ultralytics`
  - YOLOv5: `pip install torch torchvision`

### 2. 检测不到目标

**可能原因**:
- 置信度阈值太高
- 模型训练不充分
- 视频中的目标与训练数据差异太大

**解决方案**:
- 降低 `--conf` 参数（如0.3或0.2）
- 检查模型是否在类似场景下训练
- 查看模型训练时的类别名称是否正确

### 3. 误检太多

**解决方案**:
- 提高 `--conf` 参数（如0.6或0.7）
- 使用 `--classes` 参数过滤特定类别
- 检查模型训练质量

### 4. 运行速度慢

**解决方案**:
- 如果有GPU，确保使用 `--device cuda`
- 检查是否正确安装了CUDA版本的PyTorch
- 降低视频分辨率进行测试

### 5. 类别名称不匹配

如果您的模型训练的类别名称与默认不同，使用 `--classes` 参数指定：

```bash
# 例如，如果您的模型类别是 "volleyball" 而不是 "sports ball"
python test_yolo_video.py \
    --model model.pt \
    --video video.mp4 \
    --classes volleyball
```

## 模型格式要求

### YOLOv8模型
- 文件格式: `.pt`
- 通常由 `ultralytics` 训练生成
- 可以直接加载

### YOLOv5模型
- 文件格式: `.pt`
- 通常由 `yolov5` 仓库训练生成
- 需要确保安装了 `torch` 和 `torchvision`

## 下一步

测试完成后，如果效果满意，可以：

1. **集成到ROS2系统**: 修改 `config/yolo_params.yaml` 中的 `model_path` 参数
2. **调整检测参数**: 根据测试结果调整置信度阈值等参数
3. **批量测试**: 使用 `--no-display` 模式处理多个视频

## 相关文件

- `yolo_detector.py`: YOLO检测器实现
- `yolo_volleyball_node.py`: ROS2节点
- `config/yolo_params.yaml`: ROS2参数配置
