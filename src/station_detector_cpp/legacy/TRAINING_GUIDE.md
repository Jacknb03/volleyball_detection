# YOLO排球检测模型训练指南

## 概述

本项目只包含YOLO模型的**推理代码**，不包含训练代码。如果需要训练自定义的排球检测模型，建议在**单独的项目空间**中进行。

## 当前项目的模型使用方式

### 模型配置

模型选择通过 `config/yolo_params.yaml` 配置：

```yaml
yolo:
  model_path: ""  # 自定义模型路径，为空则使用预训练模型
  model_type: "yolov8"  # 'yolov5' 或 'yolov8'
  conf_threshold: 0.5
  iou_threshold: 0.45
  device: "auto"  # 'auto', 'cpu', 'cuda'
```

### 模型加载逻辑

- 如果 `model_path` 为空：使用预训练模型（如 `yolov8n.pt`）
- 如果 `model_path` 有值：加载自定义训练的模型

## 训练项目建议

### 为什么推荐单独开项目？

1. **代码分离**：训练代码和推理代码职责不同，分离更清晰
2. **资源管理**：训练需要大量数据和计算资源，独立环境更合适
3. **版本控制**：可以更好地管理数据集、模型版本和实验记录
4. **不影响主项目**：训练过程中的调试和修改不会影响ROS2项目的稳定性

### 训练项目结构建议

```
volleyball_training/
├── data/
│   ├── images/          # 训练图像
│   │   ├── train/
│   │   └── val/
│   └── labels/          # YOLO格式标注
│       ├── train/
│       └── val/
├── models/              # 训练好的模型
├── configs/             # 训练配置文件
├── scripts/
│   ├── train.py         # 训练脚本
│   ├── convert_dataset.py  # 数据集转换脚本
│   └── evaluate.py      # 评估脚本
├── requirements.txt
└── README.md
```

## 训练集准备

### 1. 图像数据

需要收集包含排球的图像，建议包含：

- **不同颜色**：蓝色、黄色、白色排球
- **不同角度**：正面、侧面、俯视、仰视
- **不同距离**：近距离、中距离、远距离
- **不同光照**：明亮、昏暗、逆光、阴影
- **不同背景**：球场、天空、观众席等
- **不同状态**：静止、运动模糊、部分遮挡
- **不同尺寸**：排球在图像中的大小变化

**建议数量**：
- 训练集：至少500-1000张图像
- 验证集：至少100-200张图像
- 测试集：至少50-100张图像

### 2. 标注格式

使用YOLO格式标注（`.txt`文件）：

```
class_id center_x center_y width height
```

其中：
- `class_id`: 类别ID（排球通常为0）
- `center_x, center_y`: 边界框中心点坐标（归一化到0-1）
- `width, height`: 边界框宽度和高度（归一化到0-1）

**示例**：
```
0 0.5 0.5 0.2 0.2
```

### 3. 不需要的信息

**不需要在训练集中包含**：
- ❌ 距离信息：距离是通过相机标定和检测框大小计算的
- ❌ 速度信息：速度是通过卡尔曼滤波从位置序列计算的
- ❌ 3D坐标：3D位置是通过2D检测结果和相机参数估计的

这些信息都是在**检测后**通过算法计算的，不是训练数据的一部分。

## 训练流程

### 1. 使用YOLOv8训练（推荐）

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')  # 或 yolov8s.pt, yolov8m.pt 等

# 训练
results = model.train(
    data='data/dataset.yaml',  # 数据集配置文件
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda',  # 或 'cpu'
    project='runs/train',
    name='volleyball_detector'
)
```

### 2. 数据集配置文件（dataset.yaml）

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: volleyball
```

### 3. 导出训练好的模型

训练完成后，模型会保存在 `runs/train/volleyball_detector/weights/best.pt`

将模型复制到当前项目的模型目录，并在 `yolo_params.yaml` 中配置路径。

## 训练技巧

### 1. 数据增强

YOLO训练会自动应用数据增强，包括：
- 随机翻转
- 随机缩放
- 颜色抖动
- Mosaic增强
- MixUp增强

### 2. 超参数调整

- **学习率**：从0.01开始，根据loss调整
- **批次大小**：根据GPU内存调整（8, 16, 32等）
- **图像尺寸**：640x640是标准尺寸，可以尝试416或832
- **训练轮数**：通常100-300轮，观察验证集loss不再下降时停止

### 3. 模型选择

- **YOLOv8n**：最快，适合实时应用
- **YOLOv8s**：平衡速度和精度
- **YOLOv8m**：更高精度，速度较慢
- **YOLOv8l/x**：最高精度，速度最慢

## 模型评估

训练完成后，使用验证集评估模型：

```python
from ultralytics import YOLO

model = YOLO('runs/train/volleyball_detector/weights/best.pt')
results = model.val(data='data/dataset.yaml')
```

关注指标：
- **mAP50**：IoU=0.5时的平均精度
- **mAP50-95**：IoU=0.5-0.95的平均精度
- **Precision**：精确率
- **Recall**：召回率

## 模型部署

训练完成后，将模型部署到当前项目：

1. 复制模型文件到项目目录（可选）
2. 在 `config/yolo_params.yaml` 中配置模型路径：
   ```yaml
   yolo:
     model_path: "/path/to/your/best.pt"
     model_type: "yolov8"
   ```
3. 重新启动ROS2节点测试

## 常见问题

### Q: 可以使用COCO预训练模型吗？
A: 可以。COCO数据集包含"sports ball"类别，可以直接使用，但针对排球场景的精度可能不够高。

### Q: 需要多少数据才能训练？
A: 至少需要500-1000张标注图像才能训练出可用的模型。更多数据通常意味着更好的性能。

### Q: 训练需要GPU吗？
A: 强烈建议使用GPU。CPU训练会非常慢（可能需要数天），GPU训练通常几小时就能完成。

### Q: 如何标注数据？
A: 推荐使用标注工具：
- **LabelImg**：简单易用的GUI工具
- **CVAT**：功能强大的在线标注平台
- **Roboflow**：在线标注和数据集管理平台

## 参考资源

- [Ultralytics YOLOv8文档](https://docs.ultralytics.com/)
- [YOLOv8训练教程](https://docs.ultralytics.com/modes/train/)
- [YOLO标注格式说明](https://docs.ultralytics.com/datasets/)

