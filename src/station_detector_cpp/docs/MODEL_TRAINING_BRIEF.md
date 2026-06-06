# 排球检测模型 — 训练需求

> 数据：**你自己拍**；本文只规定 **怎么整理数据、怎么训、怎么交**。  
> 推理端：`ball_detector_node`（OpenCV DNN + ONNX），比赛机可能 **1260P 无独显** → 必须 **小模型 + 416 输入**。

---

## 1. 交付物

| 交付 | 说明 |
|------|------|
| **`best.onnx`** | **必须**，C++ 只认这个 |
| `best.pt` | 权重备份 |
| `dataset.yaml` | 数据集配置 |
| 训练 + 导出命令 | 完整可复制的一行命令 |
| 验证截图 | 远/近/快球各 1 张成功 + 2 张典型失败 |

ONNX 路径：`src/station_detector_cpp/model/best.onnx`（不提交 git）。

---

## 2. 模型硬性规格

| 项 | 要求 |
|----|------|
| 模型 | **YOLOv8n** |
| 类别 | **1 类**，`volleyball`，训练时 `single_cls=True` |
| 输入尺寸 | **416×416**（导出与训练一致；**不要 640**） |
| ONNX | `format=onnx`, `imgsz=416`, `opset=12`, `simplify=True` |

导出 416 后通知视觉同学改 `yolo_inference.cpp` 里输入 640→416（或等 yaml 参数化）。

---

## 3. 数据集怎么放（YOLO 标准目录）

```
volleyball_dataset/
├── images/
│   ├── train/          # 训练图
│   └── val/            # 验证图（不要用 train 里的重复帧）
└── labels/
    ├── train/          # 与 train 图片同名 .txt
    └── val/
```

**标注格式**（每图一个同名 `.txt`）：

```
0 x_center y_center width height
```

- 全是 **class 0**；坐标 **0～1 归一化**  
- 无球的图：**不要进 train**，或不给 txt（别乱标）

**`volleyball.yaml`：**

```yaml
path: /绝对路径/volleyball_dataset
train: images/train
val: images/val
nc: 1
names:
  0: volleyball
```

**数量建议：** train **≥800**，val **≥150**（你拍够就行，相邻帧少抽重复）。

---

## 4. 训练与导出（复制执行）

```bash
pip install ultralytics

yolo detect train \
  model=yolov8n.pt \
  data=/绝对路径/volleyball.yaml \
  imgsz=416 \
  epochs=150 \
  batch=16 \
  patience=30 \
  single_cls=True

yolo export \
  model=runs/detect/train/weights/best.pt \
  format=onnx \
  imgsz=416 \
  opset=12 \
  simplify=True
```

输出：`runs/detect/train/weights/best.onnx` → 拷到 `model/best.onnx`。

**可选增强**（数据少时可加）：

```bash
  hsv_h=0.015 hsv_s=0.7 hsv_v=0.4 \
  degrees=15 translate=0.1 scale=0.5 mosaic=1.0
```

---

## 5. 验收（训完自测 + 给我联调）

**离线：**

- val **mAP50 ≥ 0.85**（数据够的情况下）
- 肉眼看 val：**远球、模糊球** 别大面积漏检

**替换 ONNX 后在我仓库测：**

```bash
YOLO_DEVICE=cuda ./start_all.sh    # 或 cpu 模拟工控机
ros2 topic hz /volleyball_pose
```

| 项 | 标准 |
|----|------|
| 有球 | `/debug_image` 绿框跟球 |
| 无球 | 不长期误检锁死 |
| 4060+CUDA | pose **≥ 6 Hz** |
| 1260P+CPU（若可测） | 目标 **≥ 8 Hz**（配合 OpenVINO） |

---

## 6. 不要

- ❌ YOLOv8s/m/l  
- ❌ 多类（人、网、球混训）  
- ❌ 只导出 COCO 预训练、不训自定义数据  
- ❌ 训练 416、导出 640（尺寸必须一致）

---

## 7. 相关

- 无显卡部署：[DEPLOYMENT.md §六](DEPLOYMENT.md#六无显卡--cpu-推理与模型优化)
- 检测阈值：`conf_threshold` 默认 **0.18**（[readme.md](../readme.md)）
