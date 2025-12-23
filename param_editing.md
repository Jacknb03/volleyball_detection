# 颜色检测参数调优指南

## 概述

现在station_detector支持通过配置文件调整颜色检测参数，无需重新编译代码即可优化检测效果。

## 配置文件

参数保存在 `src/station_detector/config/station_params.yaml` 中：

```yaml
color_detection:
  # 红色检测参数 (HSV)
  red_lower_h: 0      # 红色下界H
  red_lower_s: 30     # 红色下界S
  red_lower_v: 30     # 红色下界V
  red_upper_h: 25     # 红色上界H
  red_upper_s: 255    # 红色上界S
  red_upper_v: 255    # 红色上界V
  
  # 红色上段检测参数 (HSV)
  red_upper_lower_h: 155  # 红色上段下界H
  red_upper_lower_s: 30  # 红色上段下界S
  red_upper_lower_v: 50  # 红色上段下界V
  red_upper_upper_h: 180 # 红色上段上界H
  red_upper_upper_s: 255 # 红色上段上界S
  red_upper_upper_v: 255 # 红色上段上界V
  
  # 蓝色检测参数 (HSV)
  blue_lower_h: 85    # 蓝色下界H
  blue_lower_s: 30    # 蓝色下界S
  blue_lower_v: 30    # 蓝色下界V
  blue_upper_h: 144   # 蓝色上界H
  blue_upper_s: 255   # 蓝色上界S
  blue_upper_v: 255   # 蓝色上界V
  
  # 形态学处理参数
  morphology_kernel_size: 5  # 形态学核大小
  use_adaptive_kernel: true   # 是否使用自适应核大小


## HSV颜色空间说明

- **H (Hue)**: 色调，0-180
  - 红色: 0-25 和 155-180
  - 蓝色: 85-144
- **S (Saturation)**: 饱和度，0-255
  - 越高颜色越鲜艳
- **V (Value)**: 亮度，0-255
  - 越高颜色越亮

## 常见问题

### Q: 检测不到红色/蓝色
A: 降低S和V的下限值，扩大检测范围

### Q: 检测到太多噪声
A: 提高S和V的下限值，增加形态学核大小

### Q: 检测效果不稳定
A: 使用自适应核大小，根据图像分辨率自动调整

### Q: 参数调整后没有效果
A: 确保重新启动了检测器，参数在启动时加载

## 高级调优

### 环境光照影响
- **强光环境**：提高V下限，降低S下限
- **弱光环境**：降低V下限，提高S下限
- **色温变化**：微调H范围

### 相机参数影响
- **高分辨率**：增加形态学核大小
- **低分辨率**：减少形态学核大小
- **不同相机**：可能需要完全重新调优

### 目标物体特性
- **颜色鲜艳**：提高S下限
- **颜色暗淡**：降低S下限
- **反光严重**：提高V下限
