# 安装指南 (Ubuntu)

## 前置要求

- Ubuntu 20.04 或 22.04
- ROS2 Foxy 或 Humble
- Python 3.8+

**注意**: 如果系统已通过源码编译安装了OpenCV（C++版本），请参考 `OPENCV_SETUP.md` 了解详细配置说明。

## 安装步骤

### 1. 安装系统依赖

```bash
sudo apt update
sudo apt install -y \
    python3-pip \
    python3-dev
```

**注意**: 如果系统已通过源码编译安装了OpenCV（C++版本），不需要再安装 `python3-opencv`。

### 2. 安装Python依赖

```bash
cd src/station_detector

# 安装所有Python依赖（包括opencv-python）
pip3 install -r requirements.txt
```

**重要说明**: 
- 即使系统已通过源码编译安装了C++版本的OpenCV，Python代码仍需要Python版本的OpenCV
- `opencv-python` 和源码编译的OpenCV可以共存，互不影响
- Python代码使用 `opencv-python`，C++代码使用源码编译的OpenCV库
- **pip安装的opencv-python会自动配置路径，不需要手动添加到bashrc**

### 3. 验证安装

```bash
# 检查Python OpenCV
python3 -c "import cv2; print('OpenCV Python版本:', cv2.__version__)"

# 检查PyTorch
python3 -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"

# 检查YOLO
python3 -c "from ultralytics import YOLO; print('YOLOv8可用')"
```

### 4. 编译ROS2工作空间

```bash
cd ~/volleyball_detection  # 或你的工作空间路径
colcon build --symlink-install
source install/setup.bash
```

## 常见问题

### Q: 系统已有源码编译的OpenCV，为什么还要装opencv-python？

**A**: C++的OpenCV和Python的OpenCV是独立的：
- C++代码使用源码编译的OpenCV（通过CMake链接，路径在bashrc中配置）
- Python代码使用 `opencv-python`（通过pip安装到Python site-packages）
- 两者可以共存，互不冲突

### Q: pip安装的opencv-python需要添加到bashrc吗？

**A**: **不需要！** pip安装的包会自动配置到Python的搜索路径中：
- pip会将包安装到 `~/.local/lib/python3.x/site-packages/` 或系统Python目录
- Python会自动搜索这些路径，无需手动配置
- 只有源码编译的OpenCV（C++版本）才需要在bashrc中配置环境变量

### Q: 如何确认使用的是哪个OpenCV？

**A**: 
```bash
# Python代码使用的OpenCV（应该是pip安装的版本）
python3 -c "import cv2; print('Python OpenCV路径:', cv2.__file__)"
python3 -c "import cv2; print('Python OpenCV版本:', cv2.__version__)"
# 应该显示类似: ~/.local/lib/python3.x/site-packages/cv2/...

# C++代码使用的OpenCV（源码编译的版本）
pkg-config --modversion opencv4
# 或者检查bashrc中配置的路径
echo $PKG_CONFIG_PATH
echo $LD_LIBRARY_PATH
```

### Q: 如果Python导入cv2时找不到模块？

**A**: 检查Python路径：
```bash
# 查看Python搜索路径
python3 -c "import sys; print('\n'.join(sys.path))"

# 如果opencv-python安装正确，应该能看到site-packages路径
# 如果看不到，可能需要：
pip3 install --user opencv-python  # 使用--user安装到用户目录
```

### Q: 源码编译的OpenCV和pip的opencv-python会冲突吗？

**A**: **不会冲突**，它们是独立的：
- 源码编译的OpenCV：用于C++编译（通过CMake的FindOpenCV）
- pip的opencv-python：用于Python运行时（通过import cv2）
- 两者使用不同的路径和机制，互不干扰

### Q: CUDA支持？

**A**: 如果安装了CUDA版本的PyTorch，YOLO会自动使用GPU加速：
```bash
# 检查CUDA是否可用
python3 -c "import torch; print(torch.cuda.is_available())"
```

## 卸载

如果需要卸载Python依赖：
```bash
pip3 uninstall opencv-python torch torchvision ultralytics numpy
```

注意：不会影响系统的C++ OpenCV库。

