#!/bin/bash

# ==========================================
# 配置区
# ==========================================
WS_PATH="$HOME/volleyball_detection"
VIDEO_PATH="$WS_PATH/src/station_detector_cpp/videos/1.mp4"
MODEL_PATH="$WS_PATH/src/station_detector_cpp/model/best.onnx"
YAML_PATH="$WS_PATH/src/station_detector_cpp/config/ball_detector_params.yaml"

source /opt/ros/humble/setup.bash
cd $WS_PATH
source install/setup.bash

echo "正在启动系统..."

# 1. 静态 TF
gnome-terminal --tab --title="TF_Static" -- bash -c "ros2 run tf2_ros static_transform_publisher --x 0 --y 0 --z 1 --yaw 0 --pitch 0 --roll 0 --frame-id odom --child-frame-id camera_optical_frame; exec bash"

# 2. 视频发布 (Python 直接跑)
gnome-terminal --tab --title="Video_Pub" -- bash -c "python3 $WS_PATH/src/station_detector/scripts/video_publisher.py --ros-args -p video_path:=$VIDEO_PATH -p loop:=true; exec bash"

# 3. 大脑节点 (修正了参数传递方式)
# 注意：我们这里不传空数组了，直接传一个['ball']，或者依靠你 YAML 里的设置
gnome-terminal --tab --title="Brain_Node" -- bash -c "source $WS_PATH/install/setup.bash; \
ros2 run station_detector_cpp ball_detector_node --ros-args \
--params-file $YAML_PATH \
-p yolo.model_path:=$MODEL_PATH \
-p world_frame_id:=odom \
-p camera_frame_id:=camera_optical_frame; \
exec bash"

# 4. RViz2
gnome-terminal --tab --title="RViz2" -- bash -c "rviz2; exec bash"