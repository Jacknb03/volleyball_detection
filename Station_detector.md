# Engineer_Vision

Fusion_station_detector
1,相机标定加载
  去畸变

2,图像预处理：
筛选红色&蓝色
形态学处理：自适应卷积核-->出二值图combined
debug：True--保存 raw.png 和 mask.png 到 /tmp/station_debug/

3,轮廓检测和过滤：

  过滤条件:
    面积范围：min_contour_area 到 max_contour_area
    长宽比：1.5 到 6.0
    凸性：solidity < 0.7（L形特征）
    轮廓点数：至少20个点
//
contours: 所有找到的轮廓
filtered: 过滤后的L形轮廓
debug：True--保存 contours.png（所有轮廓可视化
//

4,L形角点检测:
  对于每个L形轮廓，找到一个角点坐标
  如果找不到合适的90度角，使用轮廓质心作为后备

5. 角点排序
   xy轴排序大致分为左右上下
   返回left[0], right[0], right[1], left[1]// 左上、右上、右下、左下
   debug:True--保存 corners.png 和 combined.png

6,生成世界坐标系
  返回兑换站灯条角点在真实世界中的3D坐标（单位：米）

