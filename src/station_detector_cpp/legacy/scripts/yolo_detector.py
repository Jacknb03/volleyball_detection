#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO排球检测模块
支持YOLOv5和YOLOv8，检测多色排球（蓝/黄/白）
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
import torch


class YOLODetector:
    """
    YOLO检测器类
    支持YOLOv5和YOLOv8推理
    """
    
    def __init__(self, model_path: Optional[str] = None, 
                 model_type: str = 'yolov8',
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = 'auto'):
        """
        初始化YOLO检测器
        
        Args:
            model_path: 模型路径，如果为None则使用预训练模型
            model_type: 模型类型 'yolov5' 或 'yolov8'
            conf_threshold: 置信度阈值 (0-1)
            iou_threshold: NMS的IoU阈值 (0-1)
            device: 设备 'cpu', 'cuda', 或 'auto'
        """
        self.model_type = model_type.lower()
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.device = self._setup_device(device)
        
        # 加载模型
        self._load_model(model_path)
        
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                return 'cpu'
        return device
    
    def _load_model(self, model_path: Optional[str]):
        """加载YOLO模型"""
        try:
            if self.model_type == 'yolov8':
                from ultralytics import YOLO
                if model_path:
                    self.model = YOLO(model_path)
                else:
                    # 使用预训练的YOLOv8n（nano版本，速度快）
                    self.model = YOLO('yolov8n.pt')
                self.model.to(self.device)
                print(f"[YOLO] 加载YOLOv8模型成功，设备: {self.device}")
                
            elif self.model_type == 'yolov5':
                import torch.hub
                if model_path:
                    self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
                else:
                    # 使用预训练的YOLOv5n
                    self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                self.model.conf = self.conf_threshold
                self.model.iou = self.iou_threshold
                self.model.to(self.device)
                print(f"[YOLO] 加载YOLOv5模型成功，设备: {self.device}")
            else:
                raise ValueError(f"不支持的模型类型: {self.model_type}")
                
        except Exception as e:
            print(f"[YOLO] 模型加载失败: {e}")
            print("[YOLO] 提示: 请安装ultralytics (pip install ultralytics)")
            raise
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        检测图像中的排球
        
        Args:
            image: BGR格式的OpenCV图像
            
        Returns:
            检测结果列表，每个元素包含:
            {
                'bbox': [x1, y1, x2, y2],  # 边界框
                'center': (cx, cy),         # 中心点
                'confidence': float,        # 置信度
                'class_id': int,            # 类别ID
                'class_name': str           # 类别名称
            }
        """
        if self.model is None:
            return []
        
        results = []
        
        try:
            if self.model_type == 'yolov8':
                # YOLOv8推理
                outputs = self.model(image, conf=self.conf_threshold, 
                                    iou=self.iou_threshold, verbose=False)
                
                if len(outputs) > 0 and outputs[0].boxes is not None:
                    boxes = outputs[0].boxes
                    for i in range(len(boxes)):
                        # 获取边界框坐标 (xyxy格式)
                        box = boxes.xyxy[i].cpu().numpy()
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        cls_name = self.model.names[cls_id]
                        
                        # 计算中心点
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        
                        results.append({
                            'bbox': box.tolist(),
                            'center': (cx, cy),
                            'confidence': conf,
                            'class_id': cls_id,
                            'class_name': cls_name
                        })
                        
            elif self.model_type == 'yolov5':
                # YOLOv5推理
                outputs = self.model(image)
                detections = outputs.pandas().xyxy[0]
                
                for _, det in detections.iterrows():
                    if det['confidence'] >= self.conf_threshold:
                        results.append({
                            'bbox': [det['xmin'], det['ymin'], det['xmax'], det['ymax']],
                            'center': ((det['xmin'] + det['xmax']) / 2.0,
                                      (det['ymin'] + det['ymax']) / 2.0),
                            'confidence': det['confidence'],
                            'class_id': det['class'],
                            'class_name': det['name']
                        })
                        
        except Exception as e:
            print(f"[YOLO] 检测过程出错: {e}")
            return []
        
        return results
    
    def filter_volleyball(self, detections: List[dict], 
                         class_names: Optional[List[str]] = None) -> List[dict]:
        """
        过滤出排球检测结果
        
        Args:
            detections: 检测结果列表
            class_names: 排球可能的类别名称列表，如['sports ball', 'ball']
                        如果为None，则使用默认值
        
        Returns:
            过滤后的排球检测结果
        """
        if class_names is None:
            # 默认排球类别名称（根据COCO数据集）
            class_names = ['sports ball', 'ball', 'volleyball']
        
        volleyball_detections = []
        for det in detections:
            cls_name = det['class_name'].lower()
            # 检查是否匹配排球类别
            if any(name.lower() in cls_name for name in class_names):
                volleyball_detections.append(det)
        
        return volleyball_detections
    
    def select_best_detection(self, detections: List[dict], 
                             method: str = 'confidence') -> Optional[dict]:
        """
        从多个检测结果中选择最佳的一个
        
        Args:
            detections: 检测结果列表
            method: 选择方法 'confidence'（置信度最高）或 'center'（最接近图像中心）
        
        Returns:
            最佳检测结果，如果没有则返回None
        """
        if not detections:
            return None
        
        if method == 'confidence':
            # 选择置信度最高的
            return max(detections, key=lambda x: x['confidence'])
        
        elif method == 'center':
            # 选择最接近图像中心的（需要图像尺寸，这里简化处理）
            # 实际使用时可以传入图像尺寸
            return max(detections, key=lambda x: x['confidence'])
        
        return detections[0]
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            image: 输入图像
            detections: 检测结果列表
        
        Returns:
            绘制了检测框的图像
        """
        result_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']
            center = det['center']
            conf = det['confidence']
            cls_name = det['class_name']
            
            # 绘制边界框
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制中心点
            cx, cy = map(int, center)
            cv2.circle(result_image, (cx, cy), 5, (0, 0, 255), -1)
            
            # 绘制标签
            label = f"{cls_name}: {conf:.2f}"
            cv2.putText(result_image, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_image
    
    def update_thresholds(self, conf_threshold: float = None, 
                         iou_threshold: float = None):
        """更新检测阈值"""
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
            if self.model_type == 'yolov5' and self.model is not None:
                self.model.conf = conf_threshold
        
        if iou_threshold is not None:
            self.iou_threshold = iou_threshold
            if self.model_type == 'yolov5' and self.model is not None:
                self.model.iou = iou_threshold

