#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检测结果验证模块
用于过滤误检，提高检测稳定性
"""

import numpy as np
from typing import Optional, Tuple, List
from collections import deque


class DetectionValidator:
    """
    检测结果验证器
    通过历史信息过滤误检
    """
    
    def __init__(self,
                 max_jump_distance: float = 100.0,
                 min_consistent_detections: int = 2,
                 history_size: int = 5):
        """
        初始化验证器
        
        Args:
            max_jump_distance: 最大允许跳跃距离（像素），超过此距离认为是误检
            min_consistent_detections: 最小连续检测次数，低于此值认为是误检
            history_size: 历史记录大小
        """
        self.max_jump_distance = max_jump_distance
        self.min_consistent_detections = min_consistent_detections
        self.history_size = history_size
        
        # 历史检测记录
        self.detection_history = deque(maxlen=history_size)
        self.consistent_count = 0
        self.last_valid_center = None
    
    def validate(self, detection: dict, image_size: Tuple[int, int]) -> bool:
        """
        验证检测结果是否有效
        
        Args:
            detection: 检测结果字典，包含'center'和'confidence'
            image_size: 图像尺寸 (width, height)
        
        Returns:
            是否有效
        """
        if not detection:
            return False
        
        center = detection['center']
        confidence = detection.get('confidence', 0.0)
        
        # 1. 检查中心点是否在图像范围内
        if center[0] < 0 or center[0] >= image_size[0] or \
           center[1] < 0 or center[1] >= image_size[1]:
            return False
        
        # 2. 检查置信度
        if confidence < 0.3:  # 最低置信度阈值
            return False
        
        # 3. 检查是否与历史位置一致（防止误检）
        if self.last_valid_center is not None:
            distance = np.sqrt(
                (center[0] - self.last_valid_center[0])**2 + 
                (center[1] - self.last_valid_center[1])**2
            )
            
            if distance > self.max_jump_distance:
                # 跳跃距离过大，可能是误检
                self.consistent_count = 0
                return False
        
        # 4. 更新历史记录
        self.detection_history.append(center)
        self.consistent_count += 1
        self.last_valid_center = center
        
        # 5. 检查连续检测次数
        if self.consistent_count < self.min_consistent_detections:
            return False
        
        return True
    
    def reset(self):
        """重置验证器状态"""
        self.detection_history.clear()
        self.consistent_count = 0
        self.last_valid_center = None
    
    def get_average_position(self) -> Optional[Tuple[float, float]]:
        """
        获取历史平均位置（用于平滑）
        
        Returns:
            平均位置，如果没有历史则返回None
        """
        if len(self.detection_history) < 2:
            return self.last_valid_center
        
        positions = np.array(list(self.detection_history))
        avg_pos = np.mean(positions, axis=0)
        return (float(avg_pos[0]), float(avg_pos[1]))


class MultiDetectionTracker:
    """
    多目标跟踪器
    用于处理多个检测结果，选择最可能是排球的
    """
    
    def __init__(self, max_track_distance: float = 150.0):
        """
        初始化多目标跟踪器
        
        Args:
            max_track_distance: 最大跟踪距离（像素）
        """
        self.max_track_distance = max_track_distance
        self.tracks = {}  # {track_id: {'center': (x, y), 'age': int, 'hits': int}}
        self.next_track_id = 0
    
    def update(self, detections: List[dict]) -> Optional[dict]:
        """
        更新跟踪器并返回最佳检测结果
        
        Args:
            detections: 检测结果列表
        
        Returns:
            最佳检测结果
        """
        if not detections:
            # 没有检测，更新所有track的age
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                # 删除过期的track
                if self.tracks[track_id]['age'] > 5:
                    del self.tracks[track_id]
            return None
        
        # 匹配检测结果到现有tracks
        matched = set()
        for track_id, track in self.tracks.items():
            best_match = None
            best_distance = float('inf')
            
            for i, det in enumerate(detections):
                if i in matched:
                    continue
                
                distance = np.sqrt(
                    (det['center'][0] - track['center'][0])**2 +
                    (det['center'][1] - track['center'][1])**2
                )
                
                if distance < self.max_track_distance and distance < best_distance:
                    best_match = i
                    best_distance = distance
            
            if best_match is not None:
                # 更新track
                det = detections[best_match]
                self.tracks[track_id]['center'] = det['center']
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['hits'] += 1
                matched.add(best_match)
        
        # 为未匹配的检测创建新track
        for i, det in enumerate(detections):
            if i not in matched:
                self.tracks[self.next_track_id] = {
                    'center': det['center'],
                    'age': 0,
                    'hits': 1
                }
                self.next_track_id += 1
        
        # 删除过期的tracks
        for track_id in list(self.tracks.keys()):
            if self.tracks[track_id]['age'] > 5:
                del self.tracks[track_id]
        
        # 选择最佳track（hits最多且age最小）
        if not self.tracks:
            return None
        
        best_track_id = max(
            self.tracks.keys(),
            key=lambda tid: (self.tracks[tid]['hits'], -self.tracks[tid]['age'])
        )
        
        best_track = self.tracks[best_track_id]
        
        # 返回最佳检测结果（需要从原始detections中找到对应的）
        for det in detections:
            distance = np.sqrt(
                (det['center'][0] - best_track['center'][0])**2 +
                (det['center'][1] - best_track['center'][1])**2
            )
            if distance < self.max_track_distance:
                return det
        
        return None

