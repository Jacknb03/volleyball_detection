#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卡尔曼滤波轨迹跟踪模块
用于平滑和预测排球运动轨迹
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class Detection:
    """检测结果数据类"""
    center: Tuple[float, float]  # (x, y) 图像坐标
    confidence: float
    timestamp: float


class KalmanTracker:
    """
    卡尔曼滤波器用于排球轨迹跟踪
    状态向量: [x, y, vx, vy, ax, ay] (图像坐标系)
    """
    
    def __init__(self, 
                 process_noise: float = 0.1,
                 measurement_noise: float = 5.0,
                 initial_uncertainty: float = 10.0,
                 max_missing_frames: int = 5):
        """
        初始化卡尔曼滤波器
        
        Args:
            process_noise: 过程噪声（运动模型不确定性）
            measurement_noise: 观测噪声（检测误差）
            initial_uncertainty: 初始状态不确定性
            max_missing_frames: 最大允许丢帧数，超过后重置滤波器
        """
        # 状态向量维度: [x, y, vx, vy, ax, ay] = 6维
        self.state_dim = 6
        self.measurement_dim = 2  # 只能观测到位置 (x, y)
        
        # 状态向量 [x, y, vx, vy, ax, ay]
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        
        # 状态协方差矩阵 P
        self.P = np.eye(self.state_dim) * initial_uncertainty
        
        # 过程噪声协方差 Q
        self.Q = np.eye(self.state_dim) * process_noise
        # 位置噪声较小，速度和加速度噪声较大
        self.Q[0:2, 0:2] *= 0.1  # 位置噪声
        self.Q[2:4, 2:4] *= 1.0  # 速度噪声
        self.Q[4:6, 4:6] *= 2.0  # 加速度噪声
        
        # 观测噪声协方差 R
        self.R = np.eye(self.measurement_dim) * measurement_noise
        
        # 观测矩阵 H (只能观测位置)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[0, 0] = 1.0  # 观测x
        self.H[1, 1] = 1.0  # 观测y
        
        # 时间步长（动态更新）
        self.dt = 0.033  # 默认30Hz，实际会根据帧间隔更新
        
        # 状态转移矩阵 F（会在predict时根据dt更新）
        self.F = np.eye(self.state_dim)
        
        # 跟踪状态
        self.is_initialized = False
        self.missing_frames = 0
        self.max_missing_frames = max_missing_frames
        self.last_update_time = None
        
    def initialize(self, measurement: Tuple[float, float], timestamp: float):
        """
        初始化滤波器
        
        Args:
            measurement: 初始位置 (x, y)
            timestamp: 时间戳
        """
        self.state[0] = measurement[0]  # x
        self.state[1] = measurement[1]  # y
        self.state[2:6] = 0.0  # 初始速度和加速度为0
        
        # 重置协方差
        self.P = np.eye(self.state_dim) * 10.0
        
        self.is_initialized = True
        self.missing_frames = 0
        self.last_update_time = timestamp
        
    def predict(self, timestamp: Optional[float] = None) -> Tuple[float, float]:
        """
        预测步骤
        
        Args:
            timestamp: 当前时间戳，用于计算dt
        
        Returns:
            预测的位置 (x, y)
        """
        if not self.is_initialized:
            return (0.0, 0.0)
        
        # 更新时间步长
        if timestamp is not None and self.last_update_time is not None:
            self.dt = max(0.001, timestamp - self.last_update_time)  # 最小1ms
        # 如果dt太大（>0.1s），说明可能丢帧严重，使用默认值
        if self.dt > 0.1:
            self.dt = 0.033
        
        # 更新状态转移矩阵 F
        # x = x + vx*dt + 0.5*ax*dt²
        # y = y + vy*dt + 0.5*ay*dt²
        # vx = vx + ax*dt
        # vy = vy + ay*dt
        # ax = ax (假设加速度变化缓慢)
        # ay = ay
        self.F = np.array([
            [1, 0, self.dt, 0, 0.5*self.dt*self.dt, 0],
            [0, 1, 0, self.dt, 0, 0.5*self.dt*self.dt],
            [0, 0, 1, 0, self.dt, 0],
            [0, 0, 0, 1, 0, self.dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # 预测状态
        self.state = self.F @ self.state
        
        # 预测协方差
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # 增加不确定性（因为预测会有误差）
        self.P += np.eye(self.state_dim) * 0.1
        
        return (float(self.state[0]), float(self.state[1]))
    
    def update(self, measurement: Tuple[float, float], timestamp: float):
        """
        更新步骤
        
        Args:
            measurement: 观测位置 (x, y)
            timestamp: 时间戳
        """
        if not self.is_initialized:
            self.initialize(measurement, timestamp)
            return
        
        # 计算时间步长
        if self.last_update_time is not None:
            self.dt = max(0.001, timestamp - self.last_update_time)
        if self.dt > 0.1:
            self.dt = 0.033
        
        # 先进行一次预测
        self.predict(timestamp)
        
        # 观测向量
        z = np.array([measurement[0], measurement[1]], dtype=np.float32)
        
        # 计算残差（新息）
        y = z - self.H @ self.state
        
        # 残差协方差
        S = self.H @ self.P @ self.H.T + self.R
        
        # 卡尔曼增益
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        self.state = self.state + K @ y
        
        # 更新协方差
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P
        
        # 重置丢帧计数
        self.missing_frames = 0
        self.last_update_time = timestamp
    
    def update_with_missing(self, measurement: Optional[Tuple[float, float]], 
                           timestamp: float) -> Tuple[float, float]:
        """
        更新或预测（处理丢帧情况）
        
        Args:
            measurement: 观测位置，如果为None则表示丢帧
            timestamp: 时间戳
        
        Returns:
            当前估计的位置 (x, y)
        """
        if measurement is None:
            # 丢帧，只进行预测
            self.missing_frames += 1
            
            if self.missing_frames > self.max_missing_frames:
                # 丢帧太多，重置滤波器
                self.is_initialized = False
                return (0.0, 0.0)
            
            # 只预测，不更新
            return self.predict(timestamp)
        else:
            # 有观测，更新
            self.update(measurement, timestamp)
            return (self.state[0], self.state[1])
    
    def get_state(self) -> dict:
        """
        获取完整状态信息
        
        Returns:
            状态字典，包含位置、速度、加速度
        """
        return {
            'position': (float(self.state[0]), float(self.state[1])),
            'velocity': (float(self.state[2]), float(self.state[3])),
            'acceleration': (float(self.state[4]), float(self.state[5])),
            'uncertainty': float(np.trace(self.P[:2, :2])),  # 位置不确定性
            'is_initialized': self.is_initialized,
            'missing_frames': self.missing_frames
        }
    
    def predict_future(self, time_horizon: float, num_points: int = 10) -> list:
        """
        预测未来轨迹
        
        Args:
            time_horizon: 预测时间范围（秒）
            num_points: 预测点数
        
        Returns:
            预测位置列表 [(x1, y1), (x2, y2), ...]
        """
        if not self.is_initialized:
            return []
        
        predictions = []
        dt_step = time_horizon / num_points
        
        # 保存当前状态
        state_backup = self.state.copy()
        P_backup = self.P.copy()
        
        for i in range(1, num_points + 1):
            t = i * dt_step
            # 使用运动学方程预测
            x = state_backup[0] + state_backup[2] * t + 0.5 * state_backup[4] * t * t
            y = state_backup[1] + state_backup[3] * t + 0.5 * state_backup[5] * t * t
            predictions.append((float(x), float(y)))
        
        return predictions
    
    def reset(self):
        """重置滤波器"""
        self.is_initialized = False
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 10.0
        self.missing_frames = 0
        self.last_update_time = None

