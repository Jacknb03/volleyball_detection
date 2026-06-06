#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLO模型视频测试脚本
用于测试自定义训练的YOLO模型在视频上的检测效果
"""
import sys
import os
from pathlib import Path

# OpenCV路径配置（Windows环境，根据实际情况调整）
sys.path.append(r'F:\opencv\opencv\build\python\cv2\python-3.11')
os.add_dll_directory(r'F:\opencv\opencv\build\x64\vc16\bin')

import cv2
import argparse

# 添加scripts目录到路径，以便导入yolo_detector
sys.path.insert(0, str(Path(__file__).parent))
from yolo_detector import YOLODetector


def test_yolo_on_video(model_path: str,
                      video_path: str,
                      model_type: str = 'yolov8',
                      conf_threshold: float = 0.5,
                      iou_threshold: float = 0.45,
                      device: str = 'auto',
                      output_path: str = None,
                      show_display: bool = True,
                      class_filter: list = None,
                      save_frames: bool = False):
    """
    在视频上测试YOLO模型
    
    Args:
        model_path: YOLO模型文件路径（.pt文件）
        video_path: 测试视频路径
        model_type: 模型类型 'yolov5' 或 'yolov8'
        conf_threshold: 置信度阈值
        iou_threshold: IoU阈值
        device: 计算设备 'cpu', 'cuda', 或 'auto'
        output_path: 输出视频路径（可选）
        show_display: 是否显示实时检测结果
        class_filter: 要检测的类别名称列表（如['volleyball', 'ball']），None表示检测所有类别
        save_frames: 是否保存检测到的帧
    """
    
    # 检查文件是否存在
    if not os.path.isfile(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return
    
    if not os.path.isfile(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    print(f"正在加载YOLO模型: {model_path}")
    print(f"模型类型: {model_type}")
    print(f"设备: {device}")
    
    # 初始化检测器
    try:
        detector = YOLODetector(
            model_path=model_path,
            model_type=model_type,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device
        )
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n视频信息:")
    print(f"  分辨率: {width}x{height}")
    print(f"  帧率: {fps:.2f} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames/fps:.2f} 秒")
    
    # 设置输出视频写入器
    out_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"\n将保存结果到: {output_path}")
    
    # 创建保存帧的目录
    frames_dir = None
    if save_frames:
        frames_dir = Path(video_path).parent / f"{Path(video_path).stem}_detections"
        frames_dir.mkdir(exist_ok=True)
        print(f"检测帧将保存到: {frames_dir}")
    
    # 统计信息
    frame_count = 0
    detection_count = 0
    total_detections = 0
    
    print("\n开始处理视频...")
    print("按 'q' 键退出，按 'p' 键暂停/继续，按 's' 键保存当前帧")
    
    paused = False
    
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n视频播放完毕")
                    break
                
                frame_count += 1
                
                # 运行检测
                detections = detector.detect(frame)
                
                # 如果指定了类别过滤，进行过滤
                if class_filter:
                    detections = detector.filter_volleyball(detections, class_filter)
                
                # 绘制检测结果
                result_frame = detector.draw_detections(frame, detections)
                
                # 添加统计信息
                info_text = [
                    f"Frame: {frame_count}/{total_frames}",
                    f"Detections: {len(detections)}",
                    f"FPS: {fps:.1f}"
                ]
                y_offset = 20
                for text in info_text:
                    cv2.putText(result_frame, text, (10, y_offset),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                
                # 如果有检测结果，显示详细信息
                if detections:
                    detection_count += 1
                    total_detections += len(detections)
                    best_det = detector.select_best_detection(detections)
                    if best_det:
                        det_text = f"Best: {best_det['class_name']} {best_det['confidence']:.2f}"
                        cv2.putText(result_frame, det_text, (10, y_offset),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 保存到输出视频
                if out_writer:
                    out_writer.write(result_frame)
                
                # 保存检测帧
                if save_frames and detections:
                    frame_path = frames_dir / f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(str(frame_path), result_frame)
            
            # 显示结果
            if show_display:
                cv2.imshow('YOLO Detection', result_frame if not paused else frame)
            
            # 处理按键
            if show_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户中断")
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'暂停' if paused else '继续'}")
                elif key == ord('s') and not paused:
                    save_path = f"frame_{frame_count:06d}.jpg"
                    cv2.imwrite(save_path, result_frame)
                    print(f"已保存当前帧到: {save_path}")
            else:
                # 不显示时，打印进度
                if frame_count % 30 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"进度: {progress:.1f}% ({frame_count}/{total_frames})")
    
    except KeyboardInterrupt:
        print("\n\n用户中断")
    finally:
        # 清理资源
        cap.release()
        if out_writer:
            out_writer.release()
        if show_display:
            cv2.destroyAllWindows()
        
        # 打印统计信息
        print("\n" + "="*50)
        print("检测统计:")
        print(f"  总帧数: {frame_count}")
        print(f"  有检测的帧数: {detection_count}")
        print(f"  总检测数: {total_detections}")
        if frame_count > 0:
            print(f"  检测率: {detection_count/frame_count*100:.2f}%")
            print(f"  平均每帧检测数: {total_detections/frame_count:.2f}")
        print("="*50)


def main():
    # 获取脚本所在目录，用于构建默认路径
    script_dir = Path(__file__).parent
    station_detector_dir = script_dir.parent
    default_model_path = station_detector_dir / 'model' / 'best.pt'
    default_video_path = station_detector_dir / 'videos' / 'test.mp4'
    
    parser = argparse.ArgumentParser(
        description='在视频上测试YOLO模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法:
  # 使用默认模型和视频（快速测试）
  python test_yolo_video.py
  
  # 指定视频路径（使用默认模型）
  python test_yolo_video.py --video path/to/video.mp4
  
  # 指定模型路径（使用默认视频）
  python test_yolo_video.py --model path/to/model.pt
  
  # 指定模型类型和阈值
  python test_yolo_video.py --model-type yolov5 --conf 0.3
  
  # 保存结果视频
  python test_yolo_video.py --video video.mp4 --output result.mp4
  
  # 过滤特定类别
  python test_yolo_video.py --classes volleyball ball
  
  # 无显示模式（适合批量处理）
  python test_yolo_video.py --video video.mp4 --no-display

默认路径:
  模型: {default_model_path}
  视频: {default_video_path}
        """
    )
    
    parser.add_argument('--model', '-m', 
                       default=str(default_model_path),
                       help=f'YOLO模型文件路径 (.pt文件，默认: {default_model_path})')
    parser.add_argument('--video', '-v',
                       default=str(default_video_path),
                       help=f'测试视频文件路径 (默认: {default_video_path})')
    parser.add_argument('--model-type', '-t', default='yolov8',
                       choices=['yolov5', 'yolov8'],
                       help='模型类型 (默认: yolov8)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU阈值 (默认: 0.45)')
    parser.add_argument('--device', '-d', default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备 (默认: auto)')
    parser.add_argument('--output', '-o', default=None,
                       help='输出视频路径（可选）')
    parser.add_argument('--no-display', action='store_true',
                       help='不显示实时检测结果（适合批量处理）')
    parser.add_argument('--classes', '-c', nargs='+', default=None,
                       help='要检测的类别名称列表（如: volleyball ball）')
    parser.add_argument('--save-frames', action='store_true',
                       help='保存所有有检测结果的帧')
    
    args = parser.parse_args()
    
    # 验证默认路径是否存在
    model_path = Path(args.model)
    video_path = Path(args.video)
    
    if not model_path.exists():
        print(f"\n错误: 模型文件不存在: {model_path}")
        print(f"请使用 --model 参数指定正确的模型路径")
        print(f"或确保默认模型文件存在: {default_model_path}")
        sys.exit(1)
    
    if not video_path.exists():
        print(f"\n错误: 视频文件不存在: {video_path}")
        print(f"请使用 --video 参数指定正确的视频路径")
        print(f"或确保默认视频文件存在: {default_video_path}")
        sys.exit(1)
    
    print(f"使用模型: {model_path}")
    print(f"使用视频: {video_path}")
    print()
    
    test_yolo_on_video(
        model_path=args.model,
        video_path=args.video,
        model_type=args.model_type,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        output_path=args.output,
        show_display=not args.no_display,
        class_filter=args.classes,
        save_frames=args.save_frames
    )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
