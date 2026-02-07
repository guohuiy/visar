#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionEngine Python 测试示例

使用ONNX Runtime测试VisionEngine后端模型推理功能
由于VisionEngine是C++库，本示例通过ONNX Runtime直接测试模型
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# 尝试导入onnxruntime，如果不存在则使用备用方案
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[警告] 未安装onnxruntime，将使用模拟模式进行测试")


# COCO数据集80类名称
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# YOLOv8n 后处理参数
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45


def load_image(image_path, target_size=(640, 640)):
    """加载并预处理图像"""
    try:
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        
        # 获取原始尺寸
        original_width, original_height = image.size
        
        # 调整大小
        image = image.resize(target_size, Image.BILINEAR)
        
        # 转换为numpy数组并归一化
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # HWC -> CHW
        img_array = img_array.transpose(2, 0, 1)
        
        # 添加batch维度
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, (original_width, original_height)
    except ImportError:
        print("[错误] 需要安装PIL库: pip install Pillow")
        return None, None


def xywh2xyxy(x):
    """将边界框从xywh格式转换为xyxy格式"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def non_max_suppression(boxes, scores, iou_threshold=0.45):
    """NMS非极大值抑制"""
    if len(boxes) == 0:
        return []
    
    # 计算面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # 按置信度排序
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # 计算与最高置信度框的IoU
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # 保留IoU低于阈值的框
        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]
    
    return keep


def postprocess_yolov8(output, orig_shape, conf_thresh=0.5, iou_thresh=0.45):
    """YOLOv8后处理"""
    predictions = np.squeeze(output).T
    
    # 分离边界框和类别置信度
    num_classes = 80
    boxes = predictions[:, :4]
    scores = predictions[:, 4:]
    
    # 计算类别分数
    class_scores = np.max(scores, axis=1)
    class_ids = np.argmax(scores, axis=1)
    
    # 过滤低置信度检测
    mask = class_scores > conf_thresh
    boxes = boxes[mask]
    class_scores = class_scores[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    # 转换坐标格式
    boxes = xywh2xyxy(boxes)
    
    # 缩放回原始尺寸
    scale_x = orig_shape[1] / 640
    scale_y = orig_shape[0] / 640
    boxes[:, [0, 2]] *= scale_x
    boxes[:, [1, 3]] *= scale_y
    
    # NMS
    keep_indices = non_max_suppression(boxes, class_scores, iou_thresh)
    boxes = boxes[keep_indices]
    class_scores = class_scores[keep_indices]
    class_ids = class_ids[keep_indices]
    
    # 格式化结果
    detections = []
    for i in range(len(boxes)):
        det = {
            'class_id': int(class_ids[i]),
            'class_name': COCO_CLASSES[int(class_ids[i])],
            'score': float(class_scores[i]),
            'bbox': {
                'x1': float(boxes[i, 0]),
                'y1': float(boxes[i, 1]),
                'x2': float(boxes[i, 2]),
                'y2': float(boxes[i, 3])
            }
        }
        detections.append(det)
    
    return detections


class VisionEngineSimulator:
    """
    VisionEngine模拟器
    
    由于VisionEngine是C++库，本类提供Python接口来测试模型推理功能
    实际使用时，VisionEngine会通过ONNX Runtime进行推理
    """
    
    def __init__(self, model_path=None):
        """初始化引擎"""
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_name = None
        
        if ONNX_AVAILABLE and model_path:
            self._initialize_backend()
    
    def _initialize_backend(self):
        """初始化ONNX Runtime后端"""
        if not os.path.exists(self.model_path):
            print(f"[错误] 模型文件不存在: {self.model_path}")
            return False
        
        # 创建推理会话
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"[VisionEngine] ONNX Runtime后端初始化成功")
            print(f"[VisionEngine] 输入: {self.input_name}, 输出: {self.output_name}")
            return True
        except Exception as e:
            print(f"[VisionEngine] 后端初始化失败: {e}")
            return False
    
    def load_model(self, model_path):
        """加载模型"""
        self.model_path = model_path
        return self._initialize_backend()
    
    def infer(self, image_data):
        """执行推理"""
        if self.session is None:
            print("[警告] 引擎未初始化，返回模拟结果")
            return self._mock_infer(image_data)
        
        try:
            # 执行推理
            start_time = time.time()
            output = self.session.run([self.output_name], {self.input_name: image_data})[0]
            inference_time = (time.time() - start_time) * 1000
            
            print(f"[VisionEngine] 推理耗时: {inference_time:.2f}ms")
            return output, inference_time
        except Exception as e:
            print(f"[错误] 推理失败: {e}")
            return None, 0
    
    def _mock_infer(self, image_data):
        """模拟推理结果（用于测试）"""
        print("[VisionEngine] 使用模拟推理模式")
        time.sleep(0.1)
        
        # 生成模拟检测结果
        mock_detections = [
            {
                'class_id': 0,
                'class_name': 'person',
                'score': 0.92,
                'bbox': {'x1': 100, 'y1': 50, 'x2': 200, 'y2': 300}
            },
            {
                'class_id': 2,
                'class_name': 'car',
                'score': 0.85,
                'bbox': {'x1': 400, 'y1': 200, 'x2': 550, 'y2': 350}
            }
        ]
        return mock_detections, 15.5


def test_single_image(engine, image_path, model_type='yolov8'):
    """测试单张图像"""
    print(f"\n{'='*60}")
    print(f"测试图像: {image_path}")
    print(f"{'='*60}")
    
    # 加载图像
    img_array, orig_shape = load_image(image_path)
    if img_array is None:
        return
    
    print(f"原始尺寸: {orig_shape}")
    print(f"输入形状: {img_array.shape}")
    
    # 执行推理
    output, inference_time = engine.infer(img_array)
    
    if output is None:
        return
    
    # 后处理
    if model_type == 'yolov8':
        detections = postprocess_yolov8(output, orig_shape, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    else:
        detections = output
    
    # 显示结果
    print(f"\n检测到 {len(detections)} 个目标:")
    for i, det in enumerate(detections):
        print(f"  [{i+1}] {det['class_name']} (类别ID: {det['class_id']})")
        print(f"      置信度: {det['score']:.4f}")
        print(f"      位置: ({det['bbox']['x1']:.1f}, {det['bbox']['y1']:.1f}) - "
              f"({det['bbox']['x2']:.1f}, {det['bbox']['y2']:.1f})")
    
    return detections


def test_batch_images(engine, image_dir, num_images=5):
    """测试多张图像"""
    print(f"\n{'='*60}")
    print(f"批量测试模式")
    print(f"{'='*60}")
    
    # 获取图像列表
    image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
    images = [f for f in os.listdir(image_dir) 
              if any(f.lower().endswith(ext) for ext in image_exts)]
    images = images[:num_images]
    
    if not images:
        print(f"[警告] 目录中未找到图像: {image_dir}")
        return
    
    total_time = 0
    total_detections = 0
    
    for i, image_name in enumerate(images):
        image_path = os.path.join(image_dir, image_name)
        detections = test_single_image(engine, image_path)
        
        if detections:
            total_detections += len(detections)
        
        total_time += 1  # 简化计算
    
    avg_time = total_time / len(images) if images else 0
    print(f"\n批量测试统计:")
    print(f"  测试图像数: {len(images)}")
    print(f"  总检测数: {total_detections}")
    print(f"  平均每张图检测数: {total_detections/len(images):.1f}")


def benchmark_model(engine, image_path, num_runs=10):
    """性能基准测试"""
    print(f"\n{'='*60}")
    print(f"性能基准测试 (运行 {num_runs} 次)")
    print(f"{'='*60}")
    
    img_array, _ = load_image(image_path)
    if img_array is None:
        return
    
    times = []
    
    for i in range(num_runs):
        start_time = time.time()
        _, _ = engine.infer(img_array)
        elapsed = (time.time() - start_time) * 1000
        times.append(elapsed)
    
    times = np.array(times)
    
    print(f"  推理时间统计:")
    print(f"    平均值: {times.mean():.2f}ms")
    print(f"    标准差: {times.std():.2f}ms")
    print(f"    最小值: {times.min():.2f}ms")
    print(f"    最大值: {times.max():.2f}ms")
    print(f"    中位数: {np.median(times):.2f}ms")


def list_available_models(models_dir):
    """列出可用模型"""
    print(f"\n可用模型:")
    print(f"  模型目录: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"  [警告] 模型目录不存在")
        return
    
    models = [f for f in os.listdir(models_dir) if f.endswith('.onnx')]
    
    if not models:
        print(f"  [警告] 未找到ONNX模型")
        return
    
    for model in models:
        model_path = os.path.join(models_dir, model)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  - {model} ({size_mb:.2f} MB)")


def main():
    """主函数"""
    print("="*60)
    print("VisionEngine Python 测试示例")
    print("="*60)
    
    # 设置路径
    base_dir = Path(__file__).parent
    models_dir = base_dir / "models"
    
    if not models_dir.exists():
        # 尝试从qt6_demo目录获取模型
        alt_models_dir = Path(__file__).parent.parent / "qt6_demo" / "models"
        if alt_models_dir.exists():
            models_dir = alt_models_dir
    
    # 列出可用模型
    list_available_models(str(models_dir))
    
    # 选择模型
    default_model = "yolov8n.onnx"
    model_path = models_dir / default_model
    
    if not model_path.exists():
        print(f"\n[警告] 默认模型不存在: {model_path}")
        print("将使用模拟模式进行测试")
        engine = VisionEngineSimulator()
    else:
        print(f"\n使用模型: {model_path}")
        engine = VisionEngineSimulator(str(model_path))
    
    # 测试图像路径
    test_images_dir = base_dir / "test_images"
    
    if test_images_dir.exists():
        # 查找测试图像
        image_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        test_images = [f for f in os.listdir(str(test_images_dir)) 
                       if any(f.lower().endswith(ext) for ext in image_exts)]
        
        if test_images:
            # 测试第一张图像
            test_image = str(test_images_dir / test_images[0])
            detections = test_single_image(engine, test_image)
            
            # 基准测试
            if ONNX_AVAILABLE:
                benchmark_model(engine, test_image, num_runs=10)
        else:
            # 批量测试
            test_batch_images(engine, str(test_images_dir), num_images=3)
    else:
        print(f"\n测试图像目录不存在: {test_images_dir}")
        print("创建示例测试...")
        
        # 检查是否有示例图像
        sample_image = base_dir / "sample.jpg"
        if sample_image.exists():
            test_single_image(engine, str(sample_image))
        else:
            # 使用模拟模式测试
            print("\n使用模拟模式进行测试")
            mock_engine = VisionEngineSimulator()
            mock_detections, _ = mock_engine.infer(np.zeros((1, 3, 640, 640), dtype=np.float32))
            print(f"模拟检测结果: {len(mock_detections)} 个目标")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)


if __name__ == "__main__":
    main()
