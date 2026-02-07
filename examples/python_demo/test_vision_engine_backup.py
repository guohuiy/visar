#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VisionEngine Python 测试示例

使用ONNX Runtime测试VisionEngine后端模型推理功能
支持多种NMS算法、YOLO解析器、多尺度特征融合、Fast-NMS、TTA增强等
"""

import os
import sys
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

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
        original_width, original_height = image.size
        image = image.resize(target_size, Image.BILINEAR)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = img_array.transpose(2, 0, 1)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array, (original_width, original_height)
    except ImportError:
        print("[错误] 需要安装PIL库: pip install Pillow")
        return None, None


def xywh2xyxy(x):
    """将边界框从xywh格式转换为xyxy格式"""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def calculate_iou(box_a, box_b):
    """计算两个边界框的IoU"""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def calculate_giou(box_a, box_b):
    """计算GIoU"""
    iou = calculate_iou(box_a, box_b)
    x1 = min(box_a[0], box_b[0])
    y1 = min(box_a[1], box_b[1])
    x2 = max(box_a[2], box_b[2])
    y2 = max(box_a[3], box_b[3])
    c_area = (x2 - x1) * (y2 - y1)
    if c_area == 0:
        return iou
    giou = iou - (c_area - iou * (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) - 
                   (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) + iou * (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])) / c_area
    return giou


def calculate_diou(box_a, box_b):
    """计算DIoU"""
    iou = calculate_iou(box_a, box_b)
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    c_x = (box_a[0] + box_a[2]) / 2
    c_y = (box_a[1] + box_a[3]) / 2
    d_x = (box_b[0] + box_b[2]) / 2
    d_y = (box_b[1] + box_b[3]) / 2
    c_dist_sq = (c_x - d_x) ** 2 + (c_y - d_y) ** 2
    c_w = max(0, x2 - x1)
    c_h = max(0, y2 - y1)
    if c_w == 0 or c_h == 0:
        c_w = box_a[2] - box_a[0]
        c_h = box_a[3] - box_a[1]
        if c_w == 0:
            c_w = 1
        if c_h == 0:
            c_h = 1
    c_diag_sq = c_w ** 2 + c_h ** 2
    if c_diag_sq == 0:
        return iou
    diou = iou - c_dist_sq / c_diag_sq
    return diou


def calculate_ciou(box_a, box_b):
    """计算CIoU"""
    iou = calculate_iou(box_a, box_b)
    center_x1 = (box_a[0] + box_a[2]) / 2
    center_y1 = (box_a[1] + box_a[3]) / 2
    center_x2 = (box_b[0] + box_b[2]) / 2
    center_y2 = (box_b[1] + box_b[3]) / 2
    c_dist_sq = (center_x1 - center_x2) ** 2 + (center_y1 - center_y2) ** 2
    w1 = box_a[2] - box_a[0]
    h1 = box_a[3] - box_a[1]
    w2 = box_b[2] - box_b[0]
    h2 = box_b[3] - box_b[1]
    c_w = max(box_a[2], box_b[2]) - min(box_a[0], box_b[0])
    c_h = max(box_a[3], box_b[3]) - min(box_a[1], box_b[1])
    if c_w == 0 or c_h == 0:
        c_w = 1
        c_h = 1
    v = (4 / (np.pi ** 2)) * ((np.arctan(w2 / h2) - np.arctan(w1 / h1)) ** 2)
    alpha = v / (1 - iou + v + 1e-7)
    c_diag_sq = c_w ** 2 + c_h ** 2
    if c_diag_sq == 0:
        return iou
    ciou = iou - (c_dist_sq / c_diag_sq + v * alpha)
    return ciou


def nms(boxes, scores, iou_threshold=0.45):
    """标准NMS非极大值抑制"""
    if len(boxes) == 0:
        return []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        indices = np.where(iou <= iou_threshold)[0]
        order = order[indices + 1]
    return keep


def soft_nms(boxes, scores, iou_threshold=0.45, sigma=0.5, score_threshold=0.001):
    """Soft-NMS (高斯衰减)"""
    if len(boxes) == 0:
        return []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]
        if scores[i] < score_threshold:
            break
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-7)
        decay = np.exp(-iou ** 2 / sigma)
        scores[order[1:]] *= decay
        mask = scores[order[1:]] > score_threshold
        order = order[np.concatenate([[True], mask])]
    valid_indices = np.where(scores > score_threshold)[0]
    return valid_indices.tolist()


def classwise_nms(detections, iou_threshold=0.45):
    """按类别执行NMS"""
    if len(detections) == 0:
        return []
    class_groups = {}
    for i, det in enumerate(detections):
        class_id = det['class_id']
        if class_id not in class_groups:
            class_groups[class_id] = []
        class_groups[class_id].append(i)
    keep = []
    for class_id, indices in class_groups.items():
        if len(indices) == 1:
            keep.append(indices[0])
            continue
        boxes = np.array([detections[i]['bbox'] for i in indices])
        scores = np.array([detections[i]['score'] for i in indices])
        keep_indices = nms(boxes, scores, iou_threshold)
        keep.extend([indices[i] for i in keep_indices])
    return keep


def filter_by_confidence(detections, threshold=0.5):
    """置信度过滤"""
    return [d for d in detections if d['score'] >= threshold]


def ciou_nms(detections, iou_threshold=0.45):
    """使用CIoU的NMS"""
    if len(detections) == 0:
        return []
    sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
    keep = []
    while sorted_dets:
        current = sorted_dets[0]
        keep.append(current)
        if len(sorted_dets) == 1:
            break
        remaining = []
        for det in sorted_dets[1:]:
            iou = calculate_ciou(
                [current['bbox']['x1'], current['bbox']['y1'], current['bbox']['x2'], current['bbox']['y2']],
                [det['bbox']['x1'], det['bbox']['y1'], det['bbox']['x2'], det['bbox']['y2']]
            )
            if iou < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining
    return keep


class FastNMS:
    """Fast-NMS 使用空间哈希索引加速"""
    def __init__(self, iou_threshold=0.45, max_detections=100, grid_size=64):
        self.iou_threshold = iou_threshold
        self.max_detections = max_detections
        self.grid_size = grid_size
    
    def process(self, detections):
        """执行Fast-NMS"""
        if len(detections) == 0:
            return []
        sorted_dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        grid = {}
        for i, det in enumerate(sorted_dets):
            center_x = (det['bbox']['x1'] + det['bbox']['x2']) / 2
            center_y = (det['bbox']['y1'] + det['bbox']['y2']) / 2
            grid_key = (int(center_x / self.grid_size), int(center_y / self.grid_size))
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append(i)
        keep = []
        suppressed = set()
        for i, det in enumerate(sorted_dets):
            if i in suppressed:
                continue
            keep.append(det)
            if len(keep) >= self.max_detections:
                break
            center_x = (det['bbox']['x1'] + det['bbox']['x2']) / 2
            center_y = (det['bbox']['y1'] + det['bbox']['y2']) / 2
            grid_key = (int(center_x / self.grid_size), int(center_y / self.grid_size))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_key = (grid_key[0] + dx, grid_key[1] + dy)
                    if neighbor_key in grid:
                        for j in grid[neighbor_key]:
                            if j > i and j not in suppressed:
                                other = sorted_dets[j]
                                iou = calculate_iou(
                                    [det['bbox']['x1'], det['bbox']['y1'], det['bbox']['x2'], det['bbox']['y2']],
                                    [other['bbox']['x1'], other['bbox']['y1'], other['bbox']['x2'], other['bbox']['y2']]
                                )
                                if iou >= self.iou_threshold:
                                    suppressed.add(j)
        return keep


class MultiScaleDetector:
    """多尺度特征融合检测器"""
    def __init__(self, confidence_threshold=0.5, nms_threshold=0.45, strides=None):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.strides = strides if strides else [8, 16, 32]
    
    def set_config(self, config):
        """配置检测器"""
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.strides = config.get('strides', [8, 16, 32])
    
    def detect(self, outputs, shapes):
        """执行多尺度检测"""
        all_detections = []
        for i, (output, shape) in enumerate(zip(outputs, shapes)):
            stride = self.strides[i] if i < len(self.strides) else 32
            detections = self._decode_output(output, shape, stride)
            all_detections.extend(detections)
        return all_detections
    
    def _decode_output(self, output, shape, stride):
        """解码单尺度输出"""
        predictions = np.squeeze(output).T
        num_classes = 80
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        mask = class_scores > self.confidence_threshold
        boxes = boxes[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]
        if len(boxes) == 0:
            return []
        boxes = xywh2xyxy(boxes)
        scale_x = shape[1] / 640
        scale_y = shape[0] / 640
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        detections = []
        for j in range(len(boxes)):
            det = {
                'class_id': int(class_ids[j]),
                'class_name': COCO_CLASSES[int(class_ids[j])],
                'score': float(class_scores[j]),
                'bbox': {
                    'x1': float(boxes[j, 0]),
                    'y1': float(boxes[j, 1]),
                    'x2': float(boxes[j, 2]),
                    'y2': float(boxes[j, 3])
                }
            }
            detections.append(det)
        return detections


class TTAEngine:
    """测试时增强 (TTA) 引擎"""
    def __init__(self):
        self.horizontal_flip = True
        self.multi_scale = True
        self.scales = [0.83, 1.0, 1.17]
    
    def set_config(self, config):
        """配置TTA"""
        self.horizontal_flip = config.get('horizontal_flip', True)
        self.multi_scale = config.get('multi_scale', True)
        self.scales = config.get('scales', [0.83, 1.0, 1.17])
    
    def process(self, image, width, height, infer_func):
        """执行TTA推理"""
        results = []
        
        # 原始图像推理
        if self.horizontal_flip:
            # 水平翻转推理
            flipped_image = np.flip(image, axis=3).copy()
            results.append(infer_func(flipped_image))
        
        # 多尺度推理
        if self.multi_scale:
            for scale in self.scales:
                if scale != 1.0:
                    scaled_image = self._scale_image(image, scale)
                    results.append(infer_func(scaled_image))
        
        # 原始图像推理
        results.append(infer_func(image))
        
        # 融合结果
        return self._fuse_results(results)
    
    def _scale_image(self, image, scale):
        """缩放图像"""
        orig_shape = image.shape
        new_shape = (int(orig_shape[2] * scale), int(orig_shape[3] * scale))
        scaled = np.zeros((1, 3, new_shape[1], new_shape[0]), dtype=np.float32)
        for b in range(orig_shape[0]):
            from PIL import Image
            img = Image.fromarray((image[b].transpose(1, 2, 0) * 255).astype(np.uint8))
            img = img.resize(new_shape, Image.BILINEAR)
            scaled[b] = np.array(img).transpose(2, 0, 1) / 255.0
        return scaled
    
    def _fuse_results(self, results):
        """融合多个TTA结果"""
        # 简化版：返回第一个结果
        if results:
            return results[0]
        return []


class YOLOParser:
    """YOLO检测结果解析器"""
    def __init__(self):
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.45
        self.num_classes = 80
        self.use_ciou = True
    
    def set_config(self, config):
        """配置解析器"""
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.num_classes = config.get('num_classes', 80)
        self.use_ciou = config.get('use_ciou', True)
    
    def parse_yolov8(self, output_data, output_shape, original_width, original_height,
                      scale_x, scale_y, pad_x, pad_y):
        """解析YOLOv8输出"""
        predictions = np.squeeze(output_data).T
        num_classes = self.num_classes
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        class_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)
        mask = class_scores > self.confidence_threshold
        boxes = boxes[mask]
        class_scores = class_scores[mask]
        class_ids = class_ids[mask]
        if len(boxes) == 0:
            return []
        boxes = xywh2xyxy(boxes)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        if self.use_ciou:
            keep_indices = ciou_nms([{'bbox': {'x1': b[0], 'y1': b[1], 'x2': b[2], 'y2': b[3]}, 
                                     'score': s} 
                                    for b, s in zip(boxes, class_scores)], self.nms_threshold)
        else:
            keep_indices = nms(boxes, class_scores, self.nms_threshold)
        boxes = boxes[keep_indices]
        class_scores = class_scores[keep_indices]
        class_ids = class_ids[keep_indices]
        detections = []
        for i in range(len(boxes)):
            det = {
                'class_id': int(class_ids[i]),
                'class_name': COCO_CLASSES[int(class_ids[i])],
                'score': float(class_scores[i]),
                'bbox': {
                    'x1': float(max(0, boxes[i, 0])),
                    'y1': float(max(0, boxes[i, 1])),
                    'x2': float(min(original_width, boxes[i, 2])),
                    'y2': float(min(original_height, boxes[i, 3]))
                }
            }
            detections.append(det)
        return detections


def get_coco_class_name(class_id):
    """获取COCO类别名称"""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"class_{class_id}"


class DetectionVisualizer:
    """检测结果可视化"""
    @staticmethod
    def draw_detections(image_data, width, height, detections):
        """绘制检测框"""
        try:
            from PIL import Image, ImageDraw
            if len(image_data.shape) == 4:
                image = (image_data[0].transpose(1, 2, 0) * 255).astype(np.uint8)
            else:
                image = (image_data.transpose(1, 2, 0) * 255).astype(np.uint8)
            img = Image.fromarray(image)
            draw = ImageDraw.Draw(img)
            colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
            for i, det in enumerate(detections):
                bbox = det['bbox']
                color = colors[i % len(colors)]
                draw.rectangle([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']], outline=color, width=2)
                draw.text([bbox['x1'], bbox['y1'] - 15], f"{det['class_name']}: {det['score']:.2f}", fill=color)
            return np.array(img)
        except ImportError:
            print("[警告] PIL未安装，无法可视化")
            return image_data


class TensorMemoryPool:
    """内存池模拟"""
    _instance = None
    
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def allocate_tensor(self, width, height, channels):
        """分配张量内存"""
        return np.zeros((channels, height, width), dtype=np.float32)
    
    def deallocate(self, tensor):
        """释放内存（模拟）"""
        pass


       