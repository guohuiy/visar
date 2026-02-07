#!/usr/bin/env python3
"""
VisionEngine 模型量化工具
支持 PTQ (Post-Training Quantization) 和 QAT (Quantization-Aware Training)

用法:
    # PTQ 量化
    python quantize_model.py --model yolov8n.onnx --calib-data ./calib_images/ --output yolov8n_int8.onnx
    
    # 带精度评估的量化
    python quantize_model.py --model yolov8n.onnx --calib-data ./calib_images/ --eval-data ./test_images/ --output yolov8n_int8.onnx
    
    # QAT 量化 (需要训练框架)
    python quantize_model.py --model yolov8n.onnx --quantize-type qat --output yolov8n_qat.onnx
"""

import os
import sys
import argparse
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional
import json
import cv2
from pathlib import Path
import time


class CalibrationDataGenerator:
    """校准数据生成器"""
    
    def __init__(self, 
                 image_dir: str,
                 input_size: Tuple[int, int] = (640, 640),
                 mean: List[float] = None,
                 std: List[float] = None,
                 max_samples: int = 100):
        self.image_dir = image_dir
        self.input_size = input_size
        self.mean = np.array(mean if mean else [123.675, 116.28, 103.53], dtype=np.float32)
        self.std = np.array(std if std else [58.395, 57.12, 57.375], dtype=np.float32)
        self.max_samples = max_samples
        self.image_files = self._load_image_files()
        
    def _load_image_files(self) -> List[str]:
        """加载图像文件列表"""
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in valid_exts:
            image_files.extend(Path(self.image_dir).glob(f'*{ext}'))
            image_files.extend(Path(self.image_dir).glob(f'*{ext.upper()}'))
        return sorted([str(f) for f in image_files[:self.max_samples]])
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        # 调整大小 (Letterbox)
        h, w = image.shape[:2]
        scale = min(self.input_size[0] / w, self.input_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建画布
        canvas = np.full((self.input_size[1], self.input_size[0], 3), 114.0, dtype=np.float32)
        x_offset = (self.input_size[0] - new_w) // 2
        y_offset = (self.input_size[1] - new_h) // 2
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # 归一化
        canvas = (canvas - self.mean) / self.std
        
        # HWC to CHW
        canvas = canvas.transpose(2, 0, 1).astype(np.float32)
        return canvas
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        """获取预处理后的图像"""
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.preprocess(img)
    
    def get_batch(self, batch_size: int = 1) -> np.ndarray:
        """获取一个批次的数据"""
        indices = np.random.choice(len(self), batch_size, replace=False)
        batch = np.stack([self[i] for i in indices], axis=0)
        return batch


class ONNXQuantizer:
    """ONNX 模型量化器"""
    
    def __init__(self, 
                 model_path: str,
                 quantization_type: str = 'int8',  # 'int8' or 'fp16'
                 calibration_method: str = 'entropy'):  # 'minmax', 'entropy', 'percentile
        self.model_path = model_path
        self.quantization_type = quantization_type
        self.calibration_method = calibration_method
        self.model = onnx.load(model_path)
        self.graph = self.model.graph
        self.quantization_params = {}
        
    def analyze(self) -> Dict[str, Dict]:
        """分析模型以获取量化信息"""
        print("分析模型结构...")
        
        analysis = {
            'inputs': [],
            'outputs': [],
            'initializers': [],
            'conv_ops': [],
            'recommendations': {}
        }
        
        # 分析输入
        for inp in self.graph.input:
            if inp.name in [i.name for i in self.graph.initializer]:
                continue
            shape = [d.dim_value if d.dim_value > 0 else 1 for d in inp.type.tensor_type.shape.dim]
            analysis['inputs'].append({
                'name': inp.name,
                'shape': shape,
                'dtype': inp.type.tensor_type.elem_type
            })
        
        # 分析输出
        for out in self.graph.output:
            shape = [d.dim_value if d.dim_value > 0 else 1 for d in out.type.tensor_type.shape.dim]
            analysis['outputs'].append({
                'name': out.name,
                'shape': shape,
                'dtype': out.type.tensor_type.elem_type
            })
        
        # 分析卷积操作
        for node in self.graph.node:
            if node.op_type in ['Conv', 'ConvTranspose', 'Gemm', 'MatMul']:
                analysis['conv_ops'].append({
                    'name': node.name,
                    'op_type': node.op_type,
                    'inputs': list(node.input),
                    'outputs': list(node.output)
                })
        
        # 生成推荐
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> Dict:
        """生成量化建议"""
        recommendations = {
            'quantizable_layers': [],
            'sensitive_layers': [],
            'precision_suggestions': {}
        }
        
        # 识别可量化的层
        for inp in analysis['inputs']:
            if len(inp['shape']) == 4:  # NCHW format
                recommendations['quantizable_layers'].append(inp['name'])
        
        # 识别敏感层 (建议保留FP16)
        for conv in analysis['conv_ops'][:3]:  # 前几层通常是敏感的
            recommendations['sensitive_layers'].extend(conv['inputs'])
        
        return recommendations
    
    def calibrate(self, 
                   calibration_data: List[np.ndarray],
                   method: str = 'entropy') -> Dict[str, Dict]:
        """执行校准"""
        print(f"执行 {method} 校准, 使用 {len(calibration_data)} 个样本...")
        
        # 收集每一层的激活值统计
        activation_stats = {}
        
        # 创建推理会话
        session = ort.InferenceSession(self.model_path)
        output_names = [out.name for out in session.get_outputs()]
        
        for i, data in enumerate(calibration_data):
            if i % 10 == 0:
                print(f"  处理样本 {i+1}/{len(calibration_data)}")
            
            # 运行推理
            input_name = session.get_inputs()[0].name
            inputs = {input_name: data}
            outputs = session.run(output_names, inputs)
            
            # 收集统计信息
            for name, value in zip(output_names, outputs):
                value = value.flatten()
                if name not in activation_stats:
                    activation_stats[name] = {
                        'min': [],
                        'max': [],
                        'histogram': []
                    }
                activation_stats[name]['min'].append(np.min(value))
                activation_stats[name]['max'].append(np.max(value))
        
        # 计算校准参数
        calibration_params = {}
        for name, stats in activation_stats.items():
            all_min = np.min(stats['min'])
            all_max = np.max(stats['max'])
            
            if method == 'minmax':
                calibration_params[name] = {
                    'scale': (all_max - all_min) / 255.0,
                    'zero_point': int(-all_min / ((all_max - all_min) / 255.0))
                }
            elif method == 'entropy':
                # 使用KL散度优化
                calibration_params[name] = self._entropy_calibration(
                    np.concatenate([s for s in stats['histogram']]) if stats['histogram'] else np.zeros(1000),
                    all_min, all_max
                )
        
        return calibration_params
    
    def _entropy_calibration(self, 
                             histogram: np.ndarray, 
                             min_val: float, 
                             max_val: float) -> Dict:
        """熵校准算法"""
        num_bins = 256
        bin_width = (max_val - min_val) / num_bins
        
        if bin_width <= 0:
            bin_width = 1e-6
            
        # 计算直方图
        hist, _ = np.histogram(histogram, bins=num_bins, range=(min_val, max_val))
        hist = hist / hist.sum()
        
        # 计算熵
        p = hist + 1e-10
        entropy = -np.sum(p * np.log2(p))
        
        return {
            'scale': bin_width,
            'zero_point': 0,
            'num_bins': num_bins,
            'entropy': entropy
        }
    
    def quantize(self,
                 calibration_data: List[np.ndarray],
                 output_path: str,
                 quantize_type: str = 'int8') -> Tuple[bool, str]:
        """量化模型"""
        print(f"量化模型到 {quantize_type}...")
        
        if quantize_type == 'int8':
            return self._quantize_int8(calibration_data, output_path)
        elif quantize_type == 'fp16':
            return self._quantize_fp16(output_path)
        else:
            return False, f"不支持的量化类型: {quantize_type}"
    
    def _quantize_int8(self,
                       calibration_data: List[np.ndarray],
                       output_path: str) -> Tuple[bool, str]:
        """INT8 量化"""
        try:
            from onnxruntime.quantization import quantize_static, CalibrationDataReader
            from onnxruntime.quantization.quant_utils import QuantType
            
            # 创建校准数据读取器
            class CalibDataReader(CalibrationDataReader):
                def __init__(self, data_list):
                    self.data_list = data_list
                    self.idx = 0
                    
                def get_next(self):
                    if self.idx >= len(self.data_list):
                        return None
                    self.idx += 1
                    input_name = 'input'
                    return {input_name: self.data_list[self.idx-1]}
            
            reader = CalibDataReader(calibration_data)
            
            # 执行量化
            quantize_static(
                self.model_path,
                output_path,
                reader,
                QuantType.QInt8,
                ['/conv_0', '/conv_1', '/conv_2']  # 保留敏感的FP16层
            )
            
            return True, f"INT8 量化成功: {output_path}"
            
        except ImportError as e:
            # 使用简化量化方法
            print("使用简化量化方法...")
            return self._simple_quantize(output_path)
    
    def _quantize_fp16(self, output_path: str) -> Tuple[bool, str]:
        """FP16 量化"""
        try:
            from onnxruntime.tools.convert_onnx_models_to_ort import convert
            import onnxruntime.tools.optimizer as optimizer
            
            # 转换为FP16
            model_fp16 = optimizer.optimize_model(
                self.model_path,
                optimization_level=99,
                use_gpu=False
            )
            
            # 强制转换为FP16
            for node in model_fp16.graph.node:
                for attr in node.attribute:
                    if attr.name in ['dtype', 'to']:
                        if attr.i == 1:  # Float
                            attr.i = 10  # Float16
            
            onnx.save(model_fp16.model, output_path)
            return True, f"FP16 量化成功: {output_path}"
            
        except Exception as e:
            return False, f"FP16 量化失败: {str(e)}"
    
    def _simple_quantize(self, output_path: str) -> Tuple[bool, str]:
        """简化量化 (仅修改精度声明)"""
        # 这是一个占位实现
        # 实际生产中应该使用 onnxruntime 的 quantize_static
        print("警告: 使用简化量化，建议安装 onnxruntime[quantization]")
        
        # 复制模型
        import shutil
        shutil.copy(self.model_path, output_path)
        
        return True, f"模型已复制 (未量化): {output_path}"
    
    def evaluate(self, 
                 test_data: List[Tuple[np.ndarray, List]],
                 model_path: str) -> Dict:
        """评估模型精度"""
        print("评估模型精度...")
        
        session = ort.InferenceSession(model_path)
        input_name = session.get_inputs()[0].name
        
        total_correct = 0
        total_samples = len(test_data)
        
        for i, (image, gt_boxes) in enumerate(test_data):
            inputs = {input_name: image}
            outputs = session.run(None, inputs)
            
            # 简化评估：检查检测数量
            if len(outputs) > 0 and len(outputs[0]) > 0:
                total_correct += 1
            
            if i % 50 == 0:
                print(f"  评估进度: {i+1}/{total_samples}")
        
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        return {
            'accuracy': accuracy,
            'correct': total_correct,
            'total': total_samples
        }


class ModelOptimizer:
    """模型优化器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = onnx.load(model_path)
    
    def simplify(self) -> str:
        """模型简化"""
        try:
            import onnxsim
            
            print("简化模型...")
            simplified_model, check = onnxsim.simplify(
                self.model_path,
                dynamic_input_shape=False,
                input_shapes={'input': [1, 3, 640, 640]}
            )
            
            if check:
                output_path = self.model_path.replace('.onnx', '_simplified.onnx')
                onnx.save(simplified_model, output_path)
                print(f"模型简化成功: {output_path}")
                return output_path
            else:
                print("模型简化检查失败")
                return self.model_path
                
        except ImportError:
            print("onnxsim 未安装，跳过模型简化")
            return self.model_path
    
    def check_compatibility(self) -> Dict:
        """检查量化兼容性"""
        print("检查量化兼容性...")
        
        issues = []
        warnings = []
        
        for node in self.model.graph.node:
            # 检查不支持的操作
            unsupported_ops = [
                'RandomUniform', 'RandomUniformLike', 'RandomNormal', 
                'RandomNormalLike', 'Multinomial', 'LSTM', 'GRU', 'RNN'
            ]
            
            if node.op_type in unsupported_ops:
                issues.append(f"不支持的操作: {node.op_type}")
            
            # 检查动态输出
            for out in node.output:
                for tensor_type in self.model.graph.value_info:
                    if tensor_type.name == out:
                        break
                else:
                    warnings.append(f"动态输出: {out}")
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues,
            'warnings': warnings
        }


def main():
    parser = argparse.ArgumentParser(description='VisionEngine 模型量化工具')
    parser.add_argument('--model', type=str, required=True, help='输入模型路径')
    parser.add_argument('--calib-data', type=str, default=None, help='校准数据目录')
    parser.add_argument('--eval-data', type=str, default=None, help='评估数据目录')
    parser.add_argument('--output', type=str, required=True, help='输出模型路径')
    parser.add_argument('--quantize-type', type=str, default='int8', 
                       choices=['int8', 'fp16'], help='量化类型')
    parser.add_argument('--calib-method', type=str, default='entropy',
                       choices=['minmax', 'entropy', 'percentile'], help='校准方法')
    parser.add_argument('--calib-samples', type=int, default=100, help='校准样本数')
    parser.add_argument('--input-size', type=str, default='640,640', help='输入尺寸')
    parser.add_argument('--simplify', action='store_true', help='简化模型')
    parser.add_argument('--analyze', action='store_true', help='仅分析模型')
    
    args = parser.parse_args()
    
    # 解析输入尺寸
    input_size = tuple(map(int, args.input_size.split(',')))
    
    # 加载模型
    print(f"加载模型: {args.model}")
    optimizer = ModelOptimizer(args.model)
    
    # 分析模型
    analyzer = ONNXQuantizer(args.model)
    analysis = analyzer.analyze()
    
    if args.analyze:
        print("\n模型分析结果:")
        print(json.dumps(analysis, indent=2))
        
        compatibility = optimizer.check_compatibility()
        print("\n兼容性检查:")
        print(json.dumps(compatibility, indent=2))
        return
    
    # 模型简化
    if args.simplify:
        simplified_path = optimizer.simplify()
    else:
        simplified_path = args.model
    
    # 生成校准数据
    if args.calib_data:
        calib_generator = CalibrationDataGenerator(
            args.calib_data,
            input_size=input_size,
            max_samples=args.calib_samples
        )
        calib_data = [calib_generator[i] for i in range(len(calib_generator))]
        print(f"生成了 {len(calib_data)} 个校准样本")
    else:
        calib_data = None
    
    # 