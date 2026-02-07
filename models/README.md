# YOLO模型文件下载说明

由于网络限制，请手动下载以下模型文件并放置在此目录：

## 下载地址

### YOLOv8n (推荐)
- URL: https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx
- 保存为: yolov8n.onnx
- 大小: ~6MB

### YOLOv5s
- URL: https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx
- 保存为: yolov5s.onnx
- 大小: ~28MB

### ResNet50
- URL: https://github.com/onnx/models/raw/main/vision/classification/resnet/model/resnet50-v2-7.onnx
- 保存为: resnet50.onnx
- 大小: ~97MB

### MobileNet-SSD
- URL: https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/ssd-mobilenetv1/ssd_mobilenet_v1_10.onnx
- 保存为: mobilenet_ssd.onnx
- 大小: ~24MB

## 下载后

1. 下载完成后，将文件保存到此目录 (models/)
2. 重新启动 VisionEngineDemo 程序
3. 程序将自动加载模型进行真实推理

## 临时解决方案

如果无法下载模型，程序将自动使用演示模式，生成模拟检测结果供测试使用。
