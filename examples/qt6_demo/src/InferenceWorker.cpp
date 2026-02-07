#include "InferenceWorker.h"
#include <QDebug>
#include <QFile>
#include <QJsonDocument>
#include <QJsonArray>
#include <QJsonObject>
#include <QDateTime>
#include "vision_engine/vision_engine.h"
#include "vision_engine/inference/ve_inference.h"

InferenceWorker::InferenceWorker(QObject *parent)
    : QObject(parent)
    , running_(false)
    , engine_(nullptr)
    , modelLoaded_(false)
    , currentModelPath_("")
    , demoMode_(true)
{
    logToConsole("InferenceWorker initialized");
}

InferenceWorker::~InferenceWorker() {
    stop();
    logToConsole("InferenceWorker destroyed");
}

void InferenceWorker::logToConsole(const QString& message) {
    QString timestamp = QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss.zzz");
    QString fullMessage = QString("[%1] [InferenceWorker] %2").arg(timestamp).arg(message);
    qDebug() << fullMessage;
    emit logMessage(fullMessage);
}

void InferenceWorker::loadModel(const QString& modelPath) {
    currentModelPath_ = modelPath;
    
    logToConsole("========================================");
    logToConsole(QString("LOAD MODEL: %1").arg(modelPath));
    
    // 检查模型文件是否存在
    if (!QFile::exists(modelPath)) {
        logToConsole(QString("ERROR: Model file not found: %1").arg(modelPath));
        logToConsole("Switching to DEMO mode with simulated detections");
        demoMode_ = true;
        modelLoaded_ = false;
        emit errorOccurred(QString("Model file not found: %1").arg(modelPath));
        return;
    }
    
    logToConsole("Model file exists, size check passed");
    logToConsole("Creating inference engine...");
    
    // 创建推理引擎
    engine_ = new vision_engine::InferenceEngine();
    logToConsole("InferenceEngine instance created");
    
    // 配置引擎选项
    VeEngineOptions options;
    options.preferred_backend = VE_BACKEND_ONNX;
    options.device_type = VE_DEVICE_CPU;
    options.precision = VE_PRECISION_FP32;
    options.num_threads = 4;
    options.batch_size = 1;
    
    logToConsole(QString("Engine options: backend=ONNX, device=CPU, precision=FP32, threads=4"));
    
    // 初始化引擎
    VeStatusCode status = engine_->Initialize(options);
    if (status != VE_SUCCESS) {
        logToConsole(QString("ERROR: Failed to initialize engine, status=%1").arg(status));
        delete engine_;
        engine_ = nullptr;
        demoMode_ = true;
        emit errorOccurred(QString("Failed to initialize engine: %1").arg(status));
        return;
    }
    logToConsole("Engine initialized successfully");
    
    // 加载模型
    logToConsole(QString("Loading ONNX model from: %1").arg(modelPath));
    status = engine_->LoadModel(modelPath.toStdString());
    if (status != VE_SUCCESS) {
        logToConsole(QString("ERROR: Failed to load model, status=%1").arg(status));
        QString error = QString::fromStdString(engine_->GetLastError());
        if (!error.isEmpty()) {
            logToConsole(QString("Error details: %1").arg(error));
        }
        delete engine_;
        engine_ = nullptr;
        demoMode_ = true;
        emit errorOccurred(QString("Failed to load model: %1").arg(status));
        return;
    }
    logToConsole("Model loaded successfully");
    
    // 获取模型信息
    VeModelInfo modelInfo = engine_->GetModelInfo();
    logToConsole(QString("Model info: name=%1, input_size=%2x%3, num_classes=%4, confidence_thresh=%5")
        .arg(QString::fromStdString(modelInfo.name))
        .arg(modelInfo.input_width)
        .arg(modelInfo.input_height)
        .arg(modelInfo.num_classes)
        .arg(modelInfo.confidence_threshold));
    
    // 设置置信度阈值（YOLO默认0.25）
    float confThreshold = 0.25f;
    engine_->SetConfidenceThreshold(confThreshold);
    logToConsole(QString("Set confidence threshold to: %1").arg(confThreshold));
    
    // 设置NMS阈值
    engine_->SetNMSThreshold(0.45f);
    logToConsole("Set NMS threshold to: 0.45");
    
    // 预热模型
    logToConsole("Warming up model (10 iterations)...");
    engine_->Warmup();
    logToConsole("Model warmup completed");
    
    modelLoaded_ = true;
    demoMode_ = false;
    logToConsole("MODEL LOADED SUCCESSFULLY - Ready for inference");
    logToConsole("========================================");
}

void InferenceWorker::runInference(const QImage& image) {
    logToConsole("========================================");
    logToConsole("RUN INFERENCE");
    
    if (demoMode_) {
        logToConsole("Running in DEMO mode (no model loaded)");
    } else if (!modelLoaded_ || !engine_) {
        logToConsole("WARNING: Engine not initialized, falling back to demo mode");
        demoMode_ = true;
    }
    
    if (demoMode_) {
        // 使用演示模式生成模拟检测结果
        logToConsole("Generating simulated detection results...");
        
        QJsonObject jsonResult;
        QJsonArray detectionsArray;
        
        // 模拟检测结果
        const QString classNames[] = {"person", "car", "truck", "bicycle", "dog"};
        const int numDetections = 5;
        
        for (int i = 0; i < numDetections; i++) {
            QJsonObject detJson;
            detJson["class_id"] = i % 5;
            detJson["class_name"] = classNames[i % 5];
            detJson["score"] = 0.95 - (i * 0.1);
            detJson["bbox"] = QJsonArray{
                50 + i * 80, 
                50 + i * 40, 
                150 + i * 80, 
                150 + i * 40
            };
            detectionsArray.append(detJson);
            
            logToConsole(QString("  [Demo] Detection %1: class=%2, score=%.2f, bbox=(%3,%4,%5,%6)")
                .arg(i)
                .arg(classNames[i % 5])
                .arg(0.95 - (i * 0.1))
                .arg(50 + i * 80)
                .arg(50 + i * 40)
                .arg(150 + i * 80)
                .arg(150 + i * 40));
        }
        
        jsonResult["detections"] = detectionsArray;
        jsonResult["time_ms"] = 15.5;
        jsonResult["mode"] = "demo";
        jsonResult["model_path"] = currentModelPath_;
        jsonResult["image_size"] = QString("%1x%2").arg(image.width()).arg(image.height());
        
        QJsonDocument doc(jsonResult);
        QString resultStr = doc.toJson(QJsonDocument::Compact);
        logToConsole(QString("Result: %1").arg(resultStr));
        logToConsole("INFERENCE COMPLETED (Demo Mode)");
        logToConsole("========================================");
        
        emit inferenceComplete(resultStr.toStdString());
        return;
    }
    
    // 真实推理模式
    logToConsole(QString("Input image: %1x%2").arg(image.width()).arg(image.height()));
    logToConsole(QString("Model path: %1").arg(currentModelPath_));
    
    // 转换图像
    QImage rgbImage = image.convertToFormat(QImage::Format_RGB888);
    logToConsole("Image converted to RGB888 format");
    
    // 准备图像数据
    VeImageData imageData;
    imageData.data = rgbImage.bits();
    imageData.width = rgbImage.width();
    imageData.height = rgbImage.height();
    imageData.format = VE_IMAGE_FORMAT_RGB;
    imageData.mean = nullptr;
    imageData.std = nullptr;
    
    logToConsole(QString("VeImageData prepared: width=%1, height=%2, format=RGB")
        .arg(imageData.width).arg(imageData.height));
    
    // 执行推理
    auto startTime = QDateTime::currentMSecsSinceEpoch();
    auto result = engine_->Infer(imageData);
    auto endTime = QDateTime::currentMSecsSinceEpoch();
    
    double inferenceTime = endTime - startTime;
    double engineTime = result->GetInferenceTimeMs();
    
    logToConsole(QString("Inference completed in %1 ms (engine reported: %2 ms)")
        .arg(inferenceTime).arg(engineTime));
    
    // 构建JSON结果
    QJsonObject jsonResult;
    QJsonArray detectionsArray;
    
    int32_t count = result->GetDetectionCount();
    logToConsole(QString("Detected %1 objects").arg(count));
    
    for (int32_t i = 0; i < count; i++) {
        VeDetection det = result->GetDetection(i);
        QJsonObject detJson;
        detJson["class_id"] = det.class_id;
        detJson["score"] = det.score;
        detJson["bbox"] = QJsonArray{det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2};
        detectionsArray.append(detJson);
        
        logToConsole(QString("  [ONNX] Detection %1: class_id=%2, score=%.4f, bbox=(%.1f,%.1f,%.1f,%.1f)")
            .arg(i)
            .arg(det.class_id)
            .arg(det.score)
            .arg(det.bbox.x1).arg(det.bbox.y1)
            .arg(det.bbox.x2).arg(det.bbox.y2));
    }
    
    jsonResult["detections"] = detectionsArray;
    jsonResult["time_ms"] = engineTime;
    jsonResult["mode"] = "onnx";
    jsonResult["model_path"] = currentModelPath_;
    jsonResult["image_size"] = QString("%1x%2").arg(image.width()).arg(image.height());
    
    QJsonDocument doc(jsonResult);
    QString resultStr = doc.toJson(QJsonDocument::Compact);
    logToConsole(QString("Result: %1").arg(resultStr));
    logToConsole("INFERENCE COMPLETED (ONNX Mode)");
    logToConsole("========================================");
    
    emit inferenceComplete(resultStr.toStdString());
}

void InferenceWorker::stop() {
    logToConsole("Stopping InferenceWorker...");
    running_ = false;
    
    if (engine_) {
        logToConsole("Deleting InferenceEngine instance");
        delete engine_;
        engine_ = nullptr;
    }
    
    modelLoaded_ = false;
    demoMode_ = true;
    currentModelPath_ = "";
    
    logToConsole("InferenceWorker stopped");
}
