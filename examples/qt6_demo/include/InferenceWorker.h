#pragma once

#include <QObject>
#include <QImage>
#include <QString>
#include <functional>
#include <string>

// 包含VisionEngine类型
#include "vision_engine/core/ve_types.h"
#include "vision_engine/core/ve_error.h"
#include "vision_engine/inference/ve_inference.h"

class InferenceWorker : public QObject {
    Q_OBJECT
public:
    explicit InferenceWorker(QObject *parent = nullptr);
    ~InferenceWorker() override;

public slots:
    void loadModel(const QString& modelPath);
    void runInference(const QImage& image);
    void stop();

signals:
    void inferenceComplete(const std::string& resultJson);
    void errorOccurred(const QString& error);
    void progressChanged(int progress);
    void logMessage(const QString& message);  // 日志信号

public:
    // 获取当前状态
    bool isModelLoaded() const { return modelLoaded_; }
    QString getCurrentModelPath() const { return currentModelPath_; }
    bool isDemoMode() const { return demoMode_; }

private:
    void logToConsole(const QString& message);

    bool running_ = false;
    bool modelLoaded_ = false;
    bool demoMode_ = true;
    QString currentModelPath_;
    vision_engine::InferenceEngine* engine_ = nullptr;
};
