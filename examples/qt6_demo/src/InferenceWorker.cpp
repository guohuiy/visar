#include "InferenceWorker.h"
#include <QDebug>

InferenceWorker::InferenceWorker(QObject *parent)
    : QObject(parent)
    , running_(false)
    , engine_(nullptr)
{
}

InferenceWorker::~InferenceWorker() {
    stop();
}

void InferenceWorker::loadModel(const QString& modelPath) {
    qDebug() << "Loading model:" << modelPath;
}

void InferenceWorker::runInference(const QImage& image) {
    running_ = true;
    qDebug() << "Running inference on image:" << image.size();
    
    // 模拟推理完成
    emit inferenceComplete("{\"detections\":[], \"time_ms\":10.5}");
    running_ = false;
}

void InferenceWorker::stop() {
    running_ = false;
}
