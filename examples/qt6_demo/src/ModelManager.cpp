#include "ModelManager.h"

ModelManager::ModelManager(QObject *parent)
    : QObject(parent)
{
}

ModelManager::~ModelManager() = default;

QStringList ModelManager::listModels(const QString& path) {
    QDir dir(path);
    return dir.entryList(QStringList() << "*.onnx" << "*.param", QDir::Files);
}

bool ModelManager::loadModel(const QString& path) {
    QFile file(path);
    if (!file.exists()) {
        emit errorOccurred("Model file not found: " + path);
        return false;
    }
    emit modelLoaded(path);
    return true;
}

void ModelManager::setModelPath(const QString& path) {
    modelPath_ = path;
}
