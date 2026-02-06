#pragma once

#include <QObject>
#include <QDir>

class ModelManager : public QObject {
    Q_OBJECT
public:
    explicit ModelManager(QObject *parent = nullptr);
    ~ModelManager() override;
    
    QStringList listModels(const QString& path);
    bool loadModel(const QString& path);
    void setModelPath(const QString& path);
    
signals:
    void modelLoaded(const QString& name);
    void errorOccurred(const QString& error);
    
private:
    QString modelPath_;
};
