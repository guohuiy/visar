#pragma once

#include <QObject>
#include <QImage>
#include <functional>
#include <string>

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

private:
    bool running_ = false;
    void* engine_ = nullptr;
};
