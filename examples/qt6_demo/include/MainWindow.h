#pragma once

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QComboBox>
#include <QTextEdit>
#include <QTimer>
#include <QDir>
#include <QCoreApplication>
#include <memory>
#include <string>

class InferenceWorker;
class ModelManager;
class PerformanceMonitor;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void onLoadImage();
    void onLoadVideo();
    void onStartCamera();
    void onStartInference();
    void onModelChanged(int index);
    void onBackendChanged(int index);
    void onQuantizationChanged(int index);
    void onCheckForUpdates();
    void onInferenceComplete(const std::string& result);
    void onInferenceError(const QString& error);
    void onPerformanceUpdate(double inferenceTime, double totalTime);
    void onLogMessage(const QString& message);

private:
    void scanAvailableModels();
    QString parseModelDisplayName(const QString& fileName);

private:
    void setupUI();
    void setupConnections();
    void loadEngineConfig();
    void updateStatusBar(const QString& message);

    // UI组件
    QLabel* imageLabel_;
    QLabel* resultLabel_;
    QPushButton* loadImageBtn_;
    QPushButton* loadVideoBtn_;
    QPushButton* startCameraBtn_;
    QPushButton* startInferenceBtn_;
    QPushButton* otaUpdateBtn_;
    QComboBox* modelSelector_;
    QComboBox* backendSelector_;
    QComboBox* quantizationSelector_;
    QTextEdit* logView_;
    
    // 性能监控图表
    PerformanceMonitor* perfMonitor_;
    
    // 业务逻辑
    std::unique_ptr<InferenceWorker> inferenceWorker_;
    std::unique_ptr<ModelManager> modelManager_;
    
    // 当前状态
    QString currentImagePath_;
    QImage currentImage_;
    QTimer updateTimer_;
};
