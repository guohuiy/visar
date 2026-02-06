#include "MainWindow.h"
#include "InferenceWorker.h"
#include "ModelManager.h"
#include "OTADialog.h"
#include "PerformanceMonitor.h"

#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGroupBox>
#include <QFileDialog>
#include <QMessageBox>
#include <QStatusBar>
#include <QSplitter>
#include <QDateTime>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , inferenceWorker_(std::make_unique<InferenceWorker>())
    , modelManager_(std::make_unique<ModelManager>())
{
    setupUI();
    setupConnections();
    
    setWindowTitle("VisionEngine Demo - Qt6");
    resize(1200, 800);
}

MainWindow::~MainWindow() {
    inferenceWorker_->stop();
}

void MainWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    QHBoxLayout* mainLayout = new QHBoxLayout(centralWidget);
    QSplitter* splitter = new QSplitter(Qt::Horizontal, centralWidget);
    
    // 左侧：图像显示区
    QWidget* leftWidget = new QWidget();
    QVBoxLayout* leftLayout = new QVBoxLayout(leftWidget);
    
    imageLabel_ = new QLabel("No Image Loaded");
    imageLabel_->setMinimumSize(640, 480);
    imageLabel_->setAlignment(Qt::AlignCenter);
    imageLabel_->setStyleSheet("QLabel { background-color: #2b2b2b; color: #888; "
                               "border: 2px dashed #555; }");
    imageLabel_->setScaledContents(true);
    
    resultLabel_ = new QLabel("Detection Results:");
    resultLabel_->setStyleSheet("QLabel { font-size: 14px; padding: 10px; "
                               "background-color: #1e1e1e; border-radius: 5px; }");
    resultLabel_->setMinimumHeight(80);
    resultLabel_->setWordWrap(true);
    
    leftLayout->addWidget(imageLabel_);
    leftLayout->addWidget(resultLabel_);
    
    // 右侧：控制面板
    QWidget* rightWidget = new QWidget();
    QVBoxLayout* rightLayout = new QVBoxLayout(rightWidget);
    
    // 模型选择
    QGroupBox* modelGroup = new QGroupBox("Model Settings");
    QVBoxLayout* modelLayout = new QVBoxLayout(modelGroup);
    
    modelSelector_ = new QComboBox();
    modelSelector_->addItem("YOLOv5s - Object Detection");
    modelSelector_->addItem("ResNet50 - Classification");
    modelSelector_->addItem("MobileNet-SSD");
    modelLayout->addWidget(modelSelector_);
    
    backendSelector_ = new QComboBox();
    backendSelector_->addItem("ONNX Runtime (CPU)");
    backendSelector_->addItem("ONNX Runtime (CUDA)");
    backendSelector_->addItem("TensorRT (GPU)");
    modelLayout->addWidget(backendSelector_);
    
    // 按钮
    QHBoxLayout* btnLayout = new QHBoxLayout();
    loadImageBtn_ = new QPushButton("Load Image");
    loadVideoBtn_ = new QPushButton("Load Video");
    btnLayout->addWidget(loadImageBtn_);
    btnLayout->addWidget(loadVideoBtn_);
    modelLayout->addLayout(btnLayout);
    
    startInferenceBtn_ = new QPushButton("Run Inference");
    startInferenceBtn_->setStyleSheet("QPushButton { padding: 10px; font-weight: bold; }");
    modelLayout->addWidget(startInferenceBtn_);
    
    // OTA更新
    otaUpdateBtn_ = new QPushButton("Check for Updates");
    modelLayout->addWidget(otaUpdateBtn_);
    
    rightLayout->addWidget(modelGroup);
    
    // 性能监控
    perfMonitor_ = new PerformanceMonitor();
    rightLayout->addWidget(perfMonitor_);
    
    // 日志
    logView_ = new QTextEdit();
    logView_->setMaximumHeight(150);
    logView_->setReadOnly(true);
    rightLayout->addWidget(logView_);
    
    splitter->addWidget(leftWidget);
    splitter->addWidget(rightWidget);
    mainLayout->addWidget(splitter);
    
    // 状态栏
    statusBar()->showMessage("Ready");
}

void MainWindow::setupConnections() {
    connect(loadImageBtn_, &QPushButton::clicked, this, &MainWindow::onLoadImage);
    connect(loadVideoBtn_, &QPushButton::clicked, this, &MainWindow::onLoadVideo);
    connect(startInferenceBtn_, &QPushButton::clicked, this, &MainWindow::onStartInference);
    connect(otaUpdateBtn_, &QPushButton::clicked, this, &MainWindow::onCheckForUpdates);
    
    connect(modelSelector_, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onModelChanged);
}

void MainWindow::loadEngineConfig() {
    logView_->append(QString("[%1] Engine initialized").arg(QDateTime::currentDateTime().toString()));
}

void MainWindow::updateStatusBar(const QString& message) {
    statusBar()->showMessage(message);
}

void MainWindow::onLoadImage() {
    QString fileName = QFileDialog::getOpenFileName(this,
        tr("Open Image"), "", tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
    
    if (!fileName.isEmpty()) {
        currentImagePath_ = fileName;
        QImage image(fileName);
        imageLabel_->setPixmap(QPixmap::fromImage(image.scaled(640, 480, Qt::KeepAspectRatio)));
        updateStatusBar(QString("Loaded: %1").arg(fileName));
    }
}

void MainWindow::onStartCamera() {
    updateStatusBar("Camera not available in demo mode");
}

void MainWindow::onLoadVideo() {
    updateStatusBar("Video loading not implemented yet");
}

void MainWindow::onQuantizationChanged(int index) {
    updateStatusBar(QString("Quantization changed to index: %1").arg(index));
}

void MainWindow::onStartInference() {
    if (currentImagePath_.isEmpty()) {
        QMessageBox::warning(this, "Warning", "Please load an image first");
        return;
    }
    
    QImage image(currentImagePath_);
    inferenceWorker_->runInference(image);
    updateStatusBar("Running inference...");
}

void MainWindow::onModelChanged(int index) {
    updateStatusBar(QString("Model changed to: %1").arg(modelSelector_->currentText()));
}

void MainWindow::onBackendChanged(int index) {
    updateStatusBar(QString("Backend changed to: %1").arg(backendSelector_->currentText()));
}

void MainWindow::onCheckForUpdates() {
    OTADialog ota(this);
    ota.exec();
}

void MainWindow::onInferenceComplete(const std::string& result) {
    resultLabel_->setText(QString::fromStdString(result));
    updateStatusBar("Inference completed");
}

void MainWindow::onPerformanceUpdate(double inferenceTime, double totalTime) {
    perfMonitor_->addSample(inferenceTime);
}
