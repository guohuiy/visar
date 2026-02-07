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
    // 模型列表将由scanAvailableModels()动态填充
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
    
    // 连接推理信号
    connect(inferenceWorker_.get(), &InferenceWorker::inferenceComplete,
            this, &MainWindow::onInferenceComplete);
    connect(inferenceWorker_.get(), &InferenceWorker::errorOccurred,
            this, &MainWindow::onInferenceError);
    
    // 连接日志信号
    connect(inferenceWorker_.get(), &InferenceWorker::logMessage,
            this, &MainWindow::onLogMessage);
    
    // 初始化日志
    logView_->append(QString("[%1] ===== VisionEngine Demo Started =====")
        .arg(QDateTime::currentDateTime().toString()));
    logView_->append(QString("[%1] Qt6 Demo Application Initialized")
        .arg(QDateTime::currentDateTime().toString()));
    
    // 扫描可用模型并填充模型选择器
    scanAvailableModels();
    
    // 加载默认模型（第一个可用模型）
    if (modelSelector_->count() > 0) {
        onModelChanged(modelSelector_->currentIndex());
    } else {
        logView_->append(QString("[%1] WARNING: No models found in models/ folder")
            .arg(QDateTime::currentDateTime().toString()));
    }
}

void MainWindow::scanAvailableModels() {
    // 使用程序所在目录的models文件夹
    QString appPath = QCoreApplication::applicationDirPath();
    QString modelsPath = QDir(appPath).filePath("models");
    
    logView_->append(QString("[%1] Scanning for available models...")
        .arg(QDateTime::currentDateTime().toString()));
    logView_->append(QString("[%1] Models directory: %2")
        .arg(QDateTime::currentDateTime().toString()).arg(modelsPath));
    
    QDir modelsDir(modelsPath);
    if (!modelsDir.exists()) {
        logView_->append(QString("[%1] WARNING: models/ folder not found at: %2")
            .arg(QDateTime::currentDateTime().toString()).arg(modelsPath));
        return;
    }
    
    // 获取所有onnx文件
    QStringList filters;
    filters << "*.onnx";
    modelsDir.setNameFilters(filters);
    
    QFileInfoList fileList = modelsDir.entryInfoList(QDir::Files);
    
    if (fileList.isEmpty()) {
        logView_->append(QString("[%1] WARNING: No .onnx model files found in models/")
            .arg(QDateTime::currentDateTime().toString()));
        return;
    }
    
    logView_->append(QString("[%1] Found %2 model(s):")
        .arg(QDateTime::currentDateTime().toString())
        .arg(fileList.size()));
    
    // 清空现有选项
    modelSelector_->clear();
    
    // 遍历文件并添加
    for (const QFileInfo& fileInfo : fileList) {
        QString fileName = fileInfo.fileName();
        QString displayName = fileName;
        
        // 解析模型名称
        displayName = parseModelDisplayName(fileName);
        
        // 添加到选择器
        modelSelector_->addItem(displayName, fileName);  // 保存真实文件名
        
        logView_->append(QString("[%1]   - %2 (%3)")
            .arg(QDateTime::currentDateTime().toString())
            .arg(displayName)
            .arg(fileInfo.size() / 1024.0, 0, 'f', 1).append(" KB"));
    }
    
    logView_->append(QString("[%1] Model scan completed. %2 model(s) available.")
        .arg(QDateTime::currentDateTime().toString())
        .arg(modelSelector_->count()));
}

QString MainWindow::parseModelDisplayName(const QString& fileName) {
    // 从文件名提取模型显示名称
    QString name = fileName;
    
    // 移除扩展名 (.onnx)
    if (name.endsWith(".onnx", Qt::CaseInsensitive)) {
        name = name.left(name.length() - 5);
    }
    
    // 规范化模型名称
    if (name.contains("yolov8n", Qt::CaseInsensitive)) {
        return "YOLOv8n - Object Detection";
    } else if (name.contains("yolov8s", Qt::CaseInsensitive)) {
        return "YOLOv8s - Object Detection";
    } else if (name.contains("yolov8m", Qt::CaseInsensitive)) {
        return "YOLOv8m - Object Detection";
    } else if (name.contains("yolov8l", Qt::CaseInsensitive)) {
        return "YOLOv8l - Object Detection";
    } else if (name.contains("yolov8x", Qt::CaseInsensitive)) {
        return "YOLOv8x - Object Detection";
    } else if (name.contains("yolov5n", Qt::CaseInsensitive)) {
        return "YOLOv5n - Object Detection";
    } else if (name.contains("yolov5s", Qt::CaseInsensitive)) {
        return "YOLOv5s - Object Detection";
    } else if (name.contains("yolov5m", Qt::CaseInsensitive)) {
        return "YOLOv5m - Object Detection";
    } else if (name.contains("yolov5l", Qt::CaseInsensitive)) {
        return "YOLOv5l - Object Detection";
    } else if (name.contains("yolov5x", Qt::CaseInsensitive)) {
        return "YOLOv5x - Object Detection";
    } else if (name.contains("resnet50", Qt::CaseInsensitive)) {
        return "ResNet50 - Classification";
    } else if (name.contains("resnet", Qt::CaseInsensitive)) {
        return QString("ResNet - %1").arg(name);
    } else if (name.contains("mobilenet", Qt::CaseInsensitive)) {
        return "MobileNet - Classification";
    } else if (name.contains("ssd", Qt::CaseInsensitive)) {
        return "SSD - Object Detection";
    } else if (name.contains("yolo", Qt::CaseInsensitive)) {
        return QString("YOLO - %1").arg(name);
    } else {
        // 智能格式化：下划线转空格，首字母大写
        QString formatted = name;
        formatted.replace('_', ' ');
        if (!formatted.isEmpty()) {
            formatted[0] = formatted[0].toUpper();
        }
        return formatted;
    }
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
    if (index < 0 || modelSelector_->count() == 0) {
        logView_->append(QString("[%1] WARNING: No model selected")
            .arg(QDateTime::currentDateTime().toString()));
        return;
    }
    
    QString modelName = modelSelector_->currentText();
    QString modelFileName = modelSelector_->itemData(index).toString();
    QString modelPath = QString("models/%1").arg(modelFileName);
    
    updateStatusBar(QString("Model changed to: %1").arg(modelName));
    
    logView_->append(QString("[%1] === Model Selection Changed ===")
        .arg(QDateTime::currentDateTime().toString()));
    logView_->append(QString("[%1] Selected model: %2")
        .arg(QDateTime::currentDateTime().toString()).arg(modelName));
    logView_->append(QString("[%1] Model file: %2")
        .arg(QDateTime::currentDateTime().toString()).arg(modelFileName));
    logView_->append(QString("[%1] Model path: %2")
        .arg(QDateTime::currentDateTime().toString()).arg(modelPath));
    
    // 检查模型文件是否存在
    if (!QFile::exists(modelPath)) {
        logView_->append(QString("[%1] WARNING: Model file not found: %2")
            .arg(QDateTime::currentDateTime().toString()).arg(modelPath));
        logView_->append(QString("[%1] Will use demo mode with simulated detections")
            .arg(QDateTime::currentDateTime().toString()));
    } else {
        logView_->append(QString("[%1] Model file exists, loading...")
            .arg(QDateTime::currentDateTime().toString()));
        inferenceWorker_->loadModel(modelPath);
    }
    
    logView_->append(QString("[%1] ============================")
        .arg(QDateTime::currentDateTime().toString()));
}

void MainWindow::onInferenceError(const QString& error) {
    resultLabel_->setText(QString("Error: %1").arg(error));
    logView_->append(QString("[%1] Inference Error: %2")
        .arg(QDateTime::currentDateTime().toString()).arg(error));
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

void MainWindow::onLogMessage(const QString& message) {
    logView_->append(message);
}
