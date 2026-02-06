#include "OTADialog.h"
#include <QVBoxLayout>
#include <QHBoxLayout>

OTADialog::OTADialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("OTA Update");
    setMinimumWidth(400);
    
    QVBoxLayout* layout = new QVBoxLayout(this);
    
    statusLabel_ = new QLabel("Ready to check for updates");
    layout->addWidget(statusLabel_);
    
    progressBar_ = new QProgressBar();
    progressBar_->setVisible(false);
    layout->addWidget(progressBar_);
    
    QHBoxLayout* btnLayout = new QHBoxLayout();
    checkBtn_ = new QPushButton("Check for Updates");
    downloadBtn_ = new QPushButton("Download");
    cancelBtn_ = new QPushButton("Cancel");
    cancelBtn_->setEnabled(false);
    
    btnLayout->addWidget(checkBtn_);
    btnLayout->addWidget(downloadBtn_);
    btnLayout->addWidget(cancelBtn_);
    layout->addLayout(btnLayout);
    
    connect(checkBtn_, &QPushButton::clicked, this, &OTADialog::checkForUpdates);
    connect(downloadBtn_, &QPushButton::clicked, this, &OTADialog::startDownload);
    connect(cancelBtn_, &QPushButton::clicked, this, &OTADialog::cancelDownload);
}

OTADialog::~OTADialog() = default;

void OTADialog::checkForUpdates() {
    statusLabel_->setText("Checking for updates...");
    progressBar_->setVisible(true);
    progressBar_->setValue(50);
    
    // 模拟检查完成
    statusLabel_->setText("Current version is up to date.");
    progressBar_->setVisible(false);
}

void OTADialog::startDownload() {
    statusLabel_->setText("Downloading...");
    progressBar_->setVisible(true);
    progressBar_->setValue(0);
    cancelBtn_->setEnabled(true);
    checkBtn_->setEnabled(false);
}

void OTADialog::cancelDownload() {
    progressBar_->setVisible(false);
    statusLabel_->setText("Download cancelled.");
    cancelBtn_->setEnabled(false);
    checkBtn_->setEnabled(true);
}
