#include "PerformanceMonitor.h"
#include <QVBoxLayout>
#include <QLabel>

PerformanceMonitor::PerformanceMonitor(QWidget *parent)
    : QWidget(parent)
{
    QVBoxLayout* layout = new QVBoxLayout(this);
    
    QHBoxLayout* statsLayout = new QHBoxLayout();
    avgLabel_ = new QLabel("Avg: 0ms");
    minLabel_ = new QLabel("Min: 0ms");
    maxLabel_ = new QLabel("Max: 0ms");
    statsLayout->addWidget(avgLabel_);
    statsLayout->addWidget(minLabel_);
    statsLayout->addWidget(maxLabel_);
    layout->addLayout(statsLayout);
    
    avgLabel_->setStyleSheet("color: #4CAF50;");
    minLabel_->setStyleSheet("color: #2196F3;");
    maxLabel_->setStyleSheet("color: #FF9800;");
}

PerformanceMonitor::~PerformanceMonitor() = default;

void PerformanceMonitor::addSample(double value) {
    samples_.append(value);
    
    if (samples_.size() > 100) {
        samples_.removeFirst();
    }
    
    double avg = 0, minVal = value, maxVal = value;
    for (double v : samples_) {
        avg += v;
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    avg /= samples_.size();
    
    avgLabel_->setText(QString("Avg: %1ms").arg(avg, 0, 'f', 1));
    minLabel_->setText(QString("Min: %1ms").arg(minVal, 0, 'f', 1));
    maxLabel_->setText(QString("Max: %1ms").arg(maxVal, 0, 'f', 1));
}

void PerformanceMonitor::clear() {
    samples_.clear();
    avgLabel_->setText("Avg: 0ms");
    minLabel_->setText("Min: 0ms");
    maxLabel_->setText("Max: 0ms");
}
