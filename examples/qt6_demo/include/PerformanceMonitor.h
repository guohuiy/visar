#pragma once

#include <QWidget>
#include <QVector>
#include <QLabel>

class QChartView;
class QLineSeries;

class PerformanceMonitor : public QWidget {
    Q_OBJECT
public:
    explicit PerformanceMonitor(QWidget *parent = nullptr);
    ~PerformanceMonitor() override;
    
    void addSample(double value);
    void clear();

private:
    QVector<double> samples_;
    QLabel* avgLabel_;
    QLabel* minLabel_;
    QLabel* maxLabel_;
    QChartView* chartView_;
};
