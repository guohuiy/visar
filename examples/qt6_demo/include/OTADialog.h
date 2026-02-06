#pragma once

#include <QDialog>
#include <QProgressBar>
#include <QLabel>
#include <QPushButton>

class OTADialog : public QDialog {
    Q_OBJECT
public:
    explicit OTADialog(QWidget *parent = nullptr);
    ~OTADialog() override;

private slots:
    void checkForUpdates();
    void startDownload();
    void cancelDownload();

private:
    QLabel* statusLabel_;
    QProgressBar* progressBar_;
    QPushButton* checkBtn_;
    QPushButton* downloadBtn_;
    QPushButton* cancelBtn_;
};
