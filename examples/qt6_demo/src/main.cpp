#include <QApplication>
#include <QMainWindow>
#include <QSurfaceFormat>
#include <iostream>

int main(int argc, char *argv[]) {
    // 设置OpenGL格式
    QSurfaceFormat format;
    format.setDepthBufferSize(24);
    format.setSamples(4);
    QSurfaceFormat::setDefaultFormat(format);
    
    QApplication app(argc, argv);
    app.setApplicationName("VisionEngine Demo");
    app.setApplicationVersion("1.0.0");
    
    // 创建主窗口
    QMainWindow window;
    window.show();
    
    return app.exec();
}
