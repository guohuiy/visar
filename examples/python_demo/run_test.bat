@echo off
chcp 65001 >nul
echo ============================================================
echo VisionEngine Python 测试脚本
echo ============================================================

REM 设置工作目录
cd /d "%~dp0"

REM 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo.
echo [1/3] 检查依赖...
python -c "import numpy" >nul 2>&1
if errorlevel 1 (
    echo [警告] 未安装numpy，正在安装...
    pip install numpy
)

python -c "import onnxruntime" >nul 2>&1
if errorlevel 1 (
    echo [警告] 未安装onnxruntime，正在安装...
    pip install onnxruntime
)

python -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo [警告] 未安装Pillow，正在安装...
    pip install Pillow
)

echo [2/3] 运行测试...
echo.

REM 运行主测试脚本
python test_vision_engine.py

echo.
echo [3/3] 测试完成
echo ============================================================
echo 更多信息请查看 README.md
echo ============================================================
pause
