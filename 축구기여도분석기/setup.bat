@echo off
cd /d "%~dp0"
title 축구 기여도 분석기 - 최초 설치

rem 이 폴더에 venv 를 만들고 필요한 것을 넣습니다.
rem 영상 분석(YOLO)에 PyTorch 가 필요해서 2GB 가까이 받습니다. 한 번만 하면 됩니다.
rem 옆에 SoccerTracker 폴더가 있으면 START.bat 이 그쪽 venv 를 먼저 쓰므로
rem 이 설치 자체가 필요 없습니다.

set "PYEXE="
for %%P in (py python) do (
    if not defined PYEXE (
        %%P -c "import sys,tkinter;sys.exit(0 if sys.version_info>=(3,9) else 1)" >nul 2>&1
        if not errorlevel 1 set "PYEXE=%%P"
    )
)
if not defined PYEXE (
    echo  [오류] Python 3.9 이상을 찾지 못했습니다.
    echo         https://www.python.org/downloads/ 에서 설치하고
    echo         설치 화면의 "Add Python to PATH" 를 반드시 체크하세요.
    exit /b 1
)

echo  가상환경 만드는 중...
%PYEXE% -m venv venv || exit /b 1
set "VPY=venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet

echo  계산 라이브러리 설치 중... (numpy pandas scipy openpyxl matplotlib opencv)
"%VPY%" -m pip install numpy pandas scipy openpyxl matplotlib opencv-python || exit /b 1

echo.
echo  영상 분석용 PyTorch 를 설치합니다. 2GB 가까이 받으므로 시간이 걸립니다.
where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo    GPU 가 없어 CPU 판을 받습니다.
    "%VPY%" -m pip install torch torchvision || exit /b 1
) else (
    echo    NVIDIA GPU 를 찾았습니다. CUDA 판을 받습니다.
    "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    if errorlevel 1 "%VPY%" -m pip install torch torchvision || exit /b 1
)

echo  YOLO 설치 중...
"%VPY%" -m pip install ultralytics || exit /b 1
rem lap 이 없으면 첫 추적의 track_id 가 전부 -1 로 나온다
"%VPY%" -m pip install lap || exit /b 1
"%VPY%" -m pip install openvino onnx tkinterdnd2 scikit-learn
echo.
echo  설치 완료!
exit /b 0
