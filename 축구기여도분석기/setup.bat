@echo off
cd /d "%~dp0"
title 축구 기여도 분석기 - 최초 설치

rem 이 폴더 안에 venv 를 만들고 계산에 필요한 것만 넣습니다.
rem 무거운 torch / ultralytics 는 필요 없습니다 (추적은 이미 끝난 좌표를 받으므로).

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
echo  계산 라이브러리 설치 중... (numpy pandas scipy openpyxl matplotlib)
"venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
"venv\Scripts\python.exe" -m pip install numpy pandas scipy openpyxl matplotlib || exit /b 1
echo  선택 기능 설치 중... (경기장 보정용 opencv, 드래그앤드롭)
"venv\Scripts\python.exe" -m pip install opencv-python tkinterdnd2 scikit-learn
echo.
echo  설치 완료!
exit /b 0
