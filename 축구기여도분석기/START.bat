@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"
title 축구 기여도 분석기

rem ------------------------------------------------------------------
rem  더블클릭하면 창이 열립니다. 파이썬을 이 순서로 찾습니다.
rem    1. 이 폴더의 venv
rem    2. 추적 프로그램(SoccerTracker)의 venv  - 이미 다 깔려 있습니다
rem    3. 시스템 파이썬 중 pandas 가 깔린 것
rem    4. 아무것도 없으면 setup.bat 이 필요한 것만 설치합니다
rem ------------------------------------------------------------------

set "PYW="
for %%V in ("venv" "..\venv" "..\SoccerTracker\venv" "..\..\SoccerTracker\venv") do (
    if not defined PYW if exist "%%~V\Scripts\pythonw.exe" set "PYW=%%~V\Scripts\pythonw.exe"
)
if defined PYW (
    start "" "!PYW!" "ipi_app.py" %*
    exit /b 0
)

set "PYEXE="
for %%P in (py python) do (
    if not defined PYEXE (
        %%P -c "import pandas, scipy, numpy, openpyxl" >nul 2>&1
        if not errorlevel 1 set "PYEXE=%%P"
    )
)
if defined PYEXE (
    for /f "delims=" %%W in (
        '!PYEXE! -c "import sys,os;print(os.path.join(os.path.dirname(sys.executable),""pythonw.exe""))"'
    ) do set "PYW=%%W"
    if exist "!PYW!" ( start "" "!PYW!" "ipi_app.py" %* ) else ( !PYEXE! "ipi_app.py" %* )
    exit /b 0
)

echo.
echo  ============================================================
echo    처음 실행입니다. 계산에 필요한 것만 자동으로 설치합니다.
echo    (인터넷 필요 - 2~5분. 이 창을 닫지 마세요)
echo  ============================================================
echo.
call "setup.bat"
if errorlevel 1 (
    echo.
    echo  설치에 실패했습니다. README.txt 의 "안 될 때" 를 보세요.
    pause
    exit /b 1
)
if exist "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" "ipi_app.py" %*
) else (
    echo  설치는 됐는데 실행 파일을 못 찾았습니다. setup.bat 을 다시 실행해 보세요.
    pause
)
exit /b 0
