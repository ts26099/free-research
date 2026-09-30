#!/usr/bin/env bash
# 축구 기여도 분석기 — 리눅스 / macOS 실행
#
#   bash start.sh          창을 연다
#   bash start.sh --check  환경 점검만 한다
#
# 윈도우는 START.bat 을 더블클릭하면 됩니다.

set -u
cd "$(dirname "$(readlink -f "$0" 2>/dev/null || echo "$0")")" || exit 1

info() { printf '\033[36m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m%s\033[0m\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*"; }

need="numpy pandas scipy openpyxl matplotlib"

find_python() {
    for c in python3.12 python3.11 python3.10 python3 python; do
        command -v "$c" >/dev/null 2>&1 || continue
        "$c" -c 'import sys,tkinter; sys.exit(0 if sys.version_info>=(3,9) else 1)' 2>/dev/null \
            && { echo "$c"; return 0; }
    done
    return 1
}

ready() { "$1" -c 'import pandas, scipy, numpy, openpyxl' >/dev/null 2>&1; }

PY=""
for v in ./venv/bin/python ../venv/bin/python ../SoccerTracker/venv/bin/python; do
    [ -x "$v" ] && ready "$v" && { PY="$v"; break; }
done

if [ -z "$PY" ]; then
    sys="$(find_python)" || {
        fail "Python 3.9 이상(tkinter 포함)을 찾지 못했습니다."
        echo "    sudo apt install -y python3 python3-venv python3-tk"
        exit 1
    }
    if ready "$sys"; then
        PY="$sys"
    else
        info "처음 실행입니다. 계산에 필요한 것만 설치합니다 (2~5분)."
        "$sys" -m venv venv || { fail "가상환경 생성 실패"; exit 1; }
        ./venv/bin/python -m pip install --upgrade pip --quiet
        ./venv/bin/python -m pip install $need || { fail "설치 실패"; exit 1; }
        ./venv/bin/python -m pip install opencv-python tkinterdnd2 scikit-learn >/dev/null 2>&1
        PY="./venv/bin/python"
        info "설치 완료."
    fi
fi

if [ "${1:-}" = "--check" ]; then
    echo "===== 환경 점검 ====="
    echo "파이썬   : $($PY -V 2>&1)  ($PY)"
    for m in tkinter numpy pandas scipy openpyxl matplotlib cv2 sklearn; do
        $PY -c "import $m" 2>/dev/null && echo "  $m : 있음" || echo "  $m : 없음"
    done
    echo "DISPLAY  : ${DISPLAY:-(없음 - 창을 띄울 수 없습니다)}"
    echo "====================="
    exit 0
fi

if [ -z "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    warn "화면이 없어 창을 띄울 수 없습니다. 좌표 파일을 인자로 주면 터미널에서 계산합니다:"
    warn "  $PY soccer_ipi_v8.py   (파일 경로는 soccer_ipi_v8.py 의 EXCEL_PATH 에)"
    exit 1
fi

exec "$PY" ipi_app.py "$@"
