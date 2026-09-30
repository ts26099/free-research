"""
영상 -> 좌표 -> 기여도 를 한 번에 잇는 부분.

이 프로그램은 추적을 직접 하지 않는다. 이미 잘 만들어져 있는 추적 프로그램
(SoccerTracker 의 tracker.py)을 불러다 쓴다. 검출 모델(.pt)과 조정된 설정이
그쪽에 있고, 같은 일을 두 번 만들 이유가 없기 때문이다.

하는 일
  1. SoccerTracker 폴더를 찾는다 (tracker.py + models/ 가 같이 있는 곳)
  2. 그 tracker.py 를 불러와 영상 하나를 추적한다 -> tracks.csv
  3. 나온 추적이 '선수별 기여도를 낼 만한가'를 summary.json 으로 판정한다
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

# tracker.py 와 모델이 함께 있어야 진짜 SoccerTracker 폴더다
NEEDED = ("tracker.py",)

#  흔히 두는 자리들. 사용자가 직접 고르면 그 값이 우선이다.
def candidate_dirs(here: Path) -> list[Path]:
    home = Path.home()
    out = [
        here, here.parent, here.parent / "SoccerTracker",
        here.parent.parent / "SoccerTracker",
    ]
    for base in (home, home / "Documents", home / "Desktop", home / "Downloads",
                 home / "바탕 화면", home / "바탕화면"):
        out += [base / "SoccerTracker", base]
    # Documents\카카오톡 받은 파일\SoccerTracker_2026-09-17\SoccerTracker 같은 경우
    for base in (home / "Documents", home / "Downloads"):
        if base.is_dir():
            try:
                for sub in sorted(base.iterdir()):
                    if sub.is_dir():
                        out += [sub, sub / "SoccerTracker"]
            except OSError:
                pass
    return out


def looks_like_tracker(folder: Path) -> bool:
    try:
        return all((folder / n).is_file() for n in NEEDED)
    except OSError:
        return False


def find_soccertracker(hint: str | None, here: Path) -> Path | None:
    """추적 프로그램 폴더를 찾는다. 못 찾으면 None."""
    if hint:
        p = Path(hint).expanduser()
        if p.is_file():
            p = p.parent
        if looks_like_tracker(p):
            return p
    seen = set()
    for c in candidate_dirs(here):
        try:
            c = c.resolve()
        except OSError:
            continue
        if c in seen:
            continue
        seen.add(c)
        if looks_like_tracker(c):
            return c
    return None


def load_tracker(folder: Path) -> types.ModuleType:
    """
    그 폴더의 tracker.py 를 불러온다.

    import 만으로는 안 된다 — tracker.py 는 같은 폴더의 draw_boxes.py 를
    결과 영상을 만들 때 부르고, 모델 경로도 자기 위치를 기준으로 찾는다.
    그래서 폴더를 sys.path 에 넣고 그 이름 그대로 불러온다.
    """
    f = folder / "tracker.py"
    if not f.is_file():
        raise FileNotFoundError(f"tracker.py 를 찾을 수 없습니다: {f}")
    s = str(folder)
    if s not in sys.path:
        sys.path.insert(0, s)
    mod = types.ModuleType("tracker")
    mod.__file__ = str(f)
    mod.__name__ = "tracker"
    sys.modules["tracker"] = mod          # draw_boxes 등이 참조할 수 있게
    exec(compile(f.read_text(encoding="utf-8"), str(f), "exec"), mod.__dict__)
    return mod


def missing_packages(folder: Path) -> list[str]:
    """추적에 필요한 것 중 지금 파이썬에 없는 것."""
    out = []
    for m in ("torch", "ultralytics", "cv2"):
        try:
            __import__(m)
        except ImportError:
            out.append({"cv2": "opencv-python"}.get(m, m))
    return out


PRESETS = {
    "빠르게 (640 · 3프레임에 1장)": (640, 3),
    "보통 (960 · 2프레임에 1장)": (960, 2),
    "정확하게 (1280 · 전 프레임)": (1280, 1),
    "설정 파일 그대로": (0, 0),
}


def make_config(tracker, folder: Path, preset: str, save_video: bool):
    """
    추적 설정을 만든다. 그 폴더의 config.json 을 그대로 읽어서 쓰되,
    모델 경로만 절대경로로 바꾼다.

    상대경로("models/soccer_yolo11m.pt")를 그대로 두면 ultralytics 가
    '지금 폴더' 기준으로 찾기 때문에, 이 프로그램에서 부르면 모델을 못 찾는다.
    """
    cfg = tracker.Config.load(folder / "config.json")
    size, stride = PRESETS.get(preset, (0, 0))
    if size:
        cfg.imgsz, cfg.vid_stride = size, stride
        if not str(cfg.model).strip() or not (folder / cfg.model).exists():
            cfg.model = tracker.best_local_model(folder)
    for attr in ("model", "ball_model"):
        val = str(getattr(cfg, attr, "") or "")
        if val and not Path(val).is_absolute():
            p = folder / val
            if p.exists():
                setattr(cfg, attr, str(p))
    cfg.save_video = bool(save_video)
    cfg.skip_done = False          # 이 프로그램이 직접 건너뛰기를 판단한다
    cfg.watch_folder = False
    return cfg


def run_tracking(folder: Path, video: Path, out_root: Path, preset: str,
                 save_video: bool, log, on_progress, should_stop,
                 start_sec: float = 0.0, end_sec: float | None = None):
    """
    영상 하나를 추적해 tracks.csv 를 만든다. (csv 경로, summary dict) 를 돌려준다.
    중간에 멈추면 (None, None).
    """
    tracker = load_tracker(folder)
    cfg = make_config(tracker, folder, preset, save_video)
    log(f"  모델 {Path(str(cfg.model)).name} @{cfg.imgsz}px · "
        f"{cfg.vid_stride}프레임에 1장"
        + (f" · 공 모델 {Path(str(cfg.ball_model)).name}" if str(cfg.ball_model).strip() else ""))
    out_root.mkdir(parents=True, exist_ok=True)
    done = tracker.run_batch([video], out_root, cfg, log=log,
                             on_progress=on_progress, should_stop=should_stop,
                             start_sec=start_sec, end_sec=end_sec)
    if not done:
        return None, None
    summary = done[0]
    out_dir = tracker.output_dir_for(video, out_root)
    if start_sec > 0 or end_sec is not None:
        tag = f"{tracker._clock(start_sec)}-{tracker._clock(end_sec) if end_sec else 'end'}"
        out_dir = out_dir.with_name(f"{out_dir.name} {tag}")
    csv_path = out_dir / "tracks.csv"
    return (csv_path if csv_path.exists() else None), summary


def check_tracking(summary: dict, team_size: int) -> list[tuple[str, str]]:
    """
    추적 결과가 '선수별 기여도'를 낼 만한지 본다. (등급, 문장) 목록을 돌려준다.
    등급 — ok / warn

    가장 중요한 것은 추적 ID 개수다. 한 선수가 여러 번호로 쪼개지면
    선수별 표가 선수를 가리키지 않게 된다.
    """
    out = []
    ids = int(summary.get("unique_track_ids") or 0)
    frames = int(summary.get("frames") or 0)
    ball = int(summary.get("ball_frames") or 0)
    nominal = 2 * int(team_size)

    if ids == 0:
        out.append(("warn", "추적 ID 가 하나도 없습니다. 추적이 실패했습니다."))
    elif ids <= nominal * 1.5:
        out.append(("ok", f"추적 ID {ids}개 (선수 {nominal}명) — 선수별 기여도를 낼 만합니다."))
    elif ids <= nominal * 3:
        out.append(("warn", f"추적 ID 가 {ids}개입니다 (선수 {nominal}명). 일부 선수의 궤적이 "
                            f"끊겨 있습니다. 순위는 참고만 하세요."))
    else:
        out.append(("warn", f"추적 ID 가 {ids}개입니다 (선수 {nominal}명). 한 선수가 여러 번호로 "
                            f"쪼개졌다는 뜻이라, 이 상태의 '선수별' 표는 선수를 가리키지 않습니다."))
        out.append(("warn", "  -> 프레임 간격을 3에서 2나 1로, 해상도를 640에서 960으로 "
                            "올려 다시 돌리면 크게 좋아집니다."))

    if frames and ball:
        pct = ball / frames * 100
        grade = "ok" if pct >= 60 else "warn"
        out.append((grade, f"공을 {pct:.0f}% 프레임에서 찾았습니다 "
                           f"(전술 카메라 1080p 기준 75~80%가 정상)"))
        if pct < 40:
            out.append(("warn", "  -> 공이 너무 자주 끊깁니다. 소유자 판정이 안 되는 구간이 많아집니다."))
    elif frames:
        out.append(("warn", "공을 한 번도 못 찾았습니다. 공 전용 모델이 켜져 있는지 확인하세요."))

    cc = summary.get("class_counts") or {}
    if isinstance(cc, dict) and cc and not any("goalkeep" in str(k).lower() for k in cc):
        out.append(("warn", "골키퍼가 한 번도 안 잡혔습니다. 골문 앞 공간을 센터백이 "
                            "대신 받아 공간 통제 점수가 부풀 수 있습니다."))
    return out


def read_summary(csv_path: Path) -> dict:
    """tracks.csv 옆의 summary.json 을 읽는다. 없으면 빈 dict."""
    f = Path(csv_path).parent / "summary.json"
    if not f.is_file():
        return {}
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}
DATA_EXT = {".csv", ".xlsx", ".xls", ".txt", ".tsv"}


def is_video(p: Path) -> bool:
    return p.suffix.lower() in VIDEO_EXT
