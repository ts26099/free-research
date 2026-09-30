"""
영상에서 선수·공 위치를 뽑는 부분 — 모델까지 알아서 마련한다.

검출·추적 자체는 tracker_core.py 가 한다 (축구 전용 YOLO + ByteTrack).
이 파일이 하는 일은 그것을 돌리기 전에 필요한 것을 갖춰 놓는 것이다.

모델을 구하는 순서 — 되는 것이 나올 때까지 차례로 해 본다
  1. 이 폴더의 models/ 에 이미 있으면 그것을 쓴다
  2. 옆에 SoccerTracker 폴더가 있으면 거기서 복사해 온다 (한 번만, 이후로는 독립)
  3. 인터넷에서 내려받는다 (HuggingFace, 축구 전용 모델)
  4. 그래도 안 되면 일반 COCO 모델로라도 돌린다 — 검출이 많이 나빠지므로 경고한다

사람이 할 일은 없다. 처음 한 번만 몇 분 기다리면 된다.
"""

from __future__ import annotations

import json
import os
import shutil
import ssl
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

# 축구 전용 모델 (AGPL-3.0). 파일 이름은 저장소마다 달라서 목록을 받아 고른다.
HF_REPOS = {
    "player": "martinjolif/yolo-football-player-detection",
    "ball": "martinjolif/yolo-football-ball-detection",
}
HF_API = "https://huggingface.co/api/models/{repo}"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{name}"

# 축구 모델을 못 구했을 때 쓸 일반 모델. ultralytics 가 알아서 내려받는다.
COCO_FALLBACK = "yolo11m.pt"

UA = {"User-Agent": "Mozilla/5.0 (soccer-ipi-analyzer)"}
TIMEOUT = 60


def _is_ball_name(name: str) -> bool:
    return "ball" in Path(name).name.lower()


def local_models() -> tuple[Path | None, Path | None]:
    """이 폴더 models/ 안의 (선수 모델, 공 모델). 없으면 None."""
    if not MODELS.is_dir():
        return None, None
    pts = sorted(MODELS.glob("*.pt"))
    player = next((p for p in pts if not _is_ball_name(p.name)), None)
    ball = next((p for p in pts if _is_ball_name(p.name)), None)
    return player, ball


def _copy_from_neighbour(log) -> bool:
    """옆에 SoccerTracker 가 있으면 모델만 복사해 온다."""
    try:
        import pipeline
    except ImportError:
        return False
    folder = pipeline.find_soccertracker(None, ROOT)
    if not folder:
        return False
    src = Path(folder) / "models"
    if not src.is_dir():
        return False
    got = 0
    MODELS.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.glob("*.pt")):
        dst = MODELS / f.name
        if dst.exists():
            continue
        try:
            log(f"    이미 있는 모델을 복사합니다: {f.name} ({f.stat().st_size/1e6:.0f} MB)")
            shutil.copy2(f, dst)
            got += 1
        except OSError as exc:
            log(f"    복사 실패({exc}) — 건너뜁니다")
    return got > 0


def _hf_pick(repo: str, want_ball: bool, log) -> str | None:
    """그 저장소에 들어 있는 .pt 파일 이름 하나를 고른다."""
    try:
        req = urllib.request.Request(HF_API.format(repo=repo), headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, ssl.SSLError, ValueError, OSError) as exc:
        log(f"    모델 목록을 못 받았습니다 ({type(exc).__name__})")
        return None
    names = [s.get("rfilename", "") for s in data.get("siblings", [])]
    pts = [n for n in names if n.lower().endswith(".pt")]
    if not pts:
        return None
    # 이름에 best/last 가 있으면 그것을 먼저, 아니면 첫 번째
    pts.sort(key=lambda n: (0 if "best" in n.lower() else 1, len(n)))
    return pts[0]


def _download(url: str, dst: Path, log, label: str) -> bool:
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r, tmp.open("wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            got, step = 0, max(1, total // 10) if total else 4 << 20
            nxt = step
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got >= nxt:
                    nxt += step
                    if total:
                        log(f"    {label} 내려받는 중 {got/1e6:.0f}/{total/1e6:.0f} MB")
                    else:
                        log(f"    {label} 내려받는 중 {got/1e6:.0f} MB")
        if tmp.stat().st_size < 100_000:      # 너무 작으면 실패 페이지를 받은 것
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(dst)
        return True
    except (urllib.error.URLError, ssl.SSLError, OSError) as exc:
        log(f"    내려받기 실패 ({type(exc).__name__}: {exc})")
        tmp.unlink(missing_ok=True)
        return False


def _download_models(log) -> bool:
    """HuggingFace 에서 축구 전용 모델을 받는다."""
    MODELS.mkdir(parents=True, exist_ok=True)
    got = 0
    for kind, repo in HF_REPOS.items():
        name = _hf_pick(repo, kind == "ball", log)
        if not name:
            continue
        out = MODELS / (f"soccer_{kind}_" + Path(name).name)
        if out.exists():
            got += 1
            continue
        log(f"    {kind} 모델을 내려받습니다 ({repo})")
        if _download(HF_FILE.format(repo=repo, name=name), out, log, kind):
            got += 1
    return got > 0


def ensure_models(log=print) -> tuple[str | None, str | None, str]:
    """
    쓸 모델을 마련한다. (선수 모델 경로, 공 모델 경로, 설명) 을 돌려준다.
    축구 모델을 못 구하면 (COCO 모델 이름, None, 설명).
    """
    player, ball = local_models()
    if player:
        return str(player), (str(ball) if ball else None), "이미 받아 둔 축구 전용 모델"

    log("  검출 모델이 없습니다. 자동으로 마련합니다 (처음 한 번만).")
    if _copy_from_neighbour(log):
        player, ball = local_models()
        if player:
            return str(player), (str(ball) if ball else None), "옆 폴더에서 복사한 축구 전용 모델"

    if _download_models(log):
        player, ball = local_models()
        if player:
            return str(player), (str(ball) if ball else None), "내려받은 축구 전용 모델"

    log("  ! 축구 전용 모델을 구하지 못했습니다 (인터넷이 막혀 있을 수 있습니다).")
    log("    일반 COCO 모델로 진행합니다 — 선수를 절반 정도밖에 못 찾고,")
    log("    심판·골키퍼를 구분하지 못하며, 공은 거의 못 잡습니다.")
    return COCO_FALLBACK, None, "일반 COCO 모델 (축구 전용 모델 없음)"


def has_gpu() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:            # noqa: BLE001
        return False


def auto_quality() -> tuple[int, int, str]:
    """GPU 유무를 보고 해상도와 프레임 간격을 정한다. (imgsz, stride, 설명)"""
    if has_gpu():
        return 1280, 1, "GPU 가 있어 정확하게 (1280 · 전 프레임)"
    return 960, 2, "GPU 가 없어 보통 (960 · 2프레임에 1장)"


def load_core():
    """번들된 추적 엔진(tracker_core.py)을 불러온다."""
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import tracker_core
    sys.modules.setdefault("tracker", tracker_core)   # draw_boxes 호환
    return tracker_core


def make_config(core, model, ball_model, imgsz, stride, save_video):
    cfg = core.Config()
    cfg.model = str(model)
    cfg.ball_model = str(ball_model or "")
    cfg.imgsz = int(imgsz)
    cfg.vid_stride = int(stride)
    cfg.save_video = bool(save_video)
    cfg.skip_done = False
    cfg.watch_folder = False
    cfg.use_openvino = not has_gpu()      # GPU 가 없으면 CPU 가속을 켠다
    cfg.team_imgsz = max(960, int(imgsz))  # 팀 색은 16장만 크게 본다
    return cfg


def track(video: Path, out_root: Path, log, on_progress, should_stop,
          start_sec: float = 0.0, end_sec: float | None = None,
          save_video: bool = True):
    """영상 하나를 추적한다. (tracks.csv 경로, summary) 를 돌려준다."""
    model, ball_model, how = ensure_models(log)
    core = load_core()
    imgsz, stride, why = auto_quality()
    log(f"  {how} · {why}")
    cfg = make_config(core, model, ball_model, imgsz, stride, save_video)
    out_root.mkdir(parents=True, exist_ok=True)
    done = core.run_batch([video], out_root, cfg, log=log, on_progress=on_progress,
                          should_stop=should_stop, start_sec=start_sec, end_sec=end_sec)
    if not done:
        return None, None
    out_dir = core.output_dir_for(video, out_root)
    if start_sec > 0 or end_sec is not None:
        tag = f"{core._clock(start_sec)}-{core._clock(end_sec) if end_sec else 'end'}"
        out_dir = out_dir.with_name(f"{out_dir.name} {tag}")
    csv_path = out_dir / "tracks.csv"
    return (csv_path if csv_path.exists() else None), done[0]
