"""
이미 뽑아둔 tracks.csv 를 원본 영상에 그려서 확인용 영상을 만든다.

YOLO 를 다시 돌리지 않는다. CSV 만 읽어서 그리므로 빠르다.

    python draw_boxes.py                         results/ 전부
    python draw_boxes.py 영상.mp4                 그 영상 하나만
    python draw_boxes.py 영상.mp4 --from 12:30 --to 13:30
    python draw_boxes.py 영상.mp4 --scale 0.5     절반 크기로 (용량 1/4)

옵션
    --from 시각    시작 (초, MM:SS, HH:MM:SS)
    --to   시각    끝
    --scale 배율   출력 크기 (0.5 = 절반). 기본은 가로 960px 로 맞춤
    --full         원본 해상도 그대로 (용량이 4배가 된다)
    --smooth       모든 프레임 기록 (기본은 분석한 프레임만 — 용량 1/stride)
    --labels 방식  none(라벨 없음) / id(기본, 번호만) / full(번호+팀)
    --step N       N프레임마다 한 장만 기록 (용량 1/N, 재생 길이는 그대로)
    --out 경로     출력 파일 지정

90분 경기를 통째로 1080p 로 뽑으면 7GB 가 넘는다. 확인이 목적이라면
--from/--to 로 1~2분만 잘라 보는 편이 훨씬 낫다.

프레임을 건너뛰고 분석했다면(vid_stride) 데이터가 없는 프레임에는 직전
박스를 그대로 둔다. 1/25초 사이에 선수는 거의 안 움직이므로 그게 자연스럽다.
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VIDEOS = ROOT / "videos"

DEFAULT_MAX_WIDTH = 960      # 확인용이므로 이 정도면 충분하고 용량이 1/4 이 된다

# 팀별 색 (BGR)
TEAM_COLOR = {
    "0": (70, 70, 235),      # 빨강
    "1": (235, 150, 60),     # 파랑
    "-1": (160, 160, 160),   # 회색 — 심판·골키퍼
}
BALL_COLOR = (60, 240, 250)   # 노랑
INTERP_COLOR = (170, 170, 170)  # 회색 -- 메운(추정한) 공 위치
KEEPER_COLOR = (0, 190, 255)  # 주황 — 골키퍼
REF_COLOR = (210, 120, 255)   # 보라 — 심판


def _kind(class_name: str) -> str:
    """
    CSV 의 class_name 을 그리기용 종류로 옮긴다.

    COCO 모델은 person / sports ball 만 쓰고, 축구 전용 모델은
    player / goalkeeper / referee / ball 을 쓴다. 둘 다 받아야 한다.
    """
    n = (class_name or "").lower()
    if "ball" in n:
        return "ball"
    if "goalkeep" in n:
        return "keeper"
    if "refere" in n:
        return "referee"
    return "player"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_time(text: str) -> float:
    """'90' / '1:30' / '01:30:00' 을 초로."""
    parts = str(text).strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"시각을 알 수 없습니다: {text}")
    sec = 0.0
    for n in nums:
        sec = sec * 60 + n
    return sec


def load_tracks(csv_path: Path, lo: int = 0, hi: int | None = None):
    """
    프레임 번호 -> 박스 목록. lo~hi 프레임만 담는다 (원본 프레임 번호 기준).

    구간만 뽑을 때 55MB CSV 를 통째로 메모리에 올리지 않으려고 여기서 거른다.
    다만 구간 시작 직전의 박스도 필요하므로 조금 앞부터 받아둔다.
    """
    by_frame: dict[int, list] = defaultdict(list)
    seen = []
    back = max(0, lo - 300)          # 시작 시점에 이미 떠 있던 박스를 위해
    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame"])
            if f < back:
                continue
            if hi is not None and f > hi:
                break                # CSV 는 프레임 순이므로 여기서 멈춰도 된다
            by_frame[f].append(row)
            if len(seen) < 3 and f not in seen:
                seen.append(f)

    stride = 1
    keys = sorted(by_frame)
    if len(keys) > 1:
        gaps = [b - a for a, b in zip(keys, keys[1:])]
        stride = min(gaps) or 1
    return by_frame, stride


def find_source(stem: str) -> Path | None:
    """
    원본 영상을 videos/ 와 흔한 위치에서 찾는다.

    다운로더가 만든 하위 폴더(예: Videos\\4K Video Downloader+\\)에 들어 있는
    경우가 흔해서 한 단계 아래까지 훑는다. 더 깊이 뒤지지는 않는다 —
    영상이 많은 폴더에서 느려지기만 한다.
    """
    exts = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
    roots = [VIDEOS, ROOT, Path.home() / "ultralytics", Path.home() / "Downloads",
             Path.home() / "Videos", Path.home() / "Desktop"]
    for root in roots:
        for ext in exts:
            p = root / f"{stem}{ext}"
            if p.exists():
                return p
    for root in roots:
        if not root.is_dir():
            continue
        try:
            subs = [d for d in root.iterdir() if d.is_dir()]
        except OSError:
            continue
        for sub in subs:
            for ext in exts:
                p = sub / f"{stem}{ext}"
                if p.exists():
                    return p
    return None


def _writer(out_path: Path, fps: float, size: tuple[int, int]):
    """H.264 가 되면 그걸 쓰고(용량이 훨씬 작다), 안 되면 mp4v 로 되돌린다."""
    for tag in ("avc1", "H264", "mp4v"):
        vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*tag), fps, size)
        if vw.isOpened():
            return vw, tag
        vw.release()
    return None, None


def draw(video: Path, csv_path: Path, out_path: Path, log=print,
         start_sec: float = 0.0, end_sec: float | None = None,
         scale: float | None = None, smooth: bool = False,
         labels: str = "id", step: int | None = None) -> bool:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        log(f"  [실패] 영상을 열 수 없습니다: {video}")
        return False

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    lo = max(0, int(start_sec * fps))
    hi = int(end_sec * fps) if end_sec is not None else (total - 1 if total else None)

    by_frame, stride = load_tracks(csv_path, lo, hi)
    if not by_frame:
        cap.release()
        log(f"  [건너뜀] 그 구간에 기록된 박스가 없습니다")
        return False

    # 출력 크기
    if scale is None:
        scale = min(1.0, DEFAULT_MAX_WIDTH / src_w) if src_w > 0 else 1.0
    out_w = max(2, int(src_w * scale) // 2 * 2)      # 코덱이 짝수 크기를 좋아한다
    out_h = max(2, int(src_h * scale) // 2 * 2)

    # 분석하지 않은 프레임은 직전 박스를 그대로 복사한 것뿐이다. 그것까지
    # 다 기록하면 용량만 stride 배로 늘고 보이는 정보는 같다. 기본은 분석한
    # 프레임만 fps/stride 로 기록한다 — 재생 길이는 원본과 똑같다.
    # step: 몇 프레임마다 한 장을 기록할지. 지정 안 하면 분석 간격(stride)을 따른다.
    # 전 프레임 분석(stride 1)이면 그대로 다 쓰게 되어 용량이 커지므로,
    # 확인용으로는 --step 3 처럼 솎아내는 편이 낫다. 재생 길이는 그대로다.
    if step is not None:
        step = max(1, int(step))
    else:
        step = 1 if smooth else max(1, stride)
    out_fps = fps / step

    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw, codec = _writer(out_path, out_fps, (out_w, out_h))
    if vw is None:
        cap.release()
        log(f"  [실패] 출력 영상을 만들 수 없습니다: {out_path}")
        return False

    thick = max(1, round(out_w / 1100))          # 박스 선을 얇게
    fscale = max(0.4, out_w / 1800)              # 좌상단 제목용
    lscale = max(0.28, out_w / 2900)             # 라벨용 — 작게
    lthick = 1
    ball_rad = max(6, round(out_w / 90))         # 공 표시 원의 최소 반지름
    last: list = []
    written = 0
    drawn = 0

    if lo > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        # 시작 직전 박스를 미리 채워둔다
        for f in sorted(k for k in by_frame if k <= lo):
            last = by_frame[f]

    idx = lo
    try:
        while hi is None or idx <= hi:
            ok, img = cap.read()
            if not ok:
                break
            rows = by_frame.get(idx)
            if rows is not None:
                last = rows
            if (idx - lo) % step != 0:
                idx += 1
                continue                      # 이 프레임은 기록하지 않는다
            if scale != 1.0:
                img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)

            # 공은 마지막에 그린다. 공 상자는 5px 남짓이라 선수 상자·라벨이
            # 나중에 덮이면 통째로 가려진다. 실측 — 사용자가 "0~5초에 공을
            # 못 찾는다"고 한 구간은 검출이 정확했고 화면에서 안 보였을 뿐이다.
            for r in sorted(last, key=lambda x: _kind(x["class_name"]) == "ball"):
                x1 = int(float(r["x1"]) * scale); y1 = int(float(r["y1"]) * scale)
                x2 = int(float(r["x2"]) * scale); y2 = int(float(r["y2"]) * scale)
                kind = _kind(r["class_name"])
                is_ball = kind == "ball"
                if is_ball:
                    color = BALL_COLOR
                elif kind == "keeper":
                    color = KEEPER_COLOR
                elif kind == "referee":
                    color = REF_COLOR
                else:
                    color = TEAM_COLOR.get(r["team"], (200, 200, 200))
                # 앞뒤 확실한 공 위치로 메운 추정값. 측정값과 헷갈리지 않게
                # 회색 점선 원으로 다르게 그린다. 옛 CSV 에는 이 열이 없다.
                interp = str(r.get("interpolated", "0")).strip() == "1"
                if is_ball and interp:
                    bx = (x1 + x2) // 2; by = (y1 + y2) // 2
                    rad = max(ball_rad, int((x2 - x1) * 0.9) + 3)
                    for a0 in range(0, 360, 30):          # 15도 그리고 15도 비운다
                        cv2.ellipse(img, (bx, by), (rad, rad), 0, a0, a0 + 15,
                                    INTERP_COLOR, thick + 1, cv2.LINE_AA)
                    drawn += 1
                    continue
                if is_ball:
                    # 상자 대신 눈에 띄는 원으로 그린다. 원본에서 공 지름이
                    # 7~8px 이고 960px 로 줄이면 5px 라, 같은 크기의 상자로는
                    # 선수 상자들 사이에서 보이지 않는다. 위치는 그대로 두고
                    # 표시만 키운다 — 어두운 테를 둘러 잔디·흰 선 위에서도
                    # 뭉개지지 않게 한다.
                    bx = (x1 + x2) // 2; by = (y1 + y2) // 2
                    rad = max(ball_rad, int((x2 - x1) * 0.9) + 3)
                    cv2.circle(img, (bx, by), rad + 1, (20, 20, 20), thick + 2, cv2.LINE_AA)
                    cv2.circle(img, (bx, by), rad, color, thick + 1, cv2.LINE_AA)
                    cv2.drawMarker(img, (bx, by), color, cv2.MARKER_CROSS,
                                   max(4, rad // 2), thick, cv2.LINE_AA)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

                # 라벨은 작게. 팀은 박스 색으로 이미 드러나므로 글자로 또 쓰지 않는다.
                # 예전엔 배경을 채운 큰 라벨이라 선수보다 라벨이 커서, 실제보다
                # 검출이 적어 보이는 착시가 있었다.
                if labels != "none":
                    tid = r["track_id"]
                    if is_ball:
                        # 원 표시가 이미 공이라고 말한다. 글자는 상자보다
                        # 열 배 커서 정작 공을 가리므로 쓰지 않는다.
                        text = ""
                    elif kind == "keeper":
                        text = "GK"
                    elif kind == "referee":
                        text = "REF"
                    elif labels == "full":
                        text = f"{tid} T{r['team']}" if r["team"] in ("0", "1") else str(tid)
                    else:
                        text = "" if tid == "-1" else str(tid)
                    if text:
                        cv2.putText(img, text, (x1, max(9, y1 - 3)), FONT, lscale,
                                    (0, 0, 0), lthick + 2, cv2.LINE_AA)
                        cv2.putText(img, text, (x1, max(9, y1 - 3)), FONT, lscale,
                                    color, lthick, cv2.LINE_AA)
                drawn += 1

            t = idx / fps
            # cv2.putText 는 ASCII 만 그린다. 한글 파일명을 그대로 넣으면
            # 물음표가 줄줄이 찍힌다(실측). 그릴 수 없는 글자는 빼고,
            # 남는 게 없으면 시각만 표시한다 — 파일명은 폴더에 이미 있다.
            name = "".join(ch for ch in video.stem[:40] if 32 <= ord(ch) < 127).strip()
            clock = f"{int(t)//60:02d}:{int(t)%60:02d}  f{idx}"
            head = f"{name}  {clock}" if name else clock
            cv2.putText(img, head, (10, 26), FONT, fscale, (0, 0, 0), thick + 2, cv2.LINE_AA)
            cv2.putText(img, head, (10, 26), FONT, fscale, (255, 255, 255), thick, cv2.LINE_AA)
            vw.write(img)
            written += 1
            idx += 1
    finally:
        cap.release()
        vw.release()

    size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    dur = written / out_fps if out_fps else 0
    log(f"  -> {out_path.name}  ({written:,}프레임 · {int(dur)//60}분{int(dur)%60}초 · "
        f"{out_w}x{out_h} · {out_fps:.1f}fps · {codec} · {size_mb:.1f}MB)")
    if step > 1:
        log(f"     (분석한 프레임만 기록 — 길이는 원본과 같고 용량은 1/{step}. "
            f"매끄럽게 하려면 --smooth)")
    return True


def _parse_args(argv: list[str]):
    opts = {"from": None, "to": None, "scale": None, "out": None,
            "full": False, "smooth": False, "labels": "id", "step": None}
    rest = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--full", "--smooth"):
            opts[a[2:]] = True
        elif a.startswith("--") and a[2:] in ("from", "to", "scale", "out", "labels", "step"):
            if i + 1 >= len(argv):
                raise ValueError(f"{a} 뒤에 값이 필요합니다")
            opts[a[2:]] = argv[i + 1]
            i += 1
        elif a in ("-h", "--help"):
            print(__doc__)
            raise SystemExit(0)
        else:
            rest.append(a)
        i += 1
    return opts, rest


def main(argv: list[str]) -> int:
    opts, rest = _parse_args(argv)
    start = parse_time(opts["from"]) if opts["from"] else 0.0
    end = parse_time(opts["to"]) if opts["to"] else None
    scale = 1.0 if opts["full"] else (float(opts["scale"]) if opts["scale"] else None)

    print("빨강/파랑 = 양 팀 · 주황 = 골키퍼 · 보라 = 심판 · 노랑 = 공\n")
    if not RESULTS.exists():
        print(f"분석 결과 폴더가 없습니다: {RESULTS}")
        return 1

    if rest:
        target = rest[0]
        video = Path(target)
        stem = video.stem if video.suffix else target
        if not video.exists():
            found = find_source(stem)
            if found is None:
                print(f"영상을 찾지 못했습니다: {target}")
                return 1
            video = found
        csv_path = RESULTS / stem / "tracks.csv"
        if not csv_path.exists():
            print(f"분석 결과가 없습니다: {csv_path}")
            return 1
        name = f"{stem}_boxes"
        if opts["from"] or opts["to"]:
            name += f"_{int(start)}-{int(end) if end else 'end'}s"
        out = Path(opts["out"]) if opts["out"] else RESULTS / stem / f"{name}.mp4"
        print(f"[{stem}]\n  원본: {video}")
        return 0 if draw(video, csv_path, out, print, start, end, scale,
                         opts["smooth"], opts["labels"],
                         int(opts["step"]) if opts["step"] else None) else 1

    made = 0
    for d in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        csv_path = d / "tracks.csv"
        if not csv_path.exists():
            continue
        print(f"[{d.name}]")
        video = find_source(d.name)
        if video is None:
            print(f"  [건너뜀] 원본 영상을 못 찾았습니다 ({d.name}.mp4 를 videos/ 에 두세요)")
            continue
        print(f"  원본: {video}")
        if draw(video, csv_path, d / f"{d.name}_boxes.mp4", print, start, end,
                scale, opts["smooth"], opts["labels"],
                int(opts["step"]) if opts["step"] else None):
            made += 1
    print(f"\n{made}개 만들었습니다. results/ 폴더 안에 있습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
