"""
진단 자료 만들기 — 영상을 통째로 보내지 않고도 문제를 찾을 수 있게.

경기 영상은 대개 수백 MB ~ 수 GB 라 주고받기 어렵다. 그런데 무엇이 잘못됐는지
알아내는 데 필요한 것은 영상 전체가 아니라 몇 장의 화면과 기록이다.

이 파일이 만드는 압축파일에 들어가는 것
    log.txt          분석할 때 화면에 나온 내용 전부
    summary.json     추적 통계 (ID 수, 공 발견률, 설정)
    frames/*.jpg     영상 곳곳에서 뽑은 화면 6장 (가로 640px 로 줄임)
    calib/*.jpg      경기장 보정이 무엇을 봤는지 (잔디 영역, 찾은 흰 선)
    tracks_sample.csv 좌표 앞부분 2천 줄
    counts.csv       프레임별 검출 인원 (추적이 끊기는지 보는 데 쓴다)
    clip.mp4         영상 15초만 작게 (넣을지는 고를 수 있다)
    env.txt          파이썬·라이브러리 버전

전부 합쳐 보통 2~5 MB 다.
"""

from __future__ import annotations

import json
import platform
import os
import sys
import tempfile
import zipfile
from pathlib import Path


def _versions() -> str:
    out = [f"python   {sys.version.split()[0]}  ({platform.platform()})"]
    for m in ("numpy", "pandas", "scipy", "cv2", "torch", "ultralytics",
              "openpyxl", "matplotlib", "lap", "openvino", "sklearn"):
        try:
            mod = __import__(m)
            out.append(f"{m:12s} {getattr(mod, '__version__', '?')}")
        except Exception:            # noqa: BLE001
            out.append(f"{m:12s} 없음")
    try:
        import torch
        out.append(f"CUDA 사용가능 {torch.cuda.is_available()}")
        out.append(f"torch 스레드  {torch.get_num_threads()}")
    except Exception:                # noqa: BLE001
        pass
    try:
        out.append(f"CPU 코어     {os.cpu_count()}")
    except Exception:                # noqa: BLE001
        pass
    return "\n".join(out)


def _sample_frames(video: Path, tmp: Path, log, n=6, width=640):
    """영상 곳곳에서 n 장을 뽑아 작게 저장한다."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    try:
        picks = (np.linspace(total * 0.05, total * 0.95, n).astype(int)
                 if total > 0 else range(n))
        for k, idx in enumerate(picks):
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, img = cap.read()
            if not ok:
                continue
            h, w = img.shape[:2]
            s = min(1.0, width / max(1, w))
            small = cv2.resize(img, (int(w * s), int(h * s)),
                               interpolation=cv2.INTER_AREA)
            f = tmp / f"frame_{k:02d}.jpg"
            cv2.imwrite(str(f), small, [cv2.IMWRITE_JPEG_QUALITY, 72])
            out.append(f)
    finally:
        cap.release()
    return out


def _calib_pictures(video: Path, tmp: Path, pitch_l, pitch_w, log):
    """
    경기장 보정이 '무엇을 보고 있는지' 그림으로 남긴다.

    잔디로 인식한 영역과 찾아낸 흰 선을 원본 위에 겹쳐 그린다.
    보정이 틀렸을 때 왜 틀렸는지는 이 그림 한 장이면 대개 보인다.
    """
    try:
        import cv2
        import numpy as np
        import autocalib
    except ImportError:
        return []
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    try:
        for k, frac in enumerate((0.2, 0.5, 0.8)):
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
            ok, img = cap.read()
            if not ok:
                continue
            vis = img.copy()
            mask = autocalib.grass_mask(img, cv2)
            note = []
            if mask is None:
                note.append("grass: NOT FOUND")
            else:
                note.append(f"grass: {(mask > 0).mean() * 100:.0f}% of frame")
                green = np.zeros_like(img)
                green[:, :, 1] = mask
                vis = cv2.addWeighted(vis, 1.0, green, 0.25, 0)
                lines = autocalib.line_pixels(img, mask, cv2)
                h, w = img.shape[:2]
                segs = cv2.HoughLinesP(lines, 1, np.pi / 360, threshold=60,
                                       minLineLength=int(min(w, h) * 0.16),
                                       maxLineGap=25)
                note.append(f"line segments: {0 if segs is None else len(segs)}")
                if segs is not None:
                    for x1, y1, x2, y2 in segs.reshape(-1, 4)[:80]:
                        cv2.line(vis, (x1, y1), (x2, y2), (60, 200, 255), 2)
            for i, t in enumerate(note):
                cv2.putText(vis, t, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(vis, t, (12, 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 1, cv2.LINE_AA)
            h, w = vis.shape[:2]
            s = min(1.0, 900 / max(1, w))
            vis = cv2.resize(vis, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            f = tmp / f"calib_{k}.jpg"
            cv2.imwrite(str(f), vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            out.append(f)
    finally:
        cap.release()
    return out


def _clip(video: Path, tmp: Path, log, seconds=15, width=640):
    """영상 앞쪽(또는 가운데) 몇 초만 작게 잘라 낸다."""
    try:
        import cv2
    except ImportError:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        want = int(seconds * fps)
        start = max(0, (total - want) // 2) if total > want else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        ok, img = cap.read()
        if not ok:
            return None
        h, w = img.shape[:2]
        s = min(1.0, width / max(1, w))
        size = (int(w * s) // 2 * 2, int(h * s) // 2 * 2)
        out = tmp / "clip.mp4"
        vw = None
        for tag in ("avc1", "mp4v"):
            vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*tag), fps, size)
            if vw.isOpened():
                break
            vw.release()
            vw = None
        if vw is None:
            return None
        n = 0
        while n < want:
            if n:
                ok, img = cap.read()
                if not ok:
                    break
            vw.write(cv2.resize(img, size, interpolation=cv2.INTER_AREA))
            n += 1
        vw.release()
        return out if out.exists() and out.stat().st_size > 1000 else None
    finally:
        cap.release()


def _tracks_bits(csv_path: Path, tmp: Path, log, tag=""):
    """좌표 앞부분과 프레임별 인원 수를 뽑는다."""
    out = []
    try:
        import pandas as pd
    except ImportError:
        return out
    try:
        head = pd.read_csv(csv_path, nrows=2000)
        f = tmp / f"{tag}tracks_sample.csv"
        head.to_csv(f, index=False)
        out.append(f)
    except Exception:                # noqa: BLE001
        return out
    try:
        df = pd.read_csv(csv_path, usecols=lambda c: c in
                         ("frame", "track_id", "class_name", "team"))
        g = df.groupby("frame").agg(
            rows=("frame", "size"),
            ids=("track_id", "nunique"))
        if "class_name" in df.columns:
            wide = (df.assign(_n=1).pivot_table(index="frame", columns="class_name",
                                                values="_n", aggfunc="sum")
                    .fillna(0).astype(int))
            g = g.join(wide)
        f = tmp / f"{tag}counts.csv"
        g.to_csv(f)
        out.append(f)
        # 추적 ID 가 얼마나 오래 사는지 — 파편화를 보는 가장 직접적인 숫자
        life = df.groupby("track_id")["frame"].agg(["min", "max", "size"])
        life["frames_alive"] = life["max"] - life["min"] + 1
        f2 = tmp / f"{tag}track_life.csv"
        life.sort_values("size", ascending=False).to_csv(f2)
        out.append(f2)
    except Exception:                # noqa: BLE001
        pass
    return out


def make_bundle(video, result_dir, out_zip, pitch_l=105.0, pitch_w=68.0,
                include_clip=True, log=print) -> Path:
    """진단 자료 압축파일을 만든다. 만든 경로를 돌려준다."""
    result_dir = Path(result_dir)
    out_zip = Path(out_zip)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        (tmp / "env.txt").write_text(_versions(), encoding="utf-8")
        files = [(tmp / "env.txt", "env.txt")]

        for name in ("log.txt", "summary.json"):
            f = result_dir / name
            if f.is_file():
                files.append((f, name))
        cfg = Path(__file__).resolve().parent / "ipi_config.json"
        if cfg.is_file():
            files.append((cfg, "ipi_config.json"))

        # 보정 확인 그림 — 보정이 맞았는지 가리는 가장 확실한 자료
        shot = result_dir / "calib_check.png"
        if shot.is_file():
            files.append((shot, "calib_check.png"))

        csv_path = result_dir / "tracks.csv"
        if csv_path.is_file():
            log("  좌표 요약을 뽑는 중...")
            for f in _tracks_bits(csv_path, tmp, log):
                files.append((f, f.name))

        # ID 손질 뒤의 좌표도 같이 넣는다. 손질 전후를 비교해야 '얼마나
        # 쪼개져 있었나'와 '이어붙이기가 먹혔나'를 한 번에 볼 수 있다.
        st_path = result_dir / "tracks_stitched.csv"
        if st_path.is_file():
            for f in _tracks_bits(st_path, tmp, log, tag="stitched_"):
                files.append((f, f.name))

        if video and Path(video).is_file():
            video = Path(video)
            log("  화면 몇 장을 뽑는 중...")
            for f in _sample_frames(video, tmp, log):
                files.append((f, f"frames/{f.name}"))
            log("  경기장 보정이 무엇을 보는지 그리는 중...")
            for f in _calib_pictures(video, tmp, pitch_l, pitch_w, log):
                files.append((f, f"calib/{f.name}"))
            if include_clip:
                log("  영상 15초를 작게 잘라내는 중...")
                c = _clip(video, tmp, log)
                if c:
                    files.append((c, "clip.mp4"))

        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
            for src, name in files:
                try:
                    z.write(src, name)
                except OSError:
                    pass
    return out_zip
