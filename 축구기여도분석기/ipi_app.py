"""
축구 기여도 분석기 — 영상을 넣고 [분석 시작] 만 누르면 끝.

START.bat (윈도우) 또는 start.sh (리눅스/맥) 로 실행한다.
사람이 할 일은 둘뿐이다.
    1. 경기 영상을 창에 끌어다 놓는다
    2. [▶ 분석 시작] 을 누른다

나머지는 프로그램이 알아서 한다.
    · 검출 모델이 없으면 받아 온다 (처음 한 번만)
    · GPU 유무를 보고 추적 품질을 정한다
    · 영상에서 선수와 공 위치를 뽑는다                      -> tracks.csv
    · 경기장 흰 선을 찾아 화면 픽셀을 미터로 바꾼다          -> 자동 보정
    · 한 팀 인원을 데이터에서 헤아린다
    · 기여도를 계산한다 (SC · PR · PA · DPI · IPI)         -> ipi_result.xlsx

값을 직접 만지고 싶으면 [고급 설정] 을 펼치면 된다. 안 건드려도 된다.
"""

from __future__ import annotations

import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import types
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import autocalib
import autotrack
import diagnose
import pipeline
import stitch

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "soccer_ipi_v8.py"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "ipi_config.json"
LOGS_DIR = ROOT / "logs"

BG = "#12151a"
PANEL = "#1a1f27"
FG = "#e6e9ef"
MUTED = "#8b95a5"
ACCENT = "#4c8dff"
OKC = "#5fd08a"
WARN = "#ffb35c"

# 고급 설정의 '분석 속도' — 화면 글자 <-> autotrack.SPEED_PRESETS 의 키
SPEED_LABEL = {"auto": "자동", "fast": "빠르게",
               "balanced": "보통", "accurate": "정확하게"}
SPEED_KEY = {v: k for k, v in SPEED_LABEL.items()}

CONFIG_REV = 2          # 설정 파일 손질 판 번호 (load_config 참고)

DEFAULTS = {
    "pitch_l": 105.0, "pitch_w": 68.0, "team_size": 0,     # 0 = 자동
    "from_sec": "", "to_sec": "", "save_video": False,
    "speed": "auto", "stitch": True,
    "norm": "z", "exclude_gk": True, "pos_adjust": True, "pa_learn": True,
    "spec_pv_prox": False, "spec_prog_goaldist": False,
    "manual_src": None, "manual_dst": None,                # 손으로 보정했을 때만
    "results_dir": str(RESULTS_DIR), "config_rev": 0,
}


def _to_sec(text):
    """'90' / '1:30' / '01:30:00' 을 초로. 비어 있으면 None."""
    t = str(text or "").strip()
    if not t:
        return None
    try:
        sec = 0.0
        for part in t.split(":"):
            sec = sec * 60 + float(part)
        return max(0.0, sec)
    except ValueError:
        return None


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in cfg:
                        cfg[k] = v
            # 예전 설정을 한 번만 손본다. 확인용 영상은 기본이 '켬' 이었는데,
            # 이게 분석이 끝난 뒤 영상을 통째로 다시 인코딩하는 단계라 시간을
            # 크게 먹는다. 일부러 켠 사람이 아니라 기본값 그대로 쓰던 사람은
            # 여기서 꺼 준다. 다시 켜면 그 뒤로는 그대로 둔다.
            if isinstance(data, dict) and "config_rev" not in data:
                cfg["save_video"] = False
        except (json.JSONDecodeError, OSError):
            pass
    cfg["config_rev"] = CONFIG_REV
    return cfg


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    except OSError:
        pass


def load_engine() -> types.ModuleType:
    """계산 엔진의 새 사본. 파일마다 깨끗한 상태로 돌리려고 매번 새로 만든다."""
    if not ENGINE.exists():
        raise FileNotFoundError(f"계산 엔진을 찾을 수 없습니다: {ENGINE}")
    mod = types.ModuleType("ipi_engine")
    mod.__file__ = str(ENGINE)
    mod.__name__ = "ipi_engine"
    exec(compile(ENGINE.read_text(encoding="utf-8"), str(ENGINE), "exec"), mod.__dict__)
    return mod


def read_foot_points(csv_path: Path):
    """tracks.csv 에서 사람 행의 발밑 픽셀 좌표와 팀별 인원을 읽는다."""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=lambda c: c in (
        "frame", "track_id", "class_name", "team", "fx", "fy", "cx", "cy"))
    if "class_name" in df.columns:
        name = df["class_name"].astype(str).str.lower()
        df = df[~name.str.contains("ball", na=False)]
    x = df["fx"] if "fx" in df.columns else df.get("cx")
    y = df["fy"] if "fy" in df.columns else df.get("cy")
    if x is None or y is None:
        return None, None, 0
    ok = x.notna() & y.notna()
    # 한 팀 인원은 '프레임당 전체 사람 수 / 2' 로 센다.
    #   팀별로 세면 안 된다 — 골키퍼는 team=-1 로 따로 빠져 있어서
    #   팀 0/1 은 필드 선수만 세게 되고 한 명씩 모자라게 나온다.
    per_team = 0
    if "frame" in df.columns:
        cnt = df[ok].groupby("frame").size()
        if len(cnt):
            per_team = float(cnt.median()) / 2.0
    return x[ok].to_numpy(float), y[ok].to_numpy(float), per_team


def guess_team_size(per_team: float) -> int:
    """프레임당 인원에서 경기 형식을 짐작한다. 5·7·9·11 중 가까운 쪽."""
    if not per_team or per_team <= 0:
        return 11
    return min((5, 7, 9, 11), key=lambda n: abs(n - per_team))


class QueueWriter(io.TextIOBase):
    """엔진의 print 출력을 창의 로그로."""

    def __init__(self, q, tee=None):
        self.q, self.tee, self.buf = q, tee, ""

    def write(self, s):
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.q.put(("log", line))
            if self.tee:
                try:
                    self.tee.write(line + "\n")
                except OSError:
                    pass
        return len(s)

    def flush(self):
        if self.buf:
            self.q.put(("log", self.buf))
            self.buf = ""


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = load_config()
        self.q: queue.Queue = queue.Queue()
        self.pending: list[Path] = []
        self.worker: threading.Thread | None = None
        self.stop_flag = threading.Event()
        self.adv_open = False
        self.last_video = None      # 진단 자료를 만들 때 쓴다
        self.last_result = None
        for d in (Path(self.cfg["results_dir"]), LOGS_DIR):
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        self._build_ui()
        self._enable_dnd()
        for arg in sys.argv[1:]:
            self.add_paths([Path(arg)])

        self.log("경기 영상을 이 창에 끌어다 놓고 [분석 시작] 을 누르세요.", "head")
        self.log("검출 모델·경기장 보정·인원 파악까지 전부 알아서 합니다.")
        self.root.after(100, self._drain)
        self.root.after(400, self._check_env)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        r = self.root
        r.title("축구 기여도 분석기")
        r.geometry("980x760")
        r.minsize(900, 680)
        r.configure(bg=BG)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Head.TLabel", background=BG, foreground=FG,
                        font=("Malgun Gothic", 16, "bold"))
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TButton", padding=6)
        style.configure("Go.TButton", padding=(18, 12),
                        font=("Malgun Gothic", 12, "bold"))
        style.configure("TProgressbar", troughcolor=PANEL, background=ACCENT,
                        borderwidth=0, thickness=10)
        style.configure("TEntry", fieldbackground=PANEL, foreground=FG)
        style.configure("TCombobox", fieldbackground=PANEL, foreground=FG)
        style.map("TCombobox", fieldbackground=[("readonly", PANEL)],
                  foreground=[("readonly", FG)],
                  selectbackground=[("readonly", PANEL)],
                  selectforeground=[("readonly", FG)])
        pad = {"padx": 16}

        head = ttk.Frame(r)
        head.pack(fill="x", pady=(16, 2), **pad)
        ttk.Label(head, text="축구 기여도 분석기", style="Head.TLabel").pack(side="left")
        ttk.Label(head, text="영상을 넣고 시작만 누르면 선수별 기여도까지 나옵니다",
                  style="Muted.TLabel").pack(side="left", padx=(12, 0))

        # --- 대기열 ------------------------------------------------------
        qf = ttk.Frame(r)
        qf.pack(fill="x", pady=(12, 6), **pad)
        bar = ttk.Frame(qf)
        bar.pack(fill="x")
        ttk.Label(bar, text="분석할 영상").pack(side="left")
        ttk.Label(bar, text="  (창에 끌어다 놓아도 됩니다)",
                  style="Muted.TLabel").pack(side="left")
        ttk.Button(bar, text="영상 추가", command=self.pick_videos).pack(side="right")
        ttk.Button(bar, text="비우기", command=self.clear_queue).pack(side="right", padx=6)
        self.listbox = tk.Listbox(qf, height=6, bg=PANEL, fg=FG, borderwidth=0,
                                  highlightthickness=1, highlightbackground="#2a313d",
                                  selectbackground=ACCENT, activestyle="none",
                                  font=("Malgun Gothic", 10))
        self.listbox.pack(fill="x", pady=(6, 0))

        # --- 실행 --------------------------------------------------------
        run = ttk.Frame(r)
        run.pack(fill="x", pady=(12, 6), **pad)
        self.start_btn = ttk.Button(run, text="▶   분석 시작", style="Go.TButton",
                                    command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(run, text="■  중지", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=10)
        ttk.Button(run, text="결과 폴더 열기",
                   command=lambda: self.open_folder(Path(self.cfg["results_dir"]))
                   ).pack(side="right")
        ttk.Button(run, text="진단 자료 만들기", command=self.make_diag
                   ).pack(side="right", padx=8)
        self.adv_btn = ttk.Button(run, text="고급 설정  ▼", command=self.toggle_adv)
        self.adv_btn.pack(side="right", padx=8)

        prog = ttk.Frame(r)
        prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status = ttk.Label(prog, text="대기 중", style="Muted.TLabel")
        self.status.pack(anchor="w", pady=(4, 0))

        # --- 고급 설정 (기본은 접혀 있다) ---------------------------------
        self.adv = ttk.Frame(r)
        a1 = ttk.Frame(self.adv)
        a1.pack(fill="x", pady=(8, 2))
        ttk.Label(a1, text="경기장").grid(row=0, column=0, sticky="w")
        self.pl_var = tk.StringVar(value=str(self.cfg["pitch_l"]))
        self.pw_var = tk.StringVar(value=str(self.cfg["pitch_w"]))
        ttk.Entry(a1, textvariable=self.pl_var, width=6).grid(row=0, column=1, padx=(6, 2))
        ttk.Label(a1, text="x").grid(row=0, column=2)
        ttk.Entry(a1, textvariable=self.pw_var, width=6).grid(row=0, column=3, padx=(2, 4))
        ttk.Label(a1, text="m", style="Muted.TLabel").grid(row=0, column=4, padx=(0, 16))
        ttk.Label(a1, text="한 팀 인원").grid(row=0, column=5, sticky="w")
        self.ts_var = tk.StringVar(value=str(self.cfg["team_size"] or "자동"))
        ttk.Combobox(a1, textvariable=self.ts_var, width=6, state="readonly",
                     values=["자동", "5", "7", "9", "11"]).grid(row=0, column=6, padx=(6, 16))
        ttk.Label(a1, text="구간").grid(row=0, column=7, sticky="w")
        self.from_var = tk.StringVar(value=self.cfg["from_sec"])
        self.to_var = tk.StringVar(value=self.cfg["to_sec"])
        ttk.Entry(a1, textvariable=self.from_var, width=7).grid(row=0, column=8, padx=(6, 2))
        ttk.Label(a1, text="~").grid(row=0, column=9)
        ttk.Entry(a1, textvariable=self.to_var, width=7).grid(row=0, column=10, padx=(2, 4))
        ttk.Label(a1, text="예 12:00 ~ 15:00 · 비우면 전체",
                  style="Muted.TLabel").grid(row=0, column=11, sticky="w")

        a15 = ttk.Frame(self.adv)
        a15.pack(fill="x", pady=2)
        ttk.Label(a15, text="분석 속도").pack(side="left")
        self.speed_var = tk.StringVar(
            value=SPEED_LABEL.get(self.cfg.get("speed", "auto"), "자동"))
        ttk.Combobox(a15, textvariable=self.speed_var, width=8, state="readonly",
                     values=list(SPEED_LABEL.values())).pack(side="left", padx=(6, 8))
        ttk.Label(a15, text="자동 = GPU 있으면 정확하게 · 없으면 보통. "
                            "빠르게는 3배 빠르고 먼 선수를 조금 더 놓친다",
                  style="Muted.TLabel").pack(side="left")
        self.stitch_var = tk.BooleanVar(value=self.cfg.get("stitch", True))
        ttk.Checkbutton(a15, text="끊어진 ID 잇기", variable=self.stitch_var
                        ).pack(side="right")

        a2 = ttk.Frame(self.adv)
        a2.pack(fill="x", pady=2)
        self.vid_var = tk.BooleanVar(value=self.cfg["save_video"])
        self.gk_var = tk.BooleanVar(value=self.cfg["exclude_gk"])
        self.pos_var = tk.BooleanVar(value=self.cfg["pos_adjust"])
        self.learn_var = tk.BooleanVar(value=self.cfg["pa_learn"])
        self.sw1_var = tk.BooleanVar(value=self.cfg["spec_pv_prox"])
        self.sw2_var = tk.BooleanVar(value=self.cfg["spec_prog_goaldist"])
        for text, var in (("박스 그린 확인용 영상 (느려짐)", self.vid_var),
                          ("골키퍼 제외", self.gk_var),
                          ("포지션 보정", self.pos_var),
                          ("PA 지수 학습", self.learn_var)):
            ttk.Checkbutton(a2, text=text, variable=var).pack(side="left", padx=(0, 12))
        ttk.Label(a2, text="|  문서와 다르게:", style="Muted.TLabel").pack(side="left", padx=(4, 8))
        ttk.Checkbutton(a2, text="압박 속도항에 거리 반영", variable=self.sw1_var
                        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(a2, text="전진가치를 골대거리로", variable=self.sw2_var).pack(side="left")

        a3 = ttk.Frame(self.adv)
        a3.pack(fill="x", pady=(2, 6))
        ttk.Label(a3, text="경기장 보정").pack(side="left")
        self.calib_lbl = tk.Label(a3, bg=BG, fg=MUTED, text="")
        self.calib_lbl.pack(side="left", padx=10)
        ttk.Button(a3, text="손으로 보정하기", command=self.calibrate).pack(side="right")
        ttk.Button(a3, text="자동으로 되돌리기", command=self.clear_calib
                   ).pack(side="right", padx=6)
        self._show_calib()

        # --- 로그 --------------------------------------------------------
        lf = ttk.Frame(r)
        lf.pack(fill="both", expand=True, pady=(10, 14), **pad)
        self.logbox = tk.Text(lf, bg=PANEL, fg=FG, borderwidth=0, highlightthickness=1,
                              highlightbackground="#2a313d", wrap="word", state="disabled",
                              font=("Consolas", 9), padx=10, pady=8)
        sb = ttk.Scrollbar(lf, command=self.logbox.yview)
        self.logbox.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.logbox.pack(side="left", fill="both", expand=True)
        self.logbox.tag_configure("warn", foreground=WARN)
        self.logbox.tag_configure("ok", foreground=OKC)
        self.logbox.tag_configure("head", foreground=ACCENT)

    def toggle_adv(self):
        if self.adv_open:
            self.adv.pack_forget()
            self.adv_btn.configure(text="고급 설정  ▼")
        else:
            self.adv.pack(fill="x", padx=16, before=self.logbox.master)
            self.adv_btn.configure(text="고급 설정  ▲")
        self.adv_open = not self.adv_open

    def _enable_dnd(self):
        try:
            self.root.drop_target_register("DND_Files")      # type: ignore[attr-defined]
            self.root.dnd_bind("<<Drop>>", self._on_drop)    # type: ignore[attr-defined]
        except (AttributeError, tk.TclError):
            pass

    def _on_drop(self, event):
        self.add_paths([Path(p) for p in self.root.tk.splitlist(event.data)])

    def _check_env(self):
        """시작할 때 환경을 한 번 살펴 준비 상태를 알려 준다."""
        miss = [m for m in ("torch", "ultralytics") if not _have(m)]
        if miss:
            self.log(f"! 영상 분석에 필요한 {', '.join(miss)} 가 없습니다. "
                     f"setup.bat 을 한 번 실행하면 설치됩니다.", "warn")
        player, ball = autotrack.local_models()
        if player:
            self.log(f"검출 모델 준비됨: {player.name}"
                     + (f" · {ball.name}" if ball else ""), "ok")
        else:
            self.log("검출 모델은 처음 분석할 때 자동으로 마련합니다 (몇 분 걸립니다).")
        if autotrack.has_gpu():
            self.log("GPU 를 쓸 수 있습니다 — 빠르게 돌아갑니다.", "ok")
        else:
            self.log("GPU 가 없어 CPU 로 돕니다. 영상 길이의 서너 배가 걸리니 "
                     "처음에는 짧은 영상이나 [고급 설정] 의 구간을 쓰세요.")

    # -------------------------------------------------------------- 대기열
    def add_paths(self, paths):
        added = 0
        for p in paths:
            found = []
            if p.is_dir():
                for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.m4v", "*.webm"):
                    found += sorted(p.rglob(ext))
            elif p.is_file() and (pipeline.is_video(p)
                                  or p.suffix.lower() in pipeline.DATA_EXT):
                found = [p]
            for f in found:
                if f not in self.pending:
                    self.pending.append(f)
                    kind = "영상" if pipeline.is_video(f) else "좌표"
                    self.listbox.insert("end", f"   [{kind}]  {f.name}")
                    added += 1
        if added:
            self.log(f"{added}개 추가 (대기 {len(self.pending)}개)")
        return added

    def pick_videos(self):
        files = filedialog.askopenfilenames(
            title="분석할 경기 영상 선택",
            filetypes=[("영상 파일", "*.mp4 *.avi *.mov *.mkv *.m4v *.webm"),
                       ("좌표 파일", "*.csv *.xlsx"), ("모든 파일", "*.*")])
        if files:
            self.add_paths([Path(f) for f in files])

    def clear_queue(self):
        if self._busy():
            messagebox.showinfo("분석 중", "분석이 끝난 뒤에 비울 수 있습니다.")
            return
        self.pending.clear()
        self.listbox.delete(0, "end")

    # ------------------------------------------------------------- 보정
    def calibrate(self, video_hint=None):
        try:
            from calibrate import CalibrateWindow
        except ImportError as exc:
            messagebox.showerror("보정 창을 열 수 없음", f"calibrate.py 가 없습니다.\n{exc}")
            return
        try:
            pl, pw = float(self.pl_var.get()), float(self.pw_var.get())
        except ValueError:
            pl, pw = 105.0, 68.0
        hint = video_hint or next((str(p) for p in self.pending if pipeline.is_video(p)), None)

        def done(src, dst):
            self.cfg["manual_src"] = [list(map(float, p)) for p in src]
            self.cfg["manual_dst"] = [list(map(float, p)) for p in dst]
            save_config(self.cfg)
            self._show_calib()
            self.log(f"손으로 보정했습니다 — 기준점 {len(src)}개. "
                     f"이제 자동 보정 대신 이 값을 씁니다.", "ok")

        CalibrateWindow(self.root, pl, pw, on_done=done, video_hint=hint)

    def clear_calib(self):
        self.cfg["manual_src"] = self.cfg["manual_dst"] = None
        save_config(self.cfg)
        self._show_calib()
        self.log("자동 보정으로 되돌렸습니다.")

    def _show_calib(self):
        if self.cfg.get("manual_src"):
            self.calib_lbl.configure(
                text=f"손으로 지정한 기준점 {len(self.cfg['manual_src'])}개를 씁니다", fg=OKC)
        else:
            self.calib_lbl.configure(text="자동 — 영상의 흰 선을 찾아 스스로 맞춥니다", fg=MUTED)

    # ------------------------------------------------------------- 실행
    def _busy(self):
        return self.worker is not None and self.worker.is_alive()

    def collect(self) -> dict:
        def f(var, default):
            try:
                return float(var.get())
            except ValueError:
                return default
        ts = self.ts_var.get().strip()
        self.cfg.update({
            "pitch_l": f(self.pl_var, 105.0), "pitch_w": f(self.pw_var, 68.0),
            "team_size": 0 if ts in ("자동", "") else int(ts),
            "from_sec": self.from_var.get().strip(), "to_sec": self.to_var.get().strip(),
            "save_video": self.vid_var.get(), "exclude_gk": self.gk_var.get(),
            "pos_adjust": self.pos_var.get(), "pa_learn": self.learn_var.get(),
            "spec_pv_prox": self.sw1_var.get(), "spec_prog_goaldist": self.sw2_var.get(),
            "speed": SPEED_KEY.get(self.speed_var.get(), "auto"),
            "stitch": self.stitch_var.get(),
        })
        save_config(self.cfg)
        return self.cfg

    def start(self):
        if self._busy():
            return
        if not self.pending:
            messagebox.showinfo("영상이 없습니다",
                                "경기 영상을 창에 끌어다 놓거나 [영상 추가] 로 고르세요.")
            return
        cfg = self.collect()
        items = list(self.pending)
        self.stop_flag.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.log(f"\n── 분석 시작 · {len(items)}개 ──", "head")
        self.worker = threading.Thread(target=self._run, args=(items, dict(cfg)),
                                       daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag.set()
        self.status.configure(text="중지 요청 — 지금 단계가 끝나면 멈춥니다")

    def _run(self, items, cfg):
        done = 0
        out_root = Path(cfg["results_dir"])
        for item in items:
            if self.stop_flag.is_set():
                break
            try:
                self.q.put(("log", f"\n[{item.name}]", "head"))
                if pipeline.is_video(item):
                    csv_path = self._track(item, cfg, out_root)
                    self.last_video = item
                    if csv_path is None:
                        self.last_result = None
                        continue
                    out_dir = csv_path.parent
                    video = item
                    self.last_result = out_dir
                else:
                    csv_path, out_dir, video = item, out_root / item.stem, None
                    out_dir.mkdir(parents=True, exist_ok=True)
                    self.last_video, self.last_result = None, out_dir
                if self.stop_flag.is_set():
                    break
                H, calib_note = self._calibrate(video, csv_path, cfg)
                csv_path = self._stitch(csv_path, cfg, H)
                if self._analyze(csv_path, cfg, out_dir, H, calib_note):
                    done += 1
            except Exception as exc:                          # noqa: BLE001
                self.q.put(("log", f"  오류: {type(exc).__name__}: {exc}", "warn"))
                for line in traceback.format_exc().splitlines()[-6:]:
                    self.q.put(("log", "    " + line))
        self.q.put(("done", done, len(items)))

    # ------------------------------------------------------- 1단계 · 추적
    def _track(self, video: Path, cfg, out_root: Path):
        self.q.put(("log", "  1 / 3  영상에서 선수와 공을 찾는 중", "head"))
        self.q.put(("stage", "준비", 0, 1))
        for m in ("torch", "ultralytics"):
            if not _have(m):
                self.q.put(("log", f"  {m} 가 없어 영상을 분석할 수 없습니다. "
                                   f"setup.bat 을 한 번 실행하세요.", "warn"))
                return None
        lo = _to_sec(cfg.get("from_sec")) or 0.0
        hi = _to_sec(cfg.get("to_sec"))
        if hi is not None and hi <= lo:
            self.q.put(("log", "  구간의 끝이 시작보다 앞입니다. 전체를 봅니다.", "warn"))
            lo, hi = 0.0, None
        t0 = time.time()
        csv_path, summary = autotrack.track(
            video, out_root,
            log=lambda m: self.q.put(("log", "  " + str(m))),
            on_progress=lambda c, t, e=0.0: self.q.put(("stage", "영상 분석", c, max(t, 1))),
            should_stop=self.stop_flag.is_set,
            start_sec=lo, end_sec=hi, save_video=bool(cfg.get("save_video", False)),
            speed=str(cfg.get("speed", "auto")))
        if self.stop_flag.is_set():
            self.q.put(("log", "  중지했습니다.", "warn"))
            return None
        if csv_path is None or not Path(csv_path).exists():
            self.q.put(("log", "  좌표를 만들지 못했습니다.", "warn"))
            return None
        summary = summary or pipeline.read_summary(csv_path)
        self.q.put(("log", f"  좌표 완료 ({time.time() - t0:.0f}초)", "ok"))
        size = int(cfg["team_size"]) or 11
        for grade, line in pipeline.check_tracking(summary, size):
            self.q.put(("log", "  " + line, grade if grade == "warn" else "ok"))
        return Path(csv_path)

    # ------------------------------------------------------- 2단계 · 보정
    def _calibrate(self, video, csv_path: Path, cfg):
        """화면 픽셀을 미터로 바꾸는 변환을 마련한다. (H, 설명)"""
        if cfg.get("manual_src") and cfg.get("manual_dst"):
            H = autocalib.homography(cfg["manual_src"], cfg["manual_dst"])
            self.q.put(("log", "  2 / 3  경기장 보정 — 손으로 지정한 기준점을 씁니다", "head"))
            return H, "손으로 지정"
        if video is None:
            return None, "없음 (좌표 파일은 이미 미터로 봅니다)"
        self.q.put(("log", "  2 / 3  경기장 보정 — 흰 선을 찾아 미터로 바꾸는 중", "head"))
        self.q.put(("stage", "경기장 보정", 0, 1))
        try:
            px, py, per_team = read_foot_points(csv_path)
        except Exception as exc:                              # noqa: BLE001
            self.q.put(("log", f"  좌표를 못 읽었습니다: {exc}", "warn"))
            return None, "실패"
        if px is None or len(px) < 100:
            self.q.put(("log", "  선수 좌표가 너무 적어 보정을 할 수 없습니다.", "warn"))
            return None, "실패"
        if not int(cfg["team_size"]):
            guess = guess_team_size(per_team)
            cfg["team_size"] = guess
            self.q.put(("log", f"  프레임당 한 팀 {per_team:.0f}명 -> {guess}명 경기로 봅니다"))
        H, how, score, note = autocalib.calibrate(
            video, (px, py), float(cfg["pitch_l"]), float(cfg["pitch_w"]),
            log=lambda m: self.q.put(("log", "  " + str(m))))
        if H is None:
            self.q.put(("log", f"  보정 실패 — {note}", "warn"))
            self.q.put(("log", "  좌표를 미터로 바꾸지 못해 기여도 값은 의미가 없습니다. "
                               "[고급 설정] 의 [손으로 보정하기] 를 써 주세요.", "warn"))
            return None, "실패"
        self.q.put(("log", f"  보정 완료 — {how} (점수 {score:.2f})",
                    "ok" if score >= 0.8 and how == "흰 선" else "warn"))
        if note:
            self.q.put(("log", "  " + note, "warn"))
        return H, f"{how} (점수 {score:.2f})"

    # ------------------------------------------------ 2.5단계 · ID 손질
    def _stitch(self, csv_path: Path, cfg, H):
        """
        끊어진 추적 ID 를 잇고, 사람이 아닌 트랙을 버린다.

        보정(H) 뒤에 한다. 픽셀 거리로는 먼 선수와 가까운 선수의 '1px'가
        6배까지 차이 나서 이어붙일지 말지를 정할 수 없다. 보정에 실패해
        H 가 없으면 이 단계는 건너뛴다 — 그때는 어차피 기여도 값 자체가
        의미가 없으니 손질해 봐야 소용이 없다.
        """
        if H is None or not cfg.get("stitch", True):
            return csv_path
        self.q.put(("log", "  끊어진 추적 ID 를 잇는 중", "head"))
        try:
            new_path, rep = stitch.stitch(
                csv_path, H, float(cfg["pitch_l"]), float(cfg["pitch_w"]),
                log=lambda m: self.q.put(("log", "  " + str(m))))
        except Exception as exc:                              # noqa: BLE001
            self.q.put(("log", f"  손질을 건너뜁니다 ({type(exc).__name__}: {exc})", "warn"))
            return csv_path
        size = int(cfg["team_size"]) or 11
        want = size * 2 + 2                  # 양 팀 + 골키퍼 둘
        if rep["after"] > want * 2:
            self.q.put(("log",
                f"  ! 손질 뒤에도 ID 가 {rep['after']}개다 (선수는 {want}명 안팎이어야 한다). "
                f"한 사람이 여러 번호로 쪼개져 있어 점수가 흩어진다 — "
                f"[고급 설정] 에서 해상도를 올리거나 구간을 짧게 나눠 보라.", "warn"))
        elif rep["merged"] or rep["rows_dropped"]:
            self.q.put(("log", f"  손질 완료 — ID {rep['before']}개 -> {rep['after']}개", "ok"))
        return new_path

    # ----------------------------------------------------- 3단계 · 기여도
    def _analyze(self, csv_path: Path, cfg, out_dir: Path, H, calib_note) -> bool:
        self.q.put(("log", "  3 / 3  기여도를 계산하는 중", "head"))
        self.q.put(("stage", "기여도", 0, 1))
        try:
            tee = open(out_dir / "log.txt", "w", encoding="utf-8")
        except OSError:
            tee = None
        writer = QueueWriter(self.q, tee)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = writer
        ok = False
        try:
            eng = load_engine()
            self._apply(eng, cfg, csv_path, out_dir, H)
            print(f"  경기장 보정: {calib_note}")
            eng.main()
            self.q.put(("log", f"  -> {out_dir}", "ok"))
            ok = True
        except SystemExit as exc:
            writer.flush()
            self.q.put(("log", f"  멈춤: {exc}", "warn"))
        except Exception as exc:                              # noqa: BLE001
            writer.flush()
            if type(exc).__name__ == "Stopped":
                self.q.put(("log", "  중지했습니다.", "warn"))
            else:
                self.q.put(("log", f"  오류: {type(exc).__name__}: {exc}", "warn"))
                for line in traceback.format_exc().splitlines()[-6:]:
                    self.q.put(("log", "    " + line))
        finally:
            writer.flush()
            sys.stdout, sys.stderr = old_out, old_err
            if tee:
                tee.close()
        return ok

    def _apply(self, eng, cfg, path, out_dir, H):
        eng.EXCEL_PATH = str(path)
        eng.SAVE_XLSX = str(out_dir / "ipi_result.xlsx")
        eng.SAVE_PLOT = str(out_dir / "ipi_plot.png")
        eng.PITCH_L, eng.PITCH_W = float(cfg["pitch_l"]), float(cfg["pitch_w"])
        eng.apply_pitch_scale()
        eng.TEAM_SIZE = int(cfg["team_size"]) or 11
        eng.NORM_METHOD = cfg["norm"]
        eng.EXCLUDE_GK = bool(cfg["exclude_gk"])
        eng.POS_ADJUST = bool(cfg["pos_adjust"])
        eng.PA_LEARN = bool(cfg["pa_learn"])
        eng.SPEC_PR_PV_PROXIMITY = bool(cfg["spec_pv_prox"])
        eng.SPEC_PA_PROG_GOALDIST = bool(cfg["spec_prog_goaldist"])
        if H is not None:
            eng.HOMOGRAPHY_H = H
        eng.PROGRESS_CB = lambda stage, cur, total: self.q.put(("stage", stage, cur, total))
        eng.STOP_CB = self.stop_flag.is_set

    # ------------------------------------------------------- 진단 자료
    def make_diag(self):
        """
        무엇이 잘못됐는지 남에게 보여줄 작은 압축파일을 만든다.

        경기 영상은 수백 MB 라 주고받기 어렵다. 그런데 문제를 찾는 데 필요한 것은
        영상 전체가 아니라 화면 몇 장과 기록이다. 그것만 모아 2~5 MB 로 묶는다.
        """
        if self._busy():
            messagebox.showinfo("분석 중", "분석이 끝난 뒤에 만들 수 있습니다.")
            return
        result = self.last_result
        if result is None or not Path(result).is_dir():
            root = Path(self.cfg["results_dir"])
            dirs = [d for d in root.iterdir() if d.is_dir()] if root.is_dir() else []
            if not dirs:
                messagebox.showinfo(
                    "먼저 한 번 돌려 주세요",
                    "진단 자료는 분석을 한 번 돌린 뒤에 만들 수 있습니다.\n"
                    "짧은 구간이라도 한 번 [분석 시작] 을 눌러 주세요.")
                return
            result = max(dirs, key=lambda d: d.stat().st_mtime)
        video = self.last_video
        if video is None or not Path(video).is_file():
            video = next((p for p in self.pending if pipeline.is_video(p)), None)

        ask = messagebox.askyesnocancel(
            "진단 자료 만들기",
            f"결과 폴더: {Path(result).name}\n\n"
            "무엇이 잘못됐는지 알아보는 데 필요한 것만 모아 압축합니다.\n"
            "  · 화면에 나왔던 기록(log.txt)과 추적 통계\n"
            "  · 영상에서 뽑은 화면 6장 (작게 줄인 사진)\n"
            "  · 경기장 보정이 무엇을 보고 있는지 그린 그림 3장\n"
            "  · 좌표 앞부분과 프레임별 인원 수\n\n"
            "[예]   영상 15초도 작게 잘라 함께 넣습니다 (가장 도움이 됩니다)\n"
            "[아니오]  영상은 넣지 않습니다\n"
            "[취소]  그만둡니다")
        if ask is None:
            return

        out = Path(result) / f"진단자료_{Path(result).name}.zip"
        self.log("진단 자료를 만드는 중입니다...", "head")
        self.start_btn.configure(state="disabled")

        def work():
            try:
                diagnose.make_bundle(
                    video, result, out,
                    pitch_l=float(self.cfg["pitch_l"]), pitch_w=float(self.cfg["pitch_w"]),
                    include_clip=bool(ask),
                    log=lambda m: self.q.put(("log", str(m))))
                self.q.put(("diag", str(out)))
            except Exception as exc:                      # noqa: BLE001
                self.q.put(("log", f"  진단 자료 만들기 실패: "
                                   f"{type(exc).__name__}: {exc}", "warn"))
                self.q.put(("diag", ""))

        threading.Thread(target=work, daemon=True).start()

    # --------------------------------------------------------- 메시지 펌프
    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "log":
                    self.log(msg[1], msg[2] if len(msg) > 2 else None)
                elif kind == "stage":
                    _, stage, cur, total = msg
                    if total > 1:
                        pct = min(100, cur * 100 / total)
                        self.progress.configure(value=pct)
                        self.status.configure(text=f"{stage}   {cur:,} / {total:,}  ({pct:.0f}%)")
                    else:
                        self.status.configure(text=f"{stage} ...")
                elif kind == "diag":
                    self.start_btn.configure(state="normal")
                    if msg[1]:
                        mb = Path(msg[1]).stat().st_size / 1e6
                        self.log(f"진단 자료를 만들었습니다 ({mb:.1f} MB)", "ok")
                        self.log(f"   {msg[1]}", "ok")
                        self.open_folder(Path(msg[1]).parent)
                elif kind == "done":
                    self._finished(msg[1], msg[2])
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _finished(self, done, total):
        if done:
            self.listbox.delete(0, "end")
            self.pending.clear()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress.configure(value=100 if done else 0)
        self.status.configure(text=f"완료 — {done}/{total}개" if done else "대기 중")
        self.log(f"── 끝났습니다 ({done}/{total}개) ──", "head" if done else "warn")
        if done:
            self.log(f"   결과: {self.cfg['results_dir']}", "ok")

    # ------------------------------------------------------------- 잡다
    def log(self, msg, tag=None):
        self.logbox.configure(state="normal")
        if tag is None:
            low = str(msg)
            tag = ("warn" if low.strip().startswith("!") or "**FAIL**" in low
                   else ("ok" if "PASS" in low else None))
        self.logbox.insert("end", str(msg) + "\n", tag or ())
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    @staticmethod
    def open_folder(path: Path):
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))          # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError:
            pass

    def on_close(self):
        if self._busy():
            if not messagebox.askyesno("분석 중", "분석이 진행 중입니다. 정말 닫을까요?"):
                return
            self.stop_flag.set()
        self.collect()
        self.root.destroy()


def _have(mod: str) -> bool:
    import importlib.util
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def _attach_streams():
    if sys.stdout is not None and sys.stderr is not None:
        return
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        stream = open(LOGS_DIR / "console.log", "a", encoding="utf-8", buffering=1)
    except OSError:
        return
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


def main() -> int:
    _attach_streams()
    try:
        from tkinterdnd2 import TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        root = tk.Tk()
    icon = ROOT / "app.ico"
    if icon.exists():
        try:
            root.iconbitmap(str(icon))
        except tk.TclError:
            pass
    try:
        App(root)
    except Exception as exc:                     # noqa: BLE001
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        (LOGS_DIR / "crash.log").write_text(traceback.format_exc(), encoding="utf-8")
        messagebox.showerror("실행 실패",
                             f"프로그램을 여는 중 오류가 났습니다.\n\n{type(exc).__name__}: {exc}\n\n"
                             f"자세한 내용: {LOGS_DIR / 'crash.log'}")
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
