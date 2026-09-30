"""
축구 기여도 분석기 — 창 하나로 끝내는 SC · PR · PA · IPI 계산 도구.

START.bat (윈도우) 또는 start.sh (리눅스/맥) 로 실행한다.
선수 좌표 파일을 창에 끌어다 놓거나 [파일 추가] 로 고르고 [분석 시작] 을 누르면
선수별 기여도 표와 그림이 results_ipi/ 폴더에 나온다.

계산은 soccer_ipi_v8.py 가 한다. 이 파일은 그것을 창에서 쓰게 해 주는 껍데기다.
"""

from __future__ import annotations

import io
import json
import os
import queue
import subprocess
import sys
import threading
import traceback
import types
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "soccer_ipi_v8.py"
RESULTS_DIR = ROOT / "results_ipi"
CONFIG_PATH = ROOT / "ipi_config.json"
LOGS_DIR = ROOT / "logs"

BG = "#12151a"
PANEL = "#1a1f27"
FG = "#e6e9ef"
MUTED = "#8b95a5"
ACCENT = "#4c8dff"
OKC = "#5fd08a"
WARN = "#ffb35c"

DATA_EXT = ("*.csv *.xlsx *.xls *.txt *.tsv")

DEFAULTS = {
    "pitch_l": 105.0, "pitch_w": 68.0, "team_size": 11,
    "fps": "", "norm": "z",
    "team_a": "", "team_b": "",
    "homography_src": None, "homography_dst": None,
    "exclude_gk": True, "pos_adjust": True, "pa_learn": True,
    "save_plot": True, "drop_referee": True,
    "spec_pv_prox": False, "spec_prog_goaldist": False,
    "results_dir": str(RESULTS_DIR),
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in cfg:
                        cfg[k] = v
        except (json.JSONDecodeError, OSError):
            pass            # 망가진 설정 하나로 프로그램이 안 열리면 안 된다
    return cfg


def save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    except OSError:
        pass


def load_engine() -> types.ModuleType:
    """
    계산 엔진을 '새 것'으로 하나 만들어 돌려준다.

    엔진은 설정을 모듈 전역으로 들고 있어서, 한 번 돌린 뒤 그대로 다시 돌리면
    앞 실행에서 정해진 값(fps, 소유 반경 등)이 남아 다음 결과를 오염시킨다.
    파일마다 깨끗한 사본을 새로 만들어 쓰면 그 문제가 없다.
    """
    if not ENGINE.exists():
        raise FileNotFoundError(f"계산 엔진을 찾을 수 없습니다: {ENGINE}")
    mod = types.ModuleType("ipi_engine")
    mod.__file__ = str(ENGINE)
    mod.__name__ = "ipi_engine"          # __main__ 이 아니므로 자동 실행되지 않는다
    code = compile(ENGINE.read_text(encoding="utf-8"), str(ENGINE), "exec")
    exec(code, mod.__dict__)
    return mod


class QueueWriter(io.TextIOBase):
    """엔진의 print 출력을 창의 로그로 흘려보낸다."""

    def __init__(self, q: queue.Queue, tee=None):
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
        for d in (Path(self.cfg["results_dir"]), LOGS_DIR):
            try:
                Path(d).mkdir(parents=True, exist_ok=True)
            except OSError:
                pass

        self._build_ui()
        self._enable_dnd()
        for arg in sys.argv[1:]:
            self.add_paths([Path(arg)])
        self.log("준비됐습니다. 선수 좌표 파일을 창에 끌어다 놓거나 [파일 추가] 를 누르세요.")
        self.log("추적 프로그램이 만든 results/<영상이름>/tracks.csv 를 그대로 넣으면 됩니다.")
        if not self.cfg.get("homography_src"):
            self.log("! 좌표가 화면 픽셀이면 [경기장 보정] 을 먼저 해야 합니다. "
                     "이미 미터 좌표라면 그냥 시작하세요.")
        self.root.after(100, self._drain)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        r = self.root
        r.title("축구 기여도 분석기 — SC · PR · PA · IPI")
        r.geometry("1010x790")
        r.minsize(940, 700)
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
                        font=("Malgun Gothic", 15, "bold"))
        style.configure("TCheckbutton", background=BG, foreground=FG)
        style.map("TCheckbutton", background=[("active", BG)])
        style.configure("TButton", padding=6)
        style.configure("Go.TButton", padding=8, font=("Malgun Gothic", 10, "bold"))
        style.configure("TProgressbar", troughcolor=PANEL, background=ACCENT,
                        borderwidth=0, thickness=8)
        style.configure("TEntry", fieldbackground=PANEL, foreground=FG)
        style.configure("TCombobox", fieldbackground=PANEL, foreground=FG)
        # clam 테마의 읽기전용 콤보박스는 글자가 배경에 묻힌다. 직접 지정한다.
        style.map("TCombobox",
                  fieldbackground=[("readonly", PANEL)],
                  foreground=[("readonly", FG)],
                  selectbackground=[("readonly", PANEL)],
                  selectforeground=[("readonly", FG)])

        pad = {"padx": 14}

        head = ttk.Frame(r)
        head.pack(fill="x", pady=(14, 4), **pad)
        ttk.Label(head, text="축구 기여도 분석기", style="Head.TLabel").pack(side="left")
        ttk.Label(head, text="선수 좌표에서 공간 통제·압박·패스 유인을 계산해 IPI 로 냅니다",
                  style="Muted.TLabel").pack(side="left", padx=(12, 0))

        # --- 경기 설정 ---------------------------------------------------
        o1 = ttk.Frame(r)
        o1.pack(fill="x", pady=(8, 2), **pad)
        ttk.Label(o1, text="경기장").grid(row=0, column=0, sticky="w")
        self.pl_var = tk.StringVar(value=str(self.cfg["pitch_l"]))
        self.pw_var = tk.StringVar(value=str(self.cfg["pitch_w"]))
        ttk.Entry(o1, textvariable=self.pl_var, width=6).grid(row=0, column=1, padx=(6, 2))
        ttk.Label(o1, text="x").grid(row=0, column=2)
        ttk.Entry(o1, textvariable=self.pw_var, width=6).grid(row=0, column=3, padx=(2, 4))
        ttk.Label(o1, text="m", style="Muted.TLabel").grid(row=0, column=4, padx=(0, 16))

        ttk.Label(o1, text="한 팀 인원").grid(row=0, column=5, sticky="w")
        self.ts_var = tk.StringVar(value=str(self.cfg["team_size"]))
        ttk.Spinbox(o1, textvariable=self.ts_var, width=5, from_=3, to=11
                    ).grid(row=0, column=6, padx=(6, 16))

        ttk.Label(o1, text="fps").grid(row=0, column=7, sticky="w")
        self.fps_var = tk.StringVar(value=str(self.cfg["fps"]))
        ttk.Entry(o1, textvariable=self.fps_var, width=6).grid(row=0, column=8, padx=(6, 2))
        ttk.Label(o1, text="비우면 자동", style="Muted.TLabel").grid(row=0, column=9, sticky="w")

        o2 = ttk.Frame(r)
        o2.pack(fill="x", pady=2, **pad)
        ttk.Label(o2, text="팀 이름").grid(row=0, column=0, sticky="w")
        self.ta_var = tk.StringVar(value=self.cfg["team_a"])
        self.tb_var = tk.StringVar(value=self.cfg["team_b"])
        ttk.Entry(o2, textvariable=self.ta_var, width=10).grid(row=0, column=1, padx=(6, 2))
        ttk.Label(o2, text="/").grid(row=0, column=2)
        ttk.Entry(o2, textvariable=self.tb_var, width=10).grid(row=0, column=3, padx=(2, 4))
        ttk.Label(o2, text="비우면 파일의 값 그대로 (tracker 는 0 / 1)",
                  style="Muted.TLabel").grid(row=0, column=4, sticky="w", padx=(0, 18))

        ttk.Label(o2, text="정규화").grid(row=0, column=5, sticky="w")
        self.norm_var = tk.StringVar(value=self.cfg["norm"])
        ttk.Combobox(o2, textvariable=self.norm_var, width=7, state="readonly",
                     values=["z", "rank"]).grid(row=0, column=6, padx=(6, 6))
        ttk.Label(o2, text="z = 방법론 문서 값 (수식 14)",
                  style="Muted.TLabel").grid(row=0, column=7, sticky="w")

        # --- 경기장 보정 -------------------------------------------------
        o3 = ttk.Frame(r)
        o3.pack(fill="x", pady=(8, 2), **pad)
        ttk.Button(o3, text="경기장 보정  (영상에서 기준점 찍기)",
                   command=self.calibrate).pack(side="left")
        self.calib_lbl = tk.Label(o3, bg=BG, fg=MUTED, text="")
        self.calib_lbl.pack(side="left", padx=10)
        ttk.Button(o3, text="보정 지우기", command=self.clear_calib).pack(side="right")
        self._show_calib()

        # --- 체크박스 ----------------------------------------------------
        o4 = ttk.Frame(r)
        o4.pack(fill="x", pady=(6, 8), **pad)
        self.gk_var = tk.BooleanVar(value=self.cfg["exclude_gk"])
        self.pos_var = tk.BooleanVar(value=self.cfg["pos_adjust"])
        self.learn_var = tk.BooleanVar(value=self.cfg["pa_learn"])
        self.plot_var = tk.BooleanVar(value=self.cfg["save_plot"])
        self.ref_var = tk.BooleanVar(value=self.cfg["drop_referee"])
        self.sw1_var = tk.BooleanVar(value=self.cfg["spec_pv_prox"])
        self.sw2_var = tk.BooleanVar(value=self.cfg["spec_prog_goaldist"])
        for text, var in (("골키퍼 제외", self.gk_var),
                          ("포지션 보정", self.pos_var),
                          ("PA 지수 학습", self.learn_var),
                          ("그림 저장", self.plot_var),
                          ("심판 제외", self.ref_var)):
            ttk.Checkbutton(o4, text=text, variable=var).pack(side="left", padx=(0, 12))
        ttk.Label(o4, text="|  문서와 다르게:", style="Muted.TLabel").pack(side="left", padx=(6, 8))
        ttk.Checkbutton(o4, text="압박 속도항에 거리 반영", variable=self.sw1_var
                        ).pack(side="left", padx=(0, 10))
        ttk.Checkbutton(o4, text="전진가치를 골대거리로", variable=self.sw2_var
                        ).pack(side="left")

        # --- 대기열 ------------------------------------------------------
        qf = ttk.Frame(r)
        qf.pack(fill="x", pady=(4, 6), **pad)
        bar = ttk.Frame(qf)
        bar.pack(fill="x")
        ttk.Label(bar, text="분석할 좌표 파일").pack(side="left")
        ttk.Button(bar, text="파일 추가", command=self.pick_files).pack(side="right")
        ttk.Button(bar, text="비우기", command=self.clear_queue).pack(side="right", padx=6)
        self.listbox = tk.Listbox(qf, height=5, bg=PANEL, fg=FG, borderwidth=0,
                                  highlightthickness=1, highlightbackground="#2a313d",
                                  selectbackground=ACCENT, activestyle="none")
        self.listbox.pack(fill="x", pady=(4, 0))

        # --- 실행 --------------------------------------------------------
        run = ttk.Frame(r)
        run.pack(fill="x", pady=8, **pad)
        self.start_btn = ttk.Button(run, text="▶  분석 시작", style="Go.TButton",
                                    command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(run, text="■  중지", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)
        ttk.Button(run, text="결과 폴더 열기",
                   command=lambda: self.open_folder(Path(self.cfg["results_dir"]))
                   ).pack(side="right")

        prog = ttk.Frame(r)
        prog.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(prog, mode="determinate", maximum=100)
        self.progress.pack(fill="x")
        self.status = ttk.Label(prog, text="대기 중", style="Muted.TLabel")
        self.status.pack(anchor="w", pady=(4, 0))

        # --- 로그 --------------------------------------------------------
        lf = ttk.Frame(r)
        lf.pack(fill="both", expand=True, pady=(8, 14), **pad)
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

    def _enable_dnd(self):
        try:
            self.root.drop_target_register("DND_Files")      # type: ignore[attr-defined]
            self.root.dnd_bind("<<Drop>>", self._on_drop)    # type: ignore[attr-defined]
        except (AttributeError, tk.TclError):
            pass

    def _on_drop(self, event):
        self.add_paths([Path(p) for p in self.root.tk.splitlist(event.data)])

    # -------------------------------------------------------------- 대기열
    def add_paths(self, paths):
        added = 0
        for p in paths:
            found = []
            if p.is_dir():
                for ext in ("*.csv", "*.xlsx", "*.xls"):
                    found += sorted(p.rglob(ext))
            elif p.is_file() and p.suffix.lower() in (".csv", ".xlsx", ".xls", ".txt", ".tsv"):
                found = [p]
            for f in found:
                if f not in self.pending:
                    self.pending.append(f)
                    self.listbox.insert("end", f"  {f.name}      ({f.parent})")
                    added += 1
        if added:
            self.log(f"{added}개 파일 추가 (대기 {len(self.pending)}개)")
        return added

    def pick_files(self):
        files = filedialog.askopenfilenames(
            title="선수 좌표 파일 선택 (tracks.csv 등)",
            filetypes=[("좌표 파일", DATA_EXT), ("모든 파일", "*.*")])
        if files:
            self.add_paths([Path(f) for f in files])

    def clear_queue(self):
        if self._busy():
            messagebox.showinfo("분석 중", "분석이 끝난 뒤에 비울 수 있습니다.")
            return
        self.pending.clear()
        self.listbox.delete(0, "end")

    # ------------------------------------------------------------- 보정
    def calibrate(self):
        try:
            from calibrate import CalibrateWindow
        except ImportError as exc:
            messagebox.showerror("보정 창을 열 수 없음", f"calibrate.py 를 찾지 못했습니다.\n{exc}")
            return
        try:
            pl, pw = float(self.pl_var.get()), float(self.pw_var.get())
        except ValueError:
            messagebox.showwarning("경기장 크기", "경기장 크기를 숫자로 넣어 주세요.")
            return

        def done(src, dst):
            self.cfg["homography_src"] = [list(map(float, p)) for p in src]
            self.cfg["homography_dst"] = [list(map(float, p)) for p in dst]
            save_config(self.cfg)
            self._show_calib()
            self.log(f"경기장 보정 완료 — 기준점 {len(src)}개를 저장했습니다.", "ok")

        CalibrateWindow(self.root, pl, pw, on_done=done)

    def clear_calib(self):
        self.cfg["homography_src"] = self.cfg["homography_dst"] = None
        save_config(self.cfg)
        self._show_calib()
        self.log("경기장 보정을 지웠습니다. 좌표를 이미 미터로 가진 파일에만 쓰세요.")

    def _show_calib(self):
        src = self.cfg.get("homography_src")
        if src:
            self.calib_lbl.configure(text=f"보정됨 — 기준점 {len(src)}개 (픽셀 → 미터 변환을 합니다)",
                                     fg=OKC)
        else:
            self.calib_lbl.configure(text="보정 안 됨 — 파일의 좌표를 이미 미터로 보고 계산합니다",
                                     fg=MUTED)

    # ------------------------------------------------------------- 실행
    def _busy(self):
        return self.worker is not None and self.worker.is_alive()

    def collect(self) -> dict:
        def f(var, default):
            try:
                return float(var.get())
            except ValueError:
                return default
        self.cfg.update({
            "pitch_l": f(self.pl_var, 105.0), "pitch_w": f(self.pw_var, 68.0),
            "team_size": int(f(self.ts_var, 11)),
            "fps": self.fps_var.get().strip(), "norm": self.norm_var.get(),
            "team_a": self.ta_var.get().strip(), "team_b": self.tb_var.get().strip(),
            "exclude_gk": self.gk_var.get(), "pos_adjust": self.pos_var.get(),
            "pa_learn": self.learn_var.get(), "save_plot": self.plot_var.get(),
            "drop_referee": self.ref_var.get(),
            "spec_pv_prox": self.sw1_var.get(),
            "spec_prog_goaldist": self.sw2_var.get(),
        })
        save_config(self.cfg)
        return self.cfg

    def start(self):
        if self._busy():
            return
        if not self.pending:
            messagebox.showinfo("파일이 없습니다",
                                "분석할 좌표 파일을 [파일 추가] 로 고르거나 창에 끌어다 놓으세요.\n\n"
                                "추적 프로그램의 results/<영상이름>/tracks.csv 를 넣으면 됩니다.")
            return
        cfg = self.collect()
        files = list(self.pending)
        self.stop_flag.clear()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.progress.configure(value=0)
        self.log(f"── 분석 시작 · {len(files)}개 파일 "
                 f"· 경기장 {cfg['pitch_l']:g}x{cfg['pitch_w']:g} m "
                 f"· {cfg['team_size']}명 ──", "head")
        self.worker = threading.Thread(target=self._run, args=(files, dict(cfg)), daemon=True)
        self.worker.start()

    def stop(self):
        self.stop_flag.set()
        self.status.configure(text="중지 요청 — 지금 단계가 끝나면 멈춥니다")

    def _run(self, files, cfg):
        done = 0
        out_root = Path(cfg["results_dir"])
        for path in files:
            if self.stop_flag.is_set():
                break
            self.q.put(("log", f"\n[{path.name}] 분석 중...", "head"))
            self.q.put(("stage", "준비", 0, 1))
            out_dir = out_root / path.stem
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                self.q.put(("log", f"  결과 폴더를 만들 수 없습니다: {exc}", "warn"))
                continue
            log_path = out_dir / "log.txt"
            try:
                tee = open(log_path, "w", encoding="utf-8")
            except OSError:
                tee = None
            writer = QueueWriter(self.q, tee)
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = sys.stderr = writer
            try:
                eng = load_engine()
                self._apply(eng, cfg, path, out_dir)
                eng.main()
                self.q.put(("log", f"  -> {out_dir}", "ok"))
                done += 1
            except SystemExit as exc:
                # 엔진은 데이터가 쓸 수 없을 때 이유를 적어 SystemExit 을 낸다.
                writer.flush()
                self.q.put(("log", f"  멈춤: {exc}", "warn"))
            except Exception as exc:                      # noqa: BLE001
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
        self.q.put(("done", done, len(files)))

    def _apply(self, eng, cfg, path, out_dir):
        """창에서 정한 값을 엔진 전역에 꽂는다."""
        eng.EXCEL_PATH = str(path)
        eng.SAVE_XLSX = str(out_dir / "ipi_result.xlsx")
        eng.SAVE_PLOT = str(out_dir / "ipi_plot.png")
        eng.PITCH_L, eng.PITCH_W = float(cfg["pitch_l"]), float(cfg["pitch_w"])
        eng.apply_pitch_scale()                 # 수식 19 를 새 크기로 다시 적용
        eng.TEAM_SIZE = int(cfg["team_size"])
        if str(cfg["fps"]).strip():
            try:
                eng.FPS = float(cfg["fps"])
            except ValueError:
                pass
        eng.NORM_METHOD = cfg["norm"]
        eng.EXCLUDE_GK = bool(cfg["exclude_gk"])
        eng.POS_ADJUST = bool(cfg["pos_adjust"])
        eng.PA_LEARN = bool(cfg["pa_learn"])
        eng.DROP_REFEREE = bool(cfg["drop_referee"])
        eng.SPEC_PR_PV_PROXIMITY = bool(cfg["spec_pv_prox"])
        eng.SPEC_PA_PROG_GOALDIST = bool(cfg["spec_prog_goaldist"])
        if cfg.get("team_a") and cfg.get("team_b"):
            eng.TEAM_NAMES = {"0": cfg["team_a"], "1": cfg["team_b"],
                              cfg["team_a"]: cfg["team_a"], cfg["team_b"]: cfg["team_b"]}
        if cfg.get("homography_src") and cfg.get("homography_dst"):
            eng.HOMOGRAPHY_SRC = [tuple(p) for p in cfg["homography_src"]]
            eng.HOMOGRAPHY_DST = [tuple(p) for p in cfg["homography_dst"]]
        if not cfg["save_plot"]:
            eng.plot = lambda *a, **k: None
        eng.PROGRESS_CB = lambda stage, cur, total: self.q.put(("stage", stage, cur, total))
        eng.STOP_CB = self.stop_flag.is_set

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
                        self.status.configure(text=f"{stage}  {cur:,} / {total:,}  ({pct:.0f}%)")
                    else:
                        self.status.configure(text=f"{stage} ...")
                elif kind == "done":
                    self._finished(msg[1], msg[2])
        except queue.Empty:
            pass
        self.root.after(100, self._drain)

    def _finished(self, done, total):
        gone = {p.name for p in self.pending}
        if done:
            for i in range(self.listbox.size() - 1, -1, -1):
                self.listbox.delete(i)
            self.pending.clear()
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.progress.configure(value=100 if done else 0)
        self.status.configure(text=f"완료 — {done}/{total}개 분석됨" if done else "대기 중")
        self.log(f"── 끝났습니다 ({done}/{total}개) ──", "head" if done else "warn")
        if done:
            self.log(f"   결과: {self.cfg['results_dir']}  "
                     f"(ipi_result.xlsx · ipi_plot.png · log.txt)", "ok")

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


def _attach_streams():
    """pythonw.exe 로 띄우면 stdout 이 None 이라 print 가 터진다. 파일로 돌려놓는다."""
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
