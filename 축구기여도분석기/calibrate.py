"""
경기장 보정 창 — 영상에서 기준점 4개를 찍어 픽셀을 미터로 바꾼다.

추적 프로그램(tracker.py)이 내는 tracks.csv 의 좌표는 '화면 픽셀'이다.
카메라가 비스듬히 찍기 때문에 화면 위쪽 100픽셀과 아래쪽 100픽셀은
실제 거리가 다르다. 기여도 수식의 상수는 전부 미터 기준이라
이 변환을 안 하면 계산이 성립하지 않는다 (방법론 문서 §7 한계 1).

쓰는 법
  1. 분석한 영상을 고른다
  2. 화면에 보이는 경기장 지점을 클릭한다 (코너, 페널티박스 모서리 등)
  3. 클릭할 때마다 그 점이 경기장의 어디인지 목록에서 고른다
  4. 4개 이상 찍고 [적용]

찍을 때 요령
  · 네 점이 한 직선 위에 있으면 안 된다
  · 화면에 넓게 퍼질수록 정확하다 (네 모서리가 가장 좋다)
  · 선이 교차하는 지점처럼 눈으로 정확히 집을 수 있는 곳을 고른다
"""
from __future__ import annotations

import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

BG = "#12151a"
PANEL = "#1a1f27"
FG = "#e6e9ef"
MUTED = "#8b95a5"
ACCENT = "#4c8dff"
MARK = "#ffd24c"

VIDEO_EXT = ("*.mp4 *.avi *.mov *.mkv *.m4v *.webm "
             "*.png *.jpg *.jpeg *.bmp")


def landmarks(pitch_l: float, pitch_w: float) -> list[tuple[str, float, float]]:
    """
    경기장 위에서 눈으로 정확히 집을 수 있는 지점들. (이름, X, Y) 미터.

    좌표계는 왼쪽 아래가 (0, 0), 오른쪽 위가 (가로, 세로)다.
    페널티박스 16.5 m, 골에어리어 5.5 m, 센터서클 9.15 m 는 경기 규칙 값이고
    풀사이즈(105x68)가 아니면 구장 비율에 맞춰 줄여 준다.
    """
    k = min(pitch_l / 105.0, pitch_w / 68.0)
    pb, ga, cc = 16.5 * k, 5.5 * k, 9.15 * k
    pb_h, ga_h = 20.16 * k, 9.16 * k
    cy = pitch_w / 2
    return [
        ("왼쪽 아래 코너", 0.0, 0.0),
        ("오른쪽 아래 코너", pitch_l, 0.0),
        ("오른쪽 위 코너", pitch_l, pitch_w),
        ("왼쪽 위 코너", 0.0, pitch_w),
        ("하프라인 × 아래 터치라인", pitch_l / 2, 0.0),
        ("하프라인 × 위 터치라인", pitch_l / 2, pitch_w),
        ("왼쪽 페널티박스 아래 모서리", pb, cy - pb_h),
        ("왼쪽 페널티박스 위 모서리", pb, cy + pb_h),
        ("오른쪽 페널티박스 아래 모서리", pitch_l - pb, cy - pb_h),
        ("오른쪽 페널티박스 위 모서리", pitch_l - pb, cy + pb_h),
        ("왼쪽 골에어리어 아래 모서리", ga, cy - ga_h),
        ("왼쪽 골에어리어 위 모서리", ga, cy + ga_h),
        ("오른쪽 골에어리어 아래 모서리", pitch_l - ga, cy - ga_h),
        ("오른쪽 골에어리어 위 모서리", pitch_l - ga, cy + ga_h),
        ("센터서클 왼쪽 끝", pitch_l / 2 - cc, cy),
        ("센터서클 오른쪽 끝", pitch_l / 2 + cc, cy),
        ("왼쪽 골대 아래 기둥", 0.0, cy - 3.66 * k),
        ("왼쪽 골대 위 기둥", 0.0, cy + 3.66 * k),
        ("오른쪽 골대 아래 기둥", pitch_l, cy - 3.66 * k),
        ("오른쪽 골대 위 기둥", pitch_l, cy + 3.66 * k),
    ]


def check_points(src, dst):
    """
    찍은 점들로 변환이 제대로 구해지는지 확인하고 '되돌림 오차'(m)를 낸다.

    구한 변환으로 화면 점들을 다시 미터로 옮겨서, 사용자가 말한 실제 위치와
    얼마나 벗어나는지 잰다. 네 점이 한 직선에 가깝게 몰려 있으면 수식은 풀리지만
    엉뚱한 변환이 나오는데, 그런 경우가 여기서 큰 오차로 드러난다.

    변환 자체를 못 구하면 None.
    """
    import numpy as np
    try:
        A = []
        for (x, y), (u, v) in zip(src, dst):
            A += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u],
                  [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
        _, _, Vt = np.linalg.svd(np.asarray(A, float))
        H = Vt[-1].reshape(3, 3)
        if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
            return None
        H = H / H[2, 2]
        p = np.asarray(src, float).T
        q = H @ np.vstack([p, np.ones(p.shape[1])])
        w = q[2]
        if not np.all(np.abs(w) > 1e-9):
            return None
        back = np.vstack([q[0] / w, q[1] / w]).T
        return float(np.max(np.linalg.norm(back - np.asarray(dst, float), axis=1)))
    except Exception:       # noqa: BLE001 — 어떤 이유로든 못 구하면 못 구한 것이다
        return None


class CalibrateWindow(tk.Toplevel):
    """영상 한 장을 띄우고 클릭으로 기준점을 모은다."""

    MAX_W, MAX_H = 1040, 600

    def __init__(self, master, pitch_l: float, pitch_w: float,
                 on_done=None, video_hint: str | None = None):
        super().__init__(master)
        self.title("경기장 보정 — 화면의 기준점을 클릭하세요")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.on_done = on_done
        self.pitch_l, self.pitch_w = pitch_l, pitch_w
        self.marks: list[dict] = []          # {px, py, name, mx, my, canvas ids}
        self.scale = 1.0
        self.photo = None
        self.raw = None
        self.check_photo = None
        self.tmp_png = None
        self.names = [n for n, _, _ in landmarks(pitch_l, pitch_w)]
        self.lut = {n: (x, y) for n, x, y in landmarks(pitch_l, pitch_w)}

        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=12, pady=(12, 6))
        tk.Label(top, text="영상/이미지", bg=BG, fg=FG).pack(side="left")
        self.path_var = tk.StringVar(value=video_hint or "")
        tk.Entry(top, textvariable=self.path_var, width=58, bg=PANEL, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left", padx=6)
        ttk.Button(top, text="찾기", command=self.pick).pack(side="left")
        tk.Label(top, text="시각(초)", bg=BG, fg=FG).pack(side="left", padx=(14, 4))
        self.sec_var = tk.StringVar(value="0")
        tk.Entry(top, textvariable=self.sec_var, width=6, bg=PANEL, fg=FG,
                 insertbackground=FG, relief="flat").pack(side="left")
        ttk.Button(top, text="화면 불러오기", command=self.load_frame).pack(side="left", padx=6)

        self.canvas = tk.Canvas(self, width=self.MAX_W, height=self.MAX_H,
                                bg="#0b0d11", highlightthickness=1,
                                highlightbackground="#2a313d")
        self.canvas.pack(padx=12)
        self.canvas.bind("<Button-1>", self.click)
        self.canvas.create_text(self.MAX_W // 2, self.MAX_H // 2,
                                text="위에서 영상을 고르고 [화면 불러오기] 를 누르세요",
                                fill=MUTED, font=("Malgun Gothic", 11))

        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="x", padx=12, pady=(8, 0))
        tk.Label(mid, text="찍은 점", bg=BG, fg=FG).pack(side="left")
        self.count_lbl = tk.Label(mid, text="0개 (4개 이상 필요)", bg=BG, fg=MUTED)
        self.count_lbl.pack(side="left", padx=8)
        ttk.Button(mid, text="마지막 점 취소", command=self.undo).pack(side="right")
        ttk.Button(mid, text="전부 지우기", command=self.clear).pack(side="right", padx=6)

        self.listbox = tk.Listbox(self, height=5, bg=PANEL, fg=FG, borderwidth=0,
                                  highlightthickness=1, highlightbackground="#2a313d",
                                  activestyle="none")
        self.listbox.pack(fill="x", padx=12, pady=(4, 0))

        bot = tk.Frame(self, bg=BG)
        bot.pack(fill="x", padx=12, pady=12)
        self.hint = tk.Label(
            bot, bg=BG, fg=MUTED, justify="left",
            text="화면에서 경기장 지점을 클릭하면 그 점이 어디인지 고르는 창이 뜹니다. "
                 "네 모서리처럼 넓게 퍼진 점이 가장 정확합니다.")
        self.hint.pack(side="left")
        ttk.Button(bot, text="취소", command=self.destroy).pack(side="right")
        self.apply_btn = ttk.Button(bot, text="적용", command=self.apply, state="disabled")
        self.apply_btn.pack(side="right", padx=6)

        self.transient(master)
        self.grab_set()
        if video_hint and os.path.exists(video_hint):
            self.after(120, self.load_frame)

    # ----------------------------------------------------------- 화면
    def pick(self):
        f = filedialog.askopenfilename(
            title="분석한 영상(또는 한 장면 이미지) 선택",
            filetypes=[("영상/이미지", VIDEO_EXT), ("모든 파일", "*.*")])
        if f:
            self.path_var.set(f)
            self.load_frame()

    def load_frame(self):
        path = self.path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showwarning("파일 없음", "영상이나 이미지를 먼저 고르세요.", parent=self)
            return
        try:
            import cv2
        except ImportError:
            messagebox.showerror(
                "OpenCV 없음",
                "영상을 열려면 opencv 가 필요합니다.\n\n"
                "  pip install opencv-python\n\n"
                "또는 화면을 캡처한 이미지 파일(png/jpg)을 대신 고르세요.", parent=self)
            return

        img = None
        if path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img = cv2.imread(path)
        else:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                try:
                    sec = float(self.sec_var.get() or 0)
                except ValueError:
                    sec = 0.0
                if sec > 0:
                    cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
                ok, img = cap.read()
                if not ok:
                    img = None
            cap.release()
        if img is None:
            messagebox.showerror("실패", "그 시각의 화면을 읽지 못했습니다. "
                                         "시각을 바꾸거나 다른 파일을 고르세요.", parent=self)
            return

        h, w = img.shape[:2]
        self.scale = min(self.MAX_W / w, self.MAX_H / h, 1.0)
        show = cv2.resize(img, (max(1, int(w * self.scale)), max(1, int(h * self.scale))),
                          interpolation=cv2.INTER_AREA)
        self.raw = img                      # 확인 그림에 쓸 원본 화면
        fd, self.tmp_png = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(self.tmp_png, show)
        self.photo = tk.PhotoImage(file=self.tmp_png)   # 참조를 들고 있어야 안 지워진다
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.clear()
        self.hint.configure(text=f"원본 {w}x{h} · 화면에 {self.scale*100:.0f}% 로 표시 중. "
                                 f"경기장 지점을 클릭하세요.")

    # ----------------------------------------------------------- 점 찍기
    def click(self, ev):
        if self.photo is None:
            return
        px, py = ev.x / self.scale, ev.y / self.scale     # 원본 픽셀로 되돌린다
        name = self.ask_landmark()
        if not name:
            return
        mx, my = self.lut[name]
        r = 5
        ids = [self.canvas.create_oval(ev.x - r, ev.y - r, ev.x + r, ev.y + r,
                                       outline=MARK, width=2),
               self.canvas.create_text(ev.x + 9, ev.y - 9, text=str(len(self.marks) + 1),
                                       fill=MARK, anchor="w",
                                       font=("Malgun Gothic", 9, "bold"))]
        self.marks.append({"px": px, "py": py, "name": name, "mx": mx, "my": my,
                           "ids": ids})
        self.refresh()

    def ask_landmark(self) -> str | None:
        """방금 찍은 점이 경기장의 어디인지 고르게 한다."""
        dlg = tk.Toplevel(self)
        dlg.title("이 점은 어디인가요?")
        dlg.configure(bg=BG)
        dlg.resizable(False, False)
        tk.Label(dlg, text="방금 클릭한 지점을 고르세요", bg=BG, fg=FG).pack(padx=16, pady=(14, 6))
        used = {m["name"] for m in self.marks}
        left = [n for n in self.names if n not in used]
        var = tk.StringVar(value=left[0] if left else "")
        box = ttk.Combobox(dlg, textvariable=var, values=left, width=34, state="readonly")
        box.pack(padx=16)
        out = {}

        def ok():
            out["name"] = var.get()
            dlg.destroy()

        btns = tk.Frame(dlg, bg=BG)
        btns.pack(pady=12)
        ttk.Button(btns, text="확인", command=ok).pack(side="left", padx=4)
        ttk.Button(btns, text="취소", command=dlg.destroy).pack(side="left", padx=4)
        dlg.bind("<Return>", lambda _e: ok())
        dlg.transient(self)
        dlg.grab_set()
        box.focus_set()
        self.wait_window(dlg)
        return out.get("name")

    def undo(self):
        if not self.marks:
            return
        m = self.marks.pop()
        for i in m["ids"]:
            self.canvas.delete(i)
        self.refresh()

    def clear(self):
        for m in self.marks:
            for i in m["ids"]:
                self.canvas.delete(i)
        self.marks.clear()
        self.refresh()

    def refresh(self):
        self.listbox.delete(0, "end")
        for k, m in enumerate(self.marks, 1):
            self.listbox.insert("end",
                                f"  {k}. {m['name']}   화면({m['px']:.0f}, {m['py']:.0f})px"
                                f"  ->  경기장({m['mx']:.1f}, {m['my']:.1f})m")
        n = len(self.marks)
        self.count_lbl.configure(
            text=f"{n}개" + ("" if n >= 4 else f" (4개 이상 필요 — {4 - n}개 더)"))
        self.apply_btn.configure(state="normal" if n >= 4 else "disabled")

    def confirm_overlay(self, src, dst) -> bool:
        """
        찍은 점으로 구한 변환이 맞는지 눈으로 보여 주고 확인받는다.

        되돌림 오차만으로는 부족하다. 네 점 자신은 잘 맞아도 경기장 전체가
        어긋나 있을 수 있다 (점 네 개는 그 네 점만 보장한다). 실제 경기장
        선을 화면에 되돌려 그려서 흰 선 위에 얹히는지 보면 바로 안다.
        """
        if self.raw is None:
            return True
        try:
            import cv2                      # 이 파일은 cv2 를 쓰는 곳에서만 불러온다
            import numpy as np
            import autocalib
        except ImportError:
            return True
        H = autocalib.homography(src, dst)
        if H is None:
            return True
        img = self.raw.copy()
        tpl = autocalib.pitch_lines(self.pitch_l, self.pitch_w)
        try:
            q = np.linalg.inv(np.asarray(H, float)) @ np.stack(
                [tpl[:, 0], tpl[:, 1], np.ones(len(tpl))])
        except np.linalg.LinAlgError:
            return True
        ok = np.abs(q[2]) > 1e-9
        x, y = q[0][ok] / q[2][ok], q[1][ok] / q[2][ok]
        h, w = img.shape[:2]
        for xx, yy in zip(x, y):
            if -20 < xx < w + 20 and -20 < yy < h + 20:
                cv2.circle(img, (int(xx), int(yy)), 2, (0, 255, 0), -1)
        for px, py in src:
            cv2.circle(img, (int(px), int(py)), 12, (0, 255, 255), 3)
        sc = min(self.MAX_W / w, self.MAX_H / h, 1.0)
        show = cv2.resize(img, (max(1, int(w * sc)), max(1, int(h * sc))),
                          interpolation=cv2.INTER_AREA)
        fd, png = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        cv2.imwrite(png, show)

        win = tk.Toplevel(self)
        win.title("이렇게 보입니다 — 초록 선이 흰 선 위에 얹혀 있습니까?")
        win.transient(self)
        self.check_photo = tk.PhotoImage(file=png)
        tk.Label(win, image=self.check_photo).pack()
        tk.Label(win, text="초록 = 찍은 점으로 구한 경기장 선 · 노랑 = 찍은 점\n"
                           "초록이 화면의 흰 선 위에 얹혀 있어야 맞는 보정입니다.",
                 justify="left").pack(padx=10, pady=(6, 0))
        res = {"go": False}

        def yes():
            res["go"] = True
            win.destroy()

        row = tk.Frame(win)
        row.pack(pady=8)
        tk.Button(row, text="맞습니다 — 이걸로 쓰겠습니다", width=26,
                  command=yes).pack(side="left", padx=6)
        tk.Button(row, text="아니요 — 다시 찍겠습니다", width=22,
                  command=win.destroy).pack(side="left", padx=6)
        win.grab_set()
        self.wait_window(win)
        try:
            os.unlink(png)
        except OSError:
            pass
        return res["go"]

    def apply(self):
        src = [(m["px"], m["py"]) for m in self.marks]
        dst = [(m["mx"], m["my"]) for m in self.marks]
        err = check_points(src, dst)
        if err is None:
            messagebox.showerror(
                "보정 실패",
                "이 점들로는 변환을 구할 수 없습니다.\n\n"
                "네 점이 한 직선 위에 몰려 있지 않은지 확인하세요. "
                "예를 들어 하프라인 위의 점만 네 개 찍으면 안 됩니다.", parent=self)
            return
        if err > 1.0:
            # 되돌려 보니 원래 자리에서 많이 벗어난다 = 점이 한 줄에 몰렸거나
            # 엉뚱한 곳을 찍었거나, 고른 지점 이름이 실제와 다르다.
            go = messagebox.askyesno(
                "기준점이 잘 안 맞습니다",
                f"찍은 점을 변환해 되돌려 보니 실제 위치에서 최대 {err:.1f} m 벗어납니다.\n\n"
                "보통 0.5 m 안쪽이어야 합니다. 원인은 대개 셋 중 하나입니다.\n"
                "  · 네 점이 한 직선에 가깝게 몰려 있다\n"
                "  · 클릭한 자리가 실제 지점과 다르다\n"
                "  · 고른 지점 이름이 화면의 그 자리와 다르다\n\n"
                "그래도 이대로 쓸까요?", parent=self)
            if not go:
                return
        if not self.confirm_overlay(src, dst):
            return
        if self.on_done:
            self.on_done(src, dst)
        if self.tmp_png and os.path.exists(self.tmp_png):
            try:
                os.unlink(self.tmp_png)
            except OSError:
                pass
        self.destroy()
