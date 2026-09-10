# =====================================================================
#  축구 선수 비가시적 기여도 — SC / PR 계산  (단일 셀 버전)
#  코랩에 통째로 붙여넣고 실행하면 된다. 엑셀(.xlsx) / CSV 둘 다 읽는다.
# =====================================================================

# ─────────────────── 설정 (여기만 고치면 된다) ───────────────────
EXCEL_PATH   = None      # None 이면 코랩 업로드 창이 뜬다. "내파일.xlsx" / "내파일.csv" 직접 지정 가능
SHEET_TRACKS = 0         # 선수 좌표가 있는 시트 (이름 또는 0,1,2... 번호). CSV 면 무시됨
SHEET_BALL   = None      # 공 좌표가 따로 있는 시트. 없으면 None (자동으로 찾음)

FPS          = None      # None 이면 시간 열 간격에서 자동 추정
PITCH_L, PITCH_W = 105.0, 68.0     # 경기장 가로(골대~골대), 세로 (m)
                                   #   프로 풀사이즈 = 105 x 68
                                   #   7v7 유소년   = 보통 60~68 x 40~47
                                   #   풋살         = 40 x 20
                                   #   ★ 반드시 실제 규격으로 바꿀 것. 아래 상수들이 여기 딸려 있다.
SCALE_TO_PITCH = True              # True 면 '경기장 크기에 비례하는' 상수들을 자동으로 줄인다
COORD_ORIGIN = "auto"    # "corner" = 왼쪽아래가 (0,0) / "center" = 경기장 중앙이 (0,0) / "auto"
TEAM_A_ATTACKS_PLUS_X = True       # 첫 번째 팀이 +X 방향으로 공격하는가

REPAIR_IDS   = "auto"    # 프레임마다 ID 가 뒤바뀌는 데이터를 헝가리안 매칭으로 복구
                         # "auto" = 이상하면 자동 실행 / True / False

DELTA        = 3         # 중앙차분 폭 (프레임)
SPEED_MAX    = 11.0      # 이 속도(m/s) 넘으면 이상치로 버림
                         #   프로 경기에서 실측되는 최고속도는 대개 34~37 km/h(9.5~10.3 m/s)이고
                         #   리그 최고 기록도 38 km/h 안쪽이다. 기존 12.0 m/s = 43.2 km/h 는
                         #   100m 세계기록 보유자의 최고속도에 가까운 값이라, 트래킹 ID 혼선으로
                         #   생긴 튐 값이 '정상 속도'로 통과해 PR(접근속도)을 오염시킨다.
                         #   기록 상한에 여유를 둔 11.0 m/s(=39.6 km/h)로 낮춰 잡는다.
POSS_RADIUS  = None      # 공에서 이 거리(m) 안에 있어야 소유자로 인정. None 이면 데이터 보고 자동
HOLD_FRAMES  = None      # 소유팀 전환 인정에 필요한 유지 프레임. None 이면 1.5초
TEAM_SIZE    = 11        # 한 팀 인원 (골키퍼 포함). 7v7 이면 7, 풋살이면 5
MIN_TRACKED  = None      # 양 팀 최소 추적 인원. None 이면 TEAM_SIZE 의 80%
                         #   기존 규칙은 '실제로 추적된 인원의 중앙값 × 80%' 였는데,
                         #   중계/전술 카메라 한 대로 만든 좌표는 화면 밖 선수가 통째로
                         #   빠지기 때문에 그 중앙값 자체가 이미 깎여 있다.
                         #   (첨부 영상도 한 화면에 경기장의 2/3 정도만 들어온다)
                         #   그러면 22명 중 15명만 잡힌 데이터에서도 문턱이 12 -> 6 으로
                         #   같이 내려가서 게이트가 전혀 작동하지 않는다.
                         #   SC 는 105x68 격자 전체를 최근접 수비수로 나누는 계산이라
                         #   뒤쪽 수비수가 화면 밖으로 빠지면 남은 수비수가 수비 진영
                         #   전체의 공간 점유를 대신 받아 SC 가 부풀려진다.
                         #   그래서 문턱은 관측값이 아니라 명목 인원에서 뽑는다.
BALL_MAX_GAP = None      # 공 좌표가 이 프레임 수 이하로 끊기면 선형보간으로 메운다.
                         # None 이면 0.2초에 해당하는 프레임 수로 자동 설정.
                         # 0 이면 보간 안 함. 이보다 긴 구간은 비워 둔다.
PLAYER_MAX_GAP = None    # 선수 좌표도 같은 방식으로 메운다. None 이면 0.3초.
                         # 선수는 공보다 느리고 부드러워서 더 길게 잡아도 안전하다.

BALL_AIRBORNE_Z = 1.5    # 공 높이(m) 열(ball_z)이 있을 때, 이보다 높으면 '공중볼'로 본다.
                         # 크로스·롱볼·헤더 상황에서는 공의 지면 투영 좌표에 가장 가까운
                         # 선수가 실제로는 그 공을 다룰 수 없다(점프/타이밍 문제라 2D 거리로
                         # 안 잡힌다). 영상으로 봐도 박스 안 크로스·세컨볼 다툼이 잦은데,
                         # 지금 코드는 X,Y 만 쓰고 있어 이런 장면에서 소유자 판정이 틀어진다.
                         # ball_z 열이 없으면 이 값은 그냥 무시되고 기존과 동일하게 동작한다.

GRID_STEP    = 1.0       # SC 격자 크기(m). 0.5 로 하면 90분에 2시간+ 걸린다
LAMBDA_GOAL  = 18.0      # 위험도: 골대 감쇠 거리
LAMBDA_BALL  = 20.0      # 위험도: 공 감쇠 거리
PR_R         = 10.0      # PR 거리 성분 유효 반경
PR_VMAX      = 5.0       # PR 속도 성분 정규화 기준
PR_WD, PR_WV = 0.6, 0.4  # PR 내부 가중치 (거리 : 접근속도). 검증 전 초기 가설
W_SC, W_PR   = 0.5, 0.5  # DPI 합성 가중치. 선험적 근거가 없으므로 동일하게 둔다

W_LANE       = 3.0       # PA 길목: 패스선에서 이만큼 떨어져야 완전히 열린 것으로 본다 (m)
                         #   기준은 '공이 지나가는 동안 수비수가 닿을 수 있는 폭'이다.
                         #   15 m 패스가 12~15 m/s 로 날아가면 1.0~1.2초가 걸리고, 그 사이
                         #   수비수는 옆으로 두어 걸음(2 m 남짓) 움직이고 팔다리로 1 m 를
                         #   더 덮는다. 기존 2.0 m 는 이 값보다 좁아서 길목이 너무 쉽게
                         #   '완전히 열림'이 됐다 (모사 경기에서 51% 가 만점으로 포화).
PA_SPACE_R   = 8.0       # PA 수신공간: 마크와 이만큼 떨어지면 만점 (m)
                         #   같은 논리다. 5 m 는 '수비수가 볼 도착 전에 좁히고 들어올 수
                         #   있는 거리'라 자유롭다고 볼 수 없는데도 만점을 줬고, 그 결과
                         #   86% 의 행이 1.0 으로 포화돼 이 차원이 사실상 아무 정보도
                         #   나르지 못했다 (표준편차 0.16). 포화율은 아래 [지표 진단]에
                         #   매번 찍히므로 실제 데이터로 다시 조정할 것.
PA_PI_MIN    = 0.3       # PA 전진가치 하한 (후방 패스도 0은 아니다)
PA_FWD_REF   = 30.0      # 전진 이득 정규화 기준 (m)
PA_LEARN     = True      # 실제 패스에서 지수 a,b,c 를 학습할지. False 면 전부 1 (= l*phi*pi)

MAX_FRAMES   = None      # 테스트할 때 앞쪽 N 프레임만. 전체면 None
SAVE_XLSX    = "sc_pr_result.xlsx"
# ────────────────────────────────────────────────────────────────

import io, os, sys, time, warnings
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore")

# ── 경기장 크기에 따른 상수 자동 조정 ────────────────────────────────
#  상수는 두 종류다.
#    (A) 경기장 크기에 비례하는 것 — 전술적 '지리'에 대한 값이라 작은 구장에서는 같이 줄어야 한다
#          LAMBDA_GOAL, LAMBDA_BALL, PR_R, PA_FWD_REF
#    (B) 비례하지 않는 것 — 사람 몸과 공의 크기·속도라서 구장이 작아져도 그대로다
#          SPEED_MAX, PR_VMAX, W_LANE, PA_SPACE_R, POSS_RADIUS
#  둘을 같이 줄이면 "작은 구장에서는 사람이 느리게 뛴다"는 말이 되어버린다.
_FULL_L, _FULL_W = 105.0, 68.0
PITCH_SCALE = 1.0
if SCALE_TO_PITCH:
    PITCH_SCALE = float(np.sqrt((PITCH_L / _FULL_L) * (PITCH_W / _FULL_W)))
    if abs(PITCH_SCALE - 1.0) > 0.02:
        LAMBDA_GOAL *= PITCH_SCALE
        LAMBDA_BALL *= PITCH_SCALE
        PR_R        *= PITCH_SCALE
        PA_FWD_REF  *= PITCH_SCALE

IN_COLAB = "google.colab" in sys.modules


# ══════════════ 1. 파일 읽기 + 열 이름 자동 인식 ══════════════
ALIAS = {
    "frame":    ["frame", "frame_id", "frameno", "frame_number", "프레임", "프레임번호", "f"],
    "time_s":   ["time_s", "time", "timesec", "timesecond", "timeseconds", "timestamp",
                 "sec", "secs", "seconds", "elapsed", "시간", "초", "경과시간", "t"],
    "track_id": ["track_id", "trackid", "id", "player_id", "playerid", "선수", "선수id",
                 "선수번호", "등번호", "player", "pid", "name", "이름"],
    "team":     ["team", "team_id", "side", "팀", "소속", "team_name"],
    "X":        ["x", "xcoord", "x_coord", "x_m", "pitch_x", "x_meter", "posx", "pos_x",
                 "좌표x", "x좌표"],
    "Y":        ["y", "ycoord", "y_coord", "y_m", "pitch_y", "y_meter", "posy", "pos_y",
                 "좌표y", "y좌표"],
    "cls":      ["cls", "class", "label", "type", "object", "종류", "분류"],
    "kind":     ["kind", "category", "role"],
    "ball_x":   ["ball_x", "ballx", "bx", "공x", "공_x"],
    "ball_y":   ["ball_y", "bally", "by", "공y", "공_y"],
    "ball_z":   ["ball_z", "ballz", "bz", "ball_height", "ball_h", "공z", "공높이", "공_z"],
}
import re as _re
def _norm(s):
    """열 이름 비교용 정규화. 'Time(sec)' -> 'timesec' 처럼 기호를 전부 뗀다."""
    return _re.sub(r"[^0-9a-z가-힣]", "", str(s).strip().lower())


def map_columns(df, verbose=True):
    """실제 열 이름 -> 표준 이름 매핑을 만든다."""
    lut = {}
    for std, alts in ALIAS.items():
        for a in alts:
            lut[_norm(a)] = std
    ren, found = {}, {}
    for c in df.columns:
        std = lut.get(_norm(c))
        if std and std not in found:
            ren[c] = std
            found[std] = c
    if "time_s" not in found:                      # 'Time(sec)' 같은 변형 폴백
        for c in df.columns:
            if c not in ren and _norm(c).startswith("time"):
                ren[c] = "time_s"; found["time_s"] = c; break
    if verbose and ren:
        print("  열 인식:", ", ".join(f"{v} <- '{k}'" for k, v in ren.items()))
    return df.rename(columns=ren)


BALL_WORDS = ("ball", "공", "볼")


def _is_ball(series):
    """어느 열에 있든 'ball' 로 표시된 행을 찾는다."""
    s = series.astype(str).str.lower().str.strip()
    m = False
    for w in BALL_WORDS:
        m = m | (s == w) | s.str.fullmatch(rf"{w}\d*", na=False)
    return m


def load_table(path, sheet):
    if str(path).lower().endswith((".csv", ".txt", ".tsv")):
        sep = "\t" if str(path).lower().endswith(".tsv") else ","
        return pd.read_csv(path, sep=sep)
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except ImportError:
        raise SystemExit("엑셀을 읽으려면 openpyxl 이 필요하다:  !pip install openpyxl")


def get_input_path():
    global EXCEL_PATH
    if EXCEL_PATH and os.path.exists(EXCEL_PATH):
        return EXCEL_PATH
    if IN_COLAB:
        from google.colab import files
        print("파일을 선택하세요 (.xlsx 또는 .csv)")
        up = files.upload()
        if not up:
            raise SystemExit("업로드된 파일이 없다.")
        return list(up.keys())[0]
    raise SystemExit(f"파일을 찾을 수 없다: {EXCEL_PATH}")


def repair_ids(tracks, verbose=True):
    """
    프레임마다 ID 가 뒤바뀌는 데이터를 복구한다.

    트래커가 ID 를 유지하지 못하면 같은 이름표가 매 프레임 다른 사람에게 붙는다.
    그러면 '한 선수가 0.1 초에 79 m 이동' 같은 값이 나오고, 속도·접근속도(PR)가
    전부 무의미해진다. 연속 프레임 사이에서 이동거리 합이 최소가 되도록
    헝가리안 매칭(linear_sum_assignment)으로 다시 이어 붙인다.
    """
    frames = np.sort(tracks["frame"].unique())
    out = tracks.copy()
    out["stable_id"] = pd.Series([None] * len(out), index=out.index, dtype=object)

    for team, gteam in tracks.groupby("team", sort=False):
        by_f = {f: g for f, g in gteam.groupby("frame", sort=True)}
        fs = [f for f in frames if f in by_f]
        if not fs:
            continue
        g0 = by_f[fs[0]]
        cur = {i: f"{team}_{k}" for k, i in enumerate(g0["track_id"].to_numpy())}
        out.loc[g0.index, "stable_id"] = [cur[i] for i in g0["track_id"]]
        prev_xy = g0[["X", "Y"]].to_numpy(float)
        prev_lbl = [cur[i] for i in g0["track_id"]]

        for f in fs[1:]:
            g = by_f[f]
            xy = g[["X", "Y"]].to_numpy(float)
            n = min(len(prev_xy), len(xy))
            if n == 0:
                prev_xy, prev_lbl = xy, [f"{team}_x{k}" for k in range(len(xy))]
                out.loc[g.index, "stable_id"] = prev_lbl
                continue
            C = np.linalg.norm(prev_xy[:, None, :] - xy[None, :, :], axis=2)
            r, c = linear_sum_assignment(C)
            lbl = [None] * len(xy)
            for ri, ci in zip(r, c):
                lbl[ci] = prev_lbl[ri]
            for k in range(len(xy)):                    # 매칭 안 된 행은 새 ID
                if lbl[k] is None:
                    lbl[k] = f"{team}_new{f}_{k}"
            out.loc[g.index, "stable_id"] = lbl
            prev_xy, prev_lbl = xy, lbl

    out["track_id"] = out["stable_id"]
    return out.drop(columns=["stable_id"])


def fill_player_gaps(tracks, max_gap=None):
    """
    선수 좌표의 짧은 끊김을 선형보간으로 메운다.

    관측이 안 되면 행을 아예 안 적는 방식이라 결측은 '구멍'으로 남는다.
    그런데 중앙차분은 구멍을 걸치면 속도를 통째로 버리므로, 결측률의
    2~2.5 배만큼 PR 이 사라진다. 선수는 공보다 훨씬 부드럽게 움직여서
    (실측: 8프레임 구멍에서 평균오차 0.05 m, 겹침 구간에 몰려도 0.075 m)
    짧은 구간은 안전하게 메울 수 있다.

    주의: 선수 결측은 가림(occlusion) 때문에 생기고, 가림은 선수들이
    붙어 있을 때 일어난다. 즉 결측은 무작위가 아니라 '중요한 순간'에 몰린다.
    그래서 메운 행은 pos_interp=True 로 표시해 두고, 검증 단계에서는
    빼고 계산할 수 있게 한다. max_gap 을 넘는 구간은 손대지 않는다.
    """
    max_gap = PLAYER_MAX_GAP if max_gap is None else max_gap
    if max_gap <= 0:
        tracks = tracks.copy()
        tracks["pos_interp"] = False
        return tracks

    parts = []
    for tid, g in tracks.groupby("track_id", sort=False):
        g = g.sort_values("frame").drop_duplicates("frame", keep="first")
        full = pd.RangeIndex(int(g.frame.min()), int(g.frame.max()) + 1)
        gi = g.set_index("frame").reindex(full).rename_axis("frame")
        miss = gi["X"].isna()
        if miss.any():
            run = miss.groupby((~miss).cumsum()).transform("sum")
            allow = miss & (run <= max_gap)
            for c in ("X", "Y"):
                filled = gi[c].interpolate(method="index")
                gi[c] = gi[c].where(~allow, filled)
            for c in gi.columns:                      # 문자열 열은 앞값으로 채움
                if c not in ("X", "Y"):
                    gi[c] = gi[c].ffill()
            gi["pos_interp"] = allow
        else:
            gi["pos_interp"] = False
        gi = gi.dropna(subset=["X", "Y"]).reset_index()
        gi["track_id"] = tid
        parts.append(gi)

    out = pd.concat(parts, ignore_index=True)
    if "time_s" in out.columns:
        out["time_s"] = out["frame"] / FPS
    return out.sort_values(["frame", "track_id"]).reset_index(drop=True)


def implied_speed(tracks, fps):
    """ID 를 그대로 믿었을 때 나오는 이동 속도 (데이터 건전성 점검용)."""
    q = tracks.sort_values(["track_id", "frame"])
    step = np.hypot(q.groupby("track_id")["X"].diff(), q.groupby("track_id")["Y"].diff())
    dt = q.groupby("track_id")["frame"].diff() / fps
    v = (step / dt).replace([np.inf, -np.inf], np.nan).dropna()
    return v


def read_data():
    path = get_input_path()
    print(f"\n[읽기] {path}")
    if abs(PITCH_SCALE - 1.0) > 0.02:
        print(f"  경기장 {PITCH_L:.0f}x{PITCH_W:.0f} m (풀사이즈의 {PITCH_SCALE*100:.0f}%) "
              f"-> 거리 상수 조정: λ골대={LAMBDA_GOAL:.1f} λ공={LAMBDA_BALL:.1f} "
              f"PR_R={PR_R:.1f} 전진기준={PA_FWD_REF:.1f}")
        print(f"  (사람·공 크기 상수는 그대로: 최고속도 {SPEED_MAX} 길목폭 {W_LANE} 수신공간 {PA_SPACE_R})")

    if not str(path).lower().endswith((".csv", ".txt", ".tsv")):
        print(f"  시트 목록: {pd.ExcelFile(path).sheet_names}")

    tracks = map_columns(load_table(path, SHEET_TRACKS))
    tracks.columns = [str(c) for c in tracks.columns]

    # ── frame 열 먼저 만들기 (시간만 있는 데이터 대응) ───────
    global FPS, HOLD_FRAMES, MIN_TRACKED, POSS_RADIUS
    if "frame" not in tracks.columns:
        if "time_s" not in tracks.columns:
            raise SystemExit(f"frame 도 시간 열도 없다. 실제 열: {list(tracks.columns)}")
        ts = np.sort(tracks["time_s"].dropna().unique())
        dt = np.diff(ts)
        dt = dt[dt > 0]
        step = float(np.median(dt)) if len(dt) else 0.04
        if FPS is None:
            FPS = float(round(1.0 / step, 3))
        tracks["frame"] = np.round(tracks["time_s"] / step).astype(int)
        print(f"  frame 열이 없어 시간에서 생성 (간격 {step:.3f}s -> fps {FPS})")

    # ── 공 좌표 확보 ─────────────────────────────────────────
    ball, how = None, None
    if SHEET_BALL is not None:
        b = map_columns(load_table(path, SHEET_BALL))
        if "frame" not in b.columns and "time_s" in b.columns:
            b["frame"] = np.round(b["time_s"] * (FPS or 25.0)).astype(int)
        bcols = ["frame", "ball_x", "ball_y"] + (["ball_z"] if "ball_z" in b.columns else [])
        ball = (b[bcols] if {"ball_x", "ball_y"} <= set(b.columns)
                else b[["frame", "X", "Y"]].rename(columns={"X": "ball_x", "Y": "ball_y"}))
        how = "별도 시트"
    elif {"ball_x", "ball_y"} <= set(tracks.columns):
        bcols = ["ball_x", "ball_y"] + (["ball_z"] if "ball_z" in tracks.columns else [])
        ball = tracks.groupby("frame")[bcols].first().reset_index()
        tracks = tracks.drop(columns=bcols)
        how = "ball_x / ball_y 열"
    else:
        # cls, team, track_id, kind 중 아무 열에서나 'ball' 표시를 찾는다
        for col in ("cls", "kind", "team", "track_id"):
            if col in tracks.columns:
                m = _is_ball(tracks[col])
                if m.any():
                    ball = (tracks[m].groupby("frame")[["X", "Y"]].first()
                            .rename(columns={"X": "ball_x", "Y": "ball_y"}).reset_index())
                    tracks = tracks[~m].copy()
                    how = f"'{col}' 열의 ball 표시"
                    break

    if ball is None:
        raise SystemExit(
            "공 좌표를 찾지 못했다. 아래 중 하나가 필요하다:\n"
            "  (a) ball_x, ball_y 열\n"
            "  (b) cls / team / track_id 열에 'ball' 인 행\n"
            "  (c) SHEET_BALL 에 공 시트 이름 지정\n"
            f"실제 열: {list(tracks.columns)}")
    print(f"  공 좌표: {how} 에서 {len(ball)} 프레임분")

    # ── 필수 열 ──────────────────────────────────────────────
    miss = {"frame", "track_id", "X", "Y"} - set(tracks.columns)
    if miss:
        raise SystemExit(f"필수 열이 없다: {miss}\n실제 열: {list(tracks.columns)}\n"
                         "ALIAS 딕셔너리에 실제 열 이름을 추가하면 인식된다.")

    if "cls" in tracks.columns:
        keep = tracks["cls"].astype(str).str.lower().str.contains("person|player|선수", na=True)
        tracks = tracks[keep].copy()

    tracks = tracks.dropna(subset=["frame", "track_id", "X", "Y"])
    tracks["frame"] = tracks["frame"].astype(int)
    ball["frame"] = ball["frame"].astype(int)

    # ── 팀 ───────────────────────────────────────────────────
    if "team" not in tracks.columns:
        raise SystemExit("team 열이 없다. 선수마다 어느 팀인지 있어야 SC/PR 을 계산할 수 있다.")
    tracks["team"] = tracks["team"].astype(str).str.strip()
    tracks = tracks[~_is_ball(tracks["team"])]
    teams = [t for t in tracks["team"].unique()
             if not any(k in t.lower() for k in ("ref", "심판", "none", "nan", "unknown"))]
    if len(teams) != 2:
        raise SystemExit(f"팀이 2개여야 하는데 {teams} 이다. team 열을 정리할 것.")
    tracks = tracks[tracks["team"].isin(teams)].copy()

    # ── 시간 / fps ───────────────────────────────────────────
    if FPS is None:
        if "time_s" in tracks.columns and tracks["time_s"].notna().any():
            d = tracks.sort_values("frame")["time_s"].diff()
            d = d[d > 0]
            FPS = float(round(1.0 / d.median(), 3)) if len(d) else 25.0
        else:
            FPS = 25.0
            print("  ! 시간 열이 없어 fps=25 로 가정했다. 다르면 FPS 를 직접 지정할 것")
    if "time_s" not in tracks.columns or tracks["time_s"].isna().all():
        tracks["time_s"] = tracks["frame"] / FPS
    if HOLD_FRAMES is None:
        HOLD_FRAMES = max(2, int(round(1.5 * FPS)))
    global BALL_MAX_GAP
    if BALL_MAX_GAP is None:
        BALL_MAX_GAP = max(1, int(round(0.2 * FPS)))   # 0.2초까지만 보간
    global PLAYER_MAX_GAP
    if PLAYER_MAX_GAP is None:
        PLAYER_MAX_GAP = max(1, int(round(0.3 * FPS)))

    # ── 좌표계 ───────────────────────────────────────────────
    xmin, xmax = tracks.X.min(), tracks.X.max()
    ymin, ymax = tracks.Y.min(), tracks.Y.max()
    if xmax - xmin > PITCH_L * 3 or ymax - ymin > PITCH_W * 3:
        raise SystemExit(
            f"좌표 범위가 경기장보다 훨씬 크다 (X {xmin:.0f}~{xmax:.0f}, Y {ymin:.0f}~{ymax:.0f}).\n"
            "화면 픽셀 좌표로 보인다. 호모그래피로 미터 좌표로 변환한 뒤에 넣어야 한다.\n"
            "픽셀로 계산하면 화면 위/아래의 1m 가 서로 달라져서 결과가 전부 무의미해진다.")

    centered = (COORD_ORIGIN == "center") or (COORD_ORIGIN == "auto" and (xmin < -1 or ymin < -1))
    if centered:
        for df_ in (tracks, ball):
            df_[["X", "ball_x"][df_ is ball]] += PITCH_L / 2
            df_[["Y", "ball_y"][df_ is ball]] += PITCH_W / 2
        print(f"  좌표계: 중앙 원점으로 판단 -> (+{PITCH_L/2:.1f}, +{PITCH_W/2:.1f}) 평행이동")
    out_ratio = (~tracks.X.between(0, PITCH_L) | ~tracks.Y.between(0, PITCH_W)).mean()
    print(f"  좌표 범위 X {tracks.X.min():.1f}~{tracks.X.max():.1f} m , "
          f"Y {tracks.Y.min():.1f}~{tracks.Y.max():.1f} m  (경기장 밖 {out_ratio*100:.1f}%)")
    sx0 = tracks.X.max() - tracks.X.min()
    sy0 = tracks.Y.max() - tracks.Y.min()
    print(f"  데이터가 실제로 퍼진 범위 {sx0:.0f} x {sy0:.0f} m "
          f"(설정한 경기장 {PITCH_L:.0f} x {PITCH_W:.0f})")
    if out_ratio > 0.05:
        sx = tracks.X.max() - tracks.X.min()
        sy = tracks.Y.max() - tracks.Y.min()
        print(f"  ! 경기장 밖 비율이 높다. 설정한 경기장은 {PITCH_L}x{PITCH_W} m 인데 "
              f"데이터가 실제로 퍼진 범위는 약 {sx:.0f}x{sy:.0f} m 다.")
        print(f"    7v7 처럼 작은 경기장이면 PITCH_L, PITCH_W 를 실제 규격으로 바꿀 것 "
              f"(위험도 가중치와 골대 위치가 여기에 달려 있다).")

    if MAX_FRAMES:
        lim = sorted(tracks.frame.unique())[:MAX_FRAMES]
        tracks = tracks[tracks.frame.isin(lim)]
        ball = ball[ball.frame.isin(lim)]

    n_per_team = tracks.groupby(["frame", "team"]).size().groupby("team").median()
    if MIN_TRACKED is None:
        MIN_TRACKED = max(3, int(np.ceil(TEAM_SIZE * 0.8)))
    print(f"  tracks {len(tracks):,}행 / 프레임 {tracks.frame.nunique():,} / "
          f"선수 {tracks.track_id.nunique()}명 / 팀 {teams} / fps {FPS}")
    print(f"  프레임당 인원 {n_per_team.to_dict()} -> MIN_TRACKED = {MIN_TRACKED} "
          f"(명목 {TEAM_SIZE}명의 80%)")
    cover = float(n_per_team.min()) / TEAM_SIZE
    if cover < 0.9:
        print(f"  ! 프레임당 추적 인원이 명목 인원의 {cover*100:.0f}% 뿐이다. "
              f"화면 밖 선수가 빠진 좌표로 보인다.")
        print(f"    이 상태로 SC 를 계산하면 남은 수비수가 빈 자리의 공간 점유까지 "
              f"받아가서 값이 부풀려진다. usable 비율이 낮게 나오는 것이 정상이며, "
              f"경기장 전체를 덮는 좌표를 먼저 확보할 것.")

    # ── ID 안정성 점검 및 복구 ───────────────────────────────
    v = implied_speed(tracks, FPS)
    med, p95 = v.median(), v.quantile(0.95)
    print(f"  ID 기준 이동속도 중앙값 {med:.1f} m/s , 95% {p95:.1f} m/s")
    need = p95 > 15.0
    if REPAIR_IDS is True or (REPAIR_IDS == "auto" and need):
        if need:
            print("  ! 사람이 낼 수 없는 속도다. 프레임마다 ID 가 뒤바뀌는 데이터로 보인다.")
        print("  -> 헝가리안 매칭으로 ID 재연결 중...")
        tracks = repair_ids(tracks)
        v2 = implied_speed(tracks, FPS)
        print(f"     복구 후 이동속도 중앙값 {v2.median():.1f} m/s , 95% {v2.quantile(.95):.1f} m/s "
              f"(총 이동거리 {v2.sum()/max(v.sum(),1e-9)*100:.0f}%)")
        if v2.quantile(.95) > 15.0:
            print("     ! 복구 후에도 속도가 비현실적이다. 원본 좌표를 다시 확인할 것.")

    # ── 공 소유 반경 자동 결정 ───────────────────────────────
    mb = tracks.merge(ball, on="frame", how="inner")
    nn = np.hypot(mb.X - mb.ball_x, mb.Y - mb.ball_y).groupby(mb.frame).min()
    if POSS_RADIUS is None:
        for r in (2.0, 3.0, 5.0, 8.0):
            if (nn <= r).mean() >= 0.6:
                POSS_RADIUS = r
                break
        POSS_RADIUS = POSS_RADIUS or float(np.ceil(nn.quantile(0.8)))
    cov = (nn <= POSS_RADIUS).mean()
    print(f"  공-최근접선수 거리 중앙값 {nn.median():.1f} m -> POSS_RADIUS = {POSS_RADIUS} m "
          f"(프레임의 {cov*100:.0f}% 에서 소유자 판정 가능)")
    if cov < 0.4:
        print("  ! 소유자 판정이 되는 프레임이 적다. 공 좌표 품질을 먼저 확인할 것.")

    return tracks.reset_index(drop=True), ball.reset_index(drop=True), teams


# ══════════════ 2. L1 — 속도 (중앙차분) ══════════════
def calculate_velocity(df, delta=DELTA, fps=None, speed_max=SPEED_MAX, jump_max_m=0.5):
    fps = fps or FPS
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    dt = 2.0 * delta / fps
    parts = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame").drop_duplicates("frame", keep="first")   # 중복 프레임 방어
        full = pd.RangeIndex(int(g.frame.min()), int(g.frame.max()) + 1)
        gi = g.set_index("frame")
        had = gi["X"].reindex(full).notna()
        gf = gi.reindex(full)
        X, Y = gf["X"], gf["Y"]
        vx = (X.shift(-delta) - X.shift(delta)) / dt
        vy = (Y.shift(-delta) - Y.shift(delta)) / dt
        sp = np.hypot(vx, vy)
        bad = sp.isna() | (sp > speed_max)
        step = np.hypot(X.diff(), Y.diff())
        parts.append(pd.DataFrame({
            "frame": full, "track_id": tid,
            "vx": vx.mask(bad).values, "vy": vy.mask(bad).values,
            "speed": sp.mask(bad).values, "kinematics_ok": (~bad).values,
            "frame_jump_flag": (step > jump_max_m).fillna(False).values,
        })[had.values])
    res = pd.concat(parts, ignore_index=True)
    added = {"vx", "vy", "speed", "kinematics_ok", "frame_jump_flag"}
    res = res.merge(df[[c for c in df.columns if c not in added]],
                    on=["track_id", "frame"], how="left")     # team 등 입력 열 통과
    return res.sort_values(["frame", "track_id"]).reset_index(drop=True)


# ══════════════ 3. L1b — 공 소유 판정 ══════════════
def fill_ball_gaps(ball, frames, max_gap=None):
    """
    공 좌표의 짧은 끊김을 선형보간으로 메운다.

    공은 검출이 제일 잘 끊긴다(작고 빠르고 선수에 가림). 그런데 공 좌표가 없으면
    그 프레임은 소유자 판정이 안 돼서 SC·PR 이 통째로 사라진다.
    공은 굴러가거나 날아가는 물체라 짧은 구간은 안전하게 보간할 수 있다.
    긴 구간은 어디로 갔는지 알 수 없으므로 비워 둔다.

    반환에 ball_interp 열(보간된 행 표시)이 추가된다.
    """
    max_gap = BALL_MAX_GAP if max_gap is None else max_gap
    full = pd.RangeIndex(int(min(frames)), int(max(frames)) + 1)
    b = (ball.drop_duplicates("frame").set_index("frame")
              .reindex(full).rename_axis("frame"))
    missing = b["ball_x"].isna()
    if max_gap > 0 and missing.any():
        # 결측 구간마다 길이를 재서, max_gap 이하인 곳만 보간한다
        grp = (~missing).cumsum()
        run = missing.groupby(grp).transform("sum")
        allow = missing & (run <= max_gap) & b["ball_x"].ffill().notna() \
                        & b["ball_x"].bfill().notna()
        for c in ("ball_x", "ball_y"):
            b[c] = b[c].interpolate(method="index").where(allow | ~missing, np.nan)
        if "ball_z" in b.columns:
            b["ball_z"] = b["ball_z"].interpolate(method="index").where(allow | ~missing, np.nan)
        b["ball_interp"] = allow
    else:
        b["ball_interp"] = False
    return b.dropna(subset=["ball_x", "ball_y"]).reset_index()


def build_frames_table(tracks, ball, teams):
    frames = np.sort(tracks["frame"].unique())
    ball = fill_ball_gaps(ball, frames)

    m = tracks.merge(ball, on="frame", how="inner")
    m["d_ball"] = np.hypot(m.X - m.ball_x, m.Y - m.ball_y)
    cols = ["frame", "time_s", "ball_x", "ball_y", "track_id", "team", "d_ball"]
    if "ball_interp" in m.columns:
        cols.append("ball_interp")
    if "ball_z" in m.columns:
        cols.append("ball_z")
    near = m.loc[m.groupby("frame")["d_ball"].idxmin(), cols]
    near = near.rename(columns={"track_id": "raw_id", "team": "raw_team"}).reset_index(drop=True)

    ok = near.d_ball <= POSS_RADIUS
    if "ball_z" in near.columns:
        # 공이 공중에 떠 있으면(크로스·롱볼·헤더) 지면상 가장 가까운 선수가
        # 실제로 그 공을 다루고 있다는 보장이 없다. 헤더 경합처럼 점프와 타이밍으로
        # 결정되는 상황이라 2D 최근접 거리로는 소유자를 정할 수 없다.
        # 그래서 '지면 소유자 없음'으로 두어 phase 를 transition 으로 보내고,
        # PA(지상 패스선·수신공간 가정)는 이 구간에서 계산하지 않는다.
        # 팀 동료 사이의 짧은 칩패스는 아래 HOLD_FRAMES 흡수 규칙이 다시 이어 준다.
        near["ball_air"] = near["ball_z"].fillna(0) > BALL_AIRBORNE_Z
        ok = ok & ~near["ball_air"]
    near["raw_team"] = np.where(ok, near.raw_team, "none")
    near["raw_id"] = np.where(ok, near.raw_id, np.nan)

    # ── 소유 전환 판정 ────────────────────────────────────────
    #  두 가지를 지켜야 한다.
    #   (1) '행'이 아니라 '프레임' 단위로 세야 한다. 공이 끊겨 행이 빠지면
    #       행을 세는 방식은 1.5초 규칙을 조용히 3초 규칙으로 바꿔버린다.
    #   (2) 짧은 구간을 무조건 이전 값으로 덮으면 안 된다. 데이터가 성기면
    #       모든 구간이 짧아져서 전체가 첫 프레임 값 하나로 잠겨버린다.
    #       그래서 '앞뒤가 같은 값인 짧은 구간'만 잡음으로 보고 흡수한다.
    #       (A B A 에서 짧은 B 는 흡수, A B C 의 B 는 실제 전환이므로 유지)
    full = pd.RangeIndex(int(frames.min()), int(frames.max()) + 1)
    axis = near.set_index("frame").reindex(full)
    # 공이 없어 비어 있는 프레임은 '소유자 없음'이 아니라 '모름'이므로 앞값을 잇는다.
    raw = axis["raw_team"].ffill().bfill().fillna("none").to_numpy().astype(object)

    def _runs(a):
        out, i = [], 0
        while i < len(a):
            j = i
            while j < len(a) and a[j] == a[i]:
                j += 1
            out.append([i, j, a[i]])
            i = j
        return out

    for _ in range(5):                       # 흡수 후 이웃이 합쳐지므로 몇 번 반복
        rs = _runs(raw)
        changed = False
        for k in range(1, len(rs) - 1):
            i0, i1, v = rs[k]
            if (i1 - i0) < HOLD_FRAMES and rs[k - 1][2] == rs[k + 1][2] and v != rs[k - 1][2]:
                raw[i0:i1] = rs[k - 1][2]
                changed = True
        if not changed:
            break

    axis["poss_team"] = raw
    axis = axis.loc[near["frame"].to_numpy()]
    near["poss_team"] = axis["poss_team"].to_numpy()
    near["poss_id"] = np.where(near.poss_team == near.raw_team, near.raw_id, np.nan)
    near["phase"] = np.where((near.poss_team == "none") | near.poss_id.isna(),
                             "transition", "settled")

    cnt = tracks.groupby(["frame", "team"]).size().unstack(fill_value=0)
    for t in teams:
        if t not in cnt:
            cnt[t] = 0
    out = near.merge(cnt[teams].reset_index(), on="frame", how="left")
    out["usable"] = (out.poss_team != "none") & (out[teams].min(axis=1) >= MIN_TRACKED)
    return out


# ══════════════ 4. PR — 압박 기여도 ══════════════
def calculate_pr(tracks, F, delta=DELTA, fps=None):
    fps = fps or FPS
    dt = 2.0 * delta / fps
    car = tracks[["frame", "track_id", "X", "Y"]].rename(
        columns={"track_id": "poss_id", "X": "cx", "Y": "cy"}).drop_duplicates(["frame", "poss_id"])
    f = F[["frame", "ball_x", "ball_y", "poss_team", "poss_id", "phase"]].merge(
        car, on=["frame", "poss_id"], how="left")
    st, tr = f.phase == "settled", f.phase == "transition"
    f["tx"] = np.where(st, f.cx, np.where(tr, f.ball_x, np.nan))
    f["ty"] = np.where(st, f.cy, np.where(tr, f.ball_y, np.nan))
    f["target_type"] = np.where(st, "carrier", np.where(tr, "ball", "none"))

    df = tracks.merge(f[["frame", "poss_team", "tx", "ty", "target_type"]], on="frame", how="left")
    df = df[df.poss_team.notna() & (df.poss_team != "none") & (df.team != df.poss_team)].copy()
    if df.empty:
        return pd.DataFrame()
    df["d"] = np.hypot(df.X - df.tx, df.Y - df.ty)

    parts = []
    for tid, g in df.groupby("track_id", sort=False):
        g = g.sort_values("frame").drop_duplicates("frame", keep="first")
        full = pd.RangeIndex(int(g.frame.min()), int(g.frame.max()) + 1)
        gi = g.set_index("frame")
        had = gi["d"].reindex(full).notna()
        gf = gi.reindex(full)
        d = gf["d"]
        v = -(d.shift(-delta) - d.shift(delta)) / dt
        tt = gf["target_type"]
        bad = v.isna() | (tt.shift(-delta) != tt) | (tt.shift(delta) != tt)
        parts.append(pd.DataFrame({
            "frame": full, "track_id": tid, "team": gf["team"].values,
            "time_s": gf["time_s"].values, "target_type": tt.values,
            "d": d.values, "v_app": v.mask(bad).values,
            "kinematics_ok": (~bad).values})[had.values])

    r = pd.concat(parts, ignore_index=True)
    # 거리 커널. PD 는 이것의 제곱이고, 속도 성분에도 같은 커널을 곱한다.
    #   원래 PV = clip(v_app/PR_VMAX, 0, 1) 에는 거리가 전혀 안 들어갔다. 그래서
    #   공에서 40 m 떨어진 수비수가 공 쪽으로 5 m/s 로 달리기만 해도 PV=1 -> PR=0.4 가
    #   나왔고, 이는 볼 소유자에게 1 m 까지 붙어 자리를 지킨 수비수(PR=0.486)와
    #   거의 같은 값이다. 모사 경기에서 재 보니 25 m 밖에서 쌓인 값이 PR 총량의
    #   50% 였고, 선언한 가중치 60:40 과 달리 실제 기여는 거리 18 : 속도 82 였다.
    #   먼 거리에서의 복귀 주력은 '압박'이 아니라 활동량이므로 PR 에 들어오면 안 된다.
    #   PR_R 을 그대로 쓰므로 새로 정할 상수는 없다.
    prox = np.clip(1 - r.d / PR_R, 0, 1)
    r["PD"] = prox ** 2
    r["PV"] = (np.clip(r.v_app / PR_VMAX, 0, 1) * prox).where(r.kinematics_ok) + 0.0
    r["PR"] = PR_WD * r.PD + PR_WV * r.PV
    return r.sort_values(["frame", "track_id"]).reset_index(drop=True)


# ══════════════ 5. SC — 공간 통제 (Shapley 배분) ══════════════
def make_grid(step=GRID_STEP):
    xs = np.arange(step / 2, PITCH_L, step)
    ys = np.arange(step / 2, PITCH_W, step)
    gx, gy = np.meshgrid(xs, ys)
    return np.c_[gx.ravel(), gy.ravel()], step * step


def danger_weight(grid, goal, ball, cell_area):
    dg = np.linalg.norm(grid - np.asarray(goal, float), axis=1)
    db = np.linalg.norm(grid - np.asarray(ball, float), axis=1)
    return np.exp(-dg / LAMBDA_GOAL) * np.exp(-db / LAMBDA_BALL) * cell_area


def sc_shapley(grid, def_xy, att_xy, w):
    """셀 c 를 δ_A 보다 가깝게 커버하는 수비수 k 명이 w(c)/k 씩 나눠 갖는다."""
    def_xy = np.asarray(def_xy, float).reshape(-1, 2)
    att_xy = np.asarray(att_xy, float).reshape(-1, 2)
    if len(def_xy) == 0 or len(att_xy) == 0:
        return np.zeros(len(def_xy)), 0.0
    dA, _ = cKDTree(att_xy).query(grid, k=1)
    dD = np.linalg.norm(grid[:, None, :] - def_xy[None, :, :], axis=2)
    covers = dD < dA[:, None]
    k = covers.sum(axis=1)
    ctrl = k > 0
    per = np.where(ctrl, w / np.maximum(k, 1), 0.0)
    return (covers * per[:, None]).sum(axis=0), float(w[ctrl].sum())


def calculate_sc(tracks, F, teams):
    grid, area = make_grid()
    goals = {teams[0]: (0.0, PITCH_W / 2), teams[1]: (PITCH_L, PITCH_W / 2)}
    if not TEAM_A_ATTACKS_PLUS_X:
        goals = {teams[0]: (PITCH_L, PITCH_W / 2), teams[1]: (0.0, PITCH_W / 2)}

    meta = F.set_index("frame")
    rows, t0, n = [], time.perf_counter(), 0
    frames = sorted(set(tracks.frame.unique()) & set(meta.index[meta.usable]))
    for fr in frames:
        m = meta.loc[fr]
        g = tracks[tracks.frame == fr]
        att_t = m.poss_team
        def_t = teams[0] if att_t == teams[1] else teams[1]
        att = g[g.team == att_t][["X", "Y"]].to_numpy(float)
        dfd = g[g.team == def_t]
        if len(att) == 0 or len(dfd) == 0:
            continue
        w = danger_weight(grid, goals[def_t], (m.ball_x, m.ball_y), area)
        sc, tot = sc_shapley(grid, dfd[["X", "Y"]].to_numpy(float), att, w)
        # SC 는 '위험도 가중 면적'이라 프레임마다 파이의 크기 자체가 다르다.
        # (공이 자기 박스 안이면 총량 320, 상대 박스면 29 로 11배 차이)
        # 그래서 프레임끼리 비교하려면 팀 총량 대비 몫도 같이 본다. 효율성 공리
        # 덕분에 한 프레임의 share 합은 정확히 1 이다.
        rows.append(pd.DataFrame({"frame": fr, "track_id": dfd.track_id.to_numpy(),
                                  "SC": sc, "sc_team_total": tot,
                                  "SC_share": sc / tot if tot > 0 else 0.0}))
        n += 1
        if n % 500 == 0:
            print(f"    {n}/{len(frames)} 프레임 ... {(time.perf_counter()-t0)/n*1000:.1f} ms/프레임")
    if not rows:
        return pd.DataFrame(columns=["frame", "track_id", "SC", "sc_team_total", "SC_share"])
    print(f"    {n} 프레임 완료, {(time.perf_counter()-t0)/max(n,1)*1000:.1f} ms/프레임")
    return pd.concat(rows, ignore_index=True)


# ══════════════ 5b. PA — 패스 유인도 ══════════════
def _lane_clearance(carrier, receivers, defenders, w_lane=None):
    """
    패스선(볼 소유자 -> 각 아군)에 대해, 경로를 막고 있는 수비수까지의
    최소 수직거리를 구한다. 수선의 발이 선분 '안'에 있는 수비수만 센다.
    뒤쪽이나 너머에 있는 수비수가 길을 막는 것으로 잘못 계산되는 것을 막는다.
    수비수가 하나도 걸치지 않으면 w_lane(= 완전히 열림)으로 둔다.
    """
    w_lane = W_LANE if w_lane is None else w_lane
    n = len(receivers)
    if n == 0:
        return np.zeros(0)
    v = receivers - carrier                       # (n,2)
    vv = (v * v).sum(1)                           # (n,)
    vv = np.where(vv < 1e-9, 1e-9, vv)
    if len(defenders) == 0:
        return np.full(n, w_lane)
    wv = defenders - carrier                      # (m,2)
    t = (wv @ v.T) / vv                           # (m,n)
    proj = t[:, :, None] * v[None, :, :]          # (m,n,2)
    dist = np.linalg.norm(wv[:, None, :] - proj, axis=2)   # (m,n)
    inside = (t > 0.0) & (t < 1.0)
    dist = np.where(inside, dist, np.inf)
    g = dist.min(axis=0)
    return np.where(np.isfinite(g), g, w_lane)


def calculate_pa(tracks, F, teams, alpha=1.0, beta=1.0, gamma=1.0):
    """
    공을 갖지 않은 공격팀 선수가 얼마나 좋은 패스 선택지였는가.

        l   길목 개방도      = clip(경로 최소 수직거리 / W_LANE, 0, 1)
        phi 수신 공간        = clip(최근접 수비수 거리 / PA_SPACE_R, 0, 1)
        pi  전진 가치        = PA_PI_MIN ~ 1 (상대 골대에 가까워지는 만큼 큼)
        PA  = l^alpha * phi^beta * pi^gamma

    전진 가치는 '상대 골대까지의 거리가 얼마나 줄어드는가'로 잰다.
    X 성분만 보면 측면으로 넓게 벌리는 전환 패스(스위치)나 대각 패스가
    전부 '전진 0' 으로 깎인다. 실제 경기 영상에서 빌드업의 상당 부분이
    이 대각·측면 전환으로 이뤄지고, 골문에서 먼 측면으로 가는 패스와
    골문 정면 쪽으로 파고드는 패스는 가치가 분명히 다르다.
    골대까지의 유클리드 거리 감소로 재면 두 경우가 자연히 구분된다.

    지수를 전부 1 로 두면 정의서의 원안 PA = l*phi*pi 와 정확히 같다.
    곱셈 모델은 로그를 씌우면 선형모델이므로, 실제 패스 데이터로
    지수를 학습할 수 있다(learn_pa_exponents 참고).
    """
    dirs = {teams[0]: (1.0 if TEAM_A_ATTACKS_PLUS_X else -1.0)}
    dirs[teams[1]] = -dirs[teams[0]]
    meta = F.set_index("frame")
    rows = []

    for fr, g in tracks.groupby("frame", sort=True):
        if fr not in meta.index:
            continue
        m = meta.loc[fr]
        if m.phase != "settled" or pd.isna(m.poss_id):
            continue
        att_t = m.poss_team
        def_t = teams[0] if att_t == teams[1] else teams[1]
        car = g[g.track_id == m.poss_id]
        if len(car) == 0:
            continue
        c = car[["X", "Y"]].to_numpy(float)[0]
        rec = g[(g.team == att_t) & (g.track_id != m.poss_id)]
        dfd = g[g.team == def_t]
        if len(rec) == 0:
            continue
        P = rec[["X", "Y"]].to_numpy(float)
        D = dfd[["X", "Y"]].to_numpy(float)

        lane = np.clip(_lane_clearance(c, P, D) / W_LANE, 0, 1)
        if len(D):
            space = np.clip(np.linalg.norm(P[:, None, :] - D[None, :, :], axis=2).min(1)
                            / PA_SPACE_R, 0, 1)
        else:
            space = np.ones(len(P))
        goal = np.array([PITCH_L if dirs[att_t] > 0 else 0.0, PITCH_W / 2.0])
        gain = (np.linalg.norm(c - goal)
                - np.linalg.norm(P - goal, axis=1))     # + = 골대에 가까워짐
        prog = PA_PI_MIN + (1 - PA_PI_MIN) * (np.clip(gain / PA_FWD_REF, -1, 1) + 1) / 2

        rows.append(pd.DataFrame({
            "frame": fr, "track_id": rec.track_id.to_numpy(), "team": att_t,
            "PA_lane": lane, "PA_space": space, "PA_prog": prog,
            "carrier_id": m.poss_id,
            "PA": lane ** alpha * space ** beta * prog ** gamma}))

    if not rows:
        return pd.DataFrame(columns=["frame", "track_id", "team", "PA_lane",
                                     "PA_space", "PA_prog", "carrier_id", "PA"])
    return pd.concat(rows, ignore_index=True)


def detect_passes(F):
    """
    좌표만으로 패스를 찾는다. 같은 팀 안에서 볼 소유자가 바뀌면 패스로 본다.
    반환: from_frame(패스 직전 프레임), from_id, to_id
    """
    f = F[F.phase == "settled"].sort_values("frame")
    f = f[f.poss_id.notna()]
    prev_id = f.poss_id.shift()
    prev_team = f.poss_team.shift()
    prev_frame = f.frame.shift()
    m = (f.poss_id != prev_id) & (f.poss_team == prev_team) & prev_id.notna()
    return pd.DataFrame({"from_frame": prev_frame[m].astype(int),
                         "from_id": prev_id[m], "to_id": f.poss_id[m]}).reset_index(drop=True)


def learn_pa_exponents(pa_df, passes):
    """
    실제로 일어난 패스를 정답으로 삼아 지수 a, b, c 를 학습한다.
      양성: 그 프레임에 실제로 공을 받은 선수
      음성: 같은 프레임의 나머지 아군
    log 를 씌우면 곱셈 모델이 선형모델이 되므로 로지스틱 회귀로 추정된다.
    """
    if len(passes) < 30:
        return None, f"패스 표본이 {len(passes)}개뿐이라 학습을 건너뜀 (30개 이상 필요)"
    d = pa_df.merge(passes, left_on=["frame", "carrier_id"],
                    right_on=["from_frame", "from_id"], how="inner")
    if len(d) == 0:
        return None, "패스 프레임과 PA 계산 프레임이 겹치지 않음"
    d["y"] = (d.track_id == d.to_id).astype(int)
    if d.y.sum() < 15 or (1 - d.y).sum() < 15:
        return None, f"양성 {int(d.y.sum())} / 음성 {int((1-d.y).sum())} — 표본 부족"
    eps = 1e-6
    X = np.c_[np.log(d.PA_lane + eps), np.log(d.PA_space + eps), np.log(d.PA_prog + eps)]
    y = d.y.to_numpy()
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None, "sklearn 이 없어 학습을 건너뜀 (!pip install scikit-learn)"

    # 순환논법 방지 + 잡음 방지
    #   (1) 학습에 쓴 패스로 성능을 재면 안 되므로 패스 단위로 70/30 분할.
    #   (2) 표본이 적으면 우연히 좋아 보인다. 같은 데이터를 다시 쪼개는 것으로는
    #       독립적인 증거가 안 되므로, '정답을 무작위로 섞었을 때도 이만큼
    #       좋아지는가'를 직접 재는 순열검정을 한다.
    #       섞은 데이터에서의 개선폭 분포보다 확실히 클 때만 채택한다.
    ev = d["from_frame"].to_numpy()
    uniq = np.unique(ev)
    rs = np.random.default_rng(0)
    te = set(rs.choice(uniq, max(1, int(len(uniq) * 0.3)), replace=False).tolist())
    m_te = np.array([e in te for e in ev]); m_tr = ~m_te
    if m_tr.sum() < 20 or m_te.sum() < 10 \
       or len(np.unique(y[m_tr])) < 2 or len(np.unique(y[m_te])) < 2:
        return None, "학습/검증 분할에 표본이 부족해 지수 1 을 유지 (정의서 §5 A판)"

    def fit_gain(yy):
        mdl = LogisticRegression(max_iter=2000).fit(X[m_tr], yy[m_tr])
        af = roc_auc_score(yy[m_te], d.PA.to_numpy()[m_te])
        al = roc_auc_score(yy[m_te], mdl.predict_proba(X[m_te])[:, 1])
        return al - af, al, af, mdl.coef_[0]

    gain, auc_l, auc_f, co = fit_gain(y)

    null = []
    ser = pd.Series(y)
    for _ in range(60):                       # 이벤트 안에서 정답만 섞는다
        yp = ser.groupby(ev).transform(lambda v: rs.permutation(v.to_numpy())).to_numpy()
        if len(np.unique(yp[m_tr])) < 2 or len(np.unique(yp[m_te])) < 2:
            continue
        null.append(fit_gain(yp)[0])
    thr = float(np.percentile(null, 95)) if len(null) >= 20 else 0.10

    if co[0] <= 0:
        return None, f"길목 계수가 음수({co[0]:.2f}) — 신호가 뒤집혔다. 지수 1 유지"
    est = np.clip(co / co[0], 0.1, 5)
    if gain <= thr:
        return None, (f"추정된 지수는 a={est[0]:.2f} b={est[1]:.2f} c={est[2]:.2f} 이지만, "
                      f"검증 AUC 개선 {gain:+.3f} 이 정답을 섞었을 때의 {thr:+.3f} 를 못 넘는다.\n"
                      f"          -> 지수 1 을 유지한다 (정의서 §5 A판). 위 추정값은 참고용으로만 쓸 것.\n"
                      f"          패스 표본이 수천 개 쌓이면 다시 판정된다.")
    exps = tuple(np.clip(co / co[0], 0.1, 5))
    return exps, (f"패스 {len(passes)}개 -> 지수 a={exps[0]:.2f} b={exps[1]:.2f} "
                  f"c={exps[2]:.2f} , 검증 AUC {auc_f:.3f} -> {auc_l:.3f} "
                  f"(개선 {gain:+.3f} > 순열검정 문턱 {thr:+.3f}) 채택")


# ══════════════ 5c. 지표 진단 ══════════════
def metric_report(M, PAdf, agg):
    """
    값이 '상대적으로 말이 되는 범위'에 있는지 매 실행마다 확인한다.

    지표가 틀리는 방식은 두 가지다.
      (1) 바닥 효과 — 대부분의 행에서 0 이라 사실상 '몇 번 관여했나'만 재게 된다.
      (2) 포화     — 대부분의 행에서 1 이라 그 차원이 아무 정보도 나르지 못한다.
    둘 다 평균값만 보면 안 보이고, 분포를 봐야 드러난다. 상수를 바꿀 근거도
    여기서 나온다.
    """
    print("\n[지표 진단]  각 성분이 실제로 정보를 나르고 있는지")
    pr = M[M.PR.notna()] if "PR" in M.columns else M.iloc[:0]
    if len(pr):
        inz = (pr.d <= PR_R).mean()
        cPD, cPV = PR_WD * pr.PD.mean(), PR_WV * pr.PV.mean()
        tot = cPD + cPV
        print(f"  PR  압박권(d<={PR_R:.0f}m) 안 {inz*100:4.1f}% / "
              f"공까지 거리 중앙값 {pr.d.median():.1f} m")
        if tot > 0:
            print(f"      실제 기여 거리항 {cPD/tot*100:4.1f}% : 속도항 {cPV/tot*100:4.1f}% "
                  f"(선언한 가중치 {PR_WD*100:.0f}:{PR_WV*100:.0f})")
        far = pr[pr.d > 25]
        if pr.PR.sum() > 0:
            print(f"      25m 밖에서 쌓인 몫 {far.PR.sum()/pr.PR.sum()*100:4.1f}% "
                  f"— 여기가 크면 압박이 아니라 활동량을 재고 있는 것이다")
    if len(PAdf):
        for c in ("PA_lane", "PA_space", "PA_prog"):
            s = PAdf[c]
            print(f"  {c:9s} 평균 {s.mean():.3f} 표준편차 {s.std():.3f} / "
                  f"1.0 포화 {(s >= 0.999).mean()*100:4.1f}% , 0.0 바닥 {(s <= 0.001).mean()*100:4.1f}%")
        print(f"      포화가 절반을 넘으면 그 차원은 순위에 기여하지 못한다 "
              f"(W_LANE={W_LANE}, PA_SPACE_R={PA_SPACE_R} 를 키울 것)")
    # 지표가 '어디에 서는 선수인가'로 얼마나 설명되는지
    if len(agg) > 3 and "d_mean" in agg.columns:
        from scipy.stats import spearmanr as _sp
        for c in ("SC_mean", "PR_mean", "PA_mean"):
            s = agg[c].dropna()
            d = agg["d_mean"].reindex(s.index)
            ok = s.notna() & d.notna()
            if ok.sum() > 3:
                print(f"  {c:8s} vs 공까지 평균거리 ρ={_sp(s[ok], d[ok]).statistic:+.3f}", end="")
        print("\n      |ρ| 가 1 에 가까우면 그 지표는 능력이 아니라 포지션을 재고 있다")


# ══════════════ 6. 자체 검증 ══════════════
def self_test():
    print("\n[검증] 계산이 정의대로 되는지 확인")
    fps = 30.0
    f = np.array([x for x in range(60) if not (12 <= x <= 20)])
    d = pd.DataFrame({"frame": f, "time_s": f / fps, "track_id": 1, "team": "A",
                      "X": 5.0 * f / fps, "Y": 0.0})
    global FPS
    old, FPS = FPS, fps
    v = calculate_velocity(d)
    FPS = old
    c1 = np.allclose(v[(v.frame >= 40) & (v.frame <= 45)].speed, 5.0, atol=1e-6)
    c2 = v[(v.frame >= 9) & (v.frame <= 11)].speed.isna().all()
    c3 = "team" in v.columns

    grid, area = make_grid(1.0)
    rng = np.random.default_rng(0)
    D = rng.uniform([10, 5], [40, 63], size=(11, 2))
    A = rng.uniform([50, 5], [95, 63], size=(11, 2))
    w = danger_weight(grid, (0, 34), (33, 34), area)
    sc, tot = sc_shapley(grid, D, A, w)
    c4 = np.isclose(sc.sum(), tot)
    s2, _ = sc_shapley(grid, np.array([[24., 34.], [24.7, 34.]]), A, w)
    c5 = abs(s2[0] - s2[1]) / max(s2) < 0.05
    c6 = danger_weight(np.array([[85., 10.]]), (0, 34), (33, 34), 1)[0] < \
         danger_weight(np.array([[24., 34.]]), (0, 34), (33, 34), 1)[0]

    # PR: 멀리서 공 쪽으로 달리기만 하는 수비수가 압박으로 잡히면 안 된다.
    #   붙어서 자리를 지킨 수비수(1 m, 정지) vs 30 m 밖에서 5 m/s 로 달려오는 수비수.
    old, FPS = FPS, fps
    fr = np.arange(20)
    tk = [{"frame": f, "time_s": f/fps, "track_id": "A0", "team": "A", "X": 50.0, "Y": 34.0}
          for f in fr]
    tk += [{"frame": f, "time_s": f/fps, "track_id": "B_near", "team": "B", "X": 51.0, "Y": 34.0}
           for f in fr]
    tk += [{"frame": f, "time_s": f/fps, "track_id": "B_far", "team": "B",
            "X": 20.0 + 5.0*f/fps, "Y": 34.0} for f in fr]
    Ft = pd.DataFrame({"frame": fr, "ball_x": 50.0, "ball_y": 34.0,
                       "poss_team": "A", "poss_id": "A0", "phase": "settled"})
    pr = calculate_pr(pd.DataFrame(tk), Ft)
    FPS = old
    mid = pr[(pr.frame >= 8) & (pr.frame <= 11)]
    near = mid[mid.track_id == "B_near"].PR.mean()
    far = mid[mid.track_id == "B_far"].PR.mean()
    c7 = (far < 0.02) and (near > 5 * max(far, 1e-9))

    for nm, c in [("등속 5.00 m/s 정확", c1), ("추적 끊김 구간 NaN", c2),
                  ("team 열 통과", c3), (f"효율성 공리 (오차 {abs(sc.sum()-tot):.0e})", c4),
                  (f"협력 수비 보존 ({s2[0]:.0f} vs {s2[1]:.0f})", c5),
                  ("위험도 가중치 방향", c6),
                  (f"먼 거리 복귀주력은 압박 아님 (근접 {near:.3f} vs 원거리 {far:.3f})", c7)]:
        print(f"  {'PASS' if c else '**FAIL**':9s} {nm}")
    if not all([c1, c2, c3, c4, c5, c6, c7]):
        raise SystemExit("검증 실패. 아래 결과를 믿으면 안 된다.")


# ══════════════ 7. 실행 ══════════════
def main():
    self_test()
    tracks, ball, teams = read_data()

    print("\n[계산]")
    n_raw = len(tracks)
    tracks = fill_player_gaps(tracks)
    n_itp = int(tracks["pos_interp"].sum())
    print(f"  선수좌표 결측 보간 {n_itp}행 추가 (원본 {n_raw:,}행, 최대 "
          f"{PLAYER_MAX_GAP}프레임 = {PLAYER_MAX_GAP/FPS:.2f}초 구간까지)")
    L1 = calculate_velocity(tracks)
    print(f"  속도    유효 {L1.kinematics_ok.mean()*100:.0f}% , 중앙값 {L1.speed.median():.2f} m/s")

    nfr = L1.frame.nunique()
    cov0 = ball.frame.nunique() / nfr
    F = build_frames_table(L1, ball, teams)
    itp = int(F["ball_interp"].sum()) if "ball_interp" in F.columns else 0
    print(f"  공좌표  원본 {cov0*100:.0f}% 프레임에 존재"
          f" -> 보간 {itp}프레임 추가 (최대 {BALL_MAX_GAP}프레임 = {BALL_MAX_GAP/FPS:.2f}초 구간까지)")
    if cov0 < 0.7:
        print(f"    ! 공 검출이 {(1-cov0)*100:.0f}% 끊겼다. 보간으로 메울 수 있는 건 짧은 구간뿐이라"
              " 긴 구간은 그대로 버려진다. 공 검출부터 개선할 것.")
    print(f"  공소유  usable {F.usable.mean()*100:.0f}% , {F.poss_team.value_counts().to_dict()}")
    if F.usable.mean() < 0.3:
        no_poss = (F.poss_team == "none").mean()
        few = (F[teams].min(axis=1) < MIN_TRACKED).mean()
        print(f"  ! usable 이 너무 낮다. 소유자 미판정 {no_poss*100:.0f}% / "
              f"추적 인원 부족 {few*100:.0f}% 이 원인이다.")
        print("    앞쪽이 크면 공 검출·POSS_RADIUS 문제, 뒤쪽이 크면 화면 밖 선수가 "
              "빠진 좌표라 경기장 전체를 덮는 트래킹이 필요하다.")

    PRdf = calculate_pr(L1, F)
    print(f"  PR      {len(PRdf):,}행")
    print("  SC      계산 중...")
    SCdf = calculate_sc(L1, F, teams)

    PAdf = calculate_pa(L1, F, teams)
    passes = detect_passes(F)
    print(f"  PA      {len(PAdf):,}행 , 좌표에서 찾은 패스 {len(passes)}개")
    if PA_LEARN and len(PAdf):
        exps, msg = learn_pa_exponents(PAdf, passes)
        print(f"          {msg}")
        if exps:
            PAdf = calculate_pa(L1, F, teams, *exps)
    if len(PAdf) == 0:
        print("          ! PA 가 비었다. settled 프레임이나 볼 소유자 판정을 확인할 것")

    if len(SCdf):
        chk = SCdf.groupby("frame").agg(s=("SC", "sum"), t=("sc_team_total", "first"))
        print(f"  검산    효율성 공리 최대오차 {np.abs(chk.s - chk.t).max():.1e}")

    M = PRdf.merge(SCdf[["frame", "track_id", "SC", "SC_share"]],
                   on=["frame", "track_id"], how="left") \
        if len(SCdf) else PRdf.assign(SC=np.nan, SC_share=np.nan)
    # PR/SC 는 수비 중인 선수, PA 는 공격 중인 선수라 서로 다른 프레임에서 나온다.
    # 그래서 행을 합치지 않고 바깥조인으로 이어 붙인다.
    pa_cols = ["frame", "track_id", "team", "PA", "PA_lane", "PA_space", "PA_prog"]
    M = M.merge(PAdf[pa_cols] if len(PAdf) else
                pd.DataFrame(columns=pa_cols),
                on=["frame", "track_id", "team"], how="outer")

    agg = (M.groupby("track_id")
             .agg(team=("team", "first"), frames=("frame", "size"),
                  PR_mean=("PR", "mean"), PD_mean=("PD", "mean"), PV_mean=("PV", "mean"),
                  SC_mean=("SC", "mean"), SC_total=("SC", "sum"),
                  SC_share=("SC_share", "mean"), d_mean=("d", "mean"),
                  PA_mean=("PA", "mean"), PA_frames=("PA", "count"),
                  PA_lane=("PA_lane", "mean"), PA_space=("PA_space", "mean"),
                  PA_prog=("PA_prog", "mean"))
             .round(4))

    # PR_mean 하나에는 '얼마나 자주 압박 상황에 있었나'(양)와 '그때 얼마나
    # 좋았나'(질)가 섞여 있다. 수비형 미드필더는 양이 많아서, 센터백은 양이
    # 없어서 값이 갈리는데 둘 다 PR_mean 한 숫자로만 보면 구분이 안 된다.
    prm = M[M.PR.notna()] if "PR" in M.columns else M.iloc[:0]
    if len(prm):
        agg["PR_zone"] = (prm.assign(_z=prm.d <= PR_R)
                          .groupby("track_id")["_z"].mean().round(4))     # 관여율
        agg["PR_in"] = (prm[prm.d <= PR_R].groupby("track_id")["PR"]
                        .mean().round(4))                                 # 관여했을 때의 질
    metric_report(M, PAdf, agg)
    for c in ["PR_mean", "SC_mean", "PA_mean"]:
        if c in agg.columns:
            sd = agg[c].std(ddof=0)
            agg[c.replace("_mean", "_z")] = (agg[c] - agg[c].mean()) / (sd if sd else 1)

    # ── 수비 지표와 공격 지표는 합치지 않는다 ────────────────────────
    #  SC·PR 은 팀이 수비 중일 때, PA 는 공격 중일 때 나온다. 서로 다른
    #  프레임에서 나온 서로 다른 능력이라 하나의 점수로 묶으면 해석이 안 된다.
    #  실측에서도 SC 와 PA 는 강하게 음의 상관을 보였다(뒤로 설 수록 SC 는 높고
    #  PA 는 낮다). 그래서 수비는 DPI, 공격은 PA 로 따로 보고한다.
    #
    #  DPI 의 가중치는 0.5 / 0.5 로 고정한다. SC 가 PR 보다 중요하다는
    #  선험적 근거가 없고, 기존 선수 순위에 맞춰 조정하면 "이미 잘한다고
    #  알려진 선수가 높게 나오도록 식을 맞춘" 연구가 되기 때문이다.
    if {"SC_z", "PR_z"} <= set(agg.columns):
        agg["DPI"] = (W_SC * agg.SC_z + W_PR * agg.PR_z).round(4)
    if "PA_z" in agg.columns:
        agg["PA_index"] = agg.PA_z.round(4)
    agg = agg.sort_values("DPI", ascending=False)

    print(f"\n[선수별 결과]  DPI = {W_SC}·z(SC) + {W_PR}·z(PR)  (수비)  /  PA_index (공격)")
    if {"SC_z", "PR_z"} <= set(agg.columns) and len(agg) > 3:
        from scipy.stats import spearmanr as _sp
        rho = _sp(agg.SC_z, agg.PR_z).statistic
        print(f"           SC 와 PR 의 순위 상관 ρ={rho:.3f} "
              f"— 낮을수록 서로 다른 것을 재고 있다는 뜻")
        print("           w 를 바꿨을 때 순위가 얼마나 흔들리는지:", end=" ")
        base = W_SC * agg.SC_z + W_PR * agg.PR_z
        worst = min(_sp(base, w * agg.SC_z + (1 - w) * agg.PR_z).statistic
                    for w in (0.0, 0.25, 0.5, 0.75, 1.0))
        print(f"최소 ρ={worst:.3f}")
        if worst < 0.9:
            print("           ! w 선택이 순위를 바꾼다. 0.5/0.5 를 '검증 전 기준선'으로만"
                  " 쓰고, 결과 라벨이 생기면 반드시 검증할 것.")
    print(agg.to_string())

    with pd.ExcelWriter(SAVE_XLSX, engine="openpyxl") as w:
        agg.reset_index().to_excel(w, sheet_name="선수별", index=False)
        M.to_excel(w, sheet_name="프레임별", index=False)
        F.to_excel(w, sheet_name="프레임메타", index=False)
        if len(PAdf):
            PAdf.to_excel(w, sheet_name="PA프레임별", index=False)
        if len(passes):
            passes.to_excel(w, sheet_name="검출된패스", index=False)
    print(f"\n[저장] {SAVE_XLSX}  (시트 3개: 선수별 / 프레임별 / 프레임메타)")

    try:
        plot(L1, F, SCdf, agg, teams)
    except Exception as e:
        print("  (그림 생략:", e, ")")

    if IN_COLAB:
        try:
            from google.colab import files
            files.download(SAVE_XLSX)
        except Exception:
            pass
    return M, agg, F


def plot(L1, F, SCdf, agg, teams):
    import matplotlib
    if not IN_COLAB:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mp
    if not len(SCdf):
        return
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))

    a = agg.sort_values("SC_mean")
    y = np.arange(len(a))
    nz = lambda s: s / s.max() if s.max() else s
    ax[0].barh(y - .2, nz(a.SC_mean), height=.38, color="#2A6699", label="SC")
    ax[0].barh(y + .2, nz(a.PR_mean), height=.38, color="#D98F3C", label="PR")
    ax[0].set_yticks(y)
    ax[0].set_yticklabels([f"{i} ({t})" for i, t in zip(a.index, a.team)], fontsize=8)
    ax[0].set_xlabel("normalized"); ax[0].set_title("Per-player SC vs PR")
    ax[0].legend(fontsize=8); ax[0].grid(axis="x", alpha=.3)

    fr = int(SCdf.frame.median())
    m = F.set_index("frame").loc[fr]
    g = L1[L1.frame == fr]
    att_t = m.poss_team
    def_t = teams[0] if att_t == teams[1] else teams[1]
    att = g[g.team == att_t][["X", "Y"]].to_numpy()
    dfd = g[g.team == def_t][["X", "Y"]].to_numpy()
    grid, area = make_grid(1.0)
    dA, _ = cKDTree(att).query(grid, k=1)
    dD, _ = cKDTree(dfd).query(grid, k=1)
    ny, nx = len(np.arange(.5, PITCH_W, 1.)), len(np.arange(.5, PITCH_L, 1.))
    ax[1].imshow((dD < dA).reshape(ny, nx), origin="lower",
                 extent=[0, PITCH_L, 0, PITCH_W], cmap="coolwarm_r", alpha=.55, aspect="equal")
    ax[1].add_patch(mp.Rectangle((0, 0), PITCH_L, PITCH_W, fill=False, ec="w", lw=1.5))
    ax[1].plot([PITCH_L/2]*2, [0, PITCH_W], color="w", lw=1.2)
    ax[1].add_patch(mp.Circle((PITCH_L/2, PITCH_W/2), 9.15, fill=False, ec="w", lw=1.2))
    ax[1].scatter(*att.T, c="#B94E39", s=45, ec="w", zorder=3, label=f"attack ({att_t})")
    ax[1].scatter(*dfd.T, c="#2A6699", s=45, ec="w", zorder=3, label=f"defend ({def_t})")
    ax[1].scatter([m.ball_x], [m.ball_y], c="k", s=30, zorder=4, label="ball")
    ax[1].set_title(f"Space control, frame {fr}  (blue = defended)")
    ax[1].legend(fontsize=8, loc="upper right"); ax[1].axis("off")
    plt.tight_layout()
    plt.show() if IN_COLAB else plt.savefig("sc_pr_plot.png", dpi=110, bbox_inches="tight")
    if not IN_COLAB:
        print("  그림 저장: sc_pr_plot.png")


M, agg, F = main()
