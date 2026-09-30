# =====================================================================
#  축구 선수 비가시적 기여도 — SC / PR / PA / DPI / IPI
#  「방법론 — 지표 설계의 이론적 배경, 가중치, 환산 절차」 사양 구현판 (v8)
#
#  이 파일은 방법론 문서의 수식 1~19를 그대로 구현한다.
#  v7 에서 문서와 달라져 있던 부분을 전부 문서 값으로 되돌렸고,
#  달라졌던 이유가 있는 것은 껐다 켤 수 있는 스위치로 남겨 두었다(§SPEC).
#
#  내는 값 (수식 번호는 방법론 문서 기준)
#    SC    공간 통제   수비 중, 위험 가중 공간을 Shapley 로 배분        수식 10~13
#    PR    압박 기여   거리 + 접근속도                                  수식 5, 6
#    PA    패스 유인   길목 × 수신공간 × 전진가치                       수식 7 계열
#    DPI   수비 종합   0.5·z(SC) + 0.5·z(PR)                            수식 15
#    DEF_z 수비 종합을 다시 표준화                                      수식 16
#    OFF_z z(z(PA)) × 신뢰도 ρ,  ρ = n_PA/(n_PA+K)                      수식 16, 17
#    IPI   최종 점수   100 + 15·z( ½DEF_z + ½OFF_z )                    수식 18
#
#  v7 과 달라진 점 (전부 문서에 맞춘 것)
#    · W_LANE 3.0 → 2.0 m          (문서 §4 B등급)
#    · PA_SPACE_R 8.0 → 5.0 m      (문서 §4 D등급)
#    · SPEED_MAX 11.0 → 12.0 m/s   (문서 §4 B등급)
#    · PA 전진가치를 '골대까지 거리 감소' → '공격 방향 전진 이득 Δx' (수식 7-3)
#    · PR 속도항에서 근접도 곱을 제거  (수식 5 그대로)
#    · 최종 통합을 TC(1/3씩) → IPI(상황별 ½ : ½ + 신뢰도)  (수식 16~18)
#    · 정규화 기본값을 rank → z       (수식 14)
#    · 지수 학습 최소 표본 200 → 30   (문서 §6.3)
#    · 호모그래피(수식 8)와 발밑 좌표(수식 9) 추가 — 문서 §8 '할 일' 1순위
#    · DEF_z–OFF_z 평면 유형 출력 추가 — 문서 §6.4 가 반드시 함께 보라고 요구
#
#  읽는 순서
#    1) [검증]      수식이 문서의 숫자 예시를 재현하는지
#    2) [상수표]    무엇이 A/B/C/D 등급인지 (문서 §4)
#    3) [읽기]      좌표 품질 — 여기서 경고가 나오면 아래 숫자는 믿을 게 못 된다
#    4) [지표 진단] 각 성분이 실제로 정보를 나르고 있는지
#    5) [최종 점수] IPI 와 DEF–OFF 유형
# =====================================================================

# ─────────────────── 설정 (여기만 고치면 된다) ───────────────────
#  상수 옆의 [A]~[D] 는 방법론 문서 §4의 근거 등급이다.
#    [A] 이론에서 유일하게 나온 값 — 고를 여지가 없음
#    [B] 문헌·경기 규칙에서 가져온 값
#    [C] 이 연구의 데이터·실험으로 정한 값
#    [D] 그냥 정한 값 — 아직 검증되지 않음 (문서가 11개라고 밝힌 것들)
#  실행하면 [상수표] 에 등급별로 다시 찍힌다.

EXCEL_PATH   = None      # None 이면 코랩 업로드 창이 뜬다. .xlsx / .csv 둘 다 읽는다
SHEET_TRACKS = 0         # 선수 좌표 시트 (이름 또는 0,1,2...). CSV 면 무시
SHEET_BALL   = None      # 공 좌표가 따로 있는 시트. 없으면 None (자동으로 찾음)

# ── 입력이 추적 프로그램의 tracks.csv 일 때 ─────────────────────────
#  tracker.py 가 내는 tracks.csv 는 좌표가 '화면 픽셀'이다. 문서 §7 한계 1 이
#  바로 이것이고, §8 '앞으로 할 일' 1순위가 "4점 호모그래피를 적용하고
#  발밑 좌표를 쓰기" 다. 아래 네 점을 채우면 이 파일이 그 일을 대신 한다.
#
#  HOMOGRAPHY_SRC : 영상 화면에서 읽은 픽셀 좌표 4점 이상
#  HOMOGRAPHY_DST : 그 점들의 실제 경기장 좌표 (미터, 왼쪽 아래가 원점)
#  네 점은 서로 한 직선 위에 있으면 안 되고, 넓게 퍼질수록 정확하다.
#  쓸 만한 기준점 — 경기장 네 모서리, 페널티박스 네 모서리, 골에어리어 모서리,
#                  하프라인과 터치라인이 만나는 두 점, 센터서클 좌우 끝
#
#  예 (1920x1080 전술 카메라에서 경기장 네 모서리를 읽은 경우)
#     HOMOGRAPHY_SRC = [(412, 880), (1503, 872), (1740, 402), (188, 410)]
#     HOMOGRAPHY_DST = [(0, 0), (105, 0), (105, 68), (0, 68)]
HOMOGRAPHY_SRC = None    # None 이면 변환하지 않는다 (이미 미터 좌표인 입력)
HOMOGRAPHY_DST = None
PLAYER_POINT = "foot"    # 선수 대표점 (수식 9). "foot" = 발밑 중앙 (fx,fy)
                         #   "center" = 박스 중심. 호모그래피는 지면 평면 변환이라
                         #   공중에 뜬 점(배꼽)을 넣으면 좌표가 뒤로 밀린다.
BALL_POINT   = "center"  # 공은 지면에 안 붙어 있으므로 박스 중심을 쓴다
DROP_REFEREE = True      # class_name 이 referee 인 행을 뺀다 (골키퍼는 남긴다)
TEAM_NAMES   = None      # tracker 의 team 0/1 에 붙일 이름. 예 {"0":"Home","1":"Away"}

FPS          = None      # None 이면 시간 열 간격에서 자동 추정
PITCH_L, PITCH_W = 105.0, 68.0     # 경기장 가로(골대~골대), 세로 (m)  [B] FIFA 권장
                                   #   7v7 유소년 = 보통 60~68 x 40~47
                                   #   ★ 반드시 실제 규격으로 바꿀 것
SCALE_TO_PITCH = True              # 수식 19. 거리 비례 상수만 √(L/105 × W/68) 로 줄인다
COORD_ORIGIN = "auto"    # "corner" / "center" / "auto"
TEAM_A_ATTACKS_PLUS_X = "auto"     # 첫 번째 팀이 +X 로 공격하는가. "auto" / True / False

OUT_OF_PITCH_MARGIN = 3.0          # 경계에서 이보다 밖에 있는 '사람' 행은 버린다 (m)
BALL_OUT_MARGIN = 1.0    # 공이 경계에서 이보다 밖이면 데드볼로 본다 (m)

REPAIR_IDS   = "auto"    # [C] 물리적으로 불가능한 속도가 나오면 ID 재매칭
REPAIR_SLACK_M = 2.0     # ID 재연결에서 허용할 좌표 흔들림 여유 (m)

DELTA        = 3         # [C] 중앙차분 폭 (프레임). 수식 6
SPEED_MAX    = 12.0      # [B] 이 속도(m/s) 넘으면 속도 관련 값을 무효 처리
                         #     문서 §4 B등급. 사람의 단거리 최고 속도대.
POSS_RADIUS  = None      # 공에서 이 거리(m) 안이어야 소유자. None 이면 자동
HOLD_FRAMES  = None      # [C] 소유팀 전환 인정 유지 프레임. None 이면 1.5초
TEAM_SIZE    = 11        # 한 팀 인원 (골키퍼 포함). 7v7 이면 7
MIN_TRACKED  = None      # [C] 양 팀 최소 추적 인원. None 이면 정원의 80%
BALL_MAX_GAP = None      # [C] 공 좌표 보간 상한. None 이면 0.2초
PLAYER_MAX_GAP = None    # [C] 선수 좌표 보간 상한. None 이면 0.3초

# ── GUI 연결용 갈고리 (터미널·코랩에서는 건드릴 필요 없다) ──────────
PROGRESS_CB = None       # f(단계이름, 현재, 전체) 를 받는 함수. 진행률 표시용
STOP_CB = None           # () -> True 이면 계산을 중간에 멈춘다

ROLE_GK_IDS  = set()     # 역할 열에서 읽어낸 골키퍼 id (read_data 가 채운다)
BALL_REJECTED = (0, 0)   # (믿은 공 프레임, 물리 검사로 버린 공 프레임)

# ── 공 검출 품질 (문서에 없는 구현 방어. 끄려면 0) ────────────────
BALL_MIN_CONF = 0.0      # 공 신뢰도 열이 있으면 이 값 미만은 버린다
                         #   tracker.py 의 보간 공은 conf=0 이므로, 0.01 로 두면
                         #   '메운 공'만 빠진다 (문서 §3.1 ③의 취지)
BALL_SPEED_MAX = 40.0    # 공이 이 속도(m/s)를 넘게 순간이동하면 그 프레임은 안 믿는다
BALL_FROZEN_M = 0.3      # 이만큼도 안 움직인 상태가
BALL_FROZEN_S = 2.0      # 이 시간(초) 넘게 이어지면 '얼어붙은 공'으로 본다
AUDIT_MIN_GAP_S = 5.0    # 표본검사에서 뽑는 장면들 사이의 최소 간격(초)
BALL_ID_BASE = 1_000_000 # track_id 가 이 값 이상이면 공으로 본다
                         #   tracker.py 의 ball_id_base 와 같은 값이다
BALL_AIRBORNE_Z = 1.5    # 공 높이 열이 있을 때, 이보다 높으면 공중볼로 본다

# ── SC : 공간 통제 (수식 10~13) ───────────────────────────────────
GRID_STEP    = 1.0       # [C] SC 격자(m). 0.5 와 비교해 SC 차이 1% 미만, 시간 4배
LAMBDA_GOAL  = 18.0      # [D] 위험도 골대 감쇠 거리. 감도분석 필요
LAMBDA_BALL  = 20.0      # [D] 위험도 공 감쇠 거리. 감도분석 필요

# ── PR : 압박 기여 (수식 5, 6) ────────────────────────────────────
PR_R         = 10.0      # [D] 압박 유효 반경
PR_VMAX      = 5.0       # [B] 접근속도 정규화 기준
PR_WD, PR_WV = 0.6, 0.4  # [D] 거리 : 접근속도
PR_PD_POWER  = 2         # [D] 거리항 지수. 수식 5 의 제곱

# ── PA : 패스 유인 (수식 7 계열) ──────────────────────────────────
W_LANE       = 2.0       # [B] 길목 차단 판정 폭. 선수 몸 폭 + 팔다리가 닿는 범위
PA_SPACE_R   = 5.0       # [D] 수신공간 자유 판정 거리
PA_PI_MIN    = 0.3       # [D] 전진가치 하한 (완전한 백패스도 이만큼은 준다)
PA_FWD_REF   = 30.0      # [D] 전진 이득 정규화 기준 (m)
PA_LEARN     = True      # 실제 패스에서 지수 a,b,c 를 학습할지 (수식 7-4)
PA_MIN_PASSES = 30       # [C] 문서 §6.3 — 이보다 적으면 학습 자체를 시도하지 않는다
PA_MIN_TEST_PASSES = 8   # 그 중 검증(hold-out)에 들어갈 패스의 최소 개수.
                         #   문서는 이 조건을 따로 두지 않았다. 30개로 학습하면
                         #   70/30 분할에서 검증이 9개뿐이라 이 값을 10 이상으로
                         #   두면 문서의 30개 문턱이 영영 닿지 않는다. 8 로 둔다.
PASS_MAX_GAP_S = 3.0     # 두 소유 구간이 이보다 벌어지면 한 번의 패스로 안 본다 (초)

# ── 합성 : DPI · DEF_z · OFF_z · IPI (수식 15~18) ─────────────────
W_SC, W_PR   = 0.5, 0.5  # [D] DPI 가중치. 미리 알 방법이 없어 대칭값
W_DEF, W_OFF = 0.5, 0.5  # [D] IPI 가중치. 상황별로 묶고 두 상황을 반반
RELIABILITY_K = 50       # [D] 신뢰도 상수 K (수식 17). 10 fps 기준 5초치
IPI_CENTER, IPI_SCALE = 100.0, 15.0   # 수식 18 뒷부분. 눈금일 뿐 의미는 없다
NORM_METHOD  = "z"       # 수식 14. "z" 가 문서 값. "rank" 는 분포에 안 휘둘리는 대안
MIN_FRAMES_RATIO = 0.2   # 팀 중앙값 대비 이 비율보다 적게 관측된 선수는 순위에서 뺀다
GK_IDS       = None      # 골키퍼 track_id 목록. None 이면 자동 추정
GK_MAX_DIST  = 22.0      # 자기 골대까지 평균거리가 이보다 작은 '팀당 한 명'을 골키퍼로
                         #   문서 §7 한계 6 — 포지션 라벨이 없어 임시 방편이다
EXCLUDE_GK   = True      # 골키퍼를 최종 순위에서 뺄지
POS_ADJUST   = True      # 자기 골대까지 거리로 회귀한 잔차(IPI_adj)를 함께 낸다

# ── 문서와 다르게 갈 때만 켠다 (전부 False 가 문서 그대로) ─────────
SPEC_PR_PV_PROXIMITY = False   # True 면 PR 속도항에 근접도를 곱한다.
                               #   문서 수식 5 에는 없다. v7 이 넣었던 이유 —
                               #   모사 경기에서 25m 밖에서 쌓인 값이 PR 총량의
                               #   50% 였고 실제 기여가 거리 18 : 속도 82 로 갈렸다.
                               #   문서를 따를 때는 False 로 두고, [지표 진단] 의
                               #   '25m 밖 몫' 을 보고 판단할 것.
SPEC_PA_PROG_GOALDIST = False  # True 면 전진가치를 '골대까지 거리 감소'로 잰다.
                               #   문서 수식 7-3 은 공격 방향 전진 이득 Δx 다.
                               #   Δx 는 측면 전환 패스를 전진 0 으로 깎는다.

MAX_FRAMES   = None      # 테스트할 때 앞쪽 N 프레임만. 전체면 None
SAVE_XLSX    = "ipi_result.xlsx"
SAVE_PLOT    = "ipi_plot.png"   # 그림 파일 경로
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
#          LAMBDA_GOAL, LAMBDA_BALL, PR_R, PA_FWD_REF, GK_MAX_DIST
#    (B) 비례하지 않는 것 — 사람 몸과 공의 크기·속도라서 구장이 작아져도 그대로다
#          SPEED_MAX, PR_VMAX, W_LANE, PA_SPACE_R, POSS_RADIUS,
#          OUT_OF_PITCH_MARGIN, BALL_OUT_MARGIN, BALL_AIRBORNE_Z, REPAIR_SLACK_M
#  둘을 같이 줄이면 "작은 구장에서는 사람이 느리게 뛴다"는 말이 되어버린다.
_FULL_L, _FULL_W = 105.0, 68.0
PITCH_SCALE = 1.0
#  원래 값을 따로 보관한다. 경기장 크기를 바꿔 다시 적용할 때
#  이미 줄어든 값에 또 곱하면 두 번 줄어들기 때문이다 (GUI 에서 실제로 일어난다).
_BASE_SCALED = {"LAMBDA_GOAL": LAMBDA_GOAL, "LAMBDA_BALL": LAMBDA_BALL,
                "PR_R": PR_R, "PA_FWD_REF": PA_FWD_REF, "GK_MAX_DIST": GK_MAX_DIST}


def apply_pitch_scale():
    """
    수식 19 · 구장 크기 보정.  조정된 상수 = 원래 상수 × √(L/105 × W/68)

    비례해서 줄이는 것은 '전술적 지리'에 대한 값뿐이다
    (LAMBDA_GOAL, LAMBDA_BALL, PR_R, PA_FWD_REF, 그리고 골키퍼 판정 거리).
    SPEED_MAX·PR_VMAX·W_LANE·PA_SPACE_R·POSS_RADIUS 는 사람 몸과 움직임에
    대한 값이라 그대로 둔다 — 같이 줄이면 "작은 구장에서는 사람이 느리게
    뛴다"는 이상한 주장이 된다 (문서 §4).

    여러 번 불러도 결과가 같다. 매번 원래 값에서 다시 계산하기 때문이다.
    """
    global PITCH_SCALE, LAMBDA_GOAL, LAMBDA_BALL, PR_R, PA_FWD_REF, GK_MAX_DIST
    PITCH_SCALE = (float(np.sqrt((PITCH_L / _FULL_L) * (PITCH_W / _FULL_W)))
                   if SCALE_TO_PITCH else 1.0)
    k = PITCH_SCALE if abs(PITCH_SCALE - 1.0) > 0.02 else 1.0
    LAMBDA_GOAL = _BASE_SCALED["LAMBDA_GOAL"] * k
    LAMBDA_BALL = _BASE_SCALED["LAMBDA_BALL"] * k
    PR_R        = _BASE_SCALED["PR_R"] * k
    PA_FWD_REF  = _BASE_SCALED["PA_FWD_REF"] * k
    GK_MAX_DIST = _BASE_SCALED["GK_MAX_DIST"] * k
    return PITCH_SCALE


apply_pitch_scale()

IN_COLAB = "google.colab" in sys.modules


def _progress(stage, cur, total):
    """진행 상황을 GUI 로 흘려보낸다. 콜백이 없으면 아무 일도 안 한다."""
    if PROGRESS_CB is not None:
        try:
            PROGRESS_CB(stage, int(cur), int(total))
        except Exception:      # 표시가 실패해도 계산은 계속돼야 한다
            pass


def _stopped():
    """GUI 에서 중지를 눌렀는지."""
    if STOP_CB is None:
        return False
    try:
        return bool(STOP_CB())
    except Exception:
        return False


class Stopped(Exception):
    """사용자가 중지를 눌렀을 때."""


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
    "conf":     ["conf", "confidence", "score", "prob", "probability", "신뢰도", "확률"],
    "role":     ["role", "역할", "position_role", "obj_class", "objclass"],
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


def looks_centered(x, y):
    """
    좌표 원점이 경기장 중앙인지 판정한다.

    구석 원점이면 좌표의 중앙값이 경기장 한가운데(52.5, 34) 근처이고,
    중앙 원점이면 0 근처다. 최소값으로 판정하면 안 된다 — 스로인을 차려고
    라인 밖에 선 선수나 터치라인 밖 감독 한 명이면 음수 좌표가 생겨서
    판정이 뒤집히고, 전체 좌표가 반 경기장만큼 밀린 채로 조용히 계산된다.
    """
    return (abs(float(np.median(x))) < PITCH_L * 0.25
            and abs(float(np.median(y))) < PITCH_W * 0.25)


def find_height_col(df):
    """
    공 높이 열을 찾는다. 'ball_z' 로 인식된 열이 우선이고, 없으면 이름이 그냥
    'z' 인 열을 쓴다.

    'z' 를 ALIAS 에 넣지 않는 이유: 선수 표의 z 는 높이가 아닌 다른 것일 수
    있다. 하지만 공만 있는 시트나 공 행에서의 z 는 높이로 봐도 된다.
    이 함수는 그 두 곳에서만 부른다.
    """
    if "ball_z" in df.columns:
        return "ball_z"
    for c in df.columns:
        if _norm(c) == "z":
            return c
    return None


BALL_WORDS = ("ball", "공", "볼")

#  검출기가 역할을 따로 내주면 그것을 쓴다. 추정보다 항상 낫다.
#    검출 파이프라인이 선수/심판/골키퍼를 별개 클래스로 내보내는 경우가 있는데,
#    예전 필터는 'person|player|선수' 만 남겨서 '골키퍼'·'goalkeeper' 행을
#    통째로 버렸다. 그러면 팀당 인원이 한 명씩 줄고(경고도 안 뜬다),
#    골문 앞 공간을 골키퍼 대신 센터백이 받아 SC 가 부풀려진다.
GK_WORDS   = ("골키퍼", "goalkeeper", "goalie", "keeper", "gk", "수문장")
REF_WORDS  = ("심판", "부심", "주심", "대기심", "referee", "refere", "ref",
              "umpire", "linesman", "assistant referee", "official")
PLAYER_WORDS = ("선수", "player", "person", "outfield")


def _match_words(series, words):
    """열 값이 주어진 단어 중 하나를 담고 있는 행을 찾는다."""
    s = series.astype(str).str.lower().str.strip()
    m = pd.Series(False, index=series.index)
    for w in words:
        m = m | s.str.contains(_re.escape(w.lower()), na=False, regex=True)
    return m


def classify_roles(tracks):
    """
    역할 열(cls/kind/role)에서 골키퍼·심판·선수를 가려낸다.

    반환: (선수로 볼 행 마스크, 골키퍼 track_id 집합, 심판 행 수, 쓴 열 이름)
    아무 열도 역할을 담고 있지 않으면 (전부 True, 빈 집합, 0, None) 이다.
    """
    for col in ("role", "cls", "kind", "position"):
        if col not in tracks.columns:
            continue
        is_gk = _match_words(tracks[col], GK_WORDS)
        is_ref = _match_words(tracks[col], REF_WORDS)
        is_pl = _match_words(tracks[col], PLAYER_WORDS)
        # 'referee' 안에 'ref' 가 들어가듯 단어가 서로 겹치므로 골키퍼를 우선한다.
        is_ref = is_ref & ~is_gk
        if not (is_gk.any() or is_ref.any()):
            continue                       # 이 열은 역할 정보가 아니다
        keep = (is_pl | is_gk) & ~is_ref
        if not keep.any():                 # 선수 단어가 아예 없으면 심판만 뺀다
            keep = ~is_ref
        gk_ids = set(tracks.loc[is_gk & keep, "track_id"].unique()) \
            if "track_id" in tracks.columns else set()
        return keep, gk_ids, int(is_ref.sum()), col
    return pd.Series(True, index=tracks.index), set(), 0, None


def _is_ball(series):
    """
    어느 열에 있든 'ball' 로 표시된 행을 찾는다.

    track_id 가 100만 이상인 것도 공으로 본다. 공 전용 모델을 따로 돌리는
    파이프라인은 선수 추적기의 번호와 겹치지 않게 공 id 를 100만번대부터
    매기는데, 그 경우 클래스 이름이 따로 안 붙어 오기도 한다.
    """
    s = series.astype(str).str.lower().str.strip()
    m = pd.Series(False, index=series.index)
    for w in BALL_WORDS:
        m = m | (s == w) | s.str.fullmatch(rf"{w}\d*", na=False)
    num = pd.to_numeric(series, errors="coerce")
    return m | (num >= BALL_ID_BASE).fillna(False)


def solve_homography(src, dst):
    """
    수식 8 · 호모그래피 행렬 H 를 4점 이상의 대응에서 구한다 (DLT + SVD).

    H 에는 정해야 할 값이 8개 있고 짝 하나가 방정식 2개(X, Y)를 주므로
    4개면 딱 떨어진다. 그보다 많이 주면 최소제곱으로 푼다.
    cv2 를 쓰지 않는 이유는 이 파일 하나만으로 돌게 하기 위해서다.
    """
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    if len(src) != len(dst) or len(src) < 4:
        raise SystemExit("호모그래피에는 같은 개수의 점이 4쌍 이상 필요하다 "
                         f"(지금 src {len(src)}개, dst {len(dst)}개).")
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, Vt = np.linalg.svd(np.asarray(A, float))
    H = Vt[-1].reshape(3, 3)
    if abs(H[2, 2]) < 1e-12:
        raise SystemExit("호모그래피를 구하지 못했다. 네 점이 한 직선 위에 있지 않은지 확인할 것.")
    return H / H[2, 2]


def apply_homography(H, x, y):
    """픽셀 좌표 (x, y) 를 경기장 미터 좌표로 옮긴다."""
    pts = np.stack([np.asarray(x, float), np.asarray(y, float),
                    np.ones(len(x))], axis=0)            # (3, N)
    out = H @ pts
    w = np.where(np.abs(out[2]) < 1e-12, np.nan, out[2])
    return out[0] / w, out[1] / w


def prepare_tracker_csv(df, verbose=True):
    """
    추적 프로그램(tracker.py)이 내는 tracks.csv 를 이 파일이 읽을 수 있게 맞춘다.

    그 CSV 는 열 이름과 규약이 다르다.
      · 위치가 x1..y2 / cx,cy / fx,fy 로 나뉘어 있고 X, Y 가 없다
      · 종류가 class_name (player / goalkeeper / referee / ball)
      · 팀이 0 / 1 / -1(심판·골키퍼) / 빈칸(공)
      · 좌표 단위가 화면 픽셀

    여기서는 앞의 세 가지만 맞춘다. 단위 변환(수식 8)은 호출한 쪽에서 한다.
    선수는 발밑 중앙(수식 9), 공은 박스 중심을 대표점으로 쓴다 —
    호모그래피는 지면 평면 변환이라 공중에 뜬 점을 넣으면 좌표가 뒤로 밀린다.
    """
    cols = set(df.columns)
    if not ({"fx", "fy"} <= cols or {"cx", "cy"} <= cols) or "X" in cols:
        return df                       # 이미 X, Y 가 있는 평범한 입력
    df = df.copy()
    if "class_name" in cols and "cls" not in cols:
        df = df.rename(columns={"class_name": "cls"})   # classify_roles 가 읽는다
    name = df["cls"].astype(str).str.lower() if "cls" in df.columns else None
    is_ball = _is_ball(df["track_id"]) if "track_id" in df.columns else None
    if name is not None:
        is_ball = (is_ball | name.str.contains("ball", na=False)) if is_ball is not None \
            else name.str.contains("ball", na=False)
    if is_ball is None:
        is_ball = pd.Series(False, index=df.index)

    def pick(kind):
        pref = ("fx", "fy") if kind == "foot" else ("cx", "cy")
        alt = ("cx", "cy") if kind == "foot" else ("fx", "fy")
        return pref if pref[0] in df.columns else alt

    px, py = pick(PLAYER_POINT)
    bx, by = pick(BALL_POINT)
    df["X"] = np.where(is_ball, df.get(bx, np.nan), df.get(px, np.nan))
    df["Y"] = np.where(is_ball, df.get(by, np.nan), df.get(py, np.nan))
    if verbose:
        print(f"  tracker.csv 형식 인식: 선수 대표점 ({px},{py}) · 공 ({bx},{by}) -> X, Y")
    return df


def resolve_teams(tracks, verbose=True):
    """
    team 열을 정확히 두 팀으로 정리한다.

    tracker.py 는 팀을 0 / 1 / -1 로 적고, -1 에는 심판과 골키퍼가 같이 들어간다.
    심판은 classify_roles 가 이미 뺐으므로 여기 남은 -1 은 대개 골키퍼다.
    골키퍼를 통째로 버리면 골문 앞 공간을 센터백이 대신 받아 SC 가 부풀려지므로
    버리지 않고, '어느 팀 무리에 더 가까이 있었나'로 팀을 붙여 준다.
    """
    tracks = tracks.copy()
    # 숫자로 읽힌 팀 값(0, 1, -1)이 '0.0' 처럼 되는 것을 막는다.
    # 그대로 두면 TEAM_NAMES 의 키와 안 맞고, 화면에도 0.0 / 1.0 으로 찍힌다.
    t = pd.to_numeric(tracks["team"], errors="coerce")
    tracks["team"] = np.where(t.notna() & (t == t.round()),
                              t.fillna(0).astype("Int64").astype(str),
                              tracks["team"].astype(str).str.strip())
    counts = tracks["team"].value_counts()
    named = [t for t in counts.index
             if t not in ("", "-1", "nan", "none", "None")
             and not any(k in t.lower() for k in ("ref", "심판", "unknown"))]
    if len(named) < 2:
        raise SystemExit(f"팀을 두 개로 못 나눴다 (찾은 것: {list(counts.index)}). "
                         "team 열을 확인할 것.")
    teams = list(counts[named].sort_values(ascending=False).index[:2])
    rest = ~tracks["team"].isin(teams)
    if rest.any():
        cen = {t: tracks.loc[tracks.team == t, ["X", "Y"]].mean().to_numpy(float)
               for t in teams}
        moved = 0
        for tid, g in tracks[rest].groupby("track_id", sort=False):
            pos = g[["X", "Y"]].mean().to_numpy(float)
            near = min(teams, key=lambda t: float(np.hypot(*(pos - cen[t]))))
            tracks.loc[g.index, "team"] = near
            moved += 1
        if verbose:
            print(f"  팀이 안 붙어 있던 id {moved}개를 가까운 팀 무리로 배정 "
                  f"(대개 골키퍼다. 틀렸으면 GK_IDS 로 직접 지정할 것)")
    if TEAM_NAMES:
        tracks["team"] = tracks["team"].map(lambda t: TEAM_NAMES.get(str(t), t))
        teams = [TEAM_NAMES.get(str(t), t) for t in teams]
    return tracks, teams


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

        prev_f = fs[0]
        for f in fs[1:]:
            g = by_f[f]
            xy = g[["X", "Y"]].to_numpy(float)
            n = min(len(prev_xy), len(xy))
            if n == 0:
                prev_xy, prev_lbl, prev_f = xy, [f"{team}_x{k}" for k in range(len(xy))], f
                out.loc[g.index, "stable_id"] = prev_lbl
                continue
            C = np.linalg.norm(prev_xy[:, None, :] - xy[None, :, :], axis=2)
            r, c = linear_sum_assignment(C)
            # 사람이 그 사이에 갈 수 있는 거리를 넘는 연결은 받지 않는다.
            #   헝가리안 매칭은 '전체 이동거리 합'만 최소화하므로, 한 명이 화면
            #   밖으로 나가고 다른 곳에서 다른 사람이 나타나면 40 m 떨어진 둘을
            #   같은 사람으로 이어 버린다. 그러면 없던 이동거리가 생기고 PR 의
            #   접근속도가 통째로 오염된다.
            #   상한을 정확히 SPEED_MAX*dt 로 잡으면 안 된다. 25 fps 에서 그 값은
            #   0.44 m 인데, 검출 좌표의 흔들림만으로도 이걸 넘는 프레임이 흔해서
            #   멀쩡한 궤적이 토막난다(실측: 22명이 1980개 ID 로 부서졌다).
            #   여기서 막으려는 것은 '수십 m 순간이동'이지 정밀한 운동학이 아니므로
            #   좌표 흔들림 몫을 더해 넉넉히 잡는다. 같은 데이터에서 ID 는 22개로
            #   복구되고, 40 m 짜리 잘못된 연결은 여전히 거부된다.
            reach = SPEED_MAX * max(f - prev_f, 1) / FPS + REPAIR_SLACK_M
            lbl = [None] * len(xy)
            for ri, ci in zip(r, c):
                if C[ri, ci] <= reach:
                    lbl[ci] = prev_lbl[ri]
            for k in range(len(xy)):                    # 매칭 안 된 행은 새 ID
                if lbl[k] is None:
                    lbl[k] = f"{team}_new{f}_{k}"
            out.loc[g.index, "stable_id"] = lbl
            prev_xy, prev_lbl, prev_f = xy, lbl, f

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
    tracks = prepare_tracker_csv(tracks)

    # ── 수식 8 · 픽셀 -> 미터 (호모그래피) ───────────────────────
    #  문서 §7 한계 1 이 "호모그래피를 아직 안 썼다"이고 §8 할 일 1순위가 이것이다.
    #  카메라가 비스듬히 찍으면 화면 어디냐에 따라 거리 오차가 14~46% 난다.
    #  SC 와 PR 이 둘 다 거리 기반이라 그 오차가 지표 전체로 번진다.
    if HOMOGRAPHY_SRC and HOMOGRAPHY_DST:
        H = solve_homography(HOMOGRAPHY_SRC, HOMOGRAPHY_DST)
        # 되돌림 오차 — 기준점을 변환해 실제 위치와 비교한다. 네 점이 한 직선에
        # 가깝게 몰려 있으면 식은 풀려도 엉뚱한 변환이 나오는데 여기서 드러난다.
        _bx, _by = apply_homography(H, [p[0] for p in HOMOGRAPHY_SRC],
                                    [p[1] for p in HOMOGRAPHY_SRC])
        _e = float(np.max(np.hypot(_bx - np.array([p[0] for p in HOMOGRAPHY_DST], float),
                                   _by - np.array([p[1] for p in HOMOGRAPHY_DST], float))))
        print(f"  기준점 되돌림 오차 최대 {_e:.2f} m "
              f"({len(HOMOGRAPHY_SRC)}점)" + ("  <- 0.5 m 안쪽이면 좋다" if _e <= 0.5 else ""))
        if _e > 1.0:
            print(f"  ! 기준점이 잘 안 맞는다. 네 점이 한 직선에 가깝거나 클릭 위치가 "
                  f"실제 지점과 다를 수 있다. 보정을 다시 하는 편이 낫다.")
        bx0, by0 = tracks["X"].median(), tracks["Y"].median()
        tracks["X"], tracks["Y"] = apply_homography(H, tracks["X"], tracks["Y"])
        if {"ball_x", "ball_y"} <= set(tracks.columns):   # 공이 별도 열로 온 경우
            tracks["ball_x"], tracks["ball_y"] = apply_homography(
                H, tracks["ball_x"], tracks["ball_y"])
        n_bad = int(tracks[["X", "Y"]].isna().any(axis=1).sum())
        tracks = tracks.dropna(subset=["X", "Y"])
        print(f"  호모그래피 적용: 화면 중앙값 ({bx0:.0f}, {by0:.0f})px -> "
              f"({tracks.X.median():.1f}, {tracks.Y.median():.1f})m"
              + (f" · 변환 불가 {n_bad}행 제외" if n_bad else ""))
    elif "X" in tracks.columns and len(tracks):
        rng_x = float(tracks["X"].max() - tracks["X"].min())
        if rng_x > PITCH_L * 3:
            print("  ! 좌표가 미터가 아니라 화면 픽셀로 보인다. HOMOGRAPHY_SRC/DST 에")
            print("    경기장 기준점 4개를 넣으면 이 파일이 수식 8 로 변환한다.")

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
        # 높이 열은 어느 이름으로 오든 챙긴다. 예전에는 X/Y 로 된 공 시트에서
        # 높이를 통째로 버려서, 같은 데이터인데 입력 형식만 다르면 공중볼
        # 처리가 조용히 무력화됐다.
        zc = find_height_col(b)
        extra = ([zc] if zc else []) + (["conf"] if "conf" in b.columns else [])
        if {"ball_x", "ball_y"} <= set(b.columns):
            ball = b[["frame", "ball_x", "ball_y"] + extra]
        else:
            ball = b[["frame", "X", "Y"] + extra].rename(
                columns={"X": "ball_x", "Y": "ball_y"})
        if zc:
            ball = ball.rename(columns={zc: "ball_z"})
        how = "별도 시트"
    elif {"ball_x", "ball_y"} <= set(tracks.columns):
        bcols = ["ball_x", "ball_y"] + (["ball_z"] if "ball_z" in tracks.columns else [])
        ball = tracks.groupby("frame")[bcols].first().reset_index()
        tracks = tracks.drop(columns=bcols)   # conf 는 선수 행의 것이라 안 가져온다
        how = "ball_x / ball_y 열"
    else:
        # cls, team, track_id, kind 중 아무 열에서나 'ball' 표시를 찾는다
        for col in ("cls", "kind", "team", "track_id"):
            if col in tracks.columns:
                m = _is_ball(tracks[col])
                if m.any():
                    zc = find_height_col(tracks)        # 공 행의 z 는 높이다
                    take = ["X", "Y"] + ([zc] if zc else []) \
                           + (["conf"] if "conf" in tracks.columns else [])
                    ball = (tracks[m].groupby("frame")[take].first()
                            .rename(columns={"X": "ball_x", "Y": "ball_y"}).reset_index())
                    if zc:
                        ball = ball.rename(columns={zc: "ball_z"})
                        tracks = tracks.drop(columns=[zc])
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
    # 공 신뢰도가 있으면 낮은 것부터 버린다. 검출기가 내준 확신을 안 쓸 이유가 없다.
    if BALL_MIN_CONF > 0 and "conf" in ball.columns:
        n0 = len(ball)
        ball = ball[ball["conf"].fillna(1.0) >= BALL_MIN_CONF]
        if len(ball) < n0:
            print(f"  공 신뢰도 {BALL_MIN_CONF} 미만 {n0-len(ball)}프레임 제외 "
                  f"({(n0-len(ball))/max(n0,1)*100:.0f}%)")
    if "ball_z" in ball.columns:
        hi = float((ball["ball_z"].fillna(0) > BALL_AIRBORNE_Z).mean())
        print(f"  공 좌표: {how} 에서 {len(ball)} 프레임분 , 높이 열 있음 "
              f"(공중볼 {hi*100:.0f}% — 이 구간은 소유자 판정에서 빠진다)")
    else:
        print(f"  공 좌표: {how} 에서 {len(ball)} 프레임분 , 높이 열 없음 "
              f"(크로스·헤더 구간을 지상 최근접으로 판정하게 된다)")

    # ── 필수 열 ──────────────────────────────────────────────
    miss = {"frame", "track_id", "X", "Y"} - set(tracks.columns)
    if miss:
        raise SystemExit(f"필수 열이 없다: {miss}\n실제 열: {list(tracks.columns)}\n"
                         "ALIAS 딕셔너리에 실제 열 이름을 추가하면 인식된다.")

    global ROLE_GK_IDS
    keep, role_gks, n_ref, role_col = classify_roles(tracks)
    ROLE_GK_IDS = role_gks
    if role_col is not None:
        print(f"  역할 열 '{role_col}' 인식: 골키퍼 {len(role_gks)}명 / "
              f"심판 {n_ref}행 제외 / 남긴 행 {int(keep.sum()):,}")
        tracks = tracks[keep].copy()
    elif "cls" in tracks.columns:
        # 역할 정보가 없는 cls 열이면 예전처럼 사람 행만 남긴다.
        m = tracks["cls"].astype(str).str.lower()
        tracks = tracks[m.str.contains("person|player|선수", na=True)].copy()

    tracks = tracks.dropna(subset=["frame", "track_id", "X", "Y"])
    tracks["frame"] = tracks["frame"].astype(int)
    ball["frame"] = ball["frame"].astype(int)

    # ── 팀 ───────────────────────────────────────────────────
    if "team" not in tracks.columns:
        raise SystemExit("team 열이 없다. 선수마다 어느 팀인지 있어야 SC/PR 을 계산할 수 있다.")
    tracks["team"] = tracks["team"].astype(str).str.strip()
    tracks = tracks[~_is_ball(tracks["team"])]
    tracks, teams = resolve_teams(tracks)
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

    # 원점 판정은 looks_centered() 참고. '가장 바깥값'이 아니라 '데이터의 한가운데'로 한다.
    #   기존 규칙(xmin < -1 이면 중앙 원점)은 음수 좌표 하나만 있어도 뒤집힌다.
    #   스로인을 차려고 라인 밖에 선 선수, 터치라인 밖의 감독·사진기자가 딱
    #   그런 값이다. 한 번 잘못 판정하면 전체 좌표가 (+52.5, +34) 만큼 밀려
    #   모든 결과가 조용히 무의미해진다.
    #   구석 원점이면 좌표의 중앙값이 경기장 한가운데(52.5, 34) 근처이고,
    #   중앙 원점이면 0 근처다. 이건 이상치 몇 개로는 안 흔들린다.
    centered = (COORD_ORIGIN == "center") or \
               (COORD_ORIGIN == "auto" and looks_centered(tracks.X, tracks.Y))
    if centered:
        for df_ in (tracks, ball):
            df_[["X", "ball_x"][df_ is ball]] += PITCH_L / 2
            df_[["Y", "ball_y"][df_ is ball]] += PITCH_W / 2
        print(f"  좌표계: 중앙 원점으로 판단 -> (+{PITCH_L/2:.1f}, +{PITCH_W/2:.1f}) 평행이동")
    # ── 경기장 밖 사람 제거 ──────────────────────────────────
    #  감독·대기심·사진기자·안전요원은 검출기가 person 으로 잡지만 선수가 아니다.
    #  이들이 좌표에 남으면 프레임당 인원이 부풀고, SC 가 터치라인 밖 인원에게
    #  공간 점유를 나눠 준다. 스로인을 차는 선수는 실제로 라인 밖에 서므로
    #  여유(OUT_OF_PITCH_MARGIN)를 두고 그보다 먼 행만 버린다.
    if OUT_OF_PITCH_MARGIN > 0:
        m_ = OUT_OF_PITCH_MARGIN
        inside = (tracks.X.between(-m_, PITCH_L + m_)
                  & tracks.Y.between(-m_, PITCH_W + m_))
        if not inside.all():
            drop_ids = tracks.loc[~inside, "track_id"].value_counts()
            gone = [i for i, c in drop_ids.items()
                    if c > 0.5 * (tracks.track_id == i).sum()]
            print(f"  경기장 밖({m_:.0f} m 초과) {int((~inside).sum()):,}행 제거 "
                  f"— 대부분이 밖인 id {len(gone)}개는 선수가 아닐 것이다"
                  + (f": {gone[:8]}{' ...' if len(gone) > 8 else ''}" if gone else ""))
            tracks = tracks[inside].copy()

    # 규격 판단은 '사람 아닌 것'을 걸러낸 뒤에 한다. 벤치·사진기자석이 섞인 채로
    # 재면 데이터가 경기장보다 넓게 퍼진 것처럼 보여서, 멀쩡한 규격을 의심하게 된다.
    out_ratio = (~tracks.X.between(0, PITCH_L) | ~tracks.Y.between(0, PITCH_W)).mean()
    sx0 = tracks.X.max() - tracks.X.min()
    sy0 = tracks.Y.max() - tracks.Y.min()
    print(f"  좌표 범위 X {tracks.X.min():.1f}~{tracks.X.max():.1f} m , "
          f"Y {tracks.Y.min():.1f}~{tracks.Y.max():.1f} m  (경기장 밖 {out_ratio*100:.1f}%)")
    print(f"  데이터가 실제로 퍼진 범위 {sx0:.0f} x {sy0:.0f} m "
          f"(설정한 경기장 {PITCH_L:.0f} x {PITCH_W:.0f})")
    if out_ratio > 0.05:
        print(f"  ! 경기장 밖 비율이 높다. 설정한 경기장은 {PITCH_L}x{PITCH_W} m 인데 "
              f"데이터가 실제로 퍼진 범위는 약 {sx0:.0f}x{sy0:.0f} m 다.")
        print(f"    7v7 처럼 작은 경기장이면 PITCH_L, PITCH_W 와 TEAM_SIZE 를 실제 규격으로 "
              f"바꿀 것 (위험도 가중치와 골대 위치가 여기에 달려 있다).")

    if MAX_FRAMES:
        lim = sorted(tracks.frame.unique())[:MAX_FRAMES]
        tracks = tracks[tracks.frame.isin(lim)]
        ball = ball[ball.frame.isin(lim)]

    # ── 공격 방향 ────────────────────────────────────────────
    #  골키퍼는 자기 골대 앞에 상주하므로, 팀별로 '가장 낮은 X 에 상주하는 선수'를
    #  비교하면 어느 팀이 x=0 골대를 지키는지 알 수 있다.
    global TEAM_A_ATTACKS_PLUS_X
    def _lowest_x(part):
        lo = {}
        for t in teams:
            s = part[part.team == t].groupby("track_id")["X"].mean()
            lo[t] = float(s.min()) if len(s) else np.nan
        return lo

    lowest = _lowest_x(tracks)
    if any(np.isnan(v) for v in lowest.values()):
        # 한 팀이 통째로 비어 있으면 비교가 NaN 이 되고, NaN 비교는 조용히
        # False 가 되어 방향을 절반의 확률로 틀리게 잡는다. 그럴 땐 판정하지 않는다.
        guess = None
        print("  ! 한 팀의 좌표가 비어 있어 공격 방향을 판정할 수 없다. team 열을 확인할 것.")
        if TEAM_A_ATTACKS_PLUS_X == "auto":
            TEAM_A_ATTACKS_PLUS_X = True
    else:
        guess = lowest[teams[0]] < lowest[teams[1]]
    if guess is not None:
        if TEAM_A_ATTACKS_PLUS_X == "auto":
            TEAM_A_ATTACKS_PLUS_X = bool(guess)
            print(f"  공격 방향 자동 판정: {teams[0]} 가 x=0 골대를 지키고 +X 로 공격"
                  if guess else
                  f"  공격 방향 자동 판정: {teams[0]} 가 x={PITCH_L:.0f} 골대를 지키고 -X 로 공격")
            print(f"    (팀별 최소상주 X: {teams[0]} {lowest[teams[0]]:.1f} m , "
                  f"{teams[1]} {lowest[teams[1]]:.1f} m)")
        elif bool(TEAM_A_ATTACKS_PLUS_X) != bool(guess):
            print(f"  ! 설정한 공격 방향(TEAM_A_ATTACKS_PLUS_X={TEAM_A_ATTACKS_PLUS_X})이 "
                  f"좌표에서 추정한 방향과 반대다.")
            print(f"    팀별 최소상주 X: {teams[0]} {lowest[teams[0]]:.1f} m , "
                  f"{teams[1]} {lowest[teams[1]]:.1f} m -> 설정을 다시 확인할 것.")
    #  전·후반이 한 파일에 있으면 방향이 중간에 바뀐다. 그대로 돌리면 절반이
    #  반대 골대를 기준으로 계산되므로 반드시 나눠서 돌려야 한다.
    fr_mid = tracks.frame.median()
    halves = []
    for part in (tracks[tracks.frame <= fr_mid], tracks[tracks.frame > fr_mid]):
        lo = _lowest_x(part) if len(part) else {t: np.nan for t in teams}
        halves.append(None if any(np.isnan(v) for v in lo.values())
                      else lo[teams[0]] < lo[teams[1]])
    if len(halves) == 2 and None not in halves and halves[0] != halves[1]:
        print("  ! 앞구간과 뒷구간의 공격 방향이 반대다. 전·후반이 한 파일에 들어 있는 것으로 보인다.")
        print("    이대로 계산하면 절반이 반대 골대를 기준으로 잡혀 SC 와 PA 가 무의미해진다.")
        print("    전반과 후반을 따로 나눠서 돌릴 것.")

    n_per_team = tracks.groupby(["frame", "team"]).size().groupby("team").median()
    if MIN_TRACKED is None:
        MIN_TRACKED = max(3, int(np.ceil(TEAM_SIZE * 0.8)))
    print(f"  tracks {len(tracks):,}행 / 프레임 {tracks.frame.nunique():,} / "
          f"선수 {tracks.track_id.nunique()}명 / 팀 {teams} / fps {FPS}")
    print(f"  프레임당 인원 {n_per_team.to_dict()} -> MIN_TRACKED = {MIN_TRACKED} "
          f"(명목 {TEAM_SIZE}명의 80%)")
    cover = float(n_per_team.min()) / TEAM_SIZE
    if cover < 0.9:
        print(f"  ! 프레임당 추적 인원이 명목 인원({TEAM_SIZE}명)의 {cover*100:.0f}% 뿐이다. "
              f"화면 밖 선수가 빠졌거나, TEAM_SIZE 가 실제 경기 형식과 다른 것이다.")
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
        n_id = tracks.track_id.nunique()
        if n_id > 3 * TEAM_SIZE:
            print(f"     ! 복구 후 ID 가 {n_id}개다 (선수는 {2*TEAM_SIZE}명). 가림이 생길 때마다"
                  f" 새 ID 가 붙어 한 선수의 궤적이 조각나 있다는 뜻이다.")
            print(f"       조각난 ID 로는 '선수별' 표가 선수를 가리키지 않는다. 프레임 단위"
                  f" 결과(SC/PR/PA 분포)까지만 쓰고, 선수 순위는 믿지 말 것.")

    # ── 공 소유 반경 자동 결정 ───────────────────────────────
    mb = tracks.merge(ball, on="frame", how="inner")
    # 공중볼 구간은 빼고 잰다. 공이 떠 있으면 지면 최근접 거리가 원래 멀어지는데,
    # 그걸 섞어 재면 '소유 반경'이 실제보다 크게 잡힌다.
    if "ball_z" in mb.columns:
        mb = mb[mb["ball_z"].fillna(0) <= BALL_AIRBORNE_Z]
    if BALL_OUT_MARGIN > 0:                       # 데드볼 구간도 뺀다
        mo = BALL_OUT_MARGIN
        mb = mb[mb.ball_x.between(-mo, PITCH_L + mo) & mb.ball_y.between(-mo, PITCH_W + mo)]
    nn = np.hypot(mb.X - mb.ball_x, mb.Y - mb.ball_y).groupby(mb.frame).min()
    if POSS_RADIUS is None:
        for r in (2.0, 3.0, 5.0, 8.0):
            if (nn <= r).mean() >= 0.6:
                POSS_RADIUS = r
                break
        if POSS_RADIUS is None:
            # 기존 폴백(80분위 거리)은 상한이 없어서 좌표가 나쁘면 40 m 넘는
            # '소유 반경'이 나온다. 그러면 경기장 반대편 선수가 볼 소유자로
            # 잡혀 모든 지표가 말이 안 되는데도 계산은 조용히 끝난다.
            # 사람이 공을 다룰 수 있는 거리에는 물리적 상한이 있으므로 8 m 에서
            # 자르고, 대신 데이터 품질 문제라고 알린다.
            POSS_RADIUS = 8.0
            print(f"  ! 공에서 8 m 안에 아무도 없는 프레임이 "
                  f"{(1-(nn<=8.0).mean())*100:.0f}% 다 (공-최근접선수 거리 중앙값 "
                  f"{nn.median():.1f} m). 공 좌표나 선수 좌표의 정합이 깨진 것이다.")
            print(f"    POSS_RADIUS 를 8 m 로 자른다. 소유자 판정이 대부분 실패할 것이고,"
                  f" 그게 정상이다. 좌표 품질을 먼저 고칠 것.")
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


def ball_plausibility(ball):
    """
    좌표만 보고 '이건 공일 리 없다' 는 프레임을 표시한다.

    검출 단계를 아무리 조여도 공 오검출은 남는다. 남는 것들의 성질이 분명하다.
      · 페널티 마크 — 제자리에 붙어 있다 (얼어붙음)
      · 검출이 진짜 공과 가짜 공을 오갈 때 — 좌표가 한 프레임에 수십 m 튄다
    둘 다 물리로 잘라낼 수 있고, 새로 정할 임계값도 사실상 없다
    (공의 최고 속도, '멈춰 있다'의 정의).

    얼어붙은 공은 가짜이거나, 진짜라면 경기가 멈춘 시간이다. 어느 쪽이든
    기여도를 매길 구간이 아니므로 똑같이 뺀다.

    반환에 ball_bad 열(믿지 않을 행)이 붙는다.
    """
    b = ball.sort_values("frame").reset_index(drop=True).copy()
    bad = pd.Series(False, index=b.index)

    step = np.hypot(b.ball_x.diff(), b.ball_y.diff())
    dt = b.frame.diff() / FPS
    if BALL_SPEED_MAX > 0:
        v = (step / dt).replace([np.inf, -np.inf], np.nan)
        bad |= (v > BALL_SPEED_MAX).fillna(False)

    if BALL_FROZEN_M > 0 and BALL_FROZEN_S > 0:
        # '거의 안 움직인' 구간을 이어 붙여 길이를 잰다
        still = (step <= BALL_FROZEN_M).fillna(False)
        grp = (~still).cumsum()
        span = b.frame.groupby(grp).transform(lambda f: f.max() - f.min())
        bad |= still & (span >= BALL_FROZEN_S * FPS)

    b["ball_bad"] = bad
    return b


def build_frames_table(tracks, ball, teams):
    global BALL_REJECTED
    frames = np.sort(tracks["frame"].unique())
    ball = ball_plausibility(ball)
    BALL_REJECTED = (int((~ball.ball_bad).sum()), int(ball["ball_bad"].sum()))
    ball = fill_ball_gaps(ball[~ball.ball_bad].drop(columns=["ball_bad"]), frames)

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
    # 공이 라인 밖으로 나가면 그 구간은 경기가 멈춘 시간이다. 재개를 기다리며
    # 자리 잡는 배치를 공간 다툼으로 계산하면 안 되므로 소유자를 두지 않는다.
    if BALL_OUT_MARGIN > 0:
        mo = BALL_OUT_MARGIN
        near["ball_out"] = ~(near.ball_x.between(-mo, PITCH_L + mo)
                             & near.ball_y.between(-mo, PITCH_W + mo))
        ok = ok & ~near["ball_out"]
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
    # ── 수식 5 ─────────────────────────────────────────────────────
    #   PD = [clip(1 − d/R, 0, 1)]^2      거리 점수. R 밖이면 0
    #   PV = clip(v_app / v_max, 0, 1)    속도 점수. 물러나면 clip 이 0 으로 자른다
    #   PR = 0.6·PD + 0.4·PV
    #
    #   제곱이 하는 일 — 5 m(=R 의 절반)에서 0.5 가 아니라 0.25 가 된다.
    #   어중간하게 가까운 것에는 점수를 적게 주고 정말 붙었을 때만 높게 준다.
    #
    #   속도 점수의 바닥을 0 으로 막는 이유 — 압박당해 물러나는 수비수가
    #   '멀어진다'는 이유로 마이너스를 받으면 안 된다. 물러나는 것은
    #   압박을 안 하는 것이지 압박의 반대가 아니다.
    prox = np.clip(1 - r.d / PR_R, 0, 1)
    r["PD"] = prox ** PR_PD_POWER
    pv = np.clip(r.v_app / PR_VMAX, 0, 1)
    if SPEC_PR_PV_PROXIMITY:
        # 문서에 없는 변형. 먼 거리에서의 복귀 주력이 PR 로 잡히는 것을 막는다.
        # (모사 경기 실측: 25 m 밖에서 쌓인 값이 PR 총량의 50% 였다)
        pv = pv * prox
    r["PV"] = pv.where(r.kinematics_ok) + 0.0
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


def own_goals(teams):
    """각 팀이 지키는(자기) 골대 좌표. 공격 방향 설정 하나에서만 나오게 모아 둔다."""
    if TEAM_A_ATTACKS_PLUS_X:
        return {teams[0]: (0.0, PITCH_W / 2), teams[1]: (PITCH_L, PITCH_W / 2)}
    return {teams[0]: (PITCH_L, PITCH_W / 2), teams[1]: (0.0, PITCH_W / 2)}


def calculate_sc(tracks, F, teams):
    grid, area = make_grid()
    goals = own_goals(teams)

    meta = F.set_index("frame")
    rows, t0, n = [], time.perf_counter(), 0
    frames = sorted(set(tracks.frame.unique()) & set(meta.index[meta.usable]))
    # 프레임별로 미리 쪼개 둔다. 예전에는 루프 안에서 tracks[tracks.frame == fr] 로
    # 매 프레임 전체 표를 훑었는데, 이건 프레임 수 × 행 수라서 90분 경기(13.5만
    # 프레임 × 300만 행)에서는 격자 계산보다 스캔이 더 오래 걸린다.
    by_frame = {f: g for f, g in tracks.groupby("frame", sort=False)}
    for fr in frames:
        m = meta.loc[fr]
        g = by_frame[fr]
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
        if n % 25 == 0 or n == len(frames):
            _progress("SC", n, len(frames))
            if _stopped():
                raise Stopped()
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

    지수를 전부 1 로 두면 문서의 원안 PA = l*phi*pi 와 정확히 같다 (수식 7-4).
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
        # SC 와 같은 게이트를 쓴다. PA 의 수신공간·길목은 '그 프레임에 보이는
        # 수비수 전체'로 계산하므로, 수비수가 화면 밖으로 빠진 프레임에서는
        # 받는 선수가 실제보다 자유로워 보인다. SC 는 usable 로 막아 두고 PA 는
        # 안 막으면, 추적이 성긴 구간이 PA 만 부풀리는 편향이 된다.
        if "usable" in meta.columns and not bool(m.usable):
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
        # ── 수식 7-3 · 전진 가치 π ───────────────────────────────
        #   Δx = 이 패스로 공격 방향으로 몇 m 나아가는가 (전진 +, 백패스 −)
        #        좌표계의 X 축이 아니라 그 팀의 공격 방향 기준이다.
        #   π  = π_min + (1 − π_min) × (clip(Δx/D_ref, −1, 1) + 1) / 2
        if SPEC_PA_PROG_GOALDIST:
            # 문서에 없는 변형. 측면 전환·대각 패스가 Δx 로는 전진 0 으로
            # 깎이는 것을 피하려고 '골대까지의 거리 감소'로 재는 방식.
            goal = np.array([PITCH_L if dirs[att_t] > 0 else 0.0, PITCH_W / 2.0])
            gain = (np.linalg.norm(c - goal)
                    - np.linalg.norm(P - goal, axis=1))
        else:
            gain = dirs[att_t] * (P[:, 0] - c[0])       # 공격 방향 전진 이득 Δx
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

    두 소유 구간 사이가 너무 벌어지면 그것은 한 번의 패스가 아니다.
    공이 라인 밖으로 나갔다가 스로인으로 재개되거나, 공중볼 경합이 길게
    이어졌거나, 검출이 끊겼던 구간이 여기 걸린다. 가장 긴 롱볼도 3초면
    도착하므로 그보다 벌어진 쌍은 버린다. 이걸 안 하면 '공이 나가기 직전
    프레임'의 배치가 스로인의 패스 장면으로 둔갑해 PA 학습의 정답이 된다.
    """
    f = F[F.phase == "settled"].sort_values("frame")
    f = f[f.poss_id.notna()]
    prev_id = f.poss_id.shift()
    prev_team = f.poss_team.shift()
    prev_frame = f.frame.shift()
    m = (f.poss_id != prev_id) & (f.poss_team == prev_team) & prev_id.notna()
    if PASS_MAX_GAP_S > 0:
        m = m & ((f.frame - prev_frame) <= PASS_MAX_GAP_S * FPS)
    return pd.DataFrame({"from_frame": prev_frame[m].astype(int),
                         "from_id": prev_id[m], "to_id": f.poss_id[m]}).reset_index(drop=True)


def learn_pa_exponents(pa_df, passes):
    """
    실제로 일어난 패스를 정답으로 삼아 지수 a, b, c 를 학습한다.
      양성: 그 프레임에 실제로 공을 받은 선수
      음성: 같은 프레임의 나머지 아군
    log 를 씌우면 곱셈 모델이 선형모델이 되므로 로지스틱 회귀로 추정된다.
    """
    if 0 < len(passes) < 200:
        print(f"          (참고) 패스 {len(passes)}개는 지수 3개를 추정하기에 적은 표본이다. "
              f"순열검정을 통과하더라도 값을 확정된 것으로 쓰지 말 것.")
    if len(passes) < PA_MIN_PASSES:
        return None, (f"패스 표본이 {len(passes)}개뿐이라 학습을 건너뛰고 지수 1 을 유지한다 "
                      f"({PA_MIN_PASSES}개 이상 필요). 표본이 적으면 없는 신호도 "
                      f"'검증 통과'로 나온다.")
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
    n_te_ev = len(te)
    if n_te_ev < PA_MIN_TEST_PASSES or m_tr.sum() < 20 or m_te.sum() < 10 \
       or len(np.unique(y[m_tr])) < 2 or len(np.unique(y[m_te])) < 2:
        return None, (f"검증에 쓸 패스가 {n_te_ev}개뿐이라(최소 {PA_MIN_TEST_PASSES}개) "
                      f"지수 1 을 유지한다 (문서 §6.3). AUC 를 이 표본으로 재면 "
                      f"우연히 좋아 보이는 쪽을 고르게 된다.")

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
                      f"          -> 지수 1 을 유지한다 (문서 §6.3). 위 추정값은 참고용으로만 쓸 것.\n"
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
        sat = float((PAdf["PA_space"] >= 0.999).mean())
        print(f"      포화가 절반을 넘으면 그 차원은 순위에 기여하지 못한다.")
        if sat > 0.5:
            print(f"      ! 수신공간 φ 가 {sat*100:.0f}% 포화됐다. 문서 §7 한계 4 가 지적한 바로 그"
                  f" 현상이다 (7v7 은 사람이 적어 다들 넓게 서 있다).")
            print(f"        다만 φ = clip(d/R_space) 이므로 R_space 를 '줄이면' 포화가 더"
                  f" 심해진다. 문서의 '줄이거나'는 방향이 뒤집힌 서술로 보인다 —")
            print(f"        포화를 풀려면 PA_SPACE_R 을 {PA_SPACE_R:.0f} m 보다 '키우거나'"
                  f" 11v11 데이터를 써야 한다. 지금 값은 문서 그대로 둔 것이다.")
    # 분할 재현성 — 앞구간에서 잘 나온 선수가 뒷구간에서도 잘 나오는가.
    #   같은 선수를 두 번 잰 셈이므로, 순위가 안 맞으면 그 지표는 선수의 성질이
    #   아니라 그 구간의 상황(공이 어디 있었나)을 재고 있는 것이다.
    #   짧은 영상에서는 표본이 적어 낮게 나오는 것이 정상이다. 경기 시간을
    #   늘려도 안 오르면 그때는 지표 문제다.
    if "frame" in M.columns and len(M):
        from scipy.stats import spearmanr as _sp
        half = M["frame"].median()
        out = []
        for c in ("SC", "PR", "PA"):
            if c not in M.columns:
                continue
            g1 = M[M.frame <= half].groupby("track_id")[c].mean()
            g2 = M[M.frame > half].groupby("track_id")[c].mean()
            j = pd.concat([g1, g2], axis=1).dropna()
            if len(j) > 3:
                out.append(f"{c} ρ={_sp(j.iloc[:, 0], j.iloc[:, 1]).statistic:+.3f}")
        if out:
            print("  분할 재현성 (앞구간 vs 뒷구간 순위상관): " + " , ".join(out))
            print("      낮으면 표본이 부족하거나, 선수가 아니라 상황을 재고 있는 것이다")

    # 지표가 '어디에 서는 선수인가'로 얼마나 설명되는지.
    #   포지션 대용값은 '자기 골대까지의 평균거리'를 쓴다. 공까지의 거리(d_mean)로
    #   재면 PR 은 정의상 거리의 함수라 동어반복이 되고, 공이 어디 있었는지에도
    #   휘둘린다. 자기 골대까지의 거리는 공과 무관한 순수 위치 정보다.
    key = "dist_own_goal" if "dist_own_goal" in agg.columns else "d_mean"
    if len(agg) > 3 and key in agg.columns:
        from scipy.stats import spearmanr as _sp
        label = "자기골대까지 거리" if key == "dist_own_goal" else "공까지 평균거리"
        for c in ("SC_mean", "PR_mean", "PA_mean"):
            s = agg[c].dropna()
            d = agg[key].reindex(s.index)
            ok = s.notna() & d.notna()
            if ok.sum() > 3:
                print(f"  {c:8s} vs {label} ρ={_sp(s[ok], d[ok]).statistic:+.3f}", end="")
        print("\n      |ρ| 가 1 에 가까우면 그 지표는 능력이 아니라 포지션을 재고 있다")


# ══════════════ 5d. 합성 — DPI · DEF_z · OFF_z · IPI (수식 14~18) ══════════════
def normalize_scores(s, how=None):
    """
    수식 14 · z-점수.  z = (x − x̄) / σ

    기준 집단은 '그 분석에 들어간 선수 전원'이다(리그 평균이 아니다).
    측정 안 된 선수는 0 으로 채우지 않고 결측으로 둔다 — 문서 §5 ④.
    0 은 '낮다'는 뜻이고 측정 안 됨은 '모른다'는 뜻이라 전혀 다르기 때문이다.
    """
    how = NORM_METHOD if how is None else how
    v = s.dropna()
    if len(v) < 3:
        return pd.Series(np.nan, index=s.index, dtype=float)
    if how == "rank":
        from scipy.stats import norm
        out = pd.Series(norm.ppf(v.rank(method="average") / (len(v) + 1)), index=v.index)
    else:
        sd = v.std(ddof=0)
        out = (v - v.mean()) / (sd if sd else 1.0)
    return out.reindex(s.index)


def reliability(n_pa, k=None):
    """
    수식 17 · 신뢰도  ρ = n_PA / (n_PA + K)

    PA 는 우리 팀이 공을 가진 프레임에서만 측정되므로 선수마다 횟수가 크게 다르다.
    5번 잰 값을 그대로 쓰면 잡음이 점수가 되고, 그 선수를 빼버리면 평균점을
    공짜로 주는 셈이다. 그래서 '잘 모르겠으면 평균 쪽으로 당겨두는' 처리를 한다.
    ρ 는 절대 1 이 되지 않는다(분모가 항상 더 크다).
    """
    k = RELIABILITY_K if k is None else k
    n = pd.Series(n_pa).fillna(0).astype(float)
    return (n / (n + float(k))).astype(float)


def mean_dist_own_goal(tracks, teams):
    """선수별 '자기 골대까지의 평균거리'. 포지션 대용으로 쓴다."""
    g = own_goals(teams)
    gx = tracks["team"].map({t: g[t][0] for t in teams}).to_numpy(float)
    gy = tracks["team"].map({t: g[t][1] for t in teams}).to_numpy(float)
    d = np.hypot(tracks["X"].to_numpy(float) - gx, tracks["Y"].to_numpy(float) - gy)
    return pd.Series(d, index=tracks.index).groupby(tracks["track_id"]).mean()


def detect_goalkeepers(dist_goal, tracks, teams):
    """
    골키퍼를 정한다. 우선순위는 직접 지정 > 검출기의 역할 라벨 > 위치 추정이다.

    문서 §7 한계 6 — 포지션 라벨이 없어 위치로 추측하는 것은 임시 방편이다.
    입력에 goalkeeper 클래스가 있으면 그것을 먼저 쓴다.
    """
    if GK_IDS is None and ROLE_GK_IDS:
        have = set(dist_goal.index)
        keep = {i for i in ROLE_GK_IDS if i in have}
        if keep:
            return keep
    if GK_IDS is not None:
        have = set(dist_goal.index)
        keep = {i for i in GK_IDS if i in have}
        for i in set(GK_IDS) - keep:
            print(f"  ! GK_IDS 의 '{i}' 는 데이터에 없는 track_id 다 (무시).")
        if not keep:
            print("  ! GK_IDS 중 데이터와 맞는 id 가 없다. 골키퍼를 자동 추정한다.")
        else:
            return keep
    team_of = tracks.groupby("track_id")["team"].first()
    gks = set()
    for t in teams:
        s = dist_goal[team_of.reindex(dist_goal.index).to_numpy() == t]
        if len(s) and s.min() < GK_MAX_DIST:
            gks.add(s.idxmin())
    return gks


def _ids(xs):
    """선수 id 목록을 사람이 읽을 수 있게. numpy 정수가 그대로 찍히는 것을 막는다."""
    out = []
    for x in xs:
        try:
            out.append(str(int(x)))
        except (TypeError, ValueError):
            out.append(str(x))
    out = sorted(out)
    return ", ".join(out[:12]) + (f" 외 {len(out) - 12}명" if len(out) > 12 else "")


def player_type(def_z, off_z):
    """
    문서 §6.4 — IPI 하나만 보여주면 안 되고 DEF_z–OFF_z 평면 위의 위치를
    반드시 같이 표시해야 한다. SC 와 PA 는 실측 상관이 −0.855 라서
    둘을 같은 비중으로 더하면 선수 간 차이의 약 48%가 상쇄되기 때문이다.

    사분면에 이름을 붙여 둔다. IPI 가 같아도 유형이 다르면 다른 선수다.
    """
    if pd.isna(def_z) or pd.isna(off_z):
        return ""
    if def_z >= 0 and off_z >= 0:
        return "양면"          # 수비도 공격도 평균 이상
    if def_z >= 0:
        return "수비형"
    if off_z >= 0:
        return "공격형"
    return "저조"


def integrate_scores(agg, tracks, teams):
    """
    수식 15~18 · 세 지표를 하나로.

        DPI    = 0.5·z(SC) + 0.5·z(PR)                       수식 15
        DEF_z  = z( DPI )                                    수식 16
        OFF_z  = z( z(PA) ) × ρ,   ρ = n_PA/(n_PA+K)         수식 16, 17
        IPI_z  = 0.5·DEF_z + 0.5·OFF_z                       수식 18
        IPI    = 100 + 15·z( IPI_z )                         수식 18

    왜 SC·PR·PA 를 1/3 씩 나누지 않는가 (문서 §3.5)
        SC 와 PR 은 둘 다 수비 상황 지표다. 셋에 1/3 씩 주면 수비 2/3 : 공격 1/3 이
        되어 의도하지 않게 수비에 두 배 비중을 주는 셈이 된다. 그래서 상황별로
        먼저 묶고(DEF, OFF) 두 상황을 반반으로 합친다.

    왜 z 를 두 번 씌우는가 (문서 §3.5)
        z-점수 두 개를 평균 내면 서로 다른 방향의 흔들림이 상쇄되어 퍼진 정도가
        1 보다 작아진다. 실측 SC–PR 상관이 +0.098 일 때 평균의 표준편차는 약 0.74 다.
        그대로 두면 수비 점수와 공격 점수의 눈금이 서로 달라 더할 수 없으므로
        한 번 더 표준화해 눈금을 맞춘다.

    100 과 15 에는 아무 의미가 없다 (문서 §5 ⑤)
        100 은 그 분석에 들어간 선수들의 평균일 뿐이고 15 는 1 표준편차다.
        그래서 다른 경기의 IPI 와 직접 비교할 수 없다. 경기 간 비교를 하려면
        표준화 기준(x̄, σ)을 전체 경기로 고정해야 한다.
    """
    a = agg.copy()
    dist_goal = mean_dist_own_goal(tracks, teams)
    gks = detect_goalkeepers(dist_goal, tracks, teams)
    if "dist_own_goal" not in a.columns:
        a["dist_own_goal"] = dist_goal.reindex(a.index).round(1)
    a["is_gk"] = a.index.isin(gks)

    seen = a.get("frames", pd.Series(0, index=a.index)).fillna(0)
    thin = seen < MIN_FRAMES_RATIO * seen.median()
    use = ~thin & (~a["is_gk"] if EXCLUDE_GK else True)

    # ── 수식 14 : 세 지표를 각각 표준화 ─────────────────────────────
    zsc = normalize_scores(a["SC_mean"].where(use))
    zpr = normalize_scores(a["PR_mean"].where(use))
    zpa = normalize_scores(a["PA_mean"].where(use))
    a["SC_z"], a["PR_z"], a["PA_z"] = zsc.round(4), zpr.round(4), zpa.round(4)

    # ── 수식 15 : DPI ──────────────────────────────────────────────
    dpi = W_SC * zsc + W_PR * zpr
    a["DPI"] = dpi.round(4)

    # ── 수식 16 : DEF_z 와 OFF_z ───────────────────────────────────
    a["DEF_z"] = normalize_scores(dpi).round(4)
    rho = reliability(a.get("PA_frames", pd.Series(0, index=a.index)))
    a["rho"] = rho.round(3)
    a["OFF_z"] = (normalize_scores(zpa) * rho).round(4)

    # ── 수식 18 : IPI ──────────────────────────────────────────────
    ipi_z = W_DEF * a["DEF_z"] + W_OFF * a["OFF_z"]
    a["IPI_z"] = ipi_z.round(4)
    a["IPI"] = (IPI_CENTER + IPI_SCALE * normalize_scores(ipi_z)).round(1)
    a["유형"] = [player_type(d, o) for d, o in zip(a["DEF_z"], a["OFF_z"])]

    print(f"\n[최종 점수]  수식 15~18")
    miss = use & a["IPI"].isna()
    if miss.any():
        n_def = int((use & a["DEF_z"].notna()).sum())
        n_off = int((use & a["OFF_z"].notna()).sum())
        print(f"           ! IPI 를 못 낸 선수 {int(miss.sum())}명 "
              f"(DEF_z 있는 선수 {n_def}명 / OFF_z 있는 선수 {n_off}명)")
        print(f"             SC·PR 은 팀이 수비할 때만, PA 는 공격할 때만 나온다. 한 팀이")
        print(f"             관측 구간 내내 공을 갖고 있으면 한쪽은 수비 기록이, 다른 쪽은")
        print(f"             공격 기록이 아예 없어 합칠 수가 없다. 양 팀이 공수를 주고받는")
        print(f"             구간이 필요하다 (문서 §7 한계 3 — 팀 간 점유 불균형).")
    print(f"           DPI   = {W_SC}·z(SC) + {W_PR}·z(PR)")
    print(f"           DEF_z = z(DPI)   ·   OFF_z = z(z(PA)) × ρ , "
          f"ρ = n_PA/(n_PA+{RELIABILITY_K})")
    print(f"           IPI   = {IPI_CENTER:.0f} + {IPI_SCALE:.0f}·z("
          f"{W_DEF}·DEF_z + {W_OFF}·OFF_z)   n() = {NORM_METHOD} 정규화")
    if gks:
        tag = "순위에서 제외" if EXCLUDE_GK else "표시만 하고 포함"
        print(f"           골키퍼 {_ids(gks)} — {tag} "
              f"(자기 골대까지 평균 {dist_goal[list(gks)].mean():.1f} m)")
        if len(gks) < 2:
            print(f"           ! 골키퍼를 {len(gks)}명만 찾았다 (팀당 한 명이어야 한다). "
                  f"GK_MAX_DIST 를 늘리거나 GK_IDS 로 직접 지정할 것.")
    if thin.any():
        print(f"           표본 부족으로 제외: {_ids(a.index[thin])}")

    # 신뢰도가 실제로 얼마나 당겼는지 — 문서 §3.5 의 취지가 보이게 남긴다
    m_rho = a["rho"][use & a["PA_mean"].notna()]
    if len(m_rho):
        print(f"           신뢰도 ρ : 중앙값 {m_rho.median():.2f} · "
              f"최소 {m_rho.min():.2f} · 최대 {m_rho.max():.2f}"
              f"   (PA 측정 횟수 중앙값 {int(a['PA_frames'][m_rho.index].median())}회)")
        if m_rho.median() < 0.5:
            print(f"           ! PA 표본이 전반적으로 적어 공격 점수가 평균 쪽으로 크게 "
                  f"당겨졌다. 더 긴 구간이 필요하다 (문서 §7 한계 2).")

    # 문서 §6.4 — 세 지표가 서로 다른 것을 재고 있는지, 그리고 상쇄가 있는지
    from scipy.stats import spearmanr as _sp
    def _rho(x, y):
        j = pd.concat([x, y], axis=1).dropna()
        return _sp(j.iloc[:, 0], j.iloc[:, 1]).statistic if len(j) > 3 else np.nan
    r_scpr = _rho(a["SC_z"], a["PR_z"])
    r_scpa = _rho(a["SC_z"], a["PA_z"])
    r_defoff = _rho(a["DEF_z"], a["OFF_z"])
    print(f"           상관 — SC:PR {r_scpr:+.3f} (문서 실측 +0.098) · "
          f"SC:PA {r_scpa:+.3f} (−0.855) · DEF:OFF {r_defoff:+.3f} (−0.598)")
    if not np.isnan(r_defoff) and r_defoff < -0.3:
        print(f"           ! DEF 와 OFF 가 반대로 움직인다. 같은 비중으로 더하면 서로 "
              f"상쇄되므로(문서 §6.4 에서 차이의 48% 소실) IPI 하나만 보면 안 되고")
        print(f"             아래 '유형'(DEF–OFF 평면 사분면)을 반드시 함께 읽을 것.")

    if POS_ADJUST:
        m = a["IPI"].notna() & a["dist_own_goal"].notna()
        if m.sum() >= 5:
            x = a.loc[m, "dist_own_goal"].to_numpy(float)
            y = a.loc[m, "IPI"].to_numpy(float)
            b1, b0 = np.polyfit(x, y, 1)
            fit = b0 + b1 * x
            ss = ((y - y.mean()) ** 2).sum()
            r2 = 1 - ((y - fit) ** 2).sum() / ss if ss > 0 else 0.0
            a.loc[m, "IPI_adj"] = np.round(y - fit, 1)
            print(f"           IPI 의 {r2*100:.0f}% 가 '자기 골대까지의 거리' 하나로 "
                  f"설명된다 -> 포지션 보정본 IPI_adj 를 같이 본다")
            if r2 > 0.4:
                print("           ! 절반 가까이가 포지션이다. 야구 WAR 의 포지션 보정에 "
                      "해당하는 장치가 이 연구에는 아직 없다(문서 §2.6).")

    if a["IPI"].notna().sum() > 3:
        base = a["IPI"].dropna()
        worst, arg = 1.0, None
        for w in [(1, 0), (0, 1), (0.75, 0.25), (0.25, 0.75)]:
            alt = (w[0] * a["DEF_z"] + w[1] * a["OFF_z"]).reindex(base.index).dropna()
            if len(alt) > 3:
                r = _sp(base.reindex(alt.index), alt).statistic
                if r < worst:
                    worst, arg = r, w
        print(f"           가중치를 바꿨을 때 순위 상관 최소 ρ={worst:.3f} (DEF:OFF {arg})")
        if worst < 0.5:
            print("           ! 가중치가 순위를 지배한다. 0.5/0.5 는 문서 §4 D등급"
                  "('근거가 없으니 대칭으로 간다')일 뿐이다.")
    return a

# ══════════════ 5e. 표본 검사용 목록 ══════════════
def audit_frames(M, top_n=5):
    """
    선수마다 점수를 가장 많이 끌어올린 프레임을 뽑아 영상 시각과 함께 남긴다.

    집계값만 보면 그 값이 어디서 왔는지 알 수 없다. 검출 오류는 평균 뒤에
    숨고, 평균은 조용히 거짓말을 한다. 오검출을 실제로 찾아내는 방법은
    통계가 아니라 '하나씩 잘라서 눈으로 보는 것' 이다.

    그래서 상위 기여 프레임의 시각(mm:ss)을 뽑아 준다. 영상에서 그 지점을
    열어 '이 순간이 정말 그 선수의 기여였나' 를 몇 개만 확인하면, 표 전체를
    믿어도 되는지 금방 알 수 있다.
    """
    if not len(M):
        return pd.DataFrame()
    rows = []
    for metric in ("SC", "PR", "PA"):
        if metric not in M.columns:
            continue
        d = M[M[metric].notna()]
        if not len(d):
            continue
        gap = AUDIT_MIN_GAP_S * FPS
        for tid, g in d.groupby("track_id", sort=False):
            # 같은 장면의 연속 프레임을 여러 개 뽑으면 점검이 안 된다. 하나 고르면
            # 그 앞뒤 몇 초는 건너뛰고 다음을 고른다 — 서로 다른 순간이 나와야 한다.
            picked = []
            for _, r in g.sort_values(metric, ascending=False).iterrows():
                if any(abs(int(r["frame"]) - q) < gap for q in picked):
                    continue
                picked.append(int(r["frame"]))
                t = float(r["time_s"]) if ("time_s" in r and pd.notna(r["time_s"])) \
                    else float(r["frame"]) / FPS
                rows.append({"track_id": tid, "지표": metric,
                             "값": round(float(r[metric]), 4),
                             "frame": int(r["frame"]),
                             "시각": f"{int(t)//60:02d}:{int(t)%60:02d}",
                             "공까지거리": (round(float(r["d"]), 1)
                                       if ("d" in r and pd.notna(r["d"])) else np.nan)})
                if len(picked) >= top_n:
                    break
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(["지표", "track_id", "값"], ascending=[True, True, False])
            .reset_index(drop=True))


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
    mid = pr[(pr.frame >= 8) & (pr.frame <= 11)]
    near = mid[mid.track_id == "B_near"].PR.mean()
    far = mid[mid.track_id == "B_far"].PR.mean()
    far_pd = mid[mid.track_id == "B_far"].PD.mean()
    # 문서 수식 5 그대로면 압박권(R) 밖에서 거리항은 0 이고, 속도항은 남는다.
    #   그래서 30 m 밖에서 5 m/s 로 달려오는 수비수도 PR = 0.4 를 받는다.
    #   이것은 버그가 아니라 문서가 정의한 동작이고, 그 크기가 걱정되면
    #   [지표 진단] 의 '25 m 밖에서 쌓인 몫' 을 보고 판단하라는 뜻이다.
    #   SPEC_PR_PV_PROXIMITY 를 켜면 속도항에도 거리 커널이 걸려 0 이 된다.
    c7 = (abs(far_pd) < 1e-9) and (near > far)
    globals()["SPEC_PR_PV_PROXIMITY"] = True
    pr2 = calculate_pr(pd.DataFrame(tk), Ft)
    globals()["SPEC_PR_PV_PROXIMITY"] = False
    mid2 = pr2[(pr2.frame >= 8) & (pr2.frame <= 11)]
    far2 = mid2[mid2.track_id == "B_far"].PR.mean()
    c7 = c7 and (far2 < 0.02)
    FPS = old

    # 원점 판정: 라인 밖 사람이 몇 명 섞여도 구석 원점을 중앙 원점으로 오판하면 안 된다.
    rr = np.random.default_rng(1)
    cx = np.r_[rr.uniform(2, PITCH_L - 2, 400), [30., 40., 75.]]     # 벤치·사진기자석
    cy = np.r_[rr.uniform(2, PITCH_W - 2, 400), [-6., -7., 74.]]
    c8 = (not looks_centered(cx, cy)) and \
         looks_centered(cx - PITCH_L / 2, cy - PITCH_W / 2)

    # ID 재연결: 갈 수 없는 거리는 같은 사람으로 잇지 않는다 (붙은 것은 이어야 한다).
    tk2 = pd.DataFrame({"frame": [0, 0, 1, 1], "time_s": 0.0,
                        "track_id": ["p1", "p2", "q1", "q2"], "team": "A",
                        "X": [10.0, 60.0, 10.2, 95.0], "Y": 34.0})
    old2, FPS = FPS, fps
    rid = repair_ids(tk2)
    FPS = old2
    pick = lambda fr_, x_: rid[(rid.frame == fr_) & (rid.X == x_)].track_id.iloc[0]
    c9 = (pick(1, 10.2) == pick(0, 10.0)) and (pick(1, 95.0) != pick(0, 60.0))

    # 공 높이 열은 이름이 뭐로 오든 찾아야 한다. 못 찾으면 입력 형식만 달라도
    # 공중볼 처리가 조용히 꺼진다(실측: 같은 데이터에서 39프레임 -> 0프레임).
    _cols = lambda *cs: pd.DataFrame(columns=list(cs))
    c10 = (find_height_col(_cols("frame", "ball_x", "ball_y", "ball_z")) == "ball_z"
           and find_height_col(_cols("frame", "X", "Y", "Z")) == "Z"
           and find_height_col(_cols("frame", "X", "Y")) is None
           and find_height_col(_cols("frame", "Z", "ball_z")) == "ball_z")

    # 역할 열: 골키퍼는 남기고 심판은 뺀다. 'referee' 안의 'ref' 가 'goalkeeper' 를
    # 잡아먹지 않아야 한다. 예전 필터는 골키퍼 행을 통째로 버렸다.
    rt = pd.DataFrame({"track_id": ["p1", "gk1", "r1", "p2", "gk2"],
                       "cls": ["선수", "골키퍼", "심판", "player", "goalkeeper"]})
    keep_r, gks_r, nref_r, col_r = classify_roles(rt)
    c11 = (col_r == "cls" and nref_r == 1 and gks_r == {"gk1", "gk2"}
           and list(keep_r) == [True, True, False, True, True])

    # 공 물리 검사: 제자리에 붙어 있는 것과 순간이동은 공이 아니다.
    old3, FPS = FPS, fps
    # 앞 60프레임은 정상 주행, 뒤 120프레임(=4초)은 제자리에 붙어 있다.
    # 고정 구간은 BALL_FROZEN_S 보다 확실히 길어야 검사에 걸린다.
    bx = np.r_[np.linspace(10, 40, 60), np.full(120, 94.0)]
    bt = pd.DataFrame({"frame": np.arange(180), "ball_x": bx, "ball_y": 34.0})
    pl = ball_plausibility(bt)
    c12a = (not pl.ball_bad[:55].any()) and pl.ball_bad[100:].all()
    bt2 = pd.DataFrame({"frame": [0, 1, 2], "ball_x": [10.0, 12.0, 90.0], "ball_y": 34.0})
    c12b = bool(ball_plausibility(bt2).ball_bad.iloc[2])        # 78 m 이동 = 순간이동
    FPS = old3
    c12 = c12a and c12b

    for nm, c in [("등속 5.00 m/s 정확", c1), ("추적 끊김 구간 NaN", c2),
                  ("team 열 통과", c3), (f"효율성 공리 (오차 {abs(sc.sum()-tot):.0e})", c4),
                  (f"협력 수비 보존 ({s2[0]:.0f} vs {s2[1]:.0f})", c5),
                  ("위험도 가중치 방향", c6),
                  (f"수식 5 대로 R 밖 거리항 0 · 스위치 켜면 속도항도 0 "
                   f"(근접 {near:.3f} / 원거리 {far:.3f} -> {far2:.3f})", c7),
                  ("라인 밖 사람이 있어도 원점 판정 유지", c8),
                  ("ID 재연결이 순간이동을 잇지 않음", c9),
                  ("공 높이 열 인식 (ball_z / Z / 없음)", c10),
                  ("역할 열: 골키퍼 유지 · 심판 제외", c11),
                  ("공 물리 검사: 고정·순간이동 걸러냄", c12)]:
        print(f"  {'PASS' if c else '**FAIL**':9s} {nm}")
    if not all([c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]):
        raise SystemExit("검증 실패. 아래 결과를 믿으면 안 된다.")


# ══════════════ 6b. 문서의 숫자 예시로 검증 ══════════════
def spec_test():
    """
    방법론 문서에 실린 '숫자로 해보기' 예시를 그대로 재현한다.

    문서가 손으로 계산해 둔 값을 코드가 그대로 내놓는지 보는 것이라,
    수식을 잘못 옮겼으면 여기서 걸린다. 자체 검증(self_test)이 '계산이
    정의대로 도는가'를 본다면, 이쪽은 '정의가 문서와 같은가'를 본다.
    """
    print("\n[문서 검증] 방법론 문서의 숫자 예시를 재현한다")
    ok = []

    def chk(name, got, want, tol=5e-3):
        good = abs(float(got) - float(want)) <= tol
        ok.append(good)
        print(f"  {'PASS' if good else '**FAIL**':9s} {name}  "
              f"(계산 {float(got):.3f} / 문서 {float(want):.3f})")

    # 수식 4 · 칸의 위험도 -------------------------------------------
    #   골대 9 m, 공 10 m -> 0.607 × 0.607 = 0.368
    #   같은 자리인데 공만 60 m -> 0.607 × 0.050 = 0.030  (12.2배 차이)
    g = (0.0, PITCH_W / 2)
    c_near = np.array([[9.0, PITCH_W / 2]])
    w_near = danger_weight(c_near, g, (19.0, PITCH_W / 2), 1.0)[0]
    w_far = danger_weight(c_near, g, (69.0, PITCH_W / 2), 1.0)[0]
    chk("수식 4 · 위험도 (골대 9m, 공 10m)", w_near, 0.368)
    chk("수식 4 · 위험도 (공만 60m 로)", w_far, 0.030)
    chk("수식 4 · 두 경우의 비 (12.2배)", w_near / w_far, 12.2, tol=0.3)

    # 수식 13 · 선수 한 명의 점수 합산 --------------------------------
    #   문서 예시: 0.90/1 + 0.50/2 + 0.20/4 = 1.200
    chk("수식 13 · 세 칸 합산 예시", 0.90 / 1 + 0.50 / 2 + 0.20 / 4, 1.200)

    # 수식 5, 6 · PR -------------------------------------------------
    #   거리 3 m, 초당 2 m 접근 -> PD 0.490, PV 0.400, PR 0.454
    global FPS
    old_fps, FPS = FPS, 30.0
    fr = np.arange(21)
    rows = [{"frame": f, "time_s": f / 30.0, "track_id": "A0", "team": "A",
             "X": 50.0, "Y": 34.0} for f in fr]
    rows += [{"frame": f, "time_s": f / 30.0, "track_id": "B0", "team": "B",
              "X": 53.0 + 2.0 * (10 - f) / 30.0, "Y": 34.0} for f in fr]
    Ft = pd.DataFrame({"frame": fr, "ball_x": 50.0, "ball_y": 34.0,
                       "poss_team": "A", "poss_id": "A0", "phase": "settled"})
    pr = calculate_pr(pd.DataFrame(rows), Ft)
    row = pr[(pr.frame == 10) & (pr.track_id == "B0")]
    FPS = old_fps
    if len(row):
        chk("수식 5 · PD (d=3 m)", row.PD.iloc[0], 0.490)
        chk("수식 6 · PV (접근 2 m/s)", row.PV.iloc[0], 0.400)
        chk("수식 5 · PR = 0.6·PD + 0.4·PV", row.PR.iloc[0], 0.454)
    else:
        ok.append(False)
        print("  **FAIL**  수식 5 · PR 예시 — 계산 행이 비었다")

    # 수식 5 · 거리별 PD 표 ------------------------------------------
    pd_tab = [(0, 1.000), (2, 0.640), (5, 0.250), (8, 0.040), (10, 0.0), (14, 0.0)]
    got = [float(np.clip(1 - d / 10.0, 0, 1) ** 2) for d, _ in pd_tab]
    chk("수식 5 · PD 표 (0·2·5·8·10·14 m)",
        sum(abs(a - b) for a, (_, b) in zip(got, pd_tab)), 0.0, tol=1e-3)

    # 수식 7-1, 7-2, 7-3 · PA 성분 -----------------------------------
    lane_tab = [(0.0, 0.00), (0.6, 0.30), (1.5, 0.75), (3.0, 1.00)]
    chk("수식 7-1 · 길목 표 (W_lane 2.0)",
        sum(abs(np.clip(g_ / 2.0, 0, 1) - v) for g_, v in lane_tab), 0.0, tol=1e-3)
    space_tab = [(1.0, 0.20), (2.5, 0.50), (4.0, 0.80), (7.0, 1.00)]
    chk("수식 7-2 · 수신공간 표 (R_space 5.0)",
        sum(abs(np.clip(d / 5.0, 0, 1) - v) for d, v in space_tab), 0.0, tol=1e-3)
    prog_tab = [(30, 1.000), (15, 0.825), (0, 0.650), (-15, 0.475), (-30, 0.300),
                (-50, 0.300)]
    prog = lambda dx: 0.3 + 0.7 * (np.clip(dx / 30.0, -1, 1) + 1) / 2
    chk("수식 7-3 · 전진가치 표", sum(abs(prog(dx) - v) for dx, v in prog_tab), 0.0, tol=1e-3)
    chk("수식 7 · PA = 0.75 × 0.98 × 0.65", 0.75 * 0.98 * 0.65, 0.478)

    # 수식 7 계열을 실제 경로로 한 번 -------------------------------
    #   소유자 (50,34), 받을 선수 (65,34) -> Δx = 15 m -> π = 0.825
    #   수비수를 받을 선수에서 4 m 옆에 두면 φ = 0.8, 길목은 4 m 라 ℓ = 1.0
    #   PA = 1.0 × 0.8 × 0.825 = 0.660
    old_dir, globals()["TEAM_A_ATTACKS_PLUS_X"] = TEAM_A_ATTACKS_PLUS_X, True
    tk = pd.DataFrame([
        {"frame": 0, "time_s": 0.0, "track_id": "A0", "team": "A", "X": 50.0, "Y": 34.0},
        {"frame": 0, "time_s": 0.0, "track_id": "A1", "team": "A", "X": 65.0, "Y": 34.0},
        {"frame": 0, "time_s": 0.0, "track_id": "B0", "team": "B", "X": 65.0, "Y": 38.0},
    ])
    Fp = pd.DataFrame([{"frame": 0, "ball_x": 50.0, "ball_y": 34.0, "poss_team": "A",
                        "poss_id": "A0", "phase": "settled", "usable": True}])
    pa = calculate_pa(tk, Fp, ["A", "B"])
    globals()["TEAM_A_ATTACKS_PLUS_X"] = old_dir
    if len(pa):
        chk("수식 7-3 · 실제 경로 π (Δx=15 m)", pa.PA_prog.iloc[0], 0.825)
        chk("수식 7-2 · 실제 경로 φ (마크 4 m)", pa.PA_space.iloc[0], 0.800)
        chk("수식 7 · 실제 경로 PA", pa.PA.iloc[0], 0.660)
    else:
        ok.append(False)
        print("  **FAIL**  수식 7 · PA 예시 — 계산 행이 비었다")

    # 수식 17 · 신뢰도 ρ ---------------------------------------------
    rho_tab = [(10, 0.17), (50, 0.50), (150, 0.75), (450, 0.90)]
    chk("수식 17 · 신뢰도 표 (K=50)",
        sum(abs(float(reliability([n]).iloc[0]) - v) for n, v in rho_tab), 0.0, tol=0.02)

    # 수식 18 · 100 기준 눈금 ----------------------------------------
    ipi_tab = [(2.0, 130), (1.0, 115), (0.0, 100), (-1.0, 85)]
    chk("수식 18 · IPI 눈금",
        sum(abs((IPI_CENTER + IPI_SCALE * z) - v) for z, v in ipi_tab), 0.0, tol=1e-6)

    # 수식 19 · 구장 크기 보정 ---------------------------------------
    sc60 = float(np.sqrt((60.0 / 105.0) * (40.0 / 68.0)))
    chk("수식 19 · 60x40 구장 배율", sc60, 0.58, tol=0.005)
    chk("수식 19 · LAMBDA_GOAL 18 -> 10.4", 18.0 * sc60, 10.4, tol=0.1)
    chk("수식 19 · PR_R 10 -> 5.8", 10.0 * sc60, 5.8, tol=0.05)

    if not all(ok):
        raise SystemExit("문서 검증 실패. 수식이 문서와 다르게 구현돼 있다.")
    print(f"  -> {len(ok)}개 항목 모두 문서와 일치")


def print_constants():
    """
    문서 §4 의 상수 등급표를 실행할 때마다 찍는다.

    문서의 서술 원칙이 "그냥 정한 값을 근거가 있는 것처럼 쓰지 않겠다"이고,
    "D등급이 11개라는 사실 자체를 결과와 함께 보고한다"고 못박아 두었다.
    그래서 결과 위에 항상 같이 나오게 한다.
    """
    rows = [
        ("A", "Shapley 배분 w(c)/k(c)", "대칭성·효율성 공리에서 유일"),
        ("A", "중앙차분 (앞뒤 ÷ 2Δ)", "전방차분보다 오차 한 단계 작음"),
        ("A", "위험도를 곱으로", "두 조건이 동시에 만족돼야 위험"),
        ("A", "PA 를 곱으로", "하나라도 0 이면 선택지가 아님"),
        ("B", f"SPEED_MAX = {SPEED_MAX:g} m/s", "사람의 단거리 최고 속도대"),
        ("B", f"PR_VMAX = {PR_VMAX:g} m/s", "압박할 때 실제로 나오는 접근 속도대"),
        ("B", f"W_LANE = {W_LANE:g} m", "선수 몸 폭 + 팔다리가 닿는 범위"),
        ("B", f"경기장 = {PITCH_L:g} x {PITCH_W:g} m", "FIFA 권장 국제경기 규격"),
        ("C", f"GRID_STEP = {GRID_STEP:g} m", "0.5 m 와 SC 차이 1% 미만, 시간 4배"),
        ("C", f"DELTA = {DELTA} 프레임", "가상 데이터로 등속 참값 복원 확인"),
        ("C", "PLAYER_MAX_GAP = 0.3 초", "이 길이까지 보간 오차 0.4 m 이내"),
        ("C", "BALL_MAX_GAP = 0.2 초", "공은 더 빠르고 방향이 급변"),
        ("C", "HOLD_FRAMES = 1.5 초", "소유 판정 깜빡임 제거 최소 유지 시간"),
        ("C", f"MIN_TRACKED = 정원의 80%", "이보다 적으면 SC 는 값이 아니라 잡음"),
        ("C", f"PA_MIN_PASSES = {PA_MIN_PASSES}", "순열검정을 통과했을 때만 지수 채택"),
        ("D", f"LAMBDA_GOAL = {LAMBDA_GOAL:.1f} m", "12/18/25 로 바꿔 순위 변화 확인 필요"),
        ("D", f"LAMBDA_BALL = {LAMBDA_BALL:.1f} m", "위와 동일"),
        ("D", f"PR_R = {PR_R:.1f} m", "실제 공 탈취 거리 분포와 비교 필요"),
        ("D", f"PD 지수 = {PR_PD_POWER}", "지수 1/2/3 비교 필요"),
        ("D", f"PR_WD:PR_WV = {PR_WD}:{PR_WV}", "결과 라벨 확보 후 회귀로 추정"),
        ("D", f"PA_SPACE_R = {PA_SPACE_R:.1f} m", "감도분석 필요"),
        ("D", f"PA_PI_MIN = {PA_PI_MIN}", "감도분석 필요"),
        ("D", f"PA_FWD_REF = {PA_FWD_REF:.1f} m", "감도분석 필요"),
        ("D", f"W_SC:W_PR = {W_SC}:{W_PR}", "의도적 대칭. 근거 없음"),
        ("D", f"DEF:OFF = {W_DEF}:{W_OFF}", "의도적 대칭. 근거 없음"),
        ("D", f"신뢰도 K = {RELIABILITY_K}", "측정 횟수와 안정성 관계 실측 필요"),
    ]
    print("\n[상수표]  문서 §4 — 이 숫자들이 어디서 왔는가")
    for grade, label in (("A", "이론에서 유일하게 나온 값 (고를 여지 없음)"),
                         ("B", "문헌·경기 규칙에서 가져온 값"),
                         ("C", "이 연구의 데이터·실험으로 정한 값"),
                         ("D", "그냥 정한 값 — 아직 검증되지 않음")):
        items = [r for r in rows if r[0] == grade]
        print(f"  [{grade}] {label}   ({len(items)}개)")
        for _, name, why in items:
            print(f"        {name:32s} {why}")
    nd = sum(1 for r in rows if r[0] == "D")
    print(f"  D등급이 {nd}개다. 이 중 W_SC:W_PR 과 DEF:OFF 는 '근거가 없으니 대칭으로 "
          f"간다'는 명시적 판단이고,")
    print(f"  나머지 {nd - 2}개가 감도분석이 필요한 자유 파라미터다 (문서 §4).")
    off = [n for n, v in (("SPEC_PR_PV_PROXIMITY", SPEC_PR_PV_PROXIMITY),
                          ("SPEC_PA_PROG_GOALDIST", SPEC_PA_PROG_GOALDIST)) if v]
    if off:
        print(f"  ! 문서와 다르게 켜 둔 스위치: {off} — 결과를 문서 수식의 값으로 "
              f"보고하면 안 된다.")


# ══════════════ 7. 실행 ══════════════
def main():
    self_test()
    spec_test()
    print_constants()
    _progress("읽기", 0, 1)
    tracks, ball, teams = read_data()

    print("\n[계산]")
    n_raw = len(tracks)
    tracks = fill_player_gaps(tracks)
    n_itp = int(tracks["pos_interp"].sum())
    print(f"  선수좌표 결측 보간 {n_itp}행 추가 (원본 {n_raw:,}행, 최대 "
          f"{PLAYER_MAX_GAP}프레임 = {PLAYER_MAX_GAP/FPS:.2f}초 구간까지)")
    _progress("속도", 0, 1)
    L1 = calculate_velocity(tracks)
    print(f"  속도    유효 {L1.kinematics_ok.mean()*100:.0f}% , 중앙값 {L1.speed.median():.2f} m/s")

    nfr = L1.frame.nunique()
    cov0 = ball.frame.nunique() / nfr
    _progress("소유판정", 0, 1)
    F = build_frames_table(L1, ball, teams)
    itp = int(F["ball_interp"].sum()) if "ball_interp" in F.columns else 0
    print(f"  공좌표  원본 {cov0*100:.0f}% 프레임에 존재"
          f" -> 보간 {itp}프레임 추가 (최대 {BALL_MAX_GAP}프레임 = {BALL_MAX_GAP/FPS:.2f}초 구간까지)")
    if cov0 < 0.7:
        print(f"    ! 공 검출이 {(1-cov0)*100:.0f}% 끊겼다. 보간으로 메울 수 있는 건 짧은 구간뿐이라"
              " 긴 구간은 그대로 버려진다. 공 검출부터 개선할 것.")
    keptb, badb = BALL_REJECTED
    if badb:
        print(f"  공품질  물리 검사로 {badb}프레임 제외 "
              f"(순간이동 {BALL_SPEED_MAX:.0f} m/s 초과 · {BALL_FROZEN_S:.0f}초 넘게 얼어붙음)"
              f" -> 남은 공 프레임 {keptb}")
        if badb > 0.2 * (keptb + badb):
            print(f"    ! 공 프레임의 {badb/(keptb+badb)*100:.0f}% 가 물리적으로 말이 안 된다."
                  f" 공 검출에 페널티 마크 같은 고정 오검출이 섞여 있을 수 있다.")
    if "ball_out" in F.columns and F["ball_out"].any():
        print(f"  데드볼  공이 라인 밖인 프레임 {F['ball_out'].mean()*100:.0f}% "
              f"— 경기가 멈춘 시간이라 소유자를 두지 않는다")
    print(f"  공소유  usable {F.usable.mean()*100:.0f}% , {F.poss_team.value_counts().to_dict()}")
    if F.usable.mean() < 0.3:
        no_poss = (F.poss_team == "none").mean()
        few = (F[teams].min(axis=1) < MIN_TRACKED).mean()
        print(f"  ! usable 이 너무 낮다. 소유자 미판정 {no_poss*100:.0f}% / "
              f"추적 인원 부족 {few*100:.0f}% 이 원인이다.")
        print("    앞쪽이 크면 공 검출·POSS_RADIUS 문제, 뒤쪽이 크면 화면 밖 선수가 "
              "빠진 좌표라 경기장 전체를 덮는 트래킹이 필요하다.")

    _progress("PR", 0, 1)
    PRdf = calculate_pr(L1, F)
    print(f"  PR      {len(PRdf):,}행")
    print("  SC      계산 중...")
    SCdf = calculate_sc(L1, F, teams)

    _progress("PA", 0, 1)
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
    agg["dist_own_goal"] = mean_dist_own_goal(L1, teams).reindex(agg.index).round(1)
    metric_report(M, PAdf, agg)

    # 수식 14~18. z 표준화부터 IPI 까지 전부 여기서 나온다.
    agg = integrate_scores(agg, L1, teams)
    agg = agg.sort_values("IPI" if agg["IPI"].notna().any() else "DPI",
                          ascending=False)

    print(f"\n[선수별 결과]")
    print(f"           IPI 100 = 이 분석에 들어간 선수들의 평균 · 15 = 1 표준편차.")
    print(f"           다른 경기의 IPI 와 직접 비교할 수 없다 (문서 §5 ⑤).")
    n_type = ({k: v for k, v in agg["유형"].value_counts().to_dict().items() if k}
              if "유형" in agg.columns else {})
    if n_type:
        print(f"           DEF–OFF 유형 분포: {n_type}"
              f"   ← IPI 가 같아도 유형이 다르면 다른 선수다")
    # 화면에는 읽을 수 있는 만큼만. 전체 열은 엑셀에 다 들어간다.
    show = [c for c in ["team", "frames", "dist_own_goal", "is_gk",
                        "SC_mean", "SC_share", "PR_mean", "PR_zone", "PR_in",
                        "PA_mean", "PA_frames", "rho",
                        "DPI", "DEF_z", "OFF_z", "IPI", "IPI_adj", "유형"]
            if c in agg.columns]
    disp = agg[show].copy()
    for c in disp.columns:                      # 빈 칸은 '–' 로 — NaN 이 줄줄이 찍히면 못 읽는다
        if disp[c].dtype.kind in "fc":
            disp[c] = disp[c].map(lambda v: "–" if pd.isna(v) else f"{v:g}")
    print(disp.to_string())
    print(f"  (표시한 열은 {len(show)}개다. 성분별 원값과 z 점수 등 전체 "
          f"{len(agg.columns)}개 열은 {SAVE_XLSX} 의 '선수별' 시트에 있다)")

    # 엑셀 한 시트에는 1,048,576 행까지만 들어간다. 프레임별 표는 프레임 하나에
    # 열 명 남짓씩 쌓이므로 45분만 넣어도 이 한계를 넘는다(실측: 3분 72,258행
    # -> 45분 약 108만행). 그대로 to_excel 하면 SC 를 수십 분 계산한 맨 마지막에
    # 저장이 터져서 결과가 전부 날아간다. 큰 표는 CSV 로 따로 뺀다.
    XLSX_MAX_ROWS = 1_000_000
    saved, spilled = [], []
    with pd.ExcelWriter(SAVE_XLSX, engine="openpyxl") as w:
        agg.reset_index().to_excel(w, sheet_name="선수별", index=False)
        saved.append("선수별")
        for nm, df_ in (("프레임별", M), ("프레임메타", F),
                        ("PA프레임별", PAdf), ("검출된패스", passes),
                        ("표본검사", audit_frames(M))):
            if not len(df_):
                continue
            if len(df_) > XLSX_MAX_ROWS:
                path = f"{os.path.splitext(SAVE_XLSX)[0]}_{nm}.csv"
                df_.to_csv(path, index=False)
                spilled.append((nm, path, len(df_)))
            else:
                df_.to_excel(w, sheet_name=nm, index=False)
                saved.append(nm)
    print(f"\n[저장] {SAVE_XLSX}  (시트: {' / '.join(saved)})")
    for nm, path, n in spilled:
        print(f"        '{nm}' 는 {n:,}행이라 엑셀 한 시트에 안 들어간다 -> {path}")

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
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.0))

    # 왼쪽 — DEF_z / OFF_z 평면. 문서 §6.4 가 IPI 와 반드시 함께 보라고 요구한 그림이다.
    #   SC 와 PA 는 실측 상관이 -0.855 라서 둘을 같은 비중으로 더하면 선수 간 차이의
    #   약 48%가 상쇄된다. IPI 점수만으로는 '수비형이라 높은 건지 공격형이라 높은 건지'
    #   구분이 안 되므로 평면 위 위치를 같이 봐야 한다.
    _has_plane = ({"DEF_z", "OFF_z"} <= set(agg.columns)
                  and len(agg.dropna(subset=["DEF_z", "OFF_z"])) >= 2)
    if _has_plane:
        d = agg.dropna(subset=["DEF_z", "OFF_z"])
        col = {t: c for t, c in zip(teams, ("#B94E39", "#2A6699"))}
        ax[0].axhline(0, color="#999", lw=.8); ax[0].axvline(0, color="#999", lw=.8)
        for t in teams:
            q = d[d.team == t]
            ax[0].scatter(q.DEF_z, q.OFF_z, s=46, c=col.get(t, "#777"),
                          ec="w", zorder=3, label=str(t))
        for i, r in d.iterrows():
            ax[0].annotate(str(i)[:10], (r.DEF_z, r.OFF_z), fontsize=6.5,
                           xytext=(4, 3), textcoords="offset points", color="#444")
        lim = float(np.nanmax(np.abs(np.r_[d.DEF_z.values, d.OFF_z.values]))) * 1.25 + .1
        ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
        for x_, y_, txt in ((lim, lim, "both"), (lim, -lim, "defensive"),
                            (-lim, lim, "offensive"), (-lim, -lim, "low")):
            ax[0].text(x_ * .96, y_ * .93, txt, fontsize=7.5, color="#888",
                       ha="right" if x_ > 0 else "left",
                       va="top" if y_ > 0 else "bottom")
        ax[0].set_xlabel("DEF_z  (defensive)"); ax[0].set_ylabel("OFF_z  (offensive)")
        ax[0].set_title("DEF-OFF plane  (read together with IPI)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.25)
    else:
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
    plt.show() if IN_COLAB else plt.savefig(SAVE_PLOT, dpi=110, bbox_inches="tight")
    if not IN_COLAB:
        print(f"  그림 저장: {SAVE_PLOT}")


if __name__ == "__main__":
    M, agg, F = main()
