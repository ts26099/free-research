# =====================================================================
#  foot.py 출력 -> 기여도 계산용 미터 좌표로 변환
#
#  foot.py 가 내보내는 X_Coord / Y_Coord 는 미터가 아니라 '화면 폭·높이의
#  백분율'(-50 ~ +50)이다. 기여도 계산(SC/PR/PA)은 전부 미터를 전제로 한다 —
#  압박 반경 10 m, 수신 공간 8 m, 위험도 감쇠 18 m 같은 상수가 모두 그렇다.
#  백분율을 그대로 넣으면 숫자는 나오지만 전부 의미가 없다.
#
#  게다가 축이 서로 뒤바뀌어 있다. foot.py 의 미니맵은 800x1200 세로형이고
#  하프라인을 가로로 긋는다 — 즉 골대-골대 축이 화면 세로(Y_Coord)다.
#  기여도 코드는 X 를 골대-골대 축으로 본다. 그대로 넣으면 경기장이 90도
#  돌아간 채로 계산된다.
#
#  ★ 전제: 영상이 '위에서 내려다본' 화면이라 화면 위치와 경기장 위치가
#    1차식으로 이어진다는 것. foot.py 가 색깔 원을 HoughCircles 로 찾는
#    구조인 걸 보면 2D 전술 애니메이션으로 보이고, 그렇다면 이 전제가 맞다.
#    실제 중계 영상(비스듬한 카메라)이라면 원근 때문에 1차식이 성립하지 않아
#    이 변환으로는 안 되고 호모그래피가 필요하다.
# =====================================================================

SRC_CSV   = "tactical_positions.csv"
DST_CSV   = "pitch_positions.csv"

# 실제 경기장 규격 (m). 7v7 유소년은 보통 55~68 x 37~47 이다.
PITCH_L, PITCH_W = 64.0, 44.0     # 길이(골대~골대) x 폭

# 화면에서 경기장이 차지하는 구간. 영상 가장자리에 여백이 있으면 줄인다.
#   0.0 ~ 1.0 (화면 전체를 1 로 본다). 경기장이 화면을 꽉 채우면 0.0, 1.0.
LONG_FROM,  LONG_TO  = 0.0, 1.0   # 골대-골대 축이 화면에서 차지하는 구간
CROSS_FROM, CROSS_TO = 0.0, 1.0   # 폭 축이 차지하는 구간

LONG_AXIS = "auto"        # 골대-골대 축이 어느 열인가. "auto" / "X_Coord" / "Y_Coord"
DROP_OUTSIDE = 1.5        # 경기장 밖으로 이만큼(m) 넘게 나간 좌표는 버린다. 0 이면 끔
# ---------------------------------------------------------------------

import sys
import numpy as np
import pandas as pd


def load(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    need = {"Time(sec)", "ID", "Team", "X_Coord", "Y_Coord"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"열이 없다: {miss}\n실제 열: {list(df.columns)}")
    return df


def pick_long_axis(players):
    """
    골대-골대 축을 고른다.

    선수들은 폭 방향보다 길이 방향으로 더 넓게 퍼진다. 그래서 퍼진 폭이 큰
    쪽이 골대-골대 축이다. 사람이 설정에서 직접 지정할 수도 있게 해 둔다.
    """
    if LONG_AXIS in ("X_Coord", "Y_Coord"):
        return LONG_AXIS
    sx = players.X_Coord.max() - players.X_Coord.min()
    sy = players.Y_Coord.max() - players.Y_Coord.min()
    axis = "Y_Coord" if sy >= sx else "X_Coord"
    print(f"  골대-골대 축 자동 판정: {axis} "
          f"(퍼진 폭 X {sx:.1f} vs Y {sy:.1f})")
    return axis


def to_meters(v, lo, hi, length):
    """
    백분율 좌표(-50~+50)를 미터로 옮긴다.

    foot.py 는 화면 정규화 좌표에서 0.5 를 뺀 뒤 100 을 곱한다. 되돌리면
    화면 비율 = v/100 + 0.5 이고, 경기장이 화면의 lo~hi 구간을 차지하므로
    경기장 안에서의 비율 = (화면비율 - lo) / (hi - lo) 이다.
    """
    frac = v / 100.0 + 0.5
    if hi - lo <= 0:
        raise SystemExit("경기장이 화면에서 차지하는 구간 설정이 잘못됐다 (hi <= lo).")
    return (frac - lo) / (hi - lo) * length


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else SRC_CSV
    dst = sys.argv[2] if len(sys.argv) > 2 else DST_CSV
    df = load(src)
    print(f"[읽기] {src}  {len(df):,}행 / {df.ID.nunique()}개 ID")

    ball = df.ID.astype(str).str.upper().eq("BALL") | \
           df.Team.astype(str).str.lower().eq("ball")
    players = df[~ball]
    if players.empty:
        raise SystemExit("선수 행이 없다. ID/Team 열을 확인할 것.")

    long_col = pick_long_axis(players.dropna(subset=["X_Coord", "Y_Coord"]))
    cross_col = "X_Coord" if long_col == "Y_Coord" else "Y_Coord"

    out = pd.DataFrame({
        "time_s": df["Time(sec)"].astype(float),
        "track_id": df["ID"].astype(str),
        "team": df["Team"].astype(str),
        # 골대-골대 축 -> X, 폭 축 -> Y  (기여도 코드의 규약에 맞춘다)
        "X": to_meters(df[long_col].astype(float), LONG_FROM, LONG_TO, PITCH_L),
        "Y": to_meters(df[cross_col].astype(float), CROSS_FROM, CROSS_TO, PITCH_W),
    })
    out["cls"] = np.where(ball, "ball", "person")
    out = out.dropna(subset=["X", "Y"])

    # foot.py 는 0.1초 간격으로 기록하므로 프레임 번호를 그 간격으로 만든다.
    step = float(pd.Series(sorted(out.time_s.unique())).diff().median())
    if not np.isfinite(step) or step <= 0:
        raise SystemExit("시간 간격을 알 수 없다.")
    out["frame"] = np.round(out.time_s / step).astype(int)
    fps = round(1.0 / step, 3)

    print(f"  변환 결과: X(골대축) {out.X.min():.1f}~{out.X.max():.1f} m , "
          f"Y(폭) {out.Y.min():.1f}~{out.Y.max():.1f} m   [경기장 {PITCH_L}x{PITCH_W}]")
    print(f"  샘플링 {fps} Hz")

    #  선수들이 경기장의 어느 만큼을 실제로 쓰는지 본다. 이 비율이 많이 낮으면
    #  경기장이 화면을 꽉 채운다는 전제가 틀린 것이다(여백 설정을 고쳐야 한다).
    pl = out[out.cls == "person"]
    cov_l = (pl.X.max() - pl.X.min()) / PITCH_L
    cov_w = (pl.Y.max() - pl.Y.min()) / PITCH_W
    print(f"  선수가 덮는 범위: 길이 {cov_l*100:.0f}% / 폭 {cov_w*100:.0f}%")
    if cov_l < 0.5 or cov_w < 0.5:
        print("  ! 선수가 경기장의 절반도 안 쓴다. 영상에서 경기장이 화면을 꽉 채우지"
              " 않는 것 같다. LONG_FROM/TO, CROSS_FROM/TO 를 실제 여백에 맞출 것.")
        print("    이대로 두면 모든 거리가 실제보다 크게 잡혀 압박·수신공간이 전부 틀어진다.")

    if DROP_OUTSIDE > 0:
        m = DROP_OUTSIDE
        inside = out.X.between(-m, PITCH_L + m) & out.Y.between(-m, PITCH_W + m)
        if not inside.all():
            print(f"  경기장 밖 {int((~inside).sum()):,}행 제거 "
                  f"({(~inside).mean()*100:.1f}%)")
            out = out[inside]

    cols = ["frame", "time_s", "track_id", "team", "X", "Y", "cls"]
    out[cols].round(3).to_csv(dst, index=False, encoding="utf-8-sig")
    print(f"[저장] {dst}  {len(out):,}행")
    print()
    print("  기여도 스크립트에서 아래처럼 맞출 것:")
    print(f"    EXCEL_PATH   = \"{dst}\"")
    print(f"    PITCH_L, PITCH_W = {PITCH_L}, {PITCH_W}")
    print(f"    TEAM_SIZE    = {int(pl.groupby(['frame','team']).size().groupby('team').median().max())}")
    print(f"    COORD_ORIGIN = \"corner\"      # 이 파일은 이미 구석 원점이다")
    if fps <= 12:
        print(f"    DELTA        = 1            # {fps} Hz 라 3 이면 속도 창이 "
              f"{2*3/fps:.1f}초로 너무 길다")


if __name__ == "__main__":
    main()
