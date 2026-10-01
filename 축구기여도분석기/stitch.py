"""
끊어진 추적 ID 를 다시 잇는다 — 기여도 계산 전에 한 번 돌리는 손질.

왜 필요한가
-----------
ByteTrack 은 선수가 겹치거나 한 프레임 놓치면 그 자리에서 트랙을 버리고
새 ID 를 딴다. 실측(월드컵 중계 898장, 22명 경기)에서 ID 가 281개 나왔다.
선수는 22명인데 ID 가 281개면 한 사람이 평균 12토막 나 있다는 뜻이고,
그러면 기여도는 "선수의 기여"가 아니라 "토막의 기여"가 된다. 한 토막이
20프레임짜리면 SC·PR·PA 어느 것도 표본이 되지 않아 결과표가 수백 줄로
불어나고 순위는 의미를 잃는다.

무엇을 하는가
-------------
1. 사람이 아닌 트랙을 버린다
   - 경기장 밖에 주로 있는 트랙 (관중석·광고판·중계 그래픽)
   - 오래 살아 있으면서 거의 움직이지 않는 트랙 (관중은 가만히 있다)
2. 남은 트랙을 시간·거리로 이어 붙인다
   A 가 끝난 자리에서 A 의 속도로 굴러갔을 자리와, 그 직후 시작한 B 의
   첫 자리가 가까우면 같은 사람으로 본다. "가깝다"의 기준은 사람이 그
   시간 동안 달릴 수 있는 거리(SPEED_MAX)다 — 순간이동은 잇지 않는다.
3. 팀이 다르면 잇지 않는다. 시간이 겹치면 잇지 않는다 (동시에 두 자리에
   있을 수는 없다).

픽셀이 아니라 미터에서 잰다. 화면 위쪽 먼 선수는 1px 가 0.3m 인데 아래쪽
가까운 선수는 1px 가 0.05m 라, 픽셀 거리로 재면 먼 쪽만 마구 이어진다.
그래서 이 손질은 경기장 보정(호모그래피) 뒤에 돌린다.

되돌릴 수 있게, 원본 tracks.csv 는 건드리지 않고 tracks_stitched.csv 를
새로 쓴다.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

# --- 이어붙이는 기준 -------------------------------------------------------
MAX_GAP_SEC = 2.0      # 이보다 오래 끊겼으면 잇지 않는다
SPEED_MAX = 12.0       # m/s · 사람의 단거리 최고 속도대 (문서 §4 [B])
SLACK_M = 2.5          # 위치 오차 여유 (호모그래피·발끝점 오차)
MAX_JUMP_M = 15.0      # 아무리 길게 끊겼어도 이보다 멀면 잇지 않는다
VEL_SAMPLES = 5        # 속도를 잴 때 쓰는 마지막 표본 수

# --- 사람이 아닌 트랙을 버리는 기준 ---------------------------------------
OUTSIDE_M = 3.0        # 라인 밖 이만큼까지는 경기장 안으로 본다
INSIDE_MIN = 0.5       # 표본의 절반도 경기장 안이 아니면 사람이 아니다
STATIC_SEC = 3.0       # 이보다 오래 살아 있는데
STATIC_M = 2.0         # 이만큼도 안 움직였으면 관중·그래픽이다
MIN_LIFE_SEC = 1.0     # 다 이어붙인 뒤에도 이보다 짧으면 사람으로 안 본다
MIN_SAMPLES = 8        # 그리고 표본이 이보다 적으면 (둘 다일 때만 버린다)

# --- 명단 맞추기 ---------------------------------------------------------
# 경기에 뛰는 사람 수는 정해져 있다. 7v7 이면 14명, 11v11 이면 22명이다.
# 그런데 추적기는 번호를 마음대로 새로 딴다 — 실측에서 22명 경기에 281개가
# 나왔고, 이어붙이기로 101개까지 줄였지만 그래도 선수 수의 다섯 배다.
#
# 다른 접근이 있다. 번호를 '발급' 하지 말고 '정해 놓고 배정' 하는 것이다.
# 팀마다 칸을 정원만큼 만들어 두고, 토막들을 그 칸에 채워 넣는다. 칸이
# 모자라면 더 만들지 않고 남는 토막을 버린다. 그러면 결과는 반드시 정원이다.
#
# 단, 이게 성립하려면 '모든 선수가 화면에 보여야' 한다. 카메라가 따라다니며
# 일부만 비추면, 화면 밖에 있던 선수의 토막을 엉뚱한 칸에 밀어 넣게 된다.
# 그래서 프레임당 인원이 정원에 가까울 때만 켠다.
ROSTER_GAP_SEC = 10.0  # 칸을 채울 때는 이 정도 공백까지 건너뛴다
ROSTER_NEED = 0.85     # 프레임당 인원이 정원의 이만큼은 돼야 명단을 맞춘다


def apply_h(H, x, y):
    """픽셀 (x, y) 를 호모그래피로 미터 좌표에 옮긴다."""
    H = np.asarray(H, dtype=float)
    p = np.stack([np.asarray(x, float), np.asarray(y, float),
                  np.ones(len(np.atleast_1d(x)))])
    q = H @ p
    w = np.where(np.abs(q[2]) < 1e-12, np.nan, q[2])
    return q[0] / w, q[1] / w


class _Track:
    """한 ID 의 요약 — 이어붙일지 판단하는 데 필요한 것만 담는다."""

    __slots__ = ("tid", "t0", "t1", "f0", "f1", "p0", "p1", "vel",
                 "team", "n", "inside", "span")

    def __init__(self, tid, frames, times, X, Y, team):
        order = np.argsort(frames)
        frames, times, X, Y = (a[order] for a in (frames, times, X, Y))
        self.tid = tid
        self.n = len(frames)
        self.f0, self.f1 = int(frames[0]), int(frames[-1])
        self.t0, self.t1 = float(times[0]), float(times[-1])
        self.p0 = (float(X[0]), float(Y[0]))
        self.p1 = (float(X[-1]), float(Y[-1]))
        self.team = team
        # 마지막 몇 장으로 속도를 잰다 (m/s). 표본이 적으면 0 으로 둔다.
        k = min(VEL_SAMPLES, self.n)
        dt = float(times[-1] - times[-k]) if k > 1 else 0.0
        if dt > 1e-6:
            vx = (X[-1] - X[-k]) / dt
            vy = (Y[-1] - Y[-k]) / dt
            sp = math.hypot(vx, vy)
            if sp > SPEED_MAX:            # 튀는 값은 최고 속도로 눌러 둔다
                vx, vy = vx * SPEED_MAX / sp, vy * SPEED_MAX / sp
            self.vel = (vx, vy)
        else:
            self.vel = (0.0, 0.0)


def _summarize(df, H, pitch_l, pitch_w):
    """트랙별 요약과, 버릴 트랙 목록을 만든다."""
    X, Y = apply_h(H, df["_px"].to_numpy(), df["_py"].to_numpy())
    df = df.assign(_X=X, _Y=Y)
    ok = np.isfinite(X) & np.isfinite(Y)
    df = df[ok]

    tracks, junk_out, junk_static = {}, [], []
    out_only = {}          # 밖이라 뺀 것들 — 보정이 의심스러우면 되돌린다
    for tid, g in df.groupby("track_id", sort=False):
        gx, gy = g["_X"].to_numpy(), g["_Y"].to_numpy()
        inside = np.mean((gx > -OUTSIDE_M) & (gx < pitch_l + OUTSIDE_M)
                         & (gy > -OUTSIDE_M) & (gy < pitch_w + OUTSIDE_M))
        life = float(g["_t"].max() - g["_t"].min())
        span = math.hypot(float(gx.max() - gx.min()), float(gy.max() - gy.min()))
        if life >= STATIC_SEC and span < STATIC_M:
            junk_static.append(tid)
            continue
        team = None
        if "team" in g.columns:
            vals = g["team"].dropna().astype(str)
            vals = vals[~vals.isin(("", "-1", "-1.0", "nan", "None"))]
            if len(vals):
                team = vals.mode().iloc[0]
        t = _Track(tid, g["frame"].to_numpy(), g["_t"].to_numpy(), gx, gy, team)
        t.inside, t.span = float(inside), span
        if inside < INSIDE_MIN:
            junk_out.append(tid)
            out_only[tid] = t
        else:
            tracks[tid] = t
    return tracks, junk_out, junk_static, out_only


def _pairs(tracks):
    """이을 수 있는 (비용, A, B) 후보를 모두 모은다."""
    order = sorted(tracks.values(), key=lambda t: t.t0)
    out = []
    for i, a in enumerate(order):
        for b in order[i + 1:]:
            gap = b.t0 - a.t1
            if gap <= 0:                     # 시간이 겹친다 = 다른 사람
                continue
            if gap > MAX_GAP_SEC:            # 뒤는 더 멀다 — 이 A 는 여기까지
                break
            if a.team and b.team and a.team != b.team:
                continue
            # A 가 하던 대로 굴러갔을 자리
            px = a.p1[0] + a.vel[0] * gap
            py = a.p1[1] + a.vel[1] * gap
            d = math.hypot(b.p0[0] - px, b.p0[1] - py)
            reach = min(SPEED_MAX * gap + SLACK_M, MAX_JUMP_M)
            if d > reach:
                continue
            # 비용: 예측 오차를 '갈 수 있던 거리'로 나눈 값. 짧게 끊긴 쪽을
            # 먼저 잇도록 끊긴 시간을 조금 더한다.
            out.append((d / max(reach, 1e-6) + 0.1 * gap, a.tid, b.tid))
    out.sort(key=lambda r: r[0])
    return out


def _link(tracks, pairs):
    """싼 것부터 이어 붙인다. 한 트랙의 앞뒤는 각각 하나씩만."""
    nxt, prv = {}, {}
    head = {t: t for t in tracks}            # 체인의 첫 트랙
    tail = {t: t for t in tracks}            # 체인의 마지막 트랙
    for _cost, a, b in pairs:
        if a in nxt or b in prv:
            continue
        if head[a] == head[b]:               # 이미 같은 체인
            continue
        # 체인끼리 시간이 겹치면 안 된다
        if tracks[tail[head[b]]].t1 <= tracks[a].t1:
            continue
        nxt[a], prv[b] = b, a
        h, t = head[a], tail[head[b]]
        # 체인 b 전체의 head 를 a 의 head 로, 체인 a 전체의 tail 을 b 의 tail 로
        cur = head[b]
        while True:
            head[cur] = h
            if cur == t:
                break
            cur = nxt[cur]
        cur = h
        while True:
            tail[cur] = t
            if cur == a:
                break
            cur = nxt[cur]
    return head


def _chains(tracks, remap):
    """체인(이어붙인 한 묶음)별로 시간순 멤버와 양 끝 상태를 모은다."""
    byroot = {}
    for tid, root in remap.items():
        byroot.setdefault(root, []).append(tracks[tid])
    out = {}
    for root, members in byroot.items():
        members.sort(key=lambda t: t.t0)
        out[root] = {
            "members": members,
            "t0": members[0].t0, "t1": members[-1].t1,
            "p0": members[0].p0, "p1": members[-1].p1,
            "vel": members[-1].vel,
            "n": sum(m.n for m in members),
            "team": next((m.team for m in members if m.team), None),
        }
    return out


def _roster(chains, team_size, log):
    """
    팀마다 정원만큼 칸을 만들고 체인을 채워 넣는다.

    긴 체인부터 칸의 주인이 되고, 나머지는 시간이 겹치지 않고 위치가 닿는
    칸에 들어간다. 못 들어가면 버린다 — 칸을 늘리지 않는 것이 핵심이다.

    돌려주는 것: {체인 루트 -> 칸 번호}, 못 넣은 체인 목록
    """
    assign, dropped = {}, []
    teams = sorted({c["team"] for c in chains.values() if c["team"]})
    letter = {t: chr(ord("A") + i) for i, t in enumerate(teams)}
    for team in teams:
        mine = {r: c for r, c in chains.items() if c["team"] == team}
        order = sorted(mine, key=lambda r: -mine[r]["n"])
        slots = []                      # 칸마다 [채워 넣은 체인들] (시간순)
        for root in order:
            c = mine[root]
            if len(slots) < team_size:
                slots.append([root])
                assign[root] = (letter[team], len(slots) - 1)
                continue
            best, best_cost = None, None
            for k, slot in enumerate(slots):
                # 시간이 겹치면 같은 사람일 수 없다
                if any(not (c["t1"] < chains[o]["t0"] or c["t0"] > chains[o]["t1"])
                       for o in slot):
                    continue
                # 바로 앞/뒤 체인과 이어 붙일 수 있는가
                before = [o for o in slot if chains[o]["t1"] < c["t0"]]
                after = [o for o in slot if chains[o]["t0"] > c["t1"]]
                cost = None
                if before:
                    a = chains[max(before, key=lambda o: chains[o]["t1"])]
                    gap = c["t0"] - a["t1"]
                    if gap <= ROSTER_GAP_SEC:
                        px = a["p1"][0] + a["vel"][0] * min(gap, 1.0)
                        py = a["p1"][1] + a["vel"][1] * min(gap, 1.0)
                        d = math.hypot(c["p0"][0] - px, c["p0"][1] - py)
                        if d <= SPEED_MAX * gap + SLACK_M:
                            cost = d / max(SPEED_MAX * gap + SLACK_M, 1e-6)
                if cost is None and after:
                    b = chains[min(after, key=lambda o: chains[o]["t0"])]
                    gap = b["t0"] - c["t1"]
                    if gap <= ROSTER_GAP_SEC:
                        d = math.hypot(b["p0"][0] - c["p1"][0],
                                       b["p0"][1] - c["p1"][1])
                        if d <= SPEED_MAX * gap + SLACK_M:
                            cost = d / max(SPEED_MAX * gap + SLACK_M, 1e-6)
                if cost is not None and (best_cost is None or cost < best_cost):
                    best, best_cost = k, cost
            if best is None:
                dropped.append(root)
            else:
                slots[best].append(root)
                assign[root] = (letter[team], best)
    return assign, dropped


def stitch(csv_path, H, pitch_l=105.0, pitch_w=68.0, log=print,
           out_path=None, team_size=0) -> tuple[Path, dict]:
    """
    tracks.csv 를 손질해 tracks_stitched.csv 를 쓴다.
    (새 파일 경로, 보고서) 를 돌려준다. 손질할 게 없으면 원본 경로를 준다.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    report = {"before": 0, "after": 0, "dropped_outside": 0,
              "dropped_static": 0, "dropped_short": 0, "merged": 0,
              "rows_dropped": 0, "roster": 0, "dropped_roster": 0}
    if "track_id" not in df.columns or not len(df):
        return csv_path, report

    # 사람 행만 손질한다. 공은 BallPicker 가 따로 맡으므로 그대로 둔다.
    # 역할 열 이름은 만든 쪽마다 다르다 (tracker.csv 는 class_name).
    ccol = next((c for c in ("cls", "class_name", "class", "label", "name")
                 if c in df.columns), None)
    cls = (df[ccol].astype(str).str.lower() if ccol
           else pd.Series([""] * len(df), index=df.index))
    is_ball = cls.str.contains("ball", na=False)
    people = df[~is_ball].copy()
    if not len(people):
        return csv_path, report

    # 대표점: 발끝(fx, fy) 이 있으면 그걸, 없으면 상자 중심
    px = "fx" if "fx" in people.columns else "cx"
    py = "fy" if "fy" in people.columns else "cy"
    people["_px"] = pd.to_numeric(people[px], errors="coerce")
    people["_py"] = pd.to_numeric(people[py], errors="coerce")
    tcol = "time_sec" if "time_sec" in people.columns else None
    people["_t"] = (pd.to_numeric(people[tcol], errors="coerce") if tcol
                    else pd.to_numeric(people["frame"], errors="coerce") / 25.0)
    people = people.dropna(subset=["_px", "_py", "_t"])
    if not len(people):
        return csv_path, report

    report["before"] = int(people["track_id"].nunique())
    tracks, junk_out, junk_static, out_only = _summarize(people, H, pitch_l, pitch_w)
    # 보정이 틀리면 '경기장 밖' 판정이 통째로 엉망이 된다. 실제로 길이
    # 방향이 어긋난 보정에서 트랙 전부가 밖으로 나와 다 지워졌다.
    # 절반 넘게 밖이면 그건 사람이 밖에 있는 게 아니라 보정이 틀린 것이다.
    # 그럴 땐 위치로 거르는 것을 포기하고, 시간·거리로 잇는 일만 한다.
    total = len(tracks) + len(junk_out) + len(junk_static)
    if total and len(junk_out) > 0.5 * total:
        log(f"  ! 트랙의 {len(junk_out)/total*100:.0f}% 가 경기장 밖으로 나옵니다 — "
            f"사람이 아니라 보정이 틀린 것입니다. 위치로 거르는 건 건너뛰고 "
            f"끊어진 ID 잇기만 합니다. calib_check.png 를 확인하세요.")
        tracks.update(out_only)
        junk_out = []
    report["dropped_outside"] = len(junk_out)
    report["dropped_static"] = len(junk_static)
    if not tracks:
        log("  ! 손질할 트랙이 남지 않았습니다 — 원본을 그대로 씁니다")
        return csv_path, report

    head = _link(tracks, _pairs(tracks))
    # 체인의 첫 트랙 ID 를 대표로 삼는다
    remap = {tid: head[tid] for tid in tracks}
    merged_to = len(set(remap.values()))
    report["merged"] = len(remap) - merged_to

    # 다 이어붙이고도 너무 짧은 체인은 선수가 아니다.
    # 실측에서 트랙의 25%가 3프레임 이하였다 — 관중석 한 번 반짝한 오검출이다.
    # 이런 것들이 결과표를 수백 줄로 불리면서 정작 점수는 못 받는다
    # (엔진이 '표본 부족으로 제외' 하는 바로 그 줄들이다). 여기서 지운다.
    # 이어붙인 뒤에 재는 것이 중요하다 — 20프레임짜리 세 토막은 살아남는다.
    chains: dict = {}
    for tid, root in remap.items():
        t = tracks[tid]
        c = chains.setdefault(root, [t.t0, t.t1, 0])
        c[0] = min(c[0], t.t0)
        c[1] = max(c[1], t.t1)
        c[2] += t.n
    short = {r for r, (t0, t1, n) in chains.items()
             if (t1 - t0) < MIN_LIFE_SEC and n < MIN_SAMPLES}
    junk_short = [tid for tid, root in remap.items() if root in short]
    report["dropped_short"] = len(junk_short)
    report["after"] = merged_to - len(short)

    # --- 명단 맞추기 (선택) ------------------------------------------
    # 화면에 전원이 보이는 영상에서만 켠다. 카메라가 일부만 비추면
    # 화면 밖에 있던 선수의 토막을 엉뚱한 칸에 밀어 넣게 된다.
    if team_size and report["after"] > 2 * int(team_size):
        live = float(people.groupby("frame").size().median())
        need = ROSTER_NEED * 2 * int(team_size)
        if live < need:
            log(f"  명단 맞추기는 건너뜁니다 — 프레임당 인원 {live:.0f}명이 "
                f"정원 {2*int(team_size)}명에 못 미칩니다 "
                f"(카메라가 경기장 일부만 비추는 영상이다)")
        else:
            chains = {r: c for r, c in _chains(tracks, remap).items()
                      if r not in short}
            assign, roster_drop = _roster(chains, int(team_size), log)
            if assign:
                # 체인 루트 -> 'A1' / 'B3' 같은 칸 이름
                name = {root: f"{team}{k + 1}" for root, (team, k) in assign.items()}
                drop_roots = set(roster_drop)
                junk_roster = [tid for tid, root in remap.items()
                               if root in drop_roots]
                remap = {tid: name.get(root, root)
                         for tid, root in remap.items()
                         if root not in drop_roots}
                report["after"] = len(set(remap.values()))
                report["roster"] = int(team_size)
                report["dropped_roster"] = len(roster_drop)
                junk_short = list(junk_short) + junk_roster
                free = sorted(v for v in set(remap.values())
                              if not str(v)[:1].isalpha())
                log(f"  명단 맞추기: 팀마다 {team_size}칸을 두고 채워 "
                    f"A1~A{team_size} · B1~B{team_size} 로 번호를 줬습니다 "
                    f"(칸에 못 들어간 토막 {len(roster_drop)}개는 버림)")
                if free:
                    log(f"    팀이 안 붙은 토막 {len(free)}개는 명단 밖으로 "
                        f"남겼습니다 (대개 골키퍼다 — 번호가 그대로다)")

    drop = set(junk_out) | set(junk_static) | set(junk_short)
    # 명단 맞추기로 remap 에서 빠진 트랙도 버린다
    drop |= {tid for tid in tracks if tid not in remap}
    keep = ~(df["track_id"].isin(drop) & ~is_ball)
    report["rows_dropped"] = int((~keep).sum())
    out = df[keep].copy()
    # 명단을 맞추면 track_id 가 'A1' 같은 글자가 된다. 숫자 열에 글자를
    # 넣을 수 없으므로 열 자체를 글자로 올린 뒤에 바꾼다.
    m = (~is_ball[keep]).to_numpy()
    new_ids = out.loc[m, "track_id"].map(remap).fillna(out.loc[m, "track_id"])
    if new_ids.map(lambda v: isinstance(v, str)).any():
        out["track_id"] = out["track_id"].astype(object)
    out.loc[m, "track_id"] = new_ids

    out_path = Path(out_path) if out_path else csv_path.with_name("tracks_stitched.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    log(f"  ID 손질: {report['before']}개 -> {report['after']}개"
        f"  (이어붙임 {report['merged']} · 버림: 너무 짧음 {report['dropped_short']} · "
        f"경기장 밖 {report['dropped_outside']} · 안 움직임 {report['dropped_static']})")
    if report["dropped_static"]:
        log(f"    안 움직이는 트랙은 관중석·광고판·중계 그래픽이다 "
            f"({STATIC_SEC:.0f}초 넘게 살아 있으면서 {STATIC_M:.0f}m 도 안 움직인 것)")
    return out_path, report


def self_check() -> list[str]:
    """가짜 데이터로 이어붙이기가 맞게 도는지 본다. 틀린 항목을 돌려준다."""
    bad = []
    H = np.eye(3)                       # 픽셀 = 미터 인 척한다
    rows = []
    fps = 25.0

    def add(tid, f_from, f_to, x0, y0, vx, vy):
        for f in range(f_from, f_to):
            rows.append(dict(frame=f, time_sec=f / fps, track_id=tid, cls="player",
                             conf=0.9, cx=0, cy=0,
                             fx=x0 + vx * (f - f_from) / fps,
                             fy=y0 + vy * (f - f_from) / fps, team=0))

    # 한 사람이 세 토막 난 경우 — 5 m/s 로 오른쪽으로 달린다
    add(1, 0, 50, 10.0, 30.0, 5.0, 0.0)
    add(2, 55, 100, 10.0 + 5.0 * (55 / fps), 30.0, 5.0, 0.0)
    add(3, 104, 150, 10.0 + 5.0 * (104 / fps), 30.0, 5.0, 0.0)
    # 다른 사람 — 시간이 겹치므로 절대 이어지면 안 된다
    add(4, 0, 150, 60.0, 50.0, 0.0, 3.0)
    # 관중 — 오래 살아 있는데 안 움직인다
    add(5, 0, 150, 80.0, 66.0, 0.0, 0.0)
    # 광고판 — 경기장 밖
    add(6, 0, 150, 80.0, 90.0, 0.0, 1.0)

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "tracks.csv"
        pd.DataFrame(rows).to_csv(src, index=False)
        out, rep = stitch(src, H, 105.0, 68.0, log=lambda *_: None)
        got = pd.read_csv(out)
        ids = sorted(got["track_id"].unique())
        if rep["dropped_static"] != 1:
            bad.append(f"안 움직이는 트랙 {rep['dropped_static']}개 (기대 1)")
        if rep["dropped_outside"] != 1:
            bad.append(f"경기장 밖 트랙 {rep['dropped_outside']}개 (기대 1)")
        if len(ids) != 2:
            bad.append(f"이어붙인 뒤 ID {len(ids)}개 (기대 2: 달리는 사람 + 겹치는 사람)")
        n1 = int((got["track_id"] == 1).sum())
        if n1 != 50 + 45 + 46:
            bad.append(f"세 토막이 하나로 안 모였다 ({n1}행, 기대 141)")
        if 4 not in ids:
            bad.append("시간이 겹치는 다른 사람이 잘못 합쳐졌다")
    return bad


if __name__ == "__main__":
    bad = self_check()
    print("ID 손질 자체검사:", "이상 없음" if not bad else "틀림 — " + " / ".join(bad))
