"""
경기장 자동 보정 — 영상만 보고 픽셀을 미터로 바꾸는 변환을 스스로 찾는다.

왜 필요한가
    영상에서 뽑은 좌표는 화면 픽셀이다. 카메라가 비스듬히 찍으므로 화면
    위쪽의 100픽셀과 아래쪽의 100픽셀은 실제 거리가 다르다. 기여도 수식의
    상수는 전부 미터 기준이라 이 변환 없이는 계산이 성립하지 않는다.

어떻게 찾나 — 두 가지를 순서대로 해 보고 되는 것을 쓴다
    1) 흰 선으로 찾기 (find_by_lines)
       잔디를 찾고, 그 안의 흰 선을 뽑아, 경기장을 감싸는 네 개의 바깥 선을
       고른다. 네 선이 만나는 네 점이 경기장 네 모서리다. 전술 카메라처럼
       경기장 전체가 보이는 화면에서 잘 된다.

    2) 선수가 퍼진 범위로 찾기 (find_by_players)
       1)이 실패했을 때. 경기 내내 선수들이 밟고 다닌 자리를 모으면 그 모양이
       대략 보이는 경기장이다. 그 사각형을 경기장 직사각형에 맞춘다.
       어림값이라 결과에 '대략'이라고 표시하고 경고한다.

어느 쪽이든 마지막에 반드시 검산한다 (score_homography).
선수 발밑 점을 변환했을 때 경기장 안에 들어오는 비율과, 퍼진 범위가
실제 경기장 크기와 얼마나 맞는지를 본다. 점수가 낮으면 쓰지 않는다.
"""

from __future__ import annotations

import math

import numpy as np

# 잔디로 볼 HSV 색상 범위 (추적 프로그램과 같은 값)
GRASS_HUE = (30, 95)
GRASS_SAT_MIN = 30


# ────────────────────────────────────────────── 기본 도구
def homography(src, dst):
    """네 쌍 이상의 대응점에서 3x3 변환 행렬을 구한다 (DLT)."""
    src = np.asarray(src, float).reshape(-1, 2)
    dst = np.asarray(dst, float).reshape(-1, 2)
    if len(src) < 4 or len(src) != len(dst):
        return None
    A = []
    for (x, y), (u, v) in zip(src, dst):
        A += [[-x, -y, -1, 0, 0, 0, u * x, u * y, u],
              [0, 0, 0, -x, -y, -1, v * x, v * y, v]]
    try:
        _, _, Vt = np.linalg.svd(np.asarray(A, float))
    except np.linalg.LinAlgError:
        return None
    H = Vt[-1].reshape(3, 3)
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12:
        return None
    return H / H[2, 2]


def apply_h(H, x, y):
    """픽셀 -> 미터."""
    x = np.asarray(x, float).ravel()
    y = np.asarray(y, float).ravel()
    p = np.stack([x, y, np.ones(len(x))])
    q = np.asarray(H, float) @ p
    w = np.where(np.abs(q[2]) < 1e-12, np.nan, q[2])
    return q[0] / w, q[1] / w


def order_quad(pts):
    """네 점을 좌하-우하-우상-좌상 순서로 돌려준다 (화면 기준, y 는 아래가 큼)."""
    p = np.asarray(pts, float).reshape(-1, 2)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    p = p[np.argsort(-ang)]                     # 시계 반대 방향
    start = int(np.argmax(p[:, 1] - p[:, 0]))   # 좌하(아래·왼쪽)에서 시작
    return np.roll(p, -start, axis=0)


# ────────────────────────────────────────────── 검산
def inside_quad(quad, pts):
    """
    사각형 안에 있는 점의 비율. 볼록사각형이므로 네 변의 같은 쪽에 있으면 안이다.

    경기장 선 픽셀이 이 사각형 안에 얼마나 들어오는지를 재는 데 쓴다.
    페널티박스 선을 경기장 경계로 잘못 고르면, 진짜 골라인 픽셀이 밖으로
    나가기 때문에 이 값이 떨어진다. 선수 위치만으로는 구별되지 않는 차이다.
    """
    q = np.asarray(quad, float).reshape(4, 2)
    p = np.asarray(pts, float).reshape(-1, 2)
    if len(p) == 0:
        return 0.0
    sign = None
    inside = np.ones(len(p), bool)
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        cross = (b[0] - a[0]) * (p[:, 1] - a[1]) - (b[1] - a[1]) * (p[:, 0] - a[0])
        if sign is None:
            sign = 1.0 if np.sum(cross) >= 0 else -1.0
        inside &= (cross * sign) >= -1e-6
    return float(inside.mean())


# ─────────────────────────────── 실제 경기장 선 모양으로 검산
# 경기장 선의 실제 치수 (FIFA 규격, 원점은 한쪽 골대 왼쪽 모서리).
# 여기 있는 숫자는 규칙서 그대로다 — 고를 여지가 없는 값이다.
PEN_DEPTH, PEN_HALF = 16.5, 20.16     # 페널티 구역 (깊이, 반폭)
GOAL_DEPTH, GOAL_HALF = 5.5, 9.16     # 골 구역
CIRCLE_R = 9.15                       # 센터서클 · 페널티 아크 반지름
PEN_SPOT = 11.0                       # 페널티 마크까지
TPL_STEP = 0.25                       # 본뜨는 격자 간격 (m/칸)
TPL_TOL = 1.5                         # 이만큼 안에 들면 '선 위'로 본다 (m)

_TPL_CACHE: dict = {}


def pitch_lines(pitch_l, pitch_w):
    """실제 경기장 선 위의 점들을 미터 좌표로 늘어놓는다."""
    pts = []

    def seg(x0, y0, x1, y1):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / TPL_STEP))
        t = np.linspace(0, 1, n)
        pts.append(np.stack([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t], 1))

    def arc(cx, cy, r, a0, a1):
        n = max(8, int(abs(a1 - a0) * r / TPL_STEP))
        a = np.linspace(a0, a1, n)
        pts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1))

    L, W, cy = pitch_l, pitch_w, pitch_w / 2
    seg(0, 0, L, 0); seg(0, W, L, W)                  # 터치라인
    seg(0, 0, 0, W); seg(L, 0, L, W)                  # 골라인
    seg(L / 2, 0, L / 2, W)                           # 하프라인
    arc(L / 2, cy, CIRCLE_R, 0, 2 * math.pi)          # 센터서클
    for sx, x0 in ((1, 0.0), (-1, L)):                # 양쪽 페널티·골 구역
        for d, hh in ((PEN_DEPTH, PEN_HALF), (GOAL_DEPTH, GOAL_HALF)):
            x1 = x0 + sx * d
            seg(x1, cy - hh, x1, cy + hh)
            seg(x0, cy - hh, x1, cy - hh)
            seg(x0, cy + hh, x1, cy + hh)
        # 페널티 아크 — 구역 밖으로 나오는 부분만
        spot = x0 + sx * PEN_SPOT
        a = math.acos((PEN_DEPTH - PEN_SPOT) / CIRCLE_R)
        base = 0.0 if sx > 0 else math.pi
        arc(spot, cy, CIRCLE_R, base - a, base + a)
    return np.concatenate(pts, 0)


def pitch_distance_map(pitch_l, pitch_w):
    """미터 좌표 -> 가장 가까운 경기장 선까지의 거리(m). (맵, 격자간격, 여백)"""
    key = (round(pitch_l, 2), round(pitch_w, 2))
    if key in _TPL_CACHE:
        return _TPL_CACHE[key]
    pad = 8.0
    step = TPL_STEP
    nx = int((pitch_l + 2 * pad) / step) + 1
    ny = int((pitch_w + 2 * pad) / step) + 1
    grid = np.full((ny, nx), 1e9, float)
    pts = pitch_lines(pitch_l, pitch_w)
    ix = np.clip(((pts[:, 0] + pad) / step).astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] + pad) / step).astype(int), 0, ny - 1)
    grid[iy, ix] = 0.0
    # 거리 변환. scipy 가 있으면 정확하게, 없으면 두 번 훑는 근사로.
    try:
        from scipy.ndimage import distance_transform_edt
        dist = distance_transform_edt(grid > 0) * step
    except ImportError:
        dist = _chamfer(grid > 0) * step
    out = (dist, step, pad)
    _TPL_CACHE[key] = out
    return out


def _chamfer(free):
    """scipy 없을 때 쓰는 거리 근사 (앞뒤로 한 번씩 훑는다)."""
    big = 1e6
    d = np.where(free, big, 0.0)
    ny, nx = d.shape
    for y in range(ny):
        for x in range(nx):
            v = d[y, x]
            if y: v = min(v, d[y - 1, x] + 1)
            if x: v = min(v, d[y, x - 1] + 1)
            if y and x: v = min(v, d[y - 1, x - 1] + 1.414)
            d[y, x] = v
    for y in range(ny - 1, -1, -1):
        for x in range(nx - 1, -1, -1):
            v = d[y, x]
            if y < ny - 1: v = min(v, d[y + 1, x] + 1)
            if x < nx - 1: v = min(v, d[y, x + 1] + 1)
            if y < ny - 1 and x < nx - 1: v = min(v, d[y + 1, x + 1] + 1.414)
            d[y, x] = v
    return d


def line_distance_map(lines, cv2):
    """화면의 각 픽셀에서 가장 가까운 '찾은 흰 선'까지의 거리(px)."""
    return cv2.distanceTransform((lines == 0).astype(np.uint8), cv2.DIST_L2, 3)


def grass_inside(H, mask, pitch_l, pitch_w, margin=5.0, n=4000):
    """
    화면의 잔디가 '경기장 직사각형 안' 으로 옮겨지는 비율.

    보정이 맞았는지 가리는 가장 센 단서다. 화면에 보이는 녹색은 거의 다
    경기장이므로, 보정이 맞으면 그 점들이 105 x 68 안에 들어와야 한다.

    경기장 선을 얼마나 덮었나(template_score)만 보면 속는다. 경기장을
    화면 한쪽 띠로 뭉개 넣는 보정은 그 띠에 선 비슷한 것(광고판·관중석
    가장자리·터치라인)이 많아 덮은 비율이 높게 나온다. 실제로 그렇게 해서
    0.75 를 받은 보정이 경기장 전체를 오른쪽 관중석 쪽에 구겨 넣었다.
    그런데 그러면 화면 대부분을 차지하는 잔디가 경기장 밖으로 밀려난다.
    실측: 손으로 맞춘 보정 90% · 뭉개진 보정 39%.
    """
    if H is None or mask is None:
        return 0.0
    ys, xs = np.nonzero(mask)
    if len(xs) < 100:
        return 0.0
    if len(xs) > n:
        k = np.linspace(0, len(xs) - 1, n).astype(int)
        xs, ys = xs[k], ys[k]
    X, Y = apply_h(H, xs.astype(float), ys.astype(float))
    ok = np.isfinite(X) & np.isfinite(Y)
    if ok.sum() < 50:
        return 0.0
    X, Y = X[ok], Y[ok]
    return float(np.mean((X > -margin) & (X < pitch_l + margin)
                         & (Y > -margin) & (Y < pitch_w + margin)))


def sane_homography(H, shape, pitch_l, pitch_w):
    """
    화면 전체가 말이 되는 넓이의 경기장으로 옮겨지는가.

    찌부러진 답을 막는 장치다. 화면을 미터 좌표의 가느다란 띠 하나로
    몰아넣는 변환은 '경기장 선을 얼마나 덮었나' 를 1.0 가까이 받는다 —
    그 띠 안에 선 픽셀이 잔뜩 있기 때문이다. 실제로 다듬기 단계가 그런
    답으로 흘러가 점수 0.99 를 받고 센터서클을 X=15m 에 갖다 놓았다.
    화면 네 귀퉁이가 옮겨간 사각형의 가로·세로·넓이를 보면 바로 걸린다.
    """
    if H is None:
        return False
    H = np.asarray(H, float)
    h, w = shape[:2]
    c = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    # 지평선이 화면을 가로지르면 안 된다. H 의 3행이 0 이 되는 선이
    # 지평선이고, 그 너머는 '카메라 뒤' 라 좌표가 뒤집히며 수백 m 로
    # 튄다. 네 귀퉁이의 부호가 같아야 화면 전체가 지평선 한쪽에 있다.
    wq = H[2, 0] * c[:, 0] + H[2, 1] * c[:, 1] + H[2, 2]
    if np.ptp(np.sign(wq)) > 0 or np.min(np.abs(wq)) < 1e-9:
        return False
    X, Y = apply_h(H, c[:, 0], c[:, 1])
    if not (np.isfinite(X).all() and np.isfinite(Y).all()):
        return False
    if np.ptp(X) < 20.0 or np.ptp(Y) < 20.0:
        return False
    area = 0.5 * abs(np.dot(X, np.roll(Y, -1)) - np.dot(Y, np.roll(X, -1)))
    ref = pitch_l * pitch_w
    return 0.15 * ref < area < 6.0 * ref


def template_score(H, dist_px, shape, pitch_l, pitch_w, tol=3.0):
    """
    실제 경기장 선을 화면에 되돌려 그렸을 때, 찾은 흰 선 위에 얼마나 얹히나.

    방향이 중요하다. 반대로 ('찾은 흰 선이 경기장 선 근처인가') 재면 안 된다.
    화면에서 찾은 '선' 의 대부분은 잔디 깎은 줄무늬·광고판·관중석 가장자리라
    진짜 경기장 선이 아니고, 화면을 좁게 뭉개는 엉터리 보정일수록 그 잡동사니가
    우연히 무슨 선 근처엔가 떨어진다. 실제로 그렇게 재 보니 정답 보정(0.30)이
    90도 돌아간 엉터리 보정(0.34)보다 낮게 나왔다.

    이쪽 방향은 속지 않는다. 센터서클·페널티박스·하프라인은 실제로 거기
    있어야만 덮이기 때문이다. 보정이 틀리면 원이 엉뚱한 자리에 그려진다.
    """
    if H is None or dist_px is None:
        return 0.0
    if not sane_homography(H, shape, pitch_l, pitch_w):
        return 0.0
    h, w = shape[:2]
    tpl = pitch_lines(pitch_l, pitch_w)
    try:
        Hi = np.linalg.inv(np.asarray(H, float))
    except np.linalg.LinAlgError:
        return 0.0
    q = Hi @ np.stack([tpl[:, 0], tpl[:, 1], np.ones(len(tpl))])
    ok = np.abs(q[2]) > 1e-9
    if ok.sum() < 50:
        return 0.0
    x, y = q[0][ok] / q[2][ok], q[1][ok] / q[2][ok]
    vis = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    # 화면에 보이는 경기장 선이 너무 적으면 판단할 수 없다 (덮은 비율이
    # 우연히 높게 나오기 쉽다)
    if vis.sum() < 200:
        return 0.0
    # 찌부러진 답 막기. 경기장을 화면의 좁은 띠 하나에 몰아넣는 변환은
    # 그 띠에 선 픽셀이 많으면 덮은 비율이 1.0 에 가깝게 나온다. 실제로
    # 다듬기 단계가 그런 답으로 흘러가 점수 0.99 를 받은 적이 있다.
    # 경기장이 화면에서 차지하는 폭과 높이가 충분해야 점수를 준다.
    xv, yv = x[vis], y[vis]
    if (np.ptp(xv) < 0.25 * w) or (np.ptp(yv) < 0.25 * h):
        return 0.0
    d = dist_px[y[vis].astype(int), x[vis].astype(int)]
    return float(np.mean(d <= tol))


def refine_to_lines(H, line_pts, dist_px, shape, pitch_l, pitch_w, rounds=10):
    """
    거친 보정을 실제 경기장 선 위로 당겨 붙인다 (ICP).

    네 개의 바깥 선만으로 잡은 보정은 방향은 맞아도 몇 미터씩 어긋난다.
    선 하나를 잘못 집으면 센터서클이 10m 씩 밀린다. 그래서 찾은 흰 선을
    미터로 옮긴 다음, 각 점을 '가장 가까운 실제 경기장 선 위의 점'에
    짝지어 다시 풀기를 반복한다. 짝을 찾는 반경을 회를 거듭하며 좁혀
    가까운 짝만 남기므로, 잘못 집힌 선은 저절로 빠진다.

    더 나아지지 않으면 원래 것을 그대로 돌려준다.
    """
    if H is None or line_pts is None or len(line_pts) < 30:
        return H, 0.0
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return H, template_score(H, dist_px, shape, pitch_l, pitch_w)
    tpl = pitch_lines(pitch_l, pitch_w)
    tree = cKDTree(tpl)
    best_H = np.asarray(H, float)
    best_s = template_score(best_H, dist_px, shape, pitch_l, pitch_w)
    cur = best_H
    for r in range(rounds):
        X, Y = apply_h(cur, line_pts[:, 0], line_pts[:, 1])
        ok = np.isfinite(X) & np.isfinite(Y)
        if ok.sum() < 20:
            break
        d, idx = tree.query(np.stack([X[ok], Y[ok]], 1))
        # 짝 지을 반경을 6m 에서 1m 까지 좁혀 간다
        rad = 6.0 * (1.0 - r / max(1, rounds - 1)) + 1.0
        take = d < rad
        if take.sum() < 12:
            break
        src = line_pts[ok][take]
        dst = tpl[idx[take]]
        nxt = homography(src, dst)
        if nxt is None:
            break
        sc = template_score(nxt, dist_px, shape, pitch_l, pitch_w)
        if sc > best_s + 1e-4:
            best_H, best_s = nxt, sc
        cur = nxt
    return best_H, best_s


def score_homography(H, px, py, pitch_l, pitch_w):
    """
    변환이 말이 되는지 0~1 로 점수를 매긴다.

    보는 것 세 가지
      · 안에 들어오는 비율 — 선수 발밑이 경기장 안에 얼마나 떨어지는가
      · 가로 퍼짐 / 세로 퍼짐 — 실제 경기장 크기와 얼마나 맞는가
    선수들이 경기장 전체를 쓰는 것은 아니므로 퍼짐은 60~105% 면 좋게 본다.
    """
    if H is None or len(px) == 0:
        return 0.0, {}
    X, Y = apply_h(H, px, py)
    ok = np.isfinite(X) & np.isfinite(Y)
    if ok.sum() < 10:
        return 0.0, {}
    X, Y = X[ok], Y[ok]
    m = 5.0
    inside = float(np.mean((X > -m) & (X < pitch_l + m) & (Y > -m) & (Y < pitch_w + m)))
    sx = float(np.percentile(X, 98) - np.percentile(X, 2)) / pitch_l
    sy = float(np.percentile(Y, 98) - np.percentile(Y, 2)) / pitch_w

    def span_score(s):
        if s <= 0:
            return 0.0
        if 0.6 <= s <= 1.05:
            return 1.0
        return float(max(0.0, 1.0 - abs(s - 0.82) / 0.82))

    score = 0.6 * inside + 0.2 * span_score(sx) + 0.2 * span_score(sy)
    return float(score), {"inside": inside, "span_x": sx, "span_y": sy}


# ────────────────────────────────────────────── 1) 흰 선으로 찾기
def grass_mask(img, cv2):
    """잔디 영역. 못 찾으면 None."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array([GRASS_HUE[0], GRASS_SAT_MIN, 30], np.uint8),
                    np.array([GRASS_HUE[1], 255, 255], np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    if m.mean() < 0.12 * 255:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = np.where(lab == big, 255, 0).astype(np.uint8)
    # 선수 때문에 생긴 구멍을 메운다
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def line_pixels(img, mask, cv2):
    """잔디 안의 '주변보다 밝은 가는 줄' 만 남긴다."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    diff = cv2.subtract(gray, blur)             # 가는 밝은 구조만 남는다
    _, th = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(th, th, mask=mask)


def _seg_angle(s):
    x1, y1, x2, y2 = s
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def _line_from_seg(s):
    """선분을 직선 (a, b, c) 로. ax + by + c = 0, a^2+b^2=1."""
    x1, y1, x2, y2 = s
    dx, dy = x2 - x1, y2 - y1
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return None
    a, b = -dy / n, dx / n
    return a, b, -(a * x1 + b * y1)


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-9:
        return None
    return ((b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d)


def _dedupe(lines, tol_ang=4.0, tol_pos=10.0):
    """거의 같은 직선들을 하나로 묶는다. Hough 는 한 선을 여러 번 내놓는다."""
    out = []
    for ang, pos, ln, length in sorted(lines, key=lambda t: -t[3]):
        if any(abs(ang - a2) < tol_ang and abs(pos - p2) < tol_pos
               for a2, p2, _, _ in out):
            continue
        out.append((ang, pos, ln, length))
    return out


def _families(segs, w, h):
    """
    선분을 '가로쪽'(터치라인)과 '세로쪽'(골라인·하프라인)으로 나눈다.

    화면에서 어느 쪽으로 더 길게 뻗었는지로 가른다. 원근 때문에 골라인이
    기울어져도 세로 성분이 더 크므로 이 기준이 안정적이다.

    각 선은 화면 한가운데를 지나는 선과 만나는 자리(pos)로 위치를 나타낸다.
    이 값은 선분을 어느 쪽에서 그렸는지와 무관해서, '어느 쪽이 바깥인가'를
    헷갈리지 않고 판단할 수 있다.
    """
    cx, cy = w / 2.0, h / 2.0
    horiz, vert = [], []
    for x1, y1, x2, y2 in segs:
        ln = _line_from_seg((x1, y1, x2, y2))
        if ln is None:
            continue
        a, b, c = ln
        length = math.hypot(x2 - x1, y2 - y1)
        ang = _seg_angle((x1, y1, x2, y2))
        if abs(x2 - x1) >= abs(y2 - y1):        # 가로쪽 — x=cx 에서의 y
            if abs(b) < 1e-9:
                continue
            horiz.append((ang, -(a * cx + c) / b, ln, length))
        else:                                   # 세로쪽 — y=cy 에서의 x
            if abs(a) < 1e-9:
                continue
            vert.append((ang, -(b * cy + c) / a, ln, length))
    return _dedupe(horiz), _dedupe(vert)


def _quad_from(lh1, lh2, lv1, lv2, w, h):
    """네 직선의 교점으로 사각형을 만든다. 말이 안 되면 None."""
    pts = []
    for lh in (lh1, lh2):
        for lv in (lv1, lv2):
            p = _intersect(lh, lv)
            if p is None:
                return None
            if not (-1.5 * w < p[0] < 2.5 * w and -1.5 * h < p[1] < 2.5 * h):
                return None
            pts.append(p)
    return order_quad(np.asarray(pts, float))


def find_by_lines(img, pitch_l, pitch_w, cv2, px=None, py=None):
    """
    흰 선에서 경기장 네 모서리를 찾는다. 못 찾으면 None.

    바깥 선을 '이것일 것'이라고 찍지 않는다. 대신 찾은 선들로 만들 수 있는
    사각형을 모두 만들어 보고, 그 중 선수 발밑 점이 경기장 안에 가장 잘
    들어오는 것을 고른다. 페널티박스 선이나 하프라인이 섞여 있어도
    검산이 걸러 주므로 결과가 흔들리지 않는다.

    (H, 점수, 선모양 점수) 를 돌려준다.
    """
    mask = grass_mask(img, cv2)
    if mask is None:
        return None, 0.0, 0.0
    lines = line_pixels(img, mask, cv2)
    h, w = img.shape[:2]
    segs = cv2.HoughLinesP(lines, 1, np.pi / 360, threshold=60,
                           minLineLength=int(min(w, h) * 0.16), maxLineGap=25)
    if segs is None or len(segs) < 4:
        return None, 0.0, 0.0
    # 선 픽셀을 솎아 둔다 (사각형이 선을 품는지 보는 데 쓴다) 와,
    # 화면 어디서든 가장 가까운 선까지의 거리 지도 (검산에 쓴다).
    ys, xs = np.nonzero(lines)
    if len(xs) > 3000:
        pick = np.linspace(0, len(xs) - 1, 3000).astype(int)
        xs, ys = xs[pick], ys[pick]
    line_pts = np.stack([xs, ys], axis=1).astype(float)
    dist_px = line_distance_map(lines, cv2)

    horiz, vert = _families(segs.reshape(-1, 4), w, h)
    if len(horiz) < 2 or len(vert) < 2:
        return None, 0.0, 0.0
    # 바깥쪽 후보만 남긴다 (위치 기준 양 끝에서 각각 네 개까지)
    horiz = sorted(horiz, key=lambda t: t[1])
    vert = sorted(vert, key=lambda t: t[1])
    horiz = horiz[:4] + horiz[-4:]
    vert = vert[:4] + vert[-4:]

    # 사각형의 네 모서리를 경기장 어디에 붙일 것인가.
    # 예전에는 '화면 가로 = 경기장 길이' 하나만 봤다. 옆에서 찍은 중계
    # 화면은 그게 맞지만, 골대 뒤에서 찍으면 경기장 길이가 화면 세로로
    # 선다. 그때 90도 돌아간 보정이 나오면서 모든 거리가 틀어졌다.
    # 두 가지를 다 만들어 보고 선 모양이 맞는 쪽을 고른다.
    dsts = [np.float32([[0, 0], [pitch_l, 0], [pitch_l, pitch_w], [0, pitch_w]]),
            np.float32([[0, 0], [0, pitch_w], [pitch_l, pitch_w], [pitch_l, 0]])]
    min_gap_h, min_gap_v = h * 0.12, w * 0.12
    best_H, best_s, best_tpl, best_gin = None, 0.0, 0.0, 0.0
    for i in range(len(horiz)):
        for j in range(i + 1, len(horiz)):
            if abs(horiz[i][1] - horiz[j][1]) < min_gap_h:
                continue
            for k in range(len(vert)):
                for l in range(k + 1, len(vert)):
                    if abs(vert[k][1] - vert[l][1]) < min_gap_v:
                        continue
                    quad = _quad_from(horiz[i][2], horiz[j][2],
                                      vert[k][2], vert[l][2], w, h)
                    if quad is None:
                        continue
                    if cv2.contourArea(quad.astype(np.float32)) < 0.06 * w * h:
                        continue
                    for dst in dsts:
                        H = homography(quad, dst)
                        if H is None:
                            continue
                        # 잔디가 경기장 안으로 들어오는가 — 이게 먼저다.
                        # 여기서 떨어지면 나머지 점수가 아무리 높아도 버린다.
                        gin = grass_inside(H, mask, pitch_l, pitch_w)
                        if gin < GRASS_MIN:
                            continue
                        base, _ = score_homography(H, px, py, pitch_l, pitch_w)
                        # 찾은 흰 선이 실제 경기장 선 모양과 맞는가.
                        tpl = template_score(H, dist_px, img.shape, pitch_l, pitch_w)
                        s = 0.4 * tpl + 0.35 * gin + 0.25 * base
                        if s > best_s + 1e-6:
                            best_H, best_s, best_tpl, best_gin = H, s, tpl, gin
    if best_H is None:
        return None, 0.0, 0.0
    return best_H, best_s, min(best_tpl, best_gin)


# ────────────────────────────────────────────── 2) 선수가 퍼진 범위로 찾기
def find_by_players(px, py, pitch_l, pitch_w, cv2):
    """
    선수들이 밟고 다닌 자리의 최소 넓이 사각형을 경기장에 맞춘다.

    경기장 전체가 화면에 들어오고, 경기가 한쪽에만 쏠리지 않았다면 쓸 만하다.
    어디까지나 어림이므로 쓰는 쪽에서 '대략'이라고 밝혀야 한다.
    """
    pts = np.stack([np.asarray(px, float), np.asarray(py, float)], axis=1)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 200:
        return None
    hull = cv2.convexHull(pts.astype(np.float32))
    box = cv2.boxPoints(cv2.minAreaRect(hull))
    quad = order_quad(box)
    # 선수는 라인 안쪽에서 뛰므로 실제 경기장보다 조금 작게 잡힌다. 약간 넓혀 준다.
    pad_x, pad_y = pitch_l * 0.04, pitch_w * 0.06
    dst = np.float32([[pad_x, pad_y], [pitch_l - pad_x, pad_y],
                      [pitch_l - pad_x, pitch_w - pad_y], [pad_x, pitch_w - pad_y]])
    return homography(quad, dst)


# ────────────────────────────────────────────── 바깥에서 부르는 함수
def sample_frames(video, n=9, log=print):
    """영상 곳곳에서 n 장을 뽑는다."""
    try:
        import cv2
    except ImportError:
        return [], None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return [], cv2
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    try:
        if total > 0:
            for idx in np.linspace(total * 0.05, total * 0.95, n).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, img = cap.read()
                if ok:
                    out.append(img)
        else:
            while len(out) < n:
                ok, img = cap.read()
                if not ok:
                    break
                out.append(img)
    finally:
        cap.release()
    return out, cv2


# 선 모양 검산 점수의 눈금.
# 이 점수는 '실제 경기장 선을 화면에 되돌려 그렸을 때 찾은 흰 선 위에
# 얼마나 얹히나' 다. 높다고 보정이 맞았다는 보장은 못 한다 — 경기장의
# 절반만 보이는 화면에서는 보이는 쪽만 맞춰 놓고도 높게 나온다 (실측:
# 길이 방향이 1.9배 늘어난 보정이 0.75 를 받았다). 낮으면 확실히 틀렸다는
# 것만 믿을 수 있다. 그래서 어느 쪽이든 calib_check.png 를 보게 한다.
GRASS_MIN = 0.70       # 잔디의 이만큼은 경기장 안으로 들어와야 쓴다
TPL_WEAK = 0.45        # 이 아래면 확실히 틀렸다
TPL_TRUST = 0.80       # 이 위라야 '맞다고 보고 그냥 쓴다'
#
# TPL_TRUST 를 왜 이렇게 높게 뒀나.
# 실측(골대 뒤 중계 화면)에서 손으로 네 점을 찍어 맞춘 보정 — 눈으로 보면
# 빨간 선이 흰 선에 딱 얹히는 것 — 이 0.70 이었다. 같은 영상에서 눈에 띄게
# 어긋난 자동 보정이 0.63 이었다. 둘 사이가 0.07 밖에 안 된다.
# 즉 이 점수로는 '맞다' 를 증명할 수 없다. 낮으면 틀렸다는 것만 말할 수 있다.
# 그래서 확실히 좋은 경우가 아니면 기여도 계산으로 넘어가지 않고 멈춘다.
# 사람이 calib_check.png 를 보고 판단하는 편이 훨씬 정확하다.
MOVE_WARN = 8.0        # 장면마다 보정이 이만큼(m) 넘게 다르면 카메라가 움직인다


def check_picture(video, H, pitch_l, pitch_w, out_path, log=print):
    """
    실제 경기장 선을 화면에 되돌려 그린 그림을 저장한다.

    보정이 틀렸는지 한눈에 알 수 있는 가장 확실한 방법이다. 점수 0.96 이라고
    적어 놔도 사람은 그 숫자가 무슨 뜻인지 모른다. 그런데 그린 선이 화면의
    흰 선 위에 얹혀 있는지는 누구나 1초 만에 본다.
    """
    try:
        import cv2
    except ImportError:
        return None
    frames, cv2m = sample_frames(video, 3, lambda *_: None)
    if not frames:
        return None
    img = frames[len(frames) // 2]
    out = img.copy()
    tpl = pitch_lines(pitch_l, pitch_w)
    try:
        Hi = np.linalg.inv(np.asarray(H, float))
    except np.linalg.LinAlgError:
        return None
    q = Hi @ np.stack([tpl[:, 0], tpl[:, 1], np.ones(len(tpl))])
    ok = np.abs(q[2]) > 1e-9
    x, y = q[0][ok] / q[2][ok], q[1][ok] / q[2][ok]
    h, w = out.shape[:2]
    for xx, yy in zip(x, y):
        if -20 < xx < w + 20 and -20 < yy < h + 20:
            cv2.circle(out, (int(xx), int(yy)), 1, (0, 0, 255), -1)
    cv2.putText(out, "red = where the program thinks the lines are",
                (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(out, "if red does not sit on the white lines, calibration is wrong",
                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    try:
        cv2.imwrite(str(out_path), out)
        return out_path
    except Exception:              # noqa: BLE001
        return None


def calibrate(video, tracks_px, pitch_l, pitch_w, log=print):
    """
    자동 보정. (H, 방법, 점수, 설명) 을 돌려준다. 못 하면 (None, ...).

    tracks_px : (발밑 x 픽셀 배열, 발밑 y 픽셀 배열) — 추적 결과.
                검산과 두 번째 방법에 쓴다.

    점수는 '선 모양 검산' 이다. 실제 경기장 선을 화면에 되돌려 그렸을 때
    찾은 흰 선 위에 얼마나 얹히는가. 선수가 경기장 안에 들어오는지만 보는
    예전 점수는 90도 돌아간 보정에도 0.96 을 줬다 — 어떻게 돌려 붙여도
    사람은 다 안에 들어오기 때문이다.
    """
    px, py = tracks_px
    frames, cv2 = sample_frames(video, 9, log)
    if cv2 is None:
        return None, "없음", 0.0, "opencv 가 없어 자동 보정을 못 했습니다."
    best = (None, 0.0, "", 0.0)
    if frames:
        n = len(px)
        step = max(1, n // 1500)
        sx, sy = np.asarray(px, float)[::step], np.asarray(py, float)[::step]
        cands = []
        for img in frames:
            try:
                H, sc, tpl = find_by_lines(img, pitch_l, pitch_w, cv2, sx, sy)
            except Exception:           # noqa: BLE001 — 한 장 실패가 전체를 막으면 안 된다
                H, sc, tpl = None, 0.0, 0.0
            if H is None:
                continue
            cands.append((tpl, sc, H, img))
        if cands:
            cands.sort(key=lambda c: -c[0])
            tpl, sc, H, img = cands[0]
            # 여기서 ICP 로 선 위에 더 당겨 붙여 보려고 했지만 뺐다.
            # '경기장 선을 얼마나 덮었나' 를 크게 하는 쪽으로만 움직이니
            # 경기장을 화면 한쪽으로 찌부러뜨리는 답으로 흘러갔다 (점수
            # 0.99 를 받고 센터서클을 X=13m 에 갖다 놓았다). 넓이 검사로
            # 막아도 0.80 짜리 엉터리가 남았다. 확인할 수 없는 개선은
            # 넣지 않는다 — 대신 아래 calib_check.png 로 사람이 본다.
            log(f"    흰 선으로 찾기: {len(cands)}/{len(frames)}장 성공 · "
                f"선 모양 검산 {tpl:.2f}")
            # 카메라가 움직이는가 — 장마다 나온 보정이 서로 크게 다르면
            # 한 장의 보정을 영상 전체에 쓸 수 없다.
            spread = _disagreement([c[2] for c in cands[:5]], img.shape, pitch_l, pitch_w)
            best = (H, tpl, "흰 선", spread)
    if best[1] < TPL_WEAK:
        try:
            H2 = find_by_players(px, py, pitch_l, pitch_w, cv2)
        except Exception:               # noqa: BLE001
            H2 = None
        if H2 is not None and best[0] is None:
            s2, info2 = score_homography(H2, px, py, pitch_l, pitch_w)
            log(f"    선을 못 찾아 선수가 퍼진 범위로 어림잡습니다 (점수 {s2:.2f})")
            best = (H2, 0.0, "선수 범위(대략)", 0.0)
    H, tpl, how, spread = best
    if H is None:
        return None, how or "없음", 0.0, (
            "자동 보정에 실패했습니다. 경기장 전체가 보이는 고정 카메라 영상이면 "
            "잘 되고, 중계 화면처럼 카메라가 계속 움직이면 어렵습니다. "
            "[고급 설정] 의 [손으로 보정하기] 를 써 주세요.")
    notes = []
    if how.startswith("선수"):
        notes.append("선을 못 찾아 선수들이 퍼진 범위로 어림잡았습니다. "
                     "거리가 실제와 많이 다를 수 있습니다.")
    elif tpl < TPL_WEAK:
        notes.append(f"! 선 모양 검산 {tpl:.2f} — 그려 본 경기장 선이 화면의 흰 선과 "
                     f"전혀 맞지 않습니다. 미터로 바꾼 값을 믿으면 안 됩니다.")
    if spread > MOVE_WARN:
        notes.append(f"! 장면마다 보정이 평균 {spread:.0f} m 씩 다릅니다 — 카메라가 "
                     f"움직이는 중계 화면입니다. 보정 한 번을 영상 전체에 쓰므로 "
                     f"거리 값이 구간마다 틀어집니다. [고급 설정] 의 '구간' 으로 "
                     f"카메라가 거의 안 움직이는 몇 분만 잘라 보세요.")
    # 점수가 높아도 맞았다는 보장이 안 되므로, 언제나 그림을 보게 한다.
    notes.append("결과 폴더의 calib_check.png 를 꼭 한 번 보세요. 빨간 선이 화면의 "
                 "흰 선 위에 얹혀 있어야 합니다. 어긋나 있으면 [고급 설정] 의 "
                 "[손으로 보정하기] 로 네 점을 직접 찍어 주세요 — 그게 가장 정확합니다.")
    return H, how, tpl, "  ".join(notes)


def _disagreement(Hs, shape, pitch_l, pitch_w):
    """여러 장에서 나온 보정이 서로 얼마나 다른가 (화면 네 귀퉁이의 평균 어긋남, m)."""
    if len(Hs) < 2:
        return 0.0
    h, w = shape[:2]
    pts = np.float32([[w * .25, h * .25], [w * .75, h * .25],
                      [w * .25, h * .75], [w * .75, h * .75]])
    mapped = []
    for H in Hs:
        X, Y = apply_h(H, pts[:, 0], pts[:, 1])
        if not (np.isfinite(X).all() and np.isfinite(Y).all()):
            continue
        mapped.append(np.stack([X, Y], 1))
    if len(mapped) < 2:
        return 0.0
    a = np.stack(mapped)
    return float(np.mean(np.linalg.norm(a - a.mean(0), axis=2)))
