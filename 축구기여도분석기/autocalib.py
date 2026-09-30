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

    (H, 점수) 를 돌려준다.
    """
    mask = grass_mask(img, cv2)
    if mask is None:
        return None, 0.0
    lines = line_pixels(img, mask, cv2)
    h, w = img.shape[:2]
    segs = cv2.HoughLinesP(lines, 1, np.pi / 360, threshold=60,
                           minLineLength=int(min(w, h) * 0.16), maxLineGap=25)
    if segs is None or len(segs) < 4:
        return None, 0.0
    # 선 픽셀을 솎아 둔다. 후보 사각형이 경기장 선을 다 품는지 보는 데 쓴다.
    ys, xs = np.nonzero(lines)
    if len(xs) > 3000:
        pick = np.linspace(0, len(xs) - 1, 3000).astype(int)
        xs, ys = xs[pick], ys[pick]
    line_pts = np.stack([xs, ys], axis=1).astype(float)

    horiz, vert = _families(segs.reshape(-1, 4), w, h)
    if len(horiz) < 2 or len(vert) < 2:
        return None, 0.0
    # 바깥쪽 후보만 남긴다 (위치 기준 양 끝에서 각각 네 개까지)
    horiz = sorted(horiz, key=lambda t: t[1])
    vert = sorted(vert, key=lambda t: t[1])
    horiz = horiz[:4] + horiz[-4:]
    vert = vert[:4] + vert[-4:]

    dst = np.float32([[0, 0], [pitch_l, 0], [pitch_l, pitch_w], [0, pitch_w]])
    min_gap_h, min_gap_v = h * 0.12, w * 0.12
    best_H, best_s = None, 0.0
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
                    H = homography(quad, dst)
                    if H is None:
                        continue
                    s, _ = score_homography(H, px, py, pitch_l, pitch_w)
                    # 경기장 선을 얼마나 품는가. 이것이 페널티박스 선을 경계로
                    # 잘못 고르는 것을 막는 결정적인 단서다.
                    s = 0.75 * s + 0.25 * inside_quad(quad, line_pts)
                    if s > best_s + 1e-6:
                        best_H, best_s = H, s
    return best_H, best_s


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


def calibrate(video, tracks_px, pitch_l, pitch_w, log=print):
    """
    자동 보정. (H, 방법, 점수, 설명) 을 돌려준다. 못 하면 (None, ...).

    tracks_px : (발밑 x 픽셀 배열, 발밑 y 픽셀 배열) — 추적 결과.
                검산과 두 번째 방법에 쓴다.
    """
    px, py = tracks_px
    frames, cv2 = sample_frames(video, 9, log)
    if cv2 is None:
        return None, "없음", 0.0, "opencv 가 없어 자동 보정을 못 했습니다."
    best = (None, 0.0, "")
    if frames:
        # 검산에 쓸 발밑 점은 골고루 솎아 쓴다 (전부 쓰면 느리다)
        n = len(px)
        step = max(1, n // 1500)
        sx, sy = np.asarray(px, float)[::step], np.asarray(py, float)[::step]
        cands = []
        for i, img in enumerate(frames):
            try:
                H, sc = find_by_lines(img, pitch_l, pitch_w, cv2, sx, sy)
            except Exception:           # noqa: BLE001 — 한 장 실패가 전체를 막으면 안 된다
                H, sc = None, 0.0
            if H is None:
                continue
            cands.append((sc, H, score_homography(H, px, py, pitch_l, pitch_w)[1]))
        if cands:
            cands.sort(key=lambda c: -c[0])
            s, H, info = cands[0]
            log(f"    흰 선으로 찾기: {len(cands)}/{len(frames)}장 성공 · "
                f"최고 점수 {s:.2f} (경기장 안 {info.get('inside', 0)*100:.0f}% · "
                f"가로 {info.get('span_x', 0)*100:.0f}% 세로 {info.get('span_y', 0)*100:.0f}%)")
            best = (H, s, "흰 선")
    if best[1] < 0.75:
        try:
            H2 = find_by_players(px, py, pitch_l, pitch_w, cv2)
        except Exception:               # noqa: BLE001
            H2 = None
        if H2 is not None:
            s2, info2 = score_homography(H2, px, py, pitch_l, pitch_w)
            log(f"    선수가 퍼진 범위로 찾기: 점수 {s2:.2f} "
                f"(경기장 안 {info2.get('inside', 0)*100:.0f}%)")
            if s2 > best[1]:
                best = (H2, s2, "선수 범위(대략)")
    H, score, how = best
    if H is None or score < 0.45:
        return None, how or "없음", score, (
            "자동 보정에 실패했습니다. 경기장 전체가 보이는 고정 카메라 영상이면 "
            "잘 되고, 중계 화면처럼 카메라가 계속 움직이면 어렵습니다.")
    note = ""
    if how.startswith("선수"):
        note = ("선을 못 찾아 선수들이 퍼진 범위로 어림잡았습니다. "
                "거리가 실제와 다를 수 있으니 결과는 대략으로 보세요.")
    elif score < 0.8:
        note = "보정이 완벽하지는 않습니다. 결과를 참고용으로 보세요."
    return H, how, score, note
