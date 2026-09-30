"""
축구 영상 선수/공 위치 추적 코어.

GUI(app.py)에서 호출하는 것이 기본이지만, 터미널에서 단독 실행도 가능하다.
    python tracker.py videos/epl.mp4
    python tracker.py videos/            # 폴더 전체
    python tracker.py videos/ --fast     # 프리셋 지정
    python tracker.py --help
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

PERSON_CLASS = 0      # COCO 기준 person
BALL_CLASS = 32       # COCO 기준 sports ball


class ModelProfile:
    """
    쓰고 있는 모델이 무엇을 아는지 정리한다.

    두 종류를 지원한다.

    COCO 일반 모델 (yolo11n 등)
        person(0) 과 sports ball(32) 만 있다. 사람은 다 똑같이 보이므로
        선수·심판·스태프를 잔디 위치와 유니폼 색으로 추측해야 한다.

    축구 전용 모델
        player / goalkeeper / referee / ball 을 직접 구분한다. 스태프·관중은
        어느 클래스에도 안 잡히므로 경기장 필터가 필요 없고, 심판·골키퍼도
        색으로 추측할 필요가 없다.
    """

    SOCCER_WORDS = {"player", "goalkeeper", "referee"}

    def __init__(self, names: dict):
        self.names = {int(k): str(v) for k, v in (names or {}).items()}
        low = {i: n.lower() for i, n in self.names.items()}

        self.players = {i for i, n in low.items() if n == "player"}
        self.keepers = {i for i, n in low.items() if "goalkeep" in n}
        self.referees = {i for i, n in low.items() if "refere" in n}
        self.balls = {i for i, n in low.items() if "ball" in n}
        self.persons = {i for i, n in low.items() if n == "person"}

        self.is_soccer = bool(self.SOCCER_WORDS & set(low.values()))
        if not self.is_soccer:
            # COCO. person 이 사람 전부이고 팀을 나눌 대상도 그것뿐이다.
            self.players = set(self.persons)

        # '사람 계열' = 상자를 사람으로 취급할 클래스 전부
        self.people = self.players | self.keepers | self.referees | self.persons
        # 팀 번호를 색으로 정할 대상. 심판·골키퍼는 모델이 알려주므로 제외한다.
        self.team_targets = set(self.players)
        # 팀이 아님(-1)으로 고정할 대상
        self.non_team = self.keepers | self.referees

    def wanted_ids(self, want_people: bool, want_ball: bool) -> list[int] | None:
        """설정의 '선수/공' 체크박스를 이 모델의 클래스 번호로 옮긴다."""
        ids: set[int] = set()
        if want_people:
            ids |= self.people
        if want_ball:
            ids |= self.balls
        if not ids or ids == set(self.names):
            return None                      # 전부면 굳이 걸러 달라고 하지 않는다
        return sorted(ids)

    def describe(self) -> str:
        if not self.is_soccer:
            return "일반 COCO 모델 (사람/공만 구분)"
        parts = [self.names[i] for i in sorted(
            self.players | self.keepers | self.referees | self.balls)]
        return "축구 전용 모델 (" + ", ".join(parts) + ")"

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg", ".wmv"}

CSV_COLUMNS = [
    "frame",      # 프레임 번호 (0부터)
    "time_sec",   # 영상 시작 기준 경과 시간(초)
    "track_id",   # 추적 ID. 같은 선수는 같은 번호를 유지한다. 놓치면 -1
    "class_id",   # COCO 클래스 번호
    "class_name", # person / sports ball
    "conf",       # 신뢰도
    "x1", "y1", "x2", "y2",   # 바운딩 박스 좌상단 / 우하단 (픽셀)
    "cx", "cy",   # 박스 중심
    "fx", "fy",   # 발밑 중심점 (x1+x2)/2, y2 — 지면 좌표 변환용
    "w", "h",
    "team",       # 0 / 1 = 양 팀, -1 = 어느 쪽도 아님(심판·골키퍼 등), 빈칸 = 공
    "interpolated",  # 1 = 측정이 아니라 앞뒤 공 위치로 메운 추정값. 0 = 측정값
]


@dataclass
class Config:
    """분석 파라미터. config.json 으로 저장/복원된다."""

    model: str = "yolo11x.pt"
    imgsz: int = 1280
    conf: float = 0.10
    iou: float = 0.50
    # COCO 기준 0 = person, 32 = sports ball. None 이면 전체 클래스
    classes: list[int] | None = field(default_factory=lambda: [0, 32])
    tracker: str = "bytetrack.yaml"
    # ByteTrack 기본 임계값(0.25)은 conf 와 무관하게 고정이라, conf 를 0.1 로
    # 낮춰도 0.25 미만 검출은 추적 ID 를 못 받고 버려진다. 켜두면 임계값을
    # conf 에 맞춰 다시 계산한 트래커 설정을 자동 생성해서 쓴다.
    auto_tracker_thresh: bool = True
    # ByteTrack 의 2단계 연관을 되살린다. 검출을 conf 의 절반까지 낮춰 뽑아서
    # conf 미만 박스는 "이미 있는 트랙을 이어붙이는 용도"로만 쓰고 CSV 에는 안
    # 넣는다. 가려졌다 나온 선수의 ID 가 덜 끊긴다. 대신 검출이 늘어 조금 느리다.
    bytetrack_recover_low: bool = False
    # ByteTrack 이 기존 트랙과 새 검출을 같은 사람으로 이어붙일 때 허용하는
    # 거리(1 - IoU) 상한. 올리면 덜 겹쳐도 이어붙인다.
    #
    # 박스가 작은 영상에서 중요하다. 720p 에서 선수 키가 32px 면 몸통 너비가
    # 13px 남짓인데, stride 3 이면 처리 간격이 0.125초라 달리는 선수가 그
    # 사이에 자기 너비만큼 움직인다. 겹침이 0 이 되어 트랙이 끊기고 새 ID 를
    # 받는다. 실측 — 그 영상에서 트랙 지속 중앙값이 2장(0.2초)이었다.
    #
    # 실측(20초 구간, 두 영상 모두) — 0.8 → 0.9
    #     사용자 720p : 추적ID 233→139 · 트랙 지속 7→13장 · 검출 15.8→17.9명
    #     전술 1080p : 추적ID 189→117 · 트랙 지속 13→24장 · 검출 18.1→19.3명
    # 검출까지 느는 것은 2단계 연관이 살아나 끊겼던 트랙을 되살리기 때문이다.
    #
    # 올리면 다른 선수로 트랙이 옮겨붙을 위험이 생기는데, 그건 ID 수·지속
    # 지표로는 오히려 좋아 보이므로 따로 쟀다. 트랙 안에서 자기 키보다 크게
    # 튄 비율이 두 영상·세 설정 모두 0.0% 였다. ByteTrack 이 칼만 예측을
    # 함께 쓰기 때문에 IoU 문턱만 풀어도 엉뚱한 곳으로는 안 붙는다.
    match_thresh: float = 0.9
    track_buffer: int = 30     # 가려진 선수를 몇 프레임까지 기억할지 (원본 프레임 기준)
    vid_stride: int = 1        # N 프레임마다 1장. 2로 올리면 2배 빨라지고 그만큼 성겨진다
    save_video: bool = True    # 박스가 그려진 결과 영상 저장
    device: str = ""           # "" = 자동, "0" = 첫 번째 GPU, "cpu" = 강제 CPU
    # GPU 가 없을 때 모델을 OpenVINO 로 변환해 쓴다. CPU에서 1.5~2배 빨라진다.
    # GPU 가 있으면 이 설정과 무관하게 그냥 무시된다.
    use_openvino: bool = True

    # --- 경기장 밖 사람 걸러내기 -------------------------------------
    # COCO 모델에는 person 클래스 하나뿐이라 감독·코치진·볼보이·관중이
    # 전부 선수와 같은 것으로 잡힌다. 잔디 영역을 찾아서 발이 그 밖에 있는
    # 사람을 버린다. 공은 공중에 뜨므로 이 필터를 적용하지 않는다.
    pitch_filter: bool = True
    pitch_every: int = 30      # N 프레임마다 잔디 영역을 다시 계산 (원본 프레임 기준)
    pitch_hue: tuple = (30, 95)   # HSV 색상 범위. 잔디가 유난히 어둡거나 인조면 조절
    pitch_sat_min: int = 30
    pitch_shrink: int = 16     # 경기장 가장자리를 이만큼(픽셀) 안쪽으로 좁힌다
    # True 면 예전처럼 볼록껍질을 쓴다. 중계 화면에서는 껍질이 화면 대부분을
    # 삼켜 필터가 무력해지므로 기본은 껐다. 잔디가 심하게 조각나는 영상에서만.
    pitch_convex: bool = False

    # --- 중계 그래픽 속 인물 걸러내기 -----------------------------------
    # 하단 배너나 선수 소개 그래픽에 박힌 사진도 YOLO 에는 사람으로 보인다.
    # 그런데 그건 화면에 붙박여 있어서 카메라가 움직여도 제자리다. 오래
    # 잡혔는데 거의 안 움직인 추적은 사람이 아니라 그래픽으로 본다.
    drop_static_tracks: bool = True
    static_min_frames: int = 15    # 이만큼 이상 잡힌 추적만 판단 대상
    static_max_move: float = 0.10  # (이동 폭 / 평균 키) 가 이보다 작으면 그래픽

    # --- 공 오검출 걸러내기 --------------------------------------------
    # 축구 모델은 공이 아닌 것을 공이라고 자주 말한다. 실측(리버풀-레알 전체
    # 경기에서 120 프레임 표본)으로 확인한 오검출은 두 종류였다.
    #
    #   ① 페널티 마크·경기장 라인 — 잔디 위의 작고 흰 자국. 전체의 27%
    #   ② 심판·코너 깃발·선수     — 공이라기엔 너무 큰 것.   전체의 13%
    #
    # 50 초 구간을 눈으로 전수 판정한 결과 공 검출 587 건 중 281 건(48%)이
    # 페널티 마크였다. 게다가 마크가 잡히는 프레임에서는 진짜 공을 대신
    # 놓치는 일이 많았다.
    #
    # 크기로 가른다. 단 픽셀 크기를 그대로 쓰면 안 된다 — 멀리 있는 공은
    # 작게 잡히므로 '작다'와 '멀다'가 뒤섞인다. 그래서 같은 프레임에 있는
    # 사람 키의 중앙값으로 나눈다. 마크 옆의 선수도 똑같이 멀어서 작게
    # 잡히므로 거리가 상쇄된다.
    #
    #   실측 분포 — 진짜 공 0.170~0.319 (중앙 0.208)
    #               페널티 마크 0.134~0.232 (중앙 0.159)
    #   0.18~0.35 로 자르면 진짜 공 305/306(100%) 을 지키고 마크 91% 를 뺀다.
    #   공 검출의 정밀도가 52% → 92% 가 된다.
    #
    # 임계값을 축구 모델로만 측정했으므로 COCO 모델에는 적용하지 않는다.
    ball_size_check: bool = True
    # 0.18 은 선수 모델 검출로 잰 값이었다. 공 전용 모델을 붙인 뒤 다시 재니
    # 0.20 이 낫다 — 진짜 공은 97% 지키면서 페널티 마크를 83% 더 뺀다.
    # (진짜 공 5% 분위 0.202, 마크 95% 분위 0.214)
    # 0.18 → 0.20 → 0.16 으로 바뀌었다.
    #
    # 0.20 은 밝기·모양 검사가 없던 시절에 정한 값이다. 그 두 검사가 생긴
    # 뒤에는 하한이 과하게 자르고 있었다 — 전술 90초 구간에서 후보 877건 중
    # 404건(46%)이 하한에서만 탈락했고, 그 탈락분의 비율이 0.148~0.200
    # (중앙 0.184)로 문턱 바로 아래에 몰려 있었다.
    #
    # 그중 0.16 이상인 391건에 밝기·모양을 걸면 218건이 남는데(44% 는 그
    # 단계에서 걸러진다), 남은 것 39개를 무작위로 뽑아 보니 39개 전부
    # 명백한 진짜 공이었다. 공이 잡히는 프레임이 39% → 78% 로 늘어난다.
    ball_ratio_min: float = 0.16   # 이보다 작으면 잔디 위의 자국
    # 0.35 → 0.28 → 0.36 으로 두 번 바꿨다.
    #
    # 0.28 은 1080p 전술 영상에서 잰 값인데, 720p 영상에서 진짜 공이 이 문턱에
    # 걸려 탈락했다. 선수 34px 에 공 10.6px 면 비율이 0.31 이다 — 압축이 심할수록
    # 작은 물체의 박스가 상대적으로 커진다. 또 한 영상에 맞춘 절대 임계값이
    # 다른 영상에서 깨진 경우다.
    #
    # 상한을 푸는 대신 모양 검사(ball_min_fill)가 그 역할을 대신한다. 0.28 이
    # 잡던 '몸 굽힌 흰 유니폼 선수' 14건은 채움 중앙값이 0.50 이라 14/14 가
    # 모양 검사에서 걸린다. 전술 영상 정답 258건으로 재보니 상한을 0.36 으로
    # 올려도 진짜 공 183 → 185, 불량 2 → 3 으로 사실상 차이가 없다.
    ball_ratio_max: float = 0.36   # 이보다 크면 사람이나 깃발

    # 공의 흰색과 잔디 위 자국의 흰색은 밝기가 다르다. 공은 포화된 흰색이고
    # 페인트 자국은 바래서 회백색이다. 실측(정답 판정 258건, 회색조 최대값)
    #     진짜 공 248 · 페널티 마크 196 · 사진기자석 216
    # 215 로 자르면 진짜 공을 하나도 안 잃고 마크 24건을 전부 뺀다.
    #
    # 주변 잔디 대비(대비·배수)로도 재봤지만 더 나빴다 — 마크도 잔디보다는
    # 밝기 때문이다. 절대값이 맞다.
    #
    # 다만 이 값은 조명에 따라 달라진다. 야간 조명 경기에서 잰 값이라
    # 어두운 영상에서는 공을 다 버릴 수 있다. 그래서 너무 많이 걸러내면
    # 분석 끝에 경고를 남긴다. 0 이면 검사하지 않는다.
    ball_min_bright: int = 215
    # 날아가는 공은 모션블러로 흰색이 번져 최대 밝기가 190~210 으로 떨어진다.
    # 롱패스(전술 11:38)에서 신뢰도 0.77 짜리 공이 밝기 205 로 탈락했다.
    #
    # 정답 라벨의 페널티 마크 24건은 신뢰도가 최대 0.67(95% 분위 0.59)이다.
    # 그보다 확실히 위인 0.70 이상이면 밝기 검사를 면제한다 — 라벨 데이터에서
    # 마크가 하나도 새지 않는다(0.55 로 내리면 3건 샌다). 0 이면 면제 없음.
    ball_bright_exempt_conf: float = 0.70
    # 면제에도 밝기 바닥을 둔다. 1920 과 확대 창에서는 신뢰도가 전반적으로
    # 올라가 0.70 을 넘는 가짜가 생겼다. 면제로 들어온 후보 48개 전수 판정
    # (전술 10:00~13:00)
    #     터치라인 밖 여백에 놓인 공 모양 물체  146~163  (가짜, 전부 화면 아래)
    #     움직여서 흐려진 진짜 공            195~214
    # 확대 검출 결과 1,891개 중 296개(16%)가 이 물체였다 -- 확대 창이 거기
    # 머물며 스스로 강화됐다. 180 이면 둘 사이가 깨끗이 갈린다. 0 이면 바닥 없음.
    ball_exempt_min_bright: int = 180
    # 면제는 '움직이는' 후보에게만 준다. 어두운데 확신하는 공은 빨리 움직여
    # 모션블러로 번진 공이다. 멈춘 공은 번지지 않으므로 밝게 찍힌다(골킥
    # 대기 공 235~249). 반대로 페널티 마크는 제자리에 있고 밝기 181~208 이라
    # 밝기로는 진짜 흐린 공(187~214)과 안 갈렸다.
    # 이 시간 창(초) 안의 앞선 처리 프레임에서 still_px 이내에 후보가 있었으면
    # 멈춘 것으로 보고 면제하지 않는다. 확대 결과의 마크 118프레임 → 9프레임.
    ball_still_from_sec: float = 0.24
    ball_still_to_sec: float = 0.96
    ball_still_px: float = 8.0
    # 신뢰도가 이보다 낮으면서 멈춰 있는 후보는 아예 뺀다. 1920·확대로 약한
    # 후보를 많이 받게 되자 12:24 경기 중단 때 카메라맨(선수 모델도 사람으로
    # 못 잡는다)·관중석·서 있는 선수의 흰 축구화가 0.10~0.30 으로 줄줄이
    # 들어왔다. 이 규칙으로 빠진 75프레임 전수 판정: 가짜 약 59(카메라맨·
    # 관중석 37, 축구화 20) · 진짜 약 16(골키퍼가 내려놓는 공, 라인 위 공).
    # 0.4·0.5 로 올려도 가짜는 더 안 빠지고 진짜만 더 잃었다. 0 이면 끈다.
    ball_still_max_conf: float = 0.30

    # --- 공 궤적 잇기 (분석이 끝난 뒤) ------------------------------------
    # 한 프레임만 보면 달리는 선수의 흰 축구화는 공과 구별이 안 된다 (밝기
    # 222~250 · 채움 0.62~0.83 · 크기 비율 0.20~0.27 전부 공 범위).
    # 전술 10:06~10:07(f15164~15190): 공이 뜬 순간 놓치자 확대 창이 날아가던
    # 방향에 놓였고, 그 안의 축구화를 0.3~0.7 로 잡았다. 진짜 공은 그동안
    # 전체 화면에서 0.08~0.19 로 매끄럽게 굴러가고 있었지만(모션블러로 밝기·
    # 모양 검사 탈락) 축구화에 밀렸다.
    #
    # 차이는 시간축에 있다. 진짜 공의 약한 후보들은 이어져서 확실한 공으로
    # 연결되고, 축구화는 어떤 확실한 공과도 안 이어진다. 그래서 전체 화면에서
    # 신뢰도 >= ball_chain_anchor_conf 로 잡힌 공(끝점)에서 앞뒤로, 공 모델이
    # 본 모든 후보(검사 탈락 포함, 크기만 정상)를 등속 예측대로 따라간다.
    # 이어진 길이가 ball_chain_min_len 이상이면 그 프레임의 선택을 이것으로
    # 바꾸고(ball_chain_min_dist 이상 떨어져 있을 때만) 빈 프레임은 채운다.
    #
    # 끝점을 '전체 화면' 신뢰도로 정하는 이유: 확대 창은 축구화도 0.7 로
    # 부풀린다. 확대 창 신뢰도로 끝점을 잡으면 축구화 궤적이 끝점이 된다.
    #
    # 실측(전술 10:00~13:00, stride 2) 오프라인 재현: 교체 84 · 추가 161.
    # 교체 전후 78개 비교 -- 좋아짐 약 50, 나빠짐 약 5(점선 조각·손), 불분명 약 20.
    # 추가 64개 중 틀림 약 4. 허용 오차 14px 는 속도를 모를 때 너무 좁아
    # 이어지지 않았고 28px 에서 10:06 구간이 전부 바로잡혔다.
    ball_chain: bool = True
    ball_chain_anchor_conf: float = 0.60
    ball_chain_tol: float = 28.0         # 예측 위치에서 허용 거리(px)
    ball_chain_tol_speed: float = 0.6    # 속도(px/처리프레임) 비례 허용 추가분
    ball_chain_max_miss: int = 2         # 연속으로 후보가 없어도 되는 처리 프레임 수
    ball_chain_min_len: int = 3
    ball_chain_min_dist: float = 30.0

    # 흰 축구화가 공으로 잡힌다. 밝기도 포화돼 있고(선수 유니폼과 같은 흰색)
    # 크기도 공과 비슷해서 위의 두 기준을 그냥 통과한다. 무작위 104건을
    # 전수 판정하니 축구화가 19건(18%)으로 가장 큰 오류였다.
    #
    # 가르는 건 '모양이 얼마나 꽉 찼는가'다. 흰 덩어리의 넓이를 그 덩어리의
    # 외접 사각형 넓이로 나눈다. 공은 원이라 이론값 π/4 ≈ 0.785 근처고
    # 실측 중앙값이 0.77 이었다. 축구화는 끈·스트라이프 때문에 들쭉날쭉해
    # 0.52 다. 0.65 로 자르면 진짜 공 93% 를 지키면서 축구화를 19→3 으로 뺀다.
    # (정밀도 79% → 93%)
    #
    # 종횡비로도 재봤지만 못 쓴다 — 빠르게 움직이는 공은 모션블러로
    # 가로로 늘어나서 축구화와 구별이 안 된다. 채움 비율은 블러에도 버틴다.
    ball_min_fill: float = 0.65

    # 공이 한자리에 붙박여 있으면 공이 아니다. 흰 옷 입은 선수의 몸통이
    # 공으로 잡히는 경우가 실측에서 12프레임 연속 나왔다(움직임 1px).
    # 진짜 공은 프리킥으로 멈춰 있어도 결국 차이므로 트랙 전체로 보면 움직인다.
    # --- 골키퍼 트랙 확정 -----------------------------------------------
    # 모델은 골키퍼를 프레임마다 '골키퍼'와 '선수'로 오간다. 실측(전술
    # 10:00~13:00, stride 2)에서 진짜 골키퍼 트랙은 골키퍼 라벨이 3~39% 뿐이고
    # 나머지는 선수 라벨이었다. 영상에서는 골키퍼일 때만 GK 로, 나머지는 번호
    # 박스로 찍혀 번호가 떴다 사라졌다 바뀌는 것처럼 보였다.
    #
    # 반대로 골대(세로 기둥·그물)는 골키퍼 라벨이 69~100% 다 -- 학습 데이터에서
    # 골키퍼가 늘 골대 옆에 서 있어서 모델이 '기둥 = 골키퍼'로 배웠다.
    #
    # 골키퍼 라벨이 한 번이라도 붙은 트랙을 트랙 단위로 판정한다.
    #   1) 유니폼 색이 팀 0/1 이면 필드 선수 -- 골대 근처에서 잠깐 잘못 찍힌 것.
    #      골키퍼 라벨을 선수로 되돌린다.
    #   2) 팀 -1 이고 트랙 전체(모든 라벨)의 신뢰도 중앙값 < keeper_track_conf
    #      이면 골대 -- 선수 라벨 행까지 트랙을 통째로 뺀다.
    #   3) 그 외는 골키퍼 -- 트랙 전체를 골키퍼로 통일한다.
    #
    # 트랙 전체 신뢰도 중앙값 실측
    #     골대   0.22 0.30 0.34 0.34 0.36 0.41
    #     골키퍼 0.60 0.65 0.75 0.77 0.80 0.80 0.81
    # 골키퍼 라벨 행만 보면 0.41 / 0.45 로 거의 붙어 있었는데, 진짜 골키퍼는
    # 선수로 찍힌 행의 신뢰도가 높아 전체로 보면 0.41 / 0.60 으로 벌어진다.
    # 가운데인 0.50 으로 자른다. 0 이면 끈다.
    keeper_track_conf: float = 0.50

    # --- 공 빈 구간 메우기 ----------------------------------------------
    # 롱패스로 공이 높이 뜨면 공 모델이 거의 못 본다(작고 흐리고 배경이 관중석).
    # 앞뒤로 확실히 잡힌 공 위치 사이를 직선으로 이어 빈 프레임을 채운다.
    #
    # 위험: 끝점이 틀리면 틀린 것을 길게 늘여준다. 축구화 한 장이 끝점이 되면
    # 진짜 공에서 축구화까지 가짜 궤적이 생긴다. 그래서 끝점을 엄격하게 고른다.
    #
    # 실측(전술 10:00~13:00, stride 2) — 공 1,291행을 둘로 나눠 무작위 52개씩
    #     엄격한 끝점(신뢰도 >=0.6 + 앞뒤 프레임에도 가까이 이어짐) 581행 -> 52/52 진짜 공
    #     나머지 710행 -> 약 8장 불량(축구화·몸통)
    # 축구화는 신뢰도가 낮고 발이 계속 움직여 앞뒤로 이어지지 않으므로 끝점이
    # 될 수 없다.
    #
    # 메운 행은 interpolated=1 로 표시한다. 측정값이 아니므로 분석에서 가려 쓸
    # 수 있어야 하고, 영상에서도 회색 점선으로 다르게 그린다.
    ball_interpolate: bool = True
    interp_anchor_conf: float = 0.60   # 끝점이 될 신뢰도 하한
    # 이보다 긴 빈 구간은 메우지 않는다. 처음엔 2.0 초로 넣었다가 틀렸다.
    #
    # 측정된 공을 일부러 지우고 양옆 끝점으로 보간해 실제 위치와 비교했다
    # (단위: 선수 키. 0.33 이면 선수 키의 1/3 = 대략 공 표시 원 밖).
    #     빈 구간 0.16초  오차 중앙 0.01 · 1/3 이내 100%
    #             0.6초          0.18 ·           76%
    #             0.8초          0.28 ·           58%
    #             1.6초          0.91 ·           17%
    # 공은 짧은 시간에도 방향을 바꾸고(드리블·패스) 카메라가 패닝해서 픽셀상
    # 경로가 직선이 아니다. 2.0 초로 메운 점 52개를 눈으로 보니 원 안에 공이
    # 있는 것이 29% 뿐이었다.
    #
    # 상한별 (커버리지 증가 / 보간 오차 90% 지점)
    #     0.3초 +3%p / 0.03   0.4초 +4%p / 0.08   0.5초 +5%p / 0.17
    # 0.4 초면 90% 지점에서도 선수 키의 8%(약 5px) 라 공 위에 떨어진다.
    #
    # 롱패스(몇 초짜리 공백)는 이 방법으로 못 메운다. 선을 그으면 틀린다.
    interp_max_gap_sec: float = 0.4
    # 두 끝점 사이 속도 상한 (초당 선수 키). 실측 최대 14.2(약 26 m/s, 강한 킥).
    # 20 이면 약 36 m/s 로 프로 슈팅보다 빠르다 -- 이걸 넘으면 공이 아니라
    # 서로 다른 물체를 이은 것이다.
    interp_max_speed: float = 20.0
    drop_static_ball: bool = True
    static_ball_min_frames: int = 5     # 이만큼 이상 잡힌 공 트랙만 판단
    static_ball_max_move: float = 1.0   # (이동 폭 / 공 크기) 가 이보다 작으면 가짜

    # --- 공 전용 모델 -------------------------------------------------
    # 선수 모델은 공을 잘 못 찾는다. 640 에서 501 프레임 중 8 장, 1280 으로
    # 올려도 60 프레임 중 42 장이다. 공은 작아서(지름 12px 남짓) 선수와
    # 함께 학습된 모델에서는 늘 뒷전이다.
    #
    # 공만 학습한 별도 모델(yolo11n, 5.5MB)을 같은 프레임에 함께 돌린다.
    # 실측 — 같은 60 프레임에서
    #     선수 모델 @1280 : 42 장에서 발견, 968ms/장
    #     공 전용   @1280 : 50 장에서 발견, 134ms/장   (더 잘 찾고 7배 빠름)
    #     공 전용   @960  : 39 장에서 발견,  75ms/장
    #
    # 비어 있으면 쓰지 않는다 (기존 동작 그대로).
    ball_model: str = ""
    # 공은 작아서 해상도를 내리면 바로 놓친다. 1280 → 1920 (원본 해상도).
    # 전술 10:00~13:00 에서 먼 쪽 터치라인의 공(7px 남짓)을 1280 은 신뢰도
    # 0.07 로, 1920 은 0.83 으로 봤다. 공 프레임 57.0% → 68.9%, 정밀도 동일.
    # 비용: 공 모델 140 → 330ms/장.
    ball_imgsz: int = 1920
    ball_conf: float = 0.10
    # 해상도를 올리면 모델이 상자를 더 꼭 맞게 그려 '사람 키 대비 비율'이
    # 작아진다. 고신뢰 공의 비율 중앙값 1280: 0.207 · 1920: 0.184 · 확대 창:
    # 0.169. 크기 검사는 1280 에서 잰 기준이라 상자 높이를 이 값으로 나눠
    # 1280 기준으로 되돌린 뒤 검사한다. 보정 없이 1920 을 쓰면 하한에서만
    # 후보 1,244건이 탈락했다. ball_imgsz 를 바꾸면 다시 재야 한다.
    ball_ratio_scale: float = 0.89

    # --- 공 주시 확대 ---------------------------------------------------
    # 직전 공 위치 주변을 잘라 확대해 한 번 더 본다. 먼 공·선수 발밑의 공은
    # 전체 화면에서는 몇 픽셀이라 신뢰도가 바닥인데, 480px 를 960 으로
    # 2배 키우면 같은 공을 0.04 → 0.7~0.8 로 잡는다 (전술 11:42 f17598~17614).
    # 공을 놓친 지 ball_focus_hold_sec 이 지나면 더 보지 않는다.
    ball_focus: bool = True
    ball_focus_size: int = 480
    ball_focus_imgsz: int = 960
    ball_focus_hold_sec: float = 2.0
    ball_focus_ratio_scale: float = 0.82

    # --- 전체화면 공 패스를 건너뛰기 ---------------------------------------
    # 공을 이미 쫓고 있는 동안에는 확대 창(480px)만으로 충분하다. 실측(898장,
    # CPU)에서 공 프레임의 75%(669/891)가 확대 창에서 나왔고, 전체화면 1920px
    # 패스는 한 장에 330ms 로 전체 시간의 절반을 넘게 먹었다.
    # 그래서 '쫓고 있는 동안'에는 ball_full_every 장에 한 번만 전체화면을 본다
    # (놓친 공을 다시 찾기 위한 그물). 공을 놓친 상태면 매번 전체화면을 본다.
    # 0 으로 두면 예전처럼 항상 전체화면을 본다.
    ball_full_every: int = 6

    # 경기장에 공은 하나뿐이다. 후보가 여럿이면 하나만 남긴다.
    # 직전 위치에서 이어지는 쪽을 고른다 — 페널티 마크는 제자리에 붙어
    # 있고 진짜 공은 매끄럽게 움직이므로, 둘 다 보일 때 공이 이긴다.
    # 실측(500프레임): 프레임간 이동폭 90% 지점이 102px → 69px 로 줄었다.
    ball_single: bool = True
    ball_track_gap: int = 15   # 이만큼(원본 프레임) 넘게 끊기면 새 공으로 본다
    ball_reach: float = 140.0  # 연속성 점수가 절반이 되는 거리(px)
    # 연속성 보너스(최대 0.6)가 신뢰도 차이보다 크면 문제가 생긴다. 한번
    # 페널티 마크에 물리면 마크는 늘 제자리라 보너스를 통째로 받고, 멀리
    # 튄 진짜 공은 보너스를 못 받아 영영 못 빠져나온다. 스스로 강화된다.
    #
    # 실측 — 사용자 영상 f342
    #     진짜 공  (852,211) conf 0.79 + 보너스 0.00 = 0.79
    #     페널티마크 (530,615) conf 0.54 + 보너스 0.56 = 1.10  ← 이게 뽑혔다
    #
    # 그래서 '훨씬 확신하는 후보가 있으면 갈아탄다'는 안전장치를 둔다.
    # 이어붙인 후보보다 신뢰도가 이만큼 높은 후보가 있으면 그쪽으로 간다.
    # 선택이 갈린 19개 프레임을 전수로 보니 19개 모두 빨강(현재)이 마크,
    # 초록(갈아타기)이 진짜 공이었다. 0 이면 갈아타지 않는다.
    ball_switch_margin: float = 0.20
    # 공 전용 모델이 붙인 track_id 는 이 번호부터 매긴다. ByteTrack 이
    # 주는 번호와 섞이지 않게 넉넉히 떨어뜨려 둔다.
    ball_id_base: int = 1_000_000

    # --- 유니폼 색으로 팀 나누기 --------------------------------------
    # 심판은 경기장 안에 있어서 위치로는 못 거른다. 상체 색을 모아 군집을
    # 만들고, 가장 큰 두 덩어리를 양 팀으로 본다. 어디에도 안 붙는 사람은
    # team=-1 이 되며 대개 심판이나 골키퍼다. 지우지 않고 표시만 한다.
    # 상체에서 잔디색을 빼고 남은 픽셀로 유니폼 색을 정하는데, 유니폼이
    # 초록 계열이면 그 단계가 상체를 통째로 지운다. 그럴 때 패치 전체를
    # 쓰도록 물러설지. 끄면 예전처럼 그 선수를 건너뛴다.
    torso_grass_fallback: bool = True
    team_split: bool = True
    team_clusters: int = 4     # 군집 수. 이 중 큰 둘이 팀
    team_samples: int = 16     # 팀 색을 정하려고 미리 훑어볼 프레임 수
    # 남은 군집을 팀으로 흡수할 거리 기준. 두 팀 사이 거리의 몇 배까지 볼지.
    # 올리면 심판이 팀에 섞이고, 내리면 같은 팀이 -1 로 새어나간다.
    team_merge_ratio: float = 0.5
    # 팀 색을 뽑을 때만 쓰는 해상도. 0 이면 imgsz 를 그대로 쓴다.
    #
    # 검출과 팀 구분은 요구가 다르다. 축구 전용 모델은 640 에서도 선수를 거의
    # 다 찾지만(22.7명, 1280 은 23.0명), 팀 색은 상체 픽셀이 있어야 뽑힌다.
    # 640 에서는 상체가 20px 남짓이라 실측상 41% 가 team=-1 로 샜다.
    #
    # 그런데 팀 색 보정은 영상 전체에서 team_samples(16) 장만 본다. 그 16 장만
    # 높은 해상도로 처리하면 전체 비용은 거의 안 늘면서 팀 구분만 좋아진다.
    team_imgsz: int = 0
    # 같은 track_id 는 같은 팀이다. 프레임마다 따로 매기면 겹침·모션블러 때문에
    # 한 선수의 라벨이 0/1/-1 을 오간다. 켜두면 분석이 끝난 뒤 track_id 별로
    # 다수결을 내서 CSV 의 team 열을 한 번에 통일한다.
    team_vote: bool = True
    drop_non_team: bool = False  # True 면 team=-1 인 사람을 아예 빼버린다

    # 구간 분석을 할 때, 구간 시작보다 이만큼 앞에서 추적기를 미리 돌린다.
    # 기록은 구간부터 하되 ByteTrack 이 트랙을 확정할 시간을 주는 것이다.
    # 0 으로 두면 워밍업 없이 바로 시작한다 (앞부분 1~2초가 흔들린다).
    range_warmup_sec: float = 4.0

    watch_folder: bool = True  # videos/ 폴더 감시
    skip_done: bool = True     # 이미 결과가 있는 영상은 건너뛰기

    @classmethod
    def load(cls, path: Path) -> "Config":
        cfg = cls()
        if not path.exists():
            return cfg
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cfg
        if not isinstance(data, dict):
            return cfg
        for key, value in data.items():
            if not hasattr(cfg, key):
                continue
            # 손상되거나 손으로 잘못 고친 config.json 하나가 엉뚱한 지점에서
            # 터지지 않도록, 기본값의 타입에 맞춰본 뒤 안 맞으면 기본값을 지킨다.
            current = getattr(cfg, key)
            try:
                if isinstance(current, bool):
                    value = bool(value)
                elif isinstance(current, int):
                    value = int(value)
                elif isinstance(current, float):
                    value = float(value)
                elif isinstance(current, str):
                    value = str(value)
                elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
                    value = tuple(value)
            except (TypeError, ValueError):
                continue
            setattr(cfg, key, value)
        return cfg

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def find_videos(source: Path) -> list[Path]:
    """파일 하나면 그것만, 폴더면 그 안의 영상 전부를 이름순으로."""
    if source.is_file():
        return [source] if source.suffix.lower() in VIDEO_EXT else []
    if source.is_dir():
        return sorted(
            p for p in source.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXT
        )
    return []


def probe_video(path: Path) -> tuple[int, float]:
    """(총 프레임 수, fps). 읽지 못하면 (0, 0.0)."""
    try:
        import cv2
    except ImportError:
        return 0, 0.0

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0, 0.0
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return max(total, 0), fps


def _fill_holes(mask):
    """
    바깥 경계는 건드리지 않고 안쪽 구멍만 메운다.

    테두리를 1픽셀 덧대고 바깥에서 물을 채운 뒤 뒤집으면, 밖과 통하지 않는
    영역(=구멍)만 남는다. 잔디가 화면 가장자리에 닿아 있어도 안전하다.
    """
    import cv2
    import numpy as np

    pad = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = pad.copy()
    scratch = np.zeros((pad.shape[0] + 2, pad.shape[1] + 2), np.uint8)
    cv2.floodFill(flooded, scratch, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def pitch_mask(img, cfg: Config):
    """
    프레임에서 잔디(경기장) 영역을 찾아 흑백 마스크로 돌려준다.

    초록색을 골라낸 뒤 선수가 가려서 생긴 구멍을 메운다. 터치라인 밖
    벤치·관중석·광고판은 초록이 아니므로 자연히 빠진다.

    찾지 못하면 None 을 돌려주고, 그 경우 필터는 적용되지 않는다.
    """
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    scale = 4                       # 1/4 로 줄여서 계산한다. 충분하고 훨씬 빠르다
    small = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    lo = np.array([cfg.pitch_hue[0], cfg.pitch_sat_min, 30], dtype=np.uint8)
    hi = np.array([cfg.pitch_hue[1], 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)

    # 선수·라인 때문에 생긴 잔구멍을 메우고 자잘한 초록 노이즈를 지운다
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    # 초록이 화면의 15% 미만이면 잔디를 못 찾은 것으로 본다 (실내·리플레이 화면 등)
    if mask.mean() < 0.15 * 255:
        return None

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = np.where(labels == biggest, 255, 0).astype(np.uint8)

    if cfg.pitch_convex:
        # 예전 방식. 잔디의 볼록껍질을 씌운다. 구멍은 확실히 메워지지만,
        # 경기장이 화면 가장자리에 닿는 중계 화면에서는 껍질이 관중석·광고판·
        # 중계 그래픽까지 통째로 삼켜서 필터가 사실상 무력화된다.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        hull = cv2.convexHull(np.vstack(contours))
        filled = np.zeros_like(mask)
        cv2.fillConvexPoly(filled, hull, 255)
    else:
        # 바깥 윤곽은 그대로 두고 안쪽 구멍만 메운다. 선수가 잔디를 가려서
        # 생긴 구멍은 채워지고, 터치라인 바깥은 바깥으로 남는다.
        filled = _fill_holes(mask)
        filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        filled = _fill_holes(filled)

    if cfg.pitch_shrink > 0:
        k = max(1, cfg.pitch_shrink // scale)
        filled = cv2.erode(filled, np.ones((k, k), np.uint8))

    return cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)


def torso_color(img, box, cfg: Config):
    """
    사람 상자에서 상체 색을 하나 뽑는다. 못 뽑으면 None.

    머리와 다리를 빼고 가슴께만 본다. 잔디 초록 픽셀은 배경이므로 제외한다.
    Lab 색공간의 a·b 두 채널만 쓴다. 밝기(L)를 버리면 그늘에 들어간 선수와
    햇빛 아래 선수가 같은 색으로 묶인다.
    """
    import cv2
    import numpy as np

    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w < 6 or h < 12:
        return None

    ty1 = int(y1 + 0.15 * h); ty2 = int(y1 + 0.50 * h)
    tx1 = int(x1 + 0.25 * w); tx2 = int(x1 + 0.75 * w)
    ty1, tx1 = max(ty1, 0), max(tx1, 0)
    patch = img[ty1:ty2, tx1:tx2]
    if patch.size == 0 or patch.shape[0] < 2 or patch.shape[1] < 2:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(
        hsv,
        np.array([cfg.pitch_hue[0], cfg.pitch_sat_min, 30], dtype=np.uint8),
        np.array([cfg.pitch_hue[1], 255, 255], dtype=np.uint8),
    )
    keep = green == 0
    if int(keep.sum()) < 12 and cfg.torso_grass_fallback:
        # 유니폼 색이 잔디색과 겹치면 이 단계가 상체를 통째로 지운다.
        #
        # 실측 — 브라질(노랑) 대 멕시코(초록) 영상에서 상체 36개를 재니
        # 36개 전부가 잔디로 판정됐다. 초록 유니폼은 당연하고, 노란 유니폼도
        # 색조가 34 라 잔디 범위(30~95) 하한 바로 안쪽에 들어간다.
        # 그 결과 팀 보정 표본이 342명 중 41명으로 줄고 96% 가 미분류됐다.
        #
        # 이럴 때는 지우지 말고 패치 전체를 쓴다. 상체 패치는 박스의 가운데
        # 절반·위 15~50% 라 원래 유니폼이 대부분이다. 잔디가 조금 섞여도
        # 중앙값이라 버티고, 두 팀 색이 다르면 군집은 그대로 갈린다.
        # (같은 영상에서 표본 41 → 342명, 두 군집 거리 40.2 로 잘 갈렸다)
        if patch.shape[0] * patch.shape[1] < 12:
            return None
        keep = np.ones(green.shape, dtype=bool)
    elif int(keep.sum()) < 12:
        return None

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    return np.median(lab[keep][:, 1:3], axis=0).astype(np.float32)   # (a, b)


def _kmeans(points, k: int, iters: int = 25, seed: int = 0):
    """작은 k-means. sklearn 을 끌어오지 않으려고 직접 둔다."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n = len(points)
    k = min(k, n)
    # k-means++ 비슷하게 서로 먼 점부터 고른다. 초기값이 나쁘면 팀이 안 갈린다
    centers = [points[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(points - c, axis=1) for c in centers], axis=0)
        centers.append(points[int(np.argmax(d))])
    centers = np.array(centers, dtype=np.float32)

    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        dist = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dist, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            if (labels == j).any():
                centers[j] = points[labels == j].mean(axis=0)
    return centers, labels


def calibrate_teams(video: Path, model, cfg: Config, log: Callable[[str], None],
                    profile: "ModelProfile" = None):
    """
    영상 곳곳에서 몇 프레임만 뽑아 상체 색을 모으고 팀 색을 정한다.

    (centers, team_ids) 를 돌려준다. team_ids 는 centers 중 어느 둘이
    양 팀인지 가리키는 색인이다. 실패하면 None.
    """
    import cv2
    import numpy as np

    total, _ = probe_video(video)
    if total <= 0:
        return None

    picks = np.linspace(0, max(total - 1, 0), num=min(cfg.team_samples, total)).astype(int)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None

    # 이 몇 장만 높은 해상도로 봐도 전체 비용은 거의 안 는다.
    size = int(cfg.team_imgsz) or int(cfg.imgsz)
    if size != cfg.imgsz:
        log(f"  팀 색은 {size}px 로 뽑습니다 (검출은 {cfg.imgsz}px, "
            f"{len(picks)}장만 해당)")

    colors = []
    try:
        for idx in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            okay, frame = cap.read()
            if not okay:
                continue
            # 팀 색은 '선수'에서만 뽑는다. 심판·골키퍼 유니폼이 섞이면
            # 군집이 흐려진다. 축구 모델이면 클래스로, COCO 면 person 으로.
            want = sorted(profile.team_targets) if profile else [PERSON_CLASS]
            res = model.predict(frame, imgsz=size, conf=max(cfg.conf, 0.25),
                                classes=want, verbose=False)[0]
            if res.boxes is None or len(res.boxes) == 0:
                continue
            mask = pitch_mask(frame, cfg) if cfg.pitch_filter else None
            for x1, y1, x2, y2 in res.boxes.xyxy.cpu().numpy():
                if mask is not None:
                    fx = int(min(max((x1 + x2) / 2, 0), mask.shape[1] - 1))
                    fy = int(min(max(y2, 0), mask.shape[0] - 1))
                    if mask[fy, fx] == 0:
                        continue        # 경기장 밖 사람은 팀 색 계산에서 뺀다
                c = torso_color(frame, (x1, y1, x2, y2), cfg)
                if c is not None:
                    colors.append(c)
    finally:
        cap.release()

    if len(colors) < 12:
        log("  팀 구분: 표본이 모자라 건너뜁니다")
        return None

    points = np.array(colors, dtype=np.float32)
    centers, labels = _kmeans(points, cfg.team_clusters)
    counts = np.bincount(labels, minlength=len(centers))
    big = list(np.argsort(counts)[::-1][:2])        # 가장 큰 두 덩어리가 양 팀

    # 군집 → 팀 번호. 기본은 '팀 아님'.
    team_map = np.full(len(centers), -1, dtype=int)
    team_map[big[0]] = 0
    team_map[big[1]] = 1

    # 같은 팀이 그늘·조명 때문에 두 군집으로 쪼개지는 일이 잦다. 남은 군집이
    # 어느 팀 색에 충분히 가까우면 그 팀으로 합친다. 두 팀 사이 거리를 자로 쓴다.
    sep = float(np.linalg.norm(centers[big[0]] - centers[big[1]]))
    merged = 0
    if sep > 0:
        for j in range(len(centers)):
            if team_map[j] != -1 or counts[j] == 0:
                continue
            d = [np.linalg.norm(centers[j] - centers[big[0]]),
                 np.linalg.norm(centers[j] - centers[big[1]])]
            nearest = int(np.argmin(d))
            if d[nearest] < cfg.team_merge_ratio * sep:
                team_map[j] = nearest
                merged += 1

    in_team = counts[team_map != -1].sum()
    share = in_team / counts.sum() * 100
    note = f" (비슷한 색 군집 {merged}개 합침)" if merged else ""
    log(f"  팀 구분: 표본 {len(points)}명 → 두 팀이 {share:.0f}%{note}, "
        f"나머지 {100 - share:.0f}%는 심판·골키퍼로 봅니다")
    return centers, team_map


def assign_team(color, calib) -> int:
    """상체 색을 가장 가까운 군집에 붙인다. 팀이 아니면 -1."""
    import numpy as np

    if calib is None or color is None:
        return -1
    centers, team_map = calib
    j = int(np.argmin(np.linalg.norm(centers - color, axis=1)))
    return int(team_map[j])


def has_cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def prepare_model(cfg: Config, root: Path, log: Callable[[str], None] = print):
    """
    실제로 쓸 모델을 만들어 돌려준다.

    GPU 가 없으면 OpenVINO 로 한 번 변환해 두고 그걸 쓴다. CPU 추론이
    1.5~2배 빨라진다. 변환본은 해상도마다 다르므로 models/ 밑에
    해상도까지 붙여 캐시한다. GPU 가 있으면 원본 .pt 가 더 빠르다.
    """
    from ultralytics import YOLO

    if not cfg.use_openvino or has_cuda():
        return YOLO(cfg.model)

    try:
        import openvino  # noqa: F401
    except ImportError:
        log("  OpenVINO 가 없어 일반 모드로 돌립니다 (pip install openvino 하면 빨라집니다)")
        return YOLO(cfg.model)

    stem = Path(cfg.model).stem
    cache = root / "models" / f"{stem}_{cfg.imgsz}_openvino_model"

    if not (cache / "metadata.yaml").exists():
        log(f"  CPU 가속 모델 준비 중 — {stem} @{cfg.imgsz} (처음 한 번만, 1~3분)")
        try:
            exported = Path(YOLO(cfg.model).export(format="openvino", imgsz=cfg.imgsz))
            cache.parent.mkdir(parents=True, exist_ok=True)
            if cache.exists():
                shutil.rmtree(cache, ignore_errors=True)
            shutil.move(str(exported), str(cache))
        except Exception as exc:  # noqa: BLE001 — 변환 실패해도 원본으로 돌아가면 된다
            log(f"  가속 모델 변환 실패({type(exc).__name__}) — 일반 모드로 돌립니다")
            return YOLO(cfg.model)

    log("  CPU 가속 모드 (OpenVINO)")
    return YOLO(str(cache), task="detect")


def resolve_tracker(cfg: Config, work_dir: Path, log: Callable[[str], None]) -> str:
    """
    conf 에 맞춰 임계값을 조정한 트래커 설정 파일을 만들고 그 경로를 돌려준다.

    ByteTrack 은 new_track_thresh(기본 0.25) 미만인 검출로는 새 트랙을 시작하지
    않는다. 사용자가 conf=0.1 로 낮춰 약한 검출까지 받기로 했다면 트래커 쪽
    임계값도 같이 내려줘야 그 검출들이 ID 를 받는다. 안 그러면 표에 track_id 가
    전부 -1 로 찍힌다.
    """
    if not cfg.auto_tracker_thresh:
        return cfg.tracker

    try:
        import yaml
        import ultralytics
    except ImportError:
        return cfg.tracker

    base_path = Path(ultralytics.__file__).parent / "cfg" / "trackers" / cfg.tracker
    if not base_path.exists():
        return cfg.tracker

    try:
        settings = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return cfg.tracker

    stock_new = settings.get("new_track_thresh", 0.25)
    settings["track_high_thresh"] = round(cfg.conf, 4)
    settings["new_track_thresh"] = round(cfg.conf, 4)
    settings["track_low_thresh"] = round(max(0.01, cfg.conf * 0.5), 4)

    # track_buffer 는 트래커가 실제로 본 프레임 수로 센다. vid_stride 를 쓰면
    # 트래커는 N 장마다 한 장만 보므로, 30 을 그대로 두면 원본 기준 30*N 프레임을
    # 기억하는 셈이 된다. stride 3 이면 3.6초 — 그동안 다른 선수가 그 자리에
    # 들어와 ID 를 물려받기 딱 좋다. 원본 프레임 기준 의미를 유지하도록 나눈다.
    stride = max(1, int(cfg.vid_stride))
    buffer_frames = max(5, round(int(cfg.track_buffer) / stride))
    settings["track_buffer"] = buffer_frames
    settings["match_thresh"] = round(float(cfg.match_thresh), 4)

    out = work_dir / f".tracker_{Path(cfg.tracker).stem}.yaml"
    try:
        out.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    except OSError:
        return cfg.tracker

    if cfg.conf < stock_new:
        log(f"  추적 임계값을 {stock_new} → {cfg.conf} 로 낮춰 적용 "
            f"(안 그러면 약한 검출이 ID를 못 받습니다)")
    if buffer_frames != int(cfg.track_buffer):
        log(f"  {stride}프레임마다 1장 처리라 추적 기억을 "
            f"{cfg.track_buffer} → {buffer_frames}장으로 환산 (원본 기준 동일)")
    return str(out)


def _decide_team(votes: Counter) -> int:
    """
    한 track_id 의 프레임별 표를 모아 팀을 확정한다.

    표가 같으면 실제 팀(0/1)을 -1 보다 우선한다. 멀리 있어 상체가 안 잡힌
    프레임이 -1 로 세어지기 때문에, 동률일 때 -1 을 고르면 손해다.
    """
    if not votes:
        return -1
    return min(votes.items(), key=lambda kv: (-kv[1], kv[0] < 0, kv[0]))[0]


def _no_static(cfg: Config) -> Config:
    """
    drop_static_tracks 만 끈 사본.

    축구 모델을 쓰면 중계 그래픽 속 인물이 애초에 안 잡히므로 이 필터가
    할 일이 없다. 그런데 팀 다수결은 여전히 해야 하니, 같은 함수를 타되
    붙박이 제거만 빼려고 설정을 복사해서 넘긴다.
    """
    import copy

    out = copy.copy(cfg)
    out.drop_static_tracks = False
    return out


def static_tracks(motion: dict[int, list], cfg: Config) -> set[int]:
    """
    화면에 붙박여 있던 추적 id 를 골라낸다. 중계 그래픽 속 인물이 여기 걸린다.

    motion 은 track_id -> [min_cx, max_cx, min_cy, max_cy, 키 합, 등장 횟수].
    이동 폭을 사람 키로 나눠서 본다. 멀리 있어 작게 잡힌 선수도 같은 자로
    재기 위해서다. 중계 카메라는 늘 조금씩 움직이므로, 오래 잡혔는데 화면
    좌표가 그대로인 것은 실제 사람일 수 없다.
    """
    if not cfg.drop_static_tracks:
        return set()

    out: set[int] = set()
    for tid, (min_x, max_x, min_y, max_y, sum_h, count) in motion.items():
        if count < max(2, int(cfg.static_min_frames)):
            continue
        height = sum_h / count
        if height <= 0:
            continue
        moved = max(max_x - min_x, max_y - min_y)
        if moved / height < float(cfg.static_max_move):
            out.add(tid)
    return out


def ball_blob(frame_img, cx: float, cy: float, box_h: float) -> tuple[int, float]:
    """
    공 후보 자리의 흰 덩어리를 재서 (최대 밝기, 채움 비율) 을 돌려준다.

    채움 비율 = 흰 덩어리의 넓이 / 그 덩어리의 외접 사각형 넓이.

    공은 원이라 이 값이 원의 이론값 π/4 ≈ 0.785 근처에 온다. 실측 중앙값이
    0.77 이었다. 반면 축구화는 끈·스트라이프 때문에 들쭉날쭉한 모양이라
    0.52 로 낮다. 흰 축구화는 밝기도 포화돼 있고 크기도 공과 비슷해서
    다른 기준으로는 안 걸리는데, 이 값으로는 갈린다.

    상자 높이의 두 배짜리 patch 를 보고, (최대+평균)/2 을 넘는 픽셀 중
    가장 큰 연결 덩어리를 잰다. 못 재면 (0, 0.0) 을 돌려준다.
    """
    import cv2
    import numpy as np

    r = int(max(4, box_h))
    x, y = int(cx), int(cy)
    patch = frame_img[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
    if patch.size == 0:
        return 0, 0.0
    gray = (cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            if patch.ndim == 3 else patch).astype(np.float32)
    peak = int(gray.max())
    thr = (gray.max() + gray.mean()) / 2
    mask = (gray >= thr).astype(np.uint8)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(mask, 8)
    if n < 2:
        return peak, 0.0
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    bw = int(stats[i, cv2.CC_STAT_WIDTH]); bh = int(stats[i, cv2.CC_STAT_HEIGHT])
    area = int(stats[i, cv2.CC_STAT_AREA])
    return peak, area / max(1, bw * bh)


def _ball_anchors(balls: list[dict], step: int, cfg: Config) -> list[dict]:
    """
    보간 끝점으로 쓸 만큼 확실한 공만 고른다.

    신뢰도가 높고, 앞뒤 처리 프레임에도 가까운 자리에 공이 이어서 잡힌 것.
    한 장짜리로 튄 검출(축구화·몸통)은 여기서 빠진다.
    """
    by = {int(b["frame"]): b for b in balls}
    out = []
    for b in balls:
        if float(b["conf"]) < float(cfg.interp_anchor_conf):
            continue
        f = int(b["frame"]); cx = float(b["cx"]); cy = float(b["cy"])
        near = 0
        for k in (1, 2):
            reach = 60.0 * k
            for g in (f - k * step, f + k * step):
                o = by.get(g)
                if o is not None and math.hypot(float(o["cx"]) - cx, float(o["cy"]) - cy) <= reach:
                    near += 1
        if near >= 2:
            out.append(b)
    return out


def chain_ball(csv_path: Path, raw: dict[int, list], cfg: Config, step: int,
               log: Callable[[str], None] = print) -> tuple[int, int]:
    """
    확실한 공에서 앞뒤로 약한 후보를 궤적대로 이어 공 선택을 바로잡는다.

    raw: 처리 프레임 -> [(cx, cy, conf, full)] 공 모델이 본 후보 전부
         (full=True 는 전체 화면, False 는 확대 창. 크기만 넉넉히 맞는 것)
    돌려주는 값: (바꾼 행, 새로 넣은 행). 실패하면 원본을 두고 (0, 0).
    """
    if not cfg.ball_chain or not raw:
        return 0, 0
    step = max(1, int(step))
    idx = {c: i for i, c in enumerate(CSV_COLUMNS)}
    picks: dict[int, list] = {}
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
            rd = csv.reader(fh)
            if next(rd, None) is None:
                return 0, 0
            for row in rd:
                try:
                    if ("ball" in row[idx["class_name"]].lower()
                            and str(row[idx["interpolated"]]) != "1"):
                        picks[int(row[idx["frame"]])] = row
                except (IndexError, ValueError):
                    continue
    except OSError:
        return 0, 0
    if not picks:
        return 0, 0

    def pos(row):
        return float(row[idx["cx"]]), float(row[idx["cy"]])

    anc_conf = float(cfg.ball_chain_anchor_conf)

    def is_anchor(f: int) -> bool:
        row = picks.get(f)
        if row is None or float(row[idx["conf"]]) < anc_conf:
            return False
        px, py = pos(row)
        # 전체 화면에서도 같은 자리를 확실하게 봤어야 한다 (확대 창은 부풀린다)
        if not any(full and c >= anc_conf and abs(cx - px) < 6 and abs(cy - py) < 6
                   for cx, cy, c, full in raw.get(f, ())):
            return False
        for k in (1, 2):
            for g in (f - k * step, f + k * step):
                q = picks.get(g)
                if q is not None:
                    qx, qy = pos(q)
                    if math.hypot(qx - px, qy - py) <= 40.0 * k:
                        return True
        return False

    anchors = [f for f in sorted(picks) if is_anchor(f)]
    aset = set(anchors)
    tol0 = float(cfg.ball_chain_tol)
    tolv = float(cfg.ball_chain_tol_speed)
    max_miss = int(cfg.ball_chain_max_miss)
    min_len = int(cfg.ball_chain_min_len)
    min_dist = float(cfg.ball_chain_min_dist)
    changes: dict[int, tuple] = {}     # 프레임 -> (cx, cy, conf, 끝점 행)
    for a in anchors:
        arow = picks[a]
        ax, ay = pos(arow)
        for d in (-1, 1):
            # 이미 아는 쪽 이웃(가까이 이어진 것만)으로 속도를 잡는다
            vx = vy = 0.0
            for k in (1, 2, 3):
                q = picks.get(a - d * k * step)
                if q is not None:
                    qx, qy = pos(q)
                    if math.hypot(ax - qx, ay - qy) <= 40.0 * k:
                        vx, vy = (ax - qx) / k, (ay - qy) / k
                        break
            x, y, f, miss = ax, ay, a, 0
            chain = []
            while True:
                f += d * step
                if f not in raw or f in aset:
                    break
                px, py = x + vx, y + vy
                tol = tol0 + tolv * math.hypot(vx, vy) * (miss + 1)
                near = [c for c in raw[f] if math.hypot(c[0] - px, c[1] - py) <= tol]
                if not near:
                    miss += 1
                    if miss > max_miss:
                        break
                    x, y = px, py
                    continue
                c = min(near, key=lambda c: math.hypot(c[0] - px, c[1] - py) - 20.0 * c[2])
                vx = 0.5 * vx + 0.5 * (c[0] - x) / (miss + 1)
                vy = 0.5 * vy + 0.5 * (c[1] - y) / (miss + 1)
                x, y, miss = c[0], c[1], 0
                chain.append((f, c))
            if len(chain) < min_len:
                continue
            for f, c in chain:
                cur = picks.get(f)
                if cur is not None:
                    cx, cy = pos(cur)
                    if math.hypot(cx - c[0], cy - c[1]) <= min_dist:
                        continue
                changes.setdefault(f, (c[0], c[1], c[2], arow))
    if not changes:
        return 0, 0

    def make_row(f: int, ch: tuple, base: list, t_sec) -> list:
        cx, cy, conf, _arow = ch
        w = float(base[idx["w"]])
        h = float(base[idx["h"]])
        row = list(base)
        row[idx["frame"]] = f
        if t_sec is not None:
            row[idx["time_sec"]] = t_sec
        row[idx["conf"]] = round(float(conf), 4)
        row[idx["x1"]] = round(cx - w / 2, 1)
        row[idx["y1"]] = round(cy - h / 2, 1)
        row[idx["x2"]] = round(cx + w / 2, 1)
        row[idx["y2"]] = round(cy + h / 2, 1)
        row[idx["cx"]] = round(cx, 1)
        row[idx["cy"]] = round(cy, 1)
        row[idx["fx"]] = round(cx, 1)
        row[idx["fy"]] = round(cy + h / 2, 1)
        row[idx["interpolated"]] = 0
        return row

    replaced = added = 0
    adds = sorted(f for f in changes if f not in picks)
    # 새 행의 시각은 끝점 행의 (시각 / 프레임) 비율로 계산한다
    def t_of(f: int):
        arow = changes[f][3]
        try:
            fa = int(arow[idx["frame"]])
            return round(float(arow[idx["time_sec"]]) * f / fa, 3) if fa > 0 else None
        except (ValueError, IndexError):
            return None

    tmp = csv_path.with_suffix(".chain.tmp")
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as src, \
                tmp.open("w", newline="", encoding="utf-8-sig") as dst:
            rd = csv.reader(src)
            wr = csv.writer(dst)
            wr.writerow(next(rd))
            ai = 0
            for row in rd:
                try:
                    f = int(row[idx["frame"]])
                except (ValueError, IndexError):
                    wr.writerow(row)
                    continue
                # CSV 는 프레임 순이다. 이 행보다 앞 프레임의 새 행을 먼저 쓴다
                while ai < len(adds) and adds[ai] < f:
                    fa = adds[ai]
                    wr.writerow(make_row(fa, changes[fa], changes[fa][3], t_of(fa)))
                    added += 1
                    ai += 1
                if (f in changes and f in picks
                        and "ball" in row[idx["class_name"]].lower()
                        and str(row[idx["interpolated"]]) != "1"):
                    row = make_row(f, changes[f], row, None)
                    replaced += 1
                wr.writerow(row)
            while ai < len(adds):
                fa = adds[ai]
                wr.writerow(make_row(fa, changes[fa], changes[fa][3], t_of(fa)))
                added += 1
                ai += 1
        tmp.replace(csv_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        log(f"  공 궤적 잇기 실패 (원본 유지): {exc}")
        return 0, 0
    log(f"  공 궤적 잇기: 확실한 공 {len(anchors):,}개에서 이어 "
        f"선택 {replaced:,}개 교체 · {added:,}개 추가")
    return replaced, added


def interpolate_ball(csv_path: Path, cfg: Config, fps: float, step: int,
                     log: Callable[[str], None] = print) -> int:
    """
    확실한 공 위치 사이의 빈 처리 프레임을 직선으로 메운다. 메운 행 수를 돌려준다.

    메우는 조건 -- 셋 다 만족해야 한다
      - 양 끝이 엄격한 끝점(_ball_anchors)
      - 빈 구간이 interp_max_gap_sec 이하
      - 두 끝점 사이 속도가 interp_max_speed(초당 선수 키) 이하
    이미 공이 있는 프레임은 건드리지 않는다.

    CSV 를 한 번 흘려 읽으며 해당 프레임 자리에 메운 행을 끼워 넣는다.
    실패하면 원본을 그대로 두고 0 을 돌려준다.
    """
    if not cfg.ball_interpolate or fps <= 0:
        return 0
    step = max(1, int(step))
    idx = {c: i for i, c in enumerate(CSV_COLUMNS)}
    balls: list[dict] = []
    heights: dict[int, list] = {}
    frames_present: set[int] = set()
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as fh:
            for r in csv.DictReader(fh):
                try:
                    f = int(r["frame"])
                except (KeyError, ValueError, TypeError):
                    continue
                frames_present.add(f)
                name = str(r.get("class_name", "")).lower()
                if "ball" in name:
                    balls.append(r)
                elif name in ("player", "goalkeeper", "referee", "person"):
                    try:
                        heights.setdefault(f, []).append(float(r["h"]))
                    except (KeyError, ValueError, TypeError):
                        pass
    except OSError:
        return 0
    if len(balls) < 2:
        return 0

    balls.sort(key=lambda r: int(r["frame"]))
    have = {int(b["frame"]) for b in balls}
    anchors = _ball_anchors(balls, step, cfg)
    max_gap = int(round(float(cfg.interp_max_gap_sec) * fps))

    def med_h(f: int) -> float:
        v = sorted(heights.get(f, []))
        if not v:
            return 0.0
        m = len(v) // 2
        return v[m] if len(v) % 2 else (v[m - 1] + v[m]) / 2

    inserts: dict[int, list] = {}
    for a, b in zip(anchors, anchors[1:]):
        fa, fb = int(a["frame"]), int(b["frame"])
        gap = fb - fa
        if gap <= step or gap > max_gap:
            continue
        missing = [f for f in range(fa + step, fb, step)
                   if f not in have and f in frames_present]
        if not missing:
            continue
        ax, ay = float(a["cx"]), float(a["cy"])
        bx, by_ = float(b["cx"]), float(b["cy"])
        h = med_h(fa) or med_h(fb)
        if h <= 0:
            continue
        speed = math.hypot(bx - ax, by_ - ay) / gap * fps / h
        if speed > float(cfg.interp_max_speed):
            continue
        aw, ah = float(a["w"]), float(a["h"])
        bw, bh = float(b["w"]), float(b["h"])
        for f in missing:
            t = (f - fa) / gap
            cx = ax + (bx - ax) * t; cy = ay + (by_ - ay) * t
            w = aw + (bw - aw) * t; hh = ah + (bh - ah) * t
            row = [""] * len(CSV_COLUMNS)
            row[idx["frame"]] = f
            row[idx["time_sec"]] = round(f / fps, 3)
            row[idx["track_id"]] = a["track_id"]
            row[idx["class_id"]] = a["class_id"]
            row[idx["class_name"]] = a["class_name"]
            row[idx["conf"]] = 0.0
            row[idx["x1"]] = round(cx - w / 2, 1); row[idx["y1"]] = round(cy - hh / 2, 1)
            row[idx["x2"]] = round(cx + w / 2, 1); row[idx["y2"]] = round(cy + hh / 2, 1)
            row[idx["cx"]] = round(cx, 1); row[idx["cy"]] = round(cy, 1)
            row[idx["fx"]] = round(cx, 1); row[idx["fy"]] = round(cy + hh / 2, 1)
            row[idx["w"]] = round(w, 1); row[idx["h"]] = round(hh, 1)
            row[idx["team"]] = ""
            row[idx["interpolated"]] = 1
            inserts.setdefault(f, []).append(row)
    if not inserts:
        return 0

    tmp = csv_path.with_suffix(".interp.tmp")
    added = 0
    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as src, \
                tmp.open("w", newline="", encoding="utf-8-sig") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            header = next(reader, None)
            if header is None:
                tmp.unlink(missing_ok=True)
                return 0
            writer.writerow(header)
            pending = sorted(inserts)
            pi = 0
            for row in reader:
                try:
                    f = int(row[idx["frame"]])
                except (ValueError, IndexError):
                    writer.writerow(row)
                    continue
                # CSV 는 프레임 순이다. 이 행보다 앞 프레임의 메운 행을 먼저 쓴다
                # -- 그러면 메운 행이 자기 프레임의 다른 행들 바로 뒤에 놓인다.
                while pi < len(pending) and pending[pi] < f:
                    for r in inserts[pending[pi]]:
                        writer.writerow(r); added += 1
                    pi += 1
                writer.writerow(row)
            while pi < len(pending):
                for r in inserts[pending[pi]]:
                    writer.writerow(r); added += 1
                pi += 1
        tmp.replace(csv_path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        log(f"  공 빈 구간 메우기 실패 (원본 유지): {exc}")
        return 0
    if added:
        log(f"  공 빈 구간 {len(inserts):,}프레임을 앞뒤 확실한 위치로 메움 "
            f"(interpolated=1, 최대 {cfg.interp_max_gap_sec:.1f}초)")
    return added


def _hist_median(hist: list) -> float:
    """0.05 간격 20칸 신뢰도 히스토그램의 중앙값 (칸 가운데 값)."""
    total = sum(hist)
    if total <= 0:
        return 0.0
    half = total / 2
    acc = 0
    for i, n in enumerate(hist):
        acc += n
        if acc >= half:
            return (i + 0.5) / len(hist)
    return 1.0


def resolve_keeper_tracks(stats: dict[int, list] | None, final: dict[int, int],
                          cfg: Config) -> dict[int, str]:
    """
    골키퍼 라벨이 붙은 트랙이 무엇인지 판정한다.

    stats: track_id -> [선수 수, 골키퍼 수, 심판 수, 신뢰도 히스토그램 20칸...]
    final: track_id -> 다수결로 정한 팀 (선수 라벨 행의 유니폼 색)

    돌려주는 값: track_id -> 'player' | 'post' | 'keeper'
    골키퍼 라벨이 없거나 심판 라벨이 더 많은 트랙은 건드리지 않는다.
    """
    plan: dict[int, str] = {}
    floor = float(cfg.keeper_track_conf)
    if not stats or floor <= 0:
        return plan
    for tid, st_ in stats.items():
        n_player, n_gk, n_ref = st_[0], st_[1], st_[2]
        if n_gk <= 0 or n_ref > n_gk:
            continue
        team = final.get(tid, -1)
        if team in (0, 1):
            plan[tid] = "player"
        elif _hist_median(st_[3:]) < floor:
            plan[tid] = "post"
        else:
            plan[tid] = "keeper"
    return plan


def static_ball_tracks(motion: dict[int, list] | None, cfg: Config) -> set[int]:
    """
    한자리에 붙박여 있던 '공' 트랙을 찾는다.

    공은 경기장에서 가장 많이 움직이는 물체다. 프리킥으로 멈춰 서 있어도
    결국 차이므로 트랙 전체로 보면 반드시 움직인다. 실측에서 흰 옷 입은
    선수의 몸통이 12프레임 연속(움직임 1px) 공으로 잡혔는데, 그런 것을 뺀다.

    motion 은 {track_id: [min_cx, max_cx, min_cy, max_cy, 크기 합, 횟수]}.
    """
    out: set[int] = set()
    if not motion or not cfg.drop_static_ball:
        return out
    for tid, (min_x, max_x, min_y, max_y, size_sum, count) in motion.items():
        if count < max(2, int(cfg.static_ball_min_frames)):
            continue
        size = size_sum / count
        if size <= 0:
            continue
        moved = max(max_x - min_x, max_y - min_y)
        if moved / size < float(cfg.static_ball_max_move):
            out.add(tid)
    return out


class BallPicker:
    """
    공 후보 여럿 중 하나를 고른다.

    경기장에 공은 하나뿐이라는 사실이 근거다. 페널티 마크는 늘 같은 자리에
    붙어 있고 진짜 공은 매끄럽게 움직이므로, 직전 위치에서 이어지는 쪽에
    점수를 얹으면 둘 다 보이는 프레임에서 공이 이긴다.

    공을 한동안 놓쳤다가 다시 찾으면 이어붙일 근거가 없으므로 신뢰도만 본다.
    그때는 새 track_id 를 준다.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.last: tuple[float, float] | None = None
        self.last_frame: int | None = None
        self.vel = (0.0, 0.0)
        self.track_id = int(cfg.ball_id_base)

    def pick(self, frame: int, cands: list[dict], step: int = 1) -> dict | None:
        """cands 는 {'cx','cy','conf', ...} 목록. 고른 하나에 track_id 를 넣어 돌려준다."""
        if not cands:
            return None
        if not self.cfg.ball_single:
            best = max(cands, key=lambda c: c["conf"])
            best["track_id"] = self.track_id
            return best

        step = max(1, int(step))
        gap = (frame - self.last_frame) if self.last_frame is not None else None
        linked = (self.last is not None and gap is not None
                  and 0 < gap <= int(self.cfg.ball_track_gap))

        if linked:
            steps = gap / step
            px = self.last[0] + self.vel[0] * steps
            py = self.last[1] + self.vel[1] * steps
            reach = max(1.0, float(self.cfg.ball_reach))

            def score(c: dict) -> float:
                dx, dy = c["cx"] - px, c["cy"] - py
                dist = (dx * dx + dy * dy) ** 0.5
                return float(c["conf"]) + 0.6 * math.exp(-dist / reach)
        else:
            # 이어붙일 근거가 없다. 새 공으로 보고 번호를 새로 준다.
            if self.last is not None:
                self.track_id += 1

            def score(c: dict) -> float:
                return float(c["conf"])

        best = max(cands, key=score)
        switched = False
        if linked and float(self.cfg.ball_switch_margin) > 0:
            # 훨씬 확신하는 후보가 따로 있으면 갈아탄다. 연속성만 믿으면
            # 한번 잘못 물린 자리에서 못 빠져나온다.
            top = max(cands, key=lambda c: float(c["conf"]))
            if float(top["conf"]) >= float(best["conf"]) + float(self.cfg.ball_switch_margin):
                switched = best is not top
                best = top
        if linked and not switched:
            steps = gap / step
            self.vel = ((best["cx"] - self.last[0]) / steps,
                        (best["cy"] - self.last[1]) / steps)
        else:
            # 갈아탄 직후에는 속도를 모른다. 예전 자리에서 새 자리까지의
            # 거리를 속도로 삼으면 다음 예측이 엉뚱한 곳을 가리키고, 그
            # 틈에 다시 마크로 되돌아간다.
            self.vel = (0.0, 0.0)
        self.last = (best["cx"], best["cy"])
        self.last_frame = frame
        best["track_id"] = self.track_id
        return best


def finalize_tracks(
    csv_path: Path,
    votes: dict[int, Counter],
    motion: dict[int, list],
    cfg: Config,
    log: Callable[[str], None] = print,
    people_ids: set[int] | None = None,
    ball_motion: dict[int, list] | None = None,
    people_stats: dict[int, list] | None = None,
    keeper_ids: set[int] | None = None,
    player_ids: set[int] | None = None,
    class_names: dict[int, str] | None = None,
) -> dict | None:
    """
    분석이 끝난 CSV 를 한 번 훑어 team 열을 track_id 별 다수결로 통일한다.

    프레임마다 따로 매기면 같은 선수가 0 → -1 → 1 로 깜빡인다. 겹침이나
    모션블러 한 번이면 넘어가기 때문이다. track_id 라는 정보를 이미 갖고
    있으니 그걸로 확정한다.

    drop_non_team 도 여기서 처리한다. 프레임 단위로 버리면 다수결에서
    이겼을 선수가 중간중간 사라지므로, 확정된 뒤에 버려야 맞다.

    화면에 붙박여 있던 추적(중계 그래픽 속 인물)도 여기서 뺀다. 한 트랙의
    전체 궤적을 봐야 판단할 수 있으므로 분석이 끝난 뒤가 제자리다.

    한 줄씩 흘려 읽고 임시 파일에 쓴 뒤 갈아끼운다. 도중에 실패하면 원본을
    그대로 남기고 None 을 돌려준다 — 몇 시간짜리 분석 결과를 날릴 수는 없다.
    """
    if people_ids is None:
        people_ids = {PERSON_CLASS}
    frozen = static_tracks(motion, cfg)
    stuck = static_ball_tracks(ball_motion, cfg)
    keeper_ids = keeper_ids or set()
    player_ids = player_ids or set()
    class_names = class_names or {}
    final = {tid: _decide_team(v) for tid, v in votes.items()}
    plan = resolve_keeper_tracks(people_stats, final, cfg)
    posts = {t for t, k in plan.items() if k == "post"}
    keeper_tracks = {t for t, k in plan.items() if k == "keeper"}
    stray = {t for t, k in plan.items() if k == "player"}
    gk_id = min(keeper_ids) if keeper_ids else None
    pl_id = min(player_ids) if player_ids else None
    if not votes and not frozen and not stuck and not plan:
        return None

    changed_ids = sum(1 for tid, v in votes.items() if len(v) > 1)
    frozen_rows = 0
    stuck_rows = 0
    post_rows = 0
    keeper_rows = 0
    stray_rows = 0

    tmp = csv_path.with_suffix(".csv.tmp")
    written = 0
    non_team = 0
    dropped = 0
    ids_seen: set[int] = set()
    class_counts: dict[str, int] = {}
    team_idx = CSV_COLUMNS.index("team")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as src, \
                tmp.open("w", newline="", encoding="utf-8-sig") as dst:
            reader = csv.reader(src)
            writer = csv.writer(dst)
            header = next(reader, None)
            if header is None:
                tmp.unlink(missing_ok=True)
                return None
            writer.writerow(header)

            for row in reader:
                if len(row) != len(CSV_COLUMNS):
                    writer.writerow(row)          # 예상 밖의 줄은 손대지 않는다
                    written += 1
                    continue
                try:
                    tid = int(row[CSV_COLUMNS.index("track_id")])
                    cid = int(row[CSV_COLUMNS.index("class_id")])
                except ValueError:
                    writer.writerow(row)
                    written += 1
                    continue

                if cid in people_ids and tid in posts:
                    post_rows += 1
                    continue              # 골대 -- 선수 라벨 행까지 통째로 뺀다
                if cid in people_ids and tid in keeper_tracks and gk_id is not None:
                    if cid != gk_id:
                        keeper_rows += 1
                    cid = gk_id           # 트랙 전체를 골키퍼로 통일
                    row[CSV_COLUMNS.index("class_id")] = gk_id
                    row[CSV_COLUMNS.index("class_name")] = class_names.get(gk_id, "goalkeeper")
                    row[team_idx] = -1
                elif cid in keeper_ids and tid in stray and pl_id is not None:
                    stray_rows += 1       # 필드 선수가 골대 근처에서 잠깐 골키퍼로 찍힌 것
                    cid = pl_id
                    row[CSV_COLUMNS.index("class_id")] = pl_id
                    row[CSV_COLUMNS.index("class_name")] = class_names.get(pl_id, "player")

                if cid not in people_ids and tid in stuck:
                    stuck_rows += 1
                    continue              # 한자리에 붙박인 '공' — 공이 아니다

                if cid in people_ids:
                    if tid in frozen:
                        frozen_rows += 1
                        continue          # 중계 그래픽 — 사람이 아니다
                    if tid in final and tid not in keeper_tracks:
                        row[team_idx] = final[tid]
                    if str(row[team_idx]) == "-1":
                        non_team += 1
                        if cfg.drop_non_team:
                            dropped += 1
                            continue

                writer.writerow(row)
                written += 1
                if tid >= 0:
                    ids_seen.add(tid)
                name = row[CSV_COLUMNS.index("class_name")]
                class_counts[name] = class_counts.get(name, 0) + 1

        os.replace(tmp, csv_path)
    except (OSError, csv.Error) as exc:
        tmp.unlink(missing_ok=True)
        log(f"  팀 라벨 확정 실패({type(exc).__name__}) — 프레임별 값 그대로 둡니다")
        return None

    if changed_ids:
        log(f"  팀 라벨을 track_id 별 다수결로 확정 (흔들리던 {changed_ids}명 통일)")
    if frozen:
        log(f"  화면에 붙박인 추적 {len(frozen)}개 · {frozen_rows:,}행 제외 "
            f"(중계 그래픽 속 인물)")
    if posts:
        log(f"  골키퍼로 잡힌 골대 {len(posts)}개 · {post_rows:,}행 제외 "
            f"(트랙 신뢰도 중앙 {cfg.keeper_track_conf:.2f} 미만)")
    if keeper_tracks:
        log(f"  골키퍼 트랙 {len(keeper_tracks)}개를 트랙 전체 골키퍼로 통일 "
            f"(선수로 찍혀 있던 {keeper_rows:,}행)")
    if stray_rows:
        log(f"  필드 선수에 잠깐 붙은 골키퍼 라벨 {stray_rows:,}행을 선수로 되돌림")
    if stuck:
        log(f"  한자리에 붙박인 '공' {len(stuck)}개 · {stuck_rows:,}행 제외 "
            f"(흰 옷 선수·잔디 자국)")
    if dropped:
        log(f"  team=-1 로 확정된 {dropped:,}행 제외")
    return {
        "rows": written,
        "non_team": non_team,
        "ids_seen": ids_seen,
        "class_counts": class_counts,
        "static_tracks": len(frozen),
        "static_rows": frozen_rows,
        "static_ball_tracks": len(stuck),
        "static_ball_rows": stuck_rows,
        "goalpost_tracks": len(posts),
        "goalpost_rows": post_rows,
        "keeper_tracks_unified": len(keeper_tracks),
        "keeper_rows_relabeled": keeper_rows,
        "stray_keeper_rows": stray_rows,
    }


def render_boxes(video: Path, csv_path: Path, out_path: Path,
                 log: Callable[[str], None] = print,
                 start_sec: float = 0.0, end_sec: float | None = None) -> bool:
    """
    분석이 끝난 CSV 를 원본 영상에 그려 확인용 영상을 만든다.

    ultralytics 의 save=True 를 쓰지 않는 이유가 있다. 그쪽은 '처리한
    프레임'만 원본 fps 로 이어붙이기 때문에, vid_stride 를 쓰면 3배속으로
    뚝뚝 끊기는 영상이 나온다. 여기서는 원본을 처음부터 다시 훑으면서
    분석 안 한 프레임에는 직전 박스를 남겨두므로 길이도 속도도 원본과 같다.
    팀 색까지 반영된다.
    """
    try:
        import draw_boxes
    except ImportError:
        here = str(Path(__file__).resolve().parent)
        if here not in sys.path:
            sys.path.insert(0, here)
        try:
            import draw_boxes
        except ImportError:
            log("  결과 영상: draw_boxes.py 를 찾지 못해 건너뜁니다")
            return False

    try:
        return bool(draw_boxes.draw(video, csv_path, out_path, log=log,
                                    start_sec=start_sec, end_sec=end_sec))
    except Exception as exc:  # noqa: BLE001 — 영상 실패가 분석 결과를 무효로 만들면 안 된다
        log(f"  결과 영상 만들기 실패({type(exc).__name__}: {exc})")
        log("  CSV 는 정상입니다. 나중에 draw_boxes.py 로 다시 만들 수 있습니다.")
        return False


def parse_time(text: str) -> float:
    """'90' / '1:30' / '01:30:00' 을 초로."""
    nums = [float(p) for p in str(text).strip().split(":")]
    sec = 0.0
    for n in nums:
        sec = sec * 60 + n
    return sec


def _clock(sec: float | None) -> str:
    """초를 '12m30s' 같은 폴더 이름용 문자열로. 파일명에 콜론을 못 쓴다."""
    if sec is None:
        return "end"
    s = int(round(sec))
    return f"{s // 60}m{s % 60:02d}s"


def output_dir_for(video: Path, results_root: Path) -> Path:
    return results_root / video.stem


def is_done(video: Path, results_root: Path) -> bool:
    return (output_dir_for(video, results_root) / "summary.json").exists()


def analyze(
    video: Path,
    results_root: Path,
    cfg: Config,
    model=None,
    log: Callable[[str], None] = print,
    on_progress: Callable[[int, int, float], None] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> dict | None:
    """
    영상 하나를 추적하고 CSV/summary.json 을 남긴다.

    model 을 넘기면 재사용한다. 모델 로딩이 수 초 걸리므로
    여러 영상을 연속 처리할 때는 한 번만 만들어 돌려쓰는 편이 낫다.
    중간에 멈추면 None 을 돌려준다.

    start_sec/end_sec 을 주면 그 구간만 분석한다. 90분 경기 전체를 돌리면
    두 시간이 걸리므로, 보고 싶은 구간만 잘라 보는 편이 현실적이다.
    결과는 다른 폴더(<이름> 12m00s-15m00s)에 남아 전체 분석과 섞이지 않는다.
    """
    total_frames, fps = probe_video(video)
    fps = fps or 30.0
    duration = total_frames / fps if total_frames else 0.0

    # 구간을 잘못 넣는 일이 흔하다. 조용히 빈 결과를 만들지 말고 짚어준다.
    if start_sec < 0 or (end_sec is not None and end_sec < 0):
        log(f"[{video.name}] 시각이 음수입니다. 구간을 다시 확인하세요.")
        return None
    if end_sec is not None and end_sec <= start_sec:
        log(f"[{video.name}] 끝({_clock(end_sec)})이 시작({_clock(start_sec)})보다 "
            f"앞이거나 같습니다. 순서를 바꿔 보세요.")
        return None
    if duration and start_sec >= duration:
        log(f"[{video.name}] 시작 시각 {_clock(start_sec)} 이 영상 길이"
            f"({_clock(duration)})를 넘습니다. 분석할 구간이 없습니다.")
        return None
    if duration and end_sec is not None and end_sec > duration:
        log(f"  끝 시각 {_clock(end_sec)} 이 영상 길이({_clock(duration)})보다 뒤라 "
            f"영상 끝까지만 분석합니다.")
        end_sec = duration

    ranged = start_sec > 0 or end_sec is not None
    lo_frame = max(0, int(start_sec * fps))
    # 구간 시작 앞쪽 워밍업. 영상 맨 앞이면 데울 것도 없다.
    warm_frame = max(0, lo_frame - int(max(0.0, cfg.range_warmup_sec) * fps))
    hi_frame = int(end_sec * fps) if end_sec is not None else (
        total_frames - 1 if total_frames else None)

    out_dir = output_dir_for(video, results_root)
    if ranged:
        tag = f"{_clock(start_sec)}-{_clock(end_sec) if end_sec else 'end'}"
        out_dir = out_dir.with_name(f"{out_dir.name} {tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "tracks.csv"

    if ranged and warm_frame < lo_frame:
        log(f"  추적기 워밍업: {_clock(warm_frame / fps)} 부터 미리 돌리고 "
            f"기록은 {_clock(start_sec)} 부터 합니다")

    span = (hi_frame - lo_frame + 1) if hi_frame is not None else total_frames
    # vid_stride 를 쓰면 실제로 도는 프레임 수는 그만큼 줄어든다
    todo = (span + cfg.vid_stride - 1) // cfg.vid_stride if span else 0
    log(f"[{video.name}] 시작 · {total_frames or '?'}프레임 · {fps:.1f}fps"
        + (f" · 구간 {_clock(start_sec)}~{_clock(end_sec) if end_sec else '끝'}" if ranged else "")
        + (f" · {cfg.vid_stride}프레임마다 1장 → {todo}장 처리" if cfg.vid_stride > 1 else ""))

    if model is None:
        log(f"모델 로딩 중: {cfg.model}  (처음이면 다운로드에 몇 분 걸립니다)")
        model = prepare_model(cfg, results_root.parent, log)

    profile = ModelProfile(getattr(model, "names", None) or {})
    log(f"  {profile.describe()}")

    # 잔디 밖 사람은 축구 모델에서도 거른다.
    #
    # 처음엔 "축구 모델은 스태프·관중을 안 잡는다"며 껐는데 틀렸다. 실측(전술
    # 10:00~13:00, 2초마다 91장) -- 잔디 밖으로 판정된 사람 51건을 전부 눈으로
    # 보니 51건 모두 쓰레기였다.
    #     '선수' 44건: 파란 트랙 위 스태프·노란 조끼 안전요원·사진기자·코치
    #     '심판'  7건: 펜스 옆 안전요원 4 · 광고판 글자 3
    # 진짜 선수는 0건이었다. 선수는 스로인·코너킥 때도 흰 선 밖 잔디 여백
    # 위에 서므로 '초록 잔디 위' 기준이면 빠지지 않는다.
    #
    # 공은 거르지 않는다. 높이 뜬 공은 관중석을 배경으로 걸친다.
    #
    # 관중 색이 잔디와 겹치는 경기(노랑·초록 관중)에서는 마스크가 화면
    # 전체를 잔디로 봐서 아무것도 안 거른다. 해가 없는 쪽으로 실패한다.
    pitch_on = cfg.pitch_filter
    # 중계 그래픽 필터는 축구 모델에서 계속 끈다. 붙박이 판정이 골키퍼처럼
    # 한자리에 오래 서 있는 사람을 지울 수 있다.
    static_on = cfg.drop_static_tracks and not profile.is_soccer
    if profile.is_soccer and cfg.drop_static_tracks:
        log("  중계그래픽 필터는 이 모델에 불필요하므로 끕니다")

    # 공 크기 검사. 기준값을 축구 모델로만 실측했으므로 COCO 에는 걸지 않는다.
    ball_check_on = bool(cfg.ball_size_check) and profile.is_soccer and bool(profile.balls)
    # 밝기 하한. 축구 모델일 때만 건다 (COCO 경로는 건드리지 않는다).
    bright_floor = int(cfg.ball_min_bright) if profile.is_soccer else 0
    fill_floor = float(cfg.ball_min_fill) if profile.is_soccer else 0.0
    if ball_check_on:
        log(f"  공 크기 검사: 사람 키의 {cfg.ball_ratio_min:.2f}~{cfg.ball_ratio_max:.2f} 배만 공으로 인정합니다")

    # 공 전용 모델. 있으면 공은 이쪽 결과만 쓰고 주 모델의 공 검출은 버린다
    # (같은 공을 두 번 적지 않기 위해서다).
    ball_model = None
    ball_picker = None
    if str(cfg.ball_model).strip():
        bp = Path(str(cfg.ball_model))
        if not bp.is_absolute():
            # 결과 폴더 위치와 무관하게 프로그램이 있는 곳을 기준으로 찾는다.
            # (결과를 다른 데 쓰라고 하면 results_root 기준은 엉뚱한 곳을 본다)
            here = Path(__file__).resolve().parent
            for cand in (here / bp, results_root.parent / bp, Path.cwd() / bp):
                if cand.exists():
                    bp = cand
                    break
        if bp.exists():
            try:
                from ultralytics import YOLO
                ball_model = YOLO(str(bp))
                ball_picker = BallPicker(cfg)
                log(f"  공 전용 모델: {bp.name} @{cfg.ball_imgsz}"
                    + ("  (한 프레임에 공 하나만)" if cfg.ball_single else "")
                    + (f"  (주시 확대 {cfg.ball_focus_size}px→{cfg.ball_focus_imgsz})"
                       if cfg.ball_focus else ""))
            except Exception as exc:  # noqa: BLE001 — 없으면 없는 대로 간다
                log(f"  공 전용 모델을 못 읽었습니다: {exc}")
                ball_model = None
        else:
            log(f"  공 전용 모델 파일이 없습니다: {bp}")

    # 설정의 '선수/공' 의도를 이 모델의 클래스 번호로 옮긴다.
    want_people = cfg.classes is None or PERSON_CLASS in (cfg.classes or [])
    want_ball = cfg.classes is None or BALL_CLASS in (cfg.classes or [])
    wanted = profile.wanted_ids(want_people, want_ball)

    calib = None
    if cfg.team_split and want_people and profile.team_targets:
        log("  팀 색을 정하는 중...")
        try:
            calib = calibrate_teams(video, model, cfg, log, profile)
        except Exception as exc:  # noqa: BLE001 — 실패해도 분석은 계속한다
            log(f"  팀 구분 실패({type(exc).__name__}) — 팀 없이 진행합니다")

    started = time.time()
    written = 0
    frames_seen = 0
    dropped = 0          # 경기장 밖이라 버린 사람 수
    non_team = 0         # 양 팀 어디에도 안 붙은 사람 (심판·골키퍼)
    ball_rejected = 0    # 크기가 안 맞아 공에서 뺀 검출 (페널티 마크 등)
    ball_found = 0       # 공 전용 모델이 공을 찾은 프레임 수
    ball_multi = 0       # 공 후보가 둘 이상이었던 프레임 수
    ball_errors = 0      # 공 모델 추론이 실패한 프레임 수
    ball_dim = 0         # 너무 어두워서(=자국으로 보고) 뺀 공 후보 수
    ball_ragged = 0      # 모양이 안 꽉 차서(=축구화로 보고) 뺀 공 후보 수
    ball_seen = 0        # 공 모델이 내놓은 후보 총수 (밝기 경고 판단용)
    ball_still = 0       # 멈춰 있어서 밝기 면제를 못 받은 후보 수 (페널티 마크 등)
    ball_focus_calls = 0 # 주시 확대를 돌린 횟수
    ball_full_calls = 0  # 전체화면 공 패스를 돌린 횟수
    ball_from_focus = 0  # 확대 창에서 찾아 최종 공이 된 프레임 수
    # 원본 프레임 -> 그 프레임에서 공 모델이 본 후보 자리들 (멈춤 판정용)
    ball_hist: dict[int, list] = {}
    # 처리 프레임 -> 공 모델이 본 후보 전부 (궤적 잇기용)
    ball_raw: dict[int, list] = {}
    chain_on = bool(cfg.ball_chain) and ball_model is not None
    focus_on = bool(cfg.ball_focus) and ball_model is not None
    focus_hold = int(round(float(cfg.ball_focus_hold_sec) * fps)) if fps > 0 else 50
    still_from = int(round(float(cfg.ball_still_from_sec) * fps)) if fps > 0 else 6
    still_to = int(round(float(cfg.ball_still_to_sec) * fps)) if fps > 0 else 24
    still_px = float(cfg.ball_still_px)

    def ball_is_still(frame_no: int, cx_: float, cy_: float) -> bool:
        """조금 전(still_from~still_to 프레임 앞)에도 거의 같은 자리에 후보가 있었나."""
        if still_to <= 0 or still_px <= 0:
            return False
        for g in range(frame_no - still_to, frame_no - still_from + 1):
            for hx, hy in ball_hist.get(g, ()):
                if abs(hx - cx_) < still_px and abs(hy - cy_) < still_px:
                    return True
        return False
    # 공 track_id -> [min_cx, max_cx, min_cy, max_cy, 크기 합, 횟수]
    ball_motion: dict[int, list] = {}
    # 사람 track_id -> [선수 수, 골키퍼 수, 심판 수, 신뢰도 히스토그램 20칸]
    people_stats: dict[int, list] = {}
    mask = None          # 최근에 찾은 잔디 영역
    ids_seen: set[int] = set()
    class_counts: dict[str, int] = {}
    team_votes: dict[int, Counter] = {}   # track_id -> 프레임별 팀 표
    # track_id -> [min_cx, max_cx, min_cy, max_cy, 키 합, 등장 횟수]
    # 화면에서 얼마나 움직였는지 재려는 것. 중계 그래픽 걸러내기에 쓴다.
    motion: dict[int, list] = {}

    # 분석이 끝난 뒤 track_id 별 다수결로 team 열을 다시 쓸 것인지.
    # 그럴 거라면 drop_non_team 도 그때 처리해야 한다. 프레임 단위로 미리
    # 버리면 다수결에서 이겼을 선수가 중간중간 빠져버린다.
    voting = bool(cfg.team_vote) and calib is not None
    drop_now = cfg.drop_non_team and not voting

    # pitch_every 는 원본 프레임 기준이다. vid_stride 를 쓰면 루프가 도는
    # 횟수 자체가 줄어드므로 그만큼 나눠야 실제 재계산 간격이 유지된다.
    stride = max(1, int(cfg.vid_stride))
    pitch_every = max(1, round(max(1, int(cfg.pitch_every)) / stride))

    # 2단계 연관을 쓰면 검출을 conf 아래까지 뽑아 트랙 유지에만 쓴다.
    # CSV 에는 conf 이상만 남기므로 결과 표의 의미는 그대로다.
    detect_conf = round(max(0.01, cfg.conf * 0.5), 4) if cfg.bytetrack_recover_low \
        else cfg.conf
    # 이 값 미만인 검출은 표에 안 넣는다. 2단계 연관을 안 쓰면 애초에 검출이
    # conf 이상만 나오므로 0 으로 두어 걸러내는 일 자체가 없게 한다.
    row_conf_floor = cfg.conf if detect_conf < cfg.conf else 0.0
    if row_conf_floor:
        log(f"  약한 검출({detect_conf}~{cfg.conf})은 추적 유지용으로만 씁니다")

    # newline="" 은 윈도우에서 빈 줄이 끼는 것을 막는다
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_COLUMNS)

        tracker_yaml = resolve_tracker(cfg, out_dir.parent, log)
        common = dict(
            imgsz=cfg.imgsz,
            conf=detect_conf,
            iou=cfg.iou,
            classes=wanted,
            tracker=tracker_yaml,
            device=cfg.device or None,
            # ultralytics 자체 저장은 쓰지 않는다. 분석이 끝난 뒤
            # render_boxes() 로 원본 길이·속도 그대로 다시 그린다. (위 설명 참고)
            save=False,
            verbose=False,
        )

        def source_frames():
            """
            원본에서 프레임을 직접 읽어 한 장씩 추적기에 넘긴다.

            ultralytics 에 영상 경로를 통째로 맡기지 않는 이유가 둘 있다.

            1. 중간부터 시작할 수 없다. 구간 분석을 하려면 우리가 읽어야 한다.
            2. 프레임 번호가 어긋난다. ultralytics 는 vid_stride 만큼 grab()
               한 뒤 retrieve() 하므로 stride 3 이면 2, 5, 8... 을 주는데,
               그걸 0, 3, 6 으로 적으면 CSV 의 frame·time_sec 이 밀린다.
               여기서 읽으면 번호가 정의상 정확하다.

            persist=True 면 호출 사이에 추적기 상태가 유지되므로 추적 품질은
            통째로 맡길 때와 같다. 속도도 실측상 오히려 조금 빠르다.
            """
            import cv2

            cap = cv2.VideoCapture(str(video))
            if not cap.isOpened():
                log(f"[{video.name}] 영상을 열 수 없습니다")
                return
            try:
                # 샘플링은 구간 시작이 아니라 '원본 0번 프레임' 기준으로 센다.
                # 그래야 구간 분석과 전체 분석이 같은 프레임을 집어서, 두 결과를
                # 비교하거나 여러 구간을 이어 붙일 수 있다.
                start = warm_frame - (warm_frame % cfg.vid_stride)
                if start > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                idx = start
                step = max(1, int(cfg.vid_stride))
                while hi_frame is None or idx <= hi_frame:
                    if idx % step:
                        # 쓰지 않을 프레임은 grab() 만 한다. read() 와 달리
                        # 화면을 만들어내지 않아서(디코드 생략) 거의 공짜다.
                        # stride 2 에서 디코드가 절반으로 준다.
                        if not cap.grab():
                            break
                        idx += 1
                        continue
                    okay, img = cap.read()
                    if not okay:
                        break
                    yield idx, model.track(img, persist=True, **common)[0]
                    idx += 1
            finally:
                cap.release()

        for frame_idx, (src_frame, result) in enumerate(source_frames()):
            if should_stop():
                log(f"[{video.name}] 사용자 중지")
                return None

            # 카메라가 움직이므로 잔디 영역을 주기적으로 다시 잡는다.
            # 매 프레임 계산할 만큼 비싸지는 않지만 그럴 필요도 없다.
            frame_img = getattr(result, "orig_img", None)
            if pitch_on and frame_idx % pitch_every == 0:
                if frame_img is not None:
                    try:
                        new_mask = pitch_mask(frame_img, cfg)
                    except Exception:  # noqa: BLE001 — 마스크 실패가 분석을 멈추면 안 된다
                        new_mask = None
                    # 못 찾은 프레임(리플레이·클로즈업)에서는 직전 마스크를 유지한다
                    if new_mask is not None:
                        mask = new_mask

            # 워밍업 구간: 추적기와 잔디 마스크를 미리 데워두기만 하고 기록은 안 한다.
            # ByteTrack 은 갓 시작하면 트랙이 확정되지 않아 결과가 흔들린다.
            # 실측상 처음 1.5초쯤이 그렇고, 그 뒤로는 전체 분석과 93% 일치한다.
            if src_frame < lo_frame:
                continue

            frames_seen += 1
            # frame 은 원본 영상의 프레임 번호다 (구간 분석이어도 원본 기준).
            t_sec = (src_frame / fps) if fps > 0 else 0.0
            # 공 크기를 잴 때 견줄 자. 주 모델이 아무것도 못 찾으면 0 이다.
            ball_ref_h = 0.0
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                names = result.names
                xyxy = boxes.xyxy.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)
                conf_arr = boxes.conf.cpu().numpy()
                # 추적이 끊긴 프레임에서는 id 가 없다
                if boxes.id is not None:
                    id_arr = boxes.id.cpu().numpy().astype(int)
                else:
                    id_arr = [-1] * len(cls_arr)

                # 공 크기를 재려면 견줄 자가 필요하다. 같은 프레임에 있는
                # 사람 키의 중앙값을 쓴다. 중앙값이라 한둘이 이상해도 버틴다.
                if ball_check_on or ball_model is not None:
                    people_h = sorted(float(b[3] - b[1])
                                      for b, c in zip(xyxy, cls_arr)
                                      if int(c) in profile.people)
                    if people_h:
                        mid = len(people_h) // 2
                        ball_ref_h = (people_h[mid] if len(people_h) % 2
                                      else (people_h[mid - 1] + people_h[mid]) / 2)

                for (x1, y1, x2, y2), cid, cf, tid in zip(xyxy, cls_arr, conf_arr, id_arr):
                    name = names.get(int(cid), str(cid))

                    # conf 아래로 내려 뽑은 검출은 트랙을 이어붙이는 데만 쓰고
                    # 표에는 넣지 않는다. 사용자가 정한 신뢰도의 의미를 지킨다.
                    if float(cf) < row_conf_floor:
                        continue

                    # 공 전용 모델을 쓰는 중이면 주 모델의 공은 버린다.
                    # 같은 공이 두 줄로 적히는 것을 막는다.
                    if ball_model is not None and int(cid) in profile.balls:
                        continue

                    # 잔디 위의 흰 자국(페널티 마크·라인)과 사람만 한 것을
                    # 공에서 뺀다. 견줄 사람이 없는 프레임은 건드리지 않는다.
                    if ball_check_on and ball_ref_h > 0 and int(cid) in profile.balls:
                        ratio = float(y2 - y1) / ball_ref_h
                        if not (cfg.ball_ratio_min <= ratio <= cfg.ball_ratio_max):
                            ball_rejected += 1
                            continue

                    # 경기장 밖에 서 있는 사람(감독·코치·볼보이·관중)을 버린다.
                    # 판단 기준은 발밑 점이다. 공은 공중에 뜨므로 건드리지 않는다.
                    if mask is not None and int(cid) in profile.people:
                        fx_i = int(min(max((x1 + x2) / 2, 0), mask.shape[1] - 1))
                        fy_i = int(min(max(y2, 0), mask.shape[0] - 1))
                        if mask[fy_i, fx_i] == 0:
                            dropped += 1
                            continue

                    team = ""
                    if int(cid) in profile.non_team:
                        # 모델이 심판·골키퍼라고 말해준다. 색으로 추측할 이유가 없다.
                        team = -1
                    elif int(cid) in profile.team_targets and calib is not None:
                        col = (torso_color(frame_img, (x1, y1, x2, y2), cfg)
                               if frame_img is not None else None)
                        team = assign_team(col, calib)
                        # 같은 선수의 표를 모아둔다. 확정은 분석이 끝난 뒤에.
                        if voting and int(tid) >= 0:
                            team_votes.setdefault(int(tid), Counter())[team] += 1
                        if team == -1:
                            non_team += 1
                            if drop_now:
                                continue

                    writer.writerow([
                        src_frame,
                        round(float(t_sec), 3),
                        int(tid),
                        int(cid),
                        name,
                        round(float(cf), 4),
                        round(float(x1), 1), round(float(y1), 1),
                        round(float(x2), 1), round(float(y2), 1),
                        round(float((x1 + x2) / 2), 1), round(float((y1 + y2) / 2), 1),
                        round(float((x1 + x2) / 2), 1), round(float(y2), 1),
                        round(float(x2 - x1), 1), round(float(y2 - y1), 1),
                        team,
                        0,
                    ])
                    written += 1
                    if tid >= 0 and int(cid) in profile.people:
                        # 트랙 단위 골키퍼 판정용 -- 라벨별 개수와 신뢰도 분포
                        st_ = people_stats.get(int(tid))
                        if st_ is None:
                            st_ = people_stats[int(tid)] = [0, 0, 0] + [0] * 20
                        if int(cid) in profile.keepers:
                            st_[1] += 1
                        elif int(cid) in profile.referees:
                            st_[2] += 1
                        else:
                            st_[0] += 1
                        st_[3 + min(19, max(0, int(float(cf) * 20)))] += 1
                    if tid >= 0:
                        ids_seen.add(int(tid))
                        if int(cid) in profile.people:
                            cx = float((x1 + x2) / 2)
                            cy = float((y1 + y2) / 2)
                            box_h = float(y2 - y1)
                            m = motion.get(int(tid))
                            if m is None:
                                motion[int(tid)] = [cx, cx, cy, cy, box_h, 1]
                            else:
                                m[0] = min(m[0], cx); m[1] = max(m[1], cx)
                                m[2] = min(m[2], cy); m[3] = max(m[3], cy)
                                m[4] += box_h; m[5] += 1
                    class_counts[name] = class_counts.get(name, 0) + 1

            # --- 공 전용 모델 ---------------------------------------------
            # 주 모델과 같은 프레임을 따로 한 번 더 본다. 공만 학습한 모델이라
            # 작은 공을 훨씬 잘 찾는다. 추적은 ByteTrack 대신 BallPicker 가
            # 맡는다 — 공은 하나뿐이라 그 편이 확실하다.
            #
            # 다만 전체화면을 1920px 로 보는 건 한 장에 330ms 로 이 루프에서
            # 가장 비싼 한 줄이다. 공을 이미 쫓고 있는 동안에는 아래의 확대
            # 창(480px→960)만으로 대개 충분하므로 (실측 공 프레임의 75%가
            # 확대 창에서 나왔다) ball_full_every 장에 한 번만 전체화면을
            # 본다. 공을 놓쳤으면 다시 찾아야 하니 매번 본다.
            if ball_model is not None and frame_img is not None:
                # 공을 쫓고 있는 중인가 — 그렇다면 확대 창만으로 대개 충분하다.
                on_ball = (focus_on and ball_picker.last is not None
                           and src_frame - ball_picker.last_frame <= focus_hold)
                every = int(cfg.ball_full_every)
                do_full = (not on_ball) or every <= 0 or (frame_idx % every == 0)
                bres = None
                if do_full:
                    ball_full_calls += 1
                    try:
                        bres = ball_model.predict(frame_img, imgsz=int(cfg.ball_imgsz),
                                                  conf=float(cfg.ball_conf),
                                                  verbose=False)[0]
                    except Exception as exc:  # noqa: BLE001 — 한 프레임 실패로 멈추지 않는다
                        if ball_errors == 0:
                            log(f"  공 모델 추론 실패 (이후 생략): {exc}")
                        ball_errors += 1
                        bres = None
                cands = []
                seen_now: list[tuple[float, float]] = []
                raw_now: list[tuple] = []

                def ball_ok(bx1, by1, bx2, by2, bconf, ratio_scale, full=True):
                    """크기·밝기·모양 검사. 통과하면 후보 dict, 아니면 None."""
                    nonlocal ball_rejected, ball_dim, ball_ragged, ball_still
                    bh = by2 - by1
                    cx_, cy_ = (bx1 + bx2) / 2, (by1 + by2) / 2
                    seen_now.append((cx_, cy_))
                    if chain_on and ball_ref_h > 0:
                        # 궤적 잇기용: 크기만 넉넉히 맞으면 검사 결과와 무관하게 남긴다
                        r_ = bh / max(1e-6, ratio_scale) / ball_ref_h
                        if 0.12 <= r_ <= 0.40:
                            raw_now.append((round(cx_, 1), round(cy_, 1),
                                            round(bconf, 3), full))
                    if ball_check_on and ball_ref_h > 0:
                        # 해상도마다 상자 크기가 달라서 1280 기준으로 되돌린다
                        ratio = bh / max(1e-6, ratio_scale) / ball_ref_h
                        if not (cfg.ball_ratio_min <= ratio <= cfg.ball_ratio_max):
                            ball_rejected += 1
                            return None
                    if bright_floor > 0 or fill_floor > 0:
                        # 흰 덩어리를 재서 밝기와 모양을 함께 본다.
                        #   밝기  — 공은 포화된 흰색, 페인트 자국은 회백색
                        #   채움  — 공은 원(0.785), 축구화는 들쭉날쭉(0.52)
                        peak, fill = ball_blob(frame_img, cx_, cy_, bh)
                        if bright_floor > 0 and peak < bright_floor:
                            # 확신이 높고, 너무 어둡지 않고, 움직이는 후보만 면제
                            exempt = (float(cfg.ball_bright_exempt_conf) > 0
                                      and bconf >= float(cfg.ball_bright_exempt_conf)
                                      and peak >= int(cfg.ball_exempt_min_bright))
                            if exempt and ball_is_still(src_frame, cx_, cy_):
                                exempt = False
                                ball_still += 1
                            if not exempt:
                                ball_rejected += 1
                                ball_dim += 1
                                return None
                        if fill_floor > 0 and fill < fill_floor:
                            ball_rejected += 1
                            ball_ragged += 1
                            return None
                    # 약하면서 제자리에 붙은 후보 (카메라맨·관중석·서 있는 선수의 축구화)
                    if (bconf < float(cfg.ball_still_max_conf)
                            and ball_is_still(src_frame, cx_, cy_)):
                        ball_rejected += 1
                        ball_still += 1
                        return None
                    return {"x1": bx1, "y1": by1, "x2": bx2, "y2": by2,
                            "cx": cx_, "cy": cy_, "conf": bconf}

                if bres is not None and bres.boxes is not None:
                    ball_seen += len(bres.boxes)
                    for bb in bres.boxes:
                        bx1, by1, bx2, by2 = (float(v) for v in bb.xyxy[0])
                        c_ = ball_ok(bx1, by1, bx2, by2, float(bb.conf),
                                     float(cfg.ball_ratio_scale))
                        if c_ is not None:
                            cands.append(c_)

                # 주시 확대: 직전 공 자리 주변을 잘라 2배로 키워 한 번 더 본다.
                if (focus_on and ball_picker.last is not None
                        and src_frame - ball_picker.last_frame <= focus_hold):
                    H_, W_ = frame_img.shape[:2]
                    S_ = min(int(cfg.ball_focus_size), W_, H_)
                    steps = (src_frame - ball_picker.last_frame) / stride
                    steps = min(steps, 5.0)   # 오래 놓쳤으면 속도로 멀리 밀지 않는다
                    px_ = ball_picker.last[0] + ball_picker.vel[0] * steps
                    py_ = ball_picker.last[1] + ball_picker.vel[1] * steps
                    fx0 = int(min(max(px_ - S_ / 2, 0), W_ - S_))
                    fy0 = int(min(max(py_ - S_ / 2, 0), H_ - S_))
                    try:
                        fres = ball_model.predict(
                            frame_img[fy0:fy0 + S_, fx0:fx0 + S_],
                            imgsz=int(cfg.ball_focus_imgsz),
                            conf=float(cfg.ball_conf), verbose=False)[0]
                    except Exception:  # noqa: BLE001 — 확대 실패는 넘어간다
                        fres = None
                    ball_focus_calls += 1
                    if fres is not None and fres.boxes is not None:
                        for bb in fres.boxes:
                            bx1, by1, bx2, by2 = (float(v) for v in bb.xyxy[0])
                            bx1 += fx0; bx2 += fx0; by1 += fy0; by2 += fy0
                            c_ = ball_ok(bx1, by1, bx2, by2, float(bb.conf),
                                         float(cfg.ball_focus_ratio_scale), full=False)
                            if c_ is None:
                                continue
                            c_["focus"] = True
                            # 전체 화면에서 이미 본 공이면 신뢰도 높은 쪽만 남긴다
                            dup = next((k for k in cands
                                        if abs(k["cx"] - c_["cx"]) < 8
                                        and abs(k["cy"] - c_["cy"]) < 8), None)
                            if dup is None:
                                cands.append(c_)
                            elif dup["conf"] < c_["conf"]:
                                dup.update(c_)

                # 멈춤 판정용 기록. 검사 통과 여부와 무관하게 본 자리를 모두 남긴다.
                ball_hist[src_frame] = seen_now
                if chain_on:
                    ball_raw[src_frame] = raw_now
                for old in [k for k in ball_hist if k < src_frame - still_to]:
                    del ball_hist[old]
                if len(cands) > 1:
                    ball_multi += 1
                got = ball_picker.pick(src_frame, cands, step=max(1, cfg.vid_stride))
                if got is not None:
                    ball_rejected += len(cands) - 1     # 고르지 않은 후보도 뺀 것이다
                    btid = int(got["track_id"])
                    writer.writerow([
                        src_frame,
                        round(float(t_sec), 3),
                        btid,
                        BALL_CLASS,
                        "ball",
                        round(float(got["conf"]), 4),
                        round(got["x1"], 1), round(got["y1"], 1),
                        round(got["x2"], 1), round(got["y2"], 1),
                        round(got["cx"], 1), round(got["cy"], 1),
                        round(got["cx"], 1), round(got["y2"], 1),
                        round(got["x2"] - got["x1"], 1), round(got["y2"] - got["y1"], 1),
                        "",
                        0,
                    ])
                    written += 1
                    ball_found += 1
                    if got.get("focus"):
                        ball_from_focus += 1
                    ids_seen.add(btid)
                    class_counts["ball"] = class_counts.get("ball", 0) + 1
                    # 트랙 전체 궤적을 봐야 붙박이인지 알 수 있다. 모아둔다.
                    bh = got["y2"] - got["y1"]
                    mm = ball_motion.get(btid)
                    if mm is None:
                        ball_motion[btid] = [got["cx"], got["cx"],
                                             got["cy"], got["cy"], bh, 1]
                    else:
                        mm[0] = min(mm[0], got["cx"]); mm[1] = max(mm[1], got["cx"])
                        mm[2] = min(mm[2], got["cy"]); mm[3] = max(mm[3], got["cy"])
                        mm[4] += bh; mm[5] += 1

            if on_progress and frame_idx % 10 == 0:
                elapsed = time.time() - started
                eta = 0.0
                if todo and frames_seen > 5 and elapsed > 0:
                    eta = (todo - frames_seen) * elapsed / frames_seen
                on_progress(frames_seen, todo, eta)
            if frames_seen % 300 == 0:
                fh.flush()   # 도중에 끊겨도 여기까지는 남는다
                elapsed = time.time() - started
                speed = frames_seen / elapsed if elapsed else 0
                log(f"  {frames_seen}/{todo or '?'}장 · {written:,}행 · {speed:.1f}장/초")

    if frames_seen == 0:
        # 여기까지 왔는데 한 장도 못 읽었다면 결과를 남길 이유가 없다.
        # summary.json 을 안 써야 다음에 재시도된다.
        log(f"[{video.name}] 읽은 프레임이 없습니다"
            + (" — 구간을 확인하세요." if ranged else " — 영상이 손상됐을 수 있습니다."))
        csv_path.unlink(missing_ok=True)
        try:
            out_dir.rmdir()
        except OSError:
            pass
        return None

    # CSV 를 다 쓴 뒤에야 track_id 별 표와 궤적이 다 모인다. 팀 확정과
    # 붙박이 추적(중계 그래픽) 제거를 여기서 한 번에 처리한다.
    static_count = 0
    static_rows = 0
    stuck_ball_count = 0
    stuck_ball_rows = 0
    if voting or static_on or ball_motion or (people_stats and profile.keepers):
        fixed = finalize_tracks(csv_path, team_votes, motion,
                                cfg if static_on else _no_static(cfg), log,
                                people_ids=profile.people,
                                ball_motion=ball_motion,
                                people_stats=people_stats if profile.keepers else None,
                                keeper_ids=profile.keepers,
                                player_ids=profile.players,
                                class_names=profile.names)
        if fixed is not None:
            written = fixed["rows"]
            non_team = fixed["non_team"]
            ids_seen = fixed["ids_seen"]
            class_counts = fixed["class_counts"]
            static_count = fixed["static_tracks"]
            static_rows = fixed["static_rows"]
            stuck_ball_count = fixed.get("static_ball_tracks", 0)
            stuck_ball_rows = fixed.get("static_ball_rows", 0)
            ball_found = max(0, ball_found - stuck_ball_rows)

    chain_replaced = chain_added = 0
    if chain_on:
        chain_replaced, chain_added = chain_ball(csv_path, ball_raw, cfg,
                                                 max(1, cfg.vid_stride), log)
        written += chain_added
        ball_found += chain_added
        ball_raw.clear()
    interp_rows = 0
    if ball_model is not None:
        interp_rows = interpolate_ball(csv_path, cfg, fps, max(1, cfg.vid_stride), log)
        if interp_rows:
            written += interp_rows

    elapsed = time.time() - started
    summary = {
        "video": video.name,
        "result_video": None,
        "frames": frames_seen,
        "source_fps": round(fps, 2),
        "rows": written,
        "dropped_outside_pitch": dropped,
        "static_graphic_tracks": static_count,
        "dropped_static_rows": static_rows,
        "non_team_detections": non_team,
        "ball_rejected_by_size": ball_rejected,
        "ball_frames": ball_found,
        "static_ball_tracks": stuck_ball_count,
        "static_ball_rows": stuck_ball_rows,
        "ball_multi_candidate_frames": ball_multi,
        "ball_rejected_dim": ball_dim,
        "ball_rejected_ragged": ball_ragged,
        "ball_candidates": ball_seen,
        "ball_still_rejected": ball_still,
        "ball_focus_calls": ball_focus_calls,
        "ball_full_calls": ball_full_calls,
        "ball_frames_from_focus": ball_from_focus,
        "ball_chain_replaced": chain_replaced,
        "ball_chain_added": chain_added,
        "ball_interpolated_rows": interp_rows,
        "unique_track_ids": len(ids_seen),
        "class_counts": class_counts,
        "elapsed_sec": round(elapsed, 1),
        "processed_fps": round(frames_seen / elapsed, 2) if elapsed else 0,
        "config": asdict(cfg),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # 결과 영상은 summary.json 을 쓴 뒤에 만든다. 긴 경기 영상은 인코딩에만
    # 수십 분이 걸리는데, 그 도중에 끊겨서 summary.json 이 없으면 '이미 한 건
    # 건너뛰기'가 안 먹어 몇 시간짜리 분석을 처음부터 다시 돌게 된다.
    # 순서를 이렇게 두면 영상이 실패하거나 중단돼도 분석 결과는 남는다.
    if cfg.save_video:
        log("  결과 영상 만드는 중... (분석 결과는 이미 저장됐습니다)")
        target = out_dir / f"{video.stem}_boxes.mp4"
        # 구간 분석이면 영상도 그 구간만 그린다. 예전에는 구간을 안 넘겨서
        # 5초만 분석해도 90분 경기 전체를 인코딩했다 (10분 만에 548MB).
        if render_boxes(video, csv_path, target, log,
                        start_sec=(lo_frame / fps) if ranged and fps > 0 else 0.0,
                        end_sec=(hi_frame / fps) if ranged and fps > 0 and hi_frame is not None
                        else None):
            summary["result_video"] = target.name
            summary["elapsed_sec"] = round(time.time() - started, 1)
            try:
                summary_path.write_text(
                    json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except OSError:
                pass

    log(
        f"[{video.name}] 완료 · {frames_seen}장 · {written:,}행 · "
        f"추적 ID {len(ids_seen)}개 · {elapsed / 60:.1f}분"
    )
    if dropped:
        log(f"  경기장 밖 인물 {dropped:,}건 제외 (감독·코치·관중)")
    if ball_rejected:
        kept = class_counts.get("ball", 0)
        log(f"  공 오검출 {ball_rejected:,}건 제외 (페널티 마크·심판 등), "
            f"공으로 남긴 것 {kept:,}건")
    if ball_found:
        pct = ball_found / frames_seen * 100 if frames_seen else 0
        log(f"  공: {frames_seen:,}장 중 {ball_found:,}장에서 발견 ({pct:.0f}%)"
            + (f" · 후보가 둘 이상이던 프레임 {ball_multi:,}장" if ball_multi else ""))
    if ball_dim and ball_seen:
        share = ball_dim / ball_seen
        log(f"  어두워서 뺀 공 후보 {ball_dim:,}건 (전체 후보의 {share*100:.0f}%)")
        if share > 0.9:
            log(f"  ! 공 후보 대부분이 밝기 {bright_floor} 미만입니다. 이 영상의 조명이"
                f" 어두우면 config.json 의 ball_min_bright 를 낮추거나 0 으로 끄세요.")
    if ball_focus_calls:
        log(f"  주시 확대 {ball_focus_calls:,}번 · 그중 확대 창에서 찾은 공 {ball_from_focus:,}장")
        if int(cfg.ball_full_every) > 0:
            log(f"  전체화면 공 패스 {ball_full_calls:,}번 "
                f"(공을 쫓는 동안에는 {cfg.ball_full_every}장에 한 번만 — 시간을 아낀다)")
    if ball_ragged:
        log(f"  모양이 공이 아니라서 뺀 후보 {ball_ragged:,}건 (축구화·흰 스타킹)")
    if ball_errors:
        log(f"  공 모델 추론 실패 {ball_errors:,}장")
    if non_team:
        verb = "제외" if cfg.drop_non_team else "team=-1 로 표시"
        log(f"  양 팀 어느 쪽도 아닌 인물 {non_team:,}건 {verb} (심판·골키퍼)")
    log(f"  -> {csv_path}")
    if on_progress:
        on_progress(frames_seen, frames_seen, 0.0)
    return summary


def run_batch(
    videos: Sequence[Path],
    results_root: Path,
    cfg: Config,
    log: Callable[[str], None] = print,
    on_progress: Callable[[int, int, float], None] | None = None,
    should_stop: Callable[[], bool] = lambda: False,
    start_sec: float = 0.0,
    end_sec: float | None = None,
) -> list[dict]:
    """여러 영상을 한 모델로 연속 처리한다. 하나가 실패해도 나머지는 계속 간다."""
    model = None
    done: list[dict] = []

    for video in videos:
        if should_stop():
            break
        if cfg.skip_done and is_done(video, results_root):
            log(f"[{video.name}] 이미 분석됨 — 건너뜀")
            continue
        if model is None:
            log(f"모델 로딩 중: {cfg.model}  (처음이면 다운로드에 몇 분 걸립니다)")
            model = prepare_model(cfg, results_root.parent, log)
        try:
            summary = analyze(
                video, results_root, cfg, model=model,
                log=log, on_progress=on_progress, should_stop=should_stop,
                start_sec=start_sec, end_sec=end_sec,
            )
            if summary:
                done.append(summary)
        except Exception as exc:  # noqa: BLE001 — 한 영상 실패로 배치가 멈추면 안 된다
            log(f"[{video.name}] 실패: {type(exc).__name__}: {exc}")

    return done


def best_local_model(root: Path | None = None) -> str:
    """models/ 에 축구 전용 모델이 있으면 그것, 없으면 yolo11n.

    예전 CLI 프리셋은 전부 COCO 모델을 가리켜서, GPU 없는 PC 에서 옵션 없이
    돌리면 config.json 의 축구 모델이 조용히 yolo11n 으로 바뀌었다 (GUI 에서
    한 번 고친 문제가 CLI 에 그대로 남아 있었다). 같은 해상도에서 프레임당
    사람 12.1명 vs 23.1명 차이다.
    """
    folder = (root or Path(__file__).resolve().parent) / "models"
    if folder.is_dir():
        for f in sorted(folder.glob("*.pt")):
            if "ball" not in f.name.lower():
                return f"models/{f.name}"
    return "yolo11n.pt"


# CLI 프리셋. GUI(app.py)의 것과 같은 값이다. (model, imgsz, vid_stride)
# 모델은 실행할 때 best_local_model() 로 채운다.
CLI_PRESETS: dict[str, tuple[str, int, int]] = {
    "fast": ("", 640, 3),
    "balanced": ("", 960, 2),
    "accurate": ("", 1280, 1),
}

USAGE = """사용법: python tracker.py [영상 또는 폴더] [옵션]

옵션
  --fast        축구 모델 @640, 3프레임에 1장  (GPU 없을 때 권장)
  --balanced    축구 모델 @960, 2프레임에 1장
  --accurate    축구 모델 @1280, 전 프레임     (GPU 필요)
  --as-is       config.json 을 그대로 쓴다 (자동 조정 안 함)
  --from 시각   이 시각부터만 분석 (초, MM:SS, HH:MM:SS)
  --to   시각   이 시각까지만 분석
  -h, --help    이 도움말

구간 예:
  python tracker.py 경기.mp4 --from 12:00 --to 15:00
  90분 경기 전체는 두 시간 넘게 걸린다. 보고 싶은 구간만 자르는 편이 낫다.
  결과는 results/<이름> 12m00s-15m00s/ 에 따로 남는다.

옵션을 안 주면 GPU 유무를 보고 알아서 고른다.
"""


def apply_preset(cfg: Config, name: str) -> Config:
    model, imgsz, stride = CLI_PRESETS[name]
    cfg.model = model or best_local_model()
    cfg.imgsz = imgsz
    cfg.vid_stride = stride
    return cfg


def main(argv: Iterable[str]) -> int:
    args = list(argv)
    if any(a in ("-h", "--help") for a in args):
        print(USAGE)
        return 0

    root = Path(__file__).resolve().parent
    cfg = Config.load(root / "config.json")

    chosen = next((a[2:] for a in args if a.startswith("--") and a[2:] in CLI_PRESETS),
                  None)
    as_is = "--as-is" in args

    start_sec, end_sec = 0.0, None
    positional = []
    skip = False
    for i, a in enumerate(args):
        if skip:
            skip = False
            continue
        if a in ("--from", "--to"):
            if i + 1 >= len(args):
                print(f"{a} 뒤에 시각이 필요합니다 (예: --from 12:00)")
                return 1
            try:
                val = parse_time(args[i + 1])
            except ValueError:
                print(f"시각을 알 수 없습니다: {args[i + 1]}")
                return 1
            if a == "--from":
                start_sec = val
            else:
                end_sec = val
            skip = True
        elif not a.startswith("--"):
            positional.append(a)
    source = Path(positional[0]) if positional else root / "videos"

    if chosen:
        apply_preset(cfg, chosen)
        print(f"프리셋 '{chosen}' 적용 → {cfg.model} @{cfg.imgsz}, "
              f"{cfg.vid_stride}프레임에 1장")
    elif as_is:
        print(f"config.json 그대로 사용 → {cfg.model} @{cfg.imgsz}, "
              f"{cfg.vid_stride}프레임에 1장")
    elif has_cuda():
        print(f"GPU 사용 가능 → config.json 그대로 사용 ({cfg.model} @{cfg.imgsz})")
    else:
        # GUI 는 시작할 때 GPU 를 보고 프리셋을 자동 교정한다. CLI 에도 같은
        # 판단이 필요하다. 안 그러면 화면 없는 서버·컨테이너에서 기본값인
        # yolo11x @1280 전프레임이 그대로 돌아 30분 영상에 며칠이 걸린다.
        before = f"{cfg.model} @{cfg.imgsz}, {cfg.vid_stride}프레임에 1장"
        apply_preset(cfg, "fast")
        after = f"{cfg.model} @{cfg.imgsz}, {cfg.vid_stride}프레임에 1장"
        if before != after:
            print("GPU 가 없어 '빠르게' 프리셋으로 자동 전환합니다.")
            print(f"  {before}  →  {after}")
            print("  원래 설정으로 돌리려면: --as-is · 다른 선택지는 --help")
        else:
            print(f"GPU 없음 → {after}")

    videos = find_videos(source)
    if not videos:
        print(f"영상을 찾지 못했습니다: {source}")
        return 1

    if start_sec or end_sec is not None:
        print(f"구간만 분석합니다: {_clock(start_sec)} ~ "
              f"{_clock(end_sec) if end_sec else '끝'}")
    print(f"{len(videos)}개 영상을 처리합니다.")
    run_batch(videos, root / "results", cfg,
              start_sec=start_sec, end_sec=end_sec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
