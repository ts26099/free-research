"""물병던지기 시뮤레이션 — water bottle flip simulation (single file).

A partially filled bottle is released spinning, followed during free flight,
and **landed**: the run ends with a verdict, stands or falls.  The liquid is
discretised into elements that move under the fictitious forces of the
rotating frame, either as

* thin coaxial discs sliding along the bottle axis (:func:`simulate`),
* parcels that also resolve the cross section (:func:`simulate_parcels`), or
* beads moved by the forces actually acting on them — contacts with a
  restitution, attraction between beads, drag, and the fictitious forces of
  the rotating frame (:func:`simulate_beads`).

As the liquid redistributes, the moment of inertia grows and, angular
momentum being conserved in free flight, the rotation slows down: that is
what lets the bottle land upright.  Angular momentum here means
``L = J omega + L_relative`` — the liquid carries about a tenth of the total
by moving inside the bottle, so it is kept rather than dropped.

The bottle
----------
The bottle is a real PET bottle, not a cylinder: a body, a shoulder that
narrows, and a neck (:func:`radius_profile`).  The defaults are a 500 mL
bottle of 508 mL brimful capacity.  ``neck_radius = bottle_radius`` gives
back a plain cylinder if you want one.  This matters for the question below,
because the filling *fraction* is a fraction of that shape's volume, and
because the shoulder is what holds the liquid back as it is flung towards
the cap.

The research question
---------------------
"어떤 충전율에서 가장 잘 세워지는가, 그리고 그것이 액체의 종류나 질량과
무관한가" — which filling fraction stands best, and is that independent of
what the liquid is?  Two functions answer it directly:

* :func:`tolerance_scan` — for each filling fraction, sweeps the throw
  (``omega_0`` and how hard it is tossed) and measures **how many throws
  stand**.  A good filling fraction is not one that has a perfect throw; it
  is one that forgives an imperfect one.  :func:`coupling_sensitivity`
  re-runs it against the landing model's one fitted assumption, so you can
  see whether the answer depends on it.
* :func:`substance_scan` — holds the filling fraction and changes only the
  liquid.  Free flight has no mass in the equation of motion, so the liquid
  can only reach the answer through ``epsilon = m_bottle / (m_bottle +
  m_liquid)`` and through the viscous ``tau``.  The scan shows exactly that:
  a 30% spread with a real bottle, and essentially none once the bottle is
  made light.

Quick start — the physical quantities can be set three ways::

    run_interactive()                     # ask for every quantity at the prompt

    params = Parameters(bottle_height=0.21, bottle_radius=0.031,
                        water_mass=0.17, omega_0=23, model="parcels")
    results = run(params); report(results); draw(results)

    results = simulate(water_mass=0.15)   # or just override what you need
    parcel = simulate_parcels(n_pieces=19)

    scan = tolerance_scan(); print(format_tolerance(scan))   # 연구용 / research
    print(format_substance(substance_scan()))

    launch_ui()                           sliders, buttons and figures in a browser
                                          (pasting this file into a notebook
                                           cell opens it by itself)

From a shell it also works as a script::

    python 물병던지기_시뮤레이션.py --bottle-height 0.21 --water-mass 0.17
    python 물병던지기_시뮤레이션.py --interactive
    python 물병던지기_시뮤레이션.py --ui           # the Gradio interface
    python 물병던지기_시뮤레이션.py --help          # every quantity is an option

What this model does not do
---------------------------
* The bottle is followed only in free flight.  The phase in the hand, where
  gravity has not yet cancelled and the liquid is already moving, is not
  simulated; ``include_gravity`` only lets you look at that equation.
* Air drag on the bottle is neglected — over half a second it is small
  compared with everything else here.
* The wall drag is a linear ``-(2/tau) v`` per unit mass, not a solved
  boundary layer, and the free surface is never explicitly tracked.
* The landing is an energy criterion, not a simulated collision.  See the
  landing section for the one assumption it makes and how to test it.

References
----------
* P. J. Dekker et al., "Water bottle flipping physics", Am. J. Phys. 86, 733
  (2018), doi:10.1119/1.5052441.
* "The water bottle flipping experiment: a quantitative comparison between
  experiments and numerical simulations", Eur. J. Phys. 45, 065003 (2024),
  doi:10.1088/1361-6404/ad6e43.
* "Fluid Motion Makes the Bottle-Flip Challenge Mechanically Unintuitive But
  Viable", arXiv:2502.12946.
* M. Macklin and M. Müller, "Position based fluids", ACM Trans. Graph. 32,
  104 (2013), doi:10.1145/2461912.2461984.
"""

import argparse
import builtins
import dataclasses
import functools
import math
import os
import sys
import tempfile
import time
import warnings

import numpy as np


# ===========================================================================
# 기본값 — default bottle geometry and constants
# ===========================================================================
# 물병던지기 모델의 물리 상수.
#
# 병은 얇은 벽을 가진 닫힌 원통(흔한 PET 병)으로, 그 안의 물은 병 축을 따라
# 자유롭게 미끄러질 수 있는 얇은 원판("슬라이스")들이 쌓인 것으로 본다.
#
# 모든 값은 SI 단위다.
#
# Physical constants for the water bottle flip model.
#
# The bottle is idealised as a thin-walled, closed cylinder (a standard PET
# bottle), and the water inside it as a stack of thin coaxial discs ("slices")
# that are free to slide along the bottle axis.
#
# All quantities are in SI units.

# --- Bottle geometry and mass ---
# 기본값은 실제로 사람들이 던지는 병 — 500 mL PET 생수병이다.  예전 기본값
# (높이 0.28 m, 반지름 0.04 m)은 부피가 1.4 L 로, 물병던지기 실험에서 쓰는
# 병이 아니었다.  충전율의 최적값을 묻는 연구라면 병의 모양이 곧 답의 일부이므로
# 기본 병을 실제 병으로 바꾼다.
BOTTLE_HEIGHT = 0.21  # Height of the bottle, unit: m
BOTTLE_RADIUS = 0.031  # Radius of the cylindrical body, unit: m
BOTTLE_MASS = 0.022  # Mass of the empty bottle (with the cap), unit: kg

# 실제 PET 병은 원통이 아니다. 몸통이 어깨에서 좁아져 목이 되고, 물은 그 목을
# 통과해야 뚜껑 쪽에 쌓인다.  아래 세 값이 그 모양을 준다 (NECK_RADIUS 를
# BOTTLE_RADIUS 와 같게 두면 예전처럼 순수한 원통이 된다).
NECK_RADIUS = 0.0105  # Inner radius at the cap, unit: m
SHOULDER_START = 0.70  # 어깨가 시작하는 높이 (병 높이에 대한 비율)
SHOULDER_END = 0.88  # 목 반지름에 도달하는 높이 (병 높이에 대한 비율)

# --- Fluid properties ---
WATER_DENSITY = 1000.0  # Density of water, unit: kg/m**3
# 물의 동점성계수 [m**2/s]. 벽 항력의 감쇠 시간을 점성에서 유도할 때 쓴다.
WATER_KINEMATIC_VISCOSITY = 1.0e-6

# --- Environment ---
G = 9.81  # Gravity constant on Earth, unit: m/s**2


# ===========================================================================
# 병의 모양 — the shape of a real bottle
# ===========================================================================
# 실제 PET 병은 원통이 아니라 "몸통 - 어깨 - 목" 이다.  이 차이는 물병던지기
# 에서 그냥 넘길 수 있는 세부가 아니다.
#
# * 같은 충전율이라도 물기둥의 높이가 달라진다. 부피의 대부분이 몸통에 있으므로
#   물은 원통일 때보다 낮게 깔린다.
# * 물이 뚜껑 쪽으로 몰릴 때 목을 지나야 한다. 좁은 목이 물을 붙잡아, 원통
#   모델이 주는 것보다 물이 덜 퍼진다.
# * 병 자체의 관성모멘트도 질량이 몸통에 몰려 있어 원통과 다르다.
#
# 아래 함수들은 회전체 반지름 ``R(z)`` 와, 그 적분인 누적 부피 ``S(z)`` 와,
# 그 역함수를 해석적으로 준다.  누적 부피는 물을 "같은 질량의 원판" 으로 나눌
# 때 결정적이다 — 부피 좌표 ``s = S(z)`` 에서는 원판 하나가 언제나 같은 두께
# ``V_water / n`` 를 차지하므로, 목에서 원판이 얇아지고 몸통에서 두꺼워지는
# 것을 따로 다루지 않아도 비압축성 제약이 그대로 성립한다.

def radius_profile(z, radius=BOTTLE_RADIUS, neck=NECK_RADIUS,
                   height=BOTTLE_HEIGHT, shoulder_start=SHOULDER_START,
                   shoulder_end=SHOULDER_END):
    """바닥에서 ``z`` 만큼 올라간 곳의 병 안쪽 반지름 [m].

    Inner radius of the bottle at height ``z`` above the base, unit: m.

    몸통에서는 ``radius``, 어깨에서는 선형으로 좁아지고, 목에서는 ``neck`` 이다.
    """
    z = np.asarray(z, dtype=float)
    z_1 = shoulder_start * height
    z_2 = shoulder_end * height
    if z_2 <= z_1:
        return np.where(z < z_1, radius, neck)
    t = np.clip((z - z_1) / (z_2 - z_1), 0.0, 1.0)
    return radius + (neck - radius) * t


def cumulative_volume(z, radius=BOTTLE_RADIUS, neck=NECK_RADIUS,
                      height=BOTTLE_HEIGHT, shoulder_start=SHOULDER_START,
                      shoulder_end=SHOULDER_END):
    """바닥에서 ``z`` 까지의 병 내부 부피 [m**3] — 해석적으로.

    Interior volume of the bottle below ``z``, unit: m**3.

    몸통은 원기둥, 어깨는 원뿔대, 목은 다시 원기둥이므로 세 조각을 더하면 된다.
    """
    z = np.clip(np.asarray(z, dtype=float), 0.0, height)
    z_1 = shoulder_start * height
    z_2 = shoulder_end * height

    body = np.pi * radius ** 2 * np.minimum(z, z_1)

    if z_2 > z_1:
        slope = (neck - radius) / (z_2 - z_1)
        z_in = np.clip(z, z_1, z_2)
        r_in = radius + slope * (z_in - z_1)
        if abs(slope) > 1e-15:
            # ∫ pi R(z)**2 dz = pi (R(z)**3 - R_body**3) / (3 slope)
            shoulder = np.pi * (r_in ** 3 - radius ** 3) / (3 * slope)
        else:
            shoulder = np.pi * radius ** 2 * (z_in - z_1)
    else:
        shoulder = np.zeros_like(body)

    throat = np.pi * neck ** 2 * np.maximum(z - z_2, 0.0)
    return body + shoulder + throat


def height_at_volume(volume, radius=BOTTLE_RADIUS, neck=NECK_RADIUS,
                     height=BOTTLE_HEIGHT, shoulder_start=SHOULDER_START,
                     shoulder_end=SHOULDER_END):
    """누적 부피 ``S(z) = volume`` 인 높이 ``z`` [m] — :func:`cumulative_volume` 의 역함수.

    Inverse of :func:`cumulative_volume`: the height holding this volume below it.
    """
    volume = np.asarray(volume, dtype=float)
    z_1 = shoulder_start * height
    z_2 = shoulder_end * height
    v_body = np.pi * radius ** 2 * z_1
    if z_2 > z_1:
        slope = (neck - radius) / (z_2 - z_1)
        if abs(slope) > 1e-15:
            v_shoulder = np.pi * (neck ** 3 - radius ** 3) / (3 * slope)
        else:
            slope = 0.0
            v_shoulder = np.pi * radius ** 2 * (z_2 - z_1)
    else:
        slope = 0.0
        v_shoulder = 0.0

    in_body = volume <= v_body
    in_neck = volume >= v_body + v_shoulder

    z = np.empty_like(volume)
    # 몸통: 단면이 일정하다
    z_body = volume / (np.pi * radius ** 2)
    # 어깨: R**3 이 부피에 선형이다
    if slope:
        cube = np.maximum(radius ** 3 + 3 * slope
                          * (volume - v_body) / np.pi, 0.0)
        z_shoulder = z_1 + (np.cbrt(cube) - radius) / slope
    else:
        z_shoulder = z_1 + (volume - v_body) / (np.pi * radius ** 2)
    # 목: 다시 단면이 일정하다
    z_neck = z_2 + (volume - v_body - v_shoulder) / (np.pi * neck ** 2)

    z = np.where(in_body, z_body, np.where(in_neck, z_neck, z_shoulder))
    return np.clip(z, 0.0, height)


def bottle_shell_moments(radius=BOTTLE_RADIUS, neck=NECK_RADIUS,
                         height=BOTTLE_HEIGHT, shoulder_start=SHOULDER_START,
                         shoulder_end=SHOULDER_END, samples=2001):
    """빈 병의 질량이 어떻게 퍼져 있는지 — 단위 질량당 모멘트들.

    Where the mass of the empty bottle sits, as moments per unit mass.

    병의 질량은 벽면 넓이에 비례해 퍼져 있다고 본다 (같은 두께의 껍질).
    목이 좁으니 거기 있는 질량은 적고, 회전축에서 먼 몸통 벽이 대부분을
    차지한다 — 균질한 원통 껍질로 놓는 예전 식은 병의 질량중심을 너무 높게,
    관성모멘트를 몇 % 크게 잡는다.

    Returns ``(z_cm, ring_term, second_moment)`` per unit mass, from which the
    transverse moment of inertia about any axis follows in closed form::

        J = ring_term + second_moment - 2 z_axis * z_cm + z_axis**2 + y_axis**2

    (all per unit mass, so multiply by the bottle mass).
    """
    z = np.linspace(0.0, height, samples)
    r = radius_profile(z, radius, neck, height, shoulder_start, shoulder_end)
    slope = np.gradient(r, z)
    # 회전면의 넓이 요소: 2 pi R sqrt(1 + R'^2) dz. 여기에 바닥과 뚜껑 원판을
    # 더한다 — 병 질량의 일부는 거기 있고, 바닥은 반지름이 커서 무겁다.
    weight = 2 * np.pi * r * np.sqrt(1 + slope ** 2)

    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    area_wall = float(integrate(weight, z))
    area_base = np.pi * float(radius_profile(0.0, radius, neck, height,
                                             shoulder_start, shoulder_end)) ** 2
    area_cap = np.pi * float(neck) ** 2
    total = area_wall + area_base + area_cap
    if total <= 0:
        return height / 2, radius ** 2 / 2, height ** 2 / 3

    # 벽: 반지름 r 의 고리 하나는 자기 지름축에 대해 m r**2 / 2
    ring_wall = float(integrate(weight * r ** 2 / 2, z))
    first_wall = float(integrate(weight * z, z))
    second_wall = float(integrate(weight * z ** 2, z))
    # 바닥과 뚜껑: 원판 하나는 자기 지름축에 대해 m R**2 / 4
    r_base = float(radius_profile(0.0, radius, neck, height,
                                  shoulder_start, shoulder_end))
    ring_ends = area_base * r_base ** 2 / 4 + area_cap * neck ** 2 / 4
    first_ends = area_cap * height
    second_ends = area_cap * height ** 2

    z_cm = (first_wall + first_ends) / total
    ring_term = (ring_wall + ring_ends) / total
    second_moment = (second_wall + second_ends) / total
    return z_cm, ring_term, second_moment

@functools.lru_cache(maxsize=64)
def _shell_moments_cached(radius, neck, height, shoulder_start, shoulder_end):
    """모양이 같으면 :func:`bottle_shell_moments` 를 다시 적분하지 않는다."""
    return bottle_shell_moments(radius, neck, height, shoulder_start,
                                shoulder_end)


# --- Derived quantities ---
# 병의 내부 부피, 단위: m**3
BOTTLE_VOLUME = float(cumulative_volume(BOTTLE_HEIGHT))
# 병을 가득 채웠을 때의 물 질량, 단위: kg (위 치수라면 약 0.508 kg).
# 치수에서 유도해 두면, 충전율이 실제로 계산하는 병과 어긋나지 않는다.
WATER_MASS_MAX = WATER_DENSITY * BOTTLE_VOLUME


# ===========================================================================
# 물리량 입력 — every physical quantity, and the three interfaces to it
# ===========================================================================
# 시뮬레이션에 들어가는 모든 물리량과, 그것을 넣는 세 가지 방법.
#
# :class:`Parameters` 하나가 입력을 모두 들고 있고, 거기서 따라오는 값(가득
# 찼을 때의 물 질량, 충전율, 감쇠 시간 …)을 유도하며, 계산할 수 있는 값인지
# 검사한다. 시뮬레이션의 다른 어느 곳도 전역 상수를 읽지 않으므로, 병·유체·
# 중력·이산화를 코드를 건드리지 않고 바꿀 수 있다.
#
# 물리량을 넣는 길은 세 가지이며, 모두 이 필드 목록에서 만들어진다. 그래서
# 어느 한쪽이 물리량을 빠뜨릴 수 없다.
#
# * 키워드 인자 — ``simulate(water_mass=0.4, omega_0=25)``
# * 명령행 — ``--bottle-height 0.31``
# * 대화형 입력 — :func:`prompt_parameters` 가 하나씩 묻는다
#
# Every physical and numerical quantity of the simulation, in one place.
#
# :class:`Parameters` holds the inputs, derives what follows from them
# (maximum water mass, filling fraction, damping time, ...) and checks that they
# are physically usable.  Nothing else in the simulation reads a global
# constant, so a caller can change the bottle, the fluid, gravity or the
# discretisation without touching the code:
#
#     params = Parameters(bottle_height=0.31, bottle_radius=0.035,
#                         water_mass=0.4, omega_0=25)
#
# Three interfaces feed it:
#
# * keyword arguments — ``simulate(water_mass=0.4, omega_0=25)``;
# * the command line — every field gets an option, built from the field list
#   itself so none can be forgotten (``--bottle-height 0.31``);
# * an interactive prompt — :func:`prompt_parameters` asks for each quantity in
#   turn, showing the current value and keeping it when the answer is empty.

def _field(default, kind, unit, label, help_text, choices=None, span=None):
    """인터페이스들이 이 물리량을 설명할 수 있게 메타데이터를 붙인 필드.

    A parameter field carrying what the interfaces need to describe it.

    ``span`` is the ``(minimum, maximum, step)`` a slider should offer for a
    numeric quantity: sensible bounds for a hand-held bottle, not hard limits
    (:meth:`Parameters.validated` decides what is actually simulable).
    """
    metadata = {"kind": kind, "unit": unit, "label": label, "help": help_text}
    if choices is not None:
        metadata["choices"] = tuple(choices)
    if span is not None:
        metadata["span"] = tuple(span)
    return dataclasses.field(default=default, metadata=metadata)


@dataclasses.dataclass
class Parameters:
    """Inputs of one flip, with the defaults of a standard 1.4 L PET bottle."""

    # --- bottle ---
    bottle_height: float = _field(
        BOTTLE_HEIGHT, "float", "m", "병 높이 / bottle height",
        "height of the bottle",
        span=(0.1, 0.5, 0.005))
    bottle_radius: float = _field(
        BOTTLE_RADIUS, "float", "m", "병 반지름 / bottle radius",
        "inner radius of the bottle",
        span=(0.015, 0.09, 0.0025))
    bottle_mass: float = _field(
        BOTTLE_MASS, "float", "kg", "빈 병 질량 / empty bottle mass",
        "mass of the empty bottle",
        span=(0.005, 0.2, 0.001))
    neck_radius: float = _field(
        NECK_RADIUS, "float", "m", "목 반지름 / neck radius",
        "inner radius at the cap; set it equal to bottle_radius for a plain "
        "cylinder",
        span=(0.004, 0.09, 0.0005))
    shoulder_start: float = _field(
        SHOULDER_START, "float", "1", "어깨 시작 / shoulder starts at",
        "height where the body starts narrowing, as a fraction of the height",
        span=(0.3, 1.0, 0.01))
    shoulder_end: float = _field(
        SHOULDER_END, "float", "1", "어깨 끝 / shoulder ends at",
        "height where the neck radius is reached, as a fraction of the height",
        span=(0.3, 1.0, 0.01))
    base_radius: float = _field(
        0.0, "float", "m", "바닥 접지 반지름 / base contact radius",
        "radius of the ring the bottle actually stands on; 0 takes 85% of the "
        "body radius, which is what the petaloid base of a PET bottle touches",
        span=(0.0, 0.09, 0.001))

    # --- water ---
    water_mass: float = _field(
        0.17, "float", "kg", "물의 질량 / water mass",
        "mass of liquid in the bottle; a filling fraction of 0.2-0.4 is where "
        "the flip works",
        span=(0.005, 0.5, 0.005))
    water_density: float = _field(
        WATER_DENSITY, "float", "kg/m^3", "물의 밀도 / water density",
        "density of the liquid",
        span=(500.0, 2000.0, 10.0))
    kinematic_viscosity: float = _field(
        WATER_KINEMATIC_VISCOSITY, "float", "m^2/s",
        "동점성계수 / kinematic viscosity",
        "viscosity of the liquid; used only when drag_rate is 0, where the "
        "wall drag is derived from it instead of being set by hand",
        span=(1e-7, 1e-3, 1e-7))
    drag_rate: float = _field(
        10.0, "float", "1/s", "단위질량당 항력 / drag per unit mass",
        "linear wall drag per unit mass; the damping time is tau = 2/drag_rate."
        " Set it to 0 to derive it from kinematic_viscosity instead",
        span=(0.0, 60.0, 0.5))
    restitution: float = _field(
        0.3, "float", "1", "반발계수 / restitution",
        "restitution coefficient at the walls, between 0 and 1; either sign is "
        "accepted, only the magnitude is physical",
        span=(0.0, 1.0, 0.05))

    # --- throw ---
    omega_0: float = _field(
        23.0, "float", "rad/s", "초기 각속도 / initial angular velocity",
        "angular velocity at release",
        span=(1.0, 60.0, 0.5))
    theta_0: float = _field(
        0.0, "float", "rad", "초기 기울기 / initial tilt",
        "tilt of the bottle axis away from straight up at release; 0 is "
        "upright (base down), pi is upside down",
        span=(-3.1416, 3.1416, 0.05))
    gravity: float = _field(
        G, "float", "m/s^2", "중력가속도 / gravity",
        "gravitational acceleration; used for the flight time, and for the "
        "slice equation only when include_gravity is set",
        span=(0.0, 25.0, 0.05))
    include_gravity: bool = _field(
        False, "bool", "", "중력 항 포함 / keep gravity in the slice equation",
        "free flight cancels gravity in the co-moving frame; set this only "
        "for the phase where the bottle is still held")
    t_max: float = _field(
        0.6, "float", "s", "비행 시간 / flight duration",
        "duration of the simulated flight; used when flight = 'fixed'",
        span=(0.05, 2.0, 0.01))
    drop_height: float = _field(
        None, "optional_float", "m", "낙하 높이 / drop height",
        "free-fall height; used when flight = 'drop'")
    flight: str = _field(
        "launch", "choice", "", "비행 시간의 근거 / what sets the flight time",
        "'launch' throws the bottle upward at launch_speed and lands it "
        "release_drop below the release point (a real toss); 'drop' lets it "
        "fall from drop_height; 'fixed' just uses t_max",
        choices=("launch", "drop", "fixed"))
    launch_speed: float = _field(
        3.0, "float", "m/s", "던져 올리는 속도 / launch speed",
        "upward speed of the bottle at release; with the drop below, this is "
        "what sets how long the bottle is in the air",
        span=(0.0, 6.0, 0.05))
    release_drop: float = _field(
        0.1, "float", "m", "놓은 높이 - 착지 높이 / release above landing",
        "how far the bottle's centre of mass ends up below the release point",
        span=(-0.5, 1.5, 0.01))
    impact_absorption: float = _field(
        1.0, "float", "1", "충돌 흡수율 / impact absorption",
        "fraction of the falling kinetic energy the base and the water swallow "
        "on landing; 1 means all of it, and only the residual spin can topple "
        "the bottle",
        span=(0.0, 1.0, 0.05))
    water_coupling: float = _field(
        0.15, "float", "1", "충돌 시 물의 결합도 / water coupling on impact",
        "how much of the liquid's rotation is carried into tipping the bottle "
        "over. The base contact lasts a few ms while the liquid answers over "
        "tens of ms, so most of the liquid's spin never reaches the tipping "
        "motion: 0 is a fully decoupled liquid, 1 is a frozen solid. This is "
        "the one fitted assumption of the landing model - vary it and check "
        "that your conclusion does not move",
        span=(0.0, 1.0, 0.05))

    # --- discretisation ---
    model: str = _field(
        "slices", "choice", "", "모델 / model",
        "discs sliding along the axis, parcels resolving the cross section, "
        "or beads moved by the forces acting on them",
        choices=("slices", "parcels", "beads"))
    n_slices: int = _field(
        100, "int", "", "원판 개수 / number of discs", "disc model resolution",
        span=(10, 400, 10))
    n_pieces: int = _field(
        19, "int", "", "단면당 조각 수 / pieces per cross section",
        "parcel model resolution; the cost grows as its square",
        span=(3, 61, 1))
    n_steps: int = _field(
        1000, "int", "", "시간 스텝 수 / time steps",
        "number of time steps over the flight",
        span=(100, 4000, 100))
    water_at_top: bool = _field(
        True, "bool", "", "물이 뚜껑에 붙어 시작 / water starts at the cap",
        "where the water rests at the moment of release")
    incompressible: bool = _field(
        True, "bool", "", "비압축성 제약 / incompressibility",
        "keep the water elements from passing through each other")
    coriolis: bool = _field(
        True, "bool", "", "코리올리 항 / Coriolis force",
        "parcel model only; switching it off reduces the dynamics to the "
        "disc model")
    euler: bool = _field(
        True, "bool", "", "오일러 항 / Euler force",
        "parcel model only; the azimuthal force from a changing omega")
    relative_momentum: bool = _field(
        True, "bool", "", "물의 상대 각운동량 / water's own angular momentum",
        "keep the angular momentum the liquid carries by moving inside the "
        "bottle, so that omega follows from L_total = J omega + L_relative "
        "rather than from L = J omega. It is worth about 10% of the total")
    overlap_iterations: int = _field(
        3, "int", "", "겹침 투영 반복 / overlap sweeps per step",
        "parcel model only; projection sweeps used to remove overlaps",
        span=(0, 10, 1))

    # --- beads (the force-based model) ---
    n_beads: int = _field(
        400, "int", "", "구슬 개수 / number of beads",
        "how many spheres the water is made of",
        span=(20, 4000, 20))
    bead_radius: float = _field(
        0.0, "float", "m", "구슬 반지름 / bead radius",
        "radius of one bead; 0 derives it from the water volume, so that the "
        "beads at close packing fill exactly that volume",
        span=(0.0, 0.02, 0.0005))
    bead_mass: float = _field(
        0.0, "float", "kg", "구슬 질량 / bead mass",
        "mass of one bead; 0 derives it from water_mass / n_beads. Given "
        "explicitly, the water mass becomes n_beads * bead_mass",
        span=(0.0, 0.01, 0.0001))
    contact_stiffness: float = _field(
        5000.0, "float", "N/m", "접촉 강성 / contact stiffness",
        "spring constant of a bead contact; a stiffer contact needs more "
        "substeps",
        span=(10.0, 20000.0, 10.0))
    surface_tension: float = _field(
        0.072, "float", "N/m", "표면장력 / surface tension",
        "sets the attraction between beads (0.072 for water); zero gives a "
        "cohesionless granular fill",
        span=(0.0, 0.2, 0.001))
    cohesion_coefficient: float = _field(
        1.0, "float", "1", "인력 계수 / cohesion coefficient",
        "the pair attraction is cohesion_coefficient * surface_tension * "
        "bead_radius; coarse beads need it below one",
        span=(0.0, 20.0, 0.05))
    cohesion_range: float = _field(
        1.25, "float", "1", "인력 사거리 / cohesion range",
        "how far the attraction reaches, in contact diameters",
        span=(1.0, 3.0, 0.05))
    bead_substeps: int = _field(
        0, "int", "", "구슬 서브스텝 / substeps per step",
        "substeps of the contact integration per output step; 0 takes as many "
        "as the contact stiffness demands",
        span=(0, 200, 1))
    substep_limit: int = _field(
        50, "int", "", "서브스텝 상한 / substep ceiling",
        "구슬이 많아지면 안정 조건이 요구하는 서브스텝 수가 계산을 끝낼 수 "
        "없을 만큼 늘어난다. 이 상한을 넘으면 서브스텝을 여기서 자르고 접촉 "
        "강성을 그 간격에서 안정한 값까지 낮춘다 (0 = 상한 없음) / ceiling on "
        "the substeps per step; the contact stiffness is softened to stay "
        "stable at that step, so a big run finishes",
        span=(0, 400, 10))

    # --- derived quantities ---

    @property
    def shape(self):
        """반지름 곡선을 부르는 데 필요한 인자들.

        The arguments the radius-profile helpers take.
        """
        return dict(radius=self.bottle_radius, neck=self.neck_radius,
                    height=self.bottle_height,
                    shoulder_start=self.shoulder_start,
                    shoulder_end=self.shoulder_end)

    def radius_at(self, z):
        """높이 ``z`` 에서의 병 안쪽 반지름 [m].

        Inner radius of this bottle at height ``z``, unit: m.
        """
        return radius_profile(z, **self.shape)

    def volume_below(self, z):
        """높이 ``z`` 아래의 병 내부 부피 [m**3].

        Interior volume below ``z``, unit: m**3.
        """
        return cumulative_volume(z, **self.shape)

    def height_of_volume(self, volume):
        """부피 ``volume`` 을 담는 높이 [m] — :meth:`volume_below` 의 역함수.

        Height holding this volume below it, unit: m.
        """
        return height_at_volume(volume, **self.shape)

    @property
    def bottle_volume(self):
        """병의 내부 부피 [m^3] — 몸통과 어깨와 목을 합친 것.

        Interior volume of the bottle, unit: m**3.
        """
        return float(cumulative_volume(self.bottle_height, **self.shape))

    @property
    def contact_radius(self):
        """병이 실제로 딛고 서는 바닥 링의 반지름 [m].

        Radius of the ring the bottle actually stands on, unit: m.

        PET 병의 바닥은 평평하지 않고 꽃잎 모양이라, 닿는 자리는 몸통 반지름
        보다 작다.  넘어지는지 서는지를 가르는 것이 바로 이 반지름이다.
        """
        return self.base_radius if self.base_radius else 0.85 * self.bottle_radius

    @property
    def shell_moments(self):
        """빈 병의 질량 분포 — 단위 질량당 ``(z_cm, ring, second)``.

        Mass distribution of the empty bottle, per unit mass.
        """
        return _shell_moments_cached(self.bottle_radius, self.neck_radius,
                                     self.bottle_height, self.shoulder_start,
                                     self.shoulder_end)

    @property
    def bottle_center_of_mass(self):
        """빈 병만의 질량중심 높이 [m].

        Height of the centre of mass of the empty bottle, unit: m.

        원통이라면 정확히 절반이지만, 목이 좁은 실제 병은 질량이 아래에
        몰려 있어 그보다 낮다.
        """
        return self.shell_moments[0]

    def bottle_inertia(self, z_axis, y_axis=0.0):
        """``(y_axis, z_axis)`` 를 지나는 횡축에 대한 빈 병의 관성모멘트 [kg m^2].

        Transverse moment of inertia of the empty bottle about that axis.
        """
        z_cm, ring, second = self.shell_moments
        return self.bottle_mass * (ring + second - 2 * z_axis * z_cm
                                   + z_axis ** 2 + y_axis ** 2)

    @property
    def water_mass_max(self):
        """병을 가득 채웠을 때의 액체 질량 [kg].

        Mass of liquid in a completely full bottle, unit: kg.
        """
        return self.water_density * self.bottle_volume

    @property
    def water_volume(self):
        """액체의 부피 [m^3].

        Volume of the liquid, unit: m**3.
        """
        return self.water_mass / self.water_density

    @property
    def filling_fraction(self):
        """병 부피에서 액체가 차지하는 비율.

        Fraction of the bottle volume occupied by the liquid.
        """
        return self.water_volume / self.bottle_volume

    @property
    def water_height(self):
        """가만히 세워 두었을 때 물의 수면 높이 [m].

        Height of the free surface of the liquid at rest, unit: m.

        원통이라면 충전율에 병 높이를 곱한 것이지만, 목이 있는 병에서는
        부피의 대부분이 몸통에 있으므로 수면이 그보다 낮다.
        """
        return float(self.height_of_volume(self.water_volume))

    @property
    def epsilon(self):
        """빈 병이 전체 질량에서 차지하는 비율.

        Mass fraction of the empty bottle.

        자유비행에서 물의 운동방정식에는 질량이 들어가지 않는다. 물의 종류와
        질량이 결과에 들어오는 통로는 이 값 하나뿐이다 (그리고 점성을 통한
        ``tau``).  "물질의 종류나 질량과 무관하다" 는 주장을 검사할 때 보아야
        할 수는 충전율과 이 값이다.
        """
        return self.bottle_mass / (self.bottle_mass + self.water_mass)

    @property
    def tau(self):
        """벽 항력의 감쇠 시간 [s].

        Damping time of the wall drag, unit: s.

        ``drag_rate`` 를 0 으로 두면 동점성계수에서 유도한다: 병 반지름 규모의
        전단층이 운동량을 벽으로 나르므로 ``tau ~ R**2 / nu`` 이고, 물처럼
        점성이 낮은 액체에서는 이 값이 비행 시간보다 훨씬 길다 — 즉 물의
        점성은 뒤집기에 거의 관여하지 않는다.
        """
        if self.drag_rate > 0:
            return 2 / self.drag_rate
        return self.bottle_radius ** 2 / max(self.kinematic_viscosity, 1e-12)

    @property
    def drag_per_mass(self):
        """운동방정식에 들어가는 항력 계수 ``2 / tau`` [1/s].

        The drag coefficient ``2 / tau`` entering the equations of motion.
        """
        return 2 / self.tau

    @property
    def bounce(self):
        """벽 반사에 쓰는 반발계수 — 언제나 0 과 1 사이.

        Restitution used at the walls, always between 0 and 1.

        예전 코드는 이 값이 음수여야 반사가 되고 양수면 조용히 물이 벽을
        통과했다.  부호는 여기서 붙이므로, 어느 쪽으로 넣어도 물리가 같다.
        """
        return min(abs(self.restitution), 1.0)

    @property
    def duration(self):
        """시뮬레이션하는 비행 시간 [s].

        Simulated flight duration, unit: s.

        ``flight = "launch"`` 이면 실제로 던져 올린 병이 공중에 있는 시간이다:
        위로 ``launch_speed`` 로 던져 ``release_drop`` 만큼 아래에 떨어지므로
        ``t = (v + sqrt(v**2 + 2 g h)) / g``.  비행 시간은 뒤집기의 성패를
        좌우하는데, 예전처럼 0.6 s 로 고정해 두면 던지는 세기를 바꿔도 시간이
        따라오지 않아 실제 던지기와 어긋난다.
        """
        if self.flight == "launch":
            v = self.launch_speed
            g = self.gravity
            if g <= 0:
                raise ValueError("a launched flight needs a non-zero gravity")
            inside = v ** 2 + 2 * g * self.release_drop
            if inside < 0:
                raise ValueError(
                    "the bottle never comes back down to the landing height")
            return (v + math.sqrt(inside)) / g
        if self.flight == "drop":
            if self.drop_height is None:
                raise ValueError("flight = 'drop' needs a drop_height")
            return math.sqrt(2 * self.drop_height / self.gravity)
        if self.drop_height is not None:
            return math.sqrt(2 * self.drop_height / self.gravity)
        return self.t_max

    def cohesion_force(self, bead_radius):
        """맞닿은 두 구슬 사이의 인력 [N].

        Attraction between two beads in contact, unit: N.
        """
        return self.cohesion_coefficient * self.surface_tension * bead_radius

    @property
    def axial_gravity(self):
        """원판 방정식에 들어가는 중력 [m/s^2] (자유비행에서는 0).

        Gravity entering the slice equation, unit: m/s**2 (0 in free flight).
        """
        return self.gravity if self.include_gravity else 0.0

    def replace(self, **overrides):
        """몇 개 값만 바꾼 사본을 만들고 검사한다.

        A copy with some fields changed, validated.
        """
        unknown = set(overrides) - {f.name for f in dataclasses.fields(self)}
        if unknown:
            raise TypeError("unknown parameter(s): %s"
                            % ", ".join(sorted(unknown)))
        return dataclasses.replace(self, **overrides).validated()

    def validated(self):
        """계산할 수 없는 입력이면 :class:`ValueError` 를 낸다.

        Raise :class:`ValueError` if the inputs cannot be simulated.
        """
        positive = {"bottle_height": self.bottle_height,
                    "bottle_radius": self.bottle_radius,
                    "bottle_mass": self.bottle_mass,
                    "water_density": self.water_density,
                    "neck_radius": self.neck_radius,
                    "kinematic_viscosity": self.kinematic_viscosity}
        for name, value in positive.items():
            if not value > 0:
                raise ValueError("%s must be positive (got %r)" % (name, value))
        if self.drag_rate < 0:
            raise ValueError("drag_rate must not be negative")
        if self.neck_radius > self.bottle_radius:
            raise ValueError("the neck cannot be wider than the body")
        if not 0 < self.shoulder_start <= self.shoulder_end <= 1:
            raise ValueError(
                "the shoulder must satisfy 0 < shoulder_start <= shoulder_end "
                "<= 1 (got %r, %r)" % (self.shoulder_start, self.shoulder_end))
        if self.base_radius < 0 or self.base_radius > self.bottle_radius:
            raise ValueError("base_radius must be between 0 and bottle_radius")
        if self.flight not in ("launch", "drop", "fixed"):
            raise ValueError("flight must be 'launch', 'drop' or 'fixed'")
        if self.launch_speed < 0:
            raise ValueError("launch_speed must not be negative")
        if not 0 <= self.impact_absorption <= 1:
            raise ValueError("impact_absorption must be between 0 and 1")
        if not 0 <= self.water_coupling <= 1:
            raise ValueError("water_coupling must be between 0 and 1")
        if not 0 < self.water_mass < self.water_mass_max:
            raise ValueError(
                "water_mass must be between 0 and %.4f kg for this bottle "
                "(got %r)" % (self.water_mass_max, self.water_mass))
        if self.gravity < 0:
            raise ValueError("gravity must not be negative")
        if abs(self.restitution) > 1:
            raise ValueError("|restitution| must not exceed 1")
        if self.drop_height is not None:
            if self.drop_height < 0:
                raise ValueError("drop_height must not be negative")
            if self.gravity == 0:
                raise ValueError("a drop height needs a non-zero gravity")
        if self.duration <= 0:
            raise ValueError("the flight duration must be positive")
        if self.n_steps < 2:
            raise ValueError("n_steps must be at least 2")
        for name in ("n_slices", "n_pieces"):
            if getattr(self, name) < 1:
                raise ValueError("%s must be at least 1" % name)
        if self.overlap_iterations < 0:
            raise ValueError("overlap_iterations must not be negative")
        if self.model not in ("slices", "parcels", "beads"):
            raise ValueError("model must be 'slices', 'parcels' or 'beads'")
        if self.n_beads < 1:
            raise ValueError("n_beads must be at least 1")
        if self.bead_radius < 0 or self.bead_mass < 0:
            raise ValueError("bead radius and mass must not be negative")
        if not self.contact_stiffness > 0:
            raise ValueError("contact_stiffness must be positive")
        if self.surface_tension < 0 or self.cohesion_coefficient < 0:
            raise ValueError("the cohesion must not be negative")
        if self.cohesion_range < 1:
            raise ValueError("cohesion_range must be at least one diameter")
        if self.bead_substeps < 0:
            raise ValueError("bead_substeps must not be negative")
        if self.substep_limit < 0:
            raise ValueError("substep_limit must not be negative")
        return self

    def describe(self):
        """입력한 물리량과 거기서 따라오는 값들을 여러 줄로 정리한다.

        Multi-line summary of the inputs and of what follows from them.
        """
        lines = ["--- 입력 물리량 / inputs " + "-" * 30]
        for group, names in FIELD_GROUPS:
            lines.append("[%s]" % group)
            for name in names:
                field = FIELDS[name]
                value = getattr(self, name)
                unit = field.metadata["unit"]
                if isinstance(value, bool):
                    shown = "yes" if value else "no"
                elif value is None:
                    shown = "-"
                elif isinstance(value, float):
                    shown = "%.6g" % value
                else:
                    shown = str(value)
                lines.append("  %-46s %12s %s"
                             % (field.metadata["label"], shown, unit))
        lines.append("--- 유도량 / derived " + "-" * 32)
        lines.append("  병 부피 / bottle volume                      %12.1f mL"
                     % (1e6 * self.bottle_volume))
        lines.append("  물 최대 질량 / full bottle holds              %12.4f kg"
                     % self.water_mass_max)
        lines.append("  충전율 / filling fraction                    %12.4f"
                     % self.filling_fraction)
        lines.append("  수면 높이 / free surface at rest              %12.4f m"
                     % self.water_height)
        lines.append("  빈 병 질량비 / bottle mass fraction           %12.4f"
                     % self.epsilon)
        lines.append("  감쇠 시간 / damping time tau                 %12.4f s"
                     % self.tau)
        lines.append("  비행 시간 / flight duration                   %12.4f s"
                     % self.duration)
        lines.append("  접지 반지름 / base contact radius             %12.4f m"
                     % self.contact_radius)
        return "\n".join(lines)


FIELDS = {field.name: field for field in dataclasses.fields(Parameters)}

FIELD_GROUPS = [
    ("병 / bottle", ("bottle_height", "bottle_radius", "bottle_mass",
                     "neck_radius", "shoulder_start", "shoulder_end",
                     "base_radius")),
    ("물 / water", ("water_mass", "water_density", "kinematic_viscosity",
                    "drag_rate", "restitution")),
    ("던지기 / throw", ("omega_0", "theta_0", "gravity", "include_gravity",
                        "flight", "launch_speed", "release_drop",
                        "impact_absorption", "t_max", "drop_height")),
    ("이산화 / discretisation",
     ("model", "n_slices", "n_pieces", "n_steps", "water_at_top",
      "incompressible", "coriolis", "euler", "relative_momentum",
      "overlap_iterations")),
    ("구슬 / beads", ("n_beads", "bead_radius", "bead_mass",
                      "contact_stiffness", "surface_tension",
                      "cohesion_coefficient", "cohesion_range",
                      "bead_substeps", "substep_limit")),
]

# 예전 명령행 인터페이스에서 쓰던 옵션들 — 호환을 위해 남겨 둔다.
VALUE_ALIASES = {"--pieces": "n_pieces"}
BOOLEAN_ALIASES = {
    "--with-gravity": ("include_gravity", True),
    "--no-incompressibility": ("incompressible", False),
    "--water-at-bottom": ("water_at_top", False),
}


def resolve_parameters(params=None, **overrides):
    """``params`` 와 키워드 인자를 검사된 하나의 객체로 합친다.

    Turn ``params`` plus keyword overrides into one validated object.
    """
    if params is None:
        params = Parameters()
    elif not isinstance(params, Parameters):
        raise TypeError("params must be a Parameters instance")
    if overrides:
        return params.replace(**overrides)
    return params.validated()


def _option_name(field_name):
    return "--" + field_name.replace("_", "-")


def add_arguments(parser):
    """필드 목록에서 만들어, 물리량마다 명령행 옵션을 하나씩 붙인다.

    Give ``parser`` one option per parameter, built from the field list.
    """
    for group, names in FIELD_GROUPS:
        section = parser.add_argument_group(group)
        for name in names:
            field = FIELDS[name]
            kind = field.metadata["kind"]
            unit = field.metadata["unit"]
            help_text = field.metadata["help"]
            if unit:
                help_text += " [%s]" % unit
            option = _option_name(name)
            if kind == "bool":
                section.add_argument(option, dest=name, default=None,
                                     action="store_true", help=help_text)
                section.add_argument("--no-" + option[2:], dest=name,
                                     action="store_false",
                                     help="opposite of %s" % option)
            elif kind == "choice":
                section.add_argument(option, dest=name, default=None,
                                     choices=field.metadata["choices"],
                                     help=help_text)
            elif kind == "int":
                section.add_argument(option, dest=name, default=None,
                                     type=int, help=help_text)
            else:
                section.add_argument(option, dest=name, default=None,
                                     type=float, help=help_text)
    aliases = parser.add_argument_group("이전 옵션 / kept for compatibility")
    for option, name in VALUE_ALIASES.items():
        field = FIELDS[name]
        aliases.add_argument(option, dest=name, default=None,
                             type=int if field.metadata["kind"] == "int"
                             else float,
                             help="same as %s" % _option_name(name))
    for option, (name, value) in BOOLEAN_ALIASES.items():
        aliases.add_argument(option, dest=name, default=None,
                             action="store_const", const=value,
                             help="same as %s%s" % ("" if value else "--no-",
                                                    _option_name(name)[2:]))
    return parser


def parameters_from_namespace(namespace, base=None):
    """명령행이 실제로 지정한 물리량만 모은다.

    Collect the parameters a parsed command line actually set.
    """
    given = {name: getattr(namespace, name)
             for name in FIELDS
             if getattr(namespace, name, None) is not None}
    return resolve_parameters(base, **given)


def parse_value(field, answer):
    """대화형 입력의 답 하나를 해석한다.

    Interpret one answer of the interactive prompt.
    """
    kind = field.metadata["kind"]
    text = answer.strip()
    if kind == "bool":
        lowered = text.lower()
        if lowered in ("y", "yes", "true", "1", "네", "예"):
            return True
        if lowered in ("n", "no", "false", "0", "아니오", "아뇨"):
            return False
        raise ValueError("answer y or n")
    if kind == "choice":
        if text not in field.metadata["choices"]:
            raise ValueError("choose one of %s"
                             % ", ".join(field.metadata["choices"]))
        return text
    if kind == "int":
        return int(text)
    if kind == "optional_float":
        if text.lower() in ("none", "-", "없음"):
            return None
        return float(text)
    return float(text)


def prompt_parameters(base=None, ask=input, echo=print):
    """물리량을 하나씩 묻는다. 빈 칸이면 지금 값을 그대로 쓴다.

    Ask for every quantity in turn; an empty answer keeps the current one.

    ``ask`` and ``echo`` are injectable so the prompt can be driven by a test
    or by a notebook.  ``none`` (or ``-``) clears an optional value such as
    the drop height.
    """
    params = base if base is not None else Parameters()
    echo("물리량을 입력하세요. 빈 칸으로 두면 괄호 안의 현재 값을 씁니다.")
    echo("Enter the physical quantities; empty keeps the value in brackets.")
    values = {}
    for group, names in FIELD_GROUPS:
        echo("")
        echo("[%s]" % group)
        for name in names:
            field = FIELDS[name]
            current = values.get(name, getattr(params, name))
            if isinstance(current, bool):
                shown = "y" if current else "n"
            elif current is None:
                shown = "none"
            else:
                shown = "%g" % current if isinstance(current, float) \
                    else str(current)
            unit = " [%s]" % field.metadata["unit"] \
                if field.metadata["unit"] else ""
            question = "  %s%s (%s): " % (field.metadata["label"], unit, shown)
            while True:
                answer = ask(question)
                if not answer.strip():
                    break
                try:
                    values[name] = parse_value(field, answer)
                    break
                except ValueError as error:
                    echo("    입력 오류 / invalid input: %s" % error)
    return resolve_parameters(params, **values)


# ===========================================================================
# 이웃 찾기 — neighbour search on a cell grid
# ===========================================================================
# 이웃 찾기 — 셀 격자로 가까운 쌍만 골라낸다.
#
# 구슬 모델도 입자 모델도 "지금 서로 닿아 있는 것들"을 알아야 한다.  모든
# 쌍의 거리를 재면 개수의 제곱으로 비용이 늘어, 구슬 4000 개에서는 임시 배열이
# 200 MB 에 달해 큰 계산이 메모리 때문에 죽었다.
#
# 대신 한 변이 ``cutoff`` 인 셀에 요소를 나눠 담는다.  ``cutoff`` 보다 가까운
# 쌍은 같은 셀이나 맞닿은 셀 안에만 있을 수 있으므로, 훑어야 하는 후보가
# 요소 수에 비례한다.  구슬 4000 개에서 3.1 초가 0.017 초가 되고, 20000 개도
# 0.08 초에 끝난다.
#
# 셀 칸수가 ``int64`` 로 셀 수 없을 만큼 잘게 쪼개지는 경우에만 예전처럼 거리
# 행렬을 블록으로 잘라 계산한다.  어느 쪽이든 결과는 같다 — 시험이 두 경로를
# 맞대어 확인한다.

# 한 번에 메모리에 올리는 후보 쌍의 최대 개수. 쌍 하나가 int64 두 개와
# 중간 계산을 쓰므로, 4백만 쌍이면 수백 MB 가 아니라 수십 MB 로 끝난다.
PAIR_BUDGET = 4_000_000

# 자기 셀과 "절반"의 이웃 셀. 26 개 이웃을 모두 보면 셀 쌍이 두 번씩
# 나오므로, 사전순으로 앞쪽 절반만 본다. 그러면 모든 쌍이 정확히 한 번 나온다.
HALF_OFFSETS = np.array(
    [(0, 0, 0), (0, 0, 1), (0, 1, -1), (0, 1, 0), (0, 1, 1),
     (1, -1, -1), (1, -1, 0), (1, -1, 1), (1, 0, -1), (1, 0, 0),
     (1, 0, 1), (1, 1, -1), (1, 1, 0), (1, 1, 1)], dtype=np.int64)


def _cell_grid(positions, cutoff):
    """요소를 한 변이 ``cutoff`` 인 셀에 담고, 셀별 목록을 돌려준다.

    구슬을 한 변이 ``cutoff`` 인 셀에 담고, 셀별 목록을 돌려준다.

    돌려주는 것은 ``(고유 셀 좌표, 셀 시작 위치, 셀 안의 개수, 정렬 순서)``
    이고, ``순서[시작:시작+개수]`` 가 그 셀에 든 구슬의 번호다.  칸 수가
    ``int64`` 로 셀 수 없을 만큼 많아지면 ``None`` 을 돌려주어 호출한 쪽이
    거리 행렬로 되돌아가게 한다.
    """
    origin = positions.min(axis=0)
    cells = np.floor((positions - origin) / cutoff).astype(np.int64)
    dims = cells.max(axis=0) + 1
    if np.any(dims < 1) or float(dims[0]) * float(dims[1]) * float(dims[2]) \
            > 2.0 ** 62:
        return None

    keys = (cells[:, 0] * dims[1] + cells[:, 1]) * dims[2] + cells[:, 2]
    order = np.argsort(keys, kind="stable")
    unique, starts, counts = np.unique(keys[order], return_index=True,
                                       return_counts=True)
    coordinates = np.stack([unique // (dims[1] * dims[2]),
                            (unique // dims[2]) % dims[1],
                            unique % dims[2]], axis=1)
    return {"dims": dims, "keys": unique, "coordinates": coordinates,
            "starts": starts, "counts": counts, "order": order}


def _pairs_between(grid, cell_a, cell_b, budget=PAIR_BUDGET):
    """두 셀 목록 사이의 모든 구슬 쌍을, 메모리 예산만큼씩 나누어 낸다.

    셀마다 든 구슬 수가 다르므로 파이썬 루프로 풀면 셀 수만큼 반복하게
    된다.  대신 쌍의 개수를 누적해 두고 ``repeat`` 으로 펼치면, 길이가 서로
    다른 셀들도 한 번의 numpy 연산으로 모든 조합을 만들 수 있다.
    """
    counts_a = grid["counts"][cell_a].astype(np.int64)
    counts_b = grid["counts"][cell_b].astype(np.int64)
    totals = counts_a * counts_b
    keep = totals > 0
    cell_a, cell_b = cell_a[keep], cell_b[keep]
    counts_a, counts_b, totals = counts_a[keep], counts_b[keep], totals[keep]
    if totals.size == 0:
        return

    # 셀 쌍을 예산 안에 들어가는 덩어리로 자른다
    edges, carried, start = [0], 0, 0
    for index, total in enumerate(totals):
        if carried and carried + total > budget:
            edges.append(index)
            carried = 0
        carried += int(total)
    edges.append(totals.size)

    for begin, end in zip(edges[:-1], edges[1:]):
        block_a, block_b = cell_a[begin:end], cell_b[begin:end]
        block_total = totals[begin:end]
        rows = np.repeat(np.arange(block_total.size), block_total)
        offsets = np.concatenate(([0], np.cumsum(block_total)[:-1]))
        local = np.arange(int(block_total.sum())) - np.repeat(offsets,
                                                             block_total)
        width = np.repeat(counts_b[begin:end], block_total)
        first = grid["order"][np.repeat(grid["starts"][block_a], block_total)
                              + local // width]
        second = grid["order"][np.repeat(grid["starts"][block_b], block_total)
                               + local % width]
        del rows
        yield first, second


def _grid_neighbour_pairs(positions, cutoff, grid):
    """셀 격자로 찾은, ``cutoff`` 안에 있는 모든 쌍."""
    dims = grid["dims"]
    firsts, seconds = [], []
    cutoff_squared = cutoff ** 2
    every = np.arange(grid["keys"].size)

    for offset in HALF_OFFSETS:
        target = grid["coordinates"] + offset
        inside = np.all((target >= 0) & (target < dims), axis=1)
        if not inside.any():
            continue
        source = every[inside]
        target_keys = ((target[inside, 0] * dims[1] + target[inside, 1])
                       * dims[2] + target[inside, 2])
        found = np.searchsorted(grid["keys"], target_keys)
        found = np.minimum(found, grid["keys"].size - 1)
        matched = grid["keys"][found] == target_keys
        source, found = source[matched], found[matched]

        same_cell = not offset.any()
        for first, second in _pairs_between(grid, source, found):
            if same_cell:
                # 한 셀 안에서는 두 조합 (p, q) 와 (q, p) 가 모두 나오므로
                # 위쪽 삼각만 남긴다. 자기 자신과의 짝도 이때 걸러진다.
                keep = first < second
                low, high = first[keep], second[keep]
            else:
                # 서로 다른 셀이므로 번호가 겹치지 않는다. 셀 쌍이 한 번씩만
                # 나오도록 이웃을 절반만 보았으니, 순서만 맞추면 된다.
                low = np.minimum(first, second)
                high = np.maximum(first, second)
            if low.size == 0:
                continue
            delta = positions[low] - positions[high]
            close = np.einsum("ij,ij->i", delta, delta) < cutoff_squared
            firsts.append(low[close])
            seconds.append(high[close])

    if not firsts:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty
    return np.concatenate(firsts), np.concatenate(seconds)


def _dense_neighbour_pairs(positions, cutoff, chunk):
    """거리 행렬을 블록으로 잘라 쌍을 찾는다 (격자를 쓸 수 없을 때)."""
    n = positions.shape[0]
    firsts, seconds = [], []
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = positions[start:stop]
        delta = block[:, None, :] - positions[None, :, :]
        distance = np.einsum("ijk,ijk->ij", delta, delta)
        rows, columns = np.nonzero(distance < cutoff ** 2)
        rows = rows + start
        keep = rows < columns  # 쌍은 한 번만, 자기 자신과는 짝을 짓지 않는다
        firsts.append(rows[keep])
        seconds.append(columns[keep])
    return np.concatenate(firsts), np.concatenate(seconds)


def neighbour_pairs(positions, cutoff, chunk=None):
    """``cutoff`` 보다 가까운 구슬 번호의 쌍 ``(i, j)``, 항상 ``i < j``.

    한 변이 ``cutoff`` 인 셀 격자에 구슬을 나눠 담는다.  가까운 쌍은 같은
    셀이나 맞닿은 셀 안에만 있을 수 있으므로, 비용과 메모리가 구슬 수에
    비례한다.  예전 방식대로 거리 행렬을 만들면 구슬 4000 개에서 임시 배열이
    200 MB 에 달해 큰 계산이 메모리 때문에 죽었다.

    ``chunk`` 는 격자를 쓸 수 없을 때의 대비책에서 한 번에 다루는 행 수다.
    비워 두면 메모리 예산에서 정한다.
    """
    positions = np.asarray(positions, dtype=float)
    n = positions.shape[0]
    empty = np.empty(0, dtype=np.intp)
    if n < 2:
        return empty, empty
    if not cutoff > 0:
        return empty, empty

    grid = _cell_grid(positions, cutoff)
    if grid is not None:
        return _grid_neighbour_pairs(positions, cutoff, grid)
    return _dense_neighbour_pairs(positions, cutoff,
                                  chunk or max(1, PAIR_BUDGET // max(n, 1)))


# ===========================================================================
# 원판 모델 — disc model (water slides along the axis)
# ===========================================================================
# 원판 모델 — 물이 병 축을 따라 미끄러진다.
#
# 물을 같은 질량의 얇은 원판 n 개로 나누고, 각 원판이 병 축 위에서 어디에
# 있는지만 따라간다. 병과 함께 도는 좌표계에서 원판은 원심력·겉보기 중력·
# 벽 항력을 받고, 그 방정식은 계수를 얼려 두면 정확히 풀린다.
#
# 자유비행 중에는 각운동량이 보존되므로 ``omega`` 는 ``L = J omega`` 에서
# 나온다. 물이 퍼지면서 ``J`` 가 커지고 회전이 느려지는 것이 물병던지기의
# 원리다.
#
# 가장 빠르지만, 단면 안에서 물이 움직이지 못하므로 감속을 과대평가한다.
#
# Kinetics of a partially filled bottle spinning in free flight.
#
# Model
# -----
# The bottle is a rigid thin-walled cylinder rotating with angular velocity
# ``omega`` about a transverse axis through the centre of mass of the
# bottle + water system.  The water is decomposed into ``n`` thin coaxial
# slices of equal mass which are free to slide along the bottle axis; ``r``
# denotes the axial coordinate of a slice measured from the bottom of the
# bottle.
#
# In the non-inertial frame co-rotating with the bottle, a slice obeys
#
#     r'' = omega**2 * (r - r_cm) + a_axial - (2 / tau) * r'                (1)
#
# where
#
# * ``omega**2 * (r - r_cm)`` is the centrifugal (destabilising) term,
# * ``a_axial = gravity * cos(theta)`` is the axial projection of the
#   apparent gravity.  During free flight the co-moving frame falls with the
#   bottle, gravity cancels, and ``gravity`` must be passed as ``0``.  A
#   non-zero value is only meaningful while the bottle is still being
#   accelerated by the hand,
# * ``-(2 / tau) * r'`` is a linear (viscous) drag against the bottle wall,
#   with damping time ``tau``.  Because the drag is written per unit mass,
#   ``tau`` does not depend on how finely the water is discretised.
#
# Angular momentum is conserved in free flight, so ``omega`` follows from
# ``L = J * omega`` with ``J`` recomputed from the instantaneous water
# distribution; the redistribution of water raises ``J`` and therefore slows
# the rotation down, which is the mechanism behind the bottle flip.
#
# References
# ----------
# * P. J. Dekker et al., "Water bottle flipping physics",
#   Am. J. Phys. 86, 733 (2018), doi:10.1119/1.5052441.
# * "The water bottle flipping experiment: a quantitative comparison between
#   experiments and numerical simulations", Eur. J. Phys. 45, 065003 (2024),
#   doi:10.1088/1361-6404/ad6e43 (arXiv:2407.20627).

def find_center_of_mass(positions, eps, bottle_center=None,
                        length_bottle=BOTTLE_HEIGHT):
    """병과 물을 합친 계의 질량중심이 병 축에서 어디인가.

    Axial position of the centre of mass of the bottle + water system.

    ``eps`` is the mass fraction of the empty bottle,
    ``eps = m_bottle / (m_bottle + m_water)``.  ``bottle_center`` is where the
    empty bottle's own centre of mass sits; a real bottle with a narrow neck
    carries its mass low, so it is *not* mid-height (that is the default only
    for a plain cylinder).  All slices carry the same mass, so the water
    contributes the mean slice position.
    """
    if bottle_center is None:
        bottle_center = length_bottle / 2
    return eps * bottle_center + (1 - eps) * np.mean(positions)


def rotational_inertia_water(positions, center_of_mass, water_mass,
                             radius=BOTTLE_RADIUS):
    """질량중심을 지나는 횡축에 대한 물의 관성모멘트.

    Moment of inertia of the water about the transverse axis at ``center_of_mass``.

    Each slice is a solid disc: its own moment of inertia about a diameter is
    ``m * R**2 / 4``, to which the parallel axis theorem adds
    ``m * (r - r_cm)**2``.  ``radius`` may be a single number (a cylinder) or
    one radius per slice — in a real bottle a slice sitting in the neck is
    much narrower than one in the body, and using the body radius everywhere
    overestimates J for the water that has climbed into the neck.
    """
    positions = np.asarray(positions, dtype=float)
    radius = np.asarray(radius, dtype=float)
    slice_mass = water_mass / positions.size
    return slice_mass * float(np.sum(radius ** 2 / 4
                                     + (positions - center_of_mass) ** 2))


def rotational_inertia_bottle(center_of_mass, mass_bottle=BOTTLE_MASS,
                              radius=BOTTLE_RADIUS, length=BOTTLE_HEIGHT,
                              neck=None, shoulder_start=SHOULDER_START,
                              shoulder_end=SHOULDER_END, y_axis=0.0):
    """같은 횡축에 대한 빈 병의 관성모멘트.

    Moment of inertia of the empty bottle about the transverse axis at ``center_of_mass``.

    ``neck`` 를 주면 몸통-어깨-목 모양의 껍질로, 주지 않으면 예전처럼 균질한
    원통 껍질로 계산한다.
    """
    if neck is None:
        j_own = mass_bottle * (radius ** 2 / 2 + length ** 2 / 12)
        return j_own + mass_bottle * ((length / 2 - center_of_mass) ** 2
                                      + y_axis ** 2)
    z_cm, ring, second = _shell_moments_cached(radius, neck, length,
                                               shoulder_start, shoulder_end)
    return mass_bottle * (ring + second - 2 * center_of_mass * z_cm
                          + center_of_mass ** 2 + y_axis ** 2)


def update_slice_positions(positions, velocities, center_of_mass,
                           angular_velocity, angle, tau, dt, gravity=0.0):
    """식 (1) 의 정확한 해로 원판들을 ``dt`` 만큼 진행시킨다.

    Advance the slices over ``dt`` with the exact solution of Eq. (1).

    ``omega``, ``theta`` and ``r_cm`` are held constant over the step, which
    makes Eq. (1) a linear ODE with constant coefficients.  Writing
    ``u = r - r_eq`` with the equilibrium position

        r_eq = r_cm - a_axial / omega**2

    its solution is

        u(t)  = exp(-t / tau) * [u0 * cosh(W t)
                                 + (v0 + u0 / tau) / W * sinh(W t)]
        u'(t) = exp(-t / tau) * [v0 * cosh(W t)
                                 + (omega**2 * u0 - v0 / tau) / W * sinh(W t)]

    with the growth rate ``W = sqrt(omega**2 + 1 / tau**2)``.  Using ``omega``
    instead of ``W`` (as an undamped solution would) overestimates the
    damping and is only valid for ``omega * tau >> 1``.  ``W >= 1 / tau > 0``,
    so the division by ``W`` is always safe.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    if tau <= 0:
        raise ValueError("tau must be positive")

    if gravity:
        if angular_velocity == 0:
            raise ValueError(
                "the equilibrium position r_cm + g*cos(theta)/omega**2 "
                "diverges for omega = 0")
        # ``angle`` 은 병 축이 연직 위에서 벗어난 기울기다: 0 이면 똑바로 서
        # 있고, 그때 중력의 축 방향 성분은 -g (물을 바닥으로 당긴다).
        # 따라서 a_axial = -g cos(theta) 이고 r_eq = r_cm - a_axial/omega**2.
        equilibrium = center_of_mass + gravity * np.cos(angle) / angular_velocity ** 2
    else:
        # 자유비행: 함께 떨어지는 좌표계에서는 중력이 상쇄되어 사라진다.
        equilibrium = center_of_mass

    growth_rate = np.sqrt(angular_velocity ** 2 + 1 / tau ** 2)
    u_0 = positions - equilibrium
    v_0 = velocities

    decay = np.exp(-dt / tau)
    cosh = np.cosh(growth_rate * dt)
    sinh = np.sinh(growth_rate * dt)

    new_positions = equilibrium + decay * (
        u_0 * cosh + (v_0 + u_0 / tau) / growth_rate * sinh)
    new_velocities = decay * (
        v_0 * cosh
        + (angular_velocity ** 2 * u_0 - v_0 / tau) / growth_rate * sinh)
    return new_positions, new_velocities


def check_boundary_conditions(positions, velocities, restitution, l_min, l_max):
    """원판이 병의 바닥과 뚜껑에서 튀는 것을 처리한다.

    Rebound of the slices on the bottom and the cap of the bottle.

    The slice is mirrored about the wall it crossed, and the rebound
    distance is scaled by ``|restitution|`` so that position and velocity
    lose energy consistently.  (Freezing the slice at its previous position
    instead makes slices stick to the walls.)

    Only the magnitude of ``restitution`` is physical: the reflection reverses
    the velocity here, whatever sign it was given.  The old code multiplied
    the velocity by ``restitution`` as it stood, so a positive value — the
    way anybody would write a restitution coefficient — silently let the water
    keep driving into the wall instead of bouncing off it.

    ``positions`` and ``velocities`` are modified in place and returned.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    bounce = min(abs(restitution), 1.0)

    below = positions < l_min
    positions[below] = l_min + bounce * (l_min - positions[below])
    velocities[below] *= -bounce

    above = positions > l_max
    positions[above] = l_max - bounce * (positions[above] - l_max)
    velocities[above] *= -bounce

    # 한 스텝에 병 길이보다 멀리 움직인 원판은, 한 번 반사해도 반대쪽 벽을
    # 넘어갈 수 있다.
    np.clip(positions, l_min, l_max, out=positions)
    return positions, velocities


def enforce_incompressibility(positions, velocities, slice_volume,
                              s_min, s_max, to_volume, to_height):
    """물 원판들이 서로 겹치거나 지나치지 못하게 한다.

    Keep the water slices from overlapping or crossing each other.

    물은 압축되지 않으므로 원판 두 장이 같은 부피를 차지할 수 없다.  원통이면
    "원판 두께만큼 떨어져 있어야 한다" 로 끝나지만, 실제 병처럼 단면이
    변하면 원판의 두께가 있는 자리마다 다르다 — 목에서는 얇고 몸통에서는
    두껍다.

    그래서 제약을 높이 ``z`` 가 아니라 **부피 좌표** ``s = S(z)`` (바닥부터
    그 높이까지의 병 부피) 에서 건다.  같은 질량의 원판은 어디에 있든 같은
    부피 ``slice_volume`` 을 차지하므로, 부피 좌표에서는 간격이 일정한 문제가
    되어 원통일 때와 똑같은 투영으로 풀린다.  이것이 좁은 목을 지나는 물을
    제대로 다루는 방법이다: 목에서는 같은 부피가 훨씬 긴 길이를 차지한다.

    Slices that end up in contact are treated as a perfectly inelastic contact
    and share the mean velocity of their contact group, which conserves the
    momentum of that group.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    n = positions.size
    if n < 2:
        return positions, velocities
    if n * slice_volume > (s_max - s_min) + slice_volume * (1 + 1e-9):
        raise ValueError("the water does not fit inside the bottle")

    s = np.asarray(to_volume(positions), dtype=float)
    order = np.argsort(s, kind="stable")
    p = s[order]
    v = velocities[order]

    # 아래에서 위로 밀어 올리는 투영. p[i] = max(p[i], p[i-1] + dV) 를 하나씩
    # 돌리는 것과 같은데, 계단을 빼고 누적 최대를 취하면 원판 수와 무관하게
    # 한 번의 numpy 연산으로 끝난다.
    ladder = np.arange(n) * slice_volume
    p = np.maximum(p, s_min)
    p = np.maximum.accumulate(p - ladder) + ladder
    # 위에서 아래로 되밀기: 뚜껑을 뚫고 나간 기둥을 다시 내려보낸다.
    p[n - 1] = min(p[n - 1], s_max)
    p = (np.minimum.accumulate(p[::-1] + ladder) - ladder)[::-1]
    np.clip(p, s_min, s_max, out=p)

    # 완전 비탄성 접촉: 맞닿은 원판들은 함께 움직인다. 접촉 그룹마다 평균
    # 속도를 주면 그 그룹의 운동량이 보존된다.
    in_contact = np.diff(p) <= slice_volume * (1 + 1e-9)
    group = np.concatenate(([0], np.cumsum(~in_contact)))
    starts = np.flatnonzero(np.concatenate(([True], np.diff(group) != 0)))
    sums = np.add.reduceat(v, starts)
    sizes = np.diff(np.concatenate((starts, [n])))
    v = np.repeat(sums / sizes, sizes)

    positions[order] = np.asarray(to_height(p), dtype=float)
    velocities[order] = v
    return positions, velocities


def flight_time(drop_height, gravity=G):
    """``drop_height`` 에서 가만히 놓았을 때의 자유낙하 시간 [s].

    Free-fall time from ``drop_height`` (bottle released at rest), unit: s.
    """
    if drop_height < 0:
        raise ValueError("drop_height must be non-negative")
    return np.sqrt(2 * drop_height / gravity)


# ===========================================================================
# 입자 모델 — parcel model (cross section resolved)
# ===========================================================================
# 입자 모델 — 단면까지 분해한 물 조각.
#
# 원판 모델은 단면 하나에 자유도가 하나뿐이어서, 물이 측벽을 타고 오를 수
# 없다. 실제로는 원심력이 물을 "닿을 수 있는 가장 먼 곳" 으로 밀어붙이고,
# 앞서가는 물은 벽에서 떨어져 공중을 지나 반대쪽 벽에 부딪친다.
#
# 그래서 여기서는 원판을 ``n_pieces`` 개의 조각으로 쪼개어 회전면 안에서
# 따로 움직이게 한다. 코리올리 항과 오일러 항이 살아나고, 물의 부피는 조각이
# 서로 겹치지 못하게 하는 위치 투영으로 지킨다.
#
# Parcel (finite-piece) discretisation of the water in a flipping bottle.
#
# Why go beyond discs
# -------------------
# The disc model above treats the water as thin rigid discs that can only slide
# along the bottle axis.  Each disc is a single degree of freedom, so the
# water inside a cross section cannot move: the radial mass distribution is
# frozen at the uniform value ``<y**2> = R**2 / 4`` and the water can never
# climb the side wall.  Experiments and particle simulations of the flip show
# the opposite: the centrifugal acceleration "propels liquid parcels to the
# farthest volumes available", and the advancing fraction "detaches from the
# wall and moves freely through the air until it sloshes against the opposite
# side of the wall" (arXiv:2502.12946).
#
# This module cuts every disc into ``n_pieces`` small parcels and follows each
# of them individually, so the water can redistribute across the cross section
# as well as along the axis.
#
# Frame and equations of motion
# -----------------------------
# Bottle-fixed axes: ``z`` along the bottle (from the bottom), ``y`` in the
# flip plane, ``x`` perpendicular to it.  In free flight the bottle rotates
# about the transverse axis ``x`` through the system centre of mass
# ``(y_cm, z_cm)`` with angular velocity ``omega``.
#
# Because ``omega`` is directed along ``x``, none of the fictitious forces has
# a component along ``x``: a parcel released with ``vx = 0`` keeps its ``x``
# coordinate, and the dynamics are two dimensional in the flip plane.  With
# ``Y = y - y_cm`` and ``Z = z - z_cm``,
#
#     ay = omega**2 * Y + 2 * omega * vz + omega_dot * Z - (2 / tau) * vy + gy
#     az = omega**2 * Z - 2 * omega * vy - omega_dot * Y - (2 / tau) * vz + gz
#
# * ``omega**2 * (Y, Z)`` — centrifugal acceleration, away from the rotation
#   axis.  This is the term the disc model keeps.
# * ``2 * omega * (vz, -vy)`` — Coriolis acceleration.  It vanishes in the disc
#   model only because the water is pinned to the axis; as soon as a parcel
#   moves it deflects it sideways, and it is what throws the advancing water
#   across the bottle instead of straight along the wall.
# * ``omega_dot * (Z, -Y)`` — Euler (azimuthal) acceleration.  It is not small
#   here: ``omega`` drops by a factor of five during the flight.
# * ``-(2 / tau) * v`` — the same linear wall drag as in the disc model.
# * ``(gy, gz)`` — apparent gravity, zero in free flight.
#
# Over one time step ``omega``, ``omega_dot`` and the centre of mass are held
# constant, which makes the system linear with constant coefficients; the step
# is taken with the matrix exponential of the augmented system, i.e. the exact
# solution of the frozen-coefficient problem.  This is the two-dimensional
# generalisation of the closed-form step used for the discs.
#
# Incompressibility
# -----------------
# The parcels are seeded on a hexagonal close packing that fills exactly the
# volume of the water, and the pitch of that packing is their exclusion
# distance: overlaps are removed at every step by a position projection with an
# inelastic contact response, which is what keeps the volume of the water
# constant.  This is the position-based-dynamics treatment of an incompressible
# fluid (Macklin & Müller, "Position based fluids", ACM TOG 32, 104 (2013))
# reduced to its non-overlap constraint; it is not a pressure-projected
# Navier-Stokes solver, but it does give the water a finite volume, which is
# what controls the moment of inertia.
#
# References
# ----------
# * P. J. Dekker et al., Am. J. Phys. 86, 733 (2018), doi:10.1119/1.5052441.
# * Eur. J. Phys. 45, 065003 (2024), doi:10.1088/1361-6404/ad6e43.
# * "Fluid Motion Makes the Bottle-Flip Challenge Mechanically Unintuitive But
#   Viable", arXiv:2502.12946.
# * M. Macklin and M. Müller, "Position based fluids", ACM Trans. Graph. 32,
#   104 (2013), doi:10.1145/2461912.2461984.

def _hexagonal_layer(spacing, max_radius, offset=(0.0, 0.0)):
    """원 안에 놓인, 간격 ``spacing`` 인 육각격자의 점들.

    Points of a hexagonal lattice of pitch ``spacing`` inside a disc.
    """
    reach = int(np.ceil(2 * max_radius / spacing)) + 2
    i, j = np.meshgrid(np.arange(-reach, reach + 1),
                       np.arange(-reach, reach + 1), indexing="ij")
    x = (i + 0.5 * (j % 2)) * spacing + offset[0]
    y = j * (np.sqrt(3) / 2) * spacing + offset[1]
    x, y = x.ravel(), y.ravel()
    inside = x ** 2 + y ** 2 <= max_radius ** 2
    return x[inside], y[inside]


def lattice_spacing(n_pieces, radius=BOTTLE_RADIUS):
    """단면을 ``n_pieces`` 조각으로 나누는 충전 격자의 간격.

    Pitch of the packing that cuts a cross section into ``n_pieces`` pieces.

    A hexagonal lattice of pitch ``d`` carries ``2 / (sqrt(3) * d**2)`` points
    per unit area, and the centres have to stay a distance ``d / 2`` away from
    the wall, so ``n_pieces`` pieces per cross section means

        n_pieces = 2 * pi * (R - d/2)**2 / (sqrt(3) * d**2)

    which solves to ``d = sqrt(2 pi) R / (sqrt(sqrt(3) n_pieces) + sqrt(2 pi)/2)``.
    Ignoring the wall exclusion here would put roughly a third fewer pieces in
    a coarsely resolved cross section than asked for.
    """
    if n_pieces < 1:
        raise ValueError("n_pieces must be >= 1")
    root_two_pi = np.sqrt(2 * np.pi)
    return root_two_pi * radius / (np.sqrt(np.sqrt(3) * n_pieces)
                                   + root_two_pi / 2)


def _fill_column(n_pieces, water_volume, water_at_top, radius, height,
                 shape=None):
    """조각을 층으로 쌓아 물을 채워 본다.

    층을 하나씩 쌓되, 채운 구간이 담는 **병의 부피** 가 물의 부피에 이를 때까지
    쌓는다.  단면이 일정한 원통이면 예전처럼 "물기둥 높이 / 층 간격" 과 같지만,
    목이 있는 병에서는 그렇지 않다: 목은 부피가 거의 없으므로 물이 뚜껑 쪽에
    있어도 대부분은 어깨 아래 몸통에 있어야 한다.  각 층에 놓을 수 있는 조각은
    그 높이의 반지름 ``R(z)`` 이 정한다.

    돌려주는 것은 ``(x, y, z, 조각 반지름, 격자 간격, 층 수)`` 이고, 조각이
    두 개도 놓이지 않으면 ``None`` 이다.
    """
    spacing = lattice_spacing(n_pieces, radius)
    a = spacing / 2
    if a >= radius:
        return None

    def wall_at(z):
        if shape is None:
            return float(radius)
        return float(radius_profile(z, **shape))

    def volume_to(z):
        if shape is None:
            return float(np.pi * radius ** 2 * z)
        return float(cumulative_volume(z, **shape))

    layer_gap = spacing * np.sqrt(2 / 3)
    # ABAB 쌓기의 층별 어긋남: 둘째 층이 첫째 층의 오목한 곳에 앉는다.
    offsets = [(0.0, 0.0), (spacing / 2, spacing * np.sqrt(3) / 6)]
    start = volume_to(height) if water_at_top else 0.0

    xs, ys, zs = [], [], []
    layer = 0
    filled = 0.0
    max_layers = int(np.ceil(height / layer_gap)) + 2
    while filled < water_volume and layer < max_layers:
        if water_at_top:
            z = height - a - layer * layer_gap
        else:
            z = a + layer * layer_gap
        if z < a or z > height - a:
            break
        available = wall_at(z) - a
        if available > 0:
            lx, ly = _hexagonal_layer(spacing, available, offsets[layer % 2])
            if lx.size:
                xs.append(lx)
                ys.append(ly)
                zs.append(np.full(lx.size, z))
        # 이 층까지 덮은 구간이 담는 병 부피
        edge = z - layer_gap / 2 if water_at_top else z + layer_gap / 2
        edge = min(max(edge, 0.0), height)
        filled = abs(volume_to(edge) - start)
        layer += 1

    if not xs:
        return None
    x = np.concatenate(xs)
    if x.size < 2:
        return None
    return (x, np.concatenate(ys), np.concatenate(zs), a, spacing, len(xs))


def seed_parcels(water_mass, n_pieces, water_at_top=True,
                 radius=BOTTLE_RADIUS, height=BOTTLE_HEIGHT,
                 density=WATER_DENSITY, shape=None):
    """물기둥을 조각으로 채운다 — 단면마다 ``n_pieces`` 개.

    조각은 ``n_pieces`` 에서 따라오는 간격 ``d`` 의 육각 최밀충전에 놓인다.
    층은 ``d * sqrt(2/3)`` 씩 떨어져 어긋나게 쌓이므로, 이웃한 조각은 정확히
    배제 거리 ``d`` 에 앉는다.  물이 기대어 있는 뚜껑에서 시작해 높이
    ``V / (pi R**2)`` 의 기둥이 찰 때까지 층을 쌓고, 그것이 조각 수를 정한다.

    충전이 물의 부피를 정확히 덮으므로, 그 부피를 비행 중에 지켜 주는 것은
    :func:`resolve_overlaps` 의 겹침 금지 제약이다.  ``d / 2`` 는 조각의
    반지름 노릇을 한다 — 배제 반지름이지, 유체 요소의 모양에 대한 주장이
    아니다.

    좁은 병이나 ``n_pieces = 1`` 처럼 조각이 두 개도 놓이지 않는 입력은
    예전에는 오류였다.  이제는 놓일 때까지 단면을 더 잘게 쪼개고
    :class:`RuntimeWarning` 으로 알린다.

    Fill the water column with parcels, ``n_pieces`` per cross section.
    """
    water_volume = water_mass / density
    bottle_volume = float(cumulative_volume(height, **shape)) if shape \
        else float(np.pi * radius ** 2 * height)
    if water_volume > bottle_volume:
        raise ValueError(
            "물 %.4f kg (%.1f mL) 은 이 병(%.1f mL)에 들어가지 않습니다 / the "
            "water does not fit inside the bottle"
            % (water_mass, 1e6 * water_volume, 1e6 * bottle_volume))
    column_height = float(height_at_volume(water_volume, **shape)) if shape \
        else water_volume / (np.pi * radius ** 2)

    asked_pieces = n_pieces
    packed = _fill_column(n_pieces, water_volume, water_at_top, radius,
                          height, shape)
    while packed is None and n_pieces < 4096:
        n_pieces = max(4, 4 * n_pieces)
        packed = _fill_column(n_pieces, water_volume, water_at_top, radius,
                              height, shape)
    if packed is None:
        raise ValueError(
            "이 병에서는 조각을 두 개도 놓을 수 없습니다 / too few parcels: "
            "increase n_pieces")

    x, y, z, a, spacing, n_layers = packed
    n = x.size
    if n_pieces != asked_pieces:
        warnings.warn(
            "단면당 조각 수를 %d 에서 %d 로 늘렸습니다: 그대로는 이 병에 조각을 "
            "두 개도 놓을 수 없습니다 / pieces per cross section raised from "
            "%d to %d so that the water can be discretised at all"
            % (asked_pieces, n_pieces, asked_pieces, n_pieces),
            RuntimeWarning, stacklevel=2)

    return {"x": x, "y": y, "z": z, "velocities": np.zeros((n, 2)),
            "parcel_radius": a, "spacing": spacing,
            "parcel_mass": water_mass / n, "column_height": column_height,
            "n_parcels": n, "n_layers": n_layers,
            "n_pieces": n_pieces, "n_pieces_asked": asked_pieces,
            "pieces_per_layer": n / n_layers}


def parcel_center_of_mass(y, z, eps, length_bottle=BOTTLE_HEIGHT,
                          bottle_center=None):
    """병과 물을 합친 계의 질량중심 ``(y_cm, z_cm)``.

    Centre of mass ``(y_cm, z_cm)`` of the bottle + water system.

    The empty bottle sits on the axis, so it only pulls the transverse
    coordinate back towards ``y = 0``.  Its own centre of mass is
    ``bottle_center`` — mid-height only for a plain cylinder.
    """
    if bottle_center is None:
        bottle_center = length_bottle / 2
    return ((1 - eps) * float(np.mean(y)),
            eps * bottle_center + (1 - eps) * float(np.mean(z)))


def parcel_relative_angular_momentum(y, z, velocities, y_cm, z_cm, mass):
    """물이 병 안에서 움직여서 갖는 각운동량 (조각 모델).

    Angular momentum the parcels carry by moving within the bottle.
    """
    n = np.asarray(y).size
    return mass / n * float(np.sum((y - y_cm) * velocities[:, 1]
                                   - (z - z_cm) * velocities[:, 0]))


def parcel_rotational_inertia_water(y, z, y_cm, z_cm, water_mass):
    """질량중심을 지나는 횡축에 대한 조각들의 관성모멘트.

    Moment of inertia of the parcels about the transverse axis at the centre of mass.

    Rotation is about ``x``, so only the in-plane offsets ``(y, z)`` count;
    the parcels sample the water volume, so no ``R**2 / 4`` correction is
    needed here — unlike the disc model, the transverse spread of the water
    is resolved.
    """
    n = np.asarray(y).size
    return water_mass / n * np.sum((y - y_cm) ** 2 + (z - z_cm) ** 2)


def parcel_rotational_inertia_bottle(y_cm, z_cm, mass_bottle=BOTTLE_MASS,
                              radius=BOTTLE_RADIUS, length=BOTTLE_HEIGHT):
    """회전축이 병 축에서 벗어난 경우의 빈 병 관성모멘트.

    Moment of inertia of the empty bottle, with the axis off the bottle axis.
    """
    j_own = mass_bottle * (radius ** 2 / 2 + length ** 2 / 12)
    return j_own + mass_bottle * (y_cm ** 2 + (length / 2 - z_cm) ** 2)


def matrix_exponential(matrix, order=18):
    """크기를 줄여 제곱을 되풀이하는 테일러 급수로 구한 ``exp(matrix)``.

    ``exp(matrix)`` by scaling and squaring with a Taylor series.
    """
    matrix = np.asarray(matrix, dtype=float)
    norm = np.max(np.sum(np.abs(matrix), axis=1))
    squarings = 0 if norm == 0 else max(0, int(np.ceil(np.log2(norm))) + 1)
    scaled = matrix / 2.0 ** squarings
    result = np.eye(matrix.shape[0])
    term = np.eye(matrix.shape[0])
    for k in range(1, order + 1):
        term = term @ scaled / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


def step_matrix(omega, omega_dot, tau, y_cm, z_cm, dt, gravity=0.0, angle=0.0,
                coriolis=True, euler=True):
    """계수를 얼려 둔 운동방정식을 ``dt`` 동안 정확히 옮기는 전파행렬.

    Exact propagator of the frozen-coefficient equations over ``dt``.

    Returns the 5x5 matrix acting on the augmented state
    ``(y, z, vy, vz, 1)``; the last column carries the constant part of the
    acceleration, which avoids inverting a possibly singular system matrix.
    """
    if tau <= 0:
        raise ValueError("tau must be positive")
    coriolis_term = 2 * omega if coriolis else 0.0
    euler_term = omega_dot if euler else 0.0
    # 자유비행에서 중력은 0 이다. ``angle`` 은 연직 위에서 벗어난 기울기이므로
    # 병 좌표계에서 중력가속도는 -g (0, sin theta, cos theta) 이다.
    g_y = -gravity * np.sin(angle)
    g_z = -gravity * np.cos(angle)

    generator = np.zeros((5, 5))
    generator[0, 2] = 1.0
    generator[1, 3] = 1.0
    generator[2, 0] = omega ** 2
    generator[2, 1] = euler_term
    generator[2, 2] = -2 / tau
    generator[2, 3] = coriolis_term
    generator[2, 4] = -omega ** 2 * y_cm - euler_term * z_cm + g_y
    generator[3, 0] = -euler_term
    generator[3, 1] = omega ** 2
    generator[3, 2] = -coriolis_term
    generator[3, 3] = -2 / tau
    generator[3, 4] = -omega ** 2 * z_cm + euler_term * y_cm + g_z
    return matrix_exponential(generator * dt)


def advance_parcels(y, z, velocities, propagator):
    """:func:`step_matrix` 의 전파행렬을 모든 조각에 적용한다 (제자리).

    Apply a propagator from :func:`step_matrix` to every parcel in place.
    """
    state = np.column_stack([y, z, velocities[:, 0], velocities[:, 1],
                             np.ones(np.asarray(y).size)])
    new_state = state @ propagator.T
    y[:] = new_state[:, 0]
    z[:] = new_state[:, 1]
    velocities[:, 0] = new_state[:, 2]
    velocities[:, 1] = new_state[:, 3]
    return y, z, velocities


def apply_walls(x, y, z, velocities, restitution, parcel_radius_,
                radius=BOTTLE_RADIUS, height=BOTTLE_HEIGHT, shape=None):
    """조각을 측벽과 두 뚜껑에서 튕겨 낸다 (제자리 수정).

    Rebound the parcels off the side wall and the two caps, in place.

    ``x`` never changes, so the side wall limits ``|y|`` to
    ``sqrt((R - a)**2 - x**2)``.  The rebound mirrors the parcel and scales the
    excursion by ``|restitution|``, as in the disc model.
    """
    if shape is None:
        wall = np.full_like(z, float(radius))
    else:
        # 실제 병에서는 벽까지의 거리가 높이마다 다르다: 어깨 위로 올라간
        # 조각은 훨씬 좁은 곳에 있다.
        wall = radius_profile(z, **shape)
    limit = np.sqrt(np.maximum((wall - parcel_radius_) ** 2 - x ** 2, 0.0))
    bounce = min(abs(restitution), 1.0)

    above = y > limit
    y[above] = limit[above] - bounce * (y[above] - limit[above])
    below = y < -limit
    y[below] = -limit[below] + bounce * (-limit[below] - y[below])
    velocities[above | below, 0] *= -bounce
    np.clip(y, -limit, limit, out=y)

    z_min, z_max = parcel_radius_, height - parcel_radius_
    low = z < z_min
    z[low] = z_min + bounce * (z_min - z[low])
    high = z > z_max
    z[high] = z_max - bounce * (z[high] - z_max)
    velocities[low | high, 1] *= -bounce
    np.clip(z, z_min, z_max, out=z)

    # 어깨보다 굵은 자리에 있던 조각이 좁은 곳으로 올라오면 아예 들어갈 수
    # 없다. 그런 조각은 들어갈 수 있는 높이까지 도로 내려보낸다 — 그러지
    # 않으면 병 축에 눌러 붙어 물이 목을 그냥 통과해 버린다.
    if shape is not None:
        need = np.sqrt(x ** 2 + y ** 2) + parcel_radius_
        stuck = need > wall
        if np.any(stuck):
            idx = np.flatnonzero(stuck)
            for _ in range(20):
                if idx.size == 0:
                    break
                z[idx] -= 0.25 * parcel_radius_
                np.clip(z, z_min, z_max, out=z)
                wall_now = radius_profile(z[idx], **shape)
                keep = np.sqrt(x[idx] ** 2 + y[idx] ** 2) + parcel_radius_ \
                    > wall_now
                velocities[idx[keep], 1] = np.minimum(
                    velocities[idx[keep], 1], 0.0)
                idx = idx[keep]
    return y, z, velocities


def resolve_overlaps(x, y, z, velocities, parcel_radius_, iterations=1,
                     relaxation=0.8):
    """겹친 조각들을 회전면 안에서 밀어 떼어놓는다 (제자리 수정).

    조각은 ``x`` 를 그대로 지키므로, 회전축 방향으로 ``dx`` 만큼 떨어진 두
    조각은 ``sqrt(dy**2 + dz**2) >= sqrt((2a)**2 - dx**2)`` 만 지키면 된다.
    이는 3차원 거리가 ``2a`` 보다 멀다는 것과 같은 조건이라, 후보 쌍을
    :func:`neighbour_pairs` 로 한 번 찾아두면 반복마다 그 쌍만 보면 된다.
    예전에는 반복마다 N x N 거리 행렬을 만들어, 조각이 수천 개가 되면
    수백 MB 를 쓰고 느려졌다.

    떼어내는 이동량과 비탄성 접촉 반응을 쌍마다 크기가 같고 방향이 반대로
    나누어 주므로, 이 투영은 물의 질량중심과 총 운동량을 건드리지 않는다.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2 or iterations < 1:
        return 0.0

    diameter = 2 * parcel_radius_
    # 반복 중에 조각이 움직이므로 여유(skin)를 두고 후보를 한 번만 찾는다
    skin = 0.5 * diameter
    first, second = neighbour_pairs(np.column_stack([x, y, z]),
                                    diameter + skin)
    if first.size == 0:
        return 0.0

    dx = x[first] - x[second]
    min_plane = np.sqrt(np.maximum(diameter ** 2 - dx ** 2, 0.0))
    total_shift = 0.0

    for _ in range(iterations):
        dy = y[first] - y[second]
        dz = z[first] - z[second]
        distance = np.sqrt(dy ** 2 + dz ** 2)
        overlap = min_plane - distance
        active = overlap > 0
        if not active.any():
            break

        hit_first, hit_second = first[active], second[active]
        weight = overlap[active]
        near = distance[active]
        # 정확히 겹쳐 앉은 조각들은 방향이 정해지지 않으므로 z 로 떼어낸다
        safe = np.where(near > 1e-15, near, 1.0)
        normal_y = np.where(near > 1e-15, dy[active] / safe, 0.0)
        normal_z = np.where(near > 1e-15, dz[active] / safe, 1.0)

        # 쌍의 기여를 조각마다 모은다. ``bincount`` 가 ``np.add.at`` 보다
        # 훨씬 빠르고, 쌍마다 크기가 같고 방향이 반대이므로 질량중심은
        # 그대로 남는다.
        share = 0.5 * relaxation * weight
        shift_y = (np.bincount(hit_first, share * normal_y, n)
                   - np.bincount(hit_second, share * normal_y, n))
        shift_z = (np.bincount(hit_first, share * normal_z, n)
                   - np.bincount(hit_second, share * normal_z, n))
        y += shift_y
        z += shift_z
        total_shift = max(total_shift, float(np.max(np.abs(shift_y))),
                          float(np.max(np.abs(shift_z))))

        # 비탄성 접촉: 닿은 조각들을 공통 속도로 끌어당긴다. 쌍마다 주고받는
        # 양이 같고 반대이므로 (그래프 라플라시안) 총 운동량은 변하지 않는다.
        contacts = (np.bincount(hit_first, None, n)
                    + np.bincount(hit_second, None, n))
        max_contacts = contacts.max()
        if max_contacts:
            coupling = relaxation / max_contacts
            for axis in (0, 1):
                exchange = coupling * (velocities[hit_second, axis]
                                       - velocities[hit_first, axis])
                velocities[:, axis] += (np.bincount(hit_first, exchange, n)
                                        - np.bincount(hit_second, exchange, n))
    return total_shift


# ===========================================================================
# 구슬 모델 — bead model (real forces: contacts, cohesion)
# ===========================================================================
# 구슬 모델 — 실제로 작용하는 힘으로 움직이는 구슬.
#
# 원판 모델은 단면마다 자유도가 하나이고, 입자 모델은 겹침 금지라는 기하학적
# 제약으로 부피를 지킨다. 이 모델은 그런 사후 보정 없이, 3차원 구슬 하나하나에
# 대해 뉴턴 방정식을 푼다. 모든 효과가 힘으로 들어간다 — 접촉(스프링-댐퍼,
# 반발계수가 감쇠를 정한다), 구슬 사이 인력(표면장력), 벽 항력, 회전좌표계의
# 관성력, 겉보기 중력.
#
# 가장 느리지만, 입자 모델과 서로 다른 방식으로 같은 물리를 풀기 때문에 두
# 결과가 몇 % 안에서 맞는지 견주어 볼 수 있다.
#
# Bead model: the water as spheres, moved by the forces acting on them.
#
# Where the disc model gives each cross section one degree of freedom and the
# parcel model projects the water onto the flip plane with a geometric
# non-overlap constraint, this model integrates Newton's equations for a set of
# spherical beads in three dimensions.  Nothing is projected or constrained
# after the fact: every effect is a force.
#
# Forces on a bead
# ----------------
# * **Contact, bead-bead and bead-wall** — a linear spring-dashpot
#   (Cundall & Strack's soft-sphere DEM): while two beads overlap by
#   ``delta = 2a - r``, they push each other apart with
#   ``k * delta`` and lose energy through ``gamma * v_normal``.  The damping is
#   not free: for a linear contact the restitution coefficient ``e`` fixes it,
#
#       gamma = -2 ln(e) sqrt(k m_eff / (pi**2 + ln(e)**2))
#
#   so the ``restitution`` the other models apply as a velocity multiplier
#   drives the actual collisions here.
# * **Cohesion between beads** — a pairwise attraction of magnitude
#   ``cohesion_coefficient * surface_tension * a``, tapering linearly to zero at
#   ``cohesion_range`` contact diameters.  In the bulk it cancels by symmetry;
#   at the free surface it pulls inwards, which is what surface tension does.
#   Set the surface tension to zero for a cohesionless granular fill.
# * **Viscous drag** — ``-(2 m / tau) v``, the same wall drag the disc and
#   parcel models apply, with the same ``tau = 2 / drag_rate``.
# * **Fictitious forces** of the rotating bottle frame — centrifugal, Coriolis
#   and Euler, exactly as in the parcel model.  They have no component along
#   the rotation axis ``x``; the beads nevertheless move along ``x`` because
#   contacts and cohesion do.
# * **Apparent gravity** — zero in free flight, where the frame that follows
#   the bottle is falling with it; ``include_gravity`` restores it for the
#   phase in the hand.
#
# Integration and cost
# --------------------
# Velocity Verlet, with as many substeps per output step as the contact
# stiffness demands (``0.1 * pi * sqrt(m_eff / k)`` is the usual stable limit).
# Pairs come from a Verlet neighbour list rebuilt when a bead has moved half
# the skin distance, so the cost per substep is proportional to the number of
# neighbours rather than to ``n_beads**2``.
#
# Angular velocity follows from the conservation of angular momentum, as in the
# other two models, so the three can be compared directly.
#
# References
# ----------
# * P. A. Cundall and O. D. L. Strack, "A discrete numerical model for granular
#   assemblies", Géotechnique 29, 47 (1979) — soft-sphere DEM.
# * Y. Tsuji, T. Tanaka and T. Ishida, Powder Technol. 71, 239 (1992) — the
#   restitution-to-damping relation for a linear contact.
# * P. J. Dekker et al., Am. J. Phys. 86, 733 (2018); Eur. J. Phys. 45, 065003
#   (2024); arXiv:2502.12946 — the flip itself.

# 같은 크기 구를 최밀충전했을 때의 부피 채움률. 구슬 개수에서 "물의 부피를
# 채우는 구슬 크기" 를 얻는 데 쓴다.
CLOSE_PACKING = np.pi / (3 * np.sqrt(2))

# 이웃 목록에 더 얹어 두는 여유 거리 (접촉 지름 단위). 구슬이 이 여유의
# 절반만큼 움직이면 목록을 다시 만든다. 여유가 좁으면 목록을 너무 자주 다시
# 만들고 (0.35 에서는 서브스텝 1000 번에 387 번 재구성했다), 넓으면 쓸데없는
# 쌍을 들고 다닌다. 0.6 이 그 사이의 절충이다.
NEIGHBOUR_SKIN = 0.6


def bead_size(water_mass, n_beads, bead_radius=0.0, bead_mass=0.0,
              density=WATER_DENSITY):
    """구슬의 반지름과 질량, 그리고 그 합이 되는 물의 질량을 정한다.

    Resolve the bead radius and mass, and the water mass they add up to.

    ``bead_radius`` or ``bead_mass`` equal to zero means "derive it":

    * the radius from the water volume, so that ``n_beads`` beads at close
      packing occupy exactly the volume of the water;
    * the mass from ``water_mass / n_beads``.

    Giving both explicitly is allowed — the beads then carry
    ``n_beads * bead_mass`` of water, which is what the simulation uses.
    """
    if n_beads < 1:
        raise ValueError("n_beads must be at least 1")
    if bead_radius < 0 or bead_mass < 0:
        raise ValueError("bead radius and mass must not be negative")

    mass = bead_mass if bead_mass else water_mass / n_beads
    total = mass * n_beads
    if bead_radius:
        radius = bead_radius
    else:
        volume = total / density
        radius = (3 * volume * CLOSE_PACKING / (4 * np.pi * n_beads)) ** (1 / 3)
    return radius, mass, total


def largest_bead_that_fits(n_beads, radius=BOTTLE_RADIUS,
                           height=BOTTLE_HEIGHT, margin=0.98, shape=None):
    """``n_beads`` 개가 이 병에 들어갈 수 있는 구슬 반지름의 상한.

    두 가지 제약 가운데 작은 쪽이다.

    * 부피 — 구슬 ``n`` 개가 최밀충전으로도 병 부피를 넘을 수 없으니
      ``n * (4/3) pi a**3 <= CLOSE_PACKING * pi R**2 H``;
    * 측벽 — 구슬 하나는 병 안지름 안에 들어가야 하니 ``a < R``.

    ``margin`` 은 층으로 쌓는 이산성 때문에 두는 여유다.
    """
    if n_beads < 1:
        raise ValueError("n_beads must be at least 1")
    volume = float(cumulative_volume(height, **shape)) if shape \
        else float(np.pi * radius ** 2 * height)
    by_volume = (3 * CLOSE_PACKING * volume / (4 * np.pi * n_beads)) ** (1 / 3)
    return margin * min(by_volume, 0.95 * radius)


def _stack(n_beads, bead, radius, height, water_at_top, shape=None):
    """반지름 ``bead`` 인 구슬 ``n_beads`` 개를 한쪽 뚜껑에 쌓아 본다.

    격자 간격은 접촉 지름이므로 구슬들은 겹치지 않고 맞닿은 상태로 출발한다.
    각 층에 놓을 수 있는 자리는 그 높이의 병 반지름 ``R(z)`` 이 정하므로,
    목 근처의 층에는 몇 개밖에 들어가지 않고 나머지는 아래 몸통으로 내려간다 —
    실제 병에서 물이 뚜껑 쪽에 있을 때의 모습이다.

    병 안에 다 들어가지 않으면 ``None`` 을 돌려준다 — 부르는 쪽이 구슬을
    줄여서 다시 시도한다.
    """
    pitch = 2 * bead
    layer_gap = pitch * np.sqrt(2 / 3)
    if bead >= height / 2:
        return None

    def wall_at(z):
        if shape is None:
            return float(radius)
        return float(radius_profile(z, **shape))

    reach = int(np.ceil(2 * radius / pitch)) + 2
    grid_i, grid_j = np.meshgrid(np.arange(-reach, reach + 1),
                                 np.arange(-reach, reach + 1), indexing="ij")
    offsets = [(0.0, 0.0), (bead, pitch * np.sqrt(3) / 6)]

    placed = []
    count = 0
    layer = 0
    max_layers = int(np.ceil(height / layer_gap)) + 2
    while count < n_beads and layer < max_layers:
        shift = offsets[layer % 2]
        x = (grid_i + 0.5 * (grid_j % 2)) * pitch + shift[0]
        y = grid_j * (np.sqrt(3) / 2) * pitch + shift[1]
        x, y = x.ravel(), y.ravel()
        if water_at_top:
            z = height - bead - layer * layer_gap
        else:
            z = bead + layer * layer_gap
        if z < bead or z > height - bead:
            return None
        max_radius = wall_at(z) - bead
        layer += 1
        if max_radius <= 0:
            # 목이 구슬보다 좁다: 이 층은 비우고 아래로 내려간다
            continue
        inside = x ** 2 + y ** 2 <= max_radius ** 2
        lx, ly = x[inside], y[inside]
        if lx.size == 0:
            continue
        # 각 층을 축에서 바깥으로 채운다: 마지막 남은 층이 가운데 모인다
        order = np.argsort(lx ** 2 + ly ** 2)
        lx, ly = lx[order], ly[order]
        take = min(lx.size, n_beads - count)
        placed.append(np.column_stack([lx[:take], ly[:take],
                                       np.full(take, z)]))
        count += take

    if count < n_beads or not placed:
        return None
    return np.concatenate(placed), len(placed)


def seed_beads(water_mass, n_beads, bead_radius=0.0, bead_mass=0.0,
               water_at_top=True, radius=BOTTLE_RADIUS, height=BOTTLE_HEIGHT,
               density=WATER_DENSITY, shape=None):
    """``n_beads`` 개의 구슬을 한쪽 뚜껑에 최밀충전으로 쌓는다.

    구슬이 병에 들어가지 않는 조합 — 반지름을 손으로 크게 준 경우, 구슬이
    너무 많은 경우, 병이 좁은 경우 — 은 예전에는 오류였다.  이제는 들어갈 수
    있는 크기까지 반지름을 줄이고 :class:`RuntimeWarning` 으로 알린다.
    물의 질량은 그대로이므로, 계산은 언제나 끝까지 간다.
    """
    bead, mass, total = bead_size(water_mass, n_beads, bead_radius, bead_mass,
                                  density)

    limit = largest_bead_that_fits(n_beads, radius, height, shape=shape)
    asked = bead
    if bead > limit:
        bead = limit
    stacked = _stack(n_beads, bead, radius, height, water_at_top, shape)
    # 층으로 쌓는 이산성 때문에 상한을 지켰어도 마지막 한 층이 삐져나갈 수
    # 있다. 그때는 들어갈 때까지 조금씩 줄인다.
    attempts = 0
    while stacked is None and attempts < 200:
        bead *= 0.95
        stacked = _stack(n_beads, bead, radius, height, water_at_top, shape)
        attempts += 1
    if stacked is None:
        raise ValueError(
            "구슬 %d 개는 이 병(반지름 %.3f m, 높이 %.3f m)에 어떤 크기로도 "
            "들어가지 않습니다 / %d beads do not fit in this bottle at any size"
            % (n_beads, radius, height, n_beads))

    positions, layers = stacked
    if bead < asked * (1 - 1e-9):
        warnings.warn(
            "구슬 반지름을 %.4f m 에서 %.4f m 로 줄였습니다: 그대로는 구슬 "
            "%d 개가 이 병에 들어가지 않습니다. 물의 질량 %.4f kg 은 그대로 "
            "쓰므로 구슬의 유효 밀도가 물보다 높아집니다 / bead radius "
            "reduced from %.4f m to %.4f m so that %d beads fit; the water "
            "mass is unchanged."
            % (asked, bead, n_beads, total, asked, bead, n_beads),
            RuntimeWarning, stacklevel=2)

    return {"positions": positions,
            "velocities": np.zeros((n_beads, 3)),
            "bead_radius": bead, "bead_mass": mass, "water_mass": total,
            "n_layers": layers, "n_beads": n_beads,
            "bead_radius_asked": asked}


def contact_damping(stiffness, effective_mass, restitution):
    """선형 접촉에 이 반발계수를 주는 감쇠(댐퍼) 상수.

    Dashpot constant that gives a linear contact this restitution.
    """
    e = min(max(abs(restitution), 1e-6), 1 - 1e-9)
    log_e = np.log(e)
    return -2 * log_e * np.sqrt(stiffness * effective_mass
                                / (np.pi ** 2 + log_e ** 2))


def critical_time_step(bead_mass, stiffness, safety=0.1):
    """선형 접촉이 안정한 최대 시간 간격: 접촉 시간의 일부.

    Largest stable step for a linear contact: a fraction of its duration.
    """
    effective_mass = bead_mass / 2
    return safety * np.pi * np.sqrt(effective_mass / stiffness)


def substeps_for(dt, bead_mass, stiffness, safety=0.1):
    """출력 스텝 ``dt`` 를 몇 개의 서브스텝으로 잘라야 하는가.

    How many substeps an output step of ``dt`` has to be cut into.
    """
    limit = critical_time_step(bead_mass, stiffness, safety)
    return max(1, int(np.ceil(dt / limit)))


def pair_accelerations(positions, velocities, pairs, bead_radius, bead_mass,
                       stiffness, damping, cohesion, cohesion_range):
    """구슬마다 받는 접촉력과 인력에 의한 가속도.

    Contact and cohesion accelerations of every bead.
    """
    first, second = pairs
    accelerations = np.zeros_like(positions)
    if first.size == 0:
        return accelerations

    delta = positions[first] - positions[second]
    distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    distance = np.where(distance > 1e-15, distance, 1e-15)
    normal = delta / distance[:, None]

    diameter = 2 * bead_radius
    force = np.zeros(first.size)

    # 접촉: 스프링 + 다가오는 속도에 걸리는 댐퍼
    touching = distance < diameter
    if touching.any():
        overlap = diameter - distance[touching]
        relative = np.einsum("ij,ij->i",
                             velocities[first[touching]]
                             - velocities[second[touching]], normal[touching])
        force[touching] = stiffness * overlap - damping * relative

    # 응집: 끌어당기는 힘, 사거리 끝에서 0 으로 잦아든다
    if cohesion > 0:
        reach = cohesion_range * diameter
        pulling = (distance >= diameter) & (distance < reach)
        if pulling.any():
            taper = 1 - (distance[pulling] - diameter) / (reach - diameter)
            force[pulling] -= cohesion * taper

    contribution = (force / bead_mass)[:, None] * normal
    # 쌍의 기여를 구슬마다 모을 때 ``np.add.at`` 은 같은 번호가 여러 번
    # 나오는 누적을 느리게 처리한다 (전체 계산 시간의 4분의 1을 여기서
    # 썼다). 축마다 ``bincount`` 를 쓰면 같은 일이 C 루프 한 번으로 끝난다.
    count = positions.shape[0]
    for axis in range(3):
        weights = contribution[:, axis]
        accelerations[:, axis] += np.bincount(first, weights, count)
        accelerations[:, axis] -= np.bincount(second, weights, count)
    return accelerations


def wall_accelerations(positions, velocities, bead_radius, bead_mass,
                       stiffness, damping, radius=BOTTLE_RADIUS,
                       height=BOTTLE_HEIGHT, shape=None):
    """측벽과 두 뚜껑과의 접촉에서 오는 가속도.

    Contact accelerations from the side wall and the two caps.

    ``shape`` 를 주면 벽은 원통이 아니라 실제 병의 회전면 ``r = R(z)`` 이다.
    어깨에서는 벽이 기울어져 있으므로 법선이 축 방향 성분을 갖는다 — 물을
    안쪽으로만이 아니라 **아래로도** 밀어내는 이 성분이, 원심력에 밀려 올라온
    물을 목이 붙잡는 힘이다.  원통 벽에는 없는 항이고, 뒤집기에서 물이
    얼마나 퍼지는지를 실제로 좌우한다.
    """
    accelerations = np.zeros_like(positions)

    # 측벽: 겹침은 (x, y) 평면에서 반지름 방향이다
    lateral = positions[:, :2]
    span = np.sqrt(np.einsum("ij,ij->i", lateral, lateral))
    if shape is None:
        wall = np.full(span.shape, float(radius))
        slope = np.zeros_like(span)
    else:
        z = positions[:, 2]
        wall = radius_profile(z, **shape)
        # dR/dz: 몸통과 목에서는 0, 어깨에서만 음수
        z_1 = shape["shoulder_start"] * shape["height"]
        z_2 = shape["shoulder_end"] * shape["height"]
        if z_2 > z_1:
            tilt = (shape["neck"] - shape["radius"]) / (z_2 - z_1)
        else:
            tilt = 0.0
        slope = np.where((z > z_1) & (z < z_2), tilt, 0.0)

    # 기울어진 벽에서는 벽까지의 최단거리가 반지름 차이보다 짧다:
    # 원뿔면까지의 수직 거리는 (R(z) - span) * cos(alpha), tan(alpha) = |R'|.
    scale = 1.0 / np.sqrt(1 + slope ** 2)
    overlap = bead_radius - (wall - span) * scale
    pressing = overlap > 0
    if pressing.any():
        safe = np.where(span > 1e-15, span, 1.0)
        radial = lateral / safe[:, None]
        # 안쪽을 향하는 법선 (-r_hat, +slope) / sqrt(1 + slope**2)
        nx = -radial[:, 0] * scale
        ny = -radial[:, 1] * scale
        nz = slope * scale
        # 벽을 파고드는 속도 (양수면 더 깊이 들어가는 중)
        approach = -(velocities[:, 0] * nx + velocities[:, 1] * ny
                     + velocities[:, 2] * nz)
        magnitude = (stiffness * overlap[pressing]
                     + damping * approach[pressing]) / bead_mass
        accelerations[pressing, 0] += magnitude * nx[pressing]
        accelerations[pressing, 1] += magnitude * ny[pressing]
        accelerations[pressing, 2] += magnitude * nz[pressing]

    # 위아래 뚜껑
    bottom = bead_radius - positions[:, 2]
    low = bottom > 0
    if low.any():
        accelerations[low, 2] += (stiffness * bottom[low]
                                  - damping * velocities[low, 2]) / bead_mass
    top = positions[:, 2] - (height - bead_radius)
    high = top > 0
    if high.any():
        accelerations[high, 2] -= (stiffness * top[high]
                                   + damping * velocities[high, 2]) / bead_mass
    return accelerations


def frame_accelerations(positions, velocities, omega, omega_dot, y_cm, z_cm,
                        tau, gravity=0.0, angle=0.0, coriolis=True,
                        euler=True):
    """회전좌표계의 관성력, 그리고 항력과 겉보기 중력.

    Fictitious forces of the rotating frame, plus drag and apparent gravity.
    """
    accelerations = np.zeros_like(positions)
    offset_y = positions[:, 1] - y_cm
    offset_z = positions[:, 2] - z_cm

    accelerations[:, 1] += omega ** 2 * offset_y
    accelerations[:, 2] += omega ** 2 * offset_z
    if coriolis:
        accelerations[:, 1] += 2 * omega * velocities[:, 2]
        accelerations[:, 2] -= 2 * omega * velocities[:, 1]
    if euler:
        accelerations[:, 1] += omega_dot * offset_z
        accelerations[:, 2] -= omega_dot * offset_y
    if gravity:
        # theta = 0 은 똑바로 선 병이고, 그때 중력은 물을 바닥으로 당긴다.
        accelerations[:, 1] -= gravity * np.sin(angle)
        accelerations[:, 2] -= gravity * np.cos(angle)
    accelerations -= (2 / tau) * velocities
    return accelerations


def bead_rotational_inertia(positions, y_cm, z_cm, bead_mass, bead_radius):
    """질량중심을 지나는 횡축에 대한 구슬들의 관성모멘트.

    Moment of inertia of the beads about the transverse axis at the centre of mass.

    Rotation is about ``x``, so the in-plane offsets count; each bead also
    carries its own ``2/5 m a**2``, this time as a real sphere.
    """
    offset_y = positions[:, 1] - y_cm
    offset_z = positions[:, 2] - z_cm
    n = positions.shape[0]
    return bead_mass * float(np.sum(offset_y ** 2 + offset_z ** 2)) \
        + 2 / 5 * n * bead_mass * bead_radius ** 2


def bead_center_of_mass(positions, eps, length_bottle=BOTTLE_HEIGHT,
                        bottle_center=None):
    """병과 구슬을 합친 질량중심 ``(y_cm, z_cm)``.

    Centre of mass ``(y_cm, z_cm)`` of the bottle plus beads.
    """
    if bottle_center is None:
        bottle_center = length_bottle / 2
    return ((1 - eps) * float(np.mean(positions[:, 1])),
            eps * bottle_center + (1 - eps) * float(np.mean(positions[:, 2])))


def relative_angular_momentum(positions, velocities, y_cm, z_cm, bead_mass):
    """물이 병 안에서 움직여서 갖는 각운동량.

    Angular momentum the beads carry by moving within the bottle.

    The total is ``L = J omega + this``, so the angular velocity of the bottle
    follows from ``omega = (L - L_rel) / J``.  It is worth about 10% of the
    total here — dropping it, as the code used to, is not a small error.  In
    the disc model it vanishes identically: those slices sit on the axis and
    move along it, so ``(r - r_cm) x v = 0``.
    """
    offset_y = positions[:, 1] - y_cm
    offset_z = positions[:, 2] - z_cm
    return bead_mass * float(np.sum(offset_y * velocities[:, 2]
                                    - offset_z * velocities[:, 1]))


# ===========================================================================
# 시뮬레이션과 그림 — integration, reporting, plots
# ===========================================================================
# 뒤집기 한 번을 적분하고, 요약하고, 그림을 그린다.
#
# 병을 회전시켜 놓고 자유비행을 따라간다. 물은 모델에 따라 원판·조각·구슬로
# 나뉘어 회전좌표계의 힘을 받아 움직이고, 물이 퍼지면서 관성모멘트가 커지면
# 각운동량 보존에 따라 회전이 느려진다.
#
# 데이터가 커져도 끝까지 가도록 세 가지를 둔다.
#
# * 원판·조각의 위치 이력은 :func:`recorded_steps` 가 정한 스텝에만 남긴다 —
#   스텝과 요소 수를 아무리 늘려도 메모리가 예산 안에 머문다.
# * 구슬의 서브스텝은 ``substep_limit`` 에서 자르고, 그 간격에서 안정하도록
#   접촉 강성을 낮춘다 — 정확도를 조금 내주고 계산이 끝난다.
# * ``progress`` 콜백으로 진행률을 알린다 — 몇 분씩 걸리는 계산이 멈춘 것처럼
#   보이지 않게.
#
# Simulate the flight of a partially filled bottle thrown in a flip.
#
# The bottle is released spinning and followed during free flight.  The water is
# discretised into elements that move under the fictitious forces of the
# rotating frame — thin coaxial discs sliding along the bottle axis
# (:func:`simulate`), or parcels that also resolve the cross section
# (:func:`simulate_parcels`).  As the water redistributes, the moment of inertia
# grows and, angular momentum being conserved, the rotation slows down.
#
# Every physical quantity lives in :class:`Parameters` and can be set
# from keyword arguments, from the command line, or from the interactive prompt
# (:func:`run_interactive`); nothing here reads a hard-coded constant.
#
# Run ``python 물병던지기_시뮤레이션.py --help`` for the available options.

NOTEBOOK_HINT = """물병던지기 시뮤레이션 / water bottle flip — 코드로 쓰는 방법:
  launch_ui()                                 슬라이더와 버튼 / sliders and buttons
  run_interactive()                           물리량을 하나씩 입력 / one prompt each
  report(run(Parameters(water_mass=0.4)))     코드에서 직접 / straight from code"""

GRADIO_MISSING = """UI 를 열려면 gradio 가 필요합니다 / the interface needs gradio:

    %pip install -q gradio

설치한 뒤 이 셀을 다시 실행하세요. 지금은 기본 설정으로 한 번 계산해서 보여드립니다.
Install it, then run this cell again; the defaults are simulated once below."""

# 붙여넣은 셀이 스스로 UI 를 띄우지 않게 하려면 BOTTLE_FLIP_AUTO_UI=0.
AUTO_LAUNCH_UI = os.environ.get("BOTTLE_FLIP_AUTO_UI", "1") != "0"


def in_notebook():
    """IPython 커널(주피터, Colab) 안에서 돌고 있는지 알려 준다.

    True when this code runs inside an IPython kernel (Jupyter, Colab).

    ``sys.argv`` then belongs to the kernel launcher — it carries the
    kernel's own ``-f .../kernel-*.json`` — so parsing it with argparse would
    fail with ``unrecognized arguments`` and raise ``SystemExit``.  Both the
    command line entry point and the backend choice check this.
    """
    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        return True
    return hasattr(builtins, "__IPYTHON__")


# 이력 배열 하나에 담는 최대 원소 수. 스텝과 요소 수를 곱한 크기가 이보다
# 커지면 기록 간격을 벌려 솎아낸다 (원소 하나가 8 바이트이므로 약 8 MB).
# 이렇게 두지 않으면 스텝 4000 회 x 조각 5000 개에서 배열 하나가 300 MB 를
# 넘어, 계산은 되는데 메모리 때문에 그림을 못 보는 일이 생긴다.
MAX_HISTORY_VALUES = 1_000_000

# 구슬을 실제 크기의 원으로 그리는 한계. 이보다 많으면 점으로 흩뿌린다:
# 원 4000 개를 그리면 그림 한 장에 몇 분이 걸린다.
BEAD_CIRCLE_LIMIT = 600


def recorded_steps(n_steps, width, budget=MAX_HISTORY_VALUES):
    """이력을 남길 스텝 번호와 그 간격.

    ``width`` 는 한 스텝이 차지하는 원소 수(원판 수, 조각 수)다.  전체가
    예산 안에 들면 모든 스텝을, 넘으면 일정 간격으로 솎아서 남긴다.  마지막
    스텝은 착지 상태이므로 언제나 남긴다.
    """
    stride = max(1, int(np.ceil(n_steps * max(width, 1) / budget)))
    steps = np.arange(0, n_steps, stride)
    if steps[-1] != n_steps - 1:
        steps = np.append(steps, n_steps - 1)
    return steps, stride


def console_progress(label, stream=None, every=2.0):
    """진행률을 한 줄에 덧써 주는 콜백을 만든다.

    구슬을 수천 개 넣으면 계산이 몇 분씩 걸린다. 아무 것도 보이지 않으면
    멈춘 것처럼 보이므로, 몇 초에 한 번씩 어디까지 왔는지 알린다.
    """
    stream = stream or sys.stderr
    state = {"last": 0.0}

    def report(step, n_steps):
        now = time.monotonic()
        done = step >= n_steps - 1
        if not done and now - state["last"] < every:
            return
        state["last"] = now
        share = (step + 1) / n_steps
        stream.write("\r  %s %5.1f%% (%d/%d 스텝)%s"
                     % (label, 100 * share, step + 1, n_steps,
                        "\n" if done else ""))
        stream.flush()

    return report


def _substep_plan(params, dt, bead_mass):
    """서브스텝 수와, 그 간격에서 안정한 접촉 강성.

    접촉 하나가 버티는 시간 간격은 ``0.1 pi sqrt(m_eff / k)`` 이다. 구슬이
    많아지면 구슬 하나의 질량이 작아지므로 필요한 서브스텝이 늘고, 4000 개쯤
    되면 끝나지 않는 계산이 된다.  그래서 ``substep_limit`` 에서 자르고,
    잘린 간격에서도 안정하도록 강성을 ``k = m_eff (0.1 pi / dt_sub)**2`` 까지
    낮춘다.  접촉이 그만큼 부드러워져 겹침이 커지므로 경고로 알린다.
    """
    wanted = params.bead_substeps or substeps_for(dt, bead_mass,
                                                  params.contact_stiffness)
    limit = params.substep_limit
    if not limit or wanted <= limit or params.bead_substeps:
        return wanted, params.contact_stiffness, wanted

    effective_mass = bead_mass / 2
    sub_dt = dt / limit
    softened = effective_mass * (0.1 * np.pi / sub_dt) ** 2
    warnings.warn(
        "안정 조건은 스텝마다 서브스텝 %d 회를 요구하지만 상한 %d 회에서 "
        "잘랐습니다. 그 간격에서 안정하도록 접촉 강성을 %.0f N/m 에서 "
        "%.0f N/m 로 낮췄습니다: 접촉이 더 부드러워져 겹침이 커지지만 계산은 "
        "끝까지 갑니다. 정확히 풀려면 substep_limit 을 올리거나 구슬 수를 "
        "줄이세요 / substeps capped at %d and the contact stiffness softened "
        "to %.0f N/m so the run finishes"
        % (wanted, limit, params.contact_stiffness, softened, limit, softened),
        RuntimeWarning, stacklevel=3)
    return limit, softened, wanted


def _warn_if_degenerate(low, high, center_of_mass, params):
    """뒤집기가 되려면 물기둥이 질량중심을 걸치고 있어야 한다.

    The flip needs the water column to straddle the centre of mass.
    """
    if not low < center_of_mass < high:
        warnings.warn(
            "the water column (%.3f-%.3f m) lies entirely on one side of the "
            "centre of mass (%.3f m): the centrifugal force pushes every "
            "element against the same cap, so the water cannot redistribute "
            "and the angular velocity stays constant. Use a filling fraction "
            "of roughly 0.2-0.4 (currently %.3f)."
            % (low, high, center_of_mass, params.filling_fraction),
            RuntimeWarning, stacklevel=3)


def simulate(params=None, progress=None, **overrides):
    """물을 병 축을 따라 미끄러지는 원판으로 보고 뒤집기를 적분한다.

    Integrate the flip with the water as discs sliding along the axis.

    원판은 같은 **질량** 을 갖는다. 단면이 변하는 병에서는 그것이 같은 두께를
    뜻하지 않으므로, 원판의 자리는 부피 좌표에서 잡고 (:func:`height_at_volume`)
    비압축성 제약도 거기서 건다.  목에 들어간 원판은 길게 늘어나고, 관성모멘트를
    셀 때도 그 자리의 좁은 반지름을 쓴다.

    ``progress(step, n_steps)`` 를 주면 스텝마다 불러 진행 상황을 알린다.
    """
    p = resolve_parameters(params, **overrides)

    t, dt = np.linspace(0, p.duration, p.n_steps, retstep=True)

    # 부피 좌표에서 같은 질량(=같은 부피)의 원판으로 나눈다.
    bottle_volume = p.bottle_volume
    slice_volume = p.water_volume / p.n_slices
    s_min = slice_volume / 2
    s_max = bottle_volume - slice_volume / 2
    l_min = float(p.height_of_volume(s_min))
    l_max = float(p.height_of_volume(s_max))

    # 병은 자유낙하 중에 관찰한다. 놓는 순간 물기둥은 이미 뚜껑 쪽에
    # 붙어 있다.
    if p.water_at_top:
        s0 = s_max - np.arange(p.n_slices) * slice_volume
    else:
        s0 = s_min + np.arange(p.n_slices) * slice_volume
    slices_positions = np.asarray(p.height_of_volume(s0), dtype=float)
    slices_velocities = np.zeros(p.n_slices)

    bottle_center = p.bottle_center_of_mass

    def state_of(positions):
        """원판 분포 하나에서 질량중심과 관성모멘트를 얻는다."""
        cm = find_center_of_mass(positions, p.epsilon, bottle_center)
        j = (rotational_inertia_water(positions, cm, p.water_mass,
                                      p.radius_at(positions))
             + p.bottle_inertia(cm))
        return cm, j

    center_of_mass, rotational_inertia = state_of(slices_positions)
    _warn_if_degenerate(slices_positions.min(), slices_positions.max(),
                        center_of_mass, p)
    # 자유비행 중에는 외부 돌림힘이 없으므로 이 값은 상수다.
    # 원판은 병 축 위에서 축 방향으로만 움직이므로 (r - r_cm) x v = 0 이다:
    # 원판 모델에서는 물의 상대 각운동량이 정확히 0 이고, L = J omega 가 맞다.
    angular_momentum = rotational_inertia * p.omega_0

    omega_values = np.zeros(p.n_steps)
    theta_values = np.zeros(p.n_steps)
    center_of_mass_values = np.zeros(p.n_steps)
    rotational_inertia_values = np.zeros(p.n_steps)
    # 원판 위치는 스텝마다 n_slices 개이므로 예산 안에서만 남긴다
    steps, stride = recorded_steps(p.n_steps, p.n_slices)
    slices_positions_values = np.zeros((steps.size, p.n_slices))
    slices_velocities_values = np.zeros((steps.size, p.n_slices))

    omega_values[0] = p.omega_0
    theta_values[0] = p.theta_0
    slices_positions_values[0] = slices_positions
    slices_velocities_values[0] = slices_velocities
    center_of_mass_values[0] = center_of_mass
    rotational_inertia_values[0] = rotational_inertia
    recorded = 1

    for k in range(1, p.n_steps):
        omega = omega_values[k - 1]
        theta = theta_values[k - 1]

        # 원판을 옮기고, 새 분포에서 무엇을 계산하기 전에 벽 반사와 겹침
        # 금지 제약을 먼저 적용한다.
        new_positions, new_velocities = update_slice_positions(
            slices_positions, slices_velocities, center_of_mass, omega, theta,
            p.tau, dt, p.axial_gravity)
        new_positions, new_velocities = check_boundary_conditions(
            new_positions, new_velocities, p.bounce, l_min, l_max)
        if p.incompressible:
            new_positions, new_velocities = enforce_incompressibility(
                new_positions, new_velocities, slice_volume, s_min, s_max,
                p.volume_below, p.height_of_volume)

        # 질량중심과 관성모멘트는 이전 분포가 아니라 방금 구한 새 원판
        # 분포에서 나온다.
        new_center_of_mass, new_rotational_inertia = state_of(new_positions)
        new_omega = angular_momentum / new_rotational_inertia
        # 사다리꼴 적분: theta + omega * dt 와 달리 dt 에 대해 2차 정확도.
        new_theta = theta + (omega + new_omega) * dt / 2

        omega_values[k] = new_omega
        theta_values[k] = new_theta
        center_of_mass_values[k] = new_center_of_mass
        rotational_inertia_values[k] = new_rotational_inertia
        # 다음 스텝의 출발점은 방금 구한 상태다 (이력 배열이 아니라 이 변수)
        slices_positions = new_positions
        slices_velocities = new_velocities
        center_of_mass = new_center_of_mass
        if recorded < steps.size and steps[recorded] == k:
            slices_positions_values[recorded] = new_positions
            slices_velocities_values[recorded] = new_velocities
            recorded += 1
        if progress is not None:
            progress(k, p.n_steps)

    return {
        "model": "slices",
        "history_steps": steps,
        "history_t": t[steps],
        "history_stride": stride,
        "parameters": p,
        "t": t,
        "dt": dt,
        "omega": omega_values,
        "theta": theta_values,
        "slices_positions": slices_positions_values,
        "slices_velocities": slices_velocities_values,
        "center_of_mass": center_of_mass_values,
        "rotational_inertia": rotational_inertia_values,
        "angular_momentum": angular_momentum,
        "relative_angular_momentum": np.zeros(p.n_steps),
        "water_height": p.water_height,
        "slice_volume": slice_volume,
        "slice_height": slice_volume / (np.pi * p.bottle_radius ** 2),
        "water_mass_used": p.water_mass,
        "tau": p.tau,
        "filling_fraction": p.filling_fraction,
    }


def simulate_parcels(params=None, progress=None, **overrides):
    """물을 조각으로 쪼개어 뒤집기를 적분한다.

    Integrate the flip with the water cut into parcels.

    Each disc of the coarse model is cut into ``n_pieces`` pieces that move in
    the flip plane, so the water can climb the side wall and cross the bottle
    instead of only sliding along the axis.  ``coriolis`` and ``euler`` switch
    off the corresponding fictitious force, which is what reduces this model
    to the disc model.

    ``progress(step, n_steps)`` 를 주면 스텝마다 불러 진행 상황을 알리고,
    조각 위치의 이력은 :func:`recorded_steps` 가 정한 스텝에만 남긴다.
    """
    p = resolve_parameters(params, **overrides)

    t, dt = np.linspace(0, p.duration, p.n_steps, retstep=True)
    seed = seed_parcels(p.water_mass, p.n_pieces,
                        water_at_top=p.water_at_top,
                        radius=p.bottle_radius, height=p.bottle_height,
                        density=p.water_density, shape=p.shape)
    x, y, z = seed["x"], seed["y"].copy(), seed["z"].copy()
    velocities = seed["velocities"].copy()
    a = seed["parcel_radius"]
    n_parcels = seed["n_parcels"]

    bottle_center = p.bottle_center_of_mass
    y_cm, z_cm = parcel_center_of_mass(y, z, p.epsilon, p.bottle_height,
                                       bottle_center)
    _warn_if_degenerate(z.min(), z.max(), z_cm, p)
    rotational_inertia = (
        parcel_rotational_inertia_water(y, z, y_cm, z_cm, p.water_mass)
        + p.bottle_inertia(z_cm, y_cm))
    # 놓는 순간 물은 병에 대해 정지해 있으므로 상대 각운동량은 0 이고,
    # 전체 각운동량은 J * omega_0 이다. 그 뒤로는 물이 병 안에서 움직이므로
    # L = J omega + L_rel 이 되고, omega 는 그 나머지에서 나온다.
    angular_momentum = rotational_inertia * p.omega_0

    omega_values = np.zeros(p.n_steps)
    theta_values = np.zeros(p.n_steps)
    center_of_mass_values = np.zeros(p.n_steps)
    center_of_mass_y_values = np.zeros(p.n_steps)
    rotational_inertia_values = np.zeros(p.n_steps)
    relative_momentum_values = np.zeros(p.n_steps)
    # 조각이 수천 개가 되면 전체 이력이 수백 MB 이므로 예산 안에서만 남긴다
    steps, stride = recorded_steps(p.n_steps, n_parcels)
    y_values = np.zeros((steps.size, n_parcels))
    z_values = np.zeros((steps.size, n_parcels))

    omega_values[0] = p.omega_0
    theta_values[0] = p.theta_0
    y_values[0] = y
    z_values[0] = z
    center_of_mass_values[0] = z_cm
    center_of_mass_y_values[0] = y_cm
    rotational_inertia_values[0] = rotational_inertia
    recorded = 1
    omega_dot = 0.0

    for k in range(1, p.n_steps):
        omega = omega_values[k - 1]
        theta = theta_values[k - 1]

        propagator = step_matrix(omega, omega_dot, p.tau, y_cm, z_cm,
                                 dt, p.axial_gravity, theta,
                                 p.coriolis, p.euler)
        advance_parcels(y, z, velocities, propagator)
        apply_walls(x, y, z, velocities, p.restitution, a,
                    p.bottle_radius, p.bottle_height, p.shape)
        if p.incompressible:
            resolve_overlaps(x, y, z, velocities, a,
                             iterations=p.overlap_iterations)
            apply_walls(x, y, z, velocities, p.restitution, a,
                        p.bottle_radius, p.bottle_height, p.shape)

        y_cm, z_cm = parcel_center_of_mass(y, z, p.epsilon, p.bottle_height,
                                           bottle_center)
        new_rotational_inertia = (
            parcel_rotational_inertia_water(y, z, y_cm, z_cm, p.water_mass)
            + p.bottle_inertia(z_cm, y_cm))
        relative = parcel_relative_angular_momentum(
            y, z, velocities, y_cm, z_cm, p.water_mass)
        new_omega = ((angular_momentum - relative) / new_rotational_inertia
                     if p.relative_momentum
                     else angular_momentum / new_rotational_inertia)
        relative_momentum_values[k] = relative
        # 다음 스텝의 오일러 힘에 넣을 각가속도를, 방금의 변화에서 그대로
        # 추정한다.
        omega_dot = (new_omega - omega) / dt
        new_theta = theta + (omega + new_omega) * dt / 2

        omega_values[k] = new_omega
        theta_values[k] = new_theta
        center_of_mass_values[k] = z_cm
        center_of_mass_y_values[k] = y_cm
        rotational_inertia_values[k] = new_rotational_inertia
        if recorded < steps.size and steps[recorded] == k:
            y_values[recorded] = y
            z_values[recorded] = z
            recorded += 1
        if progress is not None:
            progress(k, p.n_steps)

    return {
        "model": "parcels",
        "history_steps": steps,
        "history_t": t[steps],
        "history_stride": stride,
        "n_pieces": seed.get("n_pieces", p.n_pieces),
        "parameters": p,
        "t": t,
        "dt": dt,
        "omega": omega_values,
        "theta": theta_values,
        "parcels_x": x,
        "parcels_y": y_values,
        "parcels_z": z_values,
        "center_of_mass": center_of_mass_values,
        "center_of_mass_y": center_of_mass_y_values,
        "rotational_inertia": rotational_inertia_values,
        "angular_momentum": angular_momentum,
        "relative_angular_momentum": relative_momentum_values,
        "water_height": seed["column_height"],
        "parcel_radius": a,
        "n_parcels": n_parcels,
        "n_layers": seed["n_layers"],
        "pieces_per_layer": seed["pieces_per_layer"],
        "tau": p.tau,
        "filling_fraction": p.filling_fraction,
    }


def simulate_beads(params=None, progress=None, **overrides):
    """물을 구슬로 보고, 실제로 작용하는 힘으로 뒤집기를 적분한다.

    Integrate the flip with the water as beads, moved by actual forces.

    Every effect is a force: bead-bead and bead-wall contacts (a linear
    spring-dashpot whose damping comes from ``restitution``), the attraction
    between beads that plays the role of surface tension, the same viscous
    wall drag as the other models, the fictitious forces of the rotating
    frame, and the apparent gravity.  See the bead model section above.

    The contact integration is substepped as finely as the stiffness demands,
    so ``n_steps`` only sets how often the state is recorded.

    구슬을 수천 개 넣으면 안정 조건이 요구하는 서브스텝 수가 끝나지 않을
    만큼 늘어난다.  ``substep_limit`` 이 그 수를 자르고 접촉 강성을 그 간격에서
    안정한 값까지 낮추므로, 오래 걸리더라도 계산은 끝나고 그림이 나온다.
    ``progress(step, n_steps)`` 를 주면 어디까지 왔는지 알려 준다.
    """
    p = resolve_parameters(params, **overrides)

    t, dt = np.linspace(0, p.duration, p.n_steps, retstep=True)
    seed = seed_beads(p.water_mass, p.n_beads, p.bead_radius,
                      p.bead_mass, water_at_top=p.water_at_top,
                      radius=p.bottle_radius, height=p.bottle_height,
                      density=p.water_density, shape=p.shape)
    positions = seed["positions"].copy()
    velocities = seed["velocities"].copy()
    bead_radius = seed["bead_radius"]
    bead_mass = seed["bead_mass"]
    water_mass = seed["water_mass"]
    if abs(water_mass - p.water_mass) > 1e-9 * max(water_mass, p.water_mass):
        warnings.warn(
            "n_beads * bead_mass is %.4f kg, not the %.4f kg of water_mass; "
            "the beads carry %.4f kg." % (water_mass, p.water_mass, water_mass),
            RuntimeWarning, stacklevel=2)

    epsilon = p.bottle_mass / (p.bottle_mass + water_mass)
    # 서브스텝 수와, 그 간격에서 안정한 접촉 강성을 함께 정한다
    substeps, stiffness, substeps_wanted = _substep_plan(p, dt, bead_mass)
    damping = contact_damping(stiffness, bead_mass / 2, p.restitution)
    wall_damping = contact_damping(stiffness, bead_mass, p.restitution)
    cohesion = p.cohesion_force(bead_radius)
    sub_dt = dt / substeps

    diameter = 2 * bead_radius
    reach = max(p.cohesion_range, 1.0) * diameter if cohesion > 0 else diameter
    skin = NEIGHBOUR_SKIN * diameter
    pairs = neighbour_pairs(positions, reach + skin)
    anchor = positions.copy()

    def acceleration(current_positions, current_velocities, omega, omega_dot,
                     y_cm, z_cm, angle):
        total = pair_accelerations(
            current_positions, current_velocities, pairs, bead_radius,
            bead_mass, stiffness, damping, cohesion, p.cohesion_range)
        total += wall_accelerations(
            current_positions, current_velocities, bead_radius, bead_mass,
            stiffness, wall_damping, p.bottle_radius, p.bottle_height,
            p.shape)
        total += frame_accelerations(
            current_positions, current_velocities, omega, omega_dot, y_cm,
            z_cm, p.tau, p.axial_gravity, angle, p.coriolis, p.euler)
        return total

    bottle_center = p.bottle_center_of_mass
    y_cm, z_cm = bead_center_of_mass(positions, epsilon, p.bottle_height,
                                     bottle_center)
    _warn_if_degenerate(positions[:, 2].min(), positions[:, 2].max(), z_cm, p)
    rotational_inertia = (
        bead_rotational_inertia(positions, y_cm, z_cm, bead_mass, bead_radius)
        + p.bottle_inertia(z_cm, y_cm))
    # 놓는 순간 물은 병에 대해 정지해 있으므로 L_rel = 0 이다.
    angular_momentum = rotational_inertia * p.omega_0

    omega_values = np.zeros(p.n_steps)
    theta_values = np.zeros(p.n_steps)
    center_of_mass_values = np.zeros(p.n_steps)
    center_of_mass_y_values = np.zeros(p.n_steps)
    rotational_inertia_values = np.zeros(p.n_steps)
    relative_momentum_values = np.zeros(p.n_steps)
    overlap_values = np.zeros(p.n_steps)
    rebuilds = 0

    omega_values[0] = p.omega_0
    theta_values[0] = p.theta_0
    center_of_mass_values[0] = z_cm
    center_of_mass_y_values[0] = y_cm
    rotational_inertia_values[0] = rotational_inertia
    relative_momentum_values[0] = 0.0

    # 위치는 몇 시점만 남긴다. 구슬 수천 개로 길게 돌리면 그러지 않고서는
    # 수백 MB 를 들고 다녀야 한다.
    snapshot_indices = sorted({0, p.n_steps // 4, p.n_steps // 2,
                               3 * p.n_steps // 4, p.n_steps - 1})
    snapshots = {0: positions.copy()}
    omega_dot = 0.0

    for k in range(1, p.n_steps):
        omega = omega_values[k - 1]
        theta = theta_values[k - 1]

        for _ in range(substeps):
            acc = acceleration(positions, velocities, omega, omega_dot, y_cm,
                               z_cm, theta)
            velocities += 0.5 * sub_dt * acc
            positions += sub_dt * velocities
            moved = np.max(np.abs(positions - anchor))
            if moved > 0.5 * skin:
                pairs = neighbour_pairs(positions, reach + skin)
                anchor = positions.copy()
                rebuilds += 1
            acc = acceleration(positions, velocities, omega, omega_dot, y_cm,
                               z_cm, theta)
            velocities += 0.5 * sub_dt * acc

        y_cm, z_cm = bead_center_of_mass(positions, epsilon, p.bottle_height,
                                         bottle_center)
        new_rotational_inertia = (
            bead_rotational_inertia(positions, y_cm, z_cm, bead_mass,
                                    bead_radius)
            + p.bottle_inertia(z_cm, y_cm))
        # 물이 병 안에서 도는 몫을 빼고 나서야 병의 각속도가 나온다.
        # L_total = J omega + L_rel 이고, 여기서 L_rel 은 전체의 10% 에
        # 이르므로 예전처럼 버리면 회전이 그만큼 어긋난다.
        relative = relative_angular_momentum(positions, velocities, y_cm,
                                             z_cm, bead_mass)
        new_omega = ((angular_momentum - relative) / new_rotational_inertia
                     if p.relative_momentum
                     else angular_momentum / new_rotational_inertia)
        omega_dot = (new_omega - omega) / dt
        new_theta = theta + (omega + new_omega) * dt / 2

        omega_values[k] = new_omega
        theta_values[k] = new_theta
        center_of_mass_values[k] = z_cm
        center_of_mass_y_values[k] = y_cm
        rotational_inertia_values[k] = new_rotational_inertia
        relative_momentum_values[k] = relative
        if pairs[0].size:
            delta = positions[pairs[0]] - positions[pairs[1]]
            distance = np.sqrt(np.einsum("ij,ij->i", delta, delta))
            overlap_values[k] = max(0.0, float(np.max(diameter - distance)))
        if k in snapshot_indices:
            snapshots[k] = positions.copy()
        if progress is not None:
            progress(k, p.n_steps)

    return {
        "model": "beads",
        "parameters": p,
        "t": t,
        "dt": dt,
        "omega": omega_values,
        "theta": theta_values,
        "center_of_mass": center_of_mass_values,
        "center_of_mass_y": center_of_mass_y_values,
        "rotational_inertia": rotational_inertia_values,
        "angular_momentum": angular_momentum,
        "relative_angular_momentum": relative_momentum_values,
        "max_overlap": overlap_values,
        "water_height": water_mass / p.water_density
        / (np.pi * p.bottle_radius ** 2),
        "filling_fraction": water_mass / (p.water_density * p.bottle_volume),
        "tau": p.tau,
        "bead_radius": bead_radius,
        "bead_mass": bead_mass,
        "bead_water_mass": water_mass,
        "n_beads": p.n_beads,
        "n_layers": seed["n_layers"],
        "substeps": substeps,
        "substeps_wanted": substeps_wanted,
        "contact_stiffness_used": stiffness,
        "sub_dt": sub_dt,
        "cohesion_force": cohesion,
        "contact_damping": damping,
        "neighbour_rebuilds": rebuilds,
        "snapshots": snapshots,
        "positions": positions,
    }


def run(params=None, progress=None, **overrides):
    """``params.model`` 이 고른 모델로 한 번 던진다.

    ``progress`` 는 그 모델에 그대로 넘긴다: 큰 계산에서 진행 상황을 보려면
    :func:`console_progress` 가 만든 콜백을 주면 된다.
    """
    p = resolve_parameters(params, **overrides)
    if p.model == "parcels":
        return simulate_parcels(p, progress=progress)
    if p.model == "beads":
        return simulate_beads(p, progress=progress)
    return simulate(p, progress=progress)


# ===========================================================================
# 착지 — does it actually stand up?
# ===========================================================================
# 예전 코드는 "최종 각속도" 와 "몇 바퀴 돌았나" 까지만 알려 주었다. 그런데
# 물병 세우기 연구가 묻는 것은 그것이 아니라 **섰는가 넘어졌는가** 이다.
# 회전이 느려지는 것은 필요조건일 뿐, 그것만으로는 세워지지 않는다.
#
# 병이 서려면 두 가지가 함께 맞아야 한다.
#
# 1. 닿는 순간 거의 똑바로 서 있어야 한다. 기울기가 이미 접지 반지름이 허용하는
#    한계를 넘으면, 각속도가 0 이어도 그대로 넘어간다.
# 2. 남은 회전에너지가 질량중심을 접지 링 위로 넘길 만큼 크면 안 된다.
#    바닥 모서리를 축으로 도는 문제이므로, 문턱은 질량중심을 그 모서리 바로
#    위까지 들어올리는 위치에너지다.
#
# 두 조건은 모두 접지 반지름 ``contact_radius`` 와 질량중심 높이에 달려 있고,
# 그래서 충전율이 답에 들어온다: 물이 적으면 회전이 안 느려지고, 물이 많으면
# 질량중심이 높아져 문턱이 낮아진다.
#
# 충돌에 대한 가정
# ----------------
# 떨어지는 속도가 갖는 병진 운동에너지는 기본값에서 **모두 흡수된다** 고 본다.
# 실제로 PET 병의 바닥은 찌그러지고, 안의 물은 바닥을 때리면서 자기 운동에너지를
# 거의 다 잃는다 — 물병 세우기가 성립하는 이유 자체가 이 소산이다.
# ``impact_absorption`` 을 1 보다 작게 주면 그 중 일부가 넘어뜨리는 데 쓰인다.
# 이것은 모델의 선택이지 계산 결과가 아니므로, 값을 바꿔 가며 결론이 얼마나
#흔들리는지 보는 편이 낫다.

def wrap_angle(angle):
    """각도를 ``(-pi, pi]`` 로 접는다.

    Wrap an angle into ``(-pi, pi]``.
    """
    return -(-(np.asarray(angle, dtype=float) + np.pi) % (2 * np.pi) - np.pi)


def landing_outcome(results, impact_absorption=None, water_coupling=None):
    """착지 순간의 자세와 남은 회전으로 섰는지 넘어졌는지 판정한다.

    Decide whether the bottle stands, from its attitude and residual spin.

    Returns a dict with the landing tilt, the tipping threshold, the energy
    margin and ``stands``.
    """
    p = results["parameters"]
    if impact_absorption is None:
        impact_absorption = p.impact_absorption
    coupling = p.water_coupling if water_coupling is None else water_coupling
    theta = results["theta"]
    omega = results["omega"]

    tilt = float(wrap_angle(theta[-1]))
    spin = float(omega[-1])
    water_mass = float(results.get("water_mass_used", p.water_mass))
    total_mass = p.bottle_mass + water_mass
    j_cm = float(results["rotational_inertia"][-1])

    # 병 좌표계에서의 질량중심 (축에서 벗어난 성분까지)
    z_cm = float(results["center_of_mass"][-1])
    y_cm = float(np.asarray(results.get("center_of_mass_y", 0.0)).ravel()[-1]) \
        if results.get("center_of_mass_y") is not None else 0.0

    contact = p.contact_radius

    def rim(side):
        """한쪽 바닥 모서리를 축으로 본 기하: 균형점 기울기와 지금 높이."""
        arm_y = y_cm - side * contact
        reach = math.hypot(arm_y, z_cm)
        # h(theta) = reach * cos(theta + delta):
        # 똑바로 서 있으면 z_cm, 균형점에서 reach 로 가장 높다.
        delta = math.atan2(arm_y, z_cm) if reach > 0 else 0.0
        return reach, delta, -delta, reach * math.cos(tilt + delta)

    # 닿는 순간 이미 어느 한쪽 균형점을 넘어 기울어져 있으면, 각속도가 0
    # 이어도 그대로 넘어간다. 두 모서리를 모두 본다.
    past_balance = False
    for side in (1.0, -1.0):
        _, _, tip, _ = rim(side)
        if (tilt - tip) * side >= 0:
            past_balance = True

    # 넘어뜨리는 일은 회전이 데려가는 쪽 모서리를 축으로 일어난다
    side = 1.0 if spin >= 0 else -1.0
    reach, delta, tip_tilt, height_now = rim(side)

    # 모서리를 축으로 한 관성모멘트 (평행축 정리).
    # 여기서 물과 병을 나눈다. 바닥이 상에 닿는 충돌은 몇 ms 로 끝나는데
    # 안의 물이 반응하는 데는 수십 ms 가 걸리므로, 물이 갖고 있던 회전은
    # 병을 넘어뜨리는 운동으로 거의 넘어가지 않는다.  넘어뜨리는 데 쓰이는
    # 관성은 병 껍질의 것에 물의 일부(``water_coupling``)를 더한 만큼이다.
    # 반면 병을 도로 눌러 세우는 무게는 물까지 포함한 전체 질량이다 —
    # 이 비대칭이 부분적으로 채운 병이 서는 이유다.
    j_bottle_pivot = p.bottle_inertia(z_cm, y_cm) + p.bottle_mass * reach ** 2
    j_water_pivot = max(j_cm + total_mass * reach ** 2 - j_bottle_pivot, 0.0)
    j_pivot = j_bottle_pivot + coupling * j_water_pivot
    kinetic = 0.5 * j_pivot * spin ** 2
    if impact_absorption < 1.0:
        # 흡수되지 않은 낙하 운동에너지가 넘어뜨리는 데 더해진다
        fall_speed = abs(p.gravity * results["t"][-1]
                         - (p.launch_speed if p.flight == "launch" else 0.0))
        kinetic += (1.0 - impact_absorption) * 0.5 * total_mass * fall_speed ** 2
    barrier = total_mass * p.gravity * max(reach - height_now, 0.0)

    stands = bool(not past_balance and kinetic < barrier)

    return {
        "tilt": tilt,
        "tilt_deg": math.degrees(tilt),
        "spin": spin,
        "tip_tilt": tip_tilt,
        "tip_tilt_deg": math.degrees(tip_tilt),
        "kinetic": kinetic,
        "barrier": barrier,
        "margin": barrier - kinetic,
        "stands": stands,
        "past_balance": bool(past_balance),
        "water_coupling": coupling,
        "j_pivot": j_pivot,
        "contact_radius": contact,
        "center_height": z_cm,
        "turns": float((theta[-1] - theta[0]) / (2 * np.pi)),
    }


def format_landing(results, impact_absorption=None, water_coupling=None):
    """착지 판정을 사람이 읽는 글로.

    The landing verdict, as text.
    """
    out = landing_outcome(results, impact_absorption, water_coupling)
    verdict = "섰다 / STANDS" if out["stands"] else "넘어졌다 / FALLS"
    lines = ["--- 착지 / landing " + "-" * 34,
             "판정 / verdict   : %s" % verdict,
             "착지 기울기      : %+.1f deg (넘어가는 한계 %.1f deg)"
             % (out["tilt_deg"], abs(out["tip_tilt_deg"])),
             "남은 각속도      : %.2f rad/s" % out["spin"],
             "회전에너지/문턱  : %.3e / %.3e J (여유 %+.3e J)"
             % (out["kinetic"], out["barrier"], out["margin"])]
    if out["past_balance"]:
        lines.append("이유             : 닿는 순간 이미 균형점을 넘어 기울어져 "
                     "있었다 / already past the balance point on landing")
    elif not out["stands"]:
        lines.append("이유             : 남은 회전이 질량중심을 접지 모서리 "
                     "위로 넘긴다 / the residual spin carries it over the rim")
    return "\n".join(lines)


def format_report(results, show_inputs=True):
    """입력한 값과 계산 결과를 짧은 글로 정리한다.

    The inputs and a short summary of a run, as text.
    """
    p = results["parameters"]
    theta = results["theta"]
    omega = results["omega"]
    j = results["rotational_inertia"]
    turns = (theta[-1] - theta[0]) / (2 * np.pi)

    lines = []
    if show_inputs:
        lines.append(p.describe())
        lines.append("--- 결과 / results " + "-" * 34)
    if results["model"] == "beads":
        lines.append("discretisation   : %d beads of radius %.2f mm and mass "
                     "%.3e kg in %d layers"
                     % (results["n_beads"], 1e3 * results["bead_radius"],
                        results["bead_mass"], results["n_layers"]))
        lines.append("contact          : k = %.0f N/m, damping %.3f N s/m, "
                     "%d substeps of %.2e s"
                     % (results["contact_stiffness_used"],
                        results["contact_damping"],
                        results["substeps"], results["sub_dt"]))
        if results["substeps_wanted"] > results["substeps"]:
            lines.append("서브스텝 상한     : 안정 조건은 %d 회를 요구했지만 "
                         "%d 회로 자르고 강성을 %.0f -> %.0f N/m 로 낮춤"
                         % (results["substeps_wanted"], results["substeps"],
                            p.contact_stiffness,
                            results["contact_stiffness_used"]))
        lines.append("cohesion         : %.3e N per pair up to %.2f diameters"
                     % (results["cohesion_force"], p.cohesion_range))
        lines.append("max overlap      : %.3f%% of a diameter (soft contacts)"
                     % (100 * results["max_overlap"].max()
                        / (2 * results["bead_radius"])))
        lines.append("relative L       : %.2e kg m^2/s at most, %.2f%% of "
                     "the total (%s)"
                     % (np.max(np.abs(results["relative_angular_momentum"])),
                        100 * np.max(np.abs(
                            results["relative_angular_momentum"]))
                        / max(abs(results["angular_momentum"]), 1e-30),
                        "kept in omega" if p.relative_momentum
                        else "dropped: relative_momentum is off"))
    elif results["model"] == "parcels":
        lines.append("discretisation   : %d parcels, %.1f per cross section in "
                     "%d layers, exclusion radius %.1f mm"
                     % (results["n_parcels"], results["pieces_per_layer"],
                        results["n_layers"],
                        1e3 * results["parcel_radius"]))
    else:
        lines.append("discretisation   : %d slices of %.3f mm"
                     % (p.n_slices, 1e3 * results["slice_height"]))
    lines.append("moment of inertia: %.3e -> %.3e kg m^2 (x%.2f)"
                 % (j[0], j[-1], j[-1] / j[0]))
    lines.append("angular velocity : %.2f -> %.2f rad/s (x%.2f)"
                 % (omega[0], omega[-1], omega[-1] / omega[0]))
    # 보존을 확인할 양은 J*omega 가 아니라 J*omega + L_rel 이다.
    relative = results.get("relative_angular_momentum")
    if relative is None:
        relative = np.zeros_like(omega)
    total = j * omega + (relative if p.relative_momentum else 0.0)
    lines.append("angular momentum : %.6e kg m^2/s, drift %.2e"
                 % (results["angular_momentum"],
                    np.max(np.abs(total - results["angular_momentum"]))))
    lines.append("rotation in %.2f s: %.2f rad = %.3f turns"
                 % (results["t"][-1], theta[-1] - theta[0], turns))
    lines.append("landing angle    : %.3f rad (%.0f deg from the release "
                 "attitude)"
                 % (theta[-1] % (2 * np.pi),
                    np.degrees((theta[-1] - theta[0]) % (2 * np.pi))))
    if results.get("history_stride", 1) > 1:
        lines.append("이력 기록         : %d 스텝마다 %d 장 (메모리 예산 "
                     "안에서 그림을 그리려고 솎아냄)"
                     % (results["history_stride"],
                        results["history_steps"].size))
    lines.append(format_landing(results))
    return "\n".join(lines)


def report(results, show_inputs=True):
    """:func:`format_report` 를 출력한다.

    Print :func:`format_report`.
    """
    print(format_report(results, show_inputs))


# ===========================================================================
# 연구용 스캔 — the measurements the research question actually needs
# ===========================================================================
# 이 연구의 물음은 두 가지다.
#
# 1. 충전율이 얼마일 때 가장 잘 세워지는가 (30~40% 인가)?
# 2. 그 답이 액체의 **종류나 질량** 과 무관하고 **부피 비율** 에만 달렸는가?
#
# 한 번의 던지기로는 어느 쪽도 답할 수 없다. 던지는 세기를 조금만 바꿔도
# 병은 서기도 하고 넘어지기도 하기 때문이다.  실제로 "잘 세워진다" 는 말은
# **성공하는 던지기의 범위가 넓다** 는 뜻이다 — 그것이 아래 :func:`tolerance_scan`
# 이 재는 양이다.
#
# 2 번에 대해서는 미리 알아 둘 것이 있다. 자유비행 중 물 요소의 운동방정식
#
#     r'' = omega**2 (r - r_cm) - (2/tau) r'
#
# 에는 질량도 밀도도 들어 있지 않다.  각속도는 ``omega = L / J`` 에서 나오는데
# 물만 있다면 ``L`` 과 ``J`` 가 똑같이 질량에 비례해서 약분된다.  그러므로
# 액체의 밀도와 질량이 결과에 들어오는 통로는 **딱 두 개** 뿐이다.
#
# * ``epsilon = m_bottle / (m_bottle + m_liquid)`` — 빈 병이 전체 질량에서
#   차지하는 몫. 병이 가벼울수록 0 에 가까워지고, 그때 비로소 "액체의 종류와
#   무관" 이 성립한다.
# * ``tau`` — 점성이 정하는 벽 항력의 감쇠 시간. 물처럼 묽은 액체에서는 비행
#   시간보다 훨씬 길어 거의 관여하지 않지만, 꿀이라면 이야기가 다르다.
#
# 즉 가설은 "무관하다" 가 아니라 "**epsilon 과 tau 를 통해서만 관계된다**" 가
# 맞다.  :func:`substance_scan` 이 그것을 직접 확인한다: 충전율을 고정한 채
# 밀도만 바꾸면 결과가 얼마나 움직이는가, 그리고 병을 가볍게 하면 그 차이가
# 사라지는가.

def _throw_succeeds(params, omega_0, model=None, **overrides):
    """던지기 하나를 계산해 섰는지만 돌려준다.

    Run one throw and report only whether the bottle stands.
    """
    p = params.replace(omega_0=omega_0, **overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            results = run(p) if model is None else run(p.replace(model=model))
        except ValueError:
            return None
    return landing_outcome(results)


def _flight_of(params, omega_0, **overrides):
    """던지기 하나를 계산한다 (경고는 삼킨다).

    Run one throw, swallowing the warnings.
    """
    case = params.replace(omega_0=float(omega_0), **overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return run(case)


def turns_of(params, omega_0, **overrides):
    """이 던지기가 비행 동안 도는 바퀴 수.

    How many turns this throw makes in flight.

    ``omega_0`` 에 대해 단조증가한다 — 그래서 "정확히 한 바퀴 도는 던지기" 를
    이분법으로 찾을 수 있다.
    """
    results = _flight_of(params, omega_0, **overrides)
    return float((results["theta"][-1] - results["theta"][0]) / (2 * np.pi))


def throw_for_turns(params, target=1.0, bounds=(2.0, 90.0), tol=1e-3,
                    **overrides):
    """딱 ``target`` 바퀴를 도는 초기 각속도를 찾는다 (이분법).

    The initial spin that makes exactly ``target`` turns, by bisection.

    Returns ``None`` if no throw in ``bounds`` turns that much.
    """
    low, high = bounds
    f_low = turns_of(params, low, **overrides) - target
    f_high = turns_of(params, high, **overrides) - target
    if f_low > 0 or f_high < 0:
        return None
    while high - low > tol:
        mid = 0.5 * (low + high)
        if turns_of(params, mid, **overrides) - target < 0:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def standing_window(params, target=1.0, bounds=(2.0, 90.0), tol=1e-3,
                    **overrides):
    """세워지는 초기 각속도의 구간을 정확히 잰다.

    The interval of initial spins that stand, measured exactly.

    격자를 촘촘히 훑는 대신, 먼저 ``target`` 바퀴를 도는 던지기를 이분법으로
    찾고 (거기서 병은 똑바로 선 채로 닿는다), 거기서 양쪽으로 벌려 가며
    성패가 갈리는 경계를 다시 이분법으로 찾는다.  격자 간격에 답이 좌우되지
    않으므로, 충전율끼리 견주는 데 쓸 수 있는 수가 나온다.

    ``width`` 가 곧 "손이 이만큼 흔들려도 세워진다" 는 폭 [rad/s] 이다.
    """
    center = throw_for_turns(params, target, bounds, tol, **overrides)
    if center is None:
        return {"center": None, "low": None, "high": None, "width": 0.0}

    # 세워지는 구간은 "딱 한 바퀴" 던지기에 **중심을 두지 않는다**.  넘어가는
    # 한계 기울기가 10~15 도쯤이므로 자세만 보면 한 바퀴에서 ±0.04 바퀴까지
    # 괜찮지만, 그 안에서도 남은 회전이 느린 쪽 — 즉 omega_0 이 조금 작은
    # 쪽 — 에서만 실제로 선다.  그래서 자세가 허용되는 구간 전체를 감싸는
    # 범위를 잡고 그 안을 훑는다.
    span = []
    for offset in (-0.10, 0.10):
        edge_throw = throw_for_turns(params, target + offset, bounds, tol,
                                     **overrides)
        span.append(edge_throw)
    low_bound = span[0] if span[0] is not None else bounds[0]
    high_bound = span[1] if span[1] is not None else bounds[1]
    if high_bound <= low_bound:
        return {"center": center, "low": None, "high": None, "width": 0.0}

    def stands(omega_0):
        return landing_outcome(
            _flight_of(params, omega_0, **overrides))["stands"]

    samples = np.linspace(low_bound, high_bound, 41)
    hits = [w for w in samples if stands(w)]
    if not hits:
        return {"center": center, "low": None, "high": None, "width": 0.0}

    def edge(good, bad):
        """성공과 실패 사이의 경계를 이분법으로."""
        while abs(bad - good) > tol:
            mid = 0.5 * (good + bad)
            if stands(mid):
                good = mid
            else:
                bad = mid
        return good

    step = samples[1] - samples[0]
    low = edge(hits[0], max(hits[0] - step, low_bound - step))
    high = edge(hits[-1], min(hits[-1] + step, high_bound + step))
    return {"center": center, "low": low, "high": high, "width": high - low}


def tolerance_scan(params=None, fractions=None, speeds=None, target=1.0,
                   progress=None, **overrides):
    """충전율마다, 성공하는 던지기의 폭이 얼마나 넓은지 잰다.

    For each filling fraction, how wide the range of throws that stand is.

    충전율마다 :func:`standing_window` 로 **세워지는 초기 각속도의 폭** 을
    이분법으로 정확히 재고, 던져 올리는 속도(곧 비행 시간)도 몇 가지로 바꿔
    가며 되풀이한다.

    이것이 "잘 세워진다" 의 조작적 정의다.  사람은 매번 똑같이 던지지 못하므로,
    좋은 충전율이란 어떤 한 번의 완벽한 던지기가 성공하는 충전율이 아니라
    **손이 흔들려도 성공하는 폭이 넓은** 충전율이다.  격자를 훑는 대신 경계를
    이분법으로 찾으므로, 이 폭은 격자 간격에 좌우되지 않는다.

    ``width`` 는 rad/s 단위의 절대 폭이고, ``relative`` 는 그것을 알맞은 던지기
    세기로 나눈 것이다 — 사람의 던지기 오차는 절대값보다 비율에 가깝다.
    """
    p = resolve_parameters(params, **overrides)
    if fractions is None:
        fractions = np.round(np.arange(0.05, 0.76, 0.05), 3)
    fractions = np.asarray(fractions, dtype=float)
    if speeds is None:
        speeds = np.array([2.5, 3.0, 3.5, 4.0])
    speeds = np.atleast_1d(np.asarray(speeds, dtype=float))

    width = np.zeros((fractions.size, speeds.size))
    center = np.full((fractions.size, speeds.size), np.nan)
    slowdown = np.full(fractions.size, np.nan)

    step = 0
    total_steps = fractions.size * speeds.size
    for i, fraction in enumerate(fractions):
        case = p.replace(water_mass=fraction * p.water_mass_max)
        for a, speed in enumerate(speeds):
            found = standing_window(case, target=target,
                                    launch_speed=float(speed))
            width[i, a] = found["width"]
            if found["center"] is not None:
                center[i, a] = found["center"]
            step += 1
            if progress is not None:
                progress(step, total_steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reference = run(case)
        slowdown[i] = reference["omega"][-1] / reference["omega"][0]

    best_width = width.max(axis=1)
    d_speed = float(np.mean(np.diff(speeds))) if speeds.size > 1 else 1.0
    basin = width.sum(axis=1) * d_speed
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.nanmax(np.where(width > 0, width / center, np.nan),
                             axis=1) if speeds.size else np.zeros_like(width)
    relative = np.nan_to_num(relative)

    return {"parameters": p, "fractions": fractions, "speeds": speeds,
            "width": width, "center": center, "best_width": best_width,
            "relative": relative, "basin": basin, "slowdown": slowdown,
            "target": target,
            "best_fraction": float(fractions[int(np.argmax(basin))])}


def format_tolerance(scan):
    """:func:`tolerance_scan` 의 결과를 표로.

    The tolerance scan as a table.
    """
    p = scan["parameters"]
    lines = ["충전율별 성공 범위 / how forgiving each filling fraction is",
             "병 %.0f mL, %.1f 바퀴 던지기, 던짐 속도 %s m/s"
             % (1e6 * p.bottle_volume, scan["target"],
                ", ".join("%.1f" % s for s in scan["speeds"])),
             "물의 결합도 water_coupling = %.2f (착지 모델의 유일한 가정)"
             % p.water_coupling,
             "",
             "%8s %9s %11s %10s %11s %10s"
             % ("충전율", "액체[kg]", "성공 폭", "상대 폭", "알맞은 ω",
                "ω 감속")]
    for i, fraction in enumerate(scan["fractions"]):
        centers = scan["center"][i]
        nominal = np.nanmean(centers) if np.any(np.isfinite(centers)) else np.nan
        lines.append("%8.2f %9.3f %9.3f   %8.2f%% %10.1f %10.2fx"
                     % (fraction, fraction * p.water_mass_max,
                        scan["best_width"][i], 100 * scan["relative"][i],
                        nominal, scan["slowdown"][i]))
    lines.append("")
    lines.append("성공 폭 [rad/s] = 병이 서는 초기 각속도의 폭. 이분법으로 "
                 "경계를 찾으므로 격자 간격에 좌우되지 않는다.")
    lines.append("가장 넓은 충전율 / widest at: %.2f" % scan["best_fraction"])
    good = scan["fractions"][scan["basin"] >= 0.5 * scan["basin"].max()]
    if good.size:
        lines.append("최대의 절반 이상 / at least half the best: %.2f ~ %.2f"
                     % (good.min(), good.max()))
    lines.append("")
    lines.append("주의: 거의 가득 찬 병일수록 물이 움직일 빈 공간이 없어 "
                 "실제로는 굳은 물체에 가까워진다 (water_coupling 이 1 에 "
                 "가까워진다). 이 계산은 결합도를 일정하게 두므로 높은 "
                 "충전율 쪽을 실제보다 후하게 본다 — 참 최적값은 여기 나온 "
                 "값이거나 그보다 조금 낮다.")
    return "\n".join(lines)


def coupling_sensitivity(params=None, couplings=(0.0, 0.1, 0.2, 0.35, 0.6),
                         **kwargs):
    """착지 모델의 유일한 가정을 흔들어 결론이 움직이는지 본다.

    Vary the one fitted assumption of the landing model and see if the answer moves.

    ``water_coupling`` 은 계산이 아니라 모델의 선택이다.  그러므로 결론을
    말하기 전에, 그 값을 바꿔도 **최적 충전율이 그대로인지** 확인해야 한다.
    성공률의 절대값은 당연히 달라지지만, 최적점이 움직이지 않는다면 결론은
    이 가정에 기대고 있지 않다.
    """
    p = resolve_parameters(params)
    rows = []
    for coupling in couplings:
        scan = tolerance_scan(p.replace(water_coupling=coupling), **kwargs)
        rows.append({"coupling": coupling,
                     "best_fraction": scan["best_fraction"],
                     "basin": scan["basin"],
                     "fractions": scan["fractions"]})
    return {"rows": rows}


def format_coupling_sensitivity(sensitivity):
    """:func:`coupling_sensitivity` 의 결과를 표로."""
    lines = ["착지 가정에 대한 민감도 / sensitivity to the landing assumption",
             "", "%14s %16s   %s" % ("water_coupling", "최적 충전율",
                                     "충전율별 성공 폭 (최대=100)")]
    for row in sensitivity["rows"]:
        peak = max(row["basin"].max(), 1e-30)
        shape = " ".join("%3.0f" % (100 * b / peak) for b in row["basin"])
        lines.append("%14.2f %16.2f   %s" % (row["coupling"],
                                             row["best_fraction"], shape))
    bests = [row["best_fraction"] for row in sensitivity["rows"]]
    lines.append("")
    lines.append("최적 충전율의 범위 / the optimum moves between %.2f and %.2f"
                 % (min(bests), max(bests)))
    lines.append("이 폭이 좁으면, 결론은 water_coupling 이라는 가정에 기대고 "
                 "있지 않다는 뜻이다.")
    return "\n".join(lines)


def substance_scan(params=None, fraction=0.35, densities=None,
                   viscosities=None, light_bottle=True, **overrides):
    """충전율을 고정한 채 액체의 종류(밀도·점성)만 바꾼다.

    Hold the filling fraction and change only what the liquid is.

    "물질의 종류나 질량과 무관하고 부피 비율에만 관계된다" 는 가설을 그대로
    검사한다.  같은 부피 비율에서 밀도를 바꾸면 액체의 질량이 달라지므로
    ``epsilon`` 이 달라진다 — 결과가 움직인다면 그 통로로 움직인 것이다.
    ``light_bottle`` 을 켜면 병을 아주 가볍게 한 경우도 함께 계산하는데,
    거기서는 ``epsilon`` 이 어느 액체에서나 0 에 가까우므로 가설대로 결과가
    한 점에 모여야 한다.  모이면 가설이 맞고, 갈라지면 통로가 하나 더 있다.
    """
    p = resolve_parameters(params, **overrides)
    if densities is None:
        densities = [(600.0, "가벼운 기름 / light oil"),
                     (789.0, "에탄올 / ethanol"),
                     (1000.0, "물 / water"),
                     (1260.0, "글리세린 / glycerol"),
                     (1600.0, "모래 / dry sand")]
    if viscosities is None:
        viscosities = {}

    rows = []
    for entry in densities:
        density, label = entry if isinstance(entry, tuple) else (entry, "")
        case = p.replace(water_density=density,
                         kinematic_viscosity=viscosities.get(
                             density, p.kinematic_viscosity))
        case = case.replace(water_mass=fraction * case.water_mass_max)
        variants = [("보통 병 / real bottle", case)]
        if light_bottle:
            variants.append(("가벼운 병 / massless bottle",
                             case.replace(bottle_mass=1e-4)))
        for variant, cfg in variants:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = run(cfg)
            out = landing_outcome(results)
            rows.append({"density": density, "label": label,
                         "variant": variant, "mass": cfg.water_mass,
                         "epsilon": cfg.epsilon, "tau": cfg.tau,
                         "turns": out["turns"], "tilt": out["tilt_deg"],
                         "slowdown": results["omega"][-1] / results["omega"][0],
                         "stands": out["stands"]})
    return {"parameters": p, "fraction": fraction, "rows": rows}


def format_substance(scan):
    """:func:`substance_scan` 의 결과를 표로.

    The substance scan as a table.
    """
    lines = ["같은 충전율 %.2f, 액체만 바꿈 / same filling fraction, "
             "different liquid" % scan["fraction"], "",
             "%-26s %8s %8s %8s %8s %8s"
             % ("액체 / liquid", "밀도", "질량[kg]", "epsilon", "회전수",
                "감속")]
    for variant in dict.fromkeys(row["variant"] for row in scan["rows"]):
        lines.append("[%s]" % variant)
        turns = []
        for row in scan["rows"]:
            if row["variant"] != variant:
                continue
            turns.append(row["turns"])
            lines.append("  %-24s %8.0f %8.3f %8.3f %8.3f %7.2fx"
                         % (row["label"] or "-", row["density"], row["mass"],
                            row["epsilon"], row["turns"], row["slowdown"]))
        if turns:
            spread = max(turns) - min(turns)
            lines.append("  -> 회전수 편차 / spread in turns: %.4f (%.1f%%)"
                         % (spread, 100 * spread / max(np.mean(turns), 1e-12)))
    lines.append("")
    lines.append("액체의 종류가 결과에 들어오는 통로는 epsilon 과 tau 뿐이다. "
                 "병을 가볍게 하면 epsilon 이 사라지고, 그러면 밀도가 달라도 "
                 "결과가 한 점에 모인다 — 그것이 가설이 참인 조건이다.")
    return "\n".join(lines)


def pyplot(show=False):
    """지금 있는 곳에 맞는 백엔드로 ``matplotlib.pyplot`` 을 가져온다.

    ``matplotlib.pyplot``, with a backend that suits where we are.

    A headless script needs one that has no display; inside a notebook the
    backend is already the right one, and switching it would silence every
    later figure.
    """
    import matplotlib
    if (not show and not in_notebook()
            and "matplotlib.pyplot" not in sys.modules):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _finish(figure_, plt, path, show):
    if path:
        figure_.savefig(path, dpi=150)
        print("figure written to %s" % path)
    if show:
        plt.show()
    plt.close(figure_)


def figure_slices(results, plt=None):
    """원판 모델의 시간 이력 그림 — 그림 객체를 그대로 돌려준다.

    Time histories of the disc model, as a figure the caller owns.
    """
    plt = plt or pyplot()
    p = results["parameters"]
    t = results["t"]
    positions = results["slices_positions"]
    # 이력이 솎아졌으면 위치는 그 스텝의 시각에 대응한다
    history_t = results.get("history_t", t)
    n_slices = positions.shape[1]

    fig, ax = plt.subplots(4, 1, figsize=(6, 11), sharex=True)

    ax[0].plot(t, results["theta"] / (2 * np.pi))
    ax[0].set_ylabel("Rotation [turns]")

    ax[1].plot(t, results["center_of_mass"], "--", lw=2,
               label="centre of mass")
    ax[1].plot(history_t, positions[:, 0], "k", lw=0.9,
               label="slice at the cap")
    ax[1].plot(history_t, positions[:, n_slices // 2], "k", lw=0.9, alpha=0.6,
               label="middle slice")
    ax[1].plot(history_t, positions[:, -1], "k", lw=0.9, alpha=0.3,
               label="slice at the free surface")
    ax[1].axhline(p.bottle_height, color="0.7", lw=0.8)
    ax[1].axhline(0.0, color="0.7", lw=0.8)
    ax[1].set_ylabel("Axial position [m]")
    ax[1].set_ylim(-0.02, 1.1 * p.bottle_height)
    ax[1].legend(fontsize=8, loc="lower left")

    ax[2].plot(t, results["rotational_inertia"])
    ax[2].set_ylabel(r"$J$ [kg m$^2$]")

    ax[3].plot(t, results["omega"])
    ax[3].set_ylabel(r"$\omega$ [rad/s]")
    ax[3].set_xlabel("Time [s]")
    ax[3].set_ylim(0, 1.1 * results["omega"].max())

    fig.tight_layout()
    return fig


def figure_parcels(results, plt=None):
    """회전면에서 본 물의 스냅샷과 시간 이력.

    Snapshots of the water in the flip plane plus time histories.
    """
    plt = plt or pyplot()
    p = results["parameters"]
    t = results["t"]
    y, z = results["parcels_y"], results["parcels_z"]
    # 이력을 솎아냈을 수 있으므로, 남아 있는 기록 가운데 네 장을 고른다
    history_steps = results.get("history_steps", np.arange(t.size))
    rows = np.unique(np.linspace(0, y.shape[0] - 1, 4).astype(int))
    # 조각이 많으면 점을 작게 그려야 서로 가려지지 않는다
    marker = 14 if y.shape[1] <= 400 else max(2, int(4000 / y.shape[1]))

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(3, rows.size, height_ratios=[1.6, 1, 1])

    radius = p.bottle_radius
    for column, row in enumerate(rows):
        index = int(history_steps[row])
        ax = fig.add_subplot(grid[0, column])
        ax.add_patch(plt.Rectangle((-radius, 0), 2 * radius, p.bottle_height,
                                   fill=False, edgecolor="0.4", lw=1.2))
        ax.scatter(y[row], z[row], s=marker, c="tab:blue", alpha=0.75,
                   linewidths=0)
        ax.axhline(results["center_of_mass"][index], color="tab:red", lw=1,
                   ls="--")
        ax.set_xlim(-1.6 * radius, 1.6 * radius)
        ax.set_ylim(-0.01, p.bottle_height + 0.01)
        ax.set_aspect("equal")
        ax.set_title("t = %.2f s\n$\\omega$ = %.1f rad/s"
                     % (t[index], results["omega"][index]), fontsize=9)
        ax.set_xlabel("y [m]", fontsize=8)
        if column == 0:
            ax.set_ylabel("z [m]", fontsize=8)
        ax.tick_params(labelsize=7)

    ax_j = fig.add_subplot(grid[1, :])
    ax_j.plot(t, results["rotational_inertia"])
    ax_j.set_ylabel(r"$J$ [kg m$^2$]")
    ax_j.tick_params(labelbottom=False)

    ax_w = fig.add_subplot(grid[2, :], sharex=ax_j)
    ax_w.plot(t, results["omega"])
    ax_w.set_ylabel(r"$\omega$ [rad/s]")
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylim(0, 1.1 * results["omega"].max())

    fig.tight_layout()
    return fig


def figure_beads(results, plt=None):
    """회전면에서 본 구슬의 스냅샷과 시간 이력.

    Snapshots of the beads in the flip plane plus time histories.
    """
    plt = plt or pyplot()
    p = results["parameters"]
    t = results["t"]
    radius = results["bead_radius"]
    indices = sorted(results["snapshots"])[:4]

    fig = plt.figure(figsize=(11, 8))
    grid = fig.add_gridspec(3, len(indices), height_ratios=[1.6, 1, 1])

    for column, index in enumerate(indices):
        positions = results["snapshots"][index]
        ax = fig.add_subplot(grid[0, column])
        ax.add_patch(plt.Rectangle((-p.bottle_radius, 0), 2 * p.bottle_radius,
                                   p.bottle_height, fill=False,
                                   edgecolor="0.4", lw=1.2))
        depth = np.abs(positions[:, 0]) / max(p.bottle_radius, 1e-9)
        if positions.shape[0] <= BEAD_CIRCLE_LIMIT:
            # 실제 크기로 그린다: 구슬은 이 반지름의 구다
            for (_, y, z), shade in zip(positions, depth):
                ax.add_patch(plt.Circle((y, z), radius, facecolor="tab:blue",
                                        alpha=float(0.85 - 0.5
                                                    * min(shade, 1.0)),
                                        edgecolor="none"))
        else:
            # 구슬이 수천 개면 원 하나하나를 그리는 데 몇 분이 걸린다. 그때는
            # 점으로 흩뿌린다. 점 크기는 실제 지름에 맞춘다: 이 칸은 폭이 약
            # 2.2 인치이고 x 범위가 3.2 R 이므로 지름 2a 는
            # (2a / 3.2R) * 2.2 * 72 포인트이고, scatter 의 s 는 그 제곱이다.
            points = 49.5 * 2 * radius / p.bottle_radius
            ax.scatter(positions[:, 1], positions[:, 2],
                       s=max(1.0, points ** 2), c="tab:blue", alpha=0.6,
                       linewidths=0)
        ax.axhline(results["center_of_mass"][index], color="tab:red", lw=1,
                   ls="--")
        ax.set_xlim(-1.6 * p.bottle_radius, 1.6 * p.bottle_radius)
        ax.set_ylim(-0.01, p.bottle_height + 0.01)
        ax.set_aspect("equal")
        ax.set_title("t = %.2f s\n$\\omega$ = %.1f rad/s"
                     % (t[index], results["omega"][index]), fontsize=9)
        ax.set_xlabel("y [m]", fontsize=8)
        if column == 0:
            ax.set_ylabel("z [m]", fontsize=8)
        ax.tick_params(labelsize=7)

    ax_j = fig.add_subplot(grid[1, :])
    ax_j.plot(t, results["rotational_inertia"])
    ax_j.set_ylabel(r"$J$ [kg m$^2$]")
    ax_j.tick_params(labelbottom=False)

    ax_w = fig.add_subplot(grid[2, :], sharex=ax_j)
    ax_w.plot(t, results["omega"])
    ax_w.set_ylabel(r"$\omega$ [rad/s]")
    ax_w.set_xlabel("Time [s]")
    ax_w.set_ylim(0, 1.1 * results["omega"].max())

    fig.tight_layout()
    return fig


def figure_tolerance(scan, plt=None):
    """충전율별 성공률과, 성공하는 던지기의 지도.

    Success rate against filling fraction, and the map of throws that stand.
    """
    if plt is None:
        plt = pyplot()
    fig, (ax_rate, ax_map) = plt.subplots(1, 2, figsize=(11, 4.2))

    fractions = scan["fractions"]
    ax_rate.plot(fractions, scan["best_width"], "o-", color="#1f77b4")
    ax_rate.axvspan(0.30, 0.40, color="#ff7f0e", alpha=0.15,
                    label="0.30 - 0.40")
    ax_rate.axvline(scan["best_fraction"], color="#d62728", lw=1,
                    ls="--", label="widest %.2f" % scan["best_fraction"])
    ax_rate.set_xlabel("filling fraction")
    ax_rate.set_ylabel(r"width of throws that stand [rad/s]")
    ax_rate.set_ylim(0, None)
    ax_rate.legend(fontsize=8)
    ax_rate.grid(alpha=0.3)

    # 던짐 속도(=비행 시간)마다 폭이 어떻게 달라지는지
    for a, speed in enumerate(scan["speeds"]):
        ax_map.plot(fractions, scan["width"][:, a], "o-", ms=3,
                    label="%.1f m/s" % speed)
    ax_map.set_xlabel("filling fraction")
    ax_map.set_ylabel(r"width of throws that stand [rad/s]")
    ax_map.set_ylim(0, None)
    ax_map.legend(fontsize=8, title="launch speed")
    ax_map.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def figure(results, plt=None):
    """그 계산의 모델에 맞는 그림을, 간직할 호출자에게 돌려준다.

    The figure that suits a run's model, for a caller that keeps it.
    """
    builders = {"parcels": figure_parcels, "beads": figure_beads}
    builder = builders.get(results["model"], figure_slices)
    return builder(results, plt)


def plot(results, path=None, show=True):
    """원판 그림을 보여 주거나 저장하고 닫는다.

    Show and/or save the disc figure, then release it.
    """
    plt = pyplot(show)
    _finish(figure_slices(results, plt), plt, path, show)


def plot_parcels(results, path=None, show=True):
    """입자 그림을 보여 주거나 저장하고 닫는다.

    Show and/or save the parcel figure, then release it.
    """
    plt = pyplot(show)
    _finish(figure_parcels(results, plt), plt, path, show)


def draw(results, path=None, show=True):
    """모델에 맞는 그림으로 계산 결과를 그린다.

    Plot a run with whichever figure suits its model.
    """
    plt = pyplot(show)
    _finish(figure(results, plt), plt, path, show)


def run_interactive(base=None, ask=input, echo=print, show=True, path=None,
                    progress=None):
    """물리량을 하나씩 묻고, 던지고, 결과를 보여 준다.

    Ask for every physical quantity, run the flip and show the result.
    """
    params = prompt_parameters(base, ask=ask, echo=echo)
    results = run(params, progress=progress)
    report(results)
    if show or path:
        draw(results, path=path, show=show)
    return results


def build_parser():
    """명령행 인터페이스: 물리량마다 옵션 하나.

    The command line interface: one option per physical quantity.
    """
    parser = argparse.ArgumentParser(
        prog="물병던지기_시뮤레이션.py",
        description="물병던지기 시뮤레이션 — water bottle flip simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_arguments(parser)
    output = parser.add_argument_group("출력 / output")
    output.add_argument("-i", "--interactive", action="store_true",
                        help="ask for every physical quantity at the prompt")
    output.add_argument("--ui", action="store_true",
                        help="open the Gradio interface in a browser instead "
                             "of running once (needs gradio)")
    output.add_argument("--share", action="store_true",
                        help="with --ui, publish a temporary public link")
    output.add_argument("--describe", action="store_true",
                        help="print the parameters and exit without running")
    output.add_argument("--save-figure", metavar="PATH", default=None)
    output.add_argument("--save-data", metavar="PATH", default=None)
    output.add_argument("--no-plot", action="store_true")
    output.add_argument("--quiet", action="store_true",
                        help="진행률 표시를 끈다 / do not report progress "
                             "while integrating")
    return parser


def _launch_ui(**kwargs):
    """Gradio 화면을 띄운다 — 필요할 때만 가져온다.

    Start the Gradio front end, which is only imported when it is wanted.
    """
    return launch_ui(**kwargs)


def _install_gradio():
    """설치가 일상이고 지워도 되는 곳(Colab)에서 gradio 를 설치한다.

    Install gradio, where installing things is routine and disposable.
    """
    import subprocess

    print("gradio 를 설치합니다 / installing gradio, this takes a moment ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"],
                   check=False)


def start_notebook_session():
    """이 파일을 노트북 셀에 붙여넣었을 때 사람이 보게 되는 것.

    What a human gets when this file is pasted into a notebook cell.

    The graphical interface, so there is something to click rather than a
    list of function names — installing gradio first in Colab, where that is
    routine.  If it cannot be had, the defaults are simulated once, so the
    cell always leaves a figure and a summary behind.  Returns the interface,
    or the results of that run.
    """
    if AUTO_LAUNCH_UI:
        try:
            print("UI 를 엽니다 — 왼쪽에서 물리량을 조절하고 「실행 / Run」 을 "
                  "누르세요.")
            return _launch_ui()
        except ImportError:
            if "google.colab" in sys.modules:
                _install_gradio()
                try:
                    return _launch_ui()
                except ImportError:
                    pass
            print(GRADIO_MISSING)
    else:
        print(NOTEBOOK_HINT)
    results = run()
    report(results)
    draw(results)
    return results


def main(argv=None):
    if argv is None and in_notebook():
        # 커널이 받은 인자는 우리 것이 아니다. 거기서 실패하지 말고 기본값
        # 으로 계산한다.
        argv = []
    args = build_parser().parse_args(argv)

    params = parameters_from_namespace(args)
    if args.ui:
        return _launch_ui(share=args.share)
    if args.describe:
        print(params.describe())
        return params

    if args.interactive:
        return run_interactive(params, show=not args.no_plot,
                               path=args.save_figure)

    # 큰 계산은 몇 분씩 걸린다. 진행률을 보여 주면 멈춘 것과 구분이 된다.
    progress = None if args.quiet else console_progress(
        "%s 모델 계산 중" % params.model)
    results = run(params, progress=progress)
    report(results)

    if args.save_data:
        data = np.column_stack([results["t"], results["center_of_mass"],
                                results["theta"], results["omega"],
                                results["rotational_inertia"]])
        np.savetxt(args.save_data, data, delimiter=" ",
                   header="t r_cm theta omega J")
        print("data written to %s" % args.save_data)

    if not args.no_plot or args.save_figure:
        draw(results, path=args.save_figure, show=not args.no_plot)
    return results


# ===========================================================================
# Gradio UI — 물리량 입력 · 버튼 · 그래프 / sliders, buttons and figures
# ===========================================================================
# 브라우저 화면 — 물리량을 넣고, 버튼을 누르고, 뒤집기를 본다.
#
# :class:`Parameters` 의 물리량마다 컨트롤이 하나씩 생긴다(숫자는 슬라이더,
# 스위치는 체크박스, 모델은 드롭다운, 낙하 높이는 숫자 상자). 콜백은 컨트롤
# 값만 받는 보통 함수이므로 브라우저 없이도 시험할 수 있고, ``gradio`` 는
# 화면을 만들 때에만 가져온다.
#
# A Gradio front end: type the physical quantities, press a button, see the flip.
#
# Every quantity of :class:`Parameters` gets a control — a slider for
# the numeric ones (with the bounds carried in the field metadata), a checkbox
# for the switches, a dropdown for the model, a number box for the optional drop
# height — and five buttons act on them:
#
# * **실행 / Run** — one flip: the figure of the selected model and a text summary;
# * **기본값 / Reset** — put every control back to its default;
# * **충전율 스캔 / Filling scan** — the same throw at five water masses, so the
#   degenerate regime and the useful 0.2-0.4 band are visible at once;
# * **항별 비교 / Force comparison** — the parcel model with the Coriolis force,
#   the Euler force and the incompressibility constraint switched off in turn;
# * **모델 비교 / Compare models** — the same throw through all three
#   discretisations of the water, which is the only way to see whether they
#   agree;
# * **CSV 저장 / Save CSV** — the time series of the current run as a file.
#
# The callbacks are plain functions of the control values, so they can be tested
# without a browser; ``gradio`` itself is imported only when the interface is
# built.  Usage::
#
#     launch_ui()                          # http://127.0.0.1:7860
#     launch_ui(share=True)                # e.g. from Colab
#
# Install the extra it needs with ``pip install -r requirements-ui.txt``.

# 컨트롤 순서, 따라서 콜백 인자 순서는 필드 목록 순서를 따른다.
UI_FIELDS = [name for _, names in FIELD_GROUPS for name in names]

INTRO = """# 물병던지기 시뮤레이션 — water bottle flip

물이 든 병을 회전시켜 자유낙하시키면, 물이 재분포하면서 관성모멘트 `J` 가 커집니다.
자유비행 중에는 각운동량 `L = J ω` 가 보존되므로 각속도 `ω` 가 떨어지고, 그래서 병이
똑바로 착지할 수 있습니다.

왼쪽에서 **물리량을 직접 입력**하고 버튼을 누르세요. 물을 보는 방식이 세 가지입니다.

* `slices` — 병 축을 따라 미끄러지는 원판. 가장 빠릅니다.
* `parcels` — 원판을 쪼개 단면 안에서도 움직이는 입자 (코리올리·오일러 항이 살아납니다).
* `beads` — **구슬 하나하나에 실제 힘을 계산**합니다: 접촉(스프링-댐퍼, 반발계수),
  구슬 사이 인력(표면장력), 항력, 관성력, 중력. 구슬의 크기·질량·개수를 직접 넣을
  수 있고, 0 으로 두면 물의 양에서 유도합니다. 가장 느립니다.

「모델 비교」 버튼으로 세 모델이 같은 답을 주는지 확인할 수 있습니다.

충전율이 0.1 아래면 물기둥 전체가 질량중심 한쪽에 놓여 재분포가 일어나지 않습니다.
0.2~0.4 에서 감속이 가장 큽니다. 스캔과 항별 비교는 여러 번 계산하므로 몇십 초가
걸릴 수 있습니다.

**큰 계산**: 구슬 수나 조각 수를 크게 올리면 오래 걸리지만 끝까지 갑니다. 병에
들어가지 않는 구슬 크기, 너무 굵은 조각, 메모리를 넘는 이력은 자동으로 조정하고
요약에 그 사실을 적습니다. 진행률은 UI 를 띄운 터미널에 표시됩니다."""

PRESETS = [
    # 병 높이, 반지름, 빈 병 질량, 물 질량, 초기 각속도, 모델, 조각 수
    ("500 mL 생수병 / default", 0.21, 0.031, 0.022, 0.17, 23.0, "slices", 19),
    ("적게 채움 15% / underfilled", 0.21, 0.031, 0.022, 0.076, 23.0, "slices",
     19),
    ("많이 채움 60% / overfilled", 0.21, 0.031, 0.022, 0.305, 23.0, "slices",
     19),
    ("단면 분해 / cross section resolved", 0.21, 0.031, 0.022, 0.17, 23.0,
     "parcels", 19),
    ("구슬 / beads (forces)", 0.21, 0.031, 0.022, 0.17, 23.0, "beads", 19),
    ("1.5 L 큰 병 / large bottle", 0.31, 0.044, 0.038, 0.50, 18.0, "slices",
     19),
]


def parameters_from_values(values):
    """위젯 값들을 검사된 :class:`Parameters` 로 바꾼다.

    Turn the control values into a validated :class:`Parameters`.

    Widgets are looser than the parameter object: a slider hands an integer
    quantity a float, and an empty number box arrives as ``None`` or as ``0``
    depending on the Gradio version.  An empty — or zero — drop height means
    "use the flight duration instead", not "fall from no height at all".
    """
    given = dict(zip(UI_FIELDS, values))
    for name, value in list(given.items()):
        kind = FIELDS[name].metadata["kind"]
        if kind == "int" and value is not None:
            given[name] = int(round(value))
        elif kind == "optional_float":
            if value is None or value == "" or float(value) <= 0:
                given[name] = None
            else:
                given[name] = float(value)
    return Parameters().replace(**given)


def default_values():
    """콜백이 보는 순서대로, 모든 컨트롤의 기본값.

    The default of every control, in the order the callbacks see them.
    """
    default = Parameters()
    return [getattr(default, name) for name in UI_FIELDS]


def _blank_figure(message, plt=None):
    plt = plt or pyplot()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True,
            fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    return fig


def _run_capturing_warnings(params):
    """한 번 던지고 경고까지 함께 돌려준다 — UI 는 경고를 직접 보여야 한다.

    구슬을 수천 개 넣으면 몇 분이 걸리므로, 진행률은 UI 를 띄운 콘솔에
    남긴다. 브라우저 쪽은 계산이 끝나면 그림과 요약이 한꺼번에 바뀐다.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        results = run(params, progress=console_progress(
            "%s 모델 계산 중" % params.model))
    notes = [str(entry.message) for entry in caught
             if issubclass(entry.category, RuntimeWarning)]
    return results, notes


def run_and_render(*values):
    """한 번 던지고 ``(그림, 요약 글)`` 을 돌려준다.

    Run one flip and return ``(figure, summary text)``.
    """
    try:
        params = parameters_from_values(values)
    except (ValueError, TypeError) as error:
        return (_blank_figure("invalid input - see the summary"),
                "입력 오류 / invalid input:\n%s" % error)

    results, notes = _run_capturing_warnings(params)
    text = format_report(results)
    for note in notes:
        text += "\n\n주의 / warning: %s" % note
    return figure(results), text


def filling_scan(*values, fractions=(0.05, 0.15, 0.25, 0.35, 0.45, 0.55,
                                     0.65)):
    """같은 던지기를 여러 충전율에서 되풀이한다.

    Repeat the same throw at several filling fractions.
    """
    try:
        params = parameters_from_values(values)
    except (ValueError, TypeError) as error:
        return (_blank_figure("invalid input - see the summary"),
                "입력 오류 / invalid input:\n%s" % error)

    plt = pyplot()
    fig, (ax_omega, ax_inertia) = plt.subplots(1, 2, figsize=(10, 4))
    lines = ["충전율 스캔 / filling fraction scan "
             "(같은 병, 같은 던지기 / same bottle and throw)", "",
             "%9s %9s %10s %11s %9s" % ("물 [kg]", "충전율", "J 증가",
                                        "최종 ω", "회전수")]
    for fraction in fractions:
        water_mass = fraction * params.water_mass_max
        try:
            run, notes = _run_capturing_warnings(
                params.replace(water_mass=water_mass))
        except ValueError as error:
            lines.append("%9.3f  건너뜀 / skipped: %s" % (water_mass, error))
            continue
        j = run["rotational_inertia"]
        turns = (run["theta"][-1] - run["theta"][0]) / (2 * np.pi)
        lines.append("%9.3f %9.3f %9.2fx %10.2f %9.3f%s"
                     % (water_mass, run["filling_fraction"], j[-1] / j[0],
                        run["omega"][-1], turns,
                        "  <- 재분포 없음 / no redistribution" if notes else ""))
        label = "%.2f kg (%.2f)" % (water_mass, run["filling_fraction"])
        ax_omega.plot(run["t"], run["omega"], label=label)
        ax_inertia.plot(run["t"], j)

    ax_omega.set_xlabel("Time [s]")
    ax_omega.set_ylabel(r"$\omega$ [rad/s]")
    ax_omega.set_ylim(0, None)
    ax_omega.legend(fontsize=8)
    ax_inertia.set_xlabel("Time [s]")
    ax_inertia.set_ylabel(r"$J$ [kg m$^2$]")
    fig.tight_layout()
    lines.append("")
    lines.append("물이 적으면 물기둥이 질량중심 한쪽에만 놓여 재분포가 일어나지 "
                 "않습니다.")
    return fig, "\n".join(lines)


def tolerance_render(*values):
    """UI 버튼용: 충전율별 성공 범위를 계산해 그림과 표로.

    UI entry point for :func:`tolerance_scan`.
    """
    try:
        params = parameters_from_values(values)
    except (ValueError, TypeError) as error:
        return (_blank_figure("invalid input - see the summary"),
                "입력 오류 / invalid input:\n%s" % error)
    # 화면에서 누르는 것이므로 격자를 성기게 잡는다. 연구용으로 촘촘히
    # 훑으려면 코드에서 tolerance_scan(omega=..., speeds=...) 을 직접 부른다.
    scan = tolerance_scan(params.replace(model="slices", n_slices=60,
                                         n_steps=200),
                          fractions=np.round(np.arange(0.05, 0.76, 0.05), 3),
                          speeds=np.array([2.5, 3.0, 3.5, 4.0]))
    return figure_tolerance(scan), format_tolerance(scan)


def substance_render(*values):
    """UI 버튼용: 같은 충전율에서 액체만 바꿔 본다.

    UI entry point for :func:`substance_scan`.
    """
    try:
        params = parameters_from_values(values)
    except (ValueError, TypeError) as error:
        return (_blank_figure("invalid input - see the summary"),
                "입력 오류 / invalid input:\n%s" % error)
    fraction = params.filling_fraction
    scan = substance_scan(params.replace(model="slices"), fraction=fraction)

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    for variant in dict.fromkeys(row["variant"] for row in scan["rows"]):
        rows = [row for row in scan["rows"] if row["variant"] == variant]
        ax.plot([row["density"] for row in rows],
                [row["turns"] for row in rows], "o-",
                label="real bottle" if "보통" in variant else "massless bottle")
    ax.set_xlabel(r"liquid density [kg/m$^3$]")
    ax.set_ylabel("turns in flight")
    ax.set_title("same filling fraction %.2f, different liquid" % fraction)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, format_substance(scan)


def force_comparison(*values):
    """코리올리 항, 오일러 항, 비압축성을 하나씩 꺼 본다.

    Switch off the Coriolis force, the Euler force and incompressibility.
    """
    try:
        params = parameters_from_values(values).replace(model="parcels")
    except (ValueError, TypeError) as error:
        return (_blank_figure("invalid input - see the summary"),
                "입력 오류 / invalid input:\n%s" % error)

    # matplotlib 에는 (Colab 에도) 한글 글꼴이 없다. 그래서 곡선 이름만
    # 영어로 달고, 표는 두 언어를 함께 쓴다.
    cases = [("전체 / all forces", "all forces", {}),
             ("코리올리 제거 / no Coriolis", "no Coriolis", {"coriolis": False}),
             ("오일러 제거 / no Euler", "no Euler", {"euler": False}),
             ("둘 다 제거 / neither", "neither",
              {"coriolis": False, "euler": False}),
             ("비압축성 제거 / compressible", "compressible",
              {"incompressible": False})]

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(7, 4))
    lines = ["항별 비교 / contribution of each term (입자 모델 / parcel model)",
             "", "%-34s %10s %9s %8s" % ("설정", "최종 ω", "J 증가", "차이")]
    reference = None
    for label, curve_label, overrides in cases:
        run = simulate_parcels(params.replace(**overrides))
        omega_end = run["omega"][-1]
        j = run["rotational_inertia"]
        if reference is None:
            reference = omega_end
            change = "-"
        else:
            change = "%+.1f%%" % (100 * (omega_end / reference - 1))
        lines.append("%-34s %10.2f %8.2fx %8s"
                     % (label, omega_end, j[-1] / j[0], change))
        ax.plot(run["t"], run["omega"], label=curve_label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\omega$ [rad/s]")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig, "\n".join(lines)


def compare_models(*values):
    """같은 던지기를 세 모델로 모두 돌려 견주어 본다.

    Run the same throw through all three models and compare them.

    What is comparable is how much the moment of inertia grows and where the
    rotation ends up.  Its absolute value is not: each model starts from a
    different discretisation of the same water, so ``J`` at rest differs.
    """
    try:
        params = parameters_from_values(values)
    except (ValueError, TypeError) as error:
        return (_blank_figure("invalid input - see the summary"),
                "입력 오류 / invalid input:\n%s" % error)

    plt = pyplot()
    fig, (ax_omega, ax_inertia) = plt.subplots(1, 2, figsize=(10, 4))
    lines = ["모델 비교 / the same throw through all three models", "",
             "%-26s %9s %9s %10s %9s" % ("모델", "J 처음", "J 증가", "최종 ω",
                                         "회전수")]
    labels = {"slices": "원판 / discs", "parcels": "입자 / parcels",
              "beads": "구슬 / beads"}
    for model, curve in (("slices", "discs"), ("parcels", "parcels"),
                         ("beads", "beads")):
        run, notes = _run_capturing_warnings(params.replace(model=model))
        j = run["rotational_inertia"]
        turns = (run["theta"][-1] - run["theta"][0]) / (2 * np.pi)
        lines.append("%-26s %9.2e %8.2fx %10.2f %9.3f%s"
                     % (labels[model], j[0], j[-1] / j[0], run["omega"][-1],
                        turns,
                        "  <- 재분포 없음 / no redistribution" if notes else ""))
        ax_omega.plot(run["t"], run["omega"], label=curve)
        ax_inertia.plot(run["t"], j / j[0], label=curve)

    ax_omega.set_xlabel("Time [s]")
    ax_omega.set_ylabel(r"$\omega$ [rad/s]")
    ax_omega.set_ylim(0, None)
    ax_omega.legend(fontsize=8)
    ax_inertia.set_xlabel("Time [s]")
    ax_inertia.set_ylabel(r"$J / J_0$")
    ax_inertia.legend(fontsize=8)
    fig.tight_layout()

    lines.append("")
    lines.append("원판 모델은 단면 안의 물이 움직이지 못하므로 감속을 과대평가합니다.")
    lines.append("입자·구슬 모델은 서로 다른 방식으로 같은 물리를 풀며, J 증가율과")
    lines.append("최종 각속도가 몇 % 안에서 일치해야 합니다. J 의 절대값은 초기")
    lines.append("이산화가 다르므로 같지 않습니다.")
    return fig, "\n".join(lines)


def save_csv(*values):
    """지금 계산의 시간 이력을 파일로 쓰고 그 경로를 돌려준다.

    Write the time series of the current run and return the file path.
    """
    try:
        params = parameters_from_values(values)
    except (ValueError, TypeError):
        return None
    results = run(params)
    data = np.column_stack([results["t"], results["center_of_mass"],
                            results["theta"], results["omega"],
                            results["rotational_inertia"]])
    handle, path = tempfile.mkstemp(prefix="bottle_flip_", suffix=".csv")
    os.close(handle)
    np.savetxt(path, data, delimiter=",",
               header="t,r_cm,theta,omega,J", comments="")
    return path


def _control(name, gr):
    """필드 메타데이터에서 물리량 하나의 위젯을 만든다.

    The widget for one parameter, from its field metadata.
    """
    field = FIELDS[name]
    metadata = field.metadata
    label = metadata["label"]
    if metadata["unit"]:
        label += " [%s]" % metadata["unit"]
    default = getattr(Parameters(), name)
    kind = metadata["kind"]
    if kind == "bool":
        return gr.Checkbox(value=default, label=label, info=metadata["help"])
    if kind == "choice":
        return gr.Dropdown(choices=list(metadata["choices"]), value=default,
                           label=label, info=metadata["help"])
    if kind == "optional_float":
        return gr.Number(value=default, label=label + " — 비우면 사용 안 함",
                         info=metadata["help"])
    minimum, maximum, step = metadata["span"]
    return gr.Slider(minimum=minimum, maximum=maximum, step=step,
                     value=default, label=label, info=metadata["help"])


def build_interface():
    """Gradio 화면을 조립한다 (``gradio`` 를 가져온다).

    Assemble the Gradio interface (imports ``gradio``).
    """
    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "the interface needs gradio: pip install -r requirements-ui.txt"
        ) from error

    with gr.Blocks(title="물병던지기 시뮤레이션") as demo:
        gr.Markdown(INTRO)
        controls = {}
        with gr.Row():
            with gr.Column(scale=2):
                for index, (group, names) in enumerate(FIELD_GROUPS):
                    with gr.Accordion(group, open=index < 3):
                        for name in names:
                            controls[name] = _control(name, gr)
            with gr.Column(scale=3):
                with gr.Row():
                    run_button = gr.Button("실행 / Run", variant="primary")
                    reset_button = gr.Button("기본값 / Reset")
                with gr.Row():
                    scan_button = gr.Button("충전율 스캔 / Filling scan")
                    forces_button = gr.Button("항별 비교 / Force comparison")
                    compare_button = gr.Button("모델 비교 / Compare models")
                    csv_button = gr.Button("CSV 저장 / Save CSV")
                with gr.Row():
                    tolerance_button = gr.Button(
                        "성공 범위 / Success basin", variant="primary")
                    substance_button = gr.Button("액체 바꾸기 / Change liquid")
                plot_output = gr.Plot(label="그래프 / figure")
                # 고정폭 블록: 요약은 열을 맞춘 표다
                text_output = gr.Code(label="요약 / summary", language=None,
                                      lines=22, interactive=False)
                file_output = gr.File(label="데이터 / data", visible=True)

        ordered = [controls[name] for name in UI_FIELDS]
        run_button.click(run_and_render, inputs=ordered,
                         outputs=[plot_output, text_output])
        scan_button.click(filling_scan, inputs=ordered,
                          outputs=[plot_output, text_output])
        forces_button.click(force_comparison, inputs=ordered,
                            outputs=[plot_output, text_output])
        compare_button.click(compare_models, inputs=ordered,
                             outputs=[plot_output, text_output])
        csv_button.click(save_csv, inputs=ordered, outputs=file_output)
        tolerance_button.click(tolerance_render, inputs=ordered,
                               outputs=[plot_output, text_output])
        substance_button.click(substance_render, inputs=ordered,
                               outputs=[plot_output, text_output])
        reset_button.click(lambda: default_values(), inputs=None,
                           outputs=ordered)
        # 열자마자 결과가 보이게 한다. 처음 온 사람이 빈 화면이 아니라
        # 뒤집기를 먼저 보도록.
        demo.load(run_and_render, inputs=ordered,
                  outputs=[plot_output, text_output])

        gr.Examples(
            examples=[[height, radius, mass, water, omega, model, pieces]
                      for _, height, radius, mass, water, omega, model, pieces
                      in PRESETS],
            inputs=[controls["bottle_height"], controls["bottle_radius"],
                    controls["bottle_mass"], controls["water_mass"],
                    controls["omega_0"], controls["model"],
                    controls["n_pieces"]],
            label="예시 / presets: %s" % ", ".join(name for name, *_ in PRESETS),
        )
        gr.Markdown(
            "참고문헌 / references: Dekker et al., Am. J. Phys. **86**, 733 "
            "(2018) · Eur. J. Phys. **45**, 065003 (2024) · arXiv:2502.12946 · "
            "Macklin & Müller, ACM TOG **32**, 104 (2013)")
    return demo


def launch_ui(**kwargs):
    """화면을 만들어 띄운다. 공개 링크를 원하면 ``share=True``.

    Build the interface and serve it; ``share=True`` for a public link.
    """
    demo = build_interface()
    demo.launch(**kwargs)
    return demo


if __name__ == "__main__":
    # 이 파일을 노트북 셀에 붙여넣어도 __name__ 은 "__main__" 이 된다.
    # 그래서 명령행을 건드리기 전에 지금 어디인지 본다: 노트북이면 UI 를,
    # 셸이면 받은 옵션대로 계산한다.
    if in_notebook():
        start_notebook_session()
    else:
        main()
