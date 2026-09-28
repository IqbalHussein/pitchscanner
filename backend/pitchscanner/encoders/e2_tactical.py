"""E2 - hand-picked tactical features. Interpretable, and the UI can name them."""

from __future__ import annotations

import math

import numpy as np

from pitchscanner.geometry import (
    GOAL_X,
    GOAL_Y,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PlayerPos,
    Point,
    Situation,
    distance,
    in_penalty_area,
    point_in_polygon,
)

from .base import SKETCH_KNOWN_RADIUS, Metric

#: A lane counts as blocked if a defender is within this distance of the
#: ball-to-receiver segment.
LANE_BLOCK_RADIUS = 2.5
FEATURE_NAMES: tuple[str, ...] = (
    "attackers_n",
    "defenders_n",
    "attackers_ahead_of_ball",
    "defenders_ahead_of_ball",
    "nearest_defender_distance",
    "nearest_attacker_distance",
    "defensive_line_height",
    "attack_width",
    "defensive_compactness",
    "open_passing_lanes",
    "defenders_in_goal_cone",
    "ball_x",
    "ball_y",
    "ball_distance_to_goal",
    "ball_angle_to_goal",
    "ball_in_box",
    "known_fraction",
    "numerical_advantage_ahead",
)
#: Fixed divisors keep the vector comparable across runs: a learned scaler would
#: make yesterday's index incompatible with today's query.
_SCALE = np.array(
    [11.0, 11.0, 11.0, 11.0, 40.0, 40.0, PITCH_LENGTH, PITCH_WIDTH, 40.0, 5.0,
     5.0, PITCH_LENGTH, PITCH_WIDTH, PITCH_LENGTH, math.pi, 1.0, 1.0, 11.0],
    dtype=np.float32,
)


#: Sample points for the visible-area fraction: the centres of a 20 x 20 grid of
#: cells, not their corners. Corner sampling sits exactly on polygon edges, where
#: a ray-casting test has to break ties one way or the other, and that made the
#: fraction differ between a situation and its mirror image.
_AREA_SAMPLES: tuple[tuple[float, float], ...] = tuple(
    ((i + 0.5) * PITCH_LENGTH / 20, (j + 0.5) * PITCH_WIDTH / 20)
    for i in range(20)
    for j in range(20)
)


def _point_segment_distance(p: PlayerPos, a: Point, b: PlayerPos) -> float:
    ax, ay = a.x, a.y
    bx, by = b.x, b.y
    vx, vy = bx - ax, by - ay
    seg2 = vx * vx + vy * vy
    if seg2 <= 1e-9:
        return math.hypot(p.x - ax, p.y - ay)
    t = ((p.x - ax) * vx + (p.y - ay) * vy) / seg2
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return math.hypot(p.x - (ax + t * vx), p.y - (ay + t * vy))


def _in_goal_cone(p: PlayerPos, ball: Point) -> bool:
    """Is this player inside the triangle ball - near post - far post?"""

    posts = (Point(GOAL_X, GOAL_Y - 3.66), Point(GOAL_X, GOAL_Y + 3.66))
    ax, ay = ball.x, ball.y
    d1 = (posts[0].x - ax) * (p.y - ay) - (posts[0].y - ay) * (p.x - ax)
    d2 = (posts[1].x - ax) * (p.y - ay) - (posts[1].y - ay) * (p.x - ax)
    return (d1 <= 0 <= d2) or (d2 <= 0 <= d1)


def tactical_features(situation: Situation) -> dict[str, float]:
    """Named features, so the UI and the write-up can show what matched."""

    ball = situation.ball
    attackers = [p for p in situation.attackers]
    defenders = [p for p in situation.defenders if not p.keeper]
    keepers = [p for p in situation.defenders if p.keeper]

    ahead_a = [p for p in attackers if p.x > ball.x]
    ahead_d = [p for p in defenders if p.x > ball.x]

    near_def = min((distance(ball, p) for p in defenders + keepers), default=40.0)
    near_att = min((distance(ball, p) for p in attackers), default=40.0)

    by_x = sorted((p.x for p in defenders), reverse=True)
    line_height = float(np.mean(by_x[:4])) if by_x else 0.0

    width = float(np.std([p.y for p in attackers])) if len(attackers) > 1 else 0.0
    if len(defenders) > 1:
        pts = np.array([[p.x, p.y] for p in defenders], dtype=np.float32)
        dists = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        iu = np.triu_indices(len(defenders), k=1)
        compactness = float(dists[iu].mean())
    else:
        compactness = 0.0

    open_lanes = 0
    for target in attackers:
        if distance(ball, target) < 1.0:
            continue
        if all(
            _point_segment_distance(d, ball, target) > LANE_BLOCK_RADIUS
            for d in defenders + keepers
        ):
            open_lanes += 1

    cone = sum(1 for p in defenders if _in_goal_cone(p, ball) and p.x > ball.x)

    dx = GOAL_X - ball.x
    dy = ball.y - GOAL_Y
    if situation.visible_area is not None:
        poly = situation.visible_area
        known = sum(
            1 for x, y in _AREA_SAMPLES if point_in_polygon(x, y, poly)
        ) / len(_AREA_SAMPLES)
    else:
        known = min(
            1.0,
            (len(situation.players) + 1) * math.pi * SKETCH_KNOWN_RADIUS**2
            / (PITCH_LENGTH * PITCH_WIDTH),
        )

    return {
        "attackers_n": float(len(attackers)),
        "defenders_n": float(len(defenders)),
        "attackers_ahead_of_ball": float(len(ahead_a)),
        "defenders_ahead_of_ball": float(len(ahead_d)),
        "nearest_defender_distance": float(min(near_def, 40.0)),
        "nearest_attacker_distance": float(min(near_att, 40.0)),
        "defensive_line_height": line_height,
        "attack_width": width,
        "defensive_compactness": compactness,
        "open_passing_lanes": float(open_lanes),
        "defenders_in_goal_cone": float(cone),
        "ball_x": float(ball.x),
        "ball_y": float(ball.y),
        "ball_distance_to_goal": float(math.hypot(dx, dy)),
        "ball_angle_to_goal": float(math.atan2(dy, max(dx, 1e-6))),
        "ball_in_box": 1.0 if in_penalty_area(ball.x, ball.y) else 0.0,
        "known_fraction": float(known),
        "numerical_advantage_ahead": float(len(ahead_a) - len(ahead_d)),
    }


class TacticalEncoder:
    """Eighteen numbers a coach would recognise, scaled by fixed divisors.

    The weakness is honest: it can only find patterns somebody thought to
    measure, which is why the benchmark compares it against the grids.
    """

    id = "E2"
    version = "E2-v1"
    metric: Metric = "l2"

    def __init__(self) -> None:
        self.dim = len(FEATURE_NAMES)
        self.feature_names = FEATURE_NAMES

    def encode(self, situation: Situation) -> np.ndarray:
        feats = tactical_features(situation)
        vec = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)
        return (vec / _SCALE).astype(np.float32, copy=False)
