"""Pitch geometry on StatsBomb's 120 x 80 grid.

Everything downstream assumes the coordinate conventions fixed here: x grows
towards the goal the team in possession is attacking, y grows from the left
touchline (y=0) to the right one (y=80).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Final

PITCH_LENGTH: Final = 120.0
PITCH_WIDTH: Final = 80.0
GOAL_X: Final = PITCH_LENGTH
GOAL_Y: Final = PITCH_WIDTH / 2
BOX_X: Final = 102.0  # penalty area edge
BOX_Y_MIN: Final = 18.0
BOX_Y_MAX: Final = 62.0

# Zone grid used for baselines: 6 columns x 3 rows = 18 zones (spec: "an 18-zone grid").
ZONE_COLS: Final = 6
ZONE_ROWS: Final = 3
N_ZONES: Final = ZONE_COLS * ZONE_ROWS


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class PlayerPos:
    """A player in a freeze frame or a sketch.

    ``teammate`` is relative to the team in possession, i.e. True for attackers.
    """

    x: float
    y: float
    teammate: bool
    keeper: bool = False

    def as_point(self) -> Point:
        return Point(self.x, self.y)


@dataclass(frozen=True, slots=True)
class Situation:
    """The unit the encoder consumes: a ball, some players, and a known region.

    ``visible_area`` is the broadcast camera polygon for real frames. ``None``
    means the known region has to be derived some other way (sketches).
    """

    ball: Point
    players: tuple[PlayerPos, ...]
    visible_area: tuple[Point, ...] | None = None
    mirrored: bool = False

    @property
    def attackers(self) -> tuple[PlayerPos, ...]:
        return tuple(p for p in self.players if p.teammate)

    @property
    def defenders(self) -> tuple[PlayerPos, ...]:
        return tuple(p for p in self.players if not p.teammate)


def clamp(value: float, low: float, high: float) -> float:
    return low if value < low else high if value > high else value


def clamp_to_pitch(x: float, y: float) -> tuple[float, float]:
    return clamp(x, 0.0, PITCH_LENGTH), clamp(y, 0.0, PITCH_WIDTH)


def in_pitch(x: float, y: float) -> bool:
    return 0.0 <= x <= PITCH_LENGTH and 0.0 <= y <= PITCH_WIDTH


def flip_x(x: float) -> float:
    return PITCH_LENGTH - x


def mirror_y(y: float) -> float:
    return PITCH_WIDTH - y


def orient_left_to_right(situation: Situation, *, attacks_right: bool = True) -> Situation:
    """Make the team in possession attack towards x = 120.

    StatsBomb already stores events this way, but the pipeline must not rely on
    that silently: a feed that ever flips would corrupt every vector in the
    index, so the transform is explicit and idempotent (see property tests).
    """

    if attacks_right:
        return situation
    return Situation(
        ball=Point(flip_x(situation.ball.x), mirror_y(situation.ball.y)),
        players=tuple(
            replace(p, x=flip_x(p.x), y=mirror_y(p.y)) for p in situation.players
        ),
        visible_area=(
            tuple(Point(flip_x(p.x), mirror_y(p.y)) for p in situation.visible_area)
            if situation.visible_area is not None
            else None
        ),
        mirrored=situation.mirrored,
    )


def mirror(situation: Situation) -> Situation:
    """Reflect top-to-bottom so a left-wing move can match a right-wing one."""

    return Situation(
        ball=Point(situation.ball.x, mirror_y(situation.ball.y)),
        players=tuple(replace(p, y=mirror_y(p.y)) for p in situation.players),
        visible_area=(
            tuple(Point(p.x, mirror_y(p.y)) for p in situation.visible_area)
            if situation.visible_area is not None
            else None
        ),
        mirrored=not situation.mirrored,
    )


def zone_id(x: float, y: float) -> int:
    """Index into the 6 x 3 baseline grid. Points outside the pitch clamp in."""

    x, y = clamp_to_pitch(x, y)
    col = min(int(x / (PITCH_LENGTH / ZONE_COLS)), ZONE_COLS - 1)
    row = min(int(y / (PITCH_WIDTH / ZONE_ROWS)), ZONE_ROWS - 1)
    return row * ZONE_COLS + col


def zone_label(zone: int) -> str:
    row, col = divmod(zone, ZONE_COLS)
    thirds = ("defensive", "def. middle", "middle", "att. middle", "attacking", "final")
    lanes = ("left", "centre", "right")
    return f"{thirds[col]} third, {lanes[row]} lane"


def in_penalty_area(x: float, y: float) -> bool:
    return x >= BOX_X and BOX_Y_MIN <= y <= BOX_Y_MAX


def distance(a: Point | PlayerPos, b: Point | PlayerPos) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return float((dx * dx + dy * dy) ** 0.5)


#: Points this close to an edge count as on it. The camera polygon is metres on a
#: 120x80 pitch, so a micrometre of slack costs nothing and buys determinism.
_EDGE_EPS = 1e-9


def _on_segment(
    x: float, y: float, ax: float, ay: float, bx: float, by: float
) -> bool:
    if abs((bx - ax) * (y - ay) - (by - ay) * (x - ax)) > _EDGE_EPS:
        return False
    return (
        min(ax, bx) - _EDGE_EPS <= x <= max(ax, bx) + _EDGE_EPS
        and min(ay, by) - _EDGE_EPS <= y <= max(ay, by) + _EDGE_EPS
    )


def point_in_polygon(x: float, y: float, polygon: Sequence[Point]) -> bool:
    """Ray-casting test. Used to mark camera-visible cells as known.

    Points exactly on an edge count as inside. That is the right answer for
    "did the camera cover this cell", and it is also the only way the test is
    reflection-symmetric: the bare crossing rule uses a strict `>` comparison,
    so a sample sitting exactly on a vertex's y falls inside on one side of a
    mirror and outside on the other. That asymmetry leaked into E2's
    ``known_fraction``, which is supposed to be side-agnostic.
    """

    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i].x, polygon[i].y
        xj, yj = polygon[j].x, polygon[j].y
        if _on_segment(x, y, xj, yj, xi, yi):
            return True
        if (yi > y) != (yj > y):
            denom = yj - yi
            if denom != 0.0 and x < xi + (y - yi) * (xj - xi) / denom:
                inside = not inside
        j = i
    return inside


def polygon_from_raw(raw: Iterable[Sequence[float]] | None) -> tuple[Point, ...] | None:
    """Build a polygon from raw [[x, y], ...] pairs, or None if it is degenerate."""

    if not raw:
        return None
    pts: list[Point] = []
    for pair in raw:
        values = list(pair)
        if len(values) >= 2:
            pts.append(Point(float(values[0]), float(values[1])))
    return tuple(pts) if len(pts) >= 3 else None
