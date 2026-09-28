"""Hypothesis strategies for pitch situations."""

from __future__ import annotations

from hypothesis import strategies as st

from pitchscanner.geometry import PITCH_LENGTH, PITCH_WIDTH, PlayerPos, Point, Situation

coordinate_x = st.floats(min_value=0.0, max_value=PITCH_LENGTH, allow_nan=False)
coordinate_y = st.floats(min_value=0.0, max_value=PITCH_WIDTH, allow_nan=False)


@st.composite
def players(draw, teammate: bool | None = None) -> PlayerPos:
    return PlayerPos(
        x=draw(coordinate_x),
        y=draw(coordinate_y),
        teammate=draw(st.booleans()) if teammate is None else teammate,
        keeper=draw(st.booleans()),
    )


@st.composite
def situations(draw, min_players: int = 1, max_players: int = 22, with_area: bool = False):
    ball = Point(draw(coordinate_x), draw(coordinate_y))
    squad = draw(st.lists(players(), min_size=min_players, max_size=max_players))
    area = None
    if with_area or draw(st.booleans()):
        cx, cy = ball.x, ball.y
        area = (
            Point(max(0.0, cx - 30), max(0.0, cy - 20)),
            Point(min(PITCH_LENGTH, cx + 30), max(0.0, cy - 20)),
            Point(min(PITCH_LENGTH, cx + 30), min(PITCH_WIDTH, cy + 20)),
            Point(max(0.0, cx - 30), min(PITCH_WIDTH, cy + 20)),
        )
    return Situation(ball=ball, players=tuple(squad), visible_area=area)
