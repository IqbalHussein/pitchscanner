"""Zone assignment, polygon tests and the coordinate transforms."""

from __future__ import annotations

import pytest

from pitchscanner.geometry import (
    N_ZONES,
    PITCH_LENGTH,
    PITCH_WIDTH,
    PlayerPos,
    Point,
    Situation,
    clamp_to_pitch,
    in_penalty_area,
    mirror,
    orient_left_to_right,
    point_in_polygon,
    polygon_from_raw,
    zone_id,
    zone_label,
)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [
        (0.0, 0.0, 0),
        (119.999, 79.999, N_ZONES - 1),
        (60.0, 40.0, 9),  # 4th column of 6, middle lane of 3
        (10.0, 70.0, 12),
    ],
)
def test_zone_id_known_values(x: float, y: float, expected: int) -> None:
    assert zone_id(x, y) == expected


def test_zone_id_clamps_out_of_bounds() -> None:
    assert zone_id(-5.0, -5.0) == 0
    assert zone_id(500.0, 500.0) == N_ZONES - 1


def test_every_zone_has_a_label() -> None:
    labels = {zone_label(z) for z in range(N_ZONES)}
    assert len(labels) == N_ZONES


def test_clamp_to_pitch() -> None:
    assert clamp_to_pitch(-1.0, 100.0) == (0.0, PITCH_WIDTH)
    assert clamp_to_pitch(130.0, -3.0) == (PITCH_LENGTH, 0.0)


@pytest.mark.parametrize(
    ("x", "y", "expected"),
    [(110.0, 40.0, True), (110.0, 10.0, False), (90.0, 40.0, False), (102.0, 18.0, True)],
)
def test_penalty_area(x: float, y: float, expected: bool) -> None:
    assert in_penalty_area(x, y) is expected


def test_point_in_polygon_square() -> None:
    square = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    assert point_in_polygon(5, 5, square)
    assert not point_in_polygon(15, 5, square)
    assert not point_in_polygon(5, 15, square)


def test_point_in_polygon_needs_three_points() -> None:
    assert not point_in_polygon(1, 1, [Point(0, 0), Point(1, 1)])


def test_polygon_from_raw_rejects_degenerate() -> None:
    assert polygon_from_raw(None) is None
    assert polygon_from_raw([[0, 0], [1, 1]]) is None
    assert polygon_from_raw([[0, 0], [1, 0], [1, 1]]) is not None


def _situation() -> Situation:
    return Situation(
        ball=Point(30.0, 20.0),
        players=(PlayerPos(40.0, 25.0, True), PlayerPos(50.0, 60.0, False, True)),
        visible_area=(Point(0, 0), Point(60, 0), Point(60, 40), Point(0, 40)),
    )


def test_orient_flips_when_attacking_left() -> None:
    flipped = orient_left_to_right(_situation(), attacks_right=False)
    assert flipped.ball.x == PITCH_LENGTH - 30.0
    assert flipped.ball.y == PITCH_WIDTH - 20.0
    assert flipped.players[1].keeper is True


def test_orient_is_a_noop_when_already_correct() -> None:
    situation = _situation()
    assert orient_left_to_right(situation, attacks_right=True) is situation


def test_mirror_flips_y_and_sets_flag() -> None:
    mirrored = mirror(_situation())
    assert mirrored.ball.y == PITCH_WIDTH - 20.0
    assert mirrored.mirrored is True
    assert mirrored.visible_area is not None
