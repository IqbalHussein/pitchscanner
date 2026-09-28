"""Encoder contract and the shared primitives every encoding is built from.

One module, used by both the batch job and the API. If the two sides ever
encode differently, search breaks silently, so ``tests/unit/test_encoder_parity``
pins them together and every encoder carries a ``version`` that is stored
alongside the vectors it produced.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from pitchscanner.geometry import (
    PITCH_LENGTH,
    PITCH_WIDTH,
    PlayerPos,
    Point,
    Situation,
    point_in_polygon,
)

Metric = Literal["cosine", "l2"]

#: For a sketch there is no camera polygon, so "known" is the area the user drew
#: near: a disk of this radius (metres) around the ball and every placed player.
SKETCH_KNOWN_RADIUS = 18.0


@runtime_checkable
class Encoder(Protocol):
    """A deterministic map from a situation to a fixed-length vector."""

    id: str
    version: str
    dim: int
    metric: Metric

    def encode(self, situation: Situation) -> np.ndarray: ...


def encode_many(encoder: Encoder, situations: Iterable[Situation]) -> np.ndarray:
    """Stack-encode; keeps the batch path using the exact same code as the API."""

    vectors = [encoder.encode(s) for s in situations]
    if not vectors:
        return np.zeros((0, encoder.dim), dtype=np.float32)
    return np.stack(vectors).astype(np.float32, copy=False)


def known_mask(
    cell_centres: np.ndarray,
    situation: Situation,
    *,
    sketch_radius: float = SKETCH_KNOWN_RADIUS,
) -> np.ndarray:
    """1.0 where we know what is going on in a cell, 0.0 where we do not.

    A blank area in a freeze frame is ambiguous: it can mean "nobody there" or
    "outside the broadcast camera". Treating the two the same would make every
    tightly-cropped frame look like acres of open space, so unknown cells are
    reported on their own channel instead of as emptiness.
    """

    n = cell_centres.shape[0]
    if situation.visible_area is not None:
        mask = np.empty(n, dtype=np.float32)
        for i in range(n):
            mask[i] = 1.0 if point_in_polygon(
                float(cell_centres[i, 0]), float(cell_centres[i, 1]), situation.visible_area
            ) else 0.0
        return mask

    anchors = np.array(
        [[situation.ball.x, situation.ball.y]] + [[p.x, p.y] for p in situation.players],
        dtype=np.float32,
    )
    d = np.linalg.norm(cell_centres[:, None, :] - anchors[None, :, :], axis=2)
    return (d.min(axis=1) <= sketch_radius).astype(np.float32)


def occupancy(
    cell_centres: np.ndarray,
    sigmas: np.ndarray,
    players: Sequence[PlayerPos],
) -> np.ndarray:
    """Gaussian-smoothed occupancy per cell, summed over players.

    Summation is what makes the encoding order-independent: there is no slot a
    particular defender could be assigned to, so shuffling the input cannot
    change the output (property test ``test_player_order_irrelevant``).
    """

    out = np.zeros(cell_centres.shape[0], dtype=np.float32)
    if not players:
        return out
    pts = np.array([[p.x, p.y] for p in players], dtype=np.float32)
    d2 = ((cell_centres[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
    out += np.exp(-d2 / (2.0 * (sigmas[:, None] ** 2))).sum(axis=1)
    return out.astype(np.float32, copy=False)


def l2_normalise(block: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(block))
    if norm <= 1e-9:
        return block
    return block / norm


def ball_context(ball: Point) -> np.ndarray:
    """Absolute ball features, shared by the encodings that need pitch position.

    A ball-relative encoding alone cannot tell a counter-attack in the middle
    third from the same shape at the edge of the box, so position is carried as
    a small, explicitly weighted block rather than being baked into the grid.
    """

    dx = PITCH_LENGTH - ball.x
    dy = ball.y - PITCH_WIDTH / 2
    dist_goal = math.hypot(dx, dy)
    angle = math.atan2(dy, max(dx, 1e-6))
    return np.array(
        [
            ball.x / PITCH_LENGTH,
            ball.y / PITCH_WIDTH,
            dist_goal / PITCH_LENGTH,
            angle / math.pi,
        ],
        dtype=np.float32,
    )
