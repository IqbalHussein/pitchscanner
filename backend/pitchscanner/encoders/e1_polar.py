"""E1 - ball-centred polar grid: rings and sectors around the ball, finer near it."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from pitchscanner.geometry import Situation

from .base import Metric, ball_context, known_mask, l2_normalise, occupancy

#: Ring edges in metres. The first two rings cover the ten metres that decide
#: whether a pass is even possible; the outer ring is deliberately coarse.
RING_EDGES = (0.0, 6.0, 12.0, 22.0, 40.0)
N_SECTORS = 8
#: Cell smoothing grows with distance from the ball, so near the ball a metre
#: matters and out at 35m it does not.
RING_SIGMA = (2.5, 4.0, 7.0, 12.0)
BALL_WEIGHT = 1.0
UNKNOWN_WEIGHT = 0.5


def _polar_offsets() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell centre offsets (dx, dy) from the ball, their sigmas and ring indices."""

    offsets: list[tuple[float, float]] = []
    sigmas: list[float] = []
    rings: list[int] = []
    for r, (lo, hi) in enumerate(pairwise(RING_EDGES)):
        radius = (lo + hi) / 2.0
        for s in range(N_SECTORS):
            theta = 2.0 * math.pi * (s + 0.5) / N_SECTORS
            offsets.append((radius * math.cos(theta), radius * math.sin(theta)))
            sigmas.append(RING_SIGMA[r])
            rings.append(r)
    return (
        np.array(offsets, dtype=np.float32),
        np.array(sigmas, dtype=np.float32),
        np.array(rings, dtype=np.int32),
    )


class PolarEncoder:
    """Everything is measured relative to the ball, then position is added back.

    Sectors are measured from the attacking direction (+x), so the same shape in
    the left and right channel produces the same local pattern; absolute
    position rides along in a small weighted block instead.
    """

    id = "E1"
    version = "E1-v1"
    metric: Metric = "cosine"

    def __init__(self) -> None:
        self._offsets, self._sigmas, self._rings = _polar_offsets()
        self.n_cells = self._offsets.shape[0]
        self.dim = self.n_cells * 2 + len(RING_SIGMA) + 4

    def encode(self, situation: Situation) -> np.ndarray:
        centres = self._offsets + np.array(
            [situation.ball.x, situation.ball.y], dtype=np.float32
        )
        mask = known_mask(centres, situation)
        att = occupancy(centres, self._sigmas, situation.attackers) * mask
        dfn = occupancy(centres, self._sigmas, situation.defenders) * mask

        # Unknown is summarised per ring: which cell of a ring is off-camera
        # matters much less than how much of that ring we cannot see.
        unknown = np.zeros(len(RING_SIGMA), dtype=np.float32)
        for r in range(len(RING_SIGMA)):
            in_ring = self._rings == r
            unknown[r] = float(1.0 - mask[in_ring].mean())

        vec = np.concatenate(
            [
                l2_normalise(att),
                l2_normalise(dfn),
                unknown * UNKNOWN_WEIGHT,
                ball_context(situation.ball) * BALL_WEIGHT,
            ]
        )
        return vec.astype(np.float32, copy=False)
