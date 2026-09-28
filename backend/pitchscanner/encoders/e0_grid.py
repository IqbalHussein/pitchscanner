"""E0 - occupancy grid. The baseline every other encoding is measured against."""

from __future__ import annotations

import numpy as np

from pitchscanner.geometry import PITCH_LENGTH, PITCH_WIDTH, Situation

from .base import Metric, ball_context, known_mask, l2_normalise, occupancy

GRID_COLS = 12
GRID_ROWS = 8
SIGMA = 6.0
BALL_WEIGHT = 0.6


def _cell_centres() -> np.ndarray:
    cw = PITCH_LENGTH / GRID_COLS
    ch = PITCH_WIDTH / GRID_ROWS
    xs = (np.arange(GRID_COLS) + 0.5) * cw
    ys = (np.arange(GRID_ROWS) + 0.5) * ch
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


class GridEncoder:
    """Pitch split into 12 x 8 cells, one channel per side plus an unknown channel.

    Simple and order-independent by construction, but it spends as many
    dimensions on the far touchline as on the six-yard box, which is the exact
    weakness E1 exists to fix.
    """

    id = "E0"
    version = "E0-v1"
    metric: Metric = "cosine"

    def __init__(self) -> None:
        self._centres = _cell_centres()
        self._sigmas = np.full(self._centres.shape[0], SIGMA, dtype=np.float32)
        self.dim = self._centres.shape[0] * 3 + 4

    def encode(self, situation: Situation) -> np.ndarray:
        mask = known_mask(self._centres, situation)
        att = occupancy(self._centres, self._sigmas, situation.attackers) * mask
        dfn = occupancy(self._centres, self._sigmas, situation.defenders) * mask
        unknown = 1.0 - mask
        vec = np.concatenate(
            [
                l2_normalise(att),
                l2_normalise(dfn),
                l2_normalise(unknown),
                ball_context(situation.ball) * BALL_WEIGHT,
            ]
        )
        return vec.astype(np.float32, copy=False)
