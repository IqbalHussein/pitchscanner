"""Encoding registry.

``encoding_version`` is stored with every vector so several encodings can live
in the same database at once, which the benchmarks need.
"""

from __future__ import annotations

from typing import cast

from .base import Encoder, Metric, encode_many, known_mask, occupancy
from .e0_grid import GridEncoder
from .e1_polar import PolarEncoder
from .e2_tactical import TacticalEncoder, tactical_features

DEFAULT_ENCODING = "E1"

_REGISTRY: dict[str, type] = {
    "E0": GridEncoder,
    "E1": PolarEncoder,
    "E2": TacticalEncoder,
}
_CACHE: dict[str, Encoder] = {}


def available() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def get_encoder(encoding: str | None = None) -> Encoder:
    """Return the singleton encoder for an ID. Both batch and API go through here."""

    key = (encoding or DEFAULT_ENCODING).upper()
    if key not in _REGISTRY:
        raise KeyError(f"unknown encoding {encoding!r}; available: {available()}")
    if key not in _CACHE:
        _CACHE[key] = cast(Encoder, _REGISTRY[key]())
    return _CACHE[key]


__all__ = [
    "DEFAULT_ENCODING",
    "Encoder",
    "GridEncoder",
    "Metric",
    "PolarEncoder",
    "TacticalEncoder",
    "available",
    "encode_many",
    "get_encoder",
    "known_mask",
    "occupancy",
    "tactical_features",
]
