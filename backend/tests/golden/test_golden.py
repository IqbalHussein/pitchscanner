"""Golden files: labels, vectors and baselines for pinned fixture matches.

A change in any of these numbers is either a bug or a deliberate change to the
encoding, and either way it should be visible in a diff. Refresh with
`UPDATE_GOLDEN=1 pytest tests/golden`.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from pitchscanner.encoders import available, get_encoder
from pitchscanner.geometry import PlayerPos, Point, Situation

SNAPSHOTS = Path(__file__).parent / "snapshots"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"

#: Three hand-written situations, so the snapshot does not depend on which
#: fixture frames happen to sort first.
PINNED_SITUATIONS = {
    "counter_3v2": Situation(
        ball=Point(84.0, 22.0),
        players=(
            PlayerPos(86.0, 20.0, True),
            PlayerPos(95.0, 35.0, True),
            PlayerPos(78.0, 30.0, True),
            PlayerPos(92.0, 25.0, False),
            PlayerPos(100.0, 30.0, False),
            PlayerPos(119.0, 40.0, False, True),
        ),
    ),
    "low_block": Situation(
        ball=Point(70.0, 40.0),
        players=(
            PlayerPos(72.0, 38.0, True),
            PlayerPos(88.0, 20.0, True),
            PlayerPos(88.0, 60.0, True),
            PlayerPos(100.0, 30.0, False),
            PlayerPos(100.0, 40.0, False),
            PlayerPos(100.0, 50.0, False),
            PlayerPos(105.0, 35.0, False),
            PlayerPos(105.0, 45.0, False),
        ),
    ),
    "wide_overload": Situation(
        ball=Point(100.0, 8.0),
        players=(
            PlayerPos(102.0, 10.0, True),
            PlayerPos(108.0, 24.0, True),
            PlayerPos(104.0, 14.0, False),
            PlayerPos(110.0, 30.0, False),
        ),
    ),
}


def _snapshot(name: str) -> Path:
    return SNAPSHOTS / name


def _compare(name: str, payload: dict) -> None:
    path = _snapshot(name)
    serialised = json.dumps(payload, indent=2, sort_keys=True)
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialised + "\n")
        if not UPDATE:
            pytest.skip(f"created missing snapshot {path.name}")
        return
    assert json.loads(path.read_text()) == payload, (
        f"{path.name} changed. If this is intentional, rerun with UPDATE_GOLDEN=1 "
        "and commit the diff."
    )


def test_vectors_for_pinned_situations() -> None:
    payload: dict[str, dict[str, object]] = {}
    for encoding in available():
        encoder = get_encoder(encoding)
        block: dict[str, object] = {"dim": encoder.dim, "metric": encoder.metric}
        for name, situation in PINNED_SITUATIONS.items():
            vector = encoder.encode(situation)
            block[name] = {
                # A digest plus a few summary statistics: enough to catch any real
                # change without a 292-number diff on every edit.
                "sha256": hashlib.sha256(vector.tobytes()).hexdigest()[:16],
                "sum": round(float(vector.sum()), 6),
                "norm": round(float(np.linalg.norm(vector)), 6),
                "nonzero": int((vector != 0).sum()),
            }
        payload[encoding] = block
    _compare("vectors.json", payload)


def test_labels_and_baselines_for_pinned_matches(fixture_store) -> None:
    """Outcome rates and zone baselines over the ingested fixture matches."""

    frames = fixture_store.frames()
    outcomes = fixture_store.outcomes([f.event_id for f in frames])
    payload = {
        "frames": len(frames),
        "outcomes": len(outcomes),
        "shot_rate": round(sum(o.shot for o in outcomes.values()) / len(outcomes), 6),
        "goal_rate": round(sum(o.goal for o in outcomes.values()) / len(outcomes), 6),
        "turnover_rate": round(
            sum(o.turnover for o in outcomes.values()) / len(outcomes), 6
        ),
        "box_entry_rate": round(
            sum(o.box_entry for o in outcomes.values()) / len(outcomes), 6
        ),
        "xg_mean": round(sum(o.xg_next for o in outcomes.values()) / len(outcomes), 6),
        "baselines": {
            str(b.zone_id): {
                "n": b.n,
                "shot_rate": round(b.shot_rate, 6),
                "xg_mean": round(b.xg_mean, 6),
            }
            for b in fixture_store.baselines(get_encoder("E1").version)
        },
    }
    _compare("labels.json", payload)


def test_tactical_features_are_named_and_stable() -> None:
    from pitchscanner.encoders import tactical_features

    payload = {
        name: {k: round(v, 6) for k, v in tactical_features(situation).items()}
        for name, situation in PINNED_SITUATIONS.items()
    }
    _compare("tactical_features.json", payload)
