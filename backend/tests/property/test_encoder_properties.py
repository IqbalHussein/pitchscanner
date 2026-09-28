"""The encoder invariants. These are the tests worth writing about.

Every encoding in the registry has to satisfy all of them, so adding a fourth
encoding cannot quietly break the contract search depends on.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pitchscanner.encoders import available, get_encoder
from pitchscanner.geometry import PITCH_WIDTH, Situation, mirror, orient_left_to_right
from tests.property.strategies import situations

SLOW = settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)
ENCODINGS = available()


@pytest.mark.parametrize("encoding", ENCODINGS)
@SLOW
@given(situation=situations(), seed=st.integers(min_value=0, max_value=10_000))
def test_player_order_irrelevant(encoding: str, situation: Situation, seed: int) -> None:
    """Swapping which defender is which must not change the vector."""

    encoder = get_encoder(encoding)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(situation.players))
    shuffled = Situation(
        ball=situation.ball,
        players=tuple(situation.players[i] for i in order),
        visible_area=situation.visible_area,
    )
    np.testing.assert_allclose(
        encoder.encode(situation), encoder.encode(shuffled), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("encoding", ENCODINGS)
@SLOW
@given(situation=situations())
def test_encoding_is_deterministic_and_finite(encoding: str, situation: Situation) -> None:
    encoder = get_encoder(encoding)
    first = encoder.encode(situation)
    second = encoder.encode(situation)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (encoder.dim,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()


@pytest.mark.parametrize("encoding", ENCODINGS)
@SLOW
@given(situation=situations(min_players=1, max_players=22))
def test_variable_player_count_is_accepted(encoding: str, situation: Situation) -> None:
    """Frames hold up to 22 players; sketches often hold two."""

    encoder = get_encoder(encoding)
    assert encoder.encode(situation).shape == (encoder.dim,)


@SLOW
@given(situation=situations())
def test_mirroring_twice_is_the_identity(situation: Situation) -> None:
    twice = mirror(mirror(situation))
    assert twice.mirrored == situation.mirrored
    assert twice.ball.y == pytest.approx(situation.ball.y)
    for before, after in zip(situation.players, twice.players, strict=True):
        assert after.y == pytest.approx(before.y)


@SLOW
@given(situation=situations())
def test_normalising_twice_equals_normalising_once(situation: Situation) -> None:
    once = orient_left_to_right(situation, attacks_right=False)
    twice = orient_left_to_right(once, attacks_right=True)
    assert twice.ball == once.ball
    assert twice.players == once.players


#: The grid encodings place players in space, so a mirrored situation has to
#: produce a different vector. E2 is excluded from that property: its features
#: are summaries - counts, distances, widths - that are identical either side of
#: the pitch. Only its two absolute ball features move, which the next test pins
#: down exactly. This is a documented weakness of E2, not a bug, and it is the
#: reason mirroring is applied to the query rather than baked into the index.
SPATIAL_ENCODINGS = tuple(e for e in ENCODINGS if e != "E2")
#: The E2 features that depend on which touchline the ball is nearest.
E2_Y_FEATURES = ("ball_y", "ball_angle_to_goal")


@pytest.mark.parametrize("encoding", SPATIAL_ENCODINGS)
@SLOW
@given(situation=situations(min_players=2))
def test_mirroring_a_situation_changes_the_vector(
    encoding: str, situation: Situation
) -> None:
    encoder = get_encoder(encoding)
    mirrored = mirror(situation)
    on_centre_line = all(
        abs(p.y - PITCH_WIDTH / 2) < 1e-6 for p in situation.players
    ) and abs(situation.ball.y - PITCH_WIDTH / 2) < 1e-6
    if on_centre_line:
        return  # a perfectly central situation is its own mirror
    assert not np.allclose(encoder.encode(situation), encoder.encode(mirrored), atol=1e-7)


@SLOW
@given(situation=situations(min_players=2))
def test_tactical_encoding_only_moves_its_ball_features_under_mirroring(
    situation: Situation,
) -> None:
    """E2 measures shape, not side: only the two absolute ball features change.

    Worth pinning down in both directions. If E2 ever became y-dependent
    elsewhere, the mirrored query would stop meaning "the same shape on the other
    wing"; if it lost the ball features, two different areas of the pitch would
    collapse onto one vector.
    """

    from pitchscanner.encoders.e2_tactical import FEATURE_NAMES

    encoder = get_encoder("E2")
    before = encoder.encode(situation)
    after = encoder.encode(mirror(situation))
    for index, name in enumerate(FEATURE_NAMES):
        if name in E2_Y_FEATURES:
            continue
        assert before[index] == pytest.approx(after[index], abs=1e-6), (
            f"{name} changed under mirroring"
        )


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_batch_and_api_paths_produce_identical_vectors(encoding: str) -> None:
    """One shared encoder module; this is the test that pins the two sides."""

    from pitchscanner.encoders import encode_many
    from pitchscanner.geometry import PlayerPos, Point

    encoder = get_encoder(encoding)
    situation = Situation(
        ball=Point(84.0, 22.0),
        players=(
            PlayerPos(86.0, 20.0, True),
            PlayerPos(95.0, 35.0, True),
            PlayerPos(92.0, 25.0, False),
            PlayerPos(119.0, 40.0, False, True),
        ),
    )
    batch = encode_many(encoder, [situation])[0]
    online = encoder.encode(situation)
    np.testing.assert_array_equal(batch, online)
