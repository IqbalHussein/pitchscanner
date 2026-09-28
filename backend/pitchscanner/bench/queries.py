"""Query sets the benchmarks replay: real frames plus random sketches."""

from __future__ import annotations

import random
from dataclasses import dataclass

from pitchscanner.geometry import PITCH_LENGTH, PITCH_WIDTH, PlayerPos, Point, Situation
from pitchscanner.models import FrameRecord, PointIn, SearchRequest
from pitchscanner.store.base import Store

BENCH_SEED = 4242


@dataclass(slots=True)
class BenchQuery:
    kind: str  # "frame" | "sketch" | "jitter"
    situation: Situation
    source_event_id: str | None = None

    def to_request(self, k: int = 25, encoding: str = "E1") -> SearchRequest:
        return SearchRequest(
            ball=PointIn(x=self.situation.ball.x, y=self.situation.ball.y),
            attackers=[
                PointIn(x=p.x, y=p.y, keeper=p.keeper) for p in self.situation.attackers
            ][:11],
            defenders=[
                PointIn(x=p.x, y=p.y, keeper=p.keeper) for p in self.situation.defenders
            ][:11],
            k=k,
            options={"encoding": encoding, "mirror": False},  # type: ignore[arg-type]
        )


def random_sketch(rng: random.Random) -> Situation:
    """A sketch shaped like something a user would actually draw: a few players."""

    bx = rng.uniform(40.0, 115.0)
    by = rng.uniform(5.0, PITCH_WIDTH - 5.0)
    players: list[PlayerPos] = []
    for _ in range(rng.randint(1, 4)):
        players.append(
            PlayerPos(
                x=min(PITCH_LENGTH, max(0.0, bx + rng.gauss(4, 10))),
                y=min(PITCH_WIDTH, max(0.0, by + rng.gauss(0, 12))),
                teammate=True,
            )
        )
    for _ in range(rng.randint(1, 5)):
        players.append(
            PlayerPos(
                x=min(PITCH_LENGTH, max(0.0, bx + rng.gauss(7, 9))),
                y=min(PITCH_WIDTH, max(0.0, by + rng.gauss(0, 11))),
                teammate=False,
            )
        )
    return Situation(ball=Point(bx, by), players=tuple(players))


def jitter(situation: Situation, rng: random.Random, metres: float = 2.0) -> Situation:
    """Nudge every player 1-2m. The original should still come back top-5."""

    return Situation(
        ball=Point(situation.ball.x, situation.ball.y),
        players=tuple(
            PlayerPos(
                x=min(PITCH_LENGTH, max(0.0, p.x + rng.uniform(-metres, metres))),
                y=min(PITCH_WIDTH, max(0.0, p.y + rng.uniform(-metres, metres))),
                teammate=p.teammate,
                keeper=p.keeper,
            )
            for p in situation.players
        ),
        visible_area=situation.visible_area,
    )


def build_query_set(
    store: Store, n: int, *, seed: int = BENCH_SEED, frame_share: float = 0.5
) -> list[BenchQuery]:
    """Half replayed real frames, half random sketches."""

    rng = random.Random(seed)
    frames: list[FrameRecord] = store.frames()
    rng.shuffle(frames)
    n_frames = min(int(n * frame_share), len(frames))

    queries: list[BenchQuery] = [
        BenchQuery("frame", f.to_situation(), f.event_id) for f in frames[:n_frames]
    ]
    queries.extend(BenchQuery("sketch", random_sketch(rng)) for _ in range(n - n_frames))
    rng.shuffle(queries)
    return queries


def build_jitter_set(store: Store, n: int, *, seed: int = BENCH_SEED) -> list[BenchQuery]:
    rng = random.Random(seed + 1)
    frames = store.frames()
    rng.shuffle(frames)
    return [
        BenchQuery("jitter", jitter(f.to_situation(), rng), f.event_id)
        for f in frames[: min(n, len(frames))]
    ]
