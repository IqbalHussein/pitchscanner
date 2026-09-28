#!/usr/bin/env python3
"""Generate a small fixture dataset in StatsBomb's on-disk shape.

Checked into the repo so unit, property, golden and integration tests never touch
the network, and so `make up` has something to serve before a real ingestion has
run. The numbers are synthetic but the *shape* is the real one: same folder
layout, same field names, same 120 x 80 coordinates, same flat `visible_area`.

Regenerate with `python scripts/make_fixture.py`. It is deterministic.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "fixtures" / "statsbomb"
SEED = 20260925

COMPETITIONS = [
    {
        "competition_id": 55,
        "season_id": 282,
        "competition_name": "UEFA Euro",
        "season_name": "2024",
        "stage": "Group Stage",
        "teams": [(772, "Spain"), (914, "Germany"), (791, "Morocco"), (768, "England")],
    },
    {
        "competition_id": 9,
        "season_id": 281,
        "competition_name": "1. Bundesliga",
        "season_name": "2023/2024",
        "stage": "Regular Season",
        "teams": [(904, "Bayer Leverkusen"), (169, "Bayern Munich")],
    },
]
PLAYERS = [
    (1234, "Florian Wirtz"), (5503, "Kylian Mbappé"), (3509, "Jamal Musiala"),
    (6379, "Rodri"), (3043, "Harry Kane"), (2954, "Jude Bellingham"),
    (8206, "Lamine Yamal"), (4321, "Granit Xhaka"),
]
EVENT_TYPES = ["Pass", "Carry", "Pressure", "Ball Receipt*", "Duel", "Interception"]


def uuid_for(rng: random.Random) -> str:
    return "%08x-%04x-%04x-%04x-%012x" % (
        rng.getrandbits(32), rng.getrandbits(16), rng.getrandbits(16),
        rng.getrandbits(16), rng.getrandbits(48),
    )


def visible_area(cx: float, cy: float) -> list[float]:
    """A broadcast-camera-ish polygon around the ball, as a flat list."""

    pts: list[float] = []
    for i in range(6):
        theta = 2 * math.pi * i / 6
        rx, ry = 34.0, 26.0
        x = min(120.0, max(0.0, cx + rx * math.cos(theta)))
        y = min(80.0, max(0.0, cy + ry * math.sin(theta)))
        pts.extend([round(x, 2), round(y, 2)])
    return pts


def freeze_frame(rng: random.Random, bx: float, by: float, n_att: int, n_def: int) -> list[dict]:
    """Players clustered around the ball, as the camera would see them."""

    out = [{"teammate": True, "actor": True, "keeper": False,
            "location": [round(bx, 2), round(by, 2)]}]
    for _ in range(n_att - 1):
        out.append({
            "teammate": True, "actor": False, "keeper": False,
            "location": [
                round(min(120.0, max(0.0, bx + rng.gauss(4, 11))), 2),
                round(min(80.0, max(0.0, by + rng.gauss(0, 13))), 2),
            ],
        })
    for _ in range(n_def):
        out.append({
            "teammate": False, "actor": False, "keeper": False,
            "location": [
                round(min(120.0, max(0.0, bx + rng.gauss(7, 10))), 2),
                round(min(80.0, max(0.0, by + rng.gauss(0, 12))), 2),
            ],
        })
    out.append({"teammate": False, "actor": False, "keeper": True,
                "location": [round(rng.uniform(116, 119), 2), round(rng.uniform(36, 44), 2)]})
    return out


def build_match(rng: random.Random, comp: dict, match_id: int, home: tuple, away: tuple) -> dict:
    """One match's worth of events and frames, with plausible possessions."""

    events: list[dict] = []
    frames: list[dict] = []
    index = 1
    possession = 1
    score = {home[0]: 0, away[0]: 0}

    for period in (1, 2):
        clock = 0.0
        while clock < 45 * 60 - 40:
            team = home if rng.random() < 0.52 else away
            opponent = away if team is home else home
            # A possession: a few on-ball events moving up the pitch, ending in a
            # shot, a turnover, or simply running out.
            length = rng.randint(2, 7)
            x = rng.uniform(25, 95)
            y = rng.uniform(8, 72)
            ends_in_shot = rng.random() < 0.18
            possession_events: list[dict] = []
            for step in range(length):
                clock += rng.uniform(1.2, 4.5)
                if clock > 45 * 60:
                    break
                x = min(118.0, x + rng.uniform(-3, 11))
                y = min(78.0, max(2.0, y + rng.gauss(0, 7)))
                minute = int(clock // 60) + (45 if period == 2 else 0)
                player = PLAYERS[rng.randrange(len(PLAYERS))]
                is_last = step == length - 1
                if is_last and ends_in_shot:
                    xg = round(
                        max(0.01, min(0.9, 0.55 * math.exp(-((120 - x) ** 2 + (y - 40) ** 2) / 900))),
                        4,
                    )
                    goal = rng.random() < xg
                    if goal:
                        score[team[0]] += 1
                    ev = {
                        "id": uuid_for(rng), "index": index, "period": period,
                        "timestamp": fmt_ts(clock), "minute": minute,
                        "second": int(clock % 60),
                        "type": {"id": 16, "name": "Shot"},
                        "possession": possession,
                        "possession_team": {"id": team[0], "name": team[1]},
                        "team": {"id": team[0], "name": team[1]},
                        "player": {"id": player[0], "name": player[1]},
                        "location": [round(x, 2), round(y, 2)],
                        "shot": {
                            "statsbomb_xg": xg,
                            "outcome": {"id": 97 if goal else 98,
                                        "name": "Goal" if goal else "Saved"},
                        },
                    }
                else:
                    ev = {
                        "id": uuid_for(rng), "index": index, "period": period,
                        "timestamp": fmt_ts(clock), "minute": minute,
                        "second": int(clock % 60),
                        "type": {"id": 30, "name": EVENT_TYPES[rng.randrange(len(EVENT_TYPES))]},
                        "possession": possession,
                        "possession_team": {"id": team[0], "name": team[1]},
                        "team": {"id": team[0], "name": team[1]},
                        "player": {"id": player[0], "name": player[1]},
                        "location": [round(x, 2), round(y, 2)],
                    }
                possession_events.append(ev)
                index += 1

                # ~70% of events carry a 360 frame, like the real coverage.
                if rng.random() < 0.7:
                    frames.append({
                        "event_uuid": ev["id"],
                        "visible_area": visible_area(x, y),
                        "freeze_frame": freeze_frame(
                            rng, x, y, rng.randint(3, 7), rng.randint(3, 8)
                        ),
                    })

            if not ends_in_shot and possession_events and rng.random() < 0.75:
                # Possession lost: the next event belongs to the other team.
                # One draw, so the id and the name always describe the same player.
                defender = PLAYERS[rng.randrange(len(PLAYERS))]
                clock += rng.uniform(0.5, 2.0)
                index_ev = {
                    "id": uuid_for(rng), "index": index, "period": period,
                    "timestamp": fmt_ts(clock),
                    "minute": int(clock // 60) + (45 if period == 2 else 0),
                    "second": int(clock % 60),
                    "type": {"id": 10, "name": "Interception"},
                    "possession": possession,
                    "possession_team": {"id": team[0], "name": team[1]},
                    "team": {"id": opponent[0], "name": opponent[1]},
                    "player": {"id": defender[0], "name": defender[1]},
                    "location": [round(x, 2), round(y, 2)],
                }
                possession_events.append(index_ev)
                index += 1

            events.extend(possession_events)
            possession += 1

    match = {
        "match_id": match_id,
        "match_date": "2024-06-1%d" % (match_id % 9),
        "competition": {
            "competition_id": comp["competition_id"],
            "competition_name": comp["competition_name"],
        },
        "season": {"season_id": comp["season_id"], "season_name": comp["season_name"]},
        "home_team": {"home_team_id": home[0], "home_team_name": home[1]},
        "away_team": {"away_team_id": away[0], "away_team_name": away[1]},
        "home_score": score[home[0]],
        "away_score": score[away[0]],
        "competition_stage": {"id": 10, "name": comp["stage"]},
    }
    return {"match": match, "events": events, "frames": frames}


def fmt_ts(seconds: float) -> str:
    total = max(0.0, seconds)
    hh = int(total // 3600)
    mm = int((total % 3600) // 60)
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:06.3f}"


def main() -> None:
    rng = random.Random(SEED)
    (OUT / "matches").mkdir(parents=True, exist_ok=True)
    (OUT / "events").mkdir(parents=True, exist_ok=True)
    (OUT / "three-sixty").mkdir(parents=True, exist_ok=True)

    competitions_json = []
    match_id = 3900001
    totals = {"matches": 0, "events": 0, "frames": 0}

    for comp in COMPETITIONS:
        competitions_json.append({
            "competition_id": comp["competition_id"],
            "season_id": comp["season_id"],
            "competition_name": comp["competition_name"],
            "season_name": comp["season_name"],
            "country_name": "Europe",
            "match_available_360": "2024-06-01T00:00:00.000000",
        })
        season_matches = []
        teams = comp["teams"]
        pairs = [(teams[i], teams[(i + 1) % len(teams)]) for i in range(len(teams))]
        for home, away in pairs[: 3 if len(teams) > 2 else 1]:
            built = build_match(rng, comp, match_id, home, away)
            season_matches.append(built["match"])
            (OUT / "events" / f"{match_id}.json").write_text(
                json.dumps(built["events"], separators=(",", ":")), encoding="utf-8"
            )
            (OUT / "three-sixty" / f"{match_id}.json").write_text(
                json.dumps(built["frames"], separators=(",", ":")), encoding="utf-8"
            )
            totals["matches"] += 1
            totals["events"] += len(built["events"])
            totals["frames"] += len(built["frames"])
            match_id += 1
        comp_dir = OUT / "matches" / str(comp["competition_id"])
        comp_dir.mkdir(parents=True, exist_ok=True)
        (comp_dir / f"{comp['season_id']}.json").write_text(
            json.dumps(season_matches, indent=1), encoding="utf-8"
        )

    (OUT / "competitions.json").write_text(
        json.dumps(competitions_json, indent=1), encoding="utf-8"
    )
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
