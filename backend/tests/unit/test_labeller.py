"""Outcome labelling, including the edge cases the spec says to decide and test."""

from __future__ import annotations

import pytest

from pitchscanner.geometry import PlayerPos
from pitchscanner.labeller import label_frame, label_match
from pitchscanner.models import EventRecord, FrameRecord

TEAM, OPPONENT = 1, 2


def event(
    index: int,
    ts: float,
    type_: str,
    *,
    team: int = TEAM,
    period: int = 1,
    xg: float | None = None,
    goal: bool = False,
    x: float = 80.0,
    y: float = 40.0,
    possession: int = 7,
) -> EventRecord:
    return EventRecord(
        event_id=f"e{index}",
        match_id=1,
        possession_id=possession,
        index=index,
        team_id=team,
        player_id=None,
        player_name=None,
        type=type_,
        x=x,
        y=y,
        ts=ts,
        period=period,
        minute=int(ts // 60),
        xg=xg,
        goal=goal,
        possession_team_id=TEAM,
    )


def frame(event_id: str, *, ball=(80.0, 40.0), possession: int = 7) -> FrameRecord:
    return FrameRecord(
        event_id=event_id,
        match_id=1,
        possession_id=possession,
        players=(PlayerPos(82.0, 40.0, True), PlayerPos(90.0, 42.0, False)),
        visible_area=None,
        ball_x=ball[0],
        ball_y=ball[1],
    )


def test_shot_inside_the_window_is_recorded_with_its_xg() -> None:
    events = [
        event(1, 0.0, "Pass"),
        event(2, 3.0, "Carry"),
        event(3, 6.0, "Shot", xg=0.21),
    ]
    label = label_frame(frame("e1"), events)
    assert label.shot is True
    assert label.xg_next == pytest.approx(0.21)
    assert label.turnover is False
    assert label.seconds_to_end == pytest.approx(6.0)


def test_goal_is_flagged_and_implies_a_shot() -> None:
    events = [event(1, 0.0, "Pass"), event(2, 2.0, "Shot", xg=0.4, goal=True)]
    label = label_frame(frame("e1"), events)
    assert label.goal is True
    assert label.shot is True


def test_shot_after_the_window_does_not_count() -> None:
    events = [event(1, 0.0, "Pass"), event(2, 20.0, "Shot", xg=0.3)]
    label = label_frame(frame("e1"), events, window_s=15.0)
    assert label.shot is False
    assert label.xg_next == 0.0


def test_opposition_event_is_a_turnover() -> None:
    events = [event(1, 0.0, "Pass"), event(2, 4.0, "Interception", team=OPPONENT)]
    label = label_frame(frame("e1"), events)
    assert label.turnover is True
    assert label.shot is False
    assert label.seconds_to_end == pytest.approx(4.0)


def test_a_foul_won_is_not_a_turnover() -> None:
    """Possession returns to the same team, so counting it would inflate the rate."""

    events = [
        event(1, 0.0, "Pass"),
        event(2, 3.0, "Foul Won", team=OPPONENT),
        event(3, 8.0, "Shot", xg=0.1),
    ]
    label = label_frame(frame("e1"), events)
    assert label.turnover is False
    assert label.shot is True


def test_half_ending_truncates_the_window() -> None:
    """An event in the next period cannot be the outcome of this situation."""

    events = [event(1, 2690.0, "Pass"), event(2, 2700.0, "Shot", period=2, xg=0.5)]
    label = label_frame(frame("e1"), events)
    assert label.shot is False


def test_box_entry_is_true_when_the_ball_starts_in_the_box() -> None:
    label = label_frame(frame("e1", ball=(110.0, 40.0)), [event(1, 0.0, "Pass")])
    assert label.box_entry is True


def test_box_entry_detects_a_later_entry() -> None:
    events = [event(1, 0.0, "Pass", x=70.0), event(2, 4.0, "Carry", x=110.0, y=40.0)]
    label = label_frame(frame("e1", ball=(70.0, 40.0)), events)
    assert label.box_entry is True


def test_possession_ending_without_a_shot_counts_as_a_turnover() -> None:
    events = [event(1, 0.0, "Pass"), event(2, 3.0, "Miscontrol")]
    label = label_frame(frame("e1"), events)
    assert label.turnover is True


def test_frame_with_no_following_events_is_empty_but_valid() -> None:
    label = label_frame(frame("e1"), [event(1, 0.0, "Pass")])
    assert label.shot is False
    assert label.turnover is False
    assert label.seconds_to_end == pytest.approx(15.0)


def test_unknown_anchor_returns_a_blank_label() -> None:
    label = label_frame(frame("missing"), [event(1, 0.0, "Pass")])
    assert label.event_id == "missing"
    assert label.shot is False


def test_label_match_labels_every_frame_once() -> None:
    events = [
        event(1, 0.0, "Pass", possession=1),
        event(2, 4.0, "Shot", xg=0.2, possession=1),
        event(3, 30.0, "Pass", possession=2),
        event(4, 33.0, "Interception", team=OPPONENT, possession=2),
    ]
    frames = [frame("e1", possession=1), frame("e3", possession=2)]
    labels = label_match(frames, events)
    assert [label.event_id for label in labels] == ["e1", "e3"]
    assert labels[0].shot is True
    assert labels[1].turnover is True


def test_xg_only_counts_the_possessing_team() -> None:
    events = [
        event(1, 0.0, "Pass"),
        event(2, 5.0, "Shot", team=OPPONENT, xg=0.9),
    ]
    label = label_frame(frame("e1"), events)
    assert label.shot is False
    assert label.xg_next == 0.0
    assert label.turnover is True
