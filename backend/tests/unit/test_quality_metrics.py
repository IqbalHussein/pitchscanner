"""The evaluation metrics themselves.

These are the numbers a resume line would quote, so they get tested against
hand-computed values rather than against whatever the code currently returns.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pitchscanner.bench.quality import (
    _rank_for,
    dcg,
    load_labels,
    ndcg_at_k,
    score_encodings,
)


def test_dcg_matches_the_definition() -> None:
    # 3/log2(2) + 2/log2(3) + 0/log2(4) + 1/log2(5)
    expected = 3 / 1.0 + 2 / math.log2(3) + 0.0 + 1 / math.log2(5)
    assert dcg([3, 2, 0, 1]) == pytest.approx(expected)


def test_dcg_of_nothing_relevant_is_zero() -> None:
    assert dcg([0, 0, 0]) == 0.0
    assert dcg([]) == 0.0


def test_ndcg_is_one_when_the_ranking_is_ideal() -> None:
    assert ndcg_at_k([3, 2, 1], 3, judged=[3, 2, 1]) == pytest.approx(1.0)


def test_ndcg_punishes_a_reversed_ranking() -> None:
    good = ndcg_at_k([2, 1], 2, judged=[2, 1])
    bad = ndcg_at_k([1, 2], 2, judged=[2, 1])
    assert bad < good == pytest.approx(1.0)


def test_ndcg_ideal_comes_from_everything_judged_not_from_the_run() -> None:
    """The point of pooling: the denominator is the achievable best.

    A run that returned two barely-relevant results in the right order is not
    perfect if the pool contained a highly relevant one it missed.
    """

    ranked = [1, 1]
    assert ndcg_at_k(ranked, 10, judged=ranked) == pytest.approx(1.0)
    missed_a_better_one = ndcg_at_k(ranked, 10, judged=[3, 1, 1])
    assert missed_a_better_one < 0.75


def test_ndcg_is_zero_when_nothing_relevant_was_judged() -> None:
    assert ndcg_at_k([0, 0], 2, judged=[0, 0]) == 0.0


def test_rank_for_reports_the_best_rank_and_none_when_absent() -> None:
    candidate = {
        "event_id": "e1",
        "from": [
            {"encoding": "E0", "rank": 4, "distance": 0.3},
            {"encoding": "E1", "rank": 1, "distance": 0.1},
            {"encoding": "E0", "rank": 2, "distance": 0.2},
        ],
    }
    assert _rank_for(candidate, "E0") == 2
    assert _rank_for(candidate, "E1") == 1
    assert _rank_for(candidate, "E2") is None
    assert _rank_for({"event_id": "e2", "from": []}, "E0") is None


def _write_pool(path: Path) -> None:
    """One query. E0 ranks the relevant hit first, E1 ranks it last.

    ``x`` is in the pool but was only ever returned by E1, so it must not appear
    in E0's ranking at all.
    """

    rows = [
        {
            "query_id": "q0000",
            "source_event_id": "src",
            "ball": [60.0, 40.0],
            "candidates": [
                {
                    "event_id": "good",
                    "from": [
                        {"encoding": "E0", "rank": 0, "distance": 0.1},
                        {"encoding": "E1", "rank": 2, "distance": 0.4},
                    ],
                    "label": None,
                },
                {
                    "event_id": "bad",
                    "from": [
                        {"encoding": "E0", "rank": 1, "distance": 0.2},
                        {"encoding": "E1", "rank": 0, "distance": 0.1},
                    ],
                    "label": None,
                },
                {
                    "event_id": "x",
                    "from": [{"encoding": "E1", "rank": 1, "distance": 0.2}],
                    "label": None,
                },
            ],
        }
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_score_encodings_ranks_each_encoding_on_its_own_results(tmp_path: Path) -> None:
    pool = tmp_path / "eval-pool.jsonl"
    labels = tmp_path / "eval-labels.jsonl"
    _write_pool(pool)
    labels.write_text(
        json.dumps({"query_id": "q0000", "labels": {"good": 2, "bad": 0, "x": 0}}) + "\n",
        encoding="utf-8",
    )

    scores = {
        r.encoding: r
        for r in score_encodings(pool_path=pool, labels_path=labels)
    }

    # E0 returned two candidates, one relevant: precision 1/2.
    assert scores["E0"].precision_at_10 == pytest.approx(0.5)
    # E1 returned three, one relevant: precision 1/3.
    assert scores["E1"].precision_at_10 == pytest.approx(1 / 3)
    # Both found the same single relevant result; E0 put it first, E1 put it third.
    assert scores["E0"].ndcg_at_10 == pytest.approx(1.0)
    assert scores["E1"].ndcg_at_10 == pytest.approx(1 / 2)
    # E2 returned nothing for this query, so it has no labelled queries at all.
    assert scores["E2"].labelled_queries == 0
    assert scores["E2"].precision_at_10 == 0.0


def test_unlabelled_queries_are_skipped_rather_than_counted_as_misses(
    tmp_path: Path,
) -> None:
    pool = tmp_path / "eval-pool.jsonl"
    labels = tmp_path / "eval-labels.jsonl"
    _write_pool(pool)
    labels.write_text(
        json.dumps({"query_id": "q9999", "labels": {"good": 2}}) + "\n", encoding="utf-8"
    )

    scores = score_encodings(pool_path=pool, labels_path=labels)
    assert all(r.labelled_queries == 0 for r in scores)
    assert all(r.precision_at_10 == 0.0 for r in scores)


def test_missing_label_file_yields_no_scores_rather_than_zeros(tmp_path: Path) -> None:
    pool = tmp_path / "eval-pool.jsonl"
    _write_pool(pool)
    assert load_labels(tmp_path / "nope.jsonl") == {}
    assert score_encodings(pool_path=pool, labels_path=tmp_path / "nope.jsonl") == []
