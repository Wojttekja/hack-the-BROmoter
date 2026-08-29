"""Ranking primitives: correct order and stated call complexity."""

from __future__ import annotations

import math

from promo.ranking import (
    bradley_terry_rank,
    compare_to_champion,
    merge_sort_rank,
    quickselect_top_k,
    tournament_top_k,
)
from tests.conftest import CountingJudge

ITEMS = [f"seq{c}" for c in "ABCDEFGHIJ"]  # lexicographic total order


def _expected_order(items: list[str]) -> list[str]:
    """Strongest-first == lexicographically descending for CountingJudge."""
    return sorted(items, reverse=True)


def test_merge_sort_correct_and_within_complexity() -> None:
    """merge_sort_rank returns the total order using <= n*ceil(log2 n) calls."""
    j = CountingJudge()
    ranked, calls = merge_sort_rank(j, ITEMS)
    assert ranked == _expected_order(ITEMS)
    n = len(ITEMS)
    assert calls <= n * math.ceil(math.log2(n))
    assert calls == j.calls


def test_tournament_top_k_correct_and_bounded() -> None:
    """tournament_top_k returns the true top-k within O(k*n) comparisons."""
    j = CountingJudge()
    top, calls = tournament_top_k(j, ITEMS, k=3)
    assert top == _expected_order(ITEMS)[:3]
    assert calls <= 3 * len(ITEMS)


def test_quickselect_top_k_correct() -> None:
    """quickselect_top_k returns the true top-k, ordered strongest-first."""
    j = CountingJudge()
    top, _ = quickselect_top_k(j, ITEMS, k=4)
    assert top == _expected_order(ITEMS)[:4]


def test_compare_to_champion_one_call_each() -> None:
    """compare_to_champion spends exactly one call per challenger."""
    j = CountingJudge()
    champ, calls = compare_to_champion(j, "seqE", ["seqA", "seqZ", "seqC"])
    assert calls == 3
    assert champ == "seqZ"  # the only challenger stronger than E lexicographically


def test_budget_stops_cleanly() -> None:
    """A binding budget halts ranking without error and reports calls used."""
    j = CountingJudge()
    _, calls = merge_sort_rank(j, ITEMS, budget=5)
    assert calls <= 5


def test_bradley_terry_recovers_order() -> None:
    """Bradley-Terry ranking recovers the total order from sampled comparisons."""
    j = CountingJudge()
    ranked, calls, strengths = bradley_terry_rank(j, ITEMS, n_pairs=120)
    # Top and bottom should be correct even if the middle wobbles.
    assert ranked[0] == "seqJ"
    assert ranked[-1] == "seqA"
    assert calls <= 120
    assert len(strengths) == len(ITEMS)
