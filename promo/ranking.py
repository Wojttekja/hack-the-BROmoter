"""Comparison-efficient ranking primitives over a pairwise Judge.

Every function accepts an optional ``budget`` (max real comparisons) and returns
``(result, calls_used)`` so callers can account for oracle spend precisely. When the
budget is hit, functions stop cleanly and return the best partial result they have.

"Strongest first" ordering convention: ``compare(a, b) == "A"`` means ``a`` is the
stronger promoter, so ``a`` sorts ahead of ``b``.
"""

from __future__ import annotations

import random

import numpy as np

from .interfaces import Judge


class _BudgetStop(Exception):
    """Internal signal that the comparison budget has been reached."""


class _Counter:
    """Wraps a Judge to count calls and enforce a budget."""

    def __init__(self, judge: Judge, budget: int | None) -> None:
        self.judge = judge
        self.budget = budget
        self.calls = 0

    def stronger(self, a: str, b: str) -> bool:
        """Return True if ``a`` beats ``b``; raise :class:`_BudgetStop` at budget."""
        if self.budget is not None and self.calls >= self.budget:
            raise _BudgetStop
        result = self.judge.compare(a, b)
        self.calls += 1
        return result == "A"


def compare_to_champion(
    judge: Judge,
    champion: str,
    challengers: list[str],
    budget: int | None = None,
) -> tuple[str, int]:
    """Greedily defend a champion, one comparison per challenger.

    This is our cheapest mode: each challenger costs exactly one call. A challenger
    that beats the current champion becomes the new champion.

    Returns:
        ``(champion, calls_used)``.
    """
    c = _Counter(judge, budget)
    best = champion
    try:
        for challenger in challengers:
            # New challenger as first arg; if it wins it takes the crown.
            if c.stronger(challenger, best):
                best = challenger
    except _BudgetStop:
        pass
    return best, c.calls


def merge_sort_rank(
    judge: Judge,
    items: list[str],
    budget: int | None = None,
) -> tuple[list[str], int]:
    """Rank ``items`` strongest-first using merge sort (~n*log2(n) comparisons).

    Returns:
        ``(ranked, calls_used)``. On budget exhaustion returns a best-effort partial
        order (already-merged runs concatenated).
    """
    c = _Counter(judge, budget)

    def merge(left: list[str], right: list[str]) -> list[str]:
        out: list[str] = []
        i = j = 0
        while i < len(left) and j < len(right):
            if c.stronger(left[i], right[j]):
                out.append(left[i])
                i += 1
            else:
                out.append(right[j])
                j += 1
        out.extend(left[i:])
        out.extend(right[j:])
        return out

    def sort(seq: list[str]) -> list[str]:
        if len(seq) <= 1:
            return seq
        mid = len(seq) // 2
        return merge(sort(seq[:mid]), sort(seq[mid:]))

    try:
        ranked = sort(list(items))
    except _BudgetStop:
        ranked = list(items)  # give back the input order rather than nothing
    return ranked, c.calls


def tournament_top_k(
    judge: Judge,
    items: list[str],
    k: int,
    budget: int | None = None,
) -> tuple[list[str], int]:
    """Return the top ``k`` items via repeated single-elimination brackets.

    Each bracket finds the current best in ``len(pool) - 1`` comparisons; we run ``k``
    brackets, removing the winner each time. Complexity ``O(k * n)``.

    Returns:
        ``(top_k_strongest_first, calls_used)``.
    """
    c = _Counter(judge, budget)
    pool = list(items)
    winners: list[str] = []

    def bracket(players: list[str]) -> str:
        current = players
        while len(current) > 1:
            nxt: list[str] = []
            for i in range(0, len(current) - 1, 2):
                a, b = current[i], current[i + 1]
                nxt.append(a if c.stronger(a, b) else b)
            if len(current) % 2 == 1:
                nxt.append(current[-1])
            current = nxt
        return current[0]

    try:
        for _ in range(min(k, len(pool))):
            if not pool:
                break
            win = bracket(pool)
            winners.append(win)
            pool.remove(win)
    except _BudgetStop:
        pass
    return winners, c.calls


def quickselect_top_k(
    judge: Judge,
    items: list[str],
    k: int,
    budget: int | None = None,
    rng: random.Random | None = None,
) -> tuple[list[str], int]:
    """Return the top ``k`` items via quickselect, then order them by merge sort.

    Expected ``O(n)`` comparisons to isolate the top-``k`` set, plus ``O(k log k)`` to
    order it. Returns strongest-first.

    Returns:
        ``(top_k_strongest_first, calls_used)``.
    """
    rng = rng or random.Random(0)
    c = _Counter(judge, budget)

    def select(pool: list[str], want: int) -> list[str]:
        if want <= 0 or not pool:
            return []
        if len(pool) <= want:
            return list(pool)
        pivot = pool[rng.randrange(len(pool))]
        stronger, weaker = [], []
        for x in pool:
            if x is pivot:
                continue
            (stronger if c.stronger(x, pivot) else weaker).append(x)
        if len(stronger) == want:
            return stronger
        if len(stronger) >= want:
            return select(stronger, want)
        # Need the pivot and some weaker ones too.
        return stronger + [pivot] + select(weaker, want - len(stronger) - 1)

    top: list[str] = []
    try:
        top = select(list(items), min(k, len(items)))
    except _BudgetStop:
        return top, c.calls
    # Order the isolated top set (guard the remaining budget).
    remaining = None if budget is None else max(0, budget - c.calls)
    ordered, extra = merge_sort_rank(judge, top, remaining)
    return ordered, c.calls + extra


def bradley_terry_rank(
    judge: Judge,
    items: list[str],
    budget: int | None = None,
    *,
    n_pairs: int | None = None,
    iters: int = 100,
    rng: random.Random | None = None,
) -> tuple[list[str], int, np.ndarray]:
    """Rank items by fitting a Bradley-Terry model to sampled comparisons.

    Samples up to ``n_pairs`` random pairs (bounded by ``budget``), queries the
    Judge, then fits BT strengths by the standard Zermelo/MM iteration. Robust to a
    noisy or mildly intransitive Judge because it aggregates many comparisons.

    Args:
        judge: The oracle.
        items: Sequences to rank.
        budget: Max comparisons.
        n_pairs: Target number of sampled pairs (default ``4 * n``).
        iters: MM iterations.
        rng: Random source.

    Returns:
        ``(ranked_strongest_first, calls_used, strengths)`` where ``strengths`` is
        aligned to the input ``items`` order.
    """
    rng = rng or random.Random(0)
    n = len(items)
    if n == 0:
        return [], 0, np.zeros(0)
    if n == 1:
        return list(items), 0, np.ones(1)
    target = n_pairs if n_pairs is not None else 4 * n
    if budget is not None:
        target = min(target, budget)

    idx = {s: i for i, s in enumerate(items)}
    wins = np.zeros((n, n))  # wins[i, j] = times i beat j
    c = _Counter(judge, budget)
    try:
        for _ in range(target):
            i, j = rng.sample(range(n), 2)
            if c.stronger(items[i], items[j]):
                wins[i, j] += 1
            else:
                wins[j, i] += 1
    except _BudgetStop:
        pass

    strengths = _fit_bradley_terry(wins, iters)
    order = sorted(range(n), key=lambda i: -strengths[i])
    ranked = [items[i] for i in order]
    aligned = np.array([strengths[idx[s]] for s in items])
    return ranked, c.calls, aligned


def _fit_bradley_terry(wins: np.ndarray, iters: int) -> np.ndarray:
    """Fit BT strengths from a wins matrix via MM (Zermelo) iterations.

    Returns a positive strength per item, normalized to a geometric mean of 1.
    """
    n = wins.shape[0]
    p = np.ones(n)
    games = wins + wins.T
    total_wins = wins.sum(axis=1)
    for _ in range(iters):
        new_p = np.zeros(n)
        for i in range(n):
            denom = 0.0
            for j in range(n):
                if games[i, j] > 0:
                    denom += games[i, j] / (p[i] + p[j])
            new_p[i] = total_wins[i] / denom if denom > 0 else p[i]
        # Normalize to avoid drift (geometric mean = 1).
        new_p = np.clip(new_p, 1e-9, None)
        new_p /= np.exp(np.mean(np.log(new_p)))
        p = new_p
    return p
