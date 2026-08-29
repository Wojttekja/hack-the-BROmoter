"""The `/sedzia` comparator and the ranking built on top of it.

What `judge_properties.ipynb` established about the judge, and why this module
looks the way it does:

* **deterministic within a session** -- 40 pairs x 5 repeats never disagreed
  with themselves, so one call per (a, b) is enough and answers can be cached.
* **not symmetric** -- ~20% of pairs flip when the arguments are swapped, and
  every disagreement has the same shape: whatever sits in slot A wins. A single
  call is therefore a comparison *plus a coin weighted toward the first
  argument*. `Judge` asks every pair in both orders by default and only calls
  a sequence stronger if it wins both times; a flip is reported as a tie.
* **transitive once debiased** -- no cyclic triples under the both-orders
  relation, so a sort over it is meaningful. Under the naive single-call
  relation there are cycles, so never `sorted(..., key=cmp_to_key(judge))`.

The judge cannot separate neighbours (the ties hug the diagonal of the
tournament matrix); that is a resolution limit, not noise. Prefer
`bucket_sort_sequences` when a coarse ranking is all you need.
"""

from __future__ import annotations

import itertools
import math
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import pandas as pd

from hack_the_bromoter.api import sedzia
from hack_the_bromoter.utils import ID_COL, SEQ_COL, sequence_map

__all__ = [
    "Judge",
    "bucket_sort_sequences",
    "copeland_scores",
    "sort_sequences",
]

# The server allows 600 /sedzia calls per minute; leave headroom for retries.
RATE_PER_MIN = 540
WORKERS = 6


class Judge:
    """Rate-limited, order-debiased access to `/sedzia`.

    Built over a `{id: sequence}` map (or a DataFrame with ``id``/``sequence``
    columns) so everything downstream compares *ids* and never has to carry
    800 bp strings around.

        judge = Judge(promoters)
        ranking = sort_sequences(promoters, judge.judge_many)
        print(judge.calls, "calls spent")

    `both_orders=True` (the default) asks each pair twice, once per slot, and
    counts a win only when the same sequence wins both times -- see the module
    docstring. Set it to False to spend half the budget on a biased verdict.
    """

    def __init__(
        self,
        sequences: pd.DataFrame | dict[str, str],
        rate_per_min: int = RATE_PER_MIN,
        workers: int = WORKERS,
        both_orders: bool = True,
        cache: bool = True,
    ) -> None:
        if isinstance(sequences, pd.DataFrame):
            sequences = sequence_map(sequences)
        self.sequences = dict(sequences)
        self.workers = workers
        self.both_orders = both_orders
        self.calls = 0
        self.ties = 0

        self._gap = 60.0 / rate_per_min
        self._lock = Lock()
        self._next_slot = 0.0
        self._cache: dict[tuple[str, str], dict[str, Any]] | None = {} if cache else None

    # -- one call ---------------------------------------------------------
    def _slot(self) -> float:
        """Reserve the next free send slot; returns how long to sleep for."""
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(self._next_slot, now) + self._gap
            self.calls += 1
            return wait

    def judge(self, a: str, b: str, nazwa_a: str = "a", nazwa_b: str = "b") -> dict[str, Any]:
        """One rate-limited `/sedzia` call on two sequence ids.

        Returns ``{"a", "b", "idx", "winner", "uwaga"}``; ``idx`` is 0 when
        `a` won, 1 when `b` did. Raw and order-biased -- prefer `compare`.
        """
        if self._cache is not None and (a, b) in self._cache:
            return self._cache[(a, b)]

        wait = self._slot()
        if wait:
            time.sleep(wait)
        answer = sedzia(self.sequences[a], self.sequences[b], nazwa_a=nazwa_a, nazwa_b=nazwa_b)
        idx = answer["silniejsza_idx"]
        result = {
            "a": a,
            "b": b,
            "idx": idx,
            "winner": (a, b)[idx],
            "uwaga": answer.get("uwaga", ""),
        }
        if self._cache is not None:
            self._cache[(a, b)] = result
        return result

    def compare(self, a: str, b: str) -> dict[str, Any]:
        """Debiased verdict on one pair: two calls, one per slot.

        ``tie`` is True when the two orders disagreed -- the judge is saying it
        cannot separate them. `winner` still names one of the two (`a`, for a
        stable ordering) so callers that need a total order keep working.
        """
        return self.judge_many([(a, b)])[0]

    # -- many calls -------------------------------------------------------
    def _run(self, calls: list[tuple]) -> list[dict[str, Any]]:
        """Run judge() over many argument tuples in parallel, in input order."""
        if not calls:
            return []
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(lambda c: self.judge(*c), calls))

    def judge_many(self, calls: list[tuple[str, str]]) -> list[dict[str, Any]]:
        """Judge a batch of ``(a, b)`` id pairs; results follow input order.

        This is the comparator `sort_sequences` and `bucket_sort_sequences`
        expect. With `both_orders` it issues 2 * len(calls) API calls and
        returns one debiased verdict per input pair.
        """
        calls = [tuple(c) for c in calls]
        if not self.both_orders:
            return self._run(calls)

        forward = self._run(calls)
        reverse = self._run([(b, a) for a, b in calls])

        merged = []
        for (a, b), f, r in zip(calls, forward, reverse):
            tie = f["winner"] != r["winner"]
            self.ties += tie
            merged.append({
                "a": a,
                "b": b,
                "tie": tie,
                # a flip means "too close to call": fall back to `a` so the
                # caller still gets a usable, stable ordering.
                "winner": a if tie else f["winner"],
                "idx": 0 if tie else f["idx"],
                "uwaga": f["uwaga"],
                "forward": f["winner"],
                "reverse": r["winner"],
            })
        return merged

    def beats(self, pairs: list[tuple[str, str]]) -> set[tuple[str, str]]:
        """The strict `beats` relation over `pairs`: ordered ``(winner, loser)``
        tuples for the pairs the judge decided the same way in both orders."""
        return {
            (r["winner"], r["b"] if r["winner"] == r["a"] else r["a"])
            for r in self.judge_many(pairs)
            if not r.get("tie", False)
        }

    def __repr__(self) -> str:
        return (f"Judge({len(self.sequences)} sequences, both_orders="
                f"{self.both_orders}, calls={self.calls}, ties={self.ties})")


def copeland_scores(
    df: pd.DataFrame,
    judge_many,
    id_col: str = ID_COL,
    seq_col: str = SEQ_COL,
) -> pd.DataFrame:
    """Full round robin, scored by Copeland points (opponents beaten).

    The notebook's recommended ranking when the budget allows it: n*(n-1)/2
    pairs, so O(n^2) -- fine for ~20 sequences, not for 100. Returns a
    DataFrame sorted best first with ``beats`` and ``ties`` columns.
    """
    ids = list(df[id_col])
    pairs = list(itertools.combinations(ids, 2))
    results = judge_many(pairs)

    beats = Counter({i: 0 for i in ids})
    ties = Counter({i: 0 for i in ids})
    for (a, b), res in zip(pairs, results):
        if res.get("tie", False):
            ties[a] += 1
            ties[b] += 1
        else:
            beats[res["winner"]] += 1

    order = [i for i, _ in beats.most_common()]
    return pd.DataFrame({
        id_col: order,
        "beats": [beats[i] for i in order],
        "ties": [ties[i] for i in order],
    }).reset_index(drop=True)


def sort_sequences(df, judge_many, id_col="id", seq_col="sequence"):
    """
    Sort a dataframe of sequences using a noisy pairwise comparator.

    df: pandas DataFrame with columns [id_col, seq_col]
    judge_many: the judge_many(calls) function from above -- called with
                a list of (a, b) id tuples, returns list of dicts with
                'a', 'b', 'idx', 'winner' in the SAME order as input.

    Returns: df sorted best-to-worst (list of ids, or a reordered df).

    Strategy: bottom-up merge sort, batching every comparison needed at
    a given "merge round" into a single judge_many call. This achieves
    the minimum ~O(n log n) total comparisons for a full ordering,
    while only requiring O(log n) sequential rounds (round-trips),
    since same-round merges are independent and can run in parallel.
    """
    ids = list(df[id_col])
    n = len(ids)
    if n <= 1:
        return ids

    # Each "run" is a list of ids, already sorted best-to-worst internally.
    runs = [[i] for i in ids]

    while len(runs) > 1:
        # Pair up runs to merge: (run0, run1), (run2, run3), ...
        pairs = [(runs[i], runs[i + 1]) for i in range(0, len(runs) - 1, 2)]
        leftover = runs[-1] if len(runs) % 2 == 1 else None

        # Pointers into each pair, to know which comparison is "next"
        # for that merge. We do this in waves: each wave issues one
        # comparison per still-active pair, batched into one judge_many.
        pointers = [[0, 0] for _ in pairs]       # (i into left, j into right)
        merged = [[] for _ in pairs]
        active = list(range(len(pairs)))

        while active:
            calls = []
            call_meta = []  # which pair each call belongs to
            for p in active:
                left, right = pairs[p]
                i, j = pointers[p]
                if i < len(left) and j < len(right):
                    calls.append((left[i], right[j]))
                    call_meta.append(p)
                else:
                    # one side exhausted -> flush remainder, no call needed
                    pass

            if calls:
                results = judge_many(calls)
                for (p, res) in zip(call_meta, results):
                    left, right = pairs[p]
                    i, j = pointers[p]
                    if res["winner"] == left[i]:
                        merged[p].append(left[i])
                        pointers[p][0] += 1
                    else:
                        merged[p].append(right[j])
                        pointers[p][1] += 1

            # figure out which pairs are now fully drained on one side
            still_active = []
            for p in active:
                left, right = pairs[p]
                i, j = pointers[p]
                if i < len(left) and j < len(right):
                    still_active.append(p)
                else:
                    merged[p].extend(left[i:])
                    merged[p].extend(right[j:])
            active = still_active

        runs = merged
        if leftover is not None:
            runs.append(leftover)

    return runs[0]


def bucket_sort_sequences(
    df,
    judge_many,
    id_col="id",
    seq_col="sequence",
    n_tiers=None,
    rounds=None,
    seed=None,
):
    """
    Rank sequences into coarse tiers using a noisy pairwise comparator,
    Swiss-tournament style, instead of a full sort.

    Rationale: the comparator can't reliably distinguish sequence 3 from
    sequence 4, only "roughly this group is better than that group".
    So instead of paying for a full O(n log n) sort, we run a small fixed
    number of rounds where each sequence is compared against opponents of
    similar current standing, tally wins, and bucket by win count. Total
    calls: ~n/2 * rounds, i.e. O(n) rather than O(n log n), with `rounds`
    a small constant (default scales gently with n).

    df: DataFrame with columns [id_col, seq_col]
    judge_many: judge_many(calls) -> list of dicts with 'a','b','winner',
                in the same order as the input `calls` list of (a, b) id tuples.
    n_tiers: number of output tiers/buckets. Defaults to a rough guess
             based on n (roughly n // 10, at least 1).
    rounds: number of Swiss rounds to run. Defaults to ceil(log2(n)) + 1,
            capped small (e.g. 3-7) since that's plenty of resolution
            for a comparator this coarse.
    seed: optional random seed for reproducibility of pairing/byes.

    Returns: list of tiers, each a list of ids, ordered best-to-worst tier
             first. Within a tier, order is NOT meaningful (arbitrary /
             stable by input order) since the comparator can't support it.
    """
    rng = random.Random(seed)

    ids = list(df[id_col])
    n = len(ids)
    if n <= 1:
        return [ids]

    if rounds is None:
        rounds = max(3, min(7, math.ceil(math.log2(n)) + 1))

    if n_tiers is None:
        n_tiers = max(1, n // 10)

    # Track score (wins) and opponent history per id.
    score = {i: 0.0 for i in ids}
    played = {i: set() for i in ids}  # ids already faced, to avoid repeats

    for _ in range(rounds):
        # Group by current score, then pair within score-groups where
        # possible (classic Swiss pairing), falling back to nearest
        # available score group if a group has an odd one out or
        # everyone left has already played each other.
        by_score = defaultdict(list)
        for i in ids:
            by_score[score[i]].append(i)

        # Shuffle within each score group so pairing isn't order-biased,
        # and so repeated ties don't always pick the same "first" id.
        for group in by_score.values():
            rng.shuffle(group)

        # Flatten score groups from highest to lowest score, then walk
        # down and pair consecutive not-yet-played ids. This keeps
        # pairings close in current standing (Swiss-style) without
        # needing a full matching algorithm.
        ordered = []
        for s in sorted(by_score.keys(), reverse=True):
            ordered.extend(by_score[s])

        pairs = []
        unpaired = []
        used = set()

        pool = ordered[:]
        while pool:
            a = pool.pop(0)
            if a in used:
                continue
            used.add(a)
            # find nearest not-yet-played, not-yet-used opponent
            partner = None
            for cand in pool:
                if cand not in used and cand not in played[a]:
                    partner = cand
                    break
            if partner is None:
                # everyone remaining has already played `a`, or pool empty
                # -> give a bye this round (no call needed)
                unpaired.append(a)
                continue
            pool.remove(partner)
            used.add(partner)
            pairs.append((a, partner))

        # Bye handling: a lone leftover just doesn't get a comparison
        # this round (no score change). With even n this rarely happens
        # except when history collisions force it.

        if pairs:
            calls = [(a, b) for (a, b) in pairs]
            results = judge_many(calls)
            for (a, b), res in zip(pairs, results):
                if res.get("tie", False):
                    # the judge cannot separate them -- split the point rather
                    # than handing it to whoever happened to be passed first
                    score[a] += 0.5
                    score[b] += 0.5
                else:
                    score[res["winner"]] += 1
                played[a].add(b)
                played[b].add(a)

    # Bucket by final score into n_tiers roughly-equal-sized tiers,
    # best (highest score) first.
    ranked = sorted(ids, key=lambda i: (-score[i]))

    # tiers = _split_into_tiers(ranked, n_tiers)
    return ranked


def _split_into_tiers(ranked_ids, n_tiers):
    """Split an already-best-to-worst-ordered list into n_tiers
    contiguous, roughly-equal-sized chunks."""
    n = len(ranked_ids)
    n_tiers = max(1, min(n_tiers, n))
    base, extra = divmod(n, n_tiers)
    tiers = []
    idx = 0
    for t in range(n_tiers):
        size = base + (1 if t < extra else 0)
        tiers.append(ranked_ids[idx: idx + size])
        idx += size
    return tiers