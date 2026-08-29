"""Navigator-derived fitness: a continuous score where `/sedzia` only ties.

**Why this module exists.** The judge saturates. Once a population has
converged well past the wild type, every intra-population pair comes back a
tie -- the server returns whichever sequence sits in slot A, in both orders --
so `bucket_sort_sequences` ends up ranking coin flips and selection pressure
falls to zero. Measured on the generation-2 population: 5/5 ties for the best
sequence against its nearest neighbours *and* 5/5 against the worst five,
while all 10 candidates tested against the wild type won 10/10. The
comparator still works; it has just run out of resolution on our own pool.
The tie counts in `run.log` show it arriving -- 480/700 in generation 0,
602/699 in generation 1.

`/nawigator/mapa` has not run out. One call per sequence returns, among other
things, two numbers that still spread across a converged population:

* ``zmian_pod_gatunek`` -- how many positions the model would still change to
  make this a P1 promoter. 0 means it has no further edits to suggest.
* ``nie_rekonstruuje`` -- positions the decoder cannot reproduce from its own
  latent codes, i.e. how far off the model's learned manifold the sequence
  sits.

Both are lower-is-better distances to "the model's idea of a canonical P1
promoter". On the generation-2 population they spread 0-7 and 1-20 where the
judge was flat; the wild type sits at 9 and 80.

**What this is not.** The Oracle scores promoter *strength*; this scores
*canonicality*. The two coincide only insofar as the model's canonical P1
promoter is also a strong one -- a real assumption, and the honest limitation
of the method. It is, though, the only non-saturated signal on offer, and at
one call per candidate against the judge's fourteen it screens roughly 14x
more sequences per unit of the daily budget.

    from hack_the_bromoter.fitness import NavigatorFitness

    fitness = NavigatorFitness()
    ranked = fitness.rank(population)      # best first, with score columns
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import pandas as pd

from hack_the_bromoter.api import nawigator_mapa
from hack_the_bromoter.utils import ID_COL, SEQ_COL

__all__ = ["BLAD_WEIGHT", "NavigatorFitness"]

# The server reports 3000/min per key for /nawigator/mapa and there are four
# keys in the pool; stay well under one key's share so retries have room.
# The documented limit is 3000/min per key, but that is not what binds: one
# /nawigator/mapa round trip takes ~4 s and the answers run to ~85 kB, so
# throughput plateaus on the server side. Measured end to end -- 4 workers
# 37/min, 8 workers 109/min, 16 workers 225/min, 32 workers 208/min. 16 is the
# knee; past it the only thing that grows is the 502/truncated-response rate.
# RATE_PER_MIN is therefore a safety ceiling that should never bind.
RATE_PER_MIN = 1200
WORKERS = 16

# `zmian_pod_gatunek` runs 0-9 over everything seen so far, `nie_rekonstruuje`
# 1-80. At 0.1 the reconstruction term contributes at most ~8, which puts the
# two on the same footing rather than letting either dominate outright. Tune
# it here -- it is the one free parameter of the fitness.
BLAD_WEIGHT = 0.1

# A sequence the map call failed on must not win by default; park it below
# anything real rather than dropping the row (dropping shrinks the population
# mid-run, which then quietly changes the selection fraction).
FAILED_SCORE = float("inf")


class NavigatorFitness:
    """Rate-limited `/nawigator/mapa` scoring, cached by sequence.

    The cache is keyed on the **sequence**, never on the id. `breed` re-mints
    ``elite_01``..``elite_NN`` every generation for different sequences, so an
    id-keyed cache (the mistake `Judge` makes) serves one generation's verdict
    for the next generation's sequence. Sequences are immutable and unique --
    they are the only safe key.
    """

    def __init__(
        self,
        rate_per_min: int = RATE_PER_MIN,
        workers: int = WORKERS,
        blad_weight: float = BLAD_WEIGHT,
        cache: bool = True,
    ) -> None:
        self.workers = workers
        self.blad_weight = blad_weight
        self.calls = 0
        self.failures = 0

        self._gap = 60.0 / rate_per_min
        self._lock = Lock()
        self._next_slot = 0.0
        self._cache: dict[str, dict[str, Any]] | None = {} if cache else None

    # -- one call ---------------------------------------------------------
    def _slot(self) -> float:
        """Reserve the next free send slot; returns how long to sleep for."""
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_slot - now)
            self._next_slot = max(self._next_slot, now) + self._gap
            self.calls += 1
            return wait

    def score(self, sequence: str) -> dict[str, Any]:
        """Score one sequence. ``fitness`` is lower-is-better.

        Returns ``{fitness, zmian, blad, rekon, waga}``. A failed call scores
        `FAILED_SCORE` so the sequence sinks to the bottom of the ranking
        instead of being silently promoted or silently dropped.
        """
        sequence = sequence.upper()
        if self._cache is not None and sequence in self._cache:
            return self._cache[sequence]

        wait = self._slot()
        if wait:
            time.sleep(wait)
        # `api.request` already retries; this is the outer net, so a
        # sequence the server simply will not answer for costs one row rather
        # than the whole generation. Deliberately broad: anything raised here
        # means "no score for this sequence", and there is no failure worth
        # taking the run down for halfway through a breeding round.
        answer = None
        for attempt in range(3):
            try:
                answer = nawigator_mapa(sequence, od=0, ile=len(sequence))
                break
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    with self._lock:
                        self.failures += 1
                else:
                    time.sleep(0.5 * 2**attempt)

        if answer is None:
            result = {"fitness": FAILED_SCORE, "zmian": -1.0, "blad": -1.0,
                      "rekon": 0.0, "waga": 0.0}
        else:
            zmian = float(answer["zmian_pod_gatunek"])
            blad = float(answer["nie_rekonstruuje"])
            result = {
                "fitness": zmian + self.blad_weight * blad,
                "zmian": zmian,
                "blad": blad,
                "rekon": float(answer["rekon_frakcja"]),
                "waga": sum(p["wagaP"] for p in answer["pozycje"]),
            }
        if self._cache is not None:
            self._cache[sequence] = result
        return result

    # -- many calls -------------------------------------------------------
    def score_many(self, sequences: list[str]) -> list[dict[str, Any]]:
        """Score a batch in parallel; results follow input order."""
        if not sequences:
            return []
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self.score, sequences))

    def rank(
        self,
        df: pd.DataFrame,
        id_col: str = ID_COL,
        seq_col: str = SEQ_COL,
    ) -> pd.DataFrame:
        """`df` scored and sorted best-first, with the score columns attached.

        Ties on `fitness` break on `blad` then `zmian`, so the ordering is
        total and stable rather than leaving equal-fitness rows in whatever
        order they arrived in.
        """
        scores = self.score_many([str(s) for s in df[seq_col]])
        ranked = pd.concat(
            [df.reset_index(drop=True), pd.DataFrame(scores)], axis=1
        )
        return ranked.sort_values(
            ["fitness", "blad", "zmian"], kind="mergesort"
        ).reset_index(drop=True)

    def __repr__(self) -> str:
        cached = len(self._cache) if self._cache is not None else 0
        return (f"NavigatorFitness(calls={self.calls}, cached={cached}, "
                f"failures={self.failures}, blad_weight={self.blad_weight})")
