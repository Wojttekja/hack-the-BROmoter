"""Surrogate learns to predict verdicts and generalizes to held-out sequences."""

from __future__ import annotations

import random
import time

from promo import seqs
from promo.interfaces import Comparison
from promo.mock_backend import MockJudge, MockNavigator
from promo.surrogate import BradleyTerrySurrogate, SurrogateConfig


def _build_log(n_pairs: int = 400) -> list[Comparison]:
    """Generate a comparison log from the mock judge over corpus + variants."""
    recs = seqs.read_fasta()
    nav = MockNavigator(corpus=[r.seq for r in recs])
    judge = MockJudge(navigator=nav, seed=0)
    pool = [r.seq for r in recs]
    # Add shuffled variants so there are enough distinct sequences to split.
    rng = random.Random(0)
    pool += [seqs.dinuc_shuffle(s, rng) for s in pool]
    log: list[Comparison] = []
    for _ in range(n_pairs):
        a, b = rng.sample(pool, 2)
        w = judge.compare(a, b)
        log.append(Comparison(a, b, w, timestamp=time.time(), latency_ms=0.0, source="t"))
    return log


def test_surrogate_generalizes() -> None:
    """Held-out pairwise accuracy is clearly above chance."""
    log = _build_log()
    sur = BradleyTerrySurrogate(SurrogateConfig(max_epochs=150, patience=25, seed=0))
    report = sur.fit(log)
    assert report.n_val_pairs > 0
    assert report.val_accuracy > 0.65


def test_surrogate_score_and_rank() -> None:
    """score() returns a float and rank() orders consistently with scores."""
    log = _build_log(200)
    sur = BradleyTerrySurrogate(SurrogateConfig(max_epochs=80, seed=1))
    sur.fit(log)
    seqlist = list({c.seq_a for c in log})[:10]
    ranked = sur.rank(seqlist)
    scores = [sur.score(s) for s in ranked]
    assert scores == sorted(scores, reverse=True)
