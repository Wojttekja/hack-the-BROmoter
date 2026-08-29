"""Cache crash-safety and CachedJudge no-bypass guarantees."""

from __future__ import annotations

import time
from pathlib import Path

from promo.cache import CachedJudge, ComparisonCache
from promo.interfaces import Comparison, Winner


def _cmp(a: str, b: str, w: Winner = "A") -> Comparison:
    return Comparison(a, b, w, timestamp=time.time(), latency_ms=1.0, source="test")


def test_cache_persists_and_reloads(tmp_path: Path) -> None:
    """A cache reopened from disk recovers all previously put comparisons."""
    path = tmp_path / "c.jsonl"
    cache = ComparisonCache(path)
    cache.put(_cmp("AAAA", "CCCC", "A"))
    cache.put(_cmp("GGGG", "TTTT", "B"))
    # Do NOT close: simulate a process that dies. Data must already be on disk.
    reopened = ComparisonCache(path)
    assert len(reopened) == 2
    assert reopened.get("AAAA", "CCCC") == "A"
    assert reopened.get("GGGG", "TTTT") == "B"


def test_cache_survives_torn_final_line(tmp_path: Path) -> None:
    """A half-written final JSONL line (crash mid-write) is skipped, rest intact."""
    path = tmp_path / "c.jsonl"
    cache = ComparisonCache(path)
    cache.put(_cmp("AAAA", "CCCC", "A"))
    cache.close()
    # Append a torn line as if the process died mid-flush.
    with path.open("a") as fh:
        fh.write('{"seq_a": "GG", "seq_b": "TT", "winn')
    reopened = ComparisonCache(path)
    assert len(reopened) == 1
    assert reopened.get("AAAA", "CCCC") == "A"


def test_assume_symmetric_collapses_orderings(tmp_path: Path) -> None:
    """With assume_symmetric, a reversed pair is a hit with a flipped verdict."""
    path = tmp_path / "c.jsonl"
    cache = ComparisonCache(path, assume_symmetric=True)
    cache.put(_cmp("AAAA", "CCCC", "A"))
    assert cache.get("CCCC", "AAAA") == "B"  # flipped to the reversed argument order


def test_cached_judge_no_duplicate_calls(tmp_path: Path) -> None:
    """CachedJudge issues at most one real call per distinct ordered pair."""

    class Raw:
        def __init__(self) -> None:
            self.calls = 0

        def compare(self, a: str, b: str) -> Winner:
            self.calls += 1
            return "A"

    raw = Raw()
    cache = ComparisonCache(tmp_path / "c.jsonl")
    cj = CachedJudge(raw, cache)
    cj.compare("X", "Y")
    cj.compare("X", "Y")
    cj.compare("X", "Y")
    assert raw.calls == 1
    assert cj.real_calls == 1
    assert cache.stats()["hits"] == 2


def test_export_polars(tmp_path: Path) -> None:
    """The comparison log exports to a polars DataFrame with the expected columns."""
    cache = ComparisonCache(tmp_path / "c.jsonl")
    cj = CachedJudge(_ConstJudge(), cache)
    cj.compare("X", "Y")
    df = cj.export_polars()
    assert df.height == 1
    assert set(df.columns) >= {"seq_a", "seq_b", "winner", "source"}


class _ConstJudge:
    def compare(self, a: str, b: str) -> Winner:
        return "A"
