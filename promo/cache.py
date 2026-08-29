"""Append-only JSONL comparison cache and the cache-enforcing Judge wrapper.

Design constraints (non-negotiable):

* No database. One JSONL file plus an in-memory dict.
* Crash-safe: every real verdict is appended and flushed to disk immediately, so a
  process killed at hour 9 resumes with zero data loss.
* Impossible to bypass: :class:`CachedJudge` owns the raw client privately; callers
  only ever hold the wrapper.

The cache key is the *ordered* pair by default, because the Judge may have order
bias. Once probing proves the Judge is symmetric, set ``assume_symmetric=True`` to
collapse both orderings and halve the call budget.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import polars as pl

from .interfaces import Comparison, Judge, Winner


def _flip(winner: Winner) -> Winner:
    """Return the verdict as seen with the arguments swapped."""
    return "B" if winner == "A" else "A"


class ComparisonCache:
    """In-memory dict of verdicts backed by an append-only JSONL file.

    The whole file is loaded on construction. Writes append one JSON line and flush
    with ``fsync`` so the on-disk log is always a prefix-consistent record of every
    verdict observed.
    """

    def __init__(self, path: str | Path, *, assume_symmetric: bool = False) -> None:
        """Open (creating if needed) the cache at ``path``.

        Args:
            path: JSONL file path.
            assume_symmetric: If true, ``(a, b)`` and ``(b, a)`` share one entry.
                Only enable after probing proves the Judge has no order bias.
        """
        self.path = Path(path)
        self.assume_symmetric = assume_symmetric
        self._store: dict[tuple[str, str], Comparison] = {}
        self.hits = 0
        self.misses = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        # Line-buffered append handle kept open for the process lifetime.
        self._fh = self.path.open("a", buffering=1)

    def _key(self, seq_a: str, seq_b: str) -> tuple[str, str]:
        """Return the storage key, order-collapsed iff ``assume_symmetric``."""
        if self.assume_symmetric and seq_a > seq_b:
            return (seq_b, seq_a)
        return (seq_a, seq_b)

    def _load(self) -> None:
        """Replay the JSONL log into the in-memory store, skipping bad lines."""
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = Comparison.from_json(json.loads(line))
                except (json.JSONDecodeError, KeyError):
                    # Tolerate a torn final line from a crash mid-write.
                    continue
                self._store[self._key(rec.seq_a, rec.seq_b)] = rec

    def get(self, seq_a: str, seq_b: str) -> Winner | None:
        """Return the cached verdict for ``(seq_a, seq_b)`` or ``None``.

        Under ``assume_symmetric`` a stored reversed pair is translated back into
        this call's argument order.
        """
        key = self._key(seq_a, seq_b)
        rec = self._store.get(key)
        if rec is None:
            self.misses += 1
            return None
        self.hits += 1
        if rec.seq_a == seq_a and rec.seq_b == seq_b:
            return rec.winner
        # Stored in the opposite order (only possible when assume_symmetric).
        return _flip(rec.winner)

    def put(self, comparison: Comparison) -> None:
        """Append a verdict to disk and memory, flushing immediately."""
        key = self._key(comparison.seq_a, comparison.seq_b)
        self._store[key] = comparison
        self._fh.write(json.dumps(comparison.to_json()) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def __len__(self) -> int:
        """Number of distinct cached comparisons."""
        return len(self._store)

    def __contains__(self, pair: tuple[str, str]) -> bool:
        """Whether a verdict exists for the given ordered pair."""
        return self._key(*pair) in self._store

    def stats(self) -> dict[str, Any]:
        """Return hit/miss counters and store size for progress reporting."""
        total = self.hits + self.misses
        return {
            "entries": len(self._store),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
            "assume_symmetric": self.assume_symmetric,
        }

    def close(self) -> None:
        """Flush and close the append handle."""
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> ComparisonCache:
        """Context-manager entry."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Context-manager exit closes the file handle."""
        self.close()


class CachedJudge:
    """Judge wrapper that serves cached verdicts and records every real call.

    The raw client is stored privately (name-mangled) so no caller can reach past
    the cache. Satisfies the :class:`Judge` Protocol, so it is a drop-in oracle.
    """

    def __init__(
        self,
        raw_judge: Judge,
        cache: ComparisonCache,
        *,
        source: str = "real",
    ) -> None:
        """Wrap ``raw_judge`` behind ``cache``.

        Args:
            raw_judge: The concrete oracle. Held privately; do not expose it.
            cache: The comparison cache.
            source: Provenance tag stored on freshly issued verdicts.
        """
        self.__raw = raw_judge
        self._cache = cache
        self._source = source
        self.real_calls = 0

    def compare(self, seq_a: str, seq_b: str) -> Winner:
        """Return a verdict, issuing a real call only on a cache miss."""
        cached = self._cache.get(seq_a, seq_b)
        if cached is not None:
            return cached
        start = time.perf_counter()
        winner = self.__raw.compare(seq_a, seq_b)
        latency_ms = (time.perf_counter() - start) * 1000.0
        self.real_calls += 1
        self._cache.put(
            Comparison(
                seq_a=seq_a,
                seq_b=seq_b,
                winner=winner,
                timestamp=time.time(),
                latency_ms=latency_ms,
                source=self._source,
            )
        )
        return winner

    @property
    def cache(self) -> ComparisonCache:
        """The backing cache (read access for stats; the raw client stays hidden)."""
        return self._cache

    def export_polars(self) -> pl.DataFrame:
        """Return the full comparison log as a polars DataFrame for analysis."""
        rows = [c.to_json() for c in self._cache._store.values()]
        if not rows:
            return pl.DataFrame(
                schema={
                    "seq_a": pl.Utf8,
                    "seq_b": pl.Utf8,
                    "winner": pl.Utf8,
                    "timestamp": pl.Float64,
                    "latency_ms": pl.Float64,
                    "source": pl.Utf8,
                }
            )
        return pl.DataFrame(rows)
