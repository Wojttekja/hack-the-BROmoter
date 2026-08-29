"""Protocols and dataclasses that define the oracle contract.

Everything downstream depends on these types, never on a concrete backend. The two
oracles are:

* :class:`Judge` -- pairwise strength comparison only. No numeric score.
* :class:`Navigator` -- encode sequences to latent vectors and decode back. Knows
  nothing about strength.

The real HYPPE API is unknown until the event; these Protocols are the seam we
adapt it to. See :mod:`promo.real_backend`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

Winner = Literal["A", "B"]
"""Verdict returned by a Judge: ``"A"`` means ``seq_a`` is the stronger promoter."""


class OracleError(RuntimeError):
    """Base class for backend-agnostic oracle failures.

    The probe suite catches these to detect rate limiting and budget exhaustion
    without importing any concrete backend.
    """


class RateLimitError(OracleError):
    """Raised when the oracle refuses a call due to rate limiting."""


class BudgetExhaustedError(OracleError):
    """Raised when the oracle's hard call budget is exhausted."""


@runtime_checkable
class Judge(Protocol):
    """Pairwise promoter-strength oracle.

    The only capability is: given two sequences, say which is predicted stronger.
    There is no numeric score and no notion of *how much* stronger.
    """

    def compare(self, seq_a: str, seq_b: str) -> Winner:
        """Return ``"A"`` if ``seq_a`` is predicted stronger, else ``"B"``."""
        ...


@runtime_checkable
class Navigator(Protocol):
    """Sequence <-> latent-space codec.

    The latent geometry may be Euclidean or a Poincare ball (hyperbolic). Callers
    must not assume linear arithmetic is valid; use :mod:`promo.latent` instead.
    ``dim`` and ``distance`` are optional in the real API and guarded accordingly.
    """

    def encode(self, seq: str) -> np.ndarray:
        """Encode a sequence into a latent vector."""
        ...

    def decode(self, z: np.ndarray) -> str:
        """Decode a latent vector back into a sequence."""
        ...

    @property
    def dim(self) -> int:
        """Latent dimensionality (optional in the real API)."""
        ...

    def distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Native latent distance (optional in the real API)."""
        ...


@dataclass(frozen=True, slots=True)
class Comparison:
    """A single recorded verdict from the Judge.

    Attributes:
        seq_a: First sequence as passed to ``compare``.
        seq_b: Second sequence as passed to ``compare``.
        winner: ``"A"`` or ``"B"`` -- which argument won.
        timestamp: Unix epoch seconds when the call returned.
        latency_ms: Wall-clock latency of the underlying call in milliseconds.
        source: Provenance tag, e.g. ``"real"``, ``"cache"``, ``"transitive"``.
    """

    seq_a: str
    seq_b: str
    winner: Winner
    timestamp: float
    latency_ms: float
    source: str

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict for the append-only cache log."""
        return {
            "seq_a": self.seq_a,
            "seq_b": self.seq_b,
            "winner": self.winner,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "source": self.source,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Comparison:
        """Rebuild a Comparison from a parsed JSONL record."""
        return cls(
            seq_a=d["seq_a"],
            seq_b=d["seq_b"],
            winner=d["winner"],
            timestamp=d["timestamp"],
            latency_ms=d["latency_ms"],
            source=d["source"],
        )


@dataclass(slots=True)
class Candidate:
    """A candidate promoter tracked by the optimizers.

    Attributes:
        id: Stable unique identifier.
        seq: Nucleotide sequence.
        latent: Latent vector this candidate was decoded from, if any.
        parent_id: ``id`` of the candidate this was derived from, if any.
        generation: Optimizer iteration at which it was produced.
        meta: Free-form per-candidate metadata (behaviour descriptors, notes).
    """

    id: str
    seq: str
    latent: np.ndarray | None = None
    parent_id: str | None = None
    generation: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
