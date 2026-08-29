"""Abstract optimizer base and shared latent/evaluation machinery.

All optimizers share the same ask/tell contract:

* :meth:`Optimizer.ask` proposes a batch of :class:`~promo.interfaces.Candidate`.
* The runner scores that batch with the Judge via :meth:`Optimizer.evaluate`, whose
  ranking *strategy* is optimizer-specific (merge-sort for population methods,
  defend-the-champion for KotH, per-cell duels for MAP-Elites).
* :meth:`Optimizer.tell` receives the ranked candidates and updates search state.
* :meth:`Optimizer.state_dict` / :meth:`Optimizer.load_state_dict` make the full
  optimizer state JSON-checkpointable so the runner can resume after a crash.

Optimizers never see a numeric score and never call ``_ground_truth``; they steer
purely on pairwise verdicts, exactly like the real event.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np

from ..interfaces import Candidate, Judge, Navigator
from ..latent import LatentOps
from ..ranking import merge_sort_rank


class Optimizer(abc.ABC):
    """Base class for black-box promoter optimizers over latent space."""

    def __init__(
        self,
        navigator: Navigator,
        seeds: list[Candidate],
        *,
        latent_ops: LatentOps | None = None,
        batch_size: int = 8,
        seed: int = 0,
    ) -> None:
        """Initialize shared optimizer state.

        Args:
            navigator: Sequence <-> latent codec.
            seeds: Seed candidates (must carry latent vectors).
            latent_ops: Geometry-aware latent operations. Defaults to Euclidean.
            batch_size: Candidates proposed per :meth:`ask`.
            seed: RNG seed.
        """
        if not seeds:
            raise ValueError("at least one seed candidate is required")
        self.nav = navigator
        self.seeds = seeds
        self.ops = latent_ops or LatentOps("euclidean")
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.generation = 0
        self.best: Candidate | None = None
        self._counter = 0

    # --- helpers ------------------------------------------------------------

    def _new_id(self, tag: str = "c") -> str:
        """Return a fresh unique candidate id."""
        self._counter += 1
        return f"{tag}{self.generation}_{self._counter}"

    def _decode(self, latent: np.ndarray, parent_id: str | None) -> Candidate:
        """Decode a latent vector into a fully-formed Candidate."""
        seq = self.nav.decode(latent)
        return Candidate(
            id=self._new_id(),
            seq=seq,
            latent=np.asarray(latent, dtype=np.float32),
            parent_id=parent_id,
            generation=self.generation,
        )

    def _rank(
        self, judge: Judge, candidates: list[Candidate], budget: int | None
    ) -> tuple[list[Candidate], int]:
        """Default ranking strategy: merge-sort the batch strongest-first."""
        by_seq = {c.seq: c for c in candidates}
        ranked_seqs, calls = merge_sort_rank(judge, [c.seq for c in candidates], budget)
        ranked = [by_seq[s] for s in ranked_seqs]
        return ranked, calls

    def evaluate(
        self, judge: Judge, candidates: list[Candidate], budget: int | None
    ) -> tuple[list[Candidate], int]:
        """Rank ``candidates`` and update the all-time best.

        The all-time best is kept judge-consistent by spending at most one extra
        comparison per generation (the new top vs the running best).

        Returns:
            ``(ranked_candidates, calls_used)``.
        """
        if not candidates:
            return [], 0
        ranked, calls = self._rank(judge, candidates, budget)
        if not ranked:
            return ranked, calls
        top = ranked[0]
        if self.best is None:
            self.best = top
        else:
            remaining = None if budget is None else budget - calls
            if remaining is None or remaining > 0:
                if judge.compare(top.seq, self.best.seq) == "A":
                    self.best = top
                calls += 1
        return ranked, calls

    # --- contract -----------------------------------------------------------

    @abc.abstractmethod
    def ask(self) -> list[Candidate]:
        """Propose a batch of candidates to be judged."""

    @abc.abstractmethod
    def tell(self, ranked_candidates: list[Candidate]) -> None:
        """Update search state from the ranked (strongest-first) batch."""

    @abc.abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the full optimizer state."""

    @abc.abstractmethod
    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore optimizer state produced by :meth:`state_dict`."""

    # --- shared (de)serialization helpers ----------------------------------

    def _base_state(self) -> dict[str, Any]:
        """Serialize state shared by every optimizer."""
        return {
            "generation": self.generation,
            "counter": self._counter,
            "geometry": self.ops.geometry,
            "best": _candidate_to_json(self.best),
            "rng": self.rng.bit_generator.state,
        }

    def _load_base_state(self, state: dict[str, Any]) -> None:
        """Restore state shared by every optimizer."""
        self.generation = state["generation"]
        self._counter = state["counter"]
        self.ops = LatentOps(state["geometry"])
        self.best = _candidate_from_json(state["best"])
        if state.get("rng") is not None:
            self.rng.bit_generator.state = state["rng"]


def _candidate_to_json(c: Candidate | None) -> dict[str, Any] | None:
    """Serialize a Candidate (latent as a list) for JSON checkpoints."""
    if c is None:
        return None
    return {
        "id": c.id,
        "seq": c.seq,
        "latent": None if c.latent is None else c.latent.tolist(),
        "parent_id": c.parent_id,
        "generation": c.generation,
        "meta": c.meta,
    }


def _candidate_from_json(d: dict[str, Any] | None) -> Candidate | None:
    """Rebuild a Candidate from :func:`_candidate_to_json`."""
    if d is None:
        return None
    return Candidate(
        id=d["id"],
        seq=d["seq"],
        latent=None if d["latent"] is None else np.asarray(d["latent"], dtype=np.float32),
        parent_id=d["parent_id"],
        generation=d["generation"],
        meta=d.get("meta", {}),
    )
