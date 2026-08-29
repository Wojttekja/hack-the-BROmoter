"""MAP-Elites: illuminate a behaviour archive with one duel per cell.

Each candidate is placed in a cell of a discretized behaviour space (GC, length,
TATA count, latent radius by default). A candidate only enters a cell by beating its
current occupant in a single comparison, so the archive fills with diverse, locally
strong promoters. The archive exports to a DataFrame for heatmap plotting.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .. import seqs
from ..interfaces import Candidate, Judge
from .base import Optimizer, _candidate_from_json, _candidate_to_json


@dataclass(frozen=True, slots=True)
class BehaviourAxis:
    """One behaviour dimension: a bounded descriptor discretized into ``bins`` cells."""

    name: str
    fn: Callable[[Candidate], float]
    bins: int
    lo: float
    hi: float

    def index(self, cand: Candidate) -> int:
        """Return the (clamped) bin index of ``cand`` along this axis."""
        val = self.fn(cand)
        if self.hi <= self.lo:
            return 0
        frac = (val - self.lo) / (self.hi - self.lo)
        return int(min(self.bins - 1, max(0, int(frac * self.bins))))

    def center(self, idx: int) -> float:
        """Return the descriptor value at the center of bin ``idx``."""
        width = (self.hi - self.lo) / self.bins
        return self.lo + (idx + 0.5) * width


def _axis(name: str, bins: int, ops_radius: Callable[[Candidate], float]) -> BehaviourAxis:
    """Build a named built-in behaviour axis."""
    if name == "gc":
        return BehaviourAxis("gc", lambda c: seqs.gc_content(c.seq), bins, 0.2, 0.8)
    if name == "length":
        return BehaviourAxis("length", lambda c: float(len(c.seq)), bins, 100.0, 2000.0)
    if name == "tata":
        return BehaviourAxis("tata", lambda c: float(seqs.count_tata(c.seq)), bins, 0.0, 8.0)
    if name == "latent_radius":
        return BehaviourAxis("latent_radius", ops_radius, bins, 0.0, 5.0)
    raise ValueError(f"unknown behaviour axis: {name!r}")


class MapElites(Optimizer):
    """Quality-diversity search over a discretized behaviour archive."""

    def __init__(
        self,
        *args: Any,
        axes: list[tuple[str, int]] | None = None,
        scale: float = 0.6,
        **kwargs: Any,
    ) -> None:
        """Initialize MAP-Elites.

        Args:
            axes: List of ``(axis_name, n_bins)``. Defaults to a 2-D GC x length grid
                (the two axes rendered in the heatmap); extra axes add dimensions.
            scale: Latent perturbation scale for producing offspring.
        """
        super().__init__(*args, **kwargs)
        self.scale = scale
        self._axis_cfg = axes or [("gc", 12), ("length", 12)]
        self._build_axes()
        self.archive: dict[tuple[int, ...], Candidate] = {}

    def _build_axes(self) -> None:
        """Instantiate behaviour axes from the current config."""

        def radius(c: Candidate) -> float:
            return self.ops.radius(c.latent) if c.latent is not None else 0.0

        self.axes = [_axis(name, bins, radius) for name, bins in self._axis_cfg]

    def _cell(self, cand: Candidate) -> tuple[int, ...]:
        """Return the archive cell key for ``cand``."""
        return tuple(ax.index(cand) for ax in self.axes)

    def _parents(self) -> list[Candidate]:
        """Return a parent pool: archive elites if any, else the seeds."""
        return list(self.archive.values()) if self.archive else list(self.seeds)

    def ask(self) -> list[Candidate]:
        """Produce offspring by perturbing random elites/seeds."""
        self.generation += 1
        pool = self._parents()
        out: list[Candidate] = []
        for _ in range(self.batch_size):
            parent = pool[int(self.rng.integers(len(pool)))]
            assert parent.latent is not None
            z = self.ops.perturb(parent.latent, self.scale, self.rng)
            out.append(self._decode(z, parent_id=parent.id))
        return out

    def evaluate(
        self, judge: Judge, candidates: list[Candidate], budget: int | None
    ) -> tuple[list[Candidate], int]:
        """Duel each candidate against its cell occupant; insert winners."""
        calls = 0
        accepted: list[Candidate] = []
        rejected: list[Candidate] = []
        for cand in candidates:
            if budget is not None and calls >= budget:
                break
            cell = self._cell(cand)
            occupant = self.archive.get(cell)
            if occupant is None:
                self.archive[cell] = cand
                accepted.append(cand)
                self._maybe_update_best(judge, cand, budget, calls)
                continue
            calls += 1
            if judge.compare(cand.seq, occupant.seq) == "A":
                self.archive[cell] = cand
                accepted.append(cand)
                if budget is None or calls < budget:
                    calls += self._maybe_update_best(judge, cand, budget, calls)
            else:
                rejected.append(cand)
        return accepted + rejected, calls

    def _maybe_update_best(
        self, judge: Judge, cand: Candidate, budget: int | None, calls: int
    ) -> int:
        """Update the global best with one comparison; return extra calls spent."""
        if self.best is None:
            self.best = cand
            return 0
        if budget is not None and calls >= budget:
            return 0
        if judge.compare(cand.seq, self.best.seq) == "A":
            self.best = cand
        return 1

    def tell(self, ranked_candidates: list[Candidate]) -> None:
        """Archive updates happen in :meth:`evaluate`; nothing more to do."""

    def as_dataframe(self) -> pd.DataFrame:
        """Return one row per occupied cell for heatmap plotting.

        Columns: one ``<axis>_bin`` and ``<axis>_center`` per axis, plus ``seq``,
        ``id``, ``length``, ``gc`` and ``generation``.
        """
        rows: list[dict[str, Any]] = []
        for cell, cand in self.archive.items():
            row: dict[str, Any] = {}
            for ax, idx in zip(self.axes, cell, strict=True):
                row[f"{ax.name}_bin"] = idx
                row[f"{ax.name}_center"] = ax.center(idx)
            row.update(
                id=cand.id,
                seq=cand.seq,
                length=len(cand.seq),
                gc=seqs.gc_content(cand.seq),
                generation=cand.generation,
            )
            rows.append(row)
        return pd.DataFrame(rows)

    def state_dict(self) -> dict[str, Any]:
        """Snapshot state, serializing the archive."""
        return {
            **self._base_state(),
            "scale": self.scale,
            "axis_cfg": self._axis_cfg,
            "archive": {
                ",".join(map(str, k)): _candidate_to_json(v) for k, v in self.archive.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state, rebuilding axes and archive."""
        self._load_base_state(state)
        self.scale = state["scale"]
        self._axis_cfg = [tuple(x) for x in state["axis_cfg"]]
        self._build_axes()
        self.archive = {}
        for key, cand_json in state["archive"].items():
            cell = tuple(int(x) for x in key.split(","))
            cand = _candidate_from_json(cand_json)
            if cand is not None:
                self.archive[cell] = cand
