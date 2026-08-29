"""CMA-ES over latent space, driven by ranks only (never scores).

We wrap the ``cma`` package but never give it a numeric objective -- only the rank of
each candidate within its generation, obtained from pairwise Judge comparisons. To
stay correct when the latent space is hyperbolic, CMA-ES searches in the **tangent
space at a fixed base point** and every sample is mapped back onto the manifold via
the exponential map (``LatentOps.step``), so it never does illegal linear arithmetic
inside the Poincare ball.
"""

from __future__ import annotations

import base64
import pickle
from typing import Any

import numpy as np

from ..interfaces import Candidate
from .base import Optimizer


class CMAES(Optimizer):
    """Rank-based CMA-ES in the tangent space of the latent manifold."""

    def __init__(self, *args: Any, sigma0: float = 0.5, **kwargs: Any) -> None:
        """Initialize CMA-ES.

        Args:
            sigma0: Initial CMA step size (std of the tangent-space search).
        """
        super().__init__(*args, **kwargs)
        self.sigma0 = sigma0
        self._dim = self._infer_dim()
        # Base point: the geodesic-agnostic mean of seed latents (first seed's frame).
        self._base = np.asarray(self.seeds[0].latent, dtype=float)
        self._es: Any = None
        self._pop: list[tuple[np.ndarray, str]] = []

    def _infer_dim(self) -> int:
        """Determine tangent-space dimensionality from the navigator or a seed."""
        dim = getattr(self.nav, "dim", None)
        if isinstance(dim, int):
            return dim
        assert self.seeds[0].latent is not None
        return int(self.seeds[0].latent.shape[0])

    def _ensure_es(self) -> None:
        """Lazily construct the CMA strategy (import guarded)."""
        if self._es is not None:
            return
        import cma

        seed = int(self.rng.integers(1, 2**31 - 1))
        self._es = cma.CMAEvolutionStrategy(
            np.zeros(self._dim),
            self.sigma0,
            {"popsize": self.batch_size, "seed": seed, "verbose": -9},
        )

    def ask(self) -> list[Candidate]:
        """Sample a CMA population in tangent space and decode to candidates."""
        self._ensure_es()
        self.generation += 1
        solutions = self._es.ask()
        self._pop = []
        out: list[Candidate] = []
        for x in solutions:
            x = np.asarray(x, dtype=float)
            z = self.ops.step(self._base, x, 1.0)  # expmap in hyperbolic mode
            cand = self._decode(z, parent_id=self.seeds[0].id)
            self._pop.append((x, cand.id))
            out.append(cand)
        return out

    def tell(self, ranked_candidates: list[Candidate]) -> None:
        """Feed CMA the per-candidate ranks (0 = strongest) as fitness."""
        if self._es is None or not self._pop:
            return
        rank_of = {c.id: i for i, c in enumerate(ranked_candidates)}
        xs = [x for x, _ in self._pop]
        # Missing ids (e.g. dropped on budget) get worst rank.
        worst = len(ranked_candidates)
        fitnesses = [float(rank_of.get(cid, worst)) for _, cid in self._pop]
        self._es.tell(xs, fitnesses)
        self._pop = []

    def state_dict(self) -> dict[str, Any]:
        """Snapshot state, pickling the CMA strategy into a base64 string."""
        es_blob = None
        if self._es is not None:
            es_blob = base64.b64encode(pickle.dumps(self._es)).decode("ascii")
        return {
            **self._base_state(),
            "sigma0": self.sigma0,
            "dim": self._dim,
            "base": self._base.tolist(),
            "es_pickle": es_blob,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state, including the pickled CMA strategy if present."""
        self._load_base_state(state)
        self.sigma0 = state["sigma0"]
        self._dim = state["dim"]
        self._base = np.asarray(state["base"], dtype=float)
        self._pop = []
        if state.get("es_pickle"):
            self._es = pickle.loads(base64.b64decode(state["es_pickle"]))
        else:
            self._es = None
