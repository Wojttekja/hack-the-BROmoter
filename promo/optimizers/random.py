"""RandomSearch: the baseline optimizer every other must beat.

Samples latent perturbations around the seed candidates and decodes them. It has no
memory of what worked, so it is the honest control for the improvement tests.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..interfaces import Candidate
from .base import Optimizer


class RandomSearch(Optimizer):
    """Perturb random seeds each generation; keep only the global best."""

    def __init__(self, *args: Any, scale: float = 0.5, **kwargs: Any) -> None:
        """Initialize with a fixed perturbation ``scale``."""
        super().__init__(*args, **kwargs)
        self.scale = scale

    def ask(self) -> list[Candidate]:
        """Sample ``batch_size`` perturbations of randomly chosen seeds."""
        self.generation += 1
        out: list[Candidate] = []
        for _ in range(self.batch_size):
            seed = self.seeds[self.rng.integers(len(self.seeds))]
            assert seed.latent is not None
            z = self.ops.perturb(seed.latent, self.scale, self.rng)
            out.append(self._decode(z, parent_id=seed.id))
        return out

    def tell(self, ranked_candidates: list[Candidate]) -> None:
        """No internal state to update; the base tracks the global best."""

    def state_dict(self) -> dict[str, Any]:
        """Snapshot state."""
        return {**self._base_state(), "scale": self.scale}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state."""
        self._load_base_state(state)
        self.scale = state["scale"]

    def _seed_latents(self) -> np.ndarray:
        """Return the stacked seed latents (used by tests/analysis)."""
        return np.stack([s.latent for s in self.seeds if s.latent is not None])
