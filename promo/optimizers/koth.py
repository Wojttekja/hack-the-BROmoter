"""KingOfTheHill: a single champion defended one comparison at a time.

The cheapest steerable optimizer: each generation proposes challengers by perturbing
the champion, and each challenger costs exactly one comparison against the reigning
champion. The perturbation scale grows on a win streak and shrinks on losses; when
progress stalls (or the scale collapses) the search restarts from a random seed. The
global best is preserved across restarts.
"""

from __future__ import annotations

from typing import Any

from ..interfaces import Candidate, Judge
from ..ranking import compare_to_champion
from .base import Optimizer, _candidate_from_json, _candidate_to_json


class KingOfTheHill(Optimizer):
    """Champion-vs-challenger local search with adaptive step and restarts."""

    def __init__(
        self,
        *args: Any,
        step: float = 0.5,
        grow: float = 1.3,
        shrink: float = 0.7,
        min_step: float = 0.02,
        max_step: float = 3.0,
        patience: int = 6,
        **kwargs: Any,
    ) -> None:
        """Initialize champion search.

        Args:
            step: Initial perturbation scale.
            grow: Multiplier applied to ``step`` after a win.
            shrink: Multiplier applied to ``step`` after a loss.
            min_step: Restart trigger; step is never allowed below this.
            max_step: Upper clamp on step.
            patience: Restart after this many consecutive lossy generations.
        """
        super().__init__(*args, **kwargs)
        self.step = step
        self.grow = grow
        self.shrink = shrink
        self.min_step = min_step
        self.max_step = max_step
        self.patience = patience
        self.no_improve = 0
        self._last_won = False
        # Champion starts at the strongest-guess seed (first seed by convention).
        self.champion: Candidate = self.seeds[0]

    def ask(self) -> list[Candidate]:
        """Propose challengers by perturbing the champion at the current scale."""
        self.generation += 1
        assert self.champion.latent is not None
        out: list[Candidate] = []
        for _ in range(self.batch_size):
            z = self.ops.perturb(self.champion.latent, self.step, self.rng)
            out.append(self._decode(z, parent_id=self.champion.id))
        return out

    def evaluate(
        self, judge: Judge, candidates: list[Candidate], budget: int | None
    ) -> tuple[list[Candidate], int]:
        """Defend the champion against each challenger; fold winner into best."""
        by_seq = {c.seq: c for c in candidates}
        champ_seq, calls = compare_to_champion(
            judge, self.champion.seq, [c.seq for c in candidates], budget
        )
        won = champ_seq != self.champion.seq
        if won:
            self.champion = by_seq[champ_seq]
            self.no_improve = 0
            if self.best is None:
                self.best = self.champion
            else:
                remaining = None if budget is None else budget - calls
                if remaining is None or remaining > 0:
                    if judge.compare(self.champion.seq, self.best.seq) == "A":
                        self.best = self.champion
                    calls += 1
        else:
            self.no_improve += 1
            if self.best is None:
                self.best = self.champion
        self._last_won = won
        others = [c for c in candidates if c.seq != champ_seq]
        ranked = [self.champion, *others]
        return ranked, calls

    def tell(self, ranked_candidates: list[Candidate]) -> None:
        """Adapt the step size and restart when progress stalls."""
        self.step *= self.grow if self._last_won else self.shrink
        self.step = float(min(max(self.step, self.min_step), self.max_step))
        if self.no_improve >= self.patience or self.step <= self.min_step:
            self._restart()

    def _restart(self) -> None:
        """Jump the champion to a random seed and reset the step (best preserved)."""
        idx = int(self.rng.integers(len(self.seeds)))
        self.champion = self.seeds[idx]
        self.step = float(min(1.0, self.max_step))
        self.no_improve = 0

    def state_dict(self) -> dict[str, Any]:
        """Snapshot state."""
        return {
            **self._base_state(),
            "step": self.step,
            "grow": self.grow,
            "shrink": self.shrink,
            "min_step": self.min_step,
            "max_step": self.max_step,
            "patience": self.patience,
            "no_improve": self.no_improve,
            "last_won": self._last_won,
            "champion": _candidate_to_json(self.champion),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore state."""
        self._load_base_state(state)
        self.step = state["step"]
        self.grow = state["grow"]
        self.shrink = state["shrink"]
        self.min_step = state["min_step"]
        self.max_step = state["max_step"]
        self.patience = state["patience"]
        self.no_improve = state["no_improve"]
        self._last_won = state["last_won"]
        champ = _candidate_from_json(state["champion"])
        assert champ is not None
        self.champion = champ
