"""Optimizers measurably improve the hidden ground truth within a call budget.

The key acceptance test: KingOfTheHill and CMAES both beat RandomSearch at an equal
oracle-call budget (their steering must pay for itself).
"""

from __future__ import annotations

import numpy as np
import pytest

from promo.interfaces import Candidate
from promo.latent import LatentOps
from promo.mock_backend import MockJudge, MockNavigator
from promo.optimizers import CMAES, KingOfTheHill, MapElites, Optimizer, RandomSearch


def _run_to_budget(
    cls: type[Optimizer],
    nav: MockNavigator,
    seeds: list[Candidate],
    budget: int,
    geometry: str = "euclidean",
    **kwargs: object,
) -> tuple[float, str]:
    """Run an optimizer to a fixed call budget; return (best_ground_truth, seq)."""
    judge = MockJudge(navigator=nav, seed=7)
    opt = cls(nav, seeds, latent_ops=LatentOps(geometry), batch_size=8, seed=3, **kwargs)
    while judge.n_calls < budget:
        cands = opt.ask()
        ranked, _ = opt.evaluate(judge, cands, budget - judge.n_calls)
        opt.tell(ranked)
    assert opt.best is not None
    return judge._ground_truth(opt.best.seq), opt.best.seq


@pytest.fixture
def opt_world() -> tuple[MockNavigator, list[Candidate], float]:
    """A navigator, seeds, and the native seed's ground-truth score."""
    from promo import seqs

    recs = seqs.read_fasta()
    nav = MockNavigator(corpus=[r.seq for r in recs], dim=32, geometry="euclidean")
    seeds = [Candidate(id=r.id, seq=r.seq, latent=nav.encode(r.seq)) for r in recs[:6]]
    native_gt = MockJudge(navigator=nav)._ground_truth(seeds[0].seq)
    return nav, seeds, native_gt


def test_all_optimizers_beat_native_seed(opt_world) -> None:
    """Every optimizer improves the hidden score over its starting seed."""
    nav, seeds, native_gt = opt_world
    for cls in (RandomSearch, KingOfTheHill, CMAES, MapElites):
        best_gt, _ = _run_to_budget(cls, nav, seeds, budget=300)
        assert best_gt > native_gt, f"{cls.__name__} did not beat native seed"


def test_koth_and_cmaes_beat_random(opt_world) -> None:
    """KingOfTheHill and CMAES both outperform RandomSearch at equal budget."""
    nav, seeds, _ = opt_world
    budget = 400
    random_gt, _ = _run_to_budget(RandomSearch, nav, seeds, budget)
    koth_gt, _ = _run_to_budget(KingOfTheHill, nav, seeds, budget)
    cmaes_gt, _ = _run_to_budget(CMAES, nav, seeds, budget)
    assert koth_gt > random_gt, f"KotH {koth_gt} !> random {random_gt}"
    assert cmaes_gt > random_gt, f"CMAES {cmaes_gt} !> random {random_gt}"


def test_optimizers_run_in_poincare_geometry(opt_world) -> None:
    """Optimizers improve over the seed in hyperbolic geometry too."""
    from promo import seqs

    recs = seqs.read_fasta()
    nav = MockNavigator(corpus=[r.seq for r in recs], dim=16, geometry="poincare")
    seeds = [Candidate(id=r.id, seq=r.seq, latent=nav.encode(r.seq)) for r in recs[:6]]
    native_gt = MockJudge(navigator=nav)._ground_truth(seeds[0].seq)
    for cls in (KingOfTheHill, CMAES):
        best_gt, _ = _run_to_budget(cls, nav, seeds, budget=250, geometry="poincare")
        assert best_gt > native_gt


def test_state_dict_roundtrip(opt_world) -> None:
    """Optimizer state serializes to JSON-safe dict and restores identically."""
    import json

    nav, seeds, _ = opt_world
    judge = MockJudge(navigator=nav, seed=7)
    opt = KingOfTheHill(nav, seeds, batch_size=8, seed=3)
    for _ in range(3):
        cands = opt.ask()
        ranked, _ = opt.evaluate(judge, cands, None)
        opt.tell(ranked)
    state = json.loads(json.dumps(opt.state_dict()))  # must be JSON-serializable
    restored = KingOfTheHill(nav, seeds, batch_size=8, seed=3)
    restored.load_state_dict(state)
    assert restored.generation == opt.generation
    assert restored.champion.seq == opt.champion.seq
    assert np.allclose(restored.step, opt.step)


def test_map_elites_archive_dataframe(opt_world) -> None:
    """MAP-Elites emits a non-empty archive DataFrame with axis columns."""
    nav, seeds, _ = opt_world
    judge = MockJudge(navigator=nav, seed=7)
    opt = MapElites(nav, seeds, batch_size=8, seed=3)
    for _ in range(6):
        cands = opt.ask()
        ranked, _ = opt.evaluate(judge, cands, None)
        opt.tell(ranked)
    df = opt.as_dataframe()
    assert not df.empty
    assert any(c.endswith("_bin") for c in df.columns)
    assert "seq" in df.columns
