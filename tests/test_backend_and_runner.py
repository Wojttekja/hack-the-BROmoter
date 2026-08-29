"""Backend swap point and runner checkpoint/resume behaviour."""

from __future__ import annotations

from pathlib import Path

from promo import backend
from promo.interfaces import Judge, Navigator
from promo.runner import RunnerConfig, run


def test_backend_swap_point_env(monkeypatch) -> None:
    """get_judge/get_navigator honour PROMO_BACKEND and the explicit override."""
    monkeypatch.setenv("PROMO_BACKEND", "mock")
    j = backend.get_judge()
    n = backend.get_navigator()
    assert isinstance(j, Judge)
    assert isinstance(n, Navigator)
    # Explicit override wins over the env var.
    assert isinstance(backend.get_judge("mock"), Judge)


def test_backend_rejects_unknown(monkeypatch) -> None:
    """An unknown backend name raises rather than silently defaulting."""
    import pytest

    with pytest.raises(ValueError):
        backend.get_judge("nonsense")


def _cfg(out: Path, budget: int, **kw) -> RunnerConfig:
    return RunnerConfig(
        optimizer="koth",
        backend="mock",
        budget=budget,
        wall=float("inf"),
        batch_size=8,
        geometry="euclidean",
        n_seeds=6,
        checkpoint_interval=0.0,  # checkpoint every generation
        cache_path=str(out / "cache.jsonl"),
        checkpoint_path=str(out / "ckpt.json"),
        log_path=str(out / "events.jsonl"),
        assume_symmetric=False,
        transitive=False,
        seed=0,
        **kw,
    )


def test_runner_checkpoint_and_resume(tmp_path: Path) -> None:
    """A run resumes from checkpoint and continues accumulating cached calls."""
    cfg = _cfg(tmp_path, budget=80)
    best1 = run(cfg, resume=False)
    assert (tmp_path / "ckpt.json").exists()
    lines_after_first = (tmp_path / "cache.jsonl").read_text().count("\n")
    assert lines_after_first > 0

    # Resume with a larger budget; the cache file must only grow (no data loss).
    cfg2 = _cfg(tmp_path, budget=160)
    best2 = run(cfg2, resume=True)
    lines_after_resume = (tmp_path / "cache.jsonl").read_text().count("\n")
    assert lines_after_resume >= lines_after_first
    assert best2 is not None and best1 is not None


def test_runner_produces_improvement(tmp_path: Path) -> None:
    """The runner's best beats the native seed's hidden ground truth."""
    from promo import seqs
    from promo.mock_backend import MockJudge

    cfg = _cfg(tmp_path, budget=300)
    best = run(cfg, resume=False)
    recs = seqs.read_fasta()
    # Measure with the SAME ground truth the runner's judge used: get_judge("mock")
    # builds MockJudge() with no navigator, i.e. the internal-embedding latent term.
    gt = MockJudge()._ground_truth
    native = next(r.seq for r in recs if seqs.NATIVE_PKS1_LOCUS in r.id)
    assert gt(best.seq) > gt(native)
