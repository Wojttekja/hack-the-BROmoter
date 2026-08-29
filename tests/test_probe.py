"""The probe suite detects each injected pathology (and none when clean)."""

from __future__ import annotations

from pathlib import Path

import pytest

from promo import seqs
from promo.mock_backend import MockJudge, MockNavigator, MockPathologies
from promo.probe import (
    ProbeContext,
    probe_determinism,
    probe_latency_and_limits,
    probe_order_bias,
    probe_transitivity,
)


@pytest.fixture(scope="module")
def world():
    """Records and a shared navigator for probe pathology tests."""
    recs = seqs.read_fasta()
    nav = MockNavigator(corpus=[r.seq for r in recs], dim=16)
    return recs, nav


def _ctx(recs, nav, path: MockPathologies, tmp: Path) -> ProbeContext:
    judge = MockJudge(navigator=nav, pathologies=path, seed=1)
    return ProbeContext(judge=judge, navigator=nav, records=recs, out_dir=tmp)


def test_detects_noise(world, tmp_path) -> None:
    """Noise injection breaks determinism; clean judge stays deterministic."""
    recs, nav = world
    noisy = probe_determinism(_ctx(recs, nav, MockPathologies(noise_prob=0.3), tmp_path))
    clean = probe_determinism(_ctx(recs, nav, MockPathologies(), tmp_path))
    assert noisy["deterministic"] is False
    assert clean["deterministic"] is True


def test_detects_order_bias(world, tmp_path) -> None:
    """Order-bias injection is flagged; a clean judge is not."""
    recs, nav = world
    biased = probe_order_bias(_ctx(recs, nav, MockPathologies(order_bias=0.8), tmp_path))
    clean = probe_order_bias(_ctx(recs, nav, MockPathologies(), tmp_path))
    assert biased["order_bias_detected"] is True
    assert clean["order_bias_detected"] is False


def test_detects_intransitivity(world, tmp_path) -> None:
    """Intransitivity injection produces 3-cycles; a clean judge produces none."""
    recs, nav = world
    cyc = probe_transitivity(
        _ctx(recs, nav, MockPathologies(intransitive=True, intransitive_strength=5.0), tmp_path)
    )
    clean = probe_transitivity(_ctx(recs, nav, MockPathologies(), tmp_path))
    assert cyc["n_cycles"] > 0
    assert clean["transitive"] is True


def test_detects_rate_limit(world, tmp_path) -> None:
    """A rate limit is detected under rapid calls."""
    recs, nav = world
    res = probe_latency_and_limits(
        _ctx(recs, nav, MockPathologies(rate_limit_per_min=10), tmp_path), budget=40
    )
    assert res["rate_limited"] is True


def test_detects_hard_budget(world, tmp_path) -> None:
    """A hard call budget is detected as exhaustion."""
    recs, nav = world
    res = probe_latency_and_limits(
        _ctx(recs, nav, MockPathologies(call_budget=12), tmp_path), budget=40
    )
    assert res["budget_exhausted"] is True


def test_detects_latency(world, tmp_path) -> None:
    """Simulated latency shows up in the measured mean latency."""
    recs, nav = world
    res = probe_latency_and_limits(
        _ctx(recs, nav, MockPathologies(latency_ms=5.0), tmp_path), budget=10
    )
    assert res["mean_latency_ms"] >= 4.0
