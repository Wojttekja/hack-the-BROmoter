"""Shared pytest fixtures: a small deterministic mock world for fast tests."""

from __future__ import annotations

import pytest

from promo import seqs
from promo.interfaces import Candidate, Judge, Winner
from promo.mock_backend import MockJudge, MockNavigator


@pytest.fixture(scope="session")
def corpus() -> list[str]:
    """A small slice of the real promoter corpus for speed."""
    return [r.seq for r in seqs.read_fasta()][:24]


@pytest.fixture(scope="session")
def records() -> list[seqs.FastaRecord]:
    """The full parsed corpus records."""
    return seqs.read_fasta()


@pytest.fixture
def navigator(corpus: list[str]) -> MockNavigator:
    """A Euclidean mock navigator over the small corpus."""
    return MockNavigator(corpus=corpus, dim=16, geometry="euclidean", seed=1)


@pytest.fixture
def judge(navigator: MockNavigator) -> MockJudge:
    """A clean (no-pathology) mock judge tied to the navigator's latent term."""
    return MockJudge(navigator=navigator, seed=1)


@pytest.fixture
def seeds(navigator: MockNavigator, corpus: list[str]) -> list[Candidate]:
    """Seed candidates with latents for the optimizers."""
    return [Candidate(id=f"s{i}", seq=s, latent=navigator.encode(s))
            for i, s in enumerate(corpus[:6])]


class CountingJudge:
    """A deterministic lexicographic Judge that counts calls, for ranking tests.

    ``a`` beats ``b`` iff ``a > b`` lexicographically, which is a total order, so
    correct ranking is unambiguous and transitivity holds.
    """

    def __init__(self) -> None:
        self.calls = 0

    def compare(self, seq_a: str, seq_b: str) -> Winner:
        """Count and return the lexicographic verdict."""
        self.calls += 1
        return "A" if seq_a > seq_b else "B"


@pytest.fixture
def counting_judge() -> CountingJudge:
    """A call-counting deterministic judge."""
    return CountingJudge()


def ground_truth_of(judge: Judge, seq: str) -> float:
    """Helper to read the mock hidden ground truth in tests."""
    return judge._ground_truth(seq)  # type: ignore[attr-defined]
