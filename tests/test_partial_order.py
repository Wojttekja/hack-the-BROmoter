"""Transitive closure correctness and TransitiveJudge behaviour."""

from __future__ import annotations

from promo.interfaces import Winner
from promo.partial_order import PartialOrder, TransitiveJudge


def test_transitive_closure_infers_and_never_contradicts() -> None:
    """A>B, B>C implies A>C, and the closure never flips an observed edge."""
    po = PartialOrder()
    po.add("A", "B")
    po.add("B", "C")
    assert po.implies("A", "C") == "A"
    assert po.implies("C", "A") == "B"
    assert po.implies("A", "B") == "A"
    # An unrelated pair stays unknown.
    assert po.implies("A", "Z") is None
    # A contradicting edge (C>A) is rejected, closure stays consistent.
    assert po.contradicts("C", "A")
    po.add("C", "A")
    assert po.implies("A", "C") == "A"  # unchanged


def test_needed_pairs_shrinks_with_closure() -> None:
    """needed_pairs excludes pairs already implied by transitivity."""
    po = PartialOrder()
    items = ["A", "B", "C"]
    assert len(po.needed_pairs(items)) == 3
    po.add("A", "B")
    po.add("B", "C")
    # A>C now implied, so only... actually all three implied now.
    assert po.needed_pairs(items) == []


def test_transitive_judge_saves_calls_when_enabled() -> None:
    """Enabled TransitiveJudge answers implied pairs without a real call."""

    class Raw:
        def __init__(self) -> None:
            self.calls = 0

        def compare(self, a: str, b: str) -> Winner:
            self.calls += 1
            # total order by single-char name
            return "A" if a < b else "B"

    raw = Raw()
    tj = TransitiveJudge(raw, enabled=True)
    assert tj.compare("A", "B") == "A"
    assert tj.compare("B", "C") == "A"
    calls_before = raw.calls
    # A vs C is implied (A>B>C); no new real call.
    assert tj.compare("A", "C") == "A"
    assert raw.calls == calls_before
    assert tj.saved_calls == 1


def test_transitive_judge_disabled_is_passthrough() -> None:
    """Disabled TransitiveJudge always calls the underlying judge."""

    class Raw:
        def __init__(self) -> None:
            self.calls = 0

        def compare(self, a: str, b: str) -> Winner:
            self.calls += 1
            return "A" if a < b else "B"

    raw = Raw()
    tj = TransitiveJudge(raw, enabled=False)
    tj.compare("A", "B")
    tj.compare("B", "C")
    tj.compare("A", "C")
    assert raw.calls == 3
    assert tj.saved_calls == 0
