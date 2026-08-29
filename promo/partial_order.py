"""Partial order over sequences from observed "stronger than" comparisons.

We accumulate directed edges ``winner -> loser`` and answer new queries from the
transitive closure before spending an oracle call. This is only valid if the Judge
is actually transitive, so :class:`TransitiveJudge` is **disabled by default** and
must be switched on only after the transitivity probe confirms it.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Literal

from .interfaces import Judge, Winner


class PartialOrder:
    """DAG of strict "stronger than" edges with reachability queries.

    An edge ``a -> b`` means ``a`` beat ``b``. ``implies(a, b)`` answers from
    reachability: if ``a`` reaches ``b`` then ``a`` wins; if ``b`` reaches ``a``
    then ``b`` wins; otherwise unknown.
    """

    def __init__(self) -> None:
        """Create an empty order."""
        self._succ: dict[str, set[str]] = defaultdict(set)
        self._edges: set[tuple[str, str]] = set()

    def add(self, winner: str, loser: str) -> None:
        """Record that ``winner`` beat ``loser``.

        Ignores self-loops and edges that would contradict an existing path (the
        first observation wins; contradictions are surfaced by ``contradicts``).
        """
        if winner == loser:
            return
        if self._reaches(loser, winner):
            # Adding winner->loser would create a cycle; keep the DAG acyclic and
            # drop the contradicting new edge. Callers can detect via contradicts().
            return
        self._succ[winner].add(loser)
        self._edges.add((winner, loser))

    def _reaches(self, src: str, dst: str) -> bool:
        """Whether ``dst`` is reachable from ``src`` via known edges (BFS)."""
        if src == dst:
            return True
        seen: set[str] = {src}
        queue: deque[str] = deque([src])
        while queue:
            node = queue.popleft()
            for nxt in self._succ.get(node, ()):
                if nxt == dst:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    def implies(self, a: str, b: str) -> Literal["A", "B"] | None:
        """Return the implied verdict for ``(a, b)`` from the closure, or ``None``."""
        if a == b:
            return None
        if self._reaches(a, b):
            return "A"
        if self._reaches(b, a):
            return "B"
        return None

    def contradicts(self, winner: str, loser: str) -> bool:
        """Whether recording ``winner > loser`` would contradict the closure."""
        return self._reaches(loser, winner)

    def needed_pairs(self, items: list[str]) -> list[tuple[str, str]]:
        """Return unordered pairs among ``items`` not yet implied by the closure.

        These are exactly the comparisons still worth spending a call on to fully
        rank ``items``.
        """
        pairs: list[tuple[str, str]] = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if self.implies(items[i], items[j]) is None:
                    pairs.append((items[i], items[j]))
        return pairs

    def __len__(self) -> int:
        """Number of stored (non-transitive) edges."""
        return len(self._edges)


class TransitiveJudge:
    """Judge wrapper that answers from a partial order before calling the oracle.

    Disabled unless ``enabled=True`` is passed explicitly, because consulting the
    closure is only sound when the Judge is transitive. When enabled, every real
    verdict also updates the order so future implied pairs are free.
    """

    def __init__(self, judge: Judge, *, enabled: bool = False) -> None:
        """Wrap ``judge``.

        Args:
            judge: Underlying (typically cached) Judge.
            enabled: Must be explicitly true to consult/extend the closure. When
                false this is a transparent pass-through that still learns edges.
        """
        self._judge = judge
        self.enabled = enabled
        self.order = PartialOrder()
        self.saved_calls = 0

    def compare(self, seq_a: str, seq_b: str) -> Winner:
        """Return a verdict, using the closure first when enabled."""
        if self.enabled:
            implied = self.order.implies(seq_a, seq_b)
            if implied is not None:
                self.saved_calls += 1
                return implied
        winner = self._judge.compare(seq_a, seq_b)
        if winner == "A":
            self.order.add(seq_a, seq_b)
        else:
            self.order.add(seq_b, seq_a)
        return winner
