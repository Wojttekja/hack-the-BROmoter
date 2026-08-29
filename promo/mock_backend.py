"""Fully synthetic oracles with a hidden ground truth for pre-hackathon rehearsal.

This is the most important file for preparation: it lets us measure whether our
optimizers actually work before we ever see the real HYPPE API.

``MockJudge`` decides comparisons from a **hidden** biological scoring function
(:meth:`MockJudge._ground_truth`). That function is used *only* by tests and
analysis; the real Judge exposes no score, so nothing in the optimization path may
call it. Every pathology of a real black-box oracle is available as an independently
switchable knob, all off by default, so the probe suite can be rehearsed against
each one.

``MockNavigator`` embeds sequences via k-mer counts through a fixed random
projection and decodes by blending against a codebook built from the promoter
corpus, so small latent steps yield small edit distances. It offers a Euclidean and
a Poincare-ball geometry mode to exercise our hyperbolic code paths in advance.
"""

from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import seqs
from .interfaces import BudgetExhaustedError, RateLimitError, Winner

# --- Ground-truth weights ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroundTruthWeights:
    """Weights for the hidden strength score. Tuned so every term matters."""

    tata: float = 1.6
    ccaat: float = 0.9
    ct_rich: float = 1.2
    gc_opt: float = 0.52
    gc_width: float = 0.12
    gc_weight: float = 1.4
    length_weight: float = 0.8
    length_sat: float = 1200.0
    homopolymer_penalty: float = 0.5
    crea_penalty: float = 1.1
    latent_weight: float = 1.0


@dataclass(slots=True)
class MockPathologies:
    """Independently switchable oracle pathologies. All off / neutral by default.

    Attributes:
        noise_prob: Per-call probability of flipping the verdict (breaks determinism).
        order_bias: Fraction of pairs for which the *first* argument always wins,
            chosen deterministically per unordered pair (breaks symmetry, keeps
            determinism).
        latency_ms: Simulated per-call latency in milliseconds.
        latency_jitter_ms: Uniform jitter added to ``latency_ms``.
        rate_limit_per_min: If set, raise :class:`RateLimitError` when more than this
            many calls occur within any rolling 60-second window.
        call_budget: If set, raise :class:`BudgetExhaustedError` after this many calls.
        intransitive: If true, add a rotational (non-transitive) component so 3-cycles
            appear among closely scored sequences.
        intransitive_strength: Magnitude of the rotational component.
    """

    noise_prob: float = 0.0
    order_bias: float = 0.0
    latency_ms: float = 0.0
    latency_jitter_ms: float = 0.0
    rate_limit_per_min: int | None = None
    call_budget: int | None = None
    intransitive: bool = False
    intransitive_strength: float = 2.0


def _stable_unit(text: str) -> float:
    """Deterministic value in ``[0, 1)`` from a string (stable across runs)."""
    digest = hashlib.sha256(text.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _stable_angle(seq: str) -> float:
    """Deterministic angle in ``[0, 2*pi)`` used for the intransitivity tournament."""
    return _stable_unit("angle:" + seq) * 2.0 * np.pi


class MockJudge:
    """Deterministic-by-default pairwise Judge driven by a hidden ground truth.

    Args:
        weights: Ground-truth term weights.
        pathologies: Which black-box pathologies to simulate.
        navigator: Optional navigator whose latent embedding feeds the smooth latent
            term, so latent-space search is meaningful. If omitted an internal
            embedding is used.
        seed: RNG seed for the (only stochastic) noise pathology.
    """

    def __init__(
        self,
        weights: GroundTruthWeights | None = None,
        pathologies: MockPathologies | None = None,
        navigator: MockNavigator | None = None,
        seed: int = 0,
    ) -> None:
        """Initialize the mock judge."""
        self.w = weights or GroundTruthWeights()
        self.p = pathologies or MockPathologies()
        self._nav = navigator
        self._rng = np.random.default_rng(seed)
        self._call_times: deque[float] = deque()
        self.n_calls = 0

    # --- Hidden ground truth (NOT available in the real API) ----------------

    def _latent_term(self, seq: str) -> float:
        """Smooth reward peaking at a fixed target latent point.

        This makes latent-space optimization meaningful: sequences whose embedding
        is near the hidden target score higher. Uses the navigator if one was
        supplied, else an internal deterministic embedding.
        """
        z = self._nav.encode(seq) if self._nav is not None else _internal_embed(seq)
        target = _hidden_latent_target(z.shape[0])
        dist2 = float(np.sum((z - target) ** 2))
        return float(np.exp(-dist2 / (2.0 * 0.5**2)))

    def _ground_truth(self, seq: str) -> float:
        """HIDDEN strength score. Available in the MOCK only.

        The real Judge returns no score; this exists solely so tests and analysis
        can measure optimizer progress. Never call it from the optimization path.

        The score is a weighted sum of biologically motivated terms:

        * TATA-like boxes in a plausible proximal window,
        * CCAAT boxes,
        * a CT-rich pyrimidine stretch near the 3' end,
        * a GC term with an interior optimum,
        * a mild monotone length preference saturating near 1200 bp,
        * penalties for long homopolymer runs and CreA repressor sites,
        * a smooth latent-space term.
        """
        w = self.w
        n = len(seq)
        if n == 0:
            return 0.0

        # TATA boxes, extra credit for one in the proximal window (20-160 bp from 3').
        tata_positions = seqs.find_motif(seq, seqs._TATA_RE)
        tata_score = min(len(tata_positions), 4) * 0.6
        prox_lo, prox_hi = max(0, n - 160), max(0, n - 20)
        if any(prox_lo <= pos <= prox_hi for pos in tata_positions):
            tata_score += 1.0
        tata_score *= w.tata

        ccaat_score = min(seqs.count_ccaat(seq), 4) * w.ccaat
        ct_score = seqs.ct_rich_fraction(seq) * w.ct_rich

        gc = seqs.gc_content(seq)
        gc_score = w.gc_weight * float(np.exp(-((gc - w.gc_opt) ** 2) / (2 * w.gc_width**2)))

        # Mild saturating length preference (tanh saturates near length_sat).
        length_score = w.length_weight * float(np.tanh(n / w.length_sat))

        homo = sum(1 for r in seqs.homopolymer_runs(seq, 6) if r[1] >= 8)
        homo_pen = w.homopolymer_penalty * homo
        crea_pen = w.crea_penalty * min(seqs.count_crea(seq), 5)

        latent_score = w.latent_weight * self._latent_term(seq)

        return (
            tata_score
            + ccaat_score
            + ct_score
            + gc_score
            + length_score
            + latent_score
            - homo_pen
            - crea_pen
        )

    # --- Pathology plumbing -------------------------------------------------

    def _enforce_limits(self) -> None:
        """Apply rate-limit and hard-budget pathologies; simulate latency."""
        p = self.p
        if p.call_budget is not None and self.n_calls >= p.call_budget:
            raise BudgetExhaustedError(f"call budget of {p.call_budget} exhausted")
        if p.rate_limit_per_min is not None:
            now = time.monotonic()
            while self._call_times and now - self._call_times[0] > 60.0:
                self._call_times.popleft()
            if len(self._call_times) >= p.rate_limit_per_min:
                raise RateLimitError(f"rate limit of {p.rate_limit_per_min}/min exceeded")
            self._call_times.append(now)
        if p.latency_ms > 0 or p.latency_jitter_ms > 0:
            jitter = self._rng.uniform(0, p.latency_jitter_ms) if p.latency_jitter_ms else 0.0
            time.sleep((p.latency_ms + jitter) / 1000.0)

    def _order_bias_first_wins(self, seq_a: str, seq_b: str) -> bool:
        """Whether this unordered pair is one where the first argument always wins."""
        if self.p.order_bias <= 0:
            return False
        key = "|".join(sorted((seq_a, seq_b)))
        return _stable_unit("orderbias:" + key) < self.p.order_bias

    def _intransitive_term(self, seq_a: str, seq_b: str) -> float:
        """Antisymmetric rotational signal favouring A when it 'rotationally' wins."""
        if not self.p.intransitive:
            return 0.0
        delta = (_stable_angle(seq_a) - _stable_angle(seq_b)) % (2 * np.pi)
        favors_a = 0.0 < delta < np.pi
        return self.p.intransitive_strength * (1.0 if favors_a else -1.0)

    def compare(self, seq_a: str, seq_b: str) -> Winner:
        """Return ``"A"`` or ``"B"`` per the hidden score plus active pathologies."""
        self._enforce_limits()
        self.n_calls += 1

        if self._order_bias_first_wins(seq_a, seq_b):
            base = 1.0  # first argument wins outright
        else:
            base = self._ground_truth(seq_a) - self._ground_truth(seq_b)
            base += self._intransitive_term(seq_a, seq_b)

        winner: Winner = "A" if base > 0 else "B"
        if self.p.noise_prob > 0 and self._rng.random() < self.p.noise_prob:
            winner = "B" if winner == "A" else "A"
        return winner


# --- Internal embedding for the latent term (navigator-independent) ---------

_INTERNAL_K = 3
_INTERNAL_DIM = 8


def _internal_projection() -> np.ndarray:
    """Fixed random projection from 3-mer space to a low-dim latent (seeded)."""
    rng = np.random.default_rng(1234)
    return rng.standard_normal((4**_INTERNAL_K, _INTERNAL_DIM)).astype(np.float32)


_INTERNAL_PROJ = _internal_projection()


def _internal_embed(seq: str) -> np.ndarray:
    """Deterministic low-dim embedding used by the latent term when no navigator."""
    return seqs.kmer_vector(seq, _INTERNAL_K) @ _INTERNAL_PROJ


def _hidden_latent_target(dim: int) -> np.ndarray:
    """The fixed 'good' latent point the ground-truth latent term rewards."""
    rng = np.random.default_rng(777)
    return rng.standard_normal(dim).astype(np.float32) * 0.3


# --- Mock Navigator ---------------------------------------------------------


@dataclass(slots=True)
class _Codebook:
    """Corpus-derived codebook mapping latent anchors to sequence fragments."""

    anchors: np.ndarray  # (n, dim) latent embeddings of corpus sequences
    seqs: list[str]  # aligned corpus sequences
    lengths: np.ndarray = field(default_factory=lambda: np.zeros(0))


class MockNavigator:
    """Deterministic sequence <-> latent codec with selectable geometry.

    Encoding: k-mer count vector -> fixed random projection -> ``dim`` dims. In
    ``poincare`` mode the vector is squashed into the open unit ball so hyperbolic
    code paths are exercised.

    Decoding: find the nearest codebook anchors and blend their sequences so that a
    small latent step produces a small edit distance.

    Args:
        corpus: Sequences to build the codebook from. Defaults to the bundled corpus.
        dim: Latent dimensionality.
        geometry: ``"euclidean"`` or ``"poincare"``.
        k: k-mer size for the encoder.
        seed: Seed for the fixed random projection.
    """

    def __init__(
        self,
        corpus: list[str] | None = None,
        *,
        dim: int = 32,
        geometry: str = "euclidean",
        k: int = 4,
        seed: int = 42,
    ) -> None:
        """Build the encoder projection and the decoder codebook."""
        if geometry not in ("euclidean", "poincare"):
            raise ValueError(f"unknown geometry: {geometry!r}")
        self._dim = dim
        self.geometry = geometry
        self._k = k
        if corpus is None:
            corpus = [r.seq for r in seqs.read_fasta()]
        self._corpus = corpus
        rng = np.random.default_rng(seed)
        self._proj = rng.standard_normal((4**k, dim)).astype(np.float32)
        # Scale so raw projected norms are ~O(1) before any ball squashing.
        self._proj /= np.sqrt(4**k)
        anchors = np.stack([self._raw_embed(s) for s in corpus])
        self._codebook = _Codebook(
            anchors=anchors,
            seqs=corpus,
            lengths=np.array([len(s) for s in corpus]),
        )
        # In poincare mode, spread the corpus across the ball so the *hyperbolic*
        # detection and search paths are genuinely exercised: pick a scale that maps
        # the largest raw embedding near the boundary (~0.95).
        self._spread = 1.0
        if geometry == "poincare":
            max_raw = float(np.max(np.linalg.norm(anchors, axis=1))) + 1e-9
            self._spread = float(np.arctanh(0.95) / max_raw)

    @property
    def dim(self) -> int:
        """Latent dimensionality."""
        return self._dim

    def _raw_embed(self, seq: str) -> np.ndarray:
        """Euclidean projection of the k-mer vector (pre-geometry)."""
        return seqs.kmer_vector(seq, self._k) @ self._proj

    def _to_ball(self, z: np.ndarray) -> np.ndarray:
        """Squash a Euclidean vector into the open unit ball (poincare mode)."""
        norm = float(np.linalg.norm(z))
        if norm < 1e-12:
            return z
        # tanh squashing keeps direction, maps the (spread) norm into [0, 1).
        target = np.tanh(norm * self._spread)
        return z * (target / norm)

    def encode(self, seq: str) -> np.ndarray:
        """Encode ``seq`` into a latent vector in the configured geometry."""
        z = self._raw_embed(seq)
        if self.geometry == "poincare":
            z = self._to_ball(z)
        return z.astype(np.float32)

    def _from_ball(self, z: np.ndarray) -> np.ndarray:
        """Invert :meth:`_to_ball` so decoding compares in raw embedding space."""
        norm = float(np.linalg.norm(z))
        if norm < 1e-12:
            return z
        clipped = min(norm, 1.0 - 1e-6)
        raw_norm = np.arctanh(clipped) / self._spread
        return z * (raw_norm / norm)

    def decode(self, z: np.ndarray) -> str:
        """Decode a latent vector to a sequence by blending nearest codebook anchors.

        The nearest anchor supplies the backbone; nearer-in-latent anchors nudge
        individual positions, so a small latent perturbation yields a small edit
        distance. Deterministic given ``z``.
        """
        z = np.asarray(z, dtype=np.float32)
        raw = self._from_ball(z) if self.geometry == "poincare" else z
        dists = np.linalg.norm(self._codebook.anchors - raw, axis=1)
        order = np.argsort(dists)
        base = self._codebook.seqs[int(order[0])]
        # Blend weight from latent proximity to the base determines mutation load.
        d0 = float(dists[order[0]])
        spread = float(np.median(dists)) + 1e-6
        mutate_frac = float(np.clip(d0 / spread, 0.0, 0.6))
        return self._blend(base, [self._codebook.seqs[int(i)] for i in order[1:4]], mutate_frac, z)

    def _blend(self, base: str, neighbours: list[str], frac: float, z: np.ndarray) -> str:
        """Mutate ``base`` toward ``neighbours`` at a rate set by ``frac``.

        Position choices are seeded by ``z`` so decoding is deterministic and nearby
        latent points share most mutations (hence small edit distance between them).
        """
        if not neighbours or frac <= 0:
            return base
        seed = int(abs(float(np.sum(z)) * 1e6)) % (2**32)
        rng = np.random.default_rng(seed)
        chars = list(base)
        n_mut = int(len(chars) * frac)
        if n_mut == 0:
            return base
        positions = rng.choice(len(chars), size=min(n_mut, len(chars)), replace=False)
        for pos in positions:
            donor = neighbours[rng.integers(len(neighbours))]
            if pos < len(donor):
                chars[pos] = donor[pos]
        return "".join(chars)

    def distance(self, z1: np.ndarray, z2: np.ndarray) -> float:
        """Latent distance appropriate to the geometry."""
        z1 = np.asarray(z1, dtype=float)
        z2 = np.asarray(z2, dtype=float)
        if self.geometry == "poincare":
            from .latent import poincare_distance

            return poincare_distance(z1, z2)
        return float(np.linalg.norm(z1 - z2))
