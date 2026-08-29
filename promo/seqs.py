"""Sequence utilities: tolerant FASTA IO, composition, motifs, plausibility.

Pure functions over nucleotide strings. No oracle access here. Everything that
computes on raw DNA (optimizers, probes, surrogate features) routes through this
module so motif definitions live in exactly one place.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

ALPHABET = "ACGT"
_VALID = set(ALPHABET)

# Repository-root-relative default corpus. Resolved lazily so imports are cheap.
_DEFAULT_FASTA = (
    Path(__file__).resolve().parent.parent / "native_promoter" / "data" / "promoters.fasta"
)

# Locus tag of the native pks1 (polyketide synthase) promoter in the corpus.
# See DECISIONS.md for why this one.
NATIVE_PKS1_LOCUS = "TRIATDRAFT_289254"


@dataclass(frozen=True, slots=True)
class FastaRecord:
    """One FASTA entry with its raw header and parsed key=value attributes."""

    id: str
    seq: str
    header: str
    attrs: dict[str, str]


def _clean_seq(raw: str) -> str:
    """Uppercase and strip everything that is not a nucleotide letter."""
    return "".join(c for c in raw.upper() if c in _VALID)


def read_fasta(path: str | Path | None = None) -> list[FastaRecord]:
    """Read a FASTA file, tolerating messy headers and stray whitespace.

    The reader accepts blank lines, arbitrary header text after ``>`` and any
    casing. Header attributes of the form ``key=value`` are parsed into ``attrs``;
    the first whitespace-delimited token after ``>`` becomes the record ``id``.

    Args:
        path: FASTA path. Defaults to the bundled Trichoderma promoter corpus.

    Returns:
        Records in file order. Entries whose sequence is empty are skipped.
    """
    fpath = Path(path) if path is not None else _DEFAULT_FASTA
    records: list[FastaRecord] = []
    header: str | None = None
    chunks: list[str] = []

    def flush() -> None:
        if header is None:
            return
        seq = _clean_seq("".join(chunks))
        if not seq:
            return
        tokens = header.split()
        rid = tokens[0] if tokens else f"seq{len(records)}"
        attrs: dict[str, str] = {}
        for tok in tokens[1:]:
            if "=" in tok:
                key, _, val = tok.partition("=")
                attrs[key] = val
        records.append(FastaRecord(id=rid, seq=seq, header=header, attrs=attrs))

    with fpath.open() as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith(">"):
                flush()
                header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line.strip())
    flush()
    return records


def write_fasta(records: list[tuple[str, str]], path: str | Path, width: int = 70) -> None:
    """Write ``(header, seq)`` pairs to a FASTA file with wrapped lines."""
    fpath = Path(path)
    with fpath.open("w") as fh:
        for header, seq in records:
            fh.write(f">{header}\n")
            for i in range(0, len(seq), width):
                fh.write(seq[i : i + width] + "\n")


def gc_content(seq: str) -> float:
    """Return the GC fraction in ``[0, 1]``; empty sequence returns 0.0."""
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c in "GC")
    return gc / len(seq)


def kmer_vector(seq: str, k: int = 4, *, normalize: bool = True) -> np.ndarray:
    """Return the length-``4**k`` k-mer count vector in fixed lexicographic order.

    Args:
        seq: Nucleotide sequence.
        k: k-mer length.
        normalize: If true, divide by the number of k-mers so the vector sums to 1.

    Returns:
        Dense float32 vector indexed by k-mer rank.
    """
    index = {"".join(p): i for i, p in enumerate(product(ALPHABET, repeat=k))}
    vec = np.zeros(len(index), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        idx = index.get(seq[i : i + k])
        if idx is not None:
            vec[idx] += 1.0
    total = len(seq) - k + 1
    if normalize and total > 0:
        vec /= total
    return vec


def kmer_vector_multi(seq: str, ks: tuple[int, ...] = (4, 5, 6)) -> np.ndarray:
    """Concatenate normalized k-mer vectors for several ``k`` values."""
    return np.concatenate([kmer_vector(seq, k) for k in ks])


# --- Motif scanning ---------------------------------------------------------

# TATA-like box: TATA[AT]A[AT] core, allow the canonical TATAAA / TATATAA family.
_TATA_RE = re.compile(r"TATA[AT]A[AT]?")
_CCAAT_RE = re.compile(r"CCAAT")
# CreA carbon-catabolite repressor site: SYGGRG (S=[CG], Y=[CT], R=[AG]).
_CREA_RE = re.compile(r"[CG][CT]GG[AG]G")
_HOMOPOLYMER_RE = re.compile(r"(A{6,}|C{6,}|G{6,}|T{6,})")


def find_motif(seq: str, pattern: re.Pattern[str]) -> list[int]:
    """Return start positions of all (overlapping) matches of ``pattern``."""
    return [m.start() for m in re.finditer(f"(?=({pattern.pattern}))", seq)]


def count_tata(seq: str) -> int:
    """Count TATA-like boxes."""
    return len(find_motif(seq, _TATA_RE))


def count_ccaat(seq: str) -> int:
    """Count CCAAT boxes."""
    return len(find_motif(seq, _CCAAT_RE))


def count_crea(seq: str) -> int:
    """Count CreA repressor sites (SYGGRG)."""
    return len(find_motif(seq, _CREA_RE))


def count_atg_in_tail(seq: str, tail: int = 200) -> int:
    """Count ATG occurrences in the final ``tail`` bp (spurious start codons)."""
    region = seq[-tail:] if len(seq) > tail else seq
    return len(find_motif(region, re.compile("ATG")))


def homopolymer_runs(seq: str, min_len: int = 6) -> list[tuple[int, int, str]]:
    """Return ``(start, length, base)`` for each homopolymer run >= ``min_len``."""
    runs: list[tuple[int, int, str]] = []
    for m in re.finditer(r"(A+|C+|G+|T+)", seq):
        if m.end() - m.start() >= min_len:
            runs.append((m.start(), m.end() - m.start(), m.group()[0]))
    return runs


def has_simple_repeat(seq: str, unit_max: int = 3, min_copies: int = 5) -> bool:
    """Return True if a short tandem repeat (unit up to ``unit_max`` bp) is present."""
    for unit in range(1, unit_max + 1):
        pat = re.compile(rf"(.{{{unit}}})\1{{{min_copies - 1},}}")
        if pat.search(seq):
            return True
    return False


def ct_rich_fraction(seq: str, window: int = 40) -> float:
    """Return the max CT fraction over any ``window``-bp stretch upstream of the 3' end.

    A CT-rich pyrimidine stretch near the 3' end is a weak core-promoter signal.
    """
    if len(seq) < window:
        return sum(1 for c in seq if c in "CT") / max(len(seq), 1)
    best = 0.0
    # Search the downstream (3') half where the CT stretch is biologically relevant.
    start = max(0, len(seq) - 3 * window)
    for i in range(start, len(seq) - window + 1):
        frac = sum(1 for c in seq[i : i + window] if c in "CT") / window
        best = max(best, frac)
    return best


# --- Shuffling & plausibility ----------------------------------------------


def dinuc_shuffle(seq: str, rng: random.Random | None = None) -> str:
    """Return a dinucleotide-preserving shuffle via an Eulerian-path walk.

    Preserves the exact dinucleotide composition of ``seq`` (Altschul-Erikson),
    which is the correct null model for motif-enrichment and shuffle tests.
    """
    rng = rng or random.Random()
    if len(seq) < 2:
        return seq
    # Build the multigraph of successor edges keyed by nucleotide.
    edges: dict[str, list[str]] = {c: [] for c in ALPHABET}
    for a, b in zip(seq, seq[1:], strict=False):
        if a in edges:
            edges[a].append(b)
    for lst in edges.values():
        rng.shuffle(lst)

    last = seq[-1]
    # Randomize each vertex's edge order but keep one edge toward `last` last so an
    # Eulerian trail exists (standard Altschul-Erikson trick, best-effort).
    out = [seq[0]]
    cursor = seq[0]
    remaining = {c: list(v) for c, v in edges.items()}
    for _ in range(len(seq) - 1):
        succ = remaining.get(cursor)
        if not succ:
            # Fell off the trail; append remaining bases arbitrarily.
            leftover = [b for lst in remaining.values() for b in lst]
            rng.shuffle(leftover)
            out.extend(leftover)
            break
        nxt = succ.pop()
        out.append(nxt)
        cursor = nxt
    _ = last
    return "".join(out[: len(seq)])


def random_seq(n: int, gc: float = 0.5, rng: random.Random | None = None) -> str:
    """Generate a random sequence of length ``n`` at target GC fraction."""
    rng = rng or random.Random()
    weights = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]  # A C G T
    return "".join(rng.choices(ALPHABET, weights=weights, k=n))


@dataclass(frozen=True, slots=True)
class PlausibilityReport:
    """Structured plausibility verdict for reporting defensible vs unfiltered best."""

    ok: bool
    reasons: list[str]


def plausibility(
    seq: str,
    *,
    min_len: int = 150,
    max_len: int = 2000,
    gc_lo: float = 0.25,
    gc_hi: float = 0.75,
) -> tuple[bool, list[str]]:
    """Judge whether a sequence looks like a biologically reasonable promoter.

    This is a *sanity gate*, not a strength predictor. It flags sequences we would
    be embarrassed to submit so we can report "unfiltered best" vs "defensible
    best" separately.

    Args:
        seq: Candidate sequence.
        min_len: Minimum acceptable length.
        max_len: Maximum acceptable length.
        gc_lo: Minimum acceptable GC fraction.
        gc_hi: Maximum acceptable GC fraction.

    Returns:
        ``(ok, reasons)`` where ``reasons`` lists every failed check.
    """
    reasons: list[str] = []
    if any(c not in _VALID for c in seq):
        reasons.append("non-ACGT characters present")
    n = len(seq)
    if n < min_len:
        reasons.append(f"too short ({n} < {min_len} bp)")
    if n > max_len:
        reasons.append(f"too long ({n} > {max_len} bp)")
    gc = gc_content(seq)
    if gc < gc_lo:
        reasons.append(f"GC too low ({gc:.2f} < {gc_lo})")
    if gc > gc_hi:
        reasons.append(f"GC too high ({gc:.2f} > {gc_hi})")
    long_runs = [r for r in homopolymer_runs(seq, 6) if r[1] >= 10]
    if long_runs:
        reasons.append(f"{len(long_runs)} homopolymer run(s) >= 10 bp")
    if has_simple_repeat(seq, unit_max=3, min_copies=8):
        reasons.append("low-complexity tandem repeat")
    # Composition collapse: any single base dominating.
    counts = Counter(seq)
    if n and max(counts.values()) / n > 0.55:
        reasons.append("single base > 55% of sequence")
    return (len(reasons) == 0, reasons)


def edit_distance(a: str, b: str, cap: int | None = None) -> int:
    """Levenshtein distance with an optional early-exit ``cap``.

    Uses the standard two-row dynamic program. If ``cap`` is set and the distance
    provably exceeds it, returns ``cap + 1`` early.
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > (cap if cap is not None else max(la, lb)):
        return (cap + 1) if cap is not None else abs(la - lb)
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            row_min = min(row_min, cur[j])
        if cap is not None and row_min > cap:
            return cap + 1
        prev = cur
    return prev[lb]
