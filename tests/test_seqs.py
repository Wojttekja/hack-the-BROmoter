"""Sequence utilities: parsing, composition, motifs, plausibility, shuffle."""

from __future__ import annotations

from collections import Counter

from promo import seqs


def test_read_fasta_tolerant(records) -> None:
    """The corpus parses with ids and key=value attrs from messy headers."""
    assert len(records) > 10
    r0 = records[0]
    assert set(r0.seq) <= set("ACGT")
    assert "len" in r0.attrs or "gc" in r0.attrs


def test_gc_content() -> None:
    """GC content is the fraction of G/C bases."""
    assert seqs.gc_content("GGCC") == 1.0
    assert seqs.gc_content("ATAT") == 0.0
    assert abs(seqs.gc_content("ACGT") - 0.5) < 1e-9


def test_kmer_vector_shape_and_sum() -> None:
    """Normalized k-mer vectors have length 4**k and sum ~1."""
    v = seqs.kmer_vector("ACGTACGT", k=4)
    assert v.shape == (256,)
    assert abs(v.sum() - 1.0) < 1e-6


def test_motif_counts() -> None:
    """Motif scanners find planted TATA/CCAAT/CreA sites."""
    s = "GGG" + "TATAAA" + "GGG" + "CCAAT" + "GGG" + "CCGGAG" + "GGG"
    assert seqs.count_tata(s) >= 1
    assert seqs.count_ccaat(s) == 1
    assert seqs.count_crea(s) >= 1


def test_dinuc_shuffle_preserves_length_and_alphabet() -> None:
    """Dinucleotide shuffle keeps length and alphabet."""
    s = seqs.random_seq(500, gc=0.5)
    shuf = seqs.dinuc_shuffle(s)
    assert len(shuf) == len(s)
    assert set(shuf) <= set("ACGT")
    assert Counter(shuf).keys() <= Counter(s).keys()


def test_plausibility_flags_bad_sequences() -> None:
    """Plausibility passes a normal sequence and flags degenerate ones."""
    import random

    ok_seq = seqs.random_seq(800, gc=0.5, rng=random.Random(12345))
    ok, reasons = seqs.plausibility(ok_seq)
    assert ok, reasons
    bad_ok, bad_reasons = seqs.plausibility("A" * 800)
    assert not bad_ok
    assert any("homopolymer" in r or "single base" in r or "GC" in r for r in bad_reasons)


def test_edit_distance() -> None:
    """Edit distance is 0 for equal strings and small for single edits."""
    assert seqs.edit_distance("ACGT", "ACGT") == 0
    assert seqs.edit_distance("ACGT", "ACCT") == 1
    assert seqs.edit_distance("ACGT", "ACG") == 1
