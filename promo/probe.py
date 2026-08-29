"""Oracle interrogation suite: learn the black box before trusting it.

Runs a battery of experiments against whichever backend is configured and writes
both a JSON report and a human-readable rich table. The probes deliberately use the
**raw** (uncached) Judge from :mod:`promo.backend`, because determinism, order bias
and latency are only observable when identical calls actually reach the oracle.

Every probe respects a ``--budget`` and is individually runnable::

    python -m promo.probe --backend mock --budget 400            # run all
    python -m promo.probe --backend mock --probe order_bias      # just one
    python -m promo.probe --backend mock --probe length_bias --budget 60

The outcomes drive config decisions on the morning of the event (see README's
"MORNING OF THE HACKATHON" checklist).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from scipy import stats  # noqa: E402

from . import backend, seqs  # noqa: E402
from .interfaces import (  # noqa: E402
    BudgetExhaustedError,
    Judge,
    Navigator,
    OracleError,
    RateLimitError,
)
from .latent import LatentOps, detect_geometry  # noqa: E402


class _Guard:
    """Counts Judge calls and stops a probe cleanly at its budget."""

    def __init__(self, judge: Judge, budget: int | None) -> None:
        self.judge = judge
        self.budget = budget
        self.calls = 0

    def left(self) -> bool:
        """Whether budget remains."""
        return self.budget is None or self.calls < self.budget

    def compare(self, a: str, b: str) -> str:
        """A budgeted comparison."""
        self.calls += 1
        return self.judge.compare(a, b)


@dataclass(slots=True)
class ProbeContext:
    """Shared inputs for every probe."""

    judge: Judge
    navigator: Navigator
    records: list[seqs.FastaRecord]
    out_dir: Path
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def seqs_list(self) -> list[str]:
        """All corpus sequences."""
        return [r.seq for r in self.records]


# --- Judge probes -----------------------------------------------------------


def probe_determinism(
    ctx: ProbeContext, budget: int | None = 60, repeats: int = 8
) -> dict[str, Any]:
    """Repeat the same pairs many times; a flip means the Judge is non-deterministic."""
    g = _Guard(ctx.judge, budget)
    seqlist = ctx.seqs_list()
    pairs = [(seqlist[i], seqlist[-(i + 1)]) for i in range(min(4, len(seqlist) // 2))]
    consistent = 0
    total = 0
    for a, b in pairs:
        verdicts = []
        for _ in range(repeats):
            if not g.left():
                break
            verdicts.append(g.compare(a, b))
        if verdicts:
            total += 1
            if len(set(verdicts)) == 1:
                consistent += 1
    frac = consistent / total if total else 1.0
    return {
        "consistent_fraction": frac,
        "deterministic": frac >= 0.999,
        "pairs_tested": total,
        "repeats": repeats,
        "calls": g.calls,
    }


def probe_order_bias(ctx: ProbeContext, budget: int | None = 120) -> dict[str, Any]:
    """Compare ``(a,b)`` vs ``(b,a)``; a first-position tendency signals order bias."""
    g = _Guard(ctx.judge, budget)
    seqlist = ctx.seqs_list()
    ctx.rng.shuffle(seqlist)
    first_wins = 0
    calls = 0
    disagreements = 0
    n_pairs = 0
    for i in range(0, len(seqlist) - 1, 2):
        if g.budget is not None and g.calls + 2 > g.budget:
            break
        a, b = seqlist[i], seqlist[i + 1]
        ab = g.compare(a, b)
        ba = g.compare(b, a)
        calls += 2
        first_wins += 1 if ab == "A" else 0
        first_wins += 1 if ba == "A" else 0
        # If order-invariant, ab and ba name the same real winner (ab=="A" <-> ba=="B").
        if (ab == "A") == (ba == "A"):
            disagreements += 1  # same *position* won both times -> inconsistency
        n_pairs += 1
    n_calls = 2 * n_pairs
    p = stats.binomtest(first_wins, n_calls, 0.5).pvalue if n_calls else 1.0
    rate = first_wins / n_calls if n_calls else 0.5
    return {
        "first_position_win_rate": rate,
        "binom_p_value": p,
        "position_inconsistency_rate": disagreements / n_pairs if n_pairs else 0.0,
        "order_bias_detected": bool(p < 0.05 and abs(rate - 0.5) > 0.1),
        "pairs": n_pairs,
        "calls": g.calls,
    }


def probe_transitivity(ctx: ProbeContext, budget: int | None = 90, n: int = 10) -> dict[str, Any]:
    """Compare all pairs among ``n`` sequences and count 3-cycles (A>B>C>A)."""
    g = _Guard(ctx.judge, budget)
    items = ctx.seqs_list()[:n]
    beats: dict[tuple[int, int], bool] = {}
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if not g.left():
                break
            a_wins = g.compare(items[i], items[j]) == "A"
            beats[(i, j)] = a_wins

    def wins(x: int, y: int) -> bool | None:
        if (x, y) in beats:
            return beats[(x, y)]
        if (y, x) in beats:
            return not beats[(y, x)]
        return None

    cycles = 0
    triples = 0
    m = len(items)
    for i in range(m):
        for j in range(i + 1, m):
            for k in range(j + 1, m):
                w = [wins(i, j), wins(j, k), wins(i, k)]
                if any(x is None for x in w):
                    continue
                triples += 1
                ij, jk, ik = w
                # 3-cycle: i>j>k>i or i<j<k<i.
                if ij and jk and not ik or (not ij) and (not jk) and ik:
                    cycles += 1
    return {
        "n_cycles": cycles,
        "triples_checked": triples,
        "cycle_rate": cycles / triples if triples else 0.0,
        "transitive": cycles == 0 and triples > 0,
        "calls": g.calls,
    }


def probe_latency_and_limits(ctx: ProbeContext, budget: int | None = 60) -> dict[str, Any]:
    """Measure latency distribution and detect rate limiting / hard budget."""
    seqlist = ctx.seqs_list()
    a, b = seqlist[0], seqlist[1]
    latencies: list[float] = []
    rate_limited = False
    budget_exhausted = False
    calls_before_error: int | None = None
    n = budget or 60
    for i in range(n):
        t0 = time.perf_counter()
        try:
            ctx.judge.compare(a, b)
        except RateLimitError:
            rate_limited = True
            calls_before_error = i
            break
        except BudgetExhaustedError:
            budget_exhausted = True
            calls_before_error = i
            break
        except OracleError:
            calls_before_error = i
            break
        latencies.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(latencies) if latencies else np.array([0.0])
    return {
        "mean_latency_ms": float(arr.mean()),
        "p95_latency_ms": float(np.percentile(arr, 95)),
        "max_latency_ms": float(arr.max()),
        "rate_limited": rate_limited,
        "budget_exhausted": budget_exhausted,
        "calls_before_error": calls_before_error,
        "calls": len(latencies) + (1 if calls_before_error is not None else 0),
    }


def _rank_small(g: _Guard, items: list[str]) -> list[str]:
    """Insertion-rank a small list strongest-first within the guard's budget."""
    ranked: list[str] = []
    for x in items:
        placed = False
        for idx in range(len(ranked)):
            if not g.left():
                ranked.append(x)
                placed = True
                break
            if g.compare(x, ranked[idx]) == "A":
                ranked.insert(idx, x)
                placed = True
                break
        if not placed:
            ranked.append(x)
    return ranked


def probe_length_bias(ctx: ProbeContext, budget: int | None = 80) -> dict[str, Any]:
    """Does the Judge reward content or merely size?

    Ranks nested 3'-anchored truncations of one sequence and checks monotonicity in
    length, then runs length-matched controls: a strong sequence truncated vs a weak
    one at full length, and a strong sequence padded (random and dinuc-shuffled
    filler) to a longer length.
    """
    g = _Guard(ctx.judge, budget)
    seqlist = ctx.seqs_list()
    long_seq = max(seqlist, key=len)
    lengths = [n for n in (200, 400, 600, 800, 1000, 1200) if n <= len(long_seq)]
    truncs = [long_seq[-n:] for n in lengths]  # proximal (3') regions kept
    ranked = _rank_small(g, list(truncs))
    ranked_lengths = [len(s) for s in ranked]
    monotone = ranked_lengths == sorted(ranked_lengths, reverse=True)

    # Length-matched controls need a strong and a weak sequence.
    prelim = _rank_small(g, seqlist[: min(8, len(seqlist))])
    strong = prelim[0] if prelim else long_seq
    weak = prelim[-1] if prelim else seqlist[-1]

    controls: dict[str, Any] = {}
    if g.left():
        # Strong truncated to weak's length vs weak at full length.
        tlen = min(len(strong), len(weak))
        controls["strong_trunc_beats_weak_full"] = (
            g.compare(strong[-tlen:], weak) == "A"
        )
    if g.left():
        pad = seqs.random_seq(300, gc=seqs.gc_content(strong), rng=ctx.rng)
        controls["strong_beats_self_random_padded"] = (
            g.compare(strong, strong + pad) == "A"
        )
    if g.left():
        shuf = seqs.dinuc_shuffle(strong, ctx.rng)
        controls["strong_beats_self_shuffle_padded"] = (
            g.compare(strong, strong + shuf[:300]) == "A"
        )
    return {
        "trunc_lengths": lengths,
        "ranked_lengths": ranked_lengths,
        "monotone_in_length": monotone,
        "controls": controls,
        "verdict": "size-driven" if monotone else "content-driven",
        "calls": g.calls,
    }


def probe_gc_sweep(ctx: ProbeContext, budget: int | None = 60) -> dict[str, Any]:
    """Rank synthetic sequences at controlled GC to expose a GC preference."""
    g = _Guard(ctx.judge, budget)
    gcs = [0.30, 0.40, 0.50, 0.60, 0.70]
    synth = {gc: seqs.random_seq(800, gc=gc, rng=ctx.rng) for gc in gcs}
    ranked = _rank_small(g, list(synth.values()))
    inv = {v: k for k, v in synth.items()}
    ranked_gc = [inv[s] for s in ranked]
    return {
        "gc_levels": gcs,
        "ranked_gc_best_to_worst": ranked_gc,
        "preferred_gc": ranked_gc[0] if ranked_gc else None,
        "calls": g.calls,
    }


def probe_shuffle(ctx: ProbeContext, budget: int | None = 40) -> dict[str, Any]:
    """Compare real promoters against their dinucleotide-preserving shuffles."""
    g = _Guard(ctx.judge, budget)
    real_wins = 0
    tested = 0
    for r in ctx.records:
        if not g.left():
            break
        shuf = seqs.dinuc_shuffle(r.seq, ctx.rng)
        if g.compare(r.seq, shuf) == "A":
            real_wins += 1
        tested += 1
    return {
        "real_beats_shuffle_rate": real_wins / tested if tested else 0.0,
        "tested": tested,
        "structure_matters": (real_wins / tested > 0.6) if tested else False,
        "calls": g.calls,
    }


def probe_motif_ablation(ctx: ProbeContext, budget: int | None = 30) -> dict[str, Any]:
    """Knock out TATA/CCAAT in real promoters and see if strength drops."""
    g = _Guard(ctx.judge, budget)

    def knock_out(seq: str) -> str:
        s = seqs._TATA_RE.sub(lambda m: "G" * len(m.group()), seq)
        s = seqs._CCAAT_RE.sub("GGGGG", s)
        return s

    intact_wins = 0
    tested = 0
    for r in ctx.records:
        if not g.left():
            break
        ablated = knock_out(r.seq)
        if ablated == r.seq:
            continue
        if g.compare(r.seq, ablated) == "A":
            intact_wins += 1
        tested += 1
    return {
        "intact_beats_ablated_rate": intact_wins / tested if tested else 0.0,
        "tested": tested,
        "motifs_matter": (intact_wins / tested > 0.6) if tested else False,
        "calls": g.calls,
    }


def probe_league_table(ctx: ProbeContext, budget: int | None = 200) -> dict[str, Any]:
    """Rank all corpus sequences against each other (budget-limited merge sort)."""
    from .ranking import merge_sort_rank

    seqlist = ctx.seqs_list()
    by_seq = {r.seq: r.id for r in ctx.records}
    ranked, calls = merge_sort_rank(ctx.judge, seqlist, budget)
    return {
        "ranking_ids": [by_seq.get(s, "?") for s in ranked],
        "n": len(ranked),
        "calls": calls,
    }


# --- Navigator probes -------------------------------------------------------


def probe_roundtrip(ctx: ProbeContext, budget: int | None = None) -> dict[str, Any]:
    """decode(encode(x)) fidelity: identity rate and edit-distance distribution."""
    nav = ctx.navigator
    dists: list[int] = []
    exact = 0
    for r in ctx.records:
        recon = nav.decode(nav.encode(r.seq))
        d = seqs.edit_distance(r.seq, recon, cap=len(r.seq))
        dists.append(d)
        if d == 0:
            exact += 1
    arr = np.array(dists)
    return {
        "exact_identity_rate": exact / len(dists) if dists else 0.0,
        "mean_edit_distance": float(arr.mean()),
        "median_edit_distance": float(np.median(arr)),
        "n": len(dists),
    }


def probe_geometry(ctx: ProbeContext, budget: int | None = None) -> dict[str, Any]:
    """Inspect latent norms, boundary behaviour, and whether a distance exists."""
    nav = ctx.navigator
    sample = np.stack([nav.encode(r.seq) for r in ctx.records])
    norms = np.linalg.norm(sample, axis=1)
    geom = detect_geometry(sample)
    has_distance = hasattr(nav, "distance")
    dim = getattr(nav, "dim", sample.shape[1])
    return {
        "detected_geometry": geom,
        "min_norm": float(norms.min()),
        "max_norm": float(norms.max()),
        "mean_norm": float(norms.mean()),
        "all_inside_unit_ball": bool(np.all(norms < 1.0 + 1e-6)),
        "has_native_distance": has_distance,
        "dim": int(dim),
    }


def probe_step_calibration(ctx: ProbeContext, budget: int | None = None) -> dict[str, Any]:
    """Edit distance vs perturbation magnitude (log-spaced); saves a plot.

    Reports the perturbation scale at which sequences change but remain plausible --
    our working step size for local search.
    """
    nav = ctx.navigator
    geom = detect_geometry(np.stack([nav.encode(r.seq) for r in ctx.records[:20]]))
    ops = LatentOps(geom)
    rng = np.random.default_rng(0)
    scales = np.logspace(-2, 0.5, 12)
    base = ctx.records[0].seq
    z0 = nav.encode(base)
    mean_dists: list[float] = []
    plausible_frac: list[float] = []
    for s in scales:
        ds = []
        plaus = 0
        for _ in range(5):
            z = ops.perturb(z0, float(s), rng)
            seq = nav.decode(z)
            ds.append(seqs.edit_distance(base, seq, cap=len(base)))
            plaus += 1 if seqs.plausibility(seq)[0] else 0
        mean_dists.append(float(np.mean(ds)))
        plausible_frac.append(plaus / 5)
    # Recommend the largest scale that still keeps >=60% plausible and changes seq.
    recommended = None
    for s, d, pf in zip(scales, mean_dists, plausible_frac, strict=True):
        if d >= 1 and pf >= 0.6:
            recommended = float(s)
    fig_path = ctx.out_dir / "figures" / "step_calibration.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots()
    ax1.plot(scales, mean_dists, "o-", label="edit distance")
    ax1.set_xscale("log")
    ax1.set_xlabel("perturbation scale")
    ax1.set_ylabel("mean edit distance")
    ax2 = ax1.twinx()
    ax2.plot(scales, plausible_frac, "s--", color="tab:red", label="plausible frac")
    ax2.set_ylabel("plausible fraction")
    fig.suptitle("Step-size calibration")
    fig.savefig(fig_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {
        "scales": scales.tolist(),
        "mean_edit_distance": mean_dists,
        "plausible_fraction": plausible_frac,
        "recommended_scale": recommended,
        "plot": str(fig_path),
    }


def probe_interpolation(ctx: ProbeContext, budget: int | None = None) -> dict[str, Any]:
    """Decode points along a geodesic between two promoters; check plausibility."""
    nav = ctx.navigator
    geom = detect_geometry(np.stack([nav.encode(r.seq) for r in ctx.records[:20]]))
    ops = LatentOps(geom)
    z1 = nav.encode(ctx.records[0].seq)
    z2 = nav.encode(ctx.records[1].seq)
    ts = np.linspace(0, 1, 7)
    plaus = []
    lengths = []
    for t in ts:
        z = ops.interpolate(z1, z2, float(t))
        seq = nav.decode(z)
        plaus.append(seqs.plausibility(seq)[0])
        lengths.append(len(seq))
    return {
        "t_values": ts.tolist(),
        "plausible": [bool(p) for p in plaus],
        "lengths": lengths,
        "all_intermediates_plausible": bool(all(plaus[1:-1])),
    }


def probe_radius_sweep(ctx: ProbeContext, budget: int | None = None) -> dict[str, Any]:
    """Move outward along a ray from the origin; does sequence character shift?

    This tests our hypothesis about what a hyperbolic latent buys us: outward moves
    should systematically change sequence properties (e.g. GC, length).
    """
    nav = ctx.navigator
    geom = detect_geometry(np.stack([nav.encode(r.seq) for r in ctx.records[:20]]))
    ops = LatentOps(geom)
    z0 = nav.encode(ctx.records[0].seq)
    direction = z0 / (np.linalg.norm(z0) + 1e-9)
    radii = np.linspace(0.1, 0.9, 9) if geom == "poincare" else np.linspace(0.5, 5.0, 9)
    gcs = []
    lengths = []
    origin = np.zeros_like(z0)
    for r in radii:
        z = ops.step(origin, direction, float(r))
        seq = nav.decode(z)
        gcs.append(seqs.gc_content(seq))
        lengths.append(len(seq))
    gc_trend = float(np.corrcoef(radii, gcs)[0, 1]) if len(set(gcs)) > 1 else 0.0
    return {
        "geometry": geom,
        "radii": radii.tolist(),
        "gc_by_radius": gcs,
        "length_by_radius": lengths,
        "gc_radius_correlation": gc_trend,
        "systematic_change": abs(gc_trend) > 0.5,
    }


# --- Orchestration ----------------------------------------------------------

JUDGE_PROBES: dict[str, Callable[..., dict[str, Any]]] = {
    "determinism": probe_determinism,
    "order_bias": probe_order_bias,
    "transitivity": probe_transitivity,
    "latency": probe_latency_and_limits,
    "length_bias": probe_length_bias,
    "gc_sweep": probe_gc_sweep,
    "shuffle": probe_shuffle,
    "motif_ablation": probe_motif_ablation,
    "league_table": probe_league_table,
}

NAV_PROBES: dict[str, Callable[..., dict[str, Any]]] = {
    "roundtrip": probe_roundtrip,
    "geometry": probe_geometry,
    "step_calibration": probe_step_calibration,
    "interpolation": probe_interpolation,
    "radius_sweep": probe_radius_sweep,
}

ALL_PROBES = {**JUDGE_PROBES, **NAV_PROBES}


def make_context(backend_name: str | None, out_dir: Path) -> ProbeContext:
    """Build a probe context against the selected backend."""
    judge = backend.get_judge(backend_name)
    nav = backend.get_navigator(backend_name)
    records = seqs.read_fasta()
    out_dir.mkdir(parents=True, exist_ok=True)
    return ProbeContext(judge=judge, navigator=nav, records=records, out_dir=out_dir)


def run_probes(
    ctx: ProbeContext, names: list[str], total_budget: int | None
) -> dict[str, Any]:
    """Run the named probes, sharing ``total_budget`` across the judge probes."""
    judge_names = [n for n in names if n in JUDGE_PROBES]
    per = (total_budget // max(1, len(judge_names))) if (total_budget and judge_names) else None
    results: dict[str, Any] = {}
    for name in names:
        fn = ALL_PROBES[name]
        try:
            if name in JUDGE_PROBES:
                results[name] = fn(ctx, budget=per)
            else:
                results[name] = fn(ctx)
        except OracleError as exc:  # rate limit / budget hit mid-probe
            results[name] = {"error": type(exc).__name__, "message": str(exc)}
    return results


def render_table(results: dict[str, Any], console: Console) -> None:
    """Print a compact rich summary of probe headline metrics."""
    table = Table(title="Probe report")
    table.add_column("probe")
    table.add_column("headline", overflow="fold")
    headline_keys = [
        "deterministic", "order_bias_detected", "n_cycles", "transitive",
        "rate_limited", "budget_exhausted", "mean_latency_ms", "verdict",
        "preferred_gc", "structure_matters", "motifs_matter", "detected_geometry",
        "exact_identity_rate", "recommended_scale", "systematic_change", "n",
        "monotone_in_length", "all_intermediates_plausible", "cycle_rate",
        "first_position_win_rate",
    ]
    for name, res in results.items():
        if "error" in res:
            table.add_row(name, f"[red]{res['error']}[/]")
            continue
        bits = [f"{k}={res[k]}" for k in headline_keys if k in res]
        table.add_row(name, ", ".join(bits) if bits else json.dumps(res)[:80])
    console.print(table)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Oracle interrogation suite.")
    p.add_argument("--backend", choices=("mock", "real"), default=None)
    p.add_argument("--budget", type=int, default=400, help="Total judge-call budget.")
    p.add_argument(
        "--probe",
        default="all",
        help=f"Probe name or 'all'. Choices: {', '.join(ALL_PROBES)}",
    )
    p.add_argument("--out", default="probe_out", help="Output directory.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.backend:
        import os

        os.environ["PROMO_BACKEND"] = args.backend
    out_dir = Path(args.out)
    ctx = make_context(args.backend, out_dir)
    names = list(ALL_PROBES) if args.probe == "all" else [args.probe]
    unknown = [n for n in names if n not in ALL_PROBES]
    if unknown:
        raise SystemExit(f"unknown probe(s): {unknown}. Choices: {list(ALL_PROBES)}")
    results = run_probes(ctx, names, args.budget)
    report_path = out_dir / "probe_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    console = Console()
    render_table(results, console)
    console.print(f"[green]Wrote[/] {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
