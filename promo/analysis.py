"""Analysis and plotting: did we actually improve, and why?

These functions lean on the mock's hidden ``_ground_truth`` (available only in the
mock) to measure optimizer quality during rehearsal:

* regret / improvement curve over oracle calls,
* win rate versus the native pks1 promoter,
* k-mer enrichment in winners vs losers with a significance test,
* MAP-Elites archive heatmap,
* latent-space PCA of explored points coloured by rank.

PNGs are written to ``figures/``. Run as ``python -m promo.analysis --cache ...``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

from . import seqs  # noqa: E402
from .cache import ComparisonCache  # noqa: E402

GroundTruth = Callable[[str], float]


def _figures_dir(out: str | Path) -> Path:
    """Return (and create) the figures directory."""
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    return d


def evaluated_order(cache_path: str | Path) -> list[str]:
    """Return sequences in first-seen (call) order from the comparison log."""
    cache = ComparisonCache(cache_path)
    seen: dict[str, None] = {}
    for c in sorted(cache._store.values(), key=lambda x: x.timestamp):
        seen.setdefault(c.seq_a, None)
        seen.setdefault(c.seq_b, None)
    cache.close()
    return list(seen)


def regret_curve(
    sequences: list[str], gt: GroundTruth, out: str | Path = "figures"
) -> dict[str, Any]:
    """Plot running-best ground truth vs. number of evaluated sequences.

    Regret is ``best_seen_overall - running_best``; it should fall to ~0.
    """
    scores = [gt(s) for s in sequences]
    if not scores:
        return {"regret": None}
    running = np.maximum.accumulate(scores)
    best = float(np.max(scores))
    regret = best - running
    fig, ax = plt.subplots()
    ax.plot(running, label="running best")
    ax.axhline(best, ls="--", color="grey", label="best seen")
    ax.set_xlabel("sequences evaluated")
    ax.set_ylabel("ground-truth strength")
    ax.set_title("Improvement / regret curve")
    ax.legend()
    path = _figures_dir(out) / "regret_curve.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"final_best": best, "final_regret": float(regret[-1]), "plot": str(path)}


def win_rate_vs_native(
    sequences: list[str], native_seq: str, gt: GroundTruth
) -> dict[str, Any]:
    """Fraction of explored sequences whose ground truth beats the native promoter."""
    native_score = gt(native_seq)
    scores = np.array([gt(s) for s in sequences])
    if scores.size == 0:
        return {"win_rate": 0.0, "native_score": native_score}
    return {
        "native_score": float(native_score),
        "win_rate": float(np.mean(scores > native_score)),
        "best_margin": float(scores.max() - native_score),
        "n": int(scores.size),
    }


def kmer_enrichment(
    winners: list[str], losers: list[str], k: int = 4, top: int = 15
) -> pd.DataFrame:
    """Return k-mers most enriched in winners vs losers with a t-test p-value."""
    if not winners or not losers:
        return pd.DataFrame(columns=["kmer", "delta", "p_value"])
    win_mat = np.stack([seqs.kmer_vector(s, k) for s in winners])
    lose_mat = np.stack([seqs.kmer_vector(s, k) for s in losers])
    from itertools import product

    kmers = ["".join(p) for p in product(seqs.ALPHABET, repeat=k)]
    delta = win_mat.mean(axis=0) - lose_mat.mean(axis=0)
    # Welch t-test per k-mer.
    _, pvals = stats.ttest_ind(win_mat, lose_mat, axis=0, equal_var=False)
    df = pd.DataFrame({"kmer": kmers, "delta": delta, "p_value": pvals})
    df = df.reindex(df["delta"].abs().sort_values(ascending=False).index)
    return df.head(top).reset_index(drop=True)


def winners_losers_from_cache(cache_path: str | Path) -> tuple[list[str], list[str]]:
    """Collect winning and losing sequences from every recorded comparison."""
    cache = ComparisonCache(cache_path)
    winners: list[str] = []
    losers: list[str] = []
    for c in cache._store.values():
        if c.winner == "A":
            winners.append(c.seq_a)
            losers.append(c.seq_b)
        else:
            winners.append(c.seq_b)
            losers.append(c.seq_a)
    cache.close()
    return winners, losers


def map_elites_heatmap(
    df: pd.DataFrame, gt: GroundTruth | None = None, out: str | Path = "figures"
) -> dict[str, Any]:
    """Render a MAP-Elites archive heatmap over its first two axes.

    Cells are coloured by ground-truth strength when ``gt`` is supplied, else by
    occupancy.
    """
    axis_cols = [c for c in df.columns if c.endswith("_bin")]
    if len(axis_cols) < 2 or df.empty:
        return {"plot": None, "reason": "need a 2-axis non-empty archive"}
    ax0, ax1 = axis_cols[0], axis_cols[1]
    df = df.assign(value=[gt(s) for s in df["seq"]] if gt is not None else 1.0)
    grid = df.pivot_table(index=ax1, columns=ax0, values="value", aggfunc="max")
    fig, ax = plt.subplots()
    im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel(ax0)
    ax.set_ylabel(ax1)
    ax.set_title("MAP-Elites archive")
    fig.colorbar(im, ax=ax, label="strength" if gt else "occupied")
    path = _figures_dir(out) / "map_elites_heatmap.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"plot": str(path), "occupied_cells": int(df.shape[0])}


def latent_pca(
    latents: np.ndarray, ranks: np.ndarray, out: str | Path = "figures"
) -> dict[str, Any]:
    """2-D PCA scatter of explored latent points coloured by rank."""
    if latents.shape[0] < 3:
        return {"plot": None, "reason": "need >=3 points"}
    coords = PCA(n_components=2).fit_transform(latents)
    fig, ax = plt.subplots()
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=ranks, cmap="plasma", s=18)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Explored latent space (coloured by rank)")
    fig.colorbar(sc, ax=ax, label="rank (0=best)")
    path = _figures_dir(out) / "latent_pca.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return {"plot": str(path), "n": int(latents.shape[0])}


def _mock_ground_truth() -> GroundTruth:
    """Return the mock's hidden ground truth (analysis-only; mock backend)."""
    from .mock_backend import MockJudge, MockNavigator

    nav = MockNavigator()
    judge = MockJudge(navigator=nav)
    return judge._ground_truth


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description="Analysis & plots for a run.")
    p.add_argument("--cache", required=True, help="Path to a comparison JSONL cache.")
    p.add_argument("--backend", choices=("mock", "real"), default="mock")
    p.add_argument("--out", default="figures")
    p.add_argument("--kmer", type=int, default=4)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: generate all available plots from a cache file."""
    args = _parse_args(argv)
    out = Path(args.out)
    report: dict[str, Any] = {}

    if args.backend == "mock":
        gt = _mock_ground_truth()
        order = evaluated_order(args.cache)
        report["regret"] = regret_curve(order, gt, out)
        records = seqs.read_fasta()
        native = next(
            (r.seq for r in records if seqs.NATIVE_PKS1_LOCUS in r.id), records[0].seq
        )
        report["vs_native"] = win_rate_vs_native(order, native, gt)
    else:
        report["note"] = "ground-truth plots require the mock backend"

    winners, losers = winners_losers_from_cache(args.cache)
    enr = kmer_enrichment(winners, losers, k=args.kmer)
    enr_path = out / "kmer_enrichment.csv"
    out.mkdir(parents=True, exist_ok=True)
    enr.to_csv(enr_path, index=False)
    report["kmer_enrichment_csv"] = str(enr_path)

    report_path = out / "analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
