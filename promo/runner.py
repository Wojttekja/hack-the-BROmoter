"""Main optimization loop: budget, wall-clock, checkpointing, resume.

Wires the swap-point backend, the cache-enforcing Judge, a navigator, and a chosen
optimizer into a single loop that runs until the call budget or wall-clock limit is
hit. Full optimizer state is checkpointed to JSON every ``--checkpoint-interval``
seconds and on SIGINT, so a process killed at hour 9 resumes with ``--resume`` and
zero data loss (the comparison cache is already crash-safe on its own).

Run it as::

    python -m promo.runner --optimizer koth --backend mock --budget 500
    python -m promo.runner --optimizer koth --backend mock --budget 500 --resume
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.live import Live
from rich.table import Table

from . import backend, seqs
from .cache import CachedJudge, ComparisonCache
from .interfaces import Candidate, Judge, Navigator
from .latent import LatentOps, detect_geometry
from .optimizers import REGISTRY, Optimizer
from .partial_order import TransitiveJudge


@dataclass(slots=True)
class RunnerConfig:
    """Resolved runner configuration (also mirrored into checkpoints)."""

    optimizer: str
    backend: str
    budget: int
    wall: float
    batch_size: int
    geometry: str
    n_seeds: int
    checkpoint_interval: float
    cache_path: str
    checkpoint_path: str
    log_path: str
    assume_symmetric: bool
    transitive: bool
    seed: int


def build_seeds(nav: Navigator, n_seeds: int, fasta: str | None = None) -> list[Candidate]:
    """Build seed candidates from the corpus, native pks1 promoter first.

    The native pks1 (polyketide synthase) promoter is placed at index 0 so KotH
    starts by defending the real baseline we must beat.
    """
    records = seqs.read_fasta(fasta)
    native = [r for r in records if seqs.NATIVE_PKS1_LOCUS in r.id]
    others = [r for r in records if seqs.NATIVE_PKS1_LOCUS not in r.id]
    ordered = native + others
    chosen = ordered[: max(1, n_seeds)]
    return [Candidate(id=r.id, seq=r.seq, latent=nav.encode(r.seq)) for r in chosen]


def resolve_geometry(nav: Navigator, seeds: list[Candidate], requested: str) -> str:
    """Return the latent geometry to use, auto-detecting from seed latents if asked."""
    if requested != "auto":
        return requested
    sample = np.stack([s.latent for s in seeds if s.latent is not None])
    return detect_geometry(sample)


def _wrap_judge(cfg: RunnerConfig) -> tuple[Judge, ComparisonCache, CachedJudge]:
    """Construct the cache-enforcing (and optionally transitive) Judge stack."""
    raw = backend.get_judge(cfg.backend)
    cache = ComparisonCache(cfg.cache_path, assume_symmetric=cfg.assume_symmetric)
    cached = CachedJudge(raw, cache)
    judge: Judge = cached
    if cfg.transitive:
        judge = TransitiveJudge(cached, enabled=True)
    return judge, cache, cached


def _make_navigator(cfg: RunnerConfig) -> Navigator:
    """Build the navigator, passing geometry only to the mock backend."""
    if cfg.backend == "mock":
        geom = cfg.geometry if cfg.geometry != "auto" else "euclidean"
        return backend.get_navigator(cfg.backend, geometry=geom)
    return backend.get_navigator(cfg.backend)


class _EventLog:
    """Append-only JSONL structured event log, flushed per write."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)

    def emit(self, event: str, **fields: Any) -> None:
        """Write one event record."""
        rec = {"t": time.time(), "event": event, **fields}
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Close the log."""
        if not self._fh.closed:
            self._fh.close()


def _progress_table(cfg: RunnerConfig, state: dict[str, Any]) -> Table:
    """Render the live progress table."""
    table = Table(title=f"promo.runner [{cfg.optimizer} / {cfg.backend}]")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in ("generation", "calls", "budget", "cache_hit_rate", "elapsed_s",
                "champion_len", "champion_gc", "saved_by_closure"):
        table.add_row(key, str(state.get(key, "-")))
    return table


def run(cfg: RunnerConfig, *, resume: bool = False, console: Console | None = None) -> Candidate:
    """Run the optimization loop and return the best candidate found."""
    console = console or Console()
    judge, cache, cached = _wrap_judge(cfg)
    nav = _make_navigator(cfg)
    seeds = build_seeds(nav, cfg.n_seeds)
    geometry = resolve_geometry(nav, seeds, cfg.geometry)
    ops = LatentOps(geometry)

    opt_cls: type[Optimizer] = REGISTRY[cfg.optimizer]
    opt = opt_cls(nav, seeds, latent_ops=ops, batch_size=cfg.batch_size, seed=cfg.seed)

    log = _EventLog(cfg.log_path)
    ckpt_path = Path(cfg.checkpoint_path)
    calls_offset = 0
    start = time.monotonic()

    if resume and ckpt_path.exists():
        state = json.loads(ckpt_path.read_text())
        opt.load_state_dict(state["optimizer"])
        calls_offset = state.get("calls_spent", 0)
        log.emit("resume", calls_offset=calls_offset, generation=opt.generation)
        console.print(f"[green]Resumed[/] at generation {opt.generation}, "
                      f"{calls_offset} prior calls.")

    stop = {"flag": False}

    def _on_sigint(signum: int, frame: Any) -> None:
        stop["flag"] = True

    old_handler = signal.signal(signal.SIGINT, _on_sigint)

    def total_calls() -> int:
        return calls_offset + cached.real_calls

    def checkpoint() -> None:
        state = {
            "optimizer": opt.state_dict(),
            "calls_spent": total_calls(),
            "config": asdict(cfg),
            "geometry": geometry,
        }
        tmp = ckpt_path.with_suffix(ckpt_path.suffix + ".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(state))
        tmp.replace(ckpt_path)  # atomic swap
        log.emit("checkpoint", calls=total_calls(), generation=opt.generation)

    last_ckpt = time.monotonic()
    log.emit("start", config=asdict(cfg), geometry=geometry, seeds=len(seeds))

    try:
        with Live(console=console, refresh_per_second=4) as live:
            while total_calls() < cfg.budget and (time.monotonic() - start) < cfg.wall:
                if stop["flag"]:
                    break
                remaining = cfg.budget - total_calls()
                cands = opt.ask()
                ranked, calls = opt.evaluate(judge, cands, remaining)
                opt.tell(ranked)
                log.emit("generation", generation=opt.generation, calls=total_calls(),
                         batch=len(cands), best=(opt.best.id if opt.best else None))

                now = time.monotonic()
                if now - last_ckpt >= cfg.checkpoint_interval:
                    checkpoint()
                    last_ckpt = now

                live.update(_progress_table(cfg, _snapshot(cfg, opt, cache, cached, judge, start)))
    finally:
        checkpoint()
        signal.signal(signal.SIGINT, old_handler)
        log.emit("stop", calls=total_calls(), generation=opt.generation,
                 best=(opt.best.id if opt.best else None))
        log.close()
        cache.close()

    assert opt.best is not None
    console.print(f"[bold green]Done.[/] Best={opt.best.id} "
                  f"len={len(opt.best.seq)} gc={seqs.gc_content(opt.best.seq):.3f} "
                  f"calls={total_calls()}")
    return opt.best


def _snapshot(
    cfg: RunnerConfig,
    opt: Optimizer,
    cache: ComparisonCache,
    cached: CachedJudge,
    judge: Judge,
    start: float,
) -> dict[str, Any]:
    """Assemble the live-progress metrics."""
    champ = opt.best
    saved = getattr(judge, "saved_calls", 0)
    return {
        "generation": opt.generation,
        "calls": cached.real_calls,
        "budget": cfg.budget,
        "cache_hit_rate": f"{cache.stats()['hit_rate']:.2f}",
        "elapsed_s": f"{time.monotonic() - start:.1f}",
        "champion_len": len(champ.seq) if champ else "-",
        "champion_gc": f"{seqs.gc_content(champ.seq):.3f}" if champ else "-",
        "saved_by_closure": saved,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Promoter optimization runner.")
    p.add_argument("--optimizer", choices=sorted(REGISTRY), default="koth")
    p.add_argument("--backend", choices=("mock", "real"), default=None,
                   help="Overrides PROMO_BACKEND.")
    p.add_argument("--budget", type=int, default=500, help="Max real oracle calls.")
    p.add_argument("--wall", type=float, default=float("inf"), help="Wall-clock limit (s).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--geometry", choices=("auto", "euclidean", "poincare"), default="auto")
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--checkpoint-interval", type=float, default=60.0)
    p.add_argument("--out", default="runs", help="Directory for cache/checkpoint/log.")
    p.add_argument("--cache", default=None)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--log", default=None)
    p.add_argument("--assume-symmetric", action="store_true")
    p.add_argument("--transitive", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> RunnerConfig:
    """Build a :class:`RunnerConfig` from parsed args, defaulting paths under --out."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = args.optimizer
    return RunnerConfig(
        optimizer=args.optimizer,
        backend=(args.backend or "mock"),
        budget=args.budget,
        wall=args.wall,
        batch_size=args.batch_size,
        geometry=args.geometry,
        n_seeds=args.n_seeds,
        checkpoint_interval=args.checkpoint_interval,
        cache_path=args.cache or str(out / f"{tag}_cache.jsonl"),
        checkpoint_path=args.checkpoint or str(out / f"{tag}_checkpoint.json"),
        log_path=args.log or str(out / f"{tag}_events.jsonl"),
        assume_symmetric=args.assume_symmetric,
        transitive=args.transitive,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv)
    if args.backend:
        import os

        os.environ["PROMO_BACKEND"] = args.backend
    cfg = config_from_args(args)
    run(cfg, resume=args.resume)
    return 0


if __name__ == "__main__":
    sys.exit(main())
