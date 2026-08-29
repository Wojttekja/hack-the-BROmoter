"""Offspring generation: one ranked population in, the next generation out.

Lifted out of `navigator.ipynb`, which established the shape of the recipe:

* **spread the parents out** -- taking the literal top 20% of a ranking makes
  every parent a near-clone of the same line, so `select_top` skips any
  candidate within `min_distance` substitutions of one already chosen.
* **spend the budget on the best** -- `allocate_children` weights the split by
  ``exp(-rank / tau)``, so the head of the ranking gets most of the children
  and the tail still gets a few (the ranking is noisy; the tail is not dead).
* **two sources of change per child** -- `/nawigator/edycje` moves latent
  codes, which gives big coherent jumps but only ``opcji`` distinct options
  per call, far too few for a thousand children. So every navigator variant is
  additionally point-mutated, with the positions drawn from the gradient
  weights of `/nawigator/mapa` (`wagaP`) so the mutations land where the model
  is actually sensitive, and `zmien_na` taken as a hint most of the time.
* **elitism** -- `keep_elite` parents are copied through unchanged, so a
  generation can never score worse than the one before it.

Everything talks to the server through `hack_the_bromoter.api`, so the calls
inherit the key pool: requests go round-robin over every ``HYPPE_API_KEY*``,
a key that answers 429 is parked for its minute while the others keep working,
and 503 (GPU queue full) is retried. That is also why `breed` can run parents
in parallel -- the pool is thread safe and the two navigator endpoints allow
600 calls per minute *per key*.

    from hack_the_bromoter.navigator import evolve
    from hack_the_bromoter.utils import read_dataframe

    ranking = read_dataframe("sequences.csv")   # best first
    children = evolve(ranking, k=1000)
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import pandas as pd

from hack_the_bromoter.api import (
    MAX_SCORED_SEQUENCES,
    ApiError,
    build_fasta,
    check_sequence,
    nawigator_edycje,
    nawigator_mapa,
)
from hack_the_bromoter.utils import ID_COL, SEQ_COL

__all__ = [
    "allocate_children",
    "breed",
    "dedupe_by_distance",
    "diversify_survivors",
    "evolve",
    "get_edits",
    "get_map",
    "hamming",
    "mutate",
    "select_top",
    "to_fasta",
]

# Point mutations only ever write a real base; N is legal for the server but
# never worth introducing on purpose.
BASES = "ACGT"

# Added to every `wagaP` so no position is completely unmutatable (and so
# random.choices never sees an all-zero weight vector).
WEIGHT_FLOOR = 0.05

# How often a `zmien_na` recommendation is taken instead of a random base.
HINT_PROBABILITY = 0.6

# /nawigator/edycje levels, rotated per call: 2 = L3 (2 bp slots, small
# edits), 1 = L2, 0 = L1 (16 bp slots, big jumps). Rotating gives the children
# of one parent a mix of edit sizes.
EDIT_LEVELS = (2, 1, 0)
DEFAULT_OPTIONS = 8

# Parents bred concurrently. Each one is a handful of sequential API calls, so
# this is what turns a 20-parent generation from ~20 round-trip chains into 5.
WORKERS = 4


# print() writes the text and the newline separately, so parents running in
# parallel would otherwise shuffle their progress lines into each other.
_PRINT_LOCK = Lock()


def _log(verbose: bool, message: str, indent: int = 0) -> None:
    """One progress line, flushed so a long run is followable live."""
    if verbose:
        with _PRINT_LOCK:
            print(f"{'  ' * indent}{message}", flush=True)


# --------------------------------------------------------------------------
# step 1: pick the parents
# --------------------------------------------------------------------------
def hamming(a: str, b: str) -> int:
    """How many positions two sequences differ in.

    Compares up to the shorter of the two, so a truncated sequence reads as
    similar rather than blowing up.
    """
    return sum(1 for x, y in zip(a, b) if x != y)


def dedupe_by_distance(
    ids: list[str],
    sequences: dict[str, str],
    min_distance: int,
    limit: int | None = None,
) -> list[str]:
    """`ids`, best-first, minus anything within `min_distance` of an earlier one.

    Walks the ranking greedily: a candidate is kept only if its sequence
    differs by at least `min_distance` substitutions from every
    representative already kept. That collapses each cluster of
    near-identical sequences down to its single best-ranked member, which is
    the general move behind both `select_top` (diverse parents) and
    `diversify_survivors` (a diverse next generation).

    `limit` stops once that many ids have been kept, leaving the rest of
    `ids` unexamined -- pass it when you only need the first N diverse picks
    (cheaper than filtering the whole ranking and slicing after).
    """
    kept: list[str] = []
    reps: list[str] = []
    for candidate in ids:
        if limit is not None and len(kept) >= limit:
            break
        sequence = sequences[candidate]
        if all(hamming(sequence, rep) >= min_distance for rep in reps):
            kept.append(candidate)
            reps.append(sequence)
    return kept


def select_top(
    df: pd.DataFrame,
    fraction: float = 0.2,
    min_distance: int = 8,
    id_col: str = ID_COL,
    seq_col: str = SEQ_COL,
    verbose: bool = True,
) -> pd.DataFrame:
    """The best `fraction` of a ranking, minus the near-duplicates.

    `df` must already be sorted best-first. A candidate is skipped when it is
    within `min_distance` substitutions of a parent already picked, which is
    what stops the whole generation descending from one lineage.

    Returns a fresh DataFrame with ``id``/``sequence`` columns (uppercased),
    always at least one row wide when `df` is non-empty.
    """
    wanted = max(1, int(len(df) * fraction))
    ids = [str(i) for i in df[id_col]]
    sequences = {str(i): str(s).strip().upper() for i, s in zip(df[id_col], df[seq_col])}
    kept = dedupe_by_distance(ids, sequences, min_distance, limit=wanted)

    _log(verbose, f"parents: {len(kept)} of {len(df)} "
                  f"(top {fraction:.0%}, min distance {min_distance})")
    return pd.DataFrame(
        [{ID_COL: i, SEQ_COL: sequences[i]} for i in kept],
        columns=[ID_COL, SEQ_COL],
    )


def diversify_survivors(
    ranked_ids: list[str],
    sequences: dict[str, str],
    keep: int,
    min_distance: int,
    verbose: bool = True,
) -> list[str]:
    """`ranked_ids`, best-first, collapsed to one pick per similarity cluster.

    Same move as `select_top`, applied where it was missing: choosing next
    generation's survivors. `bucket_sort_sequences` ranks best-first but has
    no notion of similarity, so slicing its output to `ranked_ids[:keep]`
    (as `optimize()` used to) can fill most of `keep` with near-clones of the
    same elite, starving every other lineage. This walks the ranking and
    keeps an id only if it is at least `min_distance` substitutions from
    every survivor already kept.

    If the distance filter leaves fewer than `keep` (a small pool, or one
    that is already diverse enough that everything clears the bar), the
    remaining slots are backfilled with the next best-ranked ids regardless
    of distance -- diversity is a tiebreaker among comparable candidates, not
    a reason to shrink the population below `keep`.
    """
    kept = dedupe_by_distance(ranked_ids, sequences, min_distance, limit=keep)
    collapsed = 0
    if len(kept) < keep:
        picked = set(kept)
        for candidate in ranked_ids:
            if len(kept) >= keep:
                break
            if candidate not in picked:
                kept.append(candidate)
                picked.add(candidate)
                collapsed += 1

    _log(verbose, f"survivors: {len(kept)} kept, {collapsed} backfilled "
                  f"after collapsing near-duplicates (min distance {min_distance})")
    return kept


# --------------------------------------------------------------------------
# step 2: how many children each parent gets
# --------------------------------------------------------------------------
def allocate_children(n_parents: int, k: int, tau: float | None = None) -> list[int]:
    """Split `k` children over `n_parents` with weights ``exp(-rank / tau)``.

    The best parent gets the most, the sum is always exactly `k` (the rounding
    remainder is handed back to the top of the ranking). `tau` defaults to a
    third of the parent count, i.e. the weight decays by ``e`` every third
    place.
    """
    if n_parents <= 0:
        raise ValueError("allocate_children() needs at least one parent")
    k = max(0, int(k))
    if k == 0:
        return [0] * n_parents

    if tau is None:
        tau = n_parents / 3
    tau = max(float(tau), 1e-9)

    weights = [math.exp(-rank / tau) for rank in range(n_parents)]
    total = sum(weights)
    quota = [int(weight / total * k) for weight in weights]

    rank = 0
    while sum(quota) < k:
        quota[rank % n_parents] += 1
        rank += 1
    return quota


# --------------------------------------------------------------------------
# step 3: what the Navigator has to say about one parent
# --------------------------------------------------------------------------
def get_map(seq: str, verbose: bool = True) -> tuple[list[float], dict[int, str]]:
    """`/nawigator/mapa` for one sequence, as ``(weights, hints)``.

    `weights` is a per-position mutation weight (the model's gradient
    magnitude, floored so every position stays reachable) and `hints` maps a
    0-based position to the base the server recommends there. A failed call
    degrades to uniform weights and no hints rather than killing the
    generation -- one parent mutating blindly costs far less than a crash
    halfway through a breeding run.
    """
    try:
        answer = nawigator_mapa(seq, od=0, ile=len(seq))
    except ApiError as error:
        _log(verbose, f"map unavailable ({error}) -- mutating uniformly", 2)
        return [1.0] * len(seq), {}

    weights = [1.0] * len(seq)
    hints: dict[int, str] = {}
    for entry in answer["pozycje"]:
        index = entry["poz"] - 1          # the server counts positions from 1
        if not 0 <= index < len(seq):
            continue
        weights[index] = max(0.0, float(entry["wagaP"])) + WEIGHT_FLOOR
        if entry["zmien_na"] in BASES:    # "." means "leave this one alone"
            hints[index] = entry["zmien_na"]
    return weights, hints


def get_edits(
    seq: str,
    needed: int,
    options: int = DEFAULT_OPTIONS,
    rng: random.Random | None = None,
    verbose: bool = True,
) -> list[str]:
    """Distinct `/nawigator/edycje` variants of one sequence.

    One call returns at most `options` of them, so this loops, rotating
    `EDIT_LEVELS` to mix edit sizes and drawing a fresh ``ziarno`` each time.
    A few extra rounds are budgeted because duplicates are dropped.

    Variants that are not the parent's length, or that the server would filter
    out of a submission anyway, are discarded: `mutate` indexes them with the
    parent's position map, which only lines up at equal length.
    """
    if needed <= 0:
        return []
    rng = rng or random.Random()

    variants: list[str] = []
    seen = {seq}
    for number in range(needed // max(1, options) + 5):
        if len(variants) >= needed:
            break
        try:
            answer = nawigator_edycje(
                seq,
                poziom=EDIT_LEVELS[number % len(EDIT_LEVELS)],
                ile_kodow=rng.randint(4, 12),
                opcji=options,
                ziarno=rng.randint(1, 999_999),
            )
        except ApiError as error:
            _log(verbose, f"edits unavailable ({error}) -- "
                          f"keeping the {len(variants)} variants so far", 2)
            break
        for option in answer["opcje"]:
            candidate = option["sekwencja"].upper()
            if candidate in seen:
                continue
            seen.add(candidate)
            if len(candidate) != len(seq) or check_sequence(candidate):
                continue
            variants.append(candidate)
    return variants


def mutate(
    seq: str,
    weights: list[float],
    hints: dict[int, str],
    n_mut: int,
    rng: random.Random | None = None,
) -> str:
    """Point-mutate `n_mut` positions, drawn proportionally to `weights`.

    Where the Navigator recommended a base, it is taken `HINT_PROBABILITY` of
    the time; otherwise a different base is picked at random. Positions are
    drawn with replacement, so the result can differ in slightly fewer than
    `n_mut` places.
    """
    draw = rng or random
    bases = list(seq)
    # A length mismatch means the map belongs to a different sequence; a
    # uniform draw is still a valid mutation, a misaligned one is not.
    aligned = weights if len(weights) == len(bases) else None

    for index in draw.choices(range(len(bases)), weights=aligned, k=n_mut):
        if index in hints and draw.random() < HINT_PROBABILITY:
            bases[index] = hints[index]
        else:
            bases[index] = draw.choice([b for b in BASES if b != bases[index]])
    return "".join(bases)


# --------------------------------------------------------------------------
# step 4: the children
# --------------------------------------------------------------------------
def _breed_one(
    rank: int,
    total: int,
    parent_id: str,
    seq: str,
    quota: int,
    options: int,
    mutations: tuple[int, int],
    rng: random.Random,
    verbose: bool,
) -> list[dict[str, Any]]:
    """Every child of one parent: one map, a pile of variants, then mutations."""
    _log(verbose, f"[{rank + 1}/{total}] {parent_id} -> {quota} children", 1)
    if quota <= 0:
        return []

    weights, hints = get_map(seq, verbose=verbose)
    variants = get_edits(seq, quota, options=options, rng=rng, verbose=verbose)
    _log(verbose, f"[{rank + 1}/{total}] {parent_id}: {len(hints)} hints, "
                  f"{len(variants)} navigator variants", 2)

    rows = []
    for number in range(quota):
        # Cycle the variants: with fewer of them than children, the point
        # mutations are what keeps the repeats apart.
        base = variants[number % len(variants)] if variants else seq
        child = mutate(base, weights, hints, rng.randint(*mutations), rng=rng)
        rows.append({
            ID_COL: f"{parent_id}_{number + 1:03d}",
            SEQ_COL: child,
            "parent_id": parent_id,
            "changes": hamming(seq, child),
        })
    return rows


def breed(
    parents: pd.DataFrame,
    k: int = 1000,
    keep_elite: int = 5,
    tau: float | None = None,
    options: int = DEFAULT_OPTIONS,
    mutations: tuple[int, int] = (1, 4),
    workers: int = WORKERS,
    seed: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Turn a parent table into (at most) `k` children.

    `parents` is what `select_top` returns: ``id``/``sequence``, best first.
    The first `keep_elite` are copied through unchanged, the rest of the
    budget is split by `allocate_children`, and each parent is bred from its
    own seeded RNG so a run is reproducible under `seed` no matter how the
    `workers` threads interleave.

    Returns ``id, sequence, parent_id, changes``, deduplicated on the
    sequence -- so expect slightly fewer than `k` rows.
    """
    parents = parents.reset_index(drop=True)
    if parents.empty:
        raise ValueError("breed() needs at least one parent")

    keep_elite = max(0, min(keep_elite, len(parents), k))
    quota = allocate_children(len(parents), k - keep_elite, tau)
    _log(verbose, f"breeding {k} children from {len(parents)} parents "
                  f"({keep_elite} elite kept): {quota}")

    rows: list[dict[str, Any]] = [
        {ID_COL: f"elite_{rank + 1:02d}",
         SEQ_COL: parents.at[rank, SEQ_COL],
         "parent_id": parents.at[rank, ID_COL],
         "changes": 0}
        for rank in range(keep_elite)
    ]

    master = random.Random(seed)
    jobs = [
        (rank, len(parents), str(parents.at[rank, ID_COL]), parents.at[rank, SEQ_COL],
         quota[rank], options, mutations, random.Random(master.getrandbits(64)), verbose)
        for rank in range(len(parents))
    ]
    with ThreadPoolExecutor(max(1, workers)) as pool:
        for produced in pool.map(lambda job: _breed_one(*job), jobs):
            rows.extend(produced)

    if not rows:
        return pd.DataFrame(columns=[ID_COL, SEQ_COL, "parent_id", "changes"])

    children = pd.DataFrame(rows).drop_duplicates(SEQ_COL).reset_index(drop=True)
    _log(verbose, f"done: {len(children)} unique sequences out of {len(rows)} bred")
    return children


def evolve(
    df: pd.DataFrame,
    k: int = 200,
    fraction: float = 0.2,
    keep_elite: int = 20,
    min_distance: int = 8,
    id_col: str = ID_COL,
    seq_col: str = SEQ_COL,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """`select_top` then `breed`: a ranked table in, a generation out.

    `df` must be sorted best-first. Extra keyword arguments (`tau`, `options`,
    `mutations`, `workers`, `seed`) go straight to `breed`.
    """
    parents = select_top(
        df,
        fraction=fraction,
        min_distance=min_distance,
        id_col=id_col,
        seq_col=seq_col,
        verbose=verbose,
    )
    return breed(parents, k=k, keep_elite=keep_elite, verbose=verbose, **kwargs)


def to_fasta(
    children: pd.DataFrame,
    limit: int = MAX_SCORED_SEQUENCES,
    drop_invalid: bool = True,
    id_col: str = ID_COL,
    seq_col: str = SEQ_COL,
) -> str:
    """Render a generation as FASTA for `/wgraj`.

    Delegates to `api.build_fasta`, so duplicates and sequences the server
    would reject are dropped. `limit` defaults to the 100 records that are
    ever scored -- pass ``limit=len(children)`` to write the whole table out.
    """
    return build_fasta(
        list(zip(children[id_col], children[seq_col])),
        limit=limit,
        drop_invalid=drop_invalid,
    )


if __name__ == "__main__":
    # Smoke test: breed a small generation off the raw hackathon CSV, whose
    # headers are still Polish (read_dataframe does not rename them).
    from hack_the_bromoter.utils import PROMOTERS_CSV, read_dataframe

    promoters = read_dataframe(PROMOTERS_CSV)
    generation = evolve(
        promoters,
        k=20,
        fraction=0.05,
        keep_elite=2,
        id_col="nazwa",
        seq_col="sekwencja",
        seed=0,
    )
    print(generation.head(10).to_string())
