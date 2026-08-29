"""Entry point: the optimizer loop.

Re-exports the API, table and judge helpers so a notebook or script can do
``from hack_the_bromoter.main import *`` and have the whole toolbox.

Everything it needs lives in the sibling modules --
`api` (the HTTP endpoints), `utils` (loading/saving tables) and `judge`
(the rate-limited, order-debiased `/sedzia` comparator and the rankers
built on it).

Every step prints a timestamped line so a long run is followable live; run
with ``python -u`` (or just rely on the flush below) to see it in real time.
"""

from __future__ import annotations

import time
import traceback

import pandas as pd

from hack_the_bromoter.api import (
    MAX_SCORED_SEQUENCES,
    ApiError,
    build_fasta,
    check_sequence,
    get_client,
    me,
    me_all,
    nawigator_edycje,  # noqa: F401
    nawigator_mapa,  # noqa: F401
    ranking,  # noqa: F401
    wgraj,  # noqa: F401
    wild_sequence,
)
from hack_the_bromoter.fitness import NavigatorFitness
from hack_the_bromoter.judge import (
    Judge,  # noqa: F401
    bucket_sort_sequences,  # noqa: F401
    copeland_scores,  # noqa: F401
    sort_sequences,  # noqa: F401
)
from hack_the_bromoter.navigator import evolve
from hack_the_bromoter.utils import (
    ID_COL,
    ROOT,
    SEQ_COL,
    convert_promoters,
    init_history,
    read_dataframe,
    save_dataframe,
    sequence_map,
)

# How many candidates survive each generation, and how many are bred.
POPULATION = 100
CHILDREN = 200
GENERATIONS = 10000

# /wgraj is capped at one upload per 5 minutes *per key*; a generation can
# finish faster than that, so a submission that finds every key still cooling
# down is skipped rather than waited out (pass wait=True to sit it out).
UPLOAD_COOLDOWN_SLACK = 2.0

# The running table of everything we have tried, `;`-separated like the rest
# of the hackathon data -- read_dataframe/save_dataframe default to that, so
# never touch it with a bare pd.read_csv (which would default to `,`).
SEQUENCES_BACKLOG = ROOT / "sequences.csv"
PROMOTERS_SOURCE = ROOT / "HackThePromotor" / "Promotory.csv"

_START = time.monotonic()


def log(message: str, indent: int = 0) -> None:
    """One timestamped progress line, flushed immediately."""
    print(f"[{time.monotonic() - _START:7.1f}s] {'  ' * indent}{message}", flush=True)


def print_account() -> dict:
    """Team, quota used today, and the /sedzia per-minute cap."""
    log("fetching /me ...")
    account = me()
    log(f"team: {account['druzyna']}"
        f" | used today: {account['zuzycie_dzis']}/{account['dzienny_limit_druzyny']}"
        f" | /sedzia per minute: {account['limity_na_minute']['/sedzia']}", 1)

    pool = me_all()
    log(f"{len(pool)} API key(s) in the pool", 1)
    for number, key in enumerate(pool, 1):
        log(f"key {number}: used {key['zuzycie_dzis']}"
            f" | upload possible in {key['zgloszenie_mozliwe_za_s']} s", 2)
    return account


def submit(population, wait: bool = False) -> dict | None:
    """Upload the current population to /wgraj and print what it scored.

    Only the best submission of the day counts for the ranking, so sending
    every generation is free -- the one cost is the 5 minute upload cooldown
    of the key that gets spent. The cooldown is per key, so the upload goes
    out on whichever key is free; when none is, the submission is skipped
    (`wait=True` sleeps until the earliest one comes back instead) and the
    function returns None.
    """
    log(f"preparing a submission from {len(population)} rows", 1)

    mapping = sequence_map(population)
    if len(mapping) < len(population):
        log(f"{len(population) - len(mapping)} rows share an id and were "
            f"collapsed -- ids must be unique", 2)

    invalid = sum(1 for sequence in mapping.values() if check_sequence(sequence.upper()))
    duplicates = len(mapping) - len({s.upper() for s in mapping.values()})

    fasta = build_fasta(mapping)
    records = fasta.count(">")
    log(f"FASTA: {records} records kept of {len(mapping)}"
        f" ({invalid} rejected by check_sequence, {duplicates} duplicate sequences,"
        f" cap {MAX_SCORED_SEQUENCES})", 2)
    if not records:
        log("submission skipped: nothing survived the FASTA filters", 2)
        return None

    log("checking the upload cooldown on every key ...", 2)
    cooldowns = [account["zgloszenie_mozliwe_za_s"] for account in me_all()]
    log("cooldowns (s): " + ", ".join(f"key {n}={c:.0f}"
                                      for n, c in enumerate(cooldowns, 1)), 3)
    index = min(range(len(cooldowns)), key=cooldowns.__getitem__)
    if cooldowns[index] > 0:
        if not wait:
            log(f"submission skipped: every key is on the upload cooldown "
                f"({cooldowns[index]:.0f} s left on the earliest)", 2)
            return None
        log(f"waiting {cooldowns[index]:.0f} s for key {index + 1} "
            f"to come off the upload cooldown", 2)
        time.sleep(cooldowns[index] + UPLOAD_COOLDOWN_SLACK)

    log(f"POST /wgraj with {records} sequences on key {index + 1} ...", 2)
    started = time.monotonic()
    try:
        answer = get_client().wgraj(fasta, key_index=index)
    except ApiError as error:
        log(f"submission failed after {time.monotonic() - started:.1f} s: {error}", 2)
        return None

    log(f"submitted in {time.monotonic() - started:.1f} s:"
        f" scored {answer['ocenionych']}"
        f" | TOP10 {answer['pozycja_top10']}"
        f" | TOP100 {answer['pozycja_top100']}"
        f" | points {answer['punkty_razem']}", 2)
    if answer.get("filtrowanie"):
        log(f"server-side filtering: {answer['filtrowanie']}", 2)
    return answer


def record(population, generation: int, scores: dict | None) -> pd.DataFrame:
    """Append this generation to the backlog CSV and return what was written."""
    results = pd.DataFrame({
        ID_COL: population[ID_COL].values,
        SEQ_COL: population[SEQ_COL].values,
        "gen": generation,
        "top10": scores["pozycja_top10"] if scores else 0,
        "pozycja_top100": scores["pozycja_top100"] if scores else 0,
        "points": scores["punkty_razem"] if scores else 0,
    })
    for column in ("parent_id", "fitness", "zmian", "blad"):
        if column in population:
            results[column] = population[column].values

    if scores is None:
        log("no score for this generation -- recording the rows with zeros", 1)

    if SEQUENCES_BACKLOG.is_file():
        previous = read_dataframe(SEQUENCES_BACKLOG)
        log(f"backlog holds {len(previous)} rows, appending {len(results)}", 1)
        results = pd.concat([previous, results], ignore_index=True)
    save_dataframe(results, SEQUENCES_BACKLOG)
    log(f"backlog saved: {len(results)} rows -> {SEQUENCES_BACKLOG}", 1)
    return results

def wild_enrichment(pool: pd.DataFrame, k=1) -> pd.DataFrame:
    """Add `k` freshly-sampled wild sequences to `pool`.

    IDs already in the backlog are a mix of plain integers (the original
    promoters) and lineage strings like ``elite_01``/``3_007`` (bred
    children), so `max() + 1` isn't a safe way to mint a new one. Use a
    `wild_NNN` id instead, bumping NNN past whatever wild ids already exist.
    """
    existing = set(pool[ID_COL].astype(str))
    used = 0
    wild_rows = []
    for _ in range(k):
        used += 1
        while f"wild_{used:03d}" in existing:
            used += 1
        new_id = f"wild_{used:03d}"
        existing.add(new_id)
        wild_rows.append({ID_COL: new_id, SEQ_COL: wild_sequence()})

    wild_df = pd.DataFrame(wild_rows)
    return pd.concat([pool, wild_df], ignore_index=True)

def _core(df: pd.DataFrame) -> pd.DataFrame:
    """Just the columns the loop carries between generations."""
    out = df[[ID_COL, SEQ_COL]].copy()
    out["parent_id"] = df.get("parent_id", pd.NA)
    return out


def optimize(population, fitness, generations=GENERATIONS, keep=POPULATION):
    """The loop: propose candidates, score them, keep the best.

    `population` is a DataFrame with [id, sequence]; the same shape comes back
    out, ordered best-first by `fitness`.

    Selection used to run on `/sedzia` through `bucket_sort_sequences`. It no
    longer does. Once the pool converged past the wild type the judge tied on
    *every* intra-population pair -- including best against worst -- so that
    ranking was returning coin flips and selection pressure had fallen to
    zero, which is what flattened the score. `NavigatorFitness` still spreads
    across the pool, and costs one call per candidate instead of fourteen.
    See `hack_the_bromoter.fitness` for the measurements.
    """
    for generation in range(generations):
        log(f"=== generation {generation}/{generations - 1}: "
            f"{len(population)} sequences in ===")

        # 1. propose new candidates from the current survivors.
        #    `wild_enrichment` used to run here; it injected the wild type --
        #    the one sequence we know is weakest -- into a pool that is then
        #    submitted, so it could only ever cost points on ALL100.
        candidates = evolve(population, k=CHILDREN)

        # `breed` re-mints elite_01..elite_NN every generation, so ids collide
        # between generations: the backlog's dedupe would collapse unrelated
        # sequences onto one row, and any id-keyed cache serves the previous
        # generation's answer. Stamp the generation in to keep them unique.
        candidates = candidates.assign(**{
            ID_COL: [f"g{generation:03d}_{n:04d}" for n in range(len(candidates))]
        })
        log(f"proposed {len(candidates)} new candidates", 1)

        # 2. score the survivors *and* the candidates together. The parents
        #    are already in the fitness cache so carrying them is free, and it
        #    is what makes elitism actually hold -- scoring the children alone
        #    lets a generation come out worse than the one before it.
        pool = pd.concat([_core(population), _core(candidates)], ignore_index=True)
        pool = pool.drop_duplicates(subset=SEQ_COL).reset_index(drop=True)

        calls_before = fitness.calls
        started = time.monotonic()
        log(f"scoring {len(pool)} sequences with /nawigator/mapa ...", 1)
        ranked = fitness.rank(pool)
        spent = fitness.calls - calls_before
        log(f"scored in {time.monotonic() - started:.1f} s"
            f" | {spent} map calls | {len(pool) - spent} cache hits"
            f" | {fitness.failures} failures so far", 1)
        log("top 5: " + ", ".join(
            f"{row[ID_COL]}(fit={row['fitness']:.1f}"
            f" zmian={row['zmian']:.0f} blad={row['blad']:.0f})"
            for _, row in ranked.head(5).iterrows()), 1)

        # 3. keep the top `keep` and go again
        population = ranked.head(keep).reset_index(drop=True)
        log(f"generation {generation}: {len(population)} survivors"
            f" | best fitness {population['fitness'].iloc[0]:.2f}"
            f" | median {population['fitness'].median():.2f}", 1)

        # 4. submit what we have now -- only the best upload counts, so there
        #    is nothing to lose by scoring every generation.
        scores = submit(population, wait=True)
        record(population, generation, scores)

    log(f"=== optimizer finished: {len(population)} sequences, "
        f"{fitness.calls} map calls ===")
    return population


def main():
    log("start")
    print_account()

    if SEQUENCES_BACKLOG.is_file():
        log(f"reusing the existing backlog at {SEQUENCES_BACKLOG}")
    else:
        log(f"seeding the backlog from {PROMOTERS_SOURCE} ...")
        written = convert_promoters(str(PROMOTERS_SOURCE), str(SEQUENCES_BACKLOG))

        init_history()
        log(f"wrote {written} rows to {SEQUENCES_BACKLOG}", 1)

    promoters = read_dataframe(SEQUENCES_BACKLOG)
    log(f"loaded {len(promoters)} rows, columns {list(promoters.columns)}", 1)

    missing = {ID_COL, SEQ_COL} - set(promoters.columns)
    if missing:
        raise KeyError(f"{SEQUENCES_BACKLOG} is missing {sorted(missing)} -- "
                       f"it must be a `;`-separated table with id and sequence")

    # Dedupe on the *sequence*, not the id: ids repeat across generations
    # (elite_01, elite_02, ...) for entirely different sequences, so an
    # id-keyed dedupe silently throws away distinct candidates.
    before = len(promoters)
    promoters = promoters.drop_duplicates(subset=SEQ_COL, keep="last").reset_index(drop=True)
    if len(promoters) != before:
        log(f"dropped {before - len(promoters)} rows repeating a sequence", 1)
    promoters[ID_COL] = [f"seed_{n:04d}" for n in range(len(promoters))]

    fitness = NavigatorFitness()
    log(f"fitness ready: {fitness!r}", 1)

    best = optimize(promoters, fitness)

    # save_dataframe(best, ROOT / "out" / "best.csv")
    return best


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("interrupted by the user")
    except Exception:
        log("crashed:")
        traceback.print_exc()
        raise
