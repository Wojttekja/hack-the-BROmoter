"""Entry point: the optimizer loop.

Re-exports the API, table and judge helpers so a notebook or script can do
``from hack_the_bromoter.main import *`` and have the whole toolbox.

Everything it needs lives in the sibling modules --
`api` (the HTTP endpoints), `utils` (loading/saving tables) and `judge`
(the rate-limited, order-debiased `/sedzia` comparator and the rankers
built on it).
"""

from __future__ import annotations

import time

import pandas as pd

from hack_the_bromoter.api import (
    ApiError,
    build_fasta,
    check_sequence,  # noqa: F401
    get_client,
    me,
    me_all,
    nawigator_edycje,  # noqa: F401
    nawigator_mapa,  # noqa: F401
    ranking,  # noqa: F401
    wgraj,  # noqa: F401
    wild_sequence,  # noqa: F401
)
from hack_the_bromoter.judge import (
    Judge,
    bucket_sort_sequences,
    copeland_scores,  # noqa: F401
    sort_sequences,  # noqa: F401
)
from hack_the_bromoter.utils import (
    ID_COL,
    ROOT,  # noqa: F401
    SEQ_COL,  # noqa: F401
    read_dataframe,  # noqa: F401
    read_promoters,
    save_dataframe,  # noqa: F401
    sequence_map,
)

# How many candidates survive each generation.
POPULATION = 20
GENERATIONS = 5

# /wgraj is capped at one upload per 5 minutes *per key*; a generation can
# finish faster than that, so a submission that finds every key still cooling
# down is skipped rather than waited out (pass wait=True to sit it out).
UPLOAD_COOLDOWN_SLACK = 2.0


def print_account() -> dict:
    """Team, quota used today, and the /sedzia per-minute cap."""
    account = me()
    print("team:", account["druzyna"],
          "| used today:", account["zuzycie_dzis"], "/", account["dzienny_limit_druzyny"],
          "| /sedzia per minute:", account["limity_na_minute"]["/sedzia"])
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
    fasta = build_fasta(sequence_map(population))
    records = fasta.count(">")
    if not records:
        print("  submission skipped: nothing survived the FASTA filters")
        return None

    cooldowns = [account["zgloszenie_mozliwe_za_s"] for account in me_all()]
    index = min(range(len(cooldowns)), key=cooldowns.__getitem__)
    if cooldowns[index] > 0:
        if not wait:
            print(f"  submission skipped: every key is on the upload cooldown "
                  f"({min(cooldowns):.0f} s left on the earliest)")
            return None
        print(f"  waiting {cooldowns[index]:.0f} s for key {index + 1} "
              f"to come off the upload cooldown")
        time.sleep(cooldowns[index] + UPLOAD_COOLDOWN_SLACK)

    try:
        answer = get_client().wgraj(fasta, key_index=index)
    except ApiError as error:
        print(f"  submission failed: {error}")
        return None

    print(f"  submitted {records} sequences on key {index + 1}:",
          f"scored {answer['ocenionych']}",
          f"| TOP10 {answer['pozycja_top10']}",
          f"| TOP100 {answer['pozycja_top100']}",
          f"| points {answer['punkty_razem']}")
    return answer


def optimize(population, judge, generations=GENERATIONS, keep=POPULATION):
    """The loop: propose candidates, rank them with the judge, keep the best.

    `population` is a DataFrame with [id, sequence]; the same shape comes back
    out, ordered best-first by the judge.
    """
    for generation in range(generations):
        # 1. propose new candidates from the current survivors
        # candidates = propose(population, judge) - chłopaki robiom
        candidates = population.iloc[:0]  # until propose() lands: no new blood

        # 2. rank the pool with the judge. sort_sequences is a full O(n log n)
        #    ordering; bucket_sort_sequences is the cheap O(n) Swiss variant
        #    for when we only need "roughly which group is better".
        pool = pd.concat([population, candidates], ignore_index=True)
        ranked_ids = bucket_sort_sequences(pool, judge.judge_many)

        # 3. keep the top `keep` and go again
        population = pool.set_index(ID_COL).loc[ranked_ids[:keep]].reset_index()

        print(f"generation {generation}: {len(population)} survivors "
              f"({judge.calls} judge calls spent)")

        # 4. submit what we have now -- only the best upload counts, so there
        #    is nothing to lose by scoring every generation.
        submit(population)

    return population


def main():
    print_account()

    promoters = read_promoters()
    judge = Judge(promoters)
    print(f"{len(promoters)} sequences loaded")

    best = optimize(promoters, judge)

    # save_dataframe(best, ROOT / "out" / "best.csv")
    return best


if __name__ == "__main__":
    main()
