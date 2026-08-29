"""Entry point: the optimizer loop.

Re-exports the API, table and judge helpers so a notebook or script can do
``from hack_the_bromoter.main import *`` and have the whole toolbox.

Everything it needs lives in the sibling modules --
`api` (the HTTP endpoints), `utils` (loading/saving tables) and `judge`
(the rate-limited, order-debiased `/sedzia` comparator and the rankers
built on it).
"""

from __future__ import annotations

from hack_the_bromoter.api import (
    build_fasta,  # noqa: F401
    check_sequence,  # noqa: F401
    me,
    nawigator_edycje,  # noqa: F401
    nawigator_mapa,  # noqa: F401
    ranking,  # noqa: F401
    wgraj,  # noqa: F401
    wild_sequence,  # noqa: F401
)
from hack_the_bromoter.judge import (
    Judge,
    bucket_sort_sequences,  # noqa: F401
    copeland_scores,  # noqa: F401
    sort_sequences,  # noqa: F401
)
from hack_the_bromoter.utils import (
    ID_COL,  # noqa: F401
    ROOT,  # noqa: F401
    SEQ_COL,  # noqa: F401
    read_dataframe,  # noqa: F401
    read_promoters,
    save_dataframe,  # noqa: F401
    sequence_map,  # noqa: F401
)

# How many candidates survive each generation.
POPULATION = 20
GENERATIONS = 5


def print_account() -> dict:
    """Team, quota used today, and the /sedzia per-minute cap."""
    account = me()
    print("team:", account["druzyna"],
          "| used today:", account["zuzycie_dzis"], "/", account["dzienny_limit_druzyny"],
          "| /sedzia per minute:", account["limity_na_minute"]["/sedzia"])
    return account


def optimize(population, judge, generations=GENERATIONS, keep=POPULATION):
    """The loop: propose candidates, rank them with the judge, keep the best.

    `population` is a DataFrame with [id, sequence]; the same shape comes back
    out, ordered best-first by the judge.
    """
    for generation in range(generations):
        # 1. propose new candidates from the current survivors
        # candidates = propose(population, judge) - chłopaki robiom

        # 2. rank the pool with the judge. sort_sequences is a full O(n log n)
        #    ordering; bucket_sort_sequences is the cheap O(n) Swiss variant
        #    for when we only need "roughly which group is better".
        pool = pd.concat([population, candidates], ignore_index=True)
        ranked_ids = bucket_sort_sequences(pool, judge.judge_many)

        # 3. keep the top `keep` and go again
        population = pool.set_index(ID_COL).loc[ranked_ids[:keep]].reset_index()

        print(f"generation {generation}: not implemented "
              f"({judge.calls} judge calls spent)")

        # Submission
        fasta = build_fasta(population)
        

    return population


def main():
    print_account()

    promoters = read_promoters()
    judge = Judge(promoters)
    print(f"{len(promoters)} sequences loaded")

    best = optimize(promoters, judge)

    # save_dataframe(best, ROOT / "out" / "best.csv")
    # wgraj(build_fasta(dict(zip(best[ID_COL], best[SEQ_COL]))))
    return best


if __name__ == "__main__":
    main()
