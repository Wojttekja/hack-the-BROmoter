import math
import random
from collections import defaultdict

def sort_sequences(df, judge_many, id_col="id", seq_col="sequence"):
    """
    Sort a dataframe of sequences using a noisy pairwise comparator.

    df: pandas DataFrame with columns [id_col, seq_col]
    judge_many: the judge_many(calls) function from above -- called with
                a list of (a, b) id tuples, returns list of dicts with
                'a', 'b', 'idx', 'winner' in the SAME order as input.

    Returns: df sorted best-to-worst (list of ids, or a reordered df).

    Strategy: bottom-up merge sort, batching every comparison needed at
    a given "merge round" into a single judge_many call. This achieves
    the minimum ~O(n log n) total comparisons for a full ordering,
    while only requiring O(log n) sequential rounds (round-trips),
    since same-round merges are independent and can run in parallel.
    """
    ids = list(df[id_col])
    n = len(ids)
    if n <= 1:
        return ids

    # Each "run" is a list of ids, already sorted best-to-worst internally.
    runs = [[i] for i in ids]

    while len(runs) > 1:
        # Pair up runs to merge: (run0, run1), (run2, run3), ...
        pairs = [(runs[i], runs[i + 1]) for i in range(0, len(runs) - 1, 2)]
        leftover = runs[-1] if len(runs) % 2 == 1 else None

        # Pointers into each pair, to know which comparison is "next"
        # for that merge. We do this in waves: each wave issues one
        # comparison per still-active pair, batched into one judge_many.
        pointers = [[0, 0] for _ in pairs]       # (i into left, j into right)
        merged = [[] for _ in pairs]
        active = list(range(len(pairs)))

        while active:
            calls = []
            call_meta = []  # which pair each call belongs to
            for p in active:
                left, right = pairs[p]
                i, j = pointers[p]
                if i < len(left) and j < len(right):
                    calls.append((left[i], right[j]))
                    call_meta.append(p)
                else:
                    # one side exhausted -> flush remainder, no call needed
                    pass

            if calls:
                results = judge_many(calls)
                for (p, res) in zip(call_meta, results):
                    left, right = pairs[p]
                    i, j = pointers[p]
                    if res["winner"] == left[i]:
                        merged[p].append(left[i])
                        pointers[p][0] += 1
                    else:
                        merged[p].append(right[j])
                        pointers[p][1] += 1

            # figure out which pairs are now fully drained on one side
            still_active = []
            for p in active:
                left, right = pairs[p]
                i, j = pointers[p]
                if i < len(left) and j < len(right):
                    still_active.append(p)
                else:
                    merged[p].extend(left[i:])
                    merged[p].extend(right[j:])
            active = still_active

        runs = merged
        if leftover is not None:
            runs.append(leftover)

    return runs[0]


def bucket_sort_sequences(
    df,
    judge_many,
    id_col="id",
    seq_col="sequence",
    n_tiers=None,
    rounds=None,
    seed=None,
):
    """
    Rank sequences into coarse tiers using a noisy pairwise comparator,
    Swiss-tournament style, instead of a full sort.

    Rationale: the comparator can't reliably distinguish sequence 3 from
    sequence 4, only "roughly this group is better than that group".
    So instead of paying for a full O(n log n) sort, we run a small fixed
    number of rounds where each sequence is compared against opponents of
    similar current standing, tally wins, and bucket by win count. Total
    calls: ~n/2 * rounds, i.e. O(n) rather than O(n log n), with `rounds`
    a small constant (default scales gently with n).

    df: DataFrame with columns [id_col, seq_col]
    judge_many: judge_many(calls) -> list of dicts with 'a','b','winner',
                in the same order as the input `calls` list of (a, b) id tuples.
    n_tiers: number of output tiers/buckets. Defaults to a rough guess
             based on n (roughly n // 10, at least 1).
    rounds: number of Swiss rounds to run. Defaults to ceil(log2(n)) + 1,
            capped small (e.g. 3-7) since that's plenty of resolution
            for a comparator this coarse.
    seed: optional random seed for reproducibility of pairing/byes.

    Returns: list of tiers, each a list of ids, ordered best-to-worst tier
             first. Within a tier, order is NOT meaningful (arbitrary /
             stable by input order) since the comparator can't support it.
    """
    rng = random.Random(seed)

    ids = list(df[id_col])
    n = len(ids)
    if n <= 1:
        return [ids]

    if rounds is None:
        rounds = max(3, min(7, math.ceil(math.log2(n)) + 1))

    if n_tiers is None:
        n_tiers = max(1, n // 10)

    # Track score (wins) and opponent history per id.
    score = {i: 0 for i in ids}
    played = {i: set() for i in ids}  # ids already faced, to avoid repeats

    for _ in range(rounds):
        # Group by current score, then pair within score-groups where
        # possible (classic Swiss pairing), falling back to nearest
        # available score group if a group has an odd one out or
        # everyone left has already played each other.
        by_score = defaultdict(list)
        for i in ids:
            by_score[score[i]].append(i)

        # Shuffle within each score group so pairing isn't order-biased,
        # and so repeated ties don't always pick the same "first" id.
        for group in by_score.values():
            rng.shuffle(group)

        # Flatten score groups from highest to lowest score, then walk
        # down and pair consecutive not-yet-played ids. This keeps
        # pairings close in current standing (Swiss-style) without
        # needing a full matching algorithm.
        ordered = []
        for s in sorted(by_score.keys(), reverse=True):
            ordered.extend(by_score[s])

        pairs = []
        unpaired = []
        used = set()

        i_ptr = 0
        pool = ordered[:]
        while pool:
            a = pool.pop(0)
            if a in used:
                continue
            used.add(a)
            # find nearest not-yet-played, not-yet-used opponent
            partner = None
            for cand in pool:
                if cand not in used and cand not in played[a]:
                    partner = cand
                    break
            if partner is None:
                # everyone remaining has already played `a`, or pool empty
                # -> give a bye this round (no call needed)
                unpaired.append(a)
                continue
            pool.remove(partner)
            used.add(partner)
            pairs.append((a, partner))

        # Bye handling: a lone leftover just doesn't get a comparison
        # this round (no score change). With even n this rarely happens
        # except when history collisions force it.

        if pairs:
            calls = [(a, b) for (a, b) in pairs]
            results = judge_many(calls)
            for (a, b), res in zip(pairs, results):
                winner = res["winner"]
                score[winner] += 1
                played[a].add(b)
                played[b].add(a)

    # Bucket by final score into n_tiers roughly-equal-sized tiers,
    # best (highest score) first.
    ranked = sorted(ids, key=lambda i: (-score[i]))

    # tiers = _split_into_tiers(ranked, n_tiers)
    return ranked


def _split_into_tiers(ranked_ids, n_tiers):
    """Split an already-best-to-worst-ordered list into n_tiers
    contiguous, roughly-equal-sized chunks."""
    n = len(ranked_ids)
    n_tiers = max(1, min(n_tiers, n))
    base, extra = divmod(n, n_tiers)
    tiers = []
    idx = 0
    for t in range(n_tiers):
        size = base + (1 if t < extra else 0)
        tiers.append(ranked_ids[idx: idx + size])
        idx += size
    return tiers