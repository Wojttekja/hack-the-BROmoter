# hack-the-BROmoter

Black-box promoter optimization for the **pks1** gene of *Trichoderma atroviride*,
against the HYPPE oracles (a pairwise **Judge** and a hyperbolic **Navigator**).

Everything is built and rehearsed against a **mock** so that on the day we swap one
import (`PROMO_BACKEND=real`) and run.

```sh
uv sync
uv run pytest                                             # 44 tests, ~10s
uv run python -m promo.probe  --backend mock --budget 400 # interrogate the oracle
uv run python -m promo.runner --optimizer koth --backend mock --budget 500
```

## Architecture (one swap point)

```
promo/
  interfaces.py    Protocols: Judge, Navigator; Comparison, Candidate; oracle errors
  backend.py       get_judge() / get_navigator()  <-- THE ONLY SWAP POINT
  mock_backend.py  full fake oracle + HIDDEN ground truth (rehearsal target)
  real_backend.py  adapter skeleton, all TODOs   <-- edit this on the morning
  cache.py         crash-safe JSONL cache + non-bypassable CachedJudge
  partial_order.py transitive closure + TransitiveJudge (off by default)
  seqs.py          FASTA IO, GC, k-mers, motifs, plausibility, dinuc-shuffle
  latent.py        Euclidean + Poincare-ball ops (Mobius/expmap/logmap), LatentOps
  ranking.py       comparison-efficient sort / top-k / champion / Bradley-Terry
  optimizers/      base, random, koth, cmaes, map_elites
  surrogate.py     Bradley-Terry model in torch (hourly refresh)
  runner.py        main loop: budget, checkpoint, resume, rich progress
  probe.py         oracle interrogation suite
  analysis.py      regret curve, k-mer enrichment, heatmaps, PCA
```

**Rule:** nothing imports a concrete backend except `backend.py`. All oracle access
goes through the Protocols in `interfaces.py`. To swap oracles, set `PROMO_BACKEND`
(or pass `--backend`); do not import `mock_backend`/`real_backend` anywhere else.

The comparison cache is an append-only JSONL file plus an in-memory dict — no
database. It flushes+fsyncs after every real call, so a process killed at hour 9
resumes with zero data loss.

---

# MORNING OF THE HACKATHON

A tight, ordered checklist. Do these in order.

## 1. Wire up the real oracles (edit ONE file)

Open `promo/real_backend.py` and fill in the numbered TODOs. Nothing else changes.

| TODO | Location | What to do |
|------|----------|------------|
| **#0** | line ~25, `import hyppe` | Replace `hyppe` with the real package name so the import guard finds it. |
| **#1** | `RealJudge.__init__` (~line 45) | Construct the judge handle (`self._judge = ...`); delete the `raise NotImplementedError`. Keep the handle private. |
| **#2** | `RealJudge.compare` (~line 62) | Map the real verdict to `"A"`/`"B"`. See the four shapes listed in the docstring (bool / winning-seq / class-index / probability). |
| **#3** | `RealNavigator.__init__` (~line 83) | Construct the navigator handle; delete the `raise`. |
| **#4** | `RealNavigator.encode` (~line 96) | Return a 1-D `np.ndarray` (`.detach().cpu().numpy()` if it hands back a tensor). |
| **#5** | `RealNavigator.decode` (~line 105) | Accept a 1-D `np.ndarray`, return an ACGT string. |
| **#6** | `RealNavigator.dim` (~line 118, optional) | Return the real latent dim, or **delete** the property (the probe infers it). |
| **#7** | `RealNavigator.distance` (~line 126, optional) | Forward to HYPPE's distance if it exists, else **delete** it (do NOT fake a Euclidean distance if the space is hyperbolic — `promo.latent` handles geometry). |

Then confirm the seam works:

```sh
PROMO_BACKEND=real uv run python -c "from promo import backend; \
  print(backend.get_judge().compare('ACGT','TTTT'))"
```

## 2. Interrogate the oracle (probes, in this order)

Run the whole suite first, then drill into anything surprising:

```sh
uv run python -m promo.probe --backend real --budget 400
```

Order and rationale (each is individually runnable with `--probe <name>`):

1. **`determinism`** — is the same pair always judged the same way? Everything else
   assumes at least mostly-deterministic verdicts.
2. **`latency`** — measures latency and detects **rate limiting** / a **hard call
   budget**. Learn your real per-call cost and ceiling *before* spending the budget.
3. **`order_bias`** — does the first argument have an edge? (binomial test)
4. **`transitivity`** — count 3-cycles among ~10 sequences.
5. **`length_bias`** — the big one: does the Judge reward **content or merely size**?
   Nested 3'-anchored truncations + length-matched controls (strong-truncated vs
   weak-full, strong vs self padded with random / dinuc-shuffled filler).
6. **`shuffle`** / **`motif_ablation`** — does sequence structure / do TATA & CCAAT
   motifs actually matter?
7. **`gc_sweep`** — preferred GC of synthetic sequences.
8. **`league_table`** — rank all native promoters against each other (our strongest
   real seeds).
9. Navigator: **`roundtrip`** → **`geometry`** → **`step_calibration`** →
   **`interpolation`** → **`radius_sweep`**. `geometry` tells you Euclidean vs
   Poincaré; `step_calibration` (saves a plot) gives the perturbation scale that
   changes sequences but keeps them plausible — use it as your optimizer step.

The report is written to `probe_out/probe_report.json` plus a rich table.

## 3. Set config flags from probe outcomes

| Probe outcome | Action |
|---------------|--------|
| **Order bias detected** (`order_bias_detected: true`) | **Keep `--assume-symmetric` OFF.** Cache each ordering separately; the reversed pair is a genuinely different question. |
| **No order bias** (clean binomial) | Turn `--assume-symmetric` **ON** to halve cache misses. |
| **Transitivity confirmed** (`n_cycles: 0`) | Enable `--transitive`: the `TransitiveJudge` answers implied pairs for free. |
| **Intransitive** (`n_cycles > 0`) | **Keep `--transitive` OFF** (closure would be unsound). Prefer `bradley_terry_rank`, which aggregates many noisy comparisons. |
| **Non-deterministic / noisy** (`deterministic: false`) | Prefer Bradley-Terry ranking + the surrogate over single-comparison decisions; consider repeating close calls. |
| **Rate limited** (`rate_limited: true`) | Lower throughput / add backoff; lean harder on the cache and the surrogate to avoid redundant calls. |
| **Hard budget** (`budget_exhausted: true`) | Set `--budget` below the ceiling; spend calls on champion defense (cheapest) not full sorts. |
| **Length bias = size-driven** (`verdict: size-driven`) | Fix candidate length near the best truncation; do not let the optimizer win by inflating length. |
| **Length bias = content-driven** | Free to vary length; optimize motifs/composition. |
| **Geometry = poincare** | Pass `--geometry poincare` (or leave `--geometry auto`); CMA-ES will optimize in the tangent space automatically. |
| **Geometry = euclidean** | `--geometry euclidean` (or `auto`). |
| **`step_calibration` recommends scale s** | Use it as the optimizer step (`step`/`scale`/`sigma0`). |

`--geometry auto` (the default) detects the geometry from a sample of real latents,
so you usually don't have to decide manually.

## 4. Optimize

Start with KotH (cheapest, most call-efficient), keep MAP-Elites running for
diversity, refresh the surrogate hourly.

```sh
# cheapest steerable search; resumes after any crash
uv run python -m promo.runner --optimizer koth --backend real \
  --budget 4000 --geometry auto --assume-symmetric --out runs

# if it dies, just add --resume (cache + checkpoint make this lossless)
uv run python -m promo.runner --optimizer koth --backend real \
  --budget 8000 --out runs --resume

# quality-diversity archive (for the heatmap and varied submissions)
uv run python -m promo.runner --optimizer map_elites --backend real --budget 4000

# refresh the surrogate from the growing cache (do this each hour)
uv run python -c "from promo.surrogate import BradleyTerrySurrogate as S; \
  print(S().retrain_from_cache('runs/koth_cache.jsonl').val_accuracy)"
```

## 5. Analyse & choose a submission

```sh
uv run python -m promo.analysis --cache runs/koth_cache.jsonl --backend real --out figures
```

Compare **unfiltered best** vs **defensible best**: run `seqs.plausibility(seq)` on
the champion. If the unfiltered winner is implausible (bad GC, homopolymers), submit
the best sequence that passes plausibility instead.

---

## Notes

- Native pks1 promoter = corpus record `Trichoderma_TRIATDRAFT_289254`
  (`product = "polyketide synthase"`), used as seed 0. See `DECISIONS.md`.
- `_ground_truth` exists **only** in the mock, for tests/analysis. The real Judge has
  no score; never call it from the optimization path.
- All ambiguities we resolved are documented in `DECISIONS.md`.
