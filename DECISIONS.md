# DECISIONS

Every ambiguity in the spec that we resolved ourselves, and why. Where the spec said
"pick the simplest option and note it", the choice is recorded here.

## Project layout & tooling

- **Package location.** `promo/` lives at the repository root (sibling of
  `native_promoter/`, which holds the data), matching the spec's layout diagram. The
  data loader resolves `native_promoter/data/promoters.fasta` relative to the repo
  root so it works regardless of the current directory.
- **Build backend.** Switched from `uv_build` (src-layout) to `hatchling` with
  `packages = ["promo"]`, and added `pythonpath = ["."]` to pytest so tests import
  `promo` even without an editable install. Removed the stale
  `[project.scripts] hack-the-bromoter` entry point.
- **Dependencies.** Kept the existing pinned numpy/pandas/polars/torch/etc. versions
  to avoid a disruptive re-resolution, and added the missing spec deps: `scipy`,
  `cma`, `tqdm`, `rich`, plus `pytest`/`ruff` as a dev group. Dropped `accelerate`,
  `ipykernel`, `seaborn` (not in the spec; unused).

## Native pks1 promoter

- The task targets the **pks1** (polyketide synthase) gene. The corpus contains
  exactly one record annotated `product = "polyketide synthase"`:
  `Trichoderma_TRIATDRAFT_289254`. We treat that as the native pks1 promoter and use
  it as seed index 0 (so KotH starts by defending the real baseline). Its locus tag
  is exposed as `seqs.NATIVE_PKS1_LOCUS`. If the organizers name a different native
  sequence on the day, change that one constant.

## Oracle contract

- **Winner encoding.** `Judge.compare` returns the literal `"A"`/`"B"` (which
  *argument* won), not the winning sequence, matching the spec. `Comparison.winner`
  stores the same.
- **Oracle exceptions live in `interfaces.py`** (`OracleError`, `RateLimitError`,
  `BudgetExhaustedError`), not in a backend, so the probe suite can catch rate-limit
  and budget conditions without importing any concrete backend.
- **`get_judge` returns the raw, uncached oracle.** Wrapping in `CachedJudge` is the
  caller's job (the runner does it). This is deliberate: the probe suite needs the
  raw handle to observe determinism, order bias and latency, which caching would
  hide.

## Mock ground truth (the rehearsal target)

- **Latent term.** To make latent-space search meaningful, the hidden score includes
  a smooth Gaussian bump around a fixed target latent point. When the judge is
  constructed *with* a navigator it uses that navigator's embedding; the default
  `get_judge("mock")` builds the judge *without* a navigator and falls back to an
  independent internal 3-mer embedding. We kept them independent by default because
  in the real event the Judge and Navigator are separate models — the optimizer
  should not assume they share a representation. Tests that measure ground-truth
  improvement always use the *same* judge configuration that produced the result.
- **Length preference is `tanh(n / 1200)`** — mild, monotone, saturating near
  1200 bp, as specified.
- **GC optimum is interior** (Gaussian around GC=0.52), so both GC-poor and GC-rich
  extremes are penalised.

## Mock pathologies (how each is made *detectable*)

- **Noise** is genuinely stochastic per call (flips the verdict with probability
  `noise_prob`), so the determinism probe sees inconsistent repeats. This is the one
  pathology that legitimately breaks determinism.
- **Order bias** is implemented as a *deterministic* per-unordered-pair coin (hashed,
  stable across runs): a fraction `order_bias` of pairs always let the first argument
  win. This keeps determinism intact while making `compare(a,b)` vs `compare(b,a)`
  asymmetric — exactly what the binomial order-bias probe detects. We chose this over
  a per-call random bias so that order bias and noise are *independently*
  detectable. The spec phrases it as a "probability of favouring the first argument";
  we realise that probability as a fraction of pairs rather than of calls.
- **Intransitivity** adds an antisymmetric *rotational tournament* term (each
  sequence gets a hidden angle; A beats B if A is within the first half-turn ahead of
  B). A rotational tournament is rich in 3-cycles, so the transitivity probe finds
  them once `intransitive_strength` is large enough to overcome score gaps.
- **Rate limit / hard budget / latency** are straightforward (rolling 60 s window,
  a call counter, and `time.sleep`).

## Mock navigator

- **Encoder** = k-mer counts → fixed seeded random projection → `dim` dims. Simple,
  deterministic, and gives small-step→small-edit-distance behaviour after decoding.
- **Decoder** blends against a codebook built from the corpus; mutation load scales
  with latent distance to the nearest anchor, and mutation positions are seeded by
  the latent vector so nearby points share most edits (small edit distance).
- **Poincaré spread.** Raw embeddings have tiny norm, so a naive tanh squash would
  leave every point near the origin and `detect_geometry` could not tell the space
  was hyperbolic. We scale by a data-driven factor that maps the largest corpus
  embedding to ~0.95, spreading the corpus across the ball so the hyperbolic
  detection *and* search paths are genuinely exercised. Encode/decode remain exact
  inverses (`_from_ball` divides by the same factor).

## Latent geometry

- **Curvature c = 1** for the Poincaré ball. A consequence: the geodesic *distance*
  reduces to Euclidean only up to the conformal factor λ₀ = 2 at the origin
  (`d_hyp → 2·d_euc`), while `expmap`, `logmap`, Möbius addition and interpolation
  reduce *exactly*. The optimizers only use distance monotonically, so the constant
  factor is irrelevant to search. The test asserts the exact factor-2 relationship
  rather than pretending distance collapses to Euclidean.
- **CMA-ES in hyperbolic space** searches the tangent space at a fixed base point
  (seed 0's latent) and maps every sample back with `expmap`, so it never does linear
  arithmetic inside the ball. We keep the base point fixed rather than re-anchoring
  each generation, to avoid the coordinate inconsistency that re-anchoring the
  tangent frame would introduce.

## Cache

- **Key is the ordered pair by default**, because the Judge may have order bias.
  `assume_symmetric=True` collapses both orderings and translates a stored reversed
  verdict on read. Only enable it after the order-bias probe proves symmetry.
- **Crash safety** = append one JSON line then `flush()` + `os.fsync()` on every put.
  A torn final line (process killed mid-write) is skipped on reload; all prior lines
  survive. This is tested.
- **Non-bypassable** = `CachedJudge` stores the raw client in a name-mangled private
  attribute; there is no public accessor, so no caller can reach the oracle without
  going through the cache.

## Ranking & optimizers

- **Every ranking primitive returns `(result, calls_used)`** and takes an optional
  budget, stopping cleanly (returning a best-effort partial result) when hit.
- **`evaluate()` is the per-optimizer ranking seam.** The spec names `ask`/`tell`/
  `state_dict`/`load_state_dict`; we added an `evaluate(judge, candidates, budget)`
  method whose *strategy* differs per optimizer (merge-sort for population methods,
  defend-the-champion for KotH, per-cell duels for MAP-Elites). The runner calls
  `ask → evaluate → tell` uniformly. `tell` still receives the ranked candidates, as
  specified.
- **Fair comparison is at equal *call budget*, not equal generations.** KotH spends
  ~1 call per challenger while population methods spend ~n·log n per generation, so
  KotH and CMAES only beat RandomSearch when all three are given the same number of
  oracle calls. The key test enforces this.
- **Global best is kept judge-consistent** by spending at most one extra comparison
  per generation (new top vs running best), so `best` never regresses across KotH
  restarts.

## Checkpoint/resume

- **Checkpoint is a single JSON file**, written atomically (`.tmp` then `replace`).
  CMA-ES's internal state is not JSON-native, so we base64-encode a pickle of the
  strategy object into the JSON — still one JSON file, still resumable.
- **Resume accounting.** The comparison cache is already crash-safe on its own; the
  checkpoint additionally stores `calls_spent` so a resumed run continues toward the
  same budget rather than restarting the count.

## Surrogate

- **Split by sequence, not by pair.** Validation pairs are those whose *both*
  sequences are held out; cross pairs (one side held out) are dropped. This prevents
  a sequence appearing in both train and val, which would inflate accuracy.

## Probe

- Probes use the **raw** judge (see above). Judge-call budget is split evenly across
  the judge probes; navigator probes are call-free. Each probe is individually
  runnable via `--probe <name>` and every probe honours `--budget`.
