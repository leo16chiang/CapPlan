# CapPlan

Forecast production prime-time peak MIPS per application, two fiscal years out,
in a form that survives an interview with an app custodian.

Two things make this harder than it looks:

1. **A peak is a maximum.** Data are thin, the tail is heavy, and mean-based
   losses are biased for a maximum.
2. **Peaks do not sum.** Applications peak in different intervals, so adding up
   application peaks overstates the LPAR. How much is a property of your
   workload, not a constant -- `capplan diagnostics` measures it, and it is the
   first thing to run.

The answer to both is the same: **the model never forecasts a peak.**

## Three stages

| Stage | What it produces | Where |
|---|---|---|
| 1 | A marginal predictive distribution per (app, future 15-min interval) | `capplan/model/` |
| 2 | The joint structure -- across time *and* across applications | `capplan/sim/residuals.py`, `bootstrap.py`, `copula.py` |
| 3 | Sampled paths, summed across apps interval by interval, then reduced | `capplan/sim/simulate.py`, `reducers.py` |

The peak falls out of Stage 3 as a property of the simulated sum. The order of
operations in `simulate()` is the substance of the whole design:

```
draw residuals -> add to marginals -> SUM ACROSS APPS -> take the maximum
```

Taking the maximum before the sum gives the sum of app peaks, which is the 33%
error. It is one transposition away, which is why the sum is a named step.

## A note on the numbers in this repository

The synthetic generator (`capplan/data/synth.py`) exists so the pipeline is
runnable and testable before your extract lands. It is a **test fixture, not
evidence**. It can show that the machinery computes what it claims -- that the
reduction takes a peak of a sum rather than a sum of peaks, that the memory
ceiling holds, that a transferred coincidence factor is applied correctly. It
can say nothing whatsoever about your workload, because its properties are
ones I chose.

Every number that matters -- the coincidence factor, the growth rates, whether
a neural model beats the baselines, how far apart the reduction conventions sit
-- has to come from your data. The commands below are how you get them.

## Start here

```bash
pip install -e '.[dev]'

capplan check                     # environment; answers the torch/proxy question
capplan ingest --synthetic        # runs end to end with no database
capplan diagnostics               # <- the go/no-go decision
```

With your own data, from Db2 (see `docs/data_input.md`):

```bash
pip install -e '.[db2]'
cp .env.example .env && $EDITOR .env               # credentials; git-ignored
capplan sources                                    # driver? env vars?
capplan tables                                     # which table may feed what
# edit config/sources.yaml -- the table names are guesses
capplan probe  --source db2 --from 2025-06-01 --to 2025-06-07
capplan ingest --source db2 --from 2022-11-01 --to 2025-10-31
```

`capplan probe` samples a few hundred rows and checks them against what CapPlan
assumes -- interval length above all. Run it after every edit to the queries.

`capplan diagnostics` is two pure-SQL queries over data you already hold, and it
can tell you to stop. If the coincidence factor comes back near 1.0, application
peaks effectively *do* coincide at your site, a joint simulation buys little,
and the effort belongs in the marginals instead. That is a legitimate and
cheap outcome, and it is why the diagnostics come before any modelling.

If it says proceed:

```bash
capplan simulate --paths 10000    # Stages 1-3, writes the serving tables
capplan evaluate                  # rolling-origin: does Stage 1 beat the baselines
capplan backtest                  # does the SIMULATION reproduce realised LPAR peaks
capplan pack                      # the custodian interview pack
```

## The two things worth building first

Both are pure SQL over data already on disk, and together they decide whether
the rest is worth doing.

**Sum-of-app-peaks vs realised LPAR peak** (`diagnostics/sql/coincidence.sql`).
The coincidence factor. Also the number every later stage is validated against:
if the simulation does not reproduce it, the forward peak is wrong in the same
direction, and nobody finds out until a hardware config is signed.

**Per-app submission bias** (`diagnostics/sql/submission_bias.sql`). What each
custodian submitted against what actually happened. Two uses: it is the honest
benchmark -- a model that cannot beat "last cycle's submission, de-biased" is
not worth deploying -- and it opens every custodian interview.

## Layout

```
capplan/
  data/         calendar, ingest, mips_normalisation, event_labels, synth
  diagnostics/  the two pure-SQL checks, plus data profiling
  model/        features, backends, train, calibrate, baselines   (Stage 1)
  sim/          residuals, bootstrap, copula, simulate, reducers  (Stages 2-3)
  eval/         rolling_origin, coverage, vs_submissions, sim_backtest
  serve/        forecast_store, scenario, pack_gen
  pipeline.py   one function per CLI verb, each writing a versioned run
  registry.py   versioned artefact dir + manifest JSON
```

## Decisions worth knowing about

**The reducer is pluggable, because the question is not settled.** Nobody has
pinned down whether the fiscal-year figure is a single annual maximum, the mean
of monthly peaks, a percentile of daily peaks, or -- for software cost -- the
monthly peak rolling 4-hour average IBM actually bills on. These are different
numbers answering different questions. So the reduction is written as
`reduce(path) -> scalar` (`capplan/sim/reducers.py`) and the architecture stops
caring. `capplan reducers` groups them by the decision they serve.

**Memory is an architectural constraint, not a tuning problem.** Materialising
paths x days x intervals x apps is 736 GB at the target settings. Paths are
chunked, samplers expose a lazy per-day stream, the sum across apps happens
inside the day loop, and only reduced `(paths, days)` series survive. Measured:
10,000 paths x 512 days in 69s with a 116 MB allocation peak.

**The block bootstrap is primary; the copula is the comparison.** Resampling
whole aligned day-blocks assumes nothing about the distribution and is about
80 lines of numpy. "Days like this have happened; here is what they looked
like" survives a review in a way a 1,260-dimensional Gaussian copula does not.
The copula is implemented too, with an explicit measurement of what its
zero-tail-dependence assumption costs. Run both: agreement is worth more than
either number alone.

**torch is optional.** `neuralforecast` (N-HiTS, TFT) is the intended Stage 1
backend, but torch is a large wheel and its presence on the internal mirror is a
week-1 unknown. Everything runs without it on the `quantile_ridge` backend --
IRLS on the pinball loss, pure numpy -- which clears the baseline gate on its
own. See `docs/week1_checklist.md`.

## Scope

**In:** production LPARs, prime time (business days 08:00-17:00), 15-minute
intervals, fiscal year November-October, top 30-40 applications.

**Out:** off-prime, non-production, cost. DR / IST / GCC SDF are out as forecast
*targets* but are labelled when they land in prime time, excluded from training
and from the residual pool, and reported separately -- out of scope to
forecast, and still something the hardware has to survive.

Capture ratio and MIPS-per-MSU are **applied as given**. CapPlan does not
estimate them; it records which value was used on every row so a disputed number
is a lookup rather than an argument.

## Everything depends on 15-minute intervals

`calendar.interval_minutes: 15` gives 36 prime-time intervals per business day
and, across ~35 apps and ~250 business days a year, the ~950k rows that justify
a neural Stage 1 at all. Confirm it in week 1. If the extract turns out to be
coarser, `capplan/data/calendar.py` is the only module that changes -- but the
argument for the neural backend changes with it.

## Documentation

- **`docs/runbook.md` -- start to finish: ingest, baselines, neural model,
  comparison, backtest, pack. Start here.**
- `docs/grain_decision.md` -- hourly vs daily, and the fallback if the hourly
  spine does not work out
- `docs/data_input.md` -- getting your SMF data in, from Db2 or a file drop
- `docs/peak_vs_average.md` -- which number you actually want, and why R4HA is
  not a choice
- `docs/week1_checklist.md` -- what to confirm before writing more code
- `docs/architecture.md` -- why each stage is shaped the way it is
- `docs/defects_found.md` -- bugs this codebase caught by measuring its own output (ten of them)
