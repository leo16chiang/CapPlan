# Runbook: start to finish

Every number that matters comes from your data. This walks through getting it
in, deciding the grain, running the baselines, running the neural model,
comparing them properly, and producing the pack.

**If you want the shortest possible path, do Part 1 and Part 2 and stop.**
Those two answer "is this worth building" and take an afternoon. Everything
after them is only worth doing if Part 2 says so.

---

## Part 0 — Install

```bash
git clone <this repo> && cd CapPlan
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev,db2]'
capplan check
```

`capplan check` lists what is present. `numpy scipy pandas duckdb pyarrow yaml`
are required; everything else is optional and the pipeline runs without it.

Credentials:

```bash
cp .env.example .env
$EDITOR .env
```

```
CAPPLAN_DB2_HOST=your_host
CAPPLAN_DB2_PORT=50000
CAPPLAN_DB2_DATABASE=your_db
CAPPLAN_DB2_USER=your_user
CAPPLAN_DB2_PASSWORD=...
CAPPLAN_DB2_DRIVER={IBM DB2 ODBC DRIVER}
```

`.env` is git-ignored. Real environment variables always beat it, so a
scheduled batch injects credentials normally. Confirm the driver name with
`odbcinst -q -d`.

```bash
capplan sources     # driver found? .env loaded? which vars are set?
```

---

## Part 1 — Get the data in

### 1.1 Decide which table is the spine

```bash
capplan tables
```

Exactly one table may feed `intervals`: a **spine** — complete, non-overlapping,
with intra-day timing. For an IZPCA database that is the SMF 72 workload
activity table, because WLM classifies every unit of dispatchable work exactly
once.

Everything else on your list overlaps it. A CICS transaction's CPU is in the
CICS table, *and* in the CICS regions' service class on the spine, *and* in the
regions' address space in SMF 30. Adding them roughly doubles the answer, and
the result looks plausible rather than wrong. `capplan ingest` refuses.

### 1.2 Find your interval

```sql
SELECT * FROM <schema>.MVSPM_TIME_RES;
```

This sets IZPCA's aggregation interval. It defaults to one hour and cannot be
finer than the SMF interval feeding it.

| Grain | Prime intervals/day | Rows, 35 apps x 3 yr |
|---|---|---|
| 15 min | 36 | ~945,000 |
| 30 min | 18 | ~470,000 |
| 60 min | 9 | ~236,000 |

Set `calendar.interval_minutes` in `config/capplan.yaml` to match.

### 1.3 Edit the queries

`config/sources.yaml`. The table and column names shipped there are guesses —
`APPL_CODE`, `MSU_USED`, `MVS_SYSTEM_ID` are plausible but not yours. Check
against your catalogue:

```sql
SELECT NAME, COLTYPE, LENGTH FROM SYSIBM.SYSCOLUMNS
WHERE TBNAME = 'MVSPM_WORKLOAD2_HV' ORDER BY COLNO;
```

The contract is only that each query returns these aliases:

| Table | Required | Useful |
|---|---|---|
| `intervals` | `ts`, `app_id`, `lpar`, and one of `msu`/`mips` | `environment`, `cpu_model` |
| `lpar_totals` | `business_date`, `lpar`, `peak_mips` | `mean_mips`, `peak_interval_idx` |
| `events` | `event_type`, `start_ts`, `end_ts` | `lpar`, `app_id`, `note` |
| `submissions` | `app_id`, `fiscal_year`, `submitted_peak_mips` | `submitted_on`, `basis` |

Two parameter markers per query, window start then window end. CapPlan binds
`date` objects; never format dates into the SQL.

### 1.4 Probe before extracting

```bash
capplan probe --source db2 --from 2025-06-01 --to 2025-06-07
```

Reads a few hundred rows and checks them against what CapPlan assumes. Run it
after every edit. What to look for:

- **`FAIL intervals: ... looks like N minutes`** — your config and your data
  disagree. Fix before anything else.
- **`FAIL intervals: only 1 distinct app_id`** — the app code is not populated
  or the join is not joining. Nothing works until this does.
- **`FAIL lpar_totals: nothing came back`** — see 1.6. This is the most
  important dependency in the project.
- **`WARN submissions: none found`** — no benchmark. The model would have
  nothing to be better than.

### 1.5 Extract

```bash
capplan ingest --source db2 --from 2022-11-01 --to 2025-10-31
```

Chunked a month at a time with `fetchmany`, reconnecting between chunks so a
long extract survives gateway idle timeouts. Start with three fiscal years; you
need at least two to backtest anything.

### 1.6 About `lpar_totals`

The realised LPAR peak is what the simulation is validated against. It should
come from SMF 70-1 (RMF partition data), **not** from summing the same rows
that feed `intervals` — otherwise the backtest compares the simulation against
its own assumption and passes regardless of how wrong the coincidence model is.

The shipped query derives it from the ungrouped spine as a fallback. That still
tests whether the coincidence structure was reconstructed (the peak of the
ungrouped sum is a genuinely different quantity from the sum of per-app peaks),
but it no longer detects work unattributed to any scoped application. Getting a
real 70-1 table remains the single most valuable data dependency here.

---

## Part 2 — The go/no-go

```bash
capplan diagnostics
```

Two pure-SQL queries over what you just landed. This can tell you to stop, and
that is a legitimate and cheap outcome.

**Coincidence factor.** `peak of the summed applications ÷ sum of application
peaks`, per day.

- **Near 1.0** → application peaks effectively coincide at your site. Summing
  them is nearly right, the three-stage simulation buys little, and the effort
  belongs in the marginals instead. *Stop and reconsider.*
- **Materially below 1** → summing application peaks overstates the LPAR by
  `1/factor - 1`. That is the error Stage 2 removes, and it is the business case
  for the rest of this. *Proceed.*

**Submission bias.** What each custodian forecast against what happened. Two
uses: it is the honest benchmark — a model that cannot beat "last cycle's
submission, de-biased" is not worth deploying — and it opens every custodian
interview.

**`n_unattributed_days`** — days where the LPAR peak exceeded the sum of scoped
application peaks. That is workload nobody has attributed, and it is a scoping
conversation before it is a modelling one.

### If your spine turns out to be daily

If the hourly table has no app code, or does not exist, see
`docs/grain_decision.md`. Short version: daily-only forecasting reproduces the
full sum-of-peaks overstatement, because with one interval a day "peak of the
sum" *is* "sum of the peaks". The fix is to measure the coincidence factor on
whatever sub-daily sample exists and transfer it:

```bash
capplan coincidence --sample-days 63
```

It reports whether your sample is adequate. Daily-grain `capplan simulate` runs
then apply it automatically.

---

## Part 3 — Baselines first

Fit the baselines and score them **before** touching the neural model, so there
is a bar to clear rather than a number to rationalise.

```bash
capplan evaluate --folds 4 --horizon-days 60
```

Rolling-origin: expanding-window fits, scored on held-out windows. Every model
goes through the same code path — a comparison where the baseline takes a
different route is not a comparison.

Baselines:

- **`seasonal_naive`** — last same-weekday value at the same interval, with
  quantiles from the empirical week-over-week ratio distribution. No trend, so
  it is expected to lose at long horizons; by how much is the number that
  justifies fitting a trend at all.
- **`interval_climatology`** — empirical quantiles by (app, interval, weekday)
  over the trailing year, projected at the fitted growth rate. **This is the
  one to beat.** At a two-fiscal-year horizon most of the signal is "what does
  this application normally do at this time on this weekday, and is it
  growing", and this computes exactly that with no model.

Optional, if installed:

```bash
pip install -e '.[baselines]'    # statsforecast + lightgbm
```

- **`sarima`** — AutoARIMA on the daily peak series. The reviewer's baseline.
  Note it forecasts a peak directly, which is what the three-stage design
  avoids; included because it is the comparison people ask for.
- **`lightgbm`** — quantile objective on the same design matrix Stage 1 uses.
  Most likely to actually win. If it does, the value was in the features, not
  the architecture — that is a finding, not a failure.

Output:

```
model                 pinball_mean  interval_score_90  pit_ks_stat  coverage_abs_error
stage1                       ...
interval_climatology         ...
seasonal_naive               ...

GATE PASSED / GATE FAILED: ...
```

**Read `pinball_mean` first** — it scores the whole predictive distribution, not
just the centre. `coverage_abs_error` says whether the intervals are honest.
`pit_ks_stat` is the sharper test: correct 90% coverage with a wrong middle is
still a wrong distribution, and for something whose upper tail becomes a
purchase order the middle being wrong is not a detail.

---

## Part 4 — The neural model

### 4.1 Decide whether it is worth it

Check the row count from Part 1.2 first. The argument for a global neural model
is that pooling ~35 applications gives enough data to fit one shared parameter
set. At 15-minute grain that is ~945k rows. At hourly it is ~236k, and the
argument is much weaker — `quantile_ridge` and LightGBM are comfortable in that
range and neither needs a large wheel through your proxy.

Do not decide this from the row count alone. Run it, score it, let Part 4.4
decide.

### 4.2 Install

```bash
pip download torch --no-deps -d /tmp/torchcheck   # confirm the mirror has it
pip install -e '.[neural]'                        # torch + neuralforecast
capplan check                                     # should now show torch ok
```

torch is a large wheel. If the internal mirror does not carry it, that is a
procurement conversation, and nothing is blocked meanwhile — `quantile_ridge`
runs the whole pipeline.

### 4.3 Configure and run

`config/capplan.yaml`:

```yaml
model:
  backend: "neuralforecast"
  neuralforecast:
    architecture: "nhits"      # nhits | tft
    input_size_days: 20
    max_steps: 500
    accelerator: "cpu"
```

```bash
capplan train                                    # fit + conformal calibration
capplan evaluate --folds 4 --horizon-days 60     # same folds as Part 3
```

Or without editing the file:

```bash
capplan --set model.backend=neuralforecast evaluate --folds 4
```

N-HiTS is the sensible default: multi-rate hierarchical interpolation, quantile
loss built in, no pretrained weights to download. TFT is worth trying if you
have strong known-future covariates, which here means the calendar block.

### 4.4 Compare properly

Same folds, same horizon, same scoring path:

```bash
capplan --set model.backend=quantile_ridge evaluate --folds 4 --horizon-days 60
capplan --set model.backend=neuralforecast evaluate --folds 4 --horizon-days 60
capplan runs evaluate
```

Each run writes `scores.parquet` and `coverage.parquet` to its artefact
directory, plus a `leaderboard` in the manifest.

Judge on:

1. **`pinball_mean`** — the primary. Whole-distribution, not just the centre.
2. **`interval_score_90`** — cannot be gamed by widening, which coverage alone
   can.
3. **`coverage_abs_error`** — are the stated intervals the real ones.
4. **`pit_ks_stat`** — is the whole distribution right.

**A win needs to be worth its cost.** A 2% pinball improvement does not justify
a torch dependency, a slower batch, and a model nobody in the team can explain
to a capacity manager. A 20% improvement does. Write the threshold down before
you look at the result.

---

## Part 5 — Simulate

```bash
capplan simulate --paths 10000 \
  --reducer annual_max \
  --reducer monthly_peak_r4ha \
  --reducer p95_of_daily_peaks
```

Stages 1-3 end to end: fit, residuals, dependence model, path sampling, sum
across applications interval by interval, then reduce.

### Choosing reducers

```bash
capplan reducers
```

Grouped by the decision they serve, because "which number do you want" is a
question about the decision, not about statistics.

- **Hardware sizing** → `annual_max`. The machine has to survive the moment.
  Most exposed to coincidence and to interval length.
- **Software (MLC) cost** → `monthly_peak_r4ha`. IBM's sub-capacity billing
  takes the highest 4-hour rolling average MSU each day and charges on the
  highest across the month. Not a convention to choose — a definition to match.
- **Trending / chargeback** → `p95_of_daily_peaks`. Estimated from hundreds of
  observations rather than one.

If you need more than one, ask for more than one. The simulation is identical
and only the final reduction differs.

### Dependence model

Default is the residual block bootstrap: resample whole aligned day-blocks (all
applications, all intervals, one historical day), so intra-day shape and
cross-application coincidence come along with no distributional assumption.

Run the comparison too:

```bash
capplan --set simulation.dependence=gaussian_copula simulate --paths 10000
```

Agreement between the two is worth more than either number alone. Disagreement
is a finding for the pack, not something to resolve by preference.

### What to read in the output

- `simulated_coincidence_daily_mean` — must sit close to what `capplan
  diagnostics` measured on history. If it does not, the forward peak is wrong in
  the same direction and everything else is downstream of that.
- `coincidence_interval_peak` vs `coincidence_r4ha` — four-hour averaging
  necessarily brings coincidence closer to 1. If the gap is small at your site,
  the Stage 2 machinery is earning less on a cost deliverable than a hardware
  one.
- `accumulator_mb`, `peak_chunk_working_mb` — the memory ceiling holding.

---

## Part 6 — Backtest the simulation

The part people skip, and the one that decides whether any of the rest is
trustworthy.

```bash
capplan backtest --paths 2000 --reducer annual_max
```

Over prior fiscal years: fit on everything before an origin, simulate the
already-realised window, and check where the realised LPAR peak fell in the
simulated distribution.

Two results, in order:

1. **Mechanism.** Simulated daily coincidence against historical. If this is
   wrong, the forward peak is wrong the same way and nothing below it matters.
2. **Level.** Where the realised figure landed (PIT). High means the simulation
   runs cold and the plan under-provisions; low means it runs hot.

Verdicts report magnitude and direction, never pass/fail. Three years of history
gives a handful of overlapping folds — a direction check, not a hypothesis test.

Anomaly days are excluded from the realised figure, with the anomaly-inclusive
number reported alongside. Comparing a simulation that excludes DR against a
realised maximum *set* by DR gives a spurious "runs cold" verdict.

---

## Part 7 — The pack

```bash
capplan pack
```

One overview page plus one page per application, in Markdown so it renders in
the wiki and diffs between cycles. Answers the six questions a custodian
interview actually asks, in the order they get asked:

1. Where did that come from → the fitted growth rate and how it was derived
2. That is not what I submitted → their submission history and its bias
3. My app peaks at 9am → the observed peak-interval distribution
4. So we need the sum of these? → no, and here is the coincidence factor
5. What about the DR test → excluded, labelled, quantified
6. How do I know this is right → the backtest verdict, verbatim

Everything traces to a run id. `capplan runs` lists them; the registry verifies
artefact hashes.

---

## The minimal loop, once it is set up

```bash
capplan ingest --source db2 --from <start> --to <end>
capplan diagnostics
capplan simulate --paths 10000 --reducer monthly_peak_r4ha
capplan backtest --paths 2000
capplan pack
```

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `Db2Unavailable: pyodbc is not installed` | `pip install -e '.[db2]'`, and the IBM ODBC driver must be on the machine too |
| `missing environment variable CAPPLAN_DB2_HOST` | `.env` not found or not filled in — `capplan sources` shows what loaded |
| `SchemaDriftError: did not return ['app_id']` | alias the column in `config/sources.yaml`, or add a `column_map` entry |
| `DoubleCountError` | a driver/attribution table configured as the `intervals` source — see `capplan tables` |
| `probe`: 1 distinct app_id | app code not populated on the spine, or the join is not joining |
| `NeuralUnavailable` | torch missing; everything runs on `quantile_ridge` meanwhile |
| Coincidence near 1.0 | possibly real — read the `STOP AND RECONSIDER` verdict before continuing |
| Backtest PIT near 1.0 | usually a scope mismatch: something in the realised figure the simulation does not model |
