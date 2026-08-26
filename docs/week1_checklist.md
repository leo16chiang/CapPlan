# Week 1 checklist

Everything below is cheap to confirm and expensive to be wrong about. Work
down the list before writing more modelling code.

## 1. Confirm the aggregation interval

**Why it comes first.** Everything downstream is indexed on it, and it decides
whether a neural Stage 1 is justified at all.

For an IZPCA / TDSz performance database, the interval is a configured value,
not a property of SMF:

```sql
SELECT * FROM <schema>.MVSPM_TIME_RES;
```

It defaults to one hour and cannot be finer than the SMF interval feeding it.

| Grain | Prime intervals/day | Rows (35 apps x 3 yr) |
|---|---|---|
| 15 min | 36 | 945,000 |
| 30 min | 18 | 472,500 |
| **60 min (default)** | **9** | **236,250** |

At hourly, the ~950k row count that justified a global neural Stage 1 is a
quarter of that. Expect `quantile_ridge` or LightGBM to win the baseline gate;
run `capplan evaluate` and let it decide rather than planning around N-HiTS.

Set `calendar.interval_minutes` to match. `capplan probe` infers the grain from
a sample and fails loudly if it disagrees with the config.

Also note: an hourly figure is an hourly *average*, so the true peak inside the
peak hour is higher. Measure that uplift -- see step 5 -- rather than assuming
it.

## 2. Confirm torch is on the internal mirror

`neuralforecast` pulls torch, which is a large wheel. Confirm it resolves
through the proxy before planning around N-HiTS:

```bash
capplan check
pip download torch --no-deps -d /tmp/torchcheck
```

Nothing is blocked either way: Stage 1 runs on `quantile_ridge` (IRLS on the
pinball loss, pure numpy) and clears the baseline gate on its own. But the
answer changes what gets promised in week 3.

## 3. Point it at the real data and probe before extracting

```bash
capplan sources                                       # driver + env vars
capplan probe --source db2 --from 2025-06-01 --to 2025-06-07
```

`probe` reads a few hundred rows and reports what came back, including the
inferred interval length. It is where step 1 is actually settled, and where a
non-joining application mapping shows up as "1 distinct app_id" rather than as a
forecast for one enormous application. See `docs/data_input.md`.

## 4. Run the two SQL diagnostics -- and be willing to stop

```bash
capplan ingest --source db2 --from 2022-11-01 --to 2025-10-31
capplan diagnostics
```

The coincidence factor is the go/no-go. Near 1.0 means application peaks
effectively do coincide, summing them is nearly right, and a three-stage
simulation is expensive theatre -- spend the time on the marginals. Below ~0.9
means the sum-of-peaks approach is overstating the LPAR by a double-digit
percentage, which is exactly what Stage 2 removes.

The submission-bias table sets the benchmark. If custodians are already
unbiased, the bar is higher than expected.

## 5. Confirm the definitions with the capacity manager

Each of these changes the answer more than any modelling choice, and none of
them is a modelling question:

- **Which decision is the number for?** Hardware sizing and MLC cost are
  different questions with different right answers, and the gap between them is
  usually larger than any modelling choice. `capplan reducers` groups the
  options by decision.
  For software cost the answer is not a choice: IBM bills on the monthly peak
  rolling 4-hour average, so `monthly_peak_r4ha` is the definition to match.
  See `docs/peak_vs_average.md`.
- **Peak-to-hourly uplift.** If the grain is hourly, measure
  `max(interval) / max(hourly average)` from one month of RMF interval data and
  put it in `normalisation.peak_to_hourly_uplift`. It multiplies the headline
  hardware number, so a reviewer will ask where it came from.
- **Prime time, for which target?** IBM's 4-hour window does not stop at 17:00,
  and at many sites the daily peak R4HA is set by the batch window. Prime-time
  scoping is right for the prime-shift hardware question and wrong for cost.
- **Prime time.** 08:00-17:00 confirmed? Local to which timezone? Does the
  mainframe clock match?
- **Fiscal year.** November-October confirmed, and is FY2027 the year *ending*
  October 2027?
- **Capture ratio.** Per-LPAR values, and the direction of the convention. The
  code assumes `mips = msu * mips_per_msu / capture_ratio`, so a ratio below 1
  grosses the figure *up*. If your site's convention is the reciprocal, invert
  it in config.
- **Business-day calendar.** Load the real holiday list into
  `config/holidays.txt`. Nothing else needs to change.

## 6. Confirm the data you need actually exists

- **SMF 70-1 LPAR totals.** Without realised LPAR peaks there is no simulation
  backtest, and without a simulation backtest there is no evidence the
  coincidence model is right. This is the single most important dependency in
  the project and it is easy to discover late.
- **Change calendar / event windows.** DR, IST and GCC SDF windows have to be
  labelled or they get resampled as ordinary variation, and one historical DR
  exercise becomes a recurring feature of the forward distribution.
- **Previous submissions.** Needed for the bias score and for the benchmark.
- **Application-to-LPAR mapping,** and whether it has changed over the history.
  A re-platformed application looks like a step change the model will
  extrapolate.

## 7. Sanity-check what the profile says

```bash
capplan diagnostics
```

Look at `data_profile`:

- `null_mips` vs `zero_mips` -- a missing SMF interval is not a zero, and the
  feature builder needs to know which it is looking at.
- `peak_to_mean` -- if it is near 1 the workload is flat and this whole exercise
  is easy; if it is above 10, the tail is what matters and the simulation is
  earning its keep.
- `anomaly_intervals` -- if this is a large share, the change calendar is
  incomplete or the labelling rule needs tightening.
- `n_unattributed_days` from the coincidence query -- days where the LPAR peak
  exceeds the sum of scoped application peaks. That is workload nobody has
  attributed, and it is a scoping conversation.
