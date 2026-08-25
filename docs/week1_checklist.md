# Week 1 checklist

Everything below is cheap to confirm and expensive to be wrong about. Work
down the list before writing more modelling code.

## 1. Confirm the SMF interval is 15 minutes

**Why it comes first.** Everything downstream assumes it. Fifteen minutes gives
36 prime-time intervals per business day, and across ~35 applications and ~250
business days a year that is roughly 950,000 prime-time rows -- which is the
data volume that justifies a global neural Stage 1 at all. At 30-minute
intervals it halves; at hourly it is 260,000 and the argument for a neural net
gets much weaker.

```sql
SELECT DISTINCT date_diff('minute', LAG(ts) OVER (PARTITION BY app_id ORDER BY ts), ts)
FROM intervals LIMIT 20;
```

If it is not 15, change `calendar.interval_minutes` in `config/capplan.yaml`.
Only `capplan/data/calendar.py` cares -- but revisit the backend choice.

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

## 3. Run the two SQL diagnostics -- and be willing to stop

```bash
capplan ingest            # point at the real extract
capplan diagnostics
```

The coincidence factor is the go/no-go. Near 1.0 means application peaks
effectively do coincide, summing them is nearly right, and a three-stage
simulation is expensive theatre -- spend the time on the marginals. Below ~0.9
means the sum-of-peaks approach is overstating the LPAR by a double-digit
percentage, which is exactly what Stage 2 removes.

The submission-bias table sets the benchmark. If custodians are already
unbiased, the bar is higher than expected.

## 4. Confirm the definitions with the capacity manager

Each of these changes the answer more than any modelling choice, and none of
them is a modelling question:

- **Which reducer?** Annual max, mean of monthly peaks, or a percentile of daily
  peaks? On the synthetic panel these span 21%. Run `capplan reducers`, put the
  list in front of them, get one written down.
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

## 5. Confirm the data you need actually exists

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

## 6. Sanity-check what the profile says

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
