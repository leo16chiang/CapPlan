# Peak vs average, revisited

You said you were not sure when you first answered this. Having looked at the
tables, the honest position is that the original question was malformed, and
not in a way you could have known.

The original framing offered three candidates -- annual max, mean of monthly
peaks, percentile of daily peaks -- and assumed all three were reductions of an
interval peak series. **The most important target on z/OS is not a peak at all,
and its definition is not yours to choose.**

## Three targets, three decisions

| | What it is | Decides | Sensitive to |
|---|---|---|---|
| **Interval peak** | Highest single interval | Hardware sizing | Coincidence, and interval length |
| **R4HA** | Highest rolling 4-hour average, per month | Software (MLC) cost | Much less: 4h averaging smooths timing |
| **Sustained level** | Percentiles / means of daily peaks | Trending, chargeback | Least |

### Interval peak

The machine has to survive the moment, not the hour. Most exposed to
coincidence, and — critically for your data — most exposed to interval length.

### R4HA — the one with an external definition

IBM's sub-capacity MLC billing works like this: every five minutes, the rolling
average of the last four hours of MSU is computed. For each day, the highest
such value is taken. The highest across all days in the month sets that month's
MLC charge for the product. The invoice is on the lower of that and any
defined/group capacity cap.

This matters for three reasons:

1. **It is not a convention you get to pick.** If the deliverable is software
   cost, matching this definition is the only correct answer, and any "peak"
   that is not an R4HA is answering a different question.
2. **Four-hour averaging smooths the timing differences that make peaks fail to
   sum.** Two applications peaking two hours apart contribute to the same
   4-hour window, so the R4HA coincidence factor is necessarily closer to 1
   than the interval-peak one -- it is an arithmetic consequence of averaging,
   not an empirical claim. How much closer is site-specific, and every
   `capplan simulate` run reports both. If the gap is small at your site, the
   Stage 2 dependence apparatus earns much less on an MLC deliverable than on
   a hardware one, which is worth knowing before spending a quarter on it.
3. **Your hourly data computes it well.** A 4-hour rolling mean of hourly MSU is
   a close approximation of the true 5-minute-stepped R4HA — closer than an
   hourly interval peak is to a true interval peak, because averaging is
   forgiving where maxima are not.

`capplan reducers` now groups by decision, and `monthly_peak_r4ha` /
`annual_peak_r4ha` are first-class.

### Sustained level

Percentiles and means of daily peaks. Estimated from hundreds of observations
rather than one, so far more stable. Right for trending and chargeback, wrong
for sizing.

## What your tables allow

Confirmed against the IZPCA/TDSz naming convention (`_H` hourly, `_D` daily,
`_M` monthly; trailing `V` = view; names clipped to 18 chars by the old Db2 for
z/OS limit):

| Your table | Grain | Role |
|---|---|---|
| `MVSPM_WORKLOAD2_HV` | **hour** | **spine** |
| `WAS_INT_SERVLETS_H` | hour | driver |
| `CICS_TRANSACTIO_DP` | day | driver |
| `IMS_SYSTEM_TRAN2_D` | day | driver |
| `KPMZ_JOB_INT_D` | day | driver |
| `MVS_ADDRSPACE_D` | day | attribution |
| `MVS_ADDRDIS_ACCT_M` | month | attribution |
| `CAP_GRP_MIPS_D` | day | validation |

Three consequences.

**Hourly is your floor.** TDSz's aggregation interval is set in
`MVSPM_TIME_RES`, defaults to one hour, and cannot be finer than the SMF
interval feeding it. Check it:

```sql
SELECT * FROM <schema>.MVSPM_TIME_RES;
```

At one hour, prime time is **9 intervals a day, not 36**.

**An hourly figure is an hourly average.** The true peak inside the peak hour is
higher, by a factor that is a property of your workload and must be measured
rather than assumed. Two ways:

- Best: pull one month of RMF interval data (SMF 70-1 or 72 at the SMF
  interval, typically 15 min) and compute
  `max(interval) / max(hourly average)` per day.
- Cheaper: temporarily lower `MVSPM_TIME_RES`, collect a month, compare.

Typical z/OS values run 1.1–1.4, but a number you measured beats a number you
read. Record it in `normalisation.peak_to_hourly_uplift` and state it in the
pack — it is a multiplier on the headline hardware number and a reviewer will
ask where it came from. It does **not** apply to R4HA, which is an average of
averages and needs no uplift.

**The row count falls, and with it the neural argument.**

| Grain | Intervals/day | Rows (35 apps × 3 yr) |
|---|---|---|
| 15 min | 36 | 945,000 |
| 30 min | 18 | 472,500 |
| **60 min** | **9** | **236,250** |

I flagged in week 1 that the ~950k figure was what justified a global neural
Stage 1 at all. At hourly grain that condition has resolved against it. 236k
rows across 35 series is comfortably in `quantile_ridge` and LightGBM territory,
and `neuralforecast` would be a large wheel and a proxy conversation for a model
that is unlikely to clear the baseline gate. Run `capplan evaluate` and let it
decide, but do not plan around N-HiTS.

## The double-counting trap

The most expensive available mistake, and it produces a plausible number rather
than an obviously wrong one.

A CICS transaction's CPU appears in `CICS_TRANSACTIO_DP`, **and** in the CICS
region's service class in `MVSPM_WORKLOAD2_HV`, **and** in the region's address
space in `MVS_ADDRSPACE_D`. These are three views of one consumption, not three
components of it. Summing MIPS across them roughly doubles the answer.

So: exactly one **spine** (`MVSPM_WORKLOAD2_HV` — WLM classifies every unit of
dispatchable work exactly once, which is the property nothing else has), and
everything else is a **driver** (volume, attribution, leading indicator) or
**validation**. `capplan tables` prints the catalogue; the ingest path raises
`DoubleCountError` rather than letting a driver feed `intervals`.

Your bottom-up app code makes this easier, not harder: because the app code is
already on the spine rows, you do not need the driver tables for attribution at
all. Their value is as growth predictors and as a cross-check on the spine's
intra-day shape.

## What changes about coincidence

The go/no-go diagnostic still works, because the spine is hourly. But it now
measures hourly coincidence, which is **less severe than 15-minute
coincidence** — two applications peaking 20 minutes apart look simultaneous at
hourly grain. So:

- Your measured coincidence factor is an **upper bound** on the true
  15-minute one. The real overstatement from summing app peaks is larger than
  the hourly number says.
- That is a conservative error in the useful direction for the go/no-go
  decision: if hourly data already says peaks do not sum, finer data would say
  it more strongly. If hourly says they do sum, that is genuinely inconclusive
  and needs a finer sample before you conclude anything.

`capplan simulate` reports both `coincidence_interval_peak` and
`coincidence_r4ha`, so the gap between them is visible in every run.

## Recommendation

1. Check `MVSPM_TIME_RES`. If hourly, set `calendar.interval_minutes: 60`.
2. **Ask which decision the number is for.** Hardware refresh and MLC forecast
   are different questions with different right answers. If both, produce both
   — the simulation is the same and only the reduction differs.
3. Default to `monthly_peak_r4ha` for cost and `annual_max` (uplifted) for
   hardware. Put both in the pack with the family label.
4. Measure the peak-to-hourly uplift once, from a month of RMF interval data.
   Do not carry an assumed 1.25 into a board pack.
5. Do not restrict to prime time for the R4HA target. IBM's window does not stop
   at 17:00, and at many sites the daily peak R4HA is set by the batch window.
   Prime time is right for the prime-shift hardware question and wrong for cost.
6. Let `capplan evaluate` decide on the neural backend. At 236k rows expect the
   baseline gate to be close, and treat that as a saved quarter rather than a
   disappointment.
