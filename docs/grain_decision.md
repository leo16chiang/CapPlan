# Should daily be the baseline?

Short answer: **no** — but the reasoning matters more than the answer, because
there is a real fallback if the hourly spine turns out not to work.

## Counting tables is the wrong metric

Seven of your eight tables are daily or monthly, so "most of my data is daily"
is true. It is also misleading, because those seven are not seven sources of
CPU. They are seven *views* of the one source:

```
MVSPM_WORKLOAD2_HV   ← all CPU, classified once by WLM, hourly
   ├── CICS_TRANSACTIO_DP    the CICS regions' share, cut by transaction
   ├── IMS_SYSTEM_TRAN2_D    the IMS regions' share, cut by transaction
   ├── WAS_INT_SERVLETS_H    the WAS servers' share, cut by servlet
   ├── KPMZ_JOB_INT_D        the batch initiators' share, cut by job
   ├── MVS_ADDRSPACE_D       all of it again, cut by address space
   └── MVS_ADDRDIS_ACCT_M    all of it again, cut by account code
```

The question is not "how many tables are daily" but "is there one complete,
non-overlapping source with intra-day timing". There is exactly one, and it is
hourly. Choosing daily as the baseline means discarding your only complete
source in favour of incomplete overlapping ones.

## What daily grain actually costs

**The deliverable becomes uncomputable.** With one number per application per
day, "peak of the sum" collapses into "sum of the peaks" — they are the same
arithmetic when there is only one interval. That is precisely the overstatement
the whole three-stage design exists to remove, and it comes back in full.

**Coincidence stops being measurable.** It is defined by *when within the day*
each application peaks. A daily maximum does not say which hour it landed in,
so two daily maxima are indistinguishable from two simultaneous ones.
Coincidence is not hard to measure from daily data — it is absent from it.

**The simulation backtest goes with it.** No coincidence measurement means no
way to check whether the forward reconstruction is right. You would be
forecasting the one thing you cannot validate.

## What daily grain is genuinely good for

Not nothing — three real uses, and the design uses all three:

**Growth estimation.** Daily series are less noisy than hourly and often go
back further. If `CAP_GRP_MIPS_D` has five years and the hourly spine has
eighteen months, fit the trend on the daily series and the shape on the hourly
one.

**Coverage.** If some applications appear in the daily tables but have no app
code on the spine, the daily tables extend the application list. Better a
daily-only application flagged as such than a missing one.

**Leading indicators.** `CICS_TRANSACTIO_DP` and `KPMZ_JOB_INT_D` give
transaction and job volumes per application. Volume growth precedes MIPS
growth, and a custodian can forecast transaction counts far more confidently
than MIPS. "Your CICS volume is growing 8% a year and your MIPS 6%" is a much
better interview opener than a MIPS number alone.

## The fallback, if the hourly spine does not work out

It might not. `MVSPM_WORKLOAD2_HV` is a view, and views filter; the app code
may not be populated on it; the site may not retain it for long. If so, daily
forecasting is workable — with one addition.

**Transfer the coincidence factor from a short sub-daily sample.** The factor is
a *ratio*, and ratios transfer across periods far better than levels do. You do
not need years of hourly data to estimate it; you need enough days to pin down
its distribution.

```bash
capplan coincidence --sample-days 63     # one quarter is usually plenty
capplan simulate                          # daily-grain runs apply it automatically
```

`capplan coincidence` estimates the factor, bootstraps its mean, and reports the
interval width as a percentage. Under ~2% the transfer adds little beside the
forecast's own uncertainty; over ~5% it dominates, and more hourly sample is the
cheapest available improvement. Below 20 days it declines to give a verdict at
all, because there the factor's own variation is indistinguishable from sampling
noise.

How much sample you need depends on how variable your coincidence factor is,
which is a property of your workload. Run it and read the verdict.

The factor is resampled per (path, day) rather than applied as a constant —
collapsing it to its mean would understate the forecast's spread by exactly the
factor's own day-to-day variation.

### What the fallback assumes, stated plainly

- The factor is **stable** between the sample window and the forecast horizon.
  If the application mix changes materially, it is not.
- Day-to-day **variation** in the factor is captured; **seasonality** in it is
  not, unless the sample spans the seasons.
- An hourly sample gives an **upper bound** on the true factor. Two
  applications peaking twenty minutes apart look simultaneous at hourly grain,
  so the real overstatement is larger than an hourly measurement says.
  `capplan coincidence` logs this every time.

Worse than measuring it directly. Much better than pretending peaks sum.

## Recommendation

1. **Try the hourly spine first.** `capplan probe` will tell you within minutes
   whether `MVSPM_WORKLOAD2_HV` has a populated app code and a sane interval.
2. **If it works, use it** — and use the daily tables for growth, coverage and
   volume drivers, never as addends.
3. **If it does not**, run daily with a transferred factor, and put the
   assumption in the pack rather than in a footnote.
4. **Either way, get some hourly data.** Even one quarter is enough to pin the
   factor down to ~1.5%, and it is the cheapest single improvement available to
   this forecast.
