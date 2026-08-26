# Architecture

Why each stage is shaped the way it is, and what each one refuses to do.

## The problem in one line

Forecast production prime-time peak MIPS per application two fiscal years out,
defensibly. Two hard parts, and they pull in different directions.

**A peak is a maximum.** Thin data, heavy tail. Mean-based losses are biased for
a maximum, and so, more subtly, is a quantile forecast of the *daily peak
series* -- you have thrown away 35 of every 36 observations to get one, and the
one you kept is the noisiest.

**Peaks do not sum.** Applications peak in different intervals. Adding their
individual peaks assumes simultaneity that does not exist. Measured on the
a workload where applications peak at genuinely different times, summing
overstates the LPAR substantially. How much is site-specific and is exactly
what `capplan diagnostics` measures.

## The resolution: never forecast a peak

```
Stage 1   marginal distribution per (app, future 15-min interval)
Stage 2   the joint structure -- across time and across applications
Stage 3   sample paths, sum across apps interval by interval, THEN reduce
```

The peak is a property of the simulated sum. Both hard parts dissolve: Stage 1
forecasts ~950,000 ordinary interval observations rather than ~250 extreme ones,
and the maximum is taken after the sum rather than before it.

## Stage 1 -- the interval forecaster

**Global, not per-application.** One parameter set across all ~35 applications,
with per-application scale handled by robust normalisation. Per-application
models on three years of data are 35 separate thin-data problems; the entire
reason there is enough data for a neural net is that the applications are
pooled.

**Backends are pluggable, and one of them has no dependencies.**
`neuralforecast` (N-HiTS, TFT) is the intended backend -- quantile loss built
in, no pretrained weights to download. But torch is a large wheel and its
availability is a week-1 unknown, so `quantile_ridge` (IRLS on the pinball loss,
pure numpy) exists to make the pipeline runnable, testable and reviewable on day
one, and to give the neural model something honest to beat. It clears the
baseline gate on its own.

**The two-fiscal-year horizon is the hard part, not the model.** Deterministic
calendar features -- interval-of-day, day-of-week, month-end, quarter-end,
fiscal-year position, damped elapsed years -- carry the forecast, because they
are the only features that mean anything 512 business days out. The lag and
rolling block is recursive for ~20 days and climatology-anchored beyond, because
pure recursion at this horizon is explosive (see `defects_found.md` #3) and a
flat anchor kills the trend (#2). Growth is therefore explicit: a per-application
annual rate fitted on daily medians, which is also the single number a custodian
can meaningfully agree or disagree with.

**Calibration is split-conformal**, defaulting to a per-quantile one-sided
variant rather than classic CQR, because a long horizon carries a location bias
and symmetric widening cannot fix one (#9). This matters more here than in
ordinary forecasting: Stage 3 turns the marginals into a distribution over a
maximum, and miscalibration accumulates exactly in the tail that gets read off
and turned into a purchase order.

**What Stage 1 refuses to do:** produce a peak, or know anything about other
applications.

## Stage 2 -- the dependence model

Marginals are not enough. A peak of a sum depends entirely on whether the
applications are high at the same moment, and the marginals contain no
information about that at all.

**Primary: residual block bootstrap.** Compute residuals against the Stage 1
median, keep them as aligned day-blocks -- all applications, all 36 intervals,
the same historical day -- and resample whole blocks. Intra-day shape and
cross-application coincidence come along for free, with no distributional
assumption and about 80 lines of numpy.

The reason it is primary is not statistical elegance. It is that "days like this
have happened; here is what they looked like" survives an interview, and a
1,260-dimensional Gaussian copula does not.

Its honest limitation: it can only produce coincidence patterns that have
occurred. With ~750 usable days that is a reasonable pool, but it cannot invent
a pattern history has never shown, and it under-samples the extreme tail. The
backtest measures exactly that: -5% on the annual maximum.

**Comparison: Gaussian copula on PIT-transformed residuals.** More principled --
it *can* generate unseen patterns, which for a tail quantity two years out is a
real advantage. Less defensible -- it assumes Gaussian dependence, which means
zero tail dependence, which means coincident extremes are systematically
understated, which is the failure that matters most here. `tail_dependence_check`
measures the gap rather than asserting it away.

Run both. Agreement is worth more than either number alone; disagreement is a
finding for the pack, not something to resolve by preference.

**Residual scaling.** Residuals are divided by half the predicted (q10, q90)
width, so they are comparable across applications, intervals and levels -- which
is what makes it legitimate to resample a residual from last March and add it to
a forecast for two Novembers hence. The floor on that divisor is *relative*, for
reasons documented painfully in `defects_found.md` #5.

**Anomaly handling.** A day enters the pool only if it is clean across *every*
application. Dropping the whole day is the price of preserving cross-application
coincidence: a day with one application missing cannot be resampled as a
coherent block.

## Stage 3 -- simulate and reduce

```
draw residuals -> add to marginals -> SUM ACROSS APPS -> take the maximum
```

**Memory is architectural.** `paths x days x intervals x apps` is 736 GB at
10,000 x 512 x 36 x 35. So: chunk over paths; samplers expose a lazy per-day
stream rather than resolving a chunk eagerly; the app axis exists only for one
`(chunk, day)` slice; a running maximum per `(path, app)` carries the
application-level numbers with no application history; only reduced
`(paths, days)` series survive. Measured peak allocation: 116 MB.

**The reducer is pluggable because the question is open.** Annual maximum, mean
of monthly peaks, 95th percentile of daily peaks, monthly peak R4HA -- and they
answer different questions, so the gaps between them are typically larger than
the modelling uncertainty within any one. `reduce(path) -> scalar` is the seam;
when the definition is settled, one function is added and nothing else
changes.

The contract is a `PathSummary` (daily peak series, daily means, dates) rather
than the raw interval cube, because handing a reducer the raw cube would require
materialising the thing the whole design avoids. Every convention anyone has
proposed is expressible from the daily peak series; a reducer that genuinely is
not declares `needs_intervals`.

**`daily_sum_app_peaks` is retained deliberately.** It costs 20 MB and it lets
the simulation report its own coincidence factor, directly comparable with the
historical SQL. It is the single most useful validation number a run produces.

## Evaluation

Two separate questions, and conflating them is the standard mistake.

**Does Stage 1 work?** `eval/rolling_origin.py`. Pinball loss, coverage, PIT,
interval score, against seasonal naive and interval climatology through the same
code path. Stage 1 has to beat them or the honest recommendation is to ship the
baseline and spend the time on the coincidence model, which is where the larger
error lives.

**Does the simulation work?** `eval/sim_backtest.py`. This is the part people
skip, and it is the one that decides whether any of the rest is trustworthy.
Over prior fiscal years, does the realised LPAR peak from SMF 70-1 fall where
the simulated distribution said it would?

The comparison is against the realised LPAR figure, not a reconstruction from
application rows -- the LPAR figure is measured at the LPAR, so it already
contains the true coincidence. Reconstructing it would test the simulation
against its own assumption.

Two things come out, in order:

1. **Mechanism.** Simulated daily coincidence against historical. If this is
   wrong, the forward peak is wrong in the same direction and nothing below it
   matters. Currently reproduced to 0.0008.
2. **Level.** Where the realised figure falls in the simulated distribution.
   Currently PIT 0.87, median 5% low, in the direction a block bootstrap is
   expected to err, with the copula agreeing at 5.3%.

Verdicts report magnitude and direction, never pass/fail. Three years of history
gives a handful of overlapping folds; that is a direction check, not a
hypothesis test, and saying so is part of the result.

## Serving

No new front end. A scheduled batch writes three small parquet tables that the
existing Dash app reads: `fy_summary` (tens of rows), `app_peaks`, and
`daily_profile`. Paths are deliberately **not** published -- 20 MB of float no
dashboard can render, and publishing it invites someone to re-derive a peak with
the wrong reduction.

`app_peaks` carries a `do_not_sum` column and a note, because someone will try.

## The pack

The deliverable is not a number. It is a number that survives an interview with
the person who owns the application, and that interview goes badly in
predictable ways. `serve/pack_gen.py` answers the six predictable questions per
application, in the order they get asked, before anyone asks them -- where the
growth rate came from, what their previous submissions did, when their
application actually peaks, why the parts do not sum to the whole, what was
excluded, and what evidence there is that any of it is right.

Markdown, because it renders in the wiki the capacity team already uses and
diffs cleanly between cycles. Last cycle's pack next to this one, with the
changed numbers visible, is worth more than any chart.
