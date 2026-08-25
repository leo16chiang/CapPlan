# Defects this codebase found by measuring its own output

Each of these produced plausible-looking numbers. None would have been caught by
reading the code, and none would have been caught by a test written against the
implementation as it stood. They are recorded because they are the argument for
where the tests are: against realised outcomes, not against intermediate values.

Every one has a regression test.

## 1. `month_end` derived from the block, not the calendar

**Symptom.** A 60-day holdout forecast ran 24% high overall, with a distinct
step: August ran +5%, September +50%, the last two days of September +5%,
October +25%.

**Cause.** `_is_month_end(days, i, n)` decided month-end by counting how many
days after position `i` fell in the same month *within the list it was given*.
For a contiguous training block that is correct. Forecasting proceeds one day at
a time, so the list had length one -- no following days, therefore month-end
true for every forecast day, therefore quarter-end true for every day in
March, June, September and December.

**Fix.** `is_month_end(grid, day, n)` walks the actual business-day calendar.
Overall bias went from 1.24 to 1.02.

**Test.** `tests/test_stage1.py::test_month_end_is_derived_from_the_calendar_not_the_block`

## 2. `years_elapsed` measured from the block's own first day

**Symptom.** With the anchored lag buffer in place, a two-fiscal-year forecast
came back at exactly 1.00x on data growing 14% a year.

**Cause.** `calendar_features` used `days[0]` as the trend epoch. During training
that is the start of history and the feature ranges 0 to 3. At predict time the
block is one day, so `days[0]` is that day and the feature is identically zero.
The trend was switched off precisely when it was the only thing carrying the
forecast.

**Fix.** The epoch is pinned in `FeatureSpec.origin` at fit time and travels
with the artefacts. Growth is additionally made explicit per app, because a
ridge coefficient on a lag feature is not something a custodian can argue with
and an annual percentage is.

**Test.** `tests/test_stage1.py::test_trend_origin_is_pinned_at_fit_time`

## 3. Recursive multi-step forecasting is explosive at this horizon

**Symptom.** Feeding the model's own median forward for 512 business days gave a
3.2x increase over two fiscal years on data whose true growth is 3% a year.

**Cause.** The lag features are near-collinear; ridge leaves their coefficients
summing above one; 512 steps of that compounds.

**Fix.** The lag buffer is recursive for the first ~20 days and
climatology-anchored thereafter, blended between. At 500 days out "what does
this application typically do at 10:15" is not a fallback, it is the correct lag
input -- the recursive alternative is the model's own error fed back 500 times.

**Test.** `tests/test_stage1.py::test_two_year_horizon_neither_explodes_nor_flatlines`

## 4. Burn-in rows given predictions from zero-filled features

**Symptom.** Simulated annual peak of 116,785 MIPS against a historical realised
peak of about 5,000. Residual standard deviation of 13.5 against a 1st/99th
percentile of -2.0/+3.0.

**Cause.** `fitted_quantiles` called `np.nan_to_num(X)` before predicting, so the
lag/rolling burn-in rows at the start of history got predictions built from
zeros. Those predictions had near-degenerate spreads. Stage 2 *divides* by that
spread, so a handful of cells produced standardised residuals in the hundreds --
and because the bootstrap resamples whole days, every path drawing one of those
days inherited it.

**Fix.** Rows the design marked invalid return NaN. Ask for a prediction on
features that do not exist and you get NaN.

**Test.** `tests/test_stage1.py::test_fitted_quantiles_return_nan_on_burn_in_rows`

## 5. An absolute floor on a quantity that scales with the application

**Symptom.** Same as #4, and it survived the fix to #4.

**Cause.** `ForecastCube.spread` floored the interquantile width at `1e-6`. For
a 2,000-MIPS application that is not a floor, it is a formality. One constant
cannot serve both a 20-MIPS application and a 2,000-MIPS one.

**Fix.** The floor is relative: `max(0.02 * median, 0.1)`. Winsorisation at
+/-8 half-widths remains as a *reported* backstop -- a large winsorisation count
means something upstream is wrong and should be fixed rather than clipped.

**Test.** `tests/test_stage1.py::test_spread_floor_is_relative_to_the_level`

## 6. A lazily-drawn sampler that was not lazy

**Symptom.** None visible -- the simulation ran. But `np.stack([sampler.draw(...)
for _ in range(size)])` materialised `size x days x apps x intervals`, which is
2.6 GB at the default settings, defeating the point of chunking.

**Fix.** Samplers expose `stream(size, n_days, rng)` returning an object with
`day(d)`. The bootstrap stores only day indices (a few MB) and resolves one
day's residual slab at a time. Measured peak allocation for 10,000 paths x 512
days: 116 MB.

**Test.** `tests/test_simulation.py::test_simulation_stays_within_its_memory_budget`

## 7. The backtest comparing against a figure the simulation does not model

**Symptom.** PIT 1.0 on every fold. Realised annual maximum around 10,000 MIPS
against a simulated median around 5,900. Read literally: the simulation runs
catastrophically cold.

**Cause.** SMF 70-1 contains the DR exercise. DR, IST and GCC SDF are out of
scope as forecast targets and the simulation correctly excludes them -- so the
comparison was between a simulation without DR and a realised maximum that was
*set* by DR.

**Fix.** The realised figure is taken on anomaly-free days. The
anomaly-inclusive figure is reported alongside it rather than dropped: 59%
higher, out of scope to forecast, and still something the hardware has to
survive. Like-for-like, the residual gap is -5%, in the direction a block
bootstrap is expected to err.

**Test.** covered by `tests/test_coincidence.py::test_unattributed_load_is_flagged_not_hidden`
and the backtest's own reporting.

## 8. IRLS converging on the median and not on the tails

**Symptom.** Nominal 0.10 quantile covering 0.164; nominal 0.90 covering 0.776.

**Cause.** Fixed-epsilon IRLS on the pinball loss converges quickly near the
median and slowly in the tails. Twelve iterations was ample for the median and
nowhere near enough for the 0.9.

**Fix.** Epsilon is annealed geometrically from 0.5 down to 1e-5, reaching exact
empirical coverage on all nine quantiles in about a dozen iterations rather than
fifty.

**Test.** `tests/test_stage1.py::test_quantile_ridge_achieves_nominal_coverage`

## 9. Symmetric CQR cannot correct a location bias

**Symptom.** After conformal calibration, holdout coverage error improved from
0.255 to 0.127 -- better, but nowhere near calibrated. The median quantile was
still covering 0.89 against a nominal 0.5.

**Cause.** Classic CQR scores an *interval* with `max(q_lo - y, y - q_hi)` and
widens symmetrically. An interval that is correctly sized but centred too high
stays centred too high.

**Fix.** A per-quantile one-sided variant is the default: score `y - q_tau`,
correct by the tau-th conformal quantile of it. Corrects location and width.
Coverage error 0.0295 to 0.0007. The symmetric version is retained for
comparison, with a test that documents exactly what it cannot do.

**Test.** `tests/test_pipeline_parts.py::test_symmetric_cqr_widens_but_cannot_move_the_centre`
