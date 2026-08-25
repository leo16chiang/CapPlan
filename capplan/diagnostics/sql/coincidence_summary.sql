-- Coincidence factor summarised per (lpar, fiscal_year).
--
-- The p05 column is the one that matters for sizing: it is the worst-case
-- (closest to additive) day, and a capacity plan that ignores it is planning
-- for the average day.
SELECT
    lpar,
    fiscal_year,
    COUNT(*) AS n_days,
    AVG(coincidence) AS coincidence_mean,
    MEDIAN(coincidence) AS coincidence_median,
    QUANTILE_CONT(coincidence, 0.05) AS coincidence_p05,
    QUANTILE_CONT(coincidence, 0.95) AS coincidence_p95,
    MIN(coincidence) AS coincidence_min,
    MAX(coincidence) AS coincidence_max,
    AVG(overstatement_mips) AS mean_overstatement_mips,
    AVG(distinct_peak_intervals) AS mean_distinct_peak_intervals,
    SUM(CASE WHEN unattributed THEN 1 ELSE 0 END) AS n_unattributed_days
FROM coincidence_daily
GROUP BY 1, 2
ORDER BY 1, 2;
