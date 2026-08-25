-- Historical sum-of-app-peaks vs realised LPAR peak.
--
-- This is the whole argument for the architecture, and it is answerable today
-- with data already on disk. If the coincidence factor comes back at 0.98,
-- peaks effectively do sum at this site, and a three-stage simulation is
-- expensive theatre. If it comes back at 0.75, every submission built by
-- adding up app peaks is overstating the LPAR by a third.
--
-- sum_app_peaks : SUM over apps of that app's own daily prime-time maximum
--                 (each app free to peak in a different interval)
-- lpar_peak     : the realised LPAR maximum from SMF 70-1, measured at the
--                 LPAR, so it already contains the true coincidence
-- coincidence   : lpar_peak / sum_app_peaks, normally in (0, 1]
--
-- A coincidence above 1 is not a rounding artefact and must not be clipped
-- away. It means the LPAR carried prime-time load that the scoped app rows do
-- not account for -- typically an anomaly day excluded from the app side but
-- still present in SMF 70-1, or workload outside the top-N apps. Either way it
-- is a data-scoping finding, so it is flagged (`unattributed`) and reported
-- rather than silently absorbed.
--
-- Parameters: $exclude_anomalies (BOOLEAN)
WITH app_daily_peak AS (
    SELECT
        business_date,
        fiscal_year,
        lpar,
        app_id,
        MAX(mips) AS app_peak_mips,
        ARG_MAX(interval_idx, mips) AS app_peak_interval
    FROM intervals
    WHERE (NOT $exclude_anomalies OR NOT is_anomaly)
    GROUP BY 1, 2, 3, 4
),
summed AS (
    SELECT
        business_date,
        fiscal_year,
        lpar,
        SUM(app_peak_mips) AS sum_app_peaks,
        COUNT(*) AS n_apps,
        COUNT(DISTINCT app_peak_interval) AS distinct_peak_intervals
    FROM app_daily_peak
    GROUP BY 1, 2, 3
)
SELECT
    s.business_date,
    s.fiscal_year,
    s.lpar,
    s.n_apps,
    s.distinct_peak_intervals,
    s.sum_app_peaks,
    t.peak_mips AS lpar_peak,
    t.peak_interval_idx AS lpar_peak_interval,
    t.peak_mips / NULLIF(s.sum_app_peaks, 0) AS coincidence,
    s.sum_app_peaks - t.peak_mips AS overstatement_mips,
    t.peak_mips > s.sum_app_peaks AS unattributed
FROM summed s
JOIN lpar_totals t
  ON t.business_date = s.business_date
 AND t.lpar = s.lpar
ORDER BY s.lpar, s.business_date;
