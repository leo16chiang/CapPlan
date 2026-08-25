-- Where each app's daily peak lands inside the prime window.
--
-- The mechanism behind the coincidence factor: if every app peaked at 10:15
-- the factor would be ~1. This is the chart that explains the whole approach
-- to a custodian in one picture.
SELECT
    app_id,
    interval_idx,
    COUNT(*) AS n_days_peaking_here,
    COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER (PARTITION BY app_id) AS share
FROM (
    SELECT
        app_id,
        business_date,
        ARG_MAX(interval_idx, mips) AS interval_idx
    FROM intervals
    WHERE NOT is_anomaly
    GROUP BY 1, 2
)
GROUP BY 1, 2
ORDER BY 1, 2;
