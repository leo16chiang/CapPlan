-- Coverage and completeness profile. Run before trusting anything else.
SELECT
    app_id,
    lpar,
    COUNT(*) AS n_intervals,
    COUNT(DISTINCT business_date) AS n_days,
    MIN(business_date) AS first_day,
    MAX(business_date) AS last_day,
    SUM(CASE WHEN is_anomaly THEN 1 ELSE 0 END) AS anomaly_intervals,
    SUM(CASE WHEN mips IS NULL THEN 1 ELSE 0 END) AS null_mips,
    SUM(CASE WHEN mips = 0 THEN 1 ELSE 0 END) AS zero_mips,
    AVG(mips) AS mean_mips,
    QUANTILE_CONT(mips, 0.99) AS p99_mips,
    MAX(mips) AS max_mips,
    MAX(mips) / NULLIF(AVG(mips), 0) AS peak_to_mean
FROM intervals
GROUP BY 1, 2
ORDER BY mean_mips DESC;
