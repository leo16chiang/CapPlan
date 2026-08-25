-- Per-app bias score across submission cycles.
--
-- The score is the mean log ratio: symmetric in over- and under-forecasting,
-- and directly usable as a multiplicative correction (EXP(-bias_log)).
-- `consistency` counts how often the sign repeated -- a custodian who is
-- reliably 30% high is more useful than one who is randomly +/-30%.
SELECT
    app_id,
    COUNT(*) AS n_cycles,
    AVG(log_ratio) AS bias_log,
    EXP(AVG(log_ratio)) AS bias_ratio,
    STDDEV_SAMP(log_ratio) AS bias_log_sd,
    AVG(ABS(ratio - 1.0)) AS mape,
    SUM(CASE WHEN log_ratio > 0 THEN 1 ELSE 0 END) AS n_over,
    SUM(CASE WHEN log_ratio <= 0 THEN 1 ELSE 0 END) AS n_under,
    GREATEST(
        SUM(CASE WHEN log_ratio > 0 THEN 1 ELSE 0 END),
        SUM(CASE WHEN log_ratio <= 0 THEN 1 ELSE 0 END)
    )::DOUBLE / COUNT(*) AS sign_consistency
FROM submission_bias_daily
GROUP BY 1
ORDER BY ABS(AVG(log_ratio)) DESC;
