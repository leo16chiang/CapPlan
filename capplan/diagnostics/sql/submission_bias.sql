-- Per-app submission bias: what the custodian said vs what actually happened.
--
-- Also pure SQL over data already held. Two uses:
--   1. It is the honest baseline. A forecast that cannot beat "last year's
--      submission, de-biased" is not worth deploying.
--   2. It is the opening of every custodian interview. "Your last three
--      submissions ran 34% high" changes the conversation.
--
-- ratio > 1 means the custodian over-forecast.
WITH realised AS (
    SELECT
        app_id,
        fiscal_year,
        MAX(mips) AS realised_peak_mips
    FROM intervals
    WHERE NOT is_anomaly
    GROUP BY 1, 2
)
SELECT
    s.app_id,
    s.fiscal_year,
    s.submitted_peak_mips,
    r.realised_peak_mips,
    s.submitted_peak_mips / NULLIF(r.realised_peak_mips, 0) AS ratio,
    s.submitted_peak_mips - r.realised_peak_mips AS error_mips,
    LN(s.submitted_peak_mips / NULLIF(r.realised_peak_mips, 0)) AS log_ratio,
    s.basis
FROM submissions s
JOIN realised r
  ON r.app_id = s.app_id
 AND r.fiscal_year = s.fiscal_year
ORDER BY s.app_id, s.fiscal_year;
