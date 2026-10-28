-- Runs inside PostgreSQL. Per account, ordered by month: what the model learns from, the value to
-- predict (next month), and the role of each row.
WITH w AS (
    SELECT "account_id",
           "month",
           "net",
           AVG("net") OVER (PARTITION BY "account_id" ORDER BY "month"
                            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)       AS "net_avg_3m",
           LAG("net", 11) OVER (PARTITION BY "account_id" ORDER BY "month") AS "net_last_year",  -- next month, 1 year ago
           LEAD("net", 1) OVER (PARTITION BY "account_id" ORDER BY "month") AS "net_next_month"  -- to predict
    FROM "${projectKey}_monthly_net"
)
SELECT w.*,
       CASE WHEN "net_next_month" IS NULL THEN 'forecast'   -- latest month: not known yet
            WHEN "net_last_year" IS NULL THEN 'warm_up'     -- first year: no last-year value
            WHEN "month" >= DATE '2026-06-01' THEN 'test'   -- last 3 known months test the model
            ELSE 'train'
       END AS "role"
FROM w
