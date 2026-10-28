-- Runs inside PostgreSQL: today's balance + the 3-month forecast => projected balances and a risk level.
--   HIGH   : the expected path goes beyond the overdraft limit
--   MEDIUM : the expected path dips below zero, or the pessimistic path (10% band) breaks the limit
--   LOW    : otherwise
WITH hist AS (
    SELECT "account_id",
           SUM("net_cashflow") AS "hist_net",
           SUM(CASE WHEN "month" > (SELECT MAX("month") FROM "${projectKey}_monthly_cashflow") - INTERVAL '3 months'
                    THEN "net_cashflow" ELSE 0 END) AS "last_3m_net"
    FROM "${projectKey}_monthly_cashflow"
    GROUP BY "account_id"
),
fc AS (
    SELECT "account_id", "inflow_forecast", "outflow_forecast", "net_forecast",
           SUM("net_forecast") OVER (PARTITION BY "account_id" ORDER BY "forecast_month") AS "cum_net",
           SUM("net_lower")    OVER (PARTITION BY "account_id" ORDER BY "forecast_month") AS "cum_low"
    FROM "${projectKey}_cashflow_forecast"
),
fca AS (
    SELECT "account_id",
           SUM("inflow_forecast")  AS "forecast_3m_inflow",
           SUM("outflow_forecast") AS "forecast_3m_outflow",
           SUM("net_forecast")     AS "forecast_3m_net",
           LEAST(MIN("cum_net"), 0) AS "min_cum_net",
           LEAST(MIN("cum_low"), 0) AS "min_cum_low"
    FROM fc
    GROUP BY "account_id"
),
b AS (
    SELECT a."account_id", a."company_name", a."sector", a."segment", a."relationship_manager",
           a."overdraft_limit",
           a."opening_balance" + h."hist_net" AS "current_balance",
           h."last_3m_net",
           f."forecast_3m_inflow", f."forecast_3m_outflow", f."forecast_3m_net",
           f."min_cum_net", f."min_cum_low"
    FROM "${projectKey}_accounts" a
    JOIN hist h ON h."account_id" = a."account_id"
    JOIN fca  f ON f."account_id" = a."account_id"
)
SELECT "account_id", "company_name", "sector", "segment", "relationship_manager", "overdraft_limit",
       "current_balance", "last_3m_net",
       "forecast_3m_inflow", "forecast_3m_outflow", "forecast_3m_net",
       "current_balance" + "forecast_3m_net" AS "projected_balance",
       "current_balance" + "min_cum_net"     AS "projected_min_balance",
       "current_balance" + "min_cum_low"     AS "worst_case_min_balance",
       CASE WHEN "current_balance" + "min_cum_net" < -"overdraft_limit" THEN 'HIGH'
            WHEN "current_balance" + "min_cum_net" < 0
              OR "current_balance" + "min_cum_low" < -"overdraft_limit" THEN 'MEDIUM'
            ELSE 'LOW' END AS "risk_level"
FROM b
