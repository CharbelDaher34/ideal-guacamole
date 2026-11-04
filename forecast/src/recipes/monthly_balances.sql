-- Month-end balance per client: the ledger balance on each account's last day of the month, added up per client.
-- Run inside the database (one row per account per day in, one row per client per month out).
WITH month_end AS (
    SELECT b.account_no,
           date_trunc('month', b.balance_date) AS month,
           b.ledger_balance,
           ROW_NUMBER() OVER (PARTITION BY b.account_no, date_trunc('month', b.balance_date)
                              ORDER BY b.balance_date DESC) AS day_rank
    FROM "${projectKey}_daily_balances" b
)
SELECT a.client_no,
       m.month,
       SUM(m.ledger_balance) AS balance_end
FROM month_end m
JOIN "${projectKey}_accounts" a ON a.account_no = m.account_no
WHERE m.day_rank = 1
GROUP BY a.client_no, m.month
