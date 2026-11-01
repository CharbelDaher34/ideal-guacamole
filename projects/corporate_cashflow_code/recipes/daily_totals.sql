-- Runs inside PostgreSQL. Money in and expenses per account, day and category, without internal transfers,
-- one-offs and loan installments (obligations come from loan_schedule). Amounts are positive: the flow
-- column says in or out. Only days with transactions appear.
SELECT "account_id",
       CAST("date" AS DATE) AS "date",
       "flow",
       "category",
       SUM(ABS("amount"))   AS "amount"
FROM "${projectKey}_transactions_clean"
WHERE "excluded" = 0 AND "flow" <> 'OBLIGATION'
GROUP BY "account_id", CAST("date" AS DATE), "flow", "category"
