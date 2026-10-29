-- Runs inside PostgreSQL. Money in and money out per account and day, without internal transfers and
-- one-offs. Amounts are positive: the flow column says in or out. Only days with transactions appear.
SELECT "account_id",
       CAST("date" AS DATE) AS "date",
       "flow",
       SUM(ABS("amount"))   AS "amount"
FROM "${projectKey}_transactions_clean"
WHERE "excluded" = 0
GROUP BY "account_id", CAST("date" AS DATE), "flow"
