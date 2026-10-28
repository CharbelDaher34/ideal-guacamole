-- Runs inside PostgreSQL: one row per account and month.
SELECT "account_id",
       "month",
       SUM(CASE WHEN "direction" = 'CREDIT' THEN "amount" ELSE 0 END)                    AS "inflow",
       SUM(CASE WHEN "direction" = 'DEBIT'  THEN "amount" ELSE 0 END)                    AS "outflow",
       SUM("signed_amount")                                                              AS "net_cashflow",
       COUNT(*)                                                                          AS "n_transactions",
       SUM(CASE WHEN "category" = 'CUSTOMER_RECEIPT' THEN "amount" ELSE 0 END)           AS "customer_receipts",
       SUM(CASE WHEN "category" = 'PAYROLL' THEN "amount" ELSE 0 END)                    AS "payroll",
       SUM(CASE WHEN "category" = 'SUPPLIER_PAYMENT' THEN "amount" ELSE 0 END)           AS "supplier_payments",
       SUM(CASE WHEN "category" IN ('LOAN_REPAYMENT', 'TAX') THEN "amount" ELSE 0 END)   AS "debt_and_tax"
FROM "${projectKey}_transactions_prepared"
GROUP BY "account_id", "month"
