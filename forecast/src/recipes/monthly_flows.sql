-- Monthly flows: money in and out per client, month and category, run inside the database.
-- Leaves out transfers between the client's own accounts and reversed postings (the reversal and its original).
-- Category: the posting's origin first, then the warehouse transaction type tag, then other in / out.
-- known_1..known_3: the part of the amount that came from obligations already known 1, 2 or 3 months before
-- (taken out of what the models must predict).
WITH reversed AS (
    SELECT original_ref AS txn_id
    FROM "${projectKey}_transactions"
    WHERE reversal_flag AND original_ref IS NOT NULL
),
postings AS (
    SELECT a.client_no,
           t.txn_id,
           t.origin_type,
           t.origin_ref,
           t.amount_sar,
           date_trunc('month', t.value_date) AS month,
           CASE WHEN t.debit_credit = 'C' THEN 'IN' ELSE 'OUT' END AS direction,
           CASE
               WHEN t.origin_type = 'PAYROLL' THEN 'PAYROLL'
               WHEN t.origin_type = 'LOAN' AND t.debit_credit = 'D' THEN 'FINANCING'
               WHEN t.origin_type = 'SADAD' THEN 'SADAD_GOV'
               WHEN t.origin_type = 'CARD_SETTLEMENT' THEN 'CARD_SALES'
               WHEN t.origin_type = 'CORP_CARD' THEN 'CORP_CARD'
               WHEN t.origin_type = 'LC' THEN 'SUPPLIERS'
               WHEN t.origin_type = 'DEPOSIT' AND t.debit_credit = 'C' THEN 'DEPOSIT_MAT'
               WHEN t.origin_type = 'CHEQUE' AND t.debit_credit = 'C' THEN 'CHEQUES_IN'
               WHEN t.origin_type = 'INVOICE' THEN 'RECEIPTS'
               WHEN t.txn_type_tag = 'RENT' THEN 'RENT'
               WHEN t.txn_type_tag = 'SUPPLIER' AND t.debit_credit = 'D' THEN 'SUPPLIERS'
               WHEN t.txn_type_tag = 'COLLECTION' AND t.debit_credit = 'C' THEN 'RECEIPTS'
               WHEN t.txn_type_tag = 'SALARY' THEN 'PAYROLL'
               WHEN t.txn_type_tag IN ('UTILITY', 'GOV') THEN 'SADAD_GOV'
               WHEN t.debit_credit = 'C' THEN 'OTHER_IN'
               ELSE 'OTHER_OUT'
           END AS category
    FROM "${projectKey}_transactions" t
    JOIN "${projectKey}_accounts" a ON a.account_no = t.account_no
    WHERE NOT t.reversal_flag
      AND NOT t.own_account_flag
      AND NOT EXISTS (SELECT 1 FROM reversed r WHERE r.txn_id = t.txn_id)
),
known AS (    -- when was each scheduled payment known? (earliest month per origin reference)
    SELECT origin_type, origin_ref, MIN(known_month) AS known_month
    FROM "${projectKey}_obligation_items"
    GROUP BY origin_type, origin_ref
),
flagged AS (
    SELECT p.*,
           (EXTRACT(YEAR FROM p.month) - EXTRACT(YEAR FROM k.known_month)) * 12
           + EXTRACT(MONTH FROM p.month) - EXTRACT(MONTH FROM k.known_month) AS lead_months
    FROM postings p
    LEFT JOIN known k ON k.origin_type = p.origin_type AND k.origin_ref = p.origin_ref
)
SELECT client_no,
       month,
       category,
       direction,
       ROUND(SUM(amount_sar)::numeric, 2)::float8 AS amount,
       ROUND(SUM(CASE WHEN lead_months >= 1 THEN amount_sar ELSE 0 END)::numeric, 2)::float8 AS known_1,
       ROUND(SUM(CASE WHEN lead_months >= 2 THEN amount_sar ELSE 0 END)::numeric, 2)::float8 AS known_2,
       ROUND(SUM(CASE WHEN lead_months >= 3 THEN amount_sar ELSE 0 END)::numeric, 2)::float8 AS known_3,
       COUNT(*) AS postings
FROM flagged
GROUP BY client_no, month, category, direction
