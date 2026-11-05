-- Digital Cash Management Forecast: the warehouse tables the forecast reads, from the bank's main data.
-- The warehouse team builds CFF_CLIENT and CFF_SCHEDULED_PYMT (tables or views, read through the database
-- connection; or CSV files in the landing folder, with the audit columns ETL_LOAD_ID and ETL_LOAD_DTTS), then
-- adds one row per table to CFF_LOAD_RECORD (LOAD_ID, LOAD_DTTS, AS_OF_DT, SRC_TBL_NM, TRGT_TBL_NM, ROW_CNT,
-- LOAD_STA_CD). Oracle SQL. Placeholders in <angle brackets> depend on the bank: fill them in after the checks.


-- 1. CFF_CLIENT: one row per business client, companies and businesses owned by one person (sole proprietors)
SELECT c.CSTMR_CIF,
       c.CSTMR_TYP,
       c.LOB_DSC,
       c.LGL_ENTY_DSC,            -- tells a company from a sole proprietor: keep both
       c.PRTY_SUB_LGL_DSC,
       c.PRTY_ID_TYP_DSC,         -- commercial registration: a business, even when CSTMR_TYP says individual
       c.RM_SGMNT_NM,
       c.SCTR_DSC,
       c.BUS_SCTR_EN_NM,
       c.INDST_EN_NM,
       c.SUB_INDST_EN_NM,
       c.CITY,
       c.STS_FLG
       -- optional, if the warehouse has them elsewhere: annual sales as ANNL_SLS_AMT, employees as NMBR_OF_EMP
FROM EDW.DIM_CUSTOMER c
WHERE <current row only>                      -- check: DIM_CUSTOMER may keep old versions of a client
  AND <business clients only>;                -- check: by LOB_DSC, LGL_ENTY_DSC or PRTY_ID_TYP_DSC, not CSTMR_TYP
-- check: SELECT COUNT(*), COUNT(DISTINCT CSTMR_CIF) ... must be equal


-- 2. The daily postings: nothing to build. The project reads EDW.FCT_STATEMENT_ENTRY itself through the read-only
-- connection and sums it per client and month in Oracle (monthly_flows_sql in the project library, warehouse.py):
-- business clients, the last 60 months, without reversed postings and transfers between the client's own accounts.
-- Add one row per load for FCT_STATEMENT_ENTRY to CFF_LOAD_RECORD (status), like the other tables.
-- check: the sign of AMT_LCY against CRDT_DBT_FLG (negative for money out); how reversals are marked (RVRSL_MRKR on
-- the reversing posting, which carries the reversed posting's TRNS_RFR); how often VAL_DT and TRNS_DT fall in
-- different months; one row per STMT_ENTRY_ID; the distinct SYS_ID and TRNS_TYP values with their amounts, which
-- the forecast's category_mapping needs:
SELECT SYS_ID, TRNS_TYP, CRDT_DBT_FLG, COUNT(*) AS POSTINGS, SUM(ABS(AMT_LCY)) AS AMOUNT
FROM EDW.FCT_STATEMENT_ENTRY
WHERE VAL_DT >= ADD_MONTHS(TRUNC(SYSDATE, 'MM'), -12)
GROUP BY SYS_ID, TRNS_TYP, CRDT_DBT_FLG
ORDER BY AMOUNT DESC;


-- 3. CFF_SCHEDULED_PYMT: every scheduled payment, past and future, with the date the bank first knew of it.
-- SYS_ID and TRNS_TYP: the values its statement entry carries when it is paid (the same category mapping applies).
SELECT l.CSTMR_CIF,
       'AA'                       AS SYS_ID,
       l.<loan reference>         AS ORGN_RFR,
       'D'                        AS CRDT_DBT_FLG,
       '<TRNS_TYP of a financing instalment>' AS TRNS_TYP,
       f.<due date>               AS DUE_DT,
       f.<instalment amount, SAR> AS AMT_LCY,
       COALESCE(l.<last reschedule date>, l.<contract start date>) AS KNOWN_DT
FROM EDW.FCT_LOAN_FUTURE_BALANCE f
JOIN EDW.DIM_LOAN l ON l.<loan reference> = f.<loan reference>
UNION ALL
SELECT d.CSTMR_CIF, '<SYS_ID of a deposit>', d.<deposit reference>, 'C', '<TRNS_TYP of a deposit maturity>',
       d.<maturity date>, d.<amount + expected profit, SAR>, d.<placement date>
FROM <deposits table> d
WHERE d.<paid to the client's account at maturity, not rolled over>
UNION ALL
SELECT o.CSTMR_CIF, 'STO', o.<instruction reference>, 'D', '<TRNS_TYP of the order: rent, supplier, ...>',
       <each future execution date>, o.<amount, SAR>, o.<creation date>
FROM <standing orders table> o;
-- later, the same way if the sources are found: letters of credit, corporate card statements, SADAD bills,
-- post-dated cheques
