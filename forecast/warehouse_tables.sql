-- Digital Cash Management Forecast: the 3 tables the forecast reads, built from the bank's main data.
-- The warehouse team writes each one as a CSV file (one folder per table) in the landing folder, with the audit
-- columns ETL_LOAD_ID and ETL_LOAD_DTTS, then adds one row per table to CFF_LOAD_RECORD (LOAD_ID, LOAD_DTTS,
-- AS_OF_DT, SRC_TBL_NM, TRGT_TBL_NM, ROW_CNT, LOAD_STA_CD).
-- Placeholders in <angle brackets> depend on the bank: fill them in after the checks marked "check".
-- Month start is written TRUNC(date, 'MM') (Oracle); use your database's own function if it differs.


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


-- 2. CFF_CLIENT_MNTHLY_FLOW: money in and out per client, month, direction, T24 module and transaction type
WITH entries AS (
    SELECT s.*
    FROM EDW.FCT_STATEMENT_ENTRY s
    JOIN CFF_CLIENT c ON c.CSTMR_CIF = s.CSTMR_CIF          -- business clients only (no bank internal accounts)
),
reversed AS (                                               -- a reversed posting and its reversal cancel out
    SELECT DISTINCT TRNS_RFR FROM entries WHERE RVRSL_MRKR IS NOT NULL      -- check: how reversals are marked
),
own_transfers AS (                                          -- money moved between the client's own accounts
    SELECT TRNS_RFR, CSTMR_CIF
    FROM entries
    WHERE TRNS_RFR IS NOT NULL
    GROUP BY TRNS_RFR, CSTMR_CIF
    HAVING COUNT(DISTINCT CRDT_DBT_FLG) = 2 AND SUM(AMT_LCY) = 0             -- a debit and a credit that net to 0
)
SELECT e.CSTMR_CIF,
       TRUNC(e.VAL_DT, 'MM')        AS MNTH,                -- the value date's month
       e.CRDT_DBT_FLG,                                      -- C: money in, D: money out
       e.SYS_ID,                                            -- T24 module: FT, TT, AA, CQ, STO, ...
       COALESCE(e.TRNS_TYP, 'NONE') AS TRNS_TYP,
       SUM(ABS(e.AMT_LCY))          AS AMT_LCY,             -- SAR, always positive
       COUNT(*)                     AS PSTNG_CNT
FROM entries e
WHERE NOT EXISTS (SELECT 1 FROM reversed r WHERE r.TRNS_RFR = e.TRNS_RFR)
  AND NOT EXISTS (SELECT 1 FROM own_transfers o WHERE o.TRNS_RFR = e.TRNS_RFR AND o.CSTMR_CIF = e.CSTMR_CIF)
GROUP BY e.CSTMR_CIF, TRUNC(e.VAL_DT, 'MM'), e.CRDT_DBT_FLG, e.SYS_ID, COALESCE(e.TRNS_TYP, 'NONE');
-- check: the sign of AMT_LCY against CRDT_DBT_FLG; how often VAL_DT and TRNS_DT fall in different months;
-- one row per STMT_ENTRY_ID (not one per load); the distinct SYS_ID and TRNS_TYP values with their amounts,
-- which the forecast's category_mapping needs


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
