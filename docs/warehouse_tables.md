# Warehouse tables: what we know

The 2 warehouse tables the forecast reads (read-only), as the bank described them. The columns the project reads
are marked **read**; everything else is never read. The mapping to the project's own names is in
[`forecast/src/warehouse.py`](../forecast/src/warehouse.py) (`COLUMNS`).

## EDW.DIM_CUSTOMER (clients)

**Source:** the `SELECT` the user sent on 2026-11-05 and the profile of a 100-row sample (retail, inactive
individuals only). This is **not the full column list**: the table has more columns than these.

**Facts measured at the bank:**
- 2026-11-08 (to confirm it was this table's count): `SELECT COUNT(*) FROM EDW.DIM_CUSTOMER` returned about **5.2 × 10⁸ rows (520 million)** in 2 min 34 s.
  A bank has a few million customers at most, so the table most likely keeps history (several rows per
  customer). Reading all of it is what made staging look stuck.
- **The bank's query for the business clients (2026-11-08):**
  `WHERE (PRIME_RM LIKE '12%' OR SCTR_DSC = 'Sole Proprietorships') AND END_DT = DATE '2199-12-31'`.
  It returns as many rows as distinct `CSTMR_CIF`: about 420k, one per customer. Without the parentheses (`AND`
  binds before `OR`) it returned over 1 million rows, because `END_DT` then only applied to sole proprietors.
  - `PRIME_RM`: the prime relationship manager's code; business clients' start with `12`.
  - `END_DT`: the end of a row's validity; `2199-12-31` on each customer's current row (the table keeps history).
  The project uses both conditions as they are (`CLIENTS_BUSINESS`, `CLIENTS_CURRENT_ROW` in `warehouse.py`).
- `SELECT ... FROM all_objects` failed (no right on that view).

| # | Column | Meaning | Values seen / notes | Project |
|---|---|---|---|---|
| 1 | `SCTR_DSC` | Sector description | sample: Other Individuals | **read** |
| 2 | `BUS_SCTR_EN_NM` | Business sector (English) | sample: Other Individuals | **read** |
| 3 | `INDST_EN_NM` | Industry (English) | sample: Others | **read** |
| 4 | `SUB_INDST_EN_NM` | Sub-industry (English) | sample: Those Industries that cannot be classified as above | **read** |
| 5 | `NAT_CNTRY` | Nationality country (personal) | sample: Saudi Arabia |  |
| 6 | `RESIDENCE_CNTRY_DSC` | Residence country (personal) | sample: Saudi Arabia |  |
| 7 | `LGL_ENTY_DSC` | Legal entity description | sample: Resident Saudi Individual | **read** |
| 8 | `PRTY_SUB_LGL_DSC` | Sub-legal entity (age / gender band for individuals) | sample: mostly Resident Minor Saudi Individual (age less than 15) | **read** |
| 9 | `JOB_TITL` | Job title (personal) | sample: always empty |  |
| 10 | `GNDR_DSC` | Gender (personal) | sample: Male, Female |  |
| 11 | `CSTMR_TYP` | Customer type | sample: PARTNER | **read** |
| 12 | `LOB_DSC` | Line of business | sample: Retail | **read** |
| 13 | `IS_EMP` | Bank employee flag (personal) | sample: N |  |
| 14 | `DEPT_DSC` | Department description | sample: IT - Service Creation |  |
| 15 | `CITY` | City | sample: many Saudi cities | **read** |
| 16 | `STS_FLG` | Status | sample: INACTIVE (ACTIVE elsewhere) | **read** |
| 17 | `PRTY_ID_TYP_DSC` | Party ID type | sample: Family Registration Book (commercial registration for companies) | **read** |
| 18 | `RM_SGMNT_NM` | Relationship manager segment | sample: 110900 RETAIL - BRANCH BANKING | **read** |
| 19 | `CRT_CHNL_CD` | Creation channel code | sample: SMRT |  |
| 20 | `CSTMR_CIF` | Customer number (CIF) | numeric; links to FCT_STATEMENT_ENTRY.CSTMR_CIF | **read** |
| 21 | `CSTMR_EN_NM` | Customer name (personal) | free text |  |
| 22 | `PRIME_RM_EN_NM` | Prime relationship manager (personal) | sample: 1109001 - Branch Manager Male |  |

Also used, in the SQL filter only (not read): `PRIME_RM` (relationship manager code) and `END_DT` (end of the row's
validity).

The 23rd field of the sample, `SCTR_DSC_1`, is the same as `SCTR_DSC` (the `SELECT` lists `SCTR_DSC` twice).

## EDW.FCT_STATEMENT_ENTRY (postings)

**Partitions (seen at the bank, 2026-11-08):** partitioned by **month on `AS_OF_DT`** (the load date), not on
`VAL_DT`. Oracle's statistics: about 291 million rows in the partition up to July 2026 (it looks like the whole
history loaded at once), about 9.7 million in August 2026; later months had no statistics yet. Staging filters on
`AS_OF_DT` too (`POSTINGS_LOAD_DATE`, margin `POSTINGS_LOAD_MARGIN_DAYS` = 31) so Oracle reads only the partitions it
needs. Still to check: how far `VAL_DT` and `AS_OF_DT` are apart (value months inside one load partition).

**Source:** the column dictionary the user sent on 2026-11-05 (Excel `Book1.xlsx`): all **123 columns**, one row
per posting to an account (T24 statement entry). The meanings come from standard T24 field names and are not the
bank's official definitions.

| # | Column | Group | Likely meaning | Use for the project | Project |
|---|---|---|---|---|---|
| 1 | `STMT_ENTRY_ID` | Needed | Unique ID of the posting | Transaction ID | **read** |
| 2 | `PARTY_ID` | Needed | Warehouse key for the client | Joins inside the warehouse |  |
| 3 | `ACNT_NMBR` | Needed | Account the posting is on | Link to accounts | **read** |
| 4 | `CSTMR_CIF` | Needed | Client number | Link to clients | **read** |
| 5 | `AMT_LCY` | Needed | Amount in SAR. In T24 usually negative for debits | Main amount | **read** |
| 6 | `CRCY` | Needed | Account currency | Currency |  |
| 7 | `AMT_FCY` | Needed | Amount in the account currency, when not SAR | Original-currency amount |  |
| 8 | `TRNS_AMT` | Needed | Transaction amount (possibly calculated by the warehouse) | Compare with AMT_LCY |  |
| 9 | `VAL_DT` | Needed | Value date: when the money counts for the balance | Candidate date for placing a transaction in a forecast period | **read** |
| 10 | `PRC_DT` | Needed | Processing date in T24 | Check against TRNS_DT |  |
| 11 | `DTTM` | Useful in some cases | Date and time stamp of the posting | Time of day |  |
| 12 | `AS_OF_DT` | Not needed | Warehouse load information | Keep for load checks only |  |
| 13 | `EXPSR_DT` | Useful in some cases | Date the funds become available (cheques, for example) | Timing of available money |  |
| 14 | `CMPNY_CD` | Useful in some cases | T24 company code (legal entity or branch group) | Filtering |  |
| 15 | `TRNS_CD` | Needed | T24 transaction code | Main key for categorising transactions |  |
| 16 | `THEIR_RFR` | Needed | Counterparty reference | May hold the payment reference |  |
| 17 | `PL_CTGRY` | Not needed | Bank profit and loss category |  |  |
| 18 | `ACNT_OFCR` | Leave out (personal or sensitive) | Relationship manager (account officer) | Not requested |  |
| 19 | `PRD_CTGRY` | Needed | Product category of the account (current, savings...) | Account type |  |
| 20 | `EXCH_RT` | Needed | Exchange rate used | Currency conversion |  |
| 21 | `NEGOTIATED_RFR_NMBR` | Not needed | FX and treasury position keeping |  |  |
| 22 | `PSTN_TYP` | Not needed | FX and treasury position keeping |  |  |
| 23 | `OUR_RFR` | Needed | Bank reference | May hold the payroll file ID or payment reference |  |
| 24 | `RVRSL_MRKR` | Needed | Reversal marker | Exclude or net reversed postings | **read** |
| 25 | `CRCY_MRKT` | Not needed | FX and treasury position keeping |  |  |
| 26 | `DEPT_CD` | Leave out (personal or sensitive) | Department of the staff member | Not requested |  |
| 27 | `TRNS_RFR` | Needed | Reference of the originating record (FT..., AA..., TT...) | Join to funds transfer, financing contract and so on | **read** |
| 28 | `SYS_ID` | Needed | T24 module that created the posting (FT funds transfer, TT teller, AA financing, AC account, IC interest, CQ cheques...) | Links the posting to its origin |  |
| 29 | `SYS_NM` | Needed | Name of that T24 module | Links the posting to its origin |  |
| 30 | `TRNS_DT` | Needed | Booking date | Candidate date for placing a transaction in a forecast period |  |
| 31 | `INPTR` | Leave out (personal or sensitive) | ID of the bank staff member who entered the posting | Staff data |  |
| 32 | `AUTHORISER` | Leave out (personal or sensitive) | ID of the bank staff member who approved the posting | Staff data |  |
| 33 | `SUSPN_CTGRY` | Not needed | Internal T24 processing |  |  |
| 34 | `SUSPN_VAL_DT` | Not needed | Internal T24 processing |  |  |
| 35 | `SUPPRESS_PSTN` | Not needed | Internal T24 processing |  |  |
| 36 | `CRF_TYP` | Not needed | T24 consolidation and general ledger reporting |  |  |
| 37 | `CRDT_DBT_FLG` | Needed | Credit or debit | Money in or out | **read** |
| 38 | `CRF_TXN_CD` | Not needed | T24 consolidation and general ledger reporting |  |  |
| 39 | `CRF_CRCY` | Not needed | T24 consolidation and general ledger reporting |  |  |
| 40 | `CONSOL_KEY` | Not needed | T24 consolidation and general ledger reporting |  |  |
| 41 | `CRF_MTUR_DT` | Not needed | T24 consolidation and general ledger reporting |  |  |
| 42 | `CRF_PRD_CTGRY` | Not needed | T24 consolidation and general ledger reporting |  |  |
| 43 | `PM_TYP` | Not needed | FX and treasury position keeping |  |  |
| 44 | `DEALER_DSK` | Not needed | FX and treasury position keeping |  |  |
| 45 | `CNTRPRTY` | Useful in some cases | Counterparty, mainly for treasury deals | Treasury flows |  |
| 46 | `LQD_MOD` | Not needed | Internal T24 processing |  |  |
| 47 | `REPAY_DT` | Useful in some cases | Repayment date (financing postings) | Financing |  |
| 48 | `OUTSTANDING_BLNC` | Useful in some cases | Outstanding balance (financing postings) | Financing |  |
| 49 | `CNTRC_INT_RAT` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 50 | `CNTRC_INT_KEY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 51 | `CYCLE_FWD` | Not needed | Internal T24 processing |  |  |
| 52 | `ORGN_LCL_EQUIV` | Useful in some cases | Original amount, local equivalent | Foreign-currency payments |  |
| 53 | `ORGN_AMOUNT` | Useful in some cases | Original amount before conversion | Foreign-currency payments |  |
| 54 | `ORGN_CRCY` | Useful in some cases | Original currency | Foreign-currency payments |  |
| 55 | `ORGN_ACNT` | Useful in some cases | Original account. May be the account on the other side | Check: possible counterparty account |  |
| 56 | `ORGN_EXCH_RAT` | Useful in some cases | Original exchange rate | Foreign-currency payments |  |
| 57 | `ORGN_AMT_LCY` | Useful in some cases | Original amount in SAR | Foreign-currency payments |  |
| 58 | `CHQ_COLL_ID` | Useful in some cases | Cheque collection ID | Cheques |  |
| 59 | `DRAWN_ACNT` | Useful in some cases | Account the cheque is drawn on | Cheques |  |
| 60 | `ACCOUNTING_DT` | Needed | Accounting date | Usually the same as the booking date |  |
| 61 | `PRC_FWD` | Not needed | Internal T24 processing |  |  |
| 62 | `AA_SUSPEN_RFR` | Needed | Financing contract reference (suspense) | Link to financing |  |
| 63 | `POS_EXP_DT` | Not needed | Internal T24 processing |  |  |
| 64 | `RC_DTL_ID` | Not needed | Internal T24 processing |  |  |
| 65 | `CUS_PL_ACNT` | Not needed | Bank profit and loss category |  |  |
| 66 | `DO_CRDT_CHK` | Not needed | Internal T24 processing |  |  |
| 67 | `POSTING_TYP` | Not needed | Internal T24 processing |  |  |
| 68 | `AMT_DEAL_CRCY` | Not needed | FX and treasury position keeping |  |  |
| 69 | `DEAL_CRCY` | Not needed | FX and treasury position keeping |  |  |
| 70 | `DEAL_EXCH_RAT` | Not needed | FX and treasury position keeping |  |  |
| 71 | `MASK_PRNT` | Not needed | T24 flag to hide details on printed statements. Not warehouse data masking |  |  |
| 72 | `MASK_NRTV` | Not needed | T24 flag to hide the description on printed statements. Not warehouse data masking |  |  |
| 73 | `STMT1_DT` | Useful in some cases | Statement date | Statements |  |
| 74 | `CHQ_TYP` | Useful in some cases | Cheque type | Cheques |  |
| 75 | `TAX_EXCH_RAT` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 76 | `NET_PARAM` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 77 | `MSTR_ACNT` | Useful in some cases | Master account in cash pooling | Hint for group structure |  |
| 78 | `INTRST_RAT` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 79 | `DD_MANDATE_RFR` | Needed | Direct debit mandate reference | Recurring payments |  |
| 80 | `DD_ITM_RFR` | Needed | Direct debit item reference | Recurring payments |  |
| 81 | `GRS_INT_TAX_LCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 82 | `TAX_ALWD_ADJ_LCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 83 | `TAX_EXM_ADJ_LCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 84 | `POOL2_ADJ_LCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 85 | `GRS_INT_TAX_FCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 86 | `TAX_ALWD_ADJ_FCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 87 | `TAX_EXM_ADJ_FCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 88 | `POOL2_ADJ_FCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 89 | `UPD_POOL_SEQ` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 90 | `NET_INT_TAX_LCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 91 | `NET_INT_TAX_FCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 92 | `TAX_ALWD_CRCY` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 93 | `POOL1_MVMT` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 94 | `POOL2_MVMT` | Not needed | Interest, tax and cash pooling calculations |  |  |
| 95 | `DD_MANDATE_DT` | Needed | Direct debit mandate date | Recurring payments |  |
| 96 | `CNTRC_BAL_ID` | Useful in some cases | Contract balance ID | Financing |  |
| 97 | `TRADE_DT` | Useful in some cases | Trade date | Treasury flows |  |
| 98 | `BLNC_TYP` | Useful in some cases | Balance type (for example due or current balance of a financing contract) | Financing |  |
| 99 | `AA_ITM_RFR` | Needed | Financing contract (T24 AA arrangement) reference | Link to financing |  |
| 100 | `ORGN_CRCY_MRKT` | Not needed | FX and treasury position keeping |  |  |
| 101 | `CRD_NMBR` | Leave out (personal or sensitive) | Card number | Card data has strict security rules |  |
| 102 | `DRAFT_PAYEE_NM` | Leave out (personal or sensitive) | Name of a payee | Personal data |  |
| 103 | `ORGN_TRNS_FLG` | Useful in some cases | Original transaction flag | Check meaning |  |
| 104 | `CHNL_CD` | Needed | Channel code | Channel (online banking, branch, ATM, SARIE...) |  |
| 105 | `CHNL_DSC` | Needed | Channel description | Channel |  |
| 106 | `TT_TRNS_CD` | Useful in some cases | Teller transaction code | Branch cash transactions |  |
| 107 | `TRNS_TYP` | Needed | Transaction type | Probably the warehouse tags (payroll, SADAD and so on) | **read** |
| 108 | `BRNCH_CD` | Useful in some cases | Branch code | Location of branch transactions |  |
| 109 | `TRNS_CHNL` | Needed | Transaction channel | Channel |  |
| 110 | `NRTV` | Leave out (personal or sensitive) | Free-text description | Can hold employee and individual names |  |
| 111 | `CALC_NRTV1` | Leave out (personal or sensitive) | Free-text description | Can hold employee and individual names |  |
| 112 | `CALC_NRTV2` | Leave out (personal or sensitive) | Free-text description | Can hold employee and individual names |  |
| 113 | `CALC_NRTV3` | Leave out (personal or sensitive) | Free-text description | Can hold employee and individual names |  |
| 114 | `CALC_NRTV4` | Leave out (personal or sensitive) | Free-text description | Can hold employee and individual names |  |
| 115 | `CALC_NRTV5` | Leave out (personal or sensitive) | Free-text description | Can hold employee and individual names |  |
| 116 | `BLR_ID` | Needed | SADAD biller ID | SADAD bills |  |
| 117 | `BLR_NMBR` | Needed | SADAD bill number | SADAD bills |  |
| 118 | `OBJ_ID` | Not needed | Internal T24 processing |  |  |
| 119 | `INSRT_DTTS` | Not needed | Warehouse load information | Keep for load checks only |  |
| 120 | `UPD_DTTS` | Not needed | Warehouse load information | Keep for load checks only |  |
| 121 | `RUN_ID` | Not needed | Warehouse load information | Keep for load checks only |  |
| 122 | `CSTMR_KEY` | Needed | Warehouse key for the client | Joins inside the warehouse |  |
| 123 | `ACNT_KEY` | Needed | Warehouse key for the account | Joins inside the warehouse |  |
