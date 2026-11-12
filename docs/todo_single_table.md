# TODO: read one table (CUST_TRANS_SAMPLE) instead of 2

Status: **planned, not started** (2026-11-11). Nothing is changed in the code yet.

Today staging reads 2 warehouse tables: `DIM_CUSTOMER` (the clients) and `FCT_STATEMENT_ENTRY` (the postings).
They will be replaced by a single table, `CUST_TRANS_SAMPLE`, in the main project and in the load test.

## The new table

- Table: `CUST_TRANS_SAMPLE`
- Connection: `EDW_AI`
- Shape: **one row per transaction**, with the client's details repeated on every row. It is not aggregated:
  the database still sums the rows per client, day and category.

| Column | Type | Max length | Used for |
|---|---|---|---|
| BUS_SCTR_EN_NM | string | 150 | client: business_sector |
| INDST_EN_NM | string | 100 | client: industry |
| SUB_INDST_EN_NM | string | 100 | client: sub_industry |
| NAT_CNTRY | string | 60 | not read |
| RESIDENCE_CNTRY_DSC | string | 50 | not read |
| LGL_ENTY_DSC | string | 150 | client: legal_entity (sole proprietor check) |
| PRTY_SUB_LGL_DSC | string | 200 | client: sub_legal_entity (sole proprietor check) |
| JOB_TITL | string | 80 | not read (personal) |
| GNDR_DSC | string | 100 | not read (personal) |
| CSTMR_TYP | string | 25 | client: customer_type |
| LOB_DSC | string | 25 | client: line_of_business |
| IS_EMP | string | 2 | not read |
| DEPT_DSC | string | 50 | not read |
| CITY | string | 80 | client: city |
| STS_FLG | string | 10 | client: status |
| PRTY_ID_TYP_DSC | string | 255 | client: id_type |
| RM_SGMNT_NM | string | 255 | client: segment |
| SCTR_DSC | string | 100 | client: sector (business filter: Sole Proprietorships) |
| CRT_CHNL_CD | string | 10 | not read |
| CSTMR_CIF | string | 50 | **client number** |
| CSTMR_EN_NM | string | 150 | not read (personal) |
| PRIME_RM_EN_NM | string | 100 | not read (the RM's name, not the code) |
| AMT_LCY | double | | **amount in SAR** |
| CRDT_DBT_FLG | string | 5 | **credit / debit** |
| CALC_NRTV5 | string | 500 | maybe the category (to check) |
| TRNS_AMT | double | | not read (amount in the transaction's currency) |
| TRNS_DT | dateonly | | **the day** (replaces VAL_DT) |
| TRNS_DESC | string | 42 | maybe the category (to check) |

## Old vs new

| What we use today | Old (2 tables) | New table |
|---|---|---|
| Client number | CSTMR_CIF | CSTMR_CIF |
| Client details | DIM_CUSTOMER columns | same names |
| Day | VAL_DT | **TRNS_DT** |
| Amount (SAR), credit/debit | AMT_LCY, CRDT_DBT_FLG | same |
| Category (transaction type) | TRNS_TYP | **missing**: TRNS_DESC or CALC_NRTV5? |
| Remove reversals | TRNS_RFR, RVRSL_MRKR | **missing** |
| Remove own-account transfers | TRNS_RFR | **missing** |
| Business clients filter | PRIME_RM LIKE '12%' or sector Sole Proprietorships | **only the sector** (PRIME_RM code missing) |
| Client's current row | END_DT = 2199-12-31 | not needed: details taken from each client's latest row |
| Load-date partition filter | AS_OF_DT | **missing** |
| Postings count | STMT_ENTRY_ID | COUNT(*) |

## Plan

- [ ] **`forecast/src/warehouse.py`**
  - [ ] `TABLES`: one table, its name a setting (default `CUST_TRANS_SAMPLE`)
  - [ ] `COLUMNS`: the client columns and the transaction columns, both from this table; `TRNS_DT` as the day
  - [ ] Clients query: the distinct `CSTMR_CIF`, with the details from each client's latest row (ROW_NUMBER over TRNS_DT)
  - [ ] Daily totals query: one table, no join, `GROUP BY` client, TRNS_DT, category; date filter on TRNS_DT
  - [ ] Remove the reversal and own-account transfer rules, `CLIENTS_CURRENT_ROW`, `POSTINGS_LOAD_DATE`
  - [ ] Category mapping on the column chosen (question 1)
- [ ] **`forecast/src/recipes/stage_tables.py`**: one input dataset instead of 2. Everything after it (Parquet
      files, groups, new days only) stays the same.
- [ ] **Setup notebook** (`cells/settings.py`, `cells/create_flow.py`): link and check one table instead of 2
- [ ] **Sandbox fake warehouse** (`simulator.py`): write one fake `CUST_TRANS_SAMPLE` in the local Postgres
- [ ] **Load test** (`loadtest.py`): no structural change (it never reads the warehouse, it makes fake daily flows);
      it only follows the new column list and categories
- [ ] **Docs**: README, `docs/how_it_works.md`, `docs/warehouse_tables.md` (add this schema), `docs/data_flow.html`
- [ ] Regenerate the notebooks (`python make_notebook.py`)
- [ ] Tested by the user at the bank (nothing run there by Claude)

## Open questions (default if no answer)

1. **Category**: which column holds the transaction type, `TRNS_DESC` or `CALC_NRTV5`? Need sample values.
   *Default: TRNS_DESC, mapping to fill in; until then everything is OTHER plus TOTAL.*
2. **Reversals and own-account transfers**: already removed in this table?
   *Default: they cannot be removed, every row counts as it is.*
3. **Business clients**: does the table hold only business clients? The RM code (PRIME_RM) is missing.
   *Default: keep the Sole Proprietorships sector filter only; ask the warehouse team for the RM code.*
4. **Name and schema**: is `CUST_TRANS_SAMPLE` the final name? Which schema?
   *Default: both are settings.*
5. **Clients with no transaction** are not in this table, so they disappear from the forecast instead of showing
   NO_POSTINGS. Is that fine?
6. Is the table indexed or partitioned on `TRNS_DT`? The nightly read filters on it.
