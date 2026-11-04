"""Warehouse tables the project reads (project library: python/warehouse.py).

The warehouse team builds one table per item of the data requirements in the schema read through the EDW_AI
connection. Table names are CFF_ + the source table's name where it is known; columns follow the EDW style
seen in FCT_PAYROLL_MSTR (CSTMR_CIF, ACNT_NMBR, TOT_AMT, ..._DT, ..._FLG with Y / N). These names are our
proposal until the warehouse team confirms them: if they differ, only this file changes.
The project's staging recipes rename every column back to the standard names used by the recipes.
"""

# standard name: (warehouse table, where the warehouse team takes it from)
TABLES = {
    "clients": ("CFF_CBG_CUSTOMER", "EDW_DM.DM_CBG_CUSTOMER"),
    "accounts": ("CFF_ACNT", "to find"),
    "transactions": ("CFF_STATEMENT_ENTRY", "EDW.FCT_STATEMENT_ENTRY"),
    "daily_balances": ("CFF_ACNT_DAILY_BAL", "to find"),
    "standing_orders": ("CFF_STANDING_ORDER", "to find"),
    "payroll_files": ("CFF_PAYROLL_MSTR", "EDW.FCT_PAYROLL_MSTR"),
    "financing_schedules": ("CFF_LOAN_FUTURE_BALANCE", "EDW.FCT_LOAN_FUTURE_BALANCE + EDW.DIM_LOAN"),
    "deposits": ("CFF_DPST", "to find"),
    "sadad_payments": ("CFF_SADAD_TRNS", "EDW.FCT_SADAD_TRNS"),
    "code_lists": ("CFF_CODE_LIST", "various"),
    "trade_finance": ("CFF_TRADE_FINANCE", "to find"),
    "card_sales": ("CFF_MRCHNT_STLMNT", "to find"),
    "corporate_cards": ("CFF_CORP_CARD", "to find"),
    "bills_issued": ("CFF_BILL_PAYMENT", "EDW.DIM_BUS_BL_ACNT + EDW.FCT_BILL_PAYMENT"),
    "client_monthly_summary": ("CFF_CBG_CUSTOMER_MNTHLY", "EDW_DM.DM_CBG_CUSTOMER"),
    "group_structure": ("CFF_CSTMR_GROUP", "to find"),
    "cheques": ("CFF_CHQ", "to find"),
}
# one row per table per load: LOAD_ID, LOAD_DTTS, AS_OF_DT, SRC_TBL_NM, TRGT_TBL_NM, ROW_CNT, LOAD_STA_CD
LOAD_RECORD = "CFF_LOAD_RECORD"
AUDIT_COLUMNS = ["ETL_LOAD_ID", "ETL_LOAD_DTTS"]      # added by the warehouse team to every table

# standard columns of each table (what the recipes use after staging)
COLUMNS = {
    "clients": ["client_no", "party_type", "segment", "sub_segment", "sector", "industry", "region", "city",
                "joining_date", "legal_form", "sole_proprietor_flag", "last_annual_sales", "number_of_employees",
                "parent_client_no", "status", "closure_date"],
    "accounts": ["account_no", "client_no", "product_type", "product_code", "currency", "open_date", "close_date",
                 "status", "restricted_flag", "payroll_account_flag"],
    "transactions": ["txn_id", "account_no", "posting_date", "value_date", "amount_sar", "amount_orig", "currency",
                     "exchange_rate", "debit_credit", "txn_code", "channel", "txn_type_tag", "counterparty_iban_code",
                     "counterparty_bank", "counterparty_name", "own_account_flag", "origin_type", "origin_ref",
                     "reversal_flag", "original_ref"],
    "daily_balances": ["account_no", "balance_date", "ledger_balance", "available_balance", "blocked_amount",
                       "hold_amount", "overdraft_limit"],
    "standing_orders": ["instruction_id", "account_no", "beneficiary_type", "beneficiary_iban_code", "amount",
                        "currency", "frequency", "start_date", "end_date", "next_payment_date", "status",
                        "created_date", "changed_date"],
    "payroll_files": ["FILE_ID", "DBT_CSTMR_CIF", "PAYROLL_ACNT_NMBR", "CRT_DTTS", "SLRY_TRNS_DT", "FND_DT", "TOT_AMT",
                      "NMBR_OF_EMP", "FILE_TYP_EN_NM", "FILE_SUB_TYP_EN_NM", "STA_CD", "DLT_FLG"],
    "financing_schedules": ["loan_no", "client_no", "product", "account_debited", "contract_start_date",
                            "maturity_date", "instalment_no", "due_date", "principal", "profit", "instalment_amount",
                            "paid_date", "paid_amount", "status", "early_settlement_flag", "credit_limit_total",
                            "credit_limit_used", "credit_limit_available"],
    "deposits": ["deposit_id", "client_no", "account_no", "product", "amount", "start_date", "maturity_date",
                 "expected_profit", "rollover_instruction", "status"],
    "sadad_payments": ["payment_id", "account_no", "client_no", "biller_id", "biller_name", "biller_category",
                       "bill_number", "bill_issue_date", "amount", "due_date", "paid_date", "channel"],
    "code_lists": ["code_type", "code", "meaning"],
    "trade_finance": ["instrument_id", "client_no", "instrument_type", "subtype", "amount", "currency", "issue_date",
                      "expiry_date", "payment_date", "cash_margin_account", "cash_margin_amount", "claim_amount",
                      "claim_date", "status"],
    "card_sales": ["merchant_id", "client_no", "sales_date", "settlement_date", "sales_amount", "fees", "chargebacks",
                   "net_amount_paid"],
    "corporate_cards": ["record_id", "record_type", "client_no", "statement_date", "amount_due", "due_date",
                        "paid_date", "paid_amount", "topup_date", "topup_amount"],
    "bills_issued": ["invoice_id", "client_no", "customer_iban_code", "issue_date", "due_date", "amount", "paid_date",
                     "paid_amount"],
    "client_monthly_summary": ["client_no", "month", "txn_count_in", "txn_amount_in", "txn_count_out",
                               "txn_amount_out", "online_amount", "cheque_count", "avg_balance", "payroll_salary",
                               "payroll_bonus"],
    "group_structure": ["parent_client_no", "child_client_no", "relationship_type", "start_date"],
    "cheques": ["cheque_id", "account_no", "direction", "status", "received_date", "cheque_date", "cleared_date",
                "amount", "post_dated_flag"],
}

ABBREVIATIONS = {
    "client": "CSTMR", "customer": "CSTMR", "account": "ACNT", "no": "NMBR", "number": "NMBR", "date": "DT",
    "amount": "AMT", "txn": "TRNS", "code": "CD", "flag": "FLG", "type": "TYP", "name": "NM", "status": "STA_CD",
    "currency": "CCY", "balance": "BAL", "count": "CNT", "instruction": "INSTR", "beneficiary": "BNFCRY",
    "frequency": "FREQ", "created": "CRT", "changed": "CHG", "start": "STRT", "next": "NXT", "payment": "PYMT",
    "maturity": "MTRTY", "profit": "PRFT", "principal": "PRNCPL", "instalment": "INSTL", "expected": "EXPCTD",
    "rollover": "RLOVR", "deposit": "DPST", "product": "PROD", "issue": "ISSU", "expiry": "EXPRY",
    "settlement": "STLMNT", "sales": "SLS", "fees": "FEE", "chargebacks": "CHRGBCK", "statement": "STMT",
    "record": "REC", "merchant": "MRCHNT", "invoice": "INV", "bank": "BNK", "counterparty": "CNTRPRTY",
    "origin": "ORGN", "ref": "REF", "reversal": "RVRSL", "original": "ORGNL", "channel": "CHNL", "value": "VAL",
    "posting": "PSTNG", "exchange": "EXCH", "rate": "RT", "debit": "DBT", "credit": "CRDT", "ledger": "LDGR",
    "available": "AVLBL", "blocked": "BLCKD", "hold": "HLD", "overdraft": "OVRDFT", "limit": "LMT",
    "segment": "SGMNT", "sector": "SCTR", "industry": "INDSTRY", "region": "RGN", "joining": "JOIN", "legal": "LGL",
    "proprietor": "PRPRTR", "last": "LST", "annual": "ANNL", "employees": "EMP", "closure": "CLSR", "close": "CLS",
    "open": "OPN", "restricted": "RSTRCTD", "biller": "BLLR", "category": "CTGRY", "bill": "BL", "cheque": "CHQ",
    "direction": "DRCTN", "received": "RCVD", "cleared": "CLRD", "dated": "DTD", "instrument": "INSTRMNT",
    "subtype": "SUB_TYP", "cash": "CSH", "margin": "MRGN", "claim": "CLM", "contract": "CNTRCT", "debited": "DBTD",
    "early": "ERLY", "total": "TOT", "relationship": "RLTNSHP", "month": "MNTH", "avg": "AVG", "online": "ONLN",
    "meaning": "MNNG", "parent": "PRNT", "child": "CHLD", "party": "PRTY", "sub": "SUB", "party_type": "PRTY_TYP",
    "topup": "TOPUP", "due": "DUE", "paid": "PAID", "used": "USED", "own": "OWN", "tag": "TAG", "orig": "ORIG",
    "salary": "SLRY", "bonus": "BONUS", "loan": "LOAN", "net": "NET",
}


def warehouse_column(column):
    """EDW-style name of a standard column: client_no -> CSTMR_CIF, value_date -> VAL_DT, own_account_flag -> OWN_ACNT_FLG."""
    if column.isupper():                        # already a warehouse name (payroll files)
        return column
    words = column.replace("client_no", "cstmr cif").replace("_", " ").split()
    return "_".join(w.upper() if w in ("cstmr", "cif") else ABBREVIATIONS.get(w, w.upper()) for w in words)


def is_flag(column):
    """Yes / no columns: true / false in the project, Y / N in the warehouse."""
    return column.endswith("_flag")


def staging_sql(table, schema):
    """SELECT that reads a warehouse table and gives back the standard column names and types."""
    warehouse_table = TABLES[table][0]
    select = []
    for column in COLUMNS[table]:
        source = f'"{warehouse_column(column)}"'
        select.append(f"({source} = 'Y') AS {column}" if is_flag(column) else f'{source} AS "{column}"')
    return (f"-- {warehouse_table} -> {table}: warehouse column names back to the standard names\n"
            "SELECT\n    " + ",\n    ".join(select) + f'\nFROM "{schema}"."{warehouse_table}"')


for _table, _columns in COLUMNS.items():          # every warehouse name must be unique within its table
    _names = [warehouse_column(c) for c in _columns]
    assert len(set(_names)) == len(_names), (_table, _names)
