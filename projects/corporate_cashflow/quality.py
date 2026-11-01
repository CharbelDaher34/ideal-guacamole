"""Data quality rules on the inputs and dashboard pieces, shared by CORP_CASHFLOW and CORP_CASHFLOW_CODE."""
from dssproj import (python_metric, rule_not_empty, rule_python, rule_record_count, rule_unique, tile_chart, tile_metric,
                     tile_scenario, tile_table, tile_text)

INPUTS = ["transactions", "accounts", "calendar", "loan_schedule"]

# transactions must reach the balances' date (D-1): otherwise yesterday's transactions are missing
TRANSACTIONS_UP_TO_DATE = '''
import dataiku
import pandas as pd


def process(last_values, dataset, partition_id):
    transactions = dataset.get_dataframe()
    accounts = dataiku.Dataset("accounts").get_dataframe()
    unknown = sorted(set(transactions["account_id"]) - set(accounts["account_id"]))
    if unknown:
        return "ERROR", f"Transactions of unknown accounts: {', '.join(unknown[:5])}"
    latest = pd.to_datetime(transactions["booking_date"]).max()
    balances = pd.to_datetime(accounts["balance_date"]).max()
    if latest < balances:
        return "ERROR", f"Transactions stop on {latest:%Y-%m-%d}, balances are of {balances:%Y-%m-%d}"
    return "OK", f"Transactions up to {latest:%Y-%m-%d}"
'''

ONE_BALANCE_DATE = '''
import pandas as pd


def process(last_values, dataset, partition_id):
    dates = pd.to_datetime(dataset.get_dataframe()["balance_date"]).dt.date.unique()
    if len(dates) != 1:
        return "ERROR", f"{len(dates)} different balance dates: every account must have yesterday's balance"
    return "OK", f"Balances of {dates[0]}"
'''

# the days to forecast need their calendar row (day of month, weekday...)
CALENDAR_COVERS_HORIZON = '''
import dataiku
import pandas as pd


def process(last_values, dataset, partition_id):
    months = int(dataiku.get_custom_variables()["horizon_months"])
    last_day = pd.to_datetime(dataset.get_dataframe()["date"]).max()
    balances = pd.to_datetime(dataiku.Dataset("accounts").get_dataframe()["balance_date"]).max()
    needed = ((balances + pd.Timedelta(days=1)).to_period("M") + months).end_time.normalize()
    if last_day < needed:
        return "ERROR", f"The calendar stops on {last_day:%Y-%m-%d}, the forecast needs it up to {needed:%Y-%m-%d}"
    return "OK", f"Calendar up to {last_day:%Y-%m-%d}"
'''

# every installment due within the transaction history was paid: a transaction of the loan, same amount, within
# 3 days of the due date
INSTALLMENTS_PAID = '''
import dataiku
import pandas as pd


def process(last_values, dataset, partition_id):
    schedule = dataset.get_dataframe()
    accounts = dataiku.Dataset("accounts").get_dataframe()
    unknown = sorted(set(schedule["account_id"]) - set(accounts["account_id"]))
    if unknown:
        return "ERROR", f"Loans of unknown accounts: {', '.join(unknown[:5])}"
    if (schedule["amount"] <= 0).any():
        return "ERROR", "Installment amounts must be positive (money to pay)"
    today = pd.to_datetime(accounts["balance_date"]).max()
    schedule["due_date"] = pd.to_datetime(schedule["due_date"])
    tx = dataiku.Dataset("transactions").get_dataframe()
    tx["booking_date"] = pd.to_datetime(tx["booking_date"])
    due = schedule[(schedule["due_date"] <= today) & (schedule["due_date"] >= tx["booking_date"].min())]
    tx = tx[tx["loan_id"].notna() & (tx["loan_id"].astype(str) != "")]
    tx = tx.assign(paid=tx["amount"].abs())
    paid = due.merge(tx[["loan_id", "booking_date", "paid"]], on="loan_id")
    paid = paid[((paid["booking_date"] - paid["due_date"]).dt.days.abs() <= 3)
                & ((paid["paid"] - paid["amount"]).abs() <= 0.01 * paid["amount"])]
    missing = due.merge(paid[["loan_id", "due_date"]].drop_duplicates(), how="left", indicator=True)
    missing = missing[missing["_merge"] == "left_only"]
    if len(missing):
        first = missing.iloc[0]
        return "ERROR", (f"{len(missing)} installments due but not paid, e.g. {first['loan_id']} "
                         f"of {first['due_date']:%Y-%m-%d}")
    return "OK", f"{len(due)} installments due up to {today:%Y-%m-%d}, all paid"
'''

# metrics of monthly_outlook, kept after each build: how many accounts at risk (an account's risk = its worst
# month), the lowest balance ahead
OUTLOOK_METRIC_NAMES = ["high_accounts", "medium_accounts", "lowest_balance"]
OUTLOOK_METRICS = '''
def process(dataset, partition_id):
    df = dataset.get_dataframe()
    worst = df.groupby("account_id")["risk_level"].agg(lambda r: "HIGH" if "HIGH" in set(r)
                                                       else "MEDIUM" if "MEDIUM" in set(r) else "LOW")
    return {"high_accounts": int((worst == "HIGH").sum()),
            "medium_accounts": int((worst == "MEDIUM").sum()),
            "lowest_balance": float(df["lowest_balance"].min())}
'''


def add_input_rules(p):
    p.data_quality("transactions", [
        rule_record_count("Has transactions", minimum=1),
        rule_not_empty("No empty field", "txn_id", "account_id", "booking_date", "amount", "description"),
        rule_unique("One row per transaction", "txn_id"),
        rule_python("Known accounts, up to yesterday", TRANSACTIONS_UP_TO_DATE),
    ])
    p.data_quality("accounts", [
        rule_record_count("Has accounts", minimum=1),
        rule_unique("One row per account", "account_id"),
        rule_not_empty("Balance and limit filled", "balance", "balance_date", "overdraft_limit"),
        rule_python("One balance date", ONE_BALANCE_DATE),
    ])
    p.data_quality("calendar", [
        rule_unique("One row per day", "date"),
        rule_python("Covers the days to forecast", CALENDAR_COVERS_HORIZON),
    ])
    p.data_quality("loan_schedule", [
        rule_not_empty("No empty field", "account_id", "loan_id", "due_date", "amount"),
        rule_python("Installments paid as scheduled", INSTALLMENTS_PAID),
    ])


# tiles a project adds with outlook_dashboard(extra=...) go below this row of the 36-column grid
EXTRA_TOP = 50


def outlook_dashboard(p, extra=()):
    """The dashboard of both projects: run buttons, accounts at risk, months ahead, balance path, drivers."""
    p.dashboard("Cash-flow outlook", [
        (tile_text("## Cash-flow outlook, next 3 months\n"
                   "Run **Daily forecast** every day once yesterday's transactions are in, **Train model** "
                   "about once a month. Outflows = expenses (forecast) + obligations (loan installments, "
                   "from the schedule)."), (0, 0, 36, 3)),
        (tile_scenario("Daily forecast", "Daily forecast"), (0, 3, 9, 4)),
        (tile_scenario("Train model", "Train model"), (9, 3, 9, 4)),
        (tile_metric("HIGH risk accounts", "monthly_outlook", python_metric("high_accounts")), (18, 3, 6, 4)),
        (tile_metric("MEDIUM risk accounts", "monthly_outlook", python_metric("medium_accounts")), (24, 3, 6, 4)),
        (tile_metric("Lowest balance ahead", "monthly_outlook", python_metric("lowest_balance")), (30, 3, 6, 4)),
        (tile_chart("All accounts per month: inflows, expenses, obligations", "monthly_outlook",
                    "grouped_columns", "month", ["inflows", "expenses", "obligations"], function="SUM"),
         (0, 7, 18, 11)),
        (tile_chart("Lowest balance per company and month", "monthly_outlook", "grouped_columns",
                    "company_name", ["lowest_balance"], subgroup="month"), (18, 7, 18, 11)),
        (tile_chart("Balance per day", "daily_balance_path", "lines", "date", ["balance"], subgroup="account_id",
                    date_x=True), (0, 18, 36, 10)),
        (tile_table("Per account and month", "monthly_outlook"), (0, 28, 36, 11)),
        (tile_table("Key drivers per account and month", "monthly_drivers"), (0, 39, 36, 11)),
        *extra,
    ])
