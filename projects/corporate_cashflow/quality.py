"""Data quality rules on the inputs and dashboard pieces, shared by CORP_CASHFLOW and CORP_CASHFLOW_CODE."""
from dssproj import (python_metric, rule_not_empty, rule_python, rule_record_count, rule_unique, tile_chart, tile_metric,
                     tile_scenario, tile_table, tile_text)

INPUTS = ["transactions", "accounts", "calendar"]

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

# the days to forecast need their calendar features
CALENDAR_COVERS_HORIZON = '''
import dataiku
import pandas as pd


def process(last_values, dataset, partition_id):
    horizon = int(dataiku.get_custom_variables()["horizon_days"])
    last_day = pd.to_datetime(dataset.get_dataframe()["date"]).max()
    balances = pd.to_datetime(dataiku.Dataset("accounts").get_dataframe()["balance_date"]).max()
    needed = balances + pd.Timedelta(days=horizon)
    if last_day < needed:
        return "ERROR", f"The calendar stops on {last_day:%Y-%m-%d}, the forecast needs it up to {needed:%Y-%m-%d}"
    return "OK", f"Calendar up to {last_day:%Y-%m-%d}"
'''

# metrics of account_outlook, kept after each build: how many accounts at risk, the lowest balance ahead
OUTLOOK_METRIC_NAMES = ["high_accounts", "medium_accounts", "lowest_balance"]
OUTLOOK_METRICS = '''
def process(dataset, partition_id):
    df = dataset.get_dataframe()
    return {"high_accounts": int((df["risk_level"] == "HIGH").sum()),
            "medium_accounts": int((df["risk_level"] == "MEDIUM").sum()),
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


def outlook_dashboard(p, extra=()):
    """The dashboard of both projects: run buttons, accounts at risk, lowest balances, balance path, outlook."""
    p.dashboard("Cash-flow outlook", [
        (tile_text("## Cash-flow outlook, next 30 days\n"
                   "Run **Daily forecast** every day once yesterday's transactions are in, **Train model** "
                   "about once a month. The numbers below refresh after each run."), (0, 0, 36, 3)),
        (tile_scenario("Daily forecast", "Daily forecast"), (0, 3, 9, 4)),
        (tile_scenario("Train model", "Train model"), (9, 3, 9, 4)),
        (tile_metric("HIGH risk accounts", "account_outlook", python_metric("high_accounts")), (18, 3, 6, 4)),
        (tile_metric("MEDIUM risk accounts", "account_outlook", python_metric("medium_accounts")), (24, 3, 6, 4)),
        (tile_metric("Lowest balance ahead", "account_outlook", python_metric("lowest_balance")), (30, 3, 6, 4)),
        (tile_chart("Lowest balance per company (expected and 1-in-10 bad case)", "account_outlook",
                    "grouped_columns", "company_name", ["lowest_balance", "pessimistic_lowest_balance"]),
         (0, 7, 18, 11)),
        (tile_chart("Balance per day", "daily_balance_path", "lines", "date", ["balance"], subgroup="account_id",
                    date_x=True), (18, 7, 18, 11)),
        (tile_table("Outlook per account", "account_outlook"), (0, 18, 36, 10)),
        *extra,
    ])
