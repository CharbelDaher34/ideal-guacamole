"""Synthetic corporate-banking data, day by day: 10 companies, 3 years of transactions (Oct 2023 - Sep 2026).

Writes data/accounts.csv, data/transactions.csv and data/calendar.csv. Fully made up, seeded so reruns match.

Each transaction has a signed amount (+ money in, - money out) and a bank-style description, like a real
statement. Customer receipts and supplier payments come on business days (Mon-Fri), more receipts early in
the month and in the sector's high season. Payroll (25th), rent (1st), loan installments (10th), utility bills
(15th), quarterly VAT (20th of Jan/Apr/Jul/Oct) and bank fees (month end) recur. Internal transfers to the
company's savings account and a few one-off deals (asset sales, loan drawdowns, ...) are mixed in: the flow
removes them before forecasting. Balances are "today" = end of 30 Sep 2026 (data up to D-1, D = 1 Oct).
"""
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"
START, END = date(2023, 10, 1), date(2026, 9, 30)     # history: up to yesterday
CALENDAR_END = date(2026, 12, 31)                     # the calendar also covers the days to forecast
rng = np.random.default_rng(7)

# sector -> revenue multiplier per calendar month (Jan..Dec)
SEASON = {
    "Retail":       [0.85, 0.80, 0.90, 0.95, 1.00, 0.95, 0.95, 1.00, 0.95, 1.05, 1.25, 1.45],
    "Hospitality":  [0.70, 0.70, 0.85, 0.95, 1.05, 1.30, 1.45, 1.45, 1.10, 0.90, 0.75, 0.90],
    "Agriculture":  [0.60, 0.60, 0.75, 0.85, 0.95, 1.00, 1.10, 1.30, 1.55, 1.45, 0.95, 0.65],
    "Construction": [0.70, 0.75, 0.95, 1.10, 1.20, 1.20, 1.15, 1.10, 1.10, 1.00, 0.90, 0.75],
    "Other":        [1.00] * 12,
}
# name, sector, monthly revenue, costs as share of revenue, revenue trend per year, has a loan,
# balance today (end of 30 Sep 2026), overdraft limit
COMPANIES = [
    ("Cedar Steelworks",        "Other",        900_000, 0.85,  0.06, True,  1_100_000, 300_000),
    ("Phoenicia Retail Group",  "Retail",       600_000, 0.86,  0.08, False,   420_000, 150_000),
    ("Levant Logistics",        "Other",        450_000, 0.97, -0.20, True,     60_000,  60_000),
    ("Mount Lebanon Hotels",    "Hospitality",  350_000, 0.90,  0.04, True,     90_000, 100_000),
    ("Bekaa Agro Holdings",     "Agriculture",  250_000, 0.82,  0.05, False,   210_000,  80_000),
    ("Beirut Digital Systems",  "Other",        300_000, 0.75,  0.12, False,   600_000, 100_000),
    ("Tripoli Textiles",        "Other",        200_000, 0.97, -0.25, True,     20_000,  40_000),
    ("Byblos Construction",     "Construction", 500_000, 0.88,  0.03, True,    260_000, 150_000),
    ("Orient Medical Supplies", "Other",        400_000, 0.84,  0.07, False,   350_000, 120_000),
    ("Sidon Port Services",     "Other",        280_000, 0.94,  0.00, False,    90_000,  50_000),
]
CUSTOMERS = ["ABC TRADING SAL", "NOUR DISTRIBUTION", "GULF IMPORTS LLC", "ATLAS HOLDING", "MEDCO SARL",
             "ZAHLE FOODS", "ALPHA CONTRACTING", "BLUE WAVE SHIPPING", "SUNRISE MARKETS", "OMEGA PHARMA"]
SUPPLIERS = ["UNITED PACKAGING", "LEBANON FUEL CO", "STEEL IMPORTS SAL", "PRIME LOGISTICS", "OFFICE PLUS",
             "TECHNO PARTS", "AGRI INPUTS SARL", "CITY MAINTENANCE"]
CARD_SECTORS = ("Retail", "Hospitality")             # these also get daily card (POS) settlements


def business_day(d):
    """Same day, or the previous Friday if it falls on a weekend."""
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d


def days(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main():
    OUT.mkdir(exist_ok=True)
    accounts, txns = [], []

    def add(acc, d, amount, description):
        txns.append({"txn_id": f"T{len(txns) + 1:06d}", "account_id": acc, "booking_date": d.isoformat(),
                     "amount": round(float(amount), 2), "description": description})

    for i, (name, sector, revenue0, cost_ratio, trend, loan, balance, overdraft) in enumerate(COMPANIES):
        acc = f"CA{1001 + i}"
        accounts.append({"account_id": acc, "company_name": name, "sector": sector,
                         "relationship_manager": ["R. Haddad", "M. Khoury", "L. Saad"][i % 3],
                         "balance": balance, "balance_date": END.isoformat(), "overdraft_limit": overdraft})
        costs = revenue0 * cost_ratio                         # monthly costs, set by the starting revenue
        customers = rng.choice(CUSTOMERS, 5, replace=False)
        suppliers = rng.choice(SUPPLIERS, 4, replace=False)

        for d in days(START, END):
            years = (d - START).days / 365
            if d.weekday() < 5:                               # business day: receipts and supplier payments
                revenue = revenue0 * (1 + trend) ** years * SEASON[sector][d.month - 1]
                daily = revenue / 21.7 * (1.5 if d.day <= 7 else 1.0) * (1.2 if d.weekday() == 0 else 1.0)
                if sector in CARD_SECTORS:                    # card sales settle every business day
                    add(acc, d, daily * 0.4 * rng.normal(1, 0.15), "POS SETTLEMENT CARD SALES")
                    daily *= 0.6
                for _ in range(rng.poisson(1.6)):
                    c = rng.choice(customers)
                    add(acc, d, daily / 1.6 * rng.lognormal(0, 0.45), f"TRF FROM {c} INV {rng.integers(1000, 9999)}")
                for _ in range(rng.poisson(1.2)):
                    s = rng.choice(suppliers)
                    add(acc, d, -costs * 0.33 / 21.7 / 1.2 * rng.lognormal(0, 0.4),
                        f"PAYMENT TO {s} REF {rng.integers(10000, 99999)}")

            mon = d.strftime("%b %Y").upper()
            # recurring payments, moved to the previous business day when the date is a weekend (rent: next)
            if d == business_day(d.replace(day=25)):
                add(acc, d, -costs * 0.40 * (1 + 0.03 * int(years)), f"SALARY {mon}")
            first = d.replace(day=1)
            first_bd = first if first.weekday() < 5 else first + timedelta(days=7 - first.weekday())
            if d == first_bd:
                add(acc, d, -costs * 0.10, f"RENT {mon} {name.split()[0].upper()} PROPERTIES")
            if loan and d == business_day(d.replace(day=10)):
                add(acc, d, -costs * 0.08, f"LOAN INSTALLMENT LN{4400 + i}")
            if d == business_day(d.replace(day=15)):
                add(acc, d, -costs * 0.05 * rng.normal(1, 0.1), f"EDL ELECTRICITY BILL {mon}")
            if d.month in (1, 4, 7, 10) and d == business_day(d.replace(day=20)):
                q = (d.month - 1) // 3 or 4
                add(acc, d, -revenue0 * 0.12, f"VAT PAYMENT Q{q} {d.year if q != 4 else d.year - 1}")
            if (d + timedelta(days=1)).day == 1:
                add(acc, d, -rng.uniform(40, 160), "ACCOUNT MAINTENANCE FEE")

            # removed by the flow: internal transfers (out to savings, back later) and one-off deals
            if d.weekday() < 5 and rng.random() < 0.02:
                add(acc, d, -revenue0 * rng.uniform(0.2, 0.5), f"INTERNAL TRF TO SAVINGS {acc}-S")
            if d.weekday() < 5 and rng.random() < 0.015:
                add(acc, d, revenue0 * rng.uniform(0.2, 0.5), f"INTERNAL TRF FROM SAVINGS {acc}-S")

        for kind, sign in rng.permutation([("SALE OF EQUIPMENT", 1), ("LOAN DRAWDOWN LN9", 1),
                                           ("CAPITAL INJECTION FROM SHAREHOLDERS", 1),
                                           ("LEGAL SETTLEMENT PAYMENT", -1)])[:2]:
            d = business_day(START + timedelta(days=int(rng.integers(30, (END - START).days - 30))))
            add(acc, d, int(sign) * revenue0 * rng.uniform(0.8, 2.0), kind)

    txns.sort(key=lambda t: (t["booking_date"], t["account_id"]))
    for n, t in enumerate(txns, 1):
        t["txn_id"] = f"T{n:06d}"

    calendar = [{"date": d.isoformat(), "day_of_month": d.day, "day_of_week": d.isoweekday(),
                 "is_weekend": int(d.weekday() >= 5), "is_month_start": int(d.day <= 3),
                 "is_month_end": int((d + timedelta(days=3)).month != d.month)}
                for d in days(START, CALENDAR_END)]

    for fname, rows in (("accounts.csv", accounts), ("transactions.csv", txns), ("calendar.csv", calendar)):
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(accounts)} accounts, {len(txns)} transactions, {len(calendar)} calendar days -> {OUT}")


if __name__ == "__main__":
    main()
