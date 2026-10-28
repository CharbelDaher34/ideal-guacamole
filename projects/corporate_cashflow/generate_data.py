"""Synthetic corporate-banking data: 10 companies, 24 months of transactions (Oct 2024 - Sep 2026).

Writes data/accounts.csv and data/transactions.csv. Fully made up, seeded so reruns match.
Revenue follows the sector's season and a yearly trend; costs (payroll, suppliers, rent) are steady,
taxes are paid every quarter. Three companies are in trouble so the forecast has risks to find.
"""
import csv
from datetime import date
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"
START, MONTHS = date(2024, 10, 1), 24
rng = np.random.default_rng(7)

# sector -> revenue multiplier per calendar month (Jan..Dec)
SEASON = {
    "Retail":        [0.85, 0.80, 0.90, 0.95, 1.00, 0.95, 0.95, 1.00, 0.95, 1.05, 1.25, 1.45],
    "Hospitality":   [0.70, 0.70, 0.85, 0.95, 1.05, 1.30, 1.45, 1.45, 1.10, 0.90, 0.75, 0.90],
    "Agriculture":   [0.60, 0.60, 0.75, 0.85, 0.95, 1.00, 1.10, 1.30, 1.55, 1.45, 0.95, 0.65],
    "Construction":  [0.70, 0.75, 0.95, 1.10, 1.20, 1.20, 1.15, 1.10, 1.10, 1.00, 0.90, 0.75],
    "Other":         [1.00] * 12,
}
# name, sector, monthly revenue, costs as share of revenue, revenue trend per year,
# balance today (end of Sep 2026), overdraft limit
COMPANIES = [
    ("Cedar Steelworks",       "Other",        900_000, 0.85,  0.06, 1_200_000, 300_000),
    ("Phoenicia Retail Group", "Retail",       600_000, 0.86,  0.08,   450_000, 150_000),
    ("Levant Logistics",       "Other",        450_000, 0.95, -0.25,    30_000,  60_000),
    ("Mount Lebanon Hotels",   "Hospitality",  350_000, 0.88,  0.04,    40_000, 100_000),
    ("Bekaa Agro Holdings",    "Agriculture",  250_000, 0.82,  0.05,   180_000,  80_000),
    ("Beirut Digital Systems", "Other",        300_000, 0.75,  0.12,   650_000, 100_000),
    ("Tripoli Textiles",       "Other",        200_000, 0.95, -0.30,   -20_000,  40_000),
    ("Byblos Construction",    "Construction", 500_000, 0.86,  0.03,   220_000, 150_000),
    ("Orient Medical Supplies", "Other",       400_000, 0.84,  0.07,   380_000, 120_000),
    ("Sidon Port Services",    "Other",        280_000, 0.93,  0.00,    15_000,  50_000),
]


def month_start(i):
    y, m = divmod(START.month - 1 + i, 12)
    return date(START.year + y, m + 1, 1)


def main():
    OUT.mkdir(exist_ok=True)
    accounts, txns = [], []
    for i, (name, sector, revenue0, cost_ratio, trend, balance, overdraft) in enumerate(COMPANIES):
        acc = f"CA{1001 + i}"
        accounts.append({"account_id": acc, "company_name": name, "sector": sector,
                         "relationship_manager": ["R. Haddad", "M. Khoury", "L. Saad"][i % 3],
                         "current_balance": balance, "overdraft_limit": overdraft})
        costs = revenue0 * cost_ratio                # steady monthly costs, set by the starting revenue
        for mi in range(MONTHS):
            month = month_start(mi)
            revenue = revenue0 * (1 + trend) ** (mi / 12) * SEASON[sector][month.month - 1] * rng.normal(1, 0.05)
            rows = [(day, revenue * share, "CREDIT", "CUSTOMER_RECEIPT")
                    for day, share in zip((5, 15, 25), rng.dirichlet([4, 4, 4]))]
            rows += [(25, costs * 0.40, "DEBIT", "PAYROLL"), (1, costs * 0.10, "DEBIT", "RENT"),
                     (10, costs * 0.25 * rng.normal(1, 0.1), "DEBIT", "SUPPLIER_PAYMENT"),
                     (20, costs * 0.25 * rng.normal(1, 0.1), "DEBIT", "SUPPLIER_PAYMENT")]
            if month.month in (1, 4, 7, 10):
                rows.append((20, revenue0 * 0.15, "DEBIT", "TAX"))
            for day, amount, direction, category in rows:
                txns.append({"txn_id": f"T{len(txns) + 1:05d}", "account_id": acc,
                             "booking_date": month.replace(day=day).isoformat(), "amount": round(amount, 2),
                             "direction": direction, "category": category})

    for fname, rows in (("accounts.csv", accounts), ("transactions.csv", txns)):
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(accounts)} accounts, {len(txns)} transactions -> {OUT}")


if __name__ == "__main__":
    main()
