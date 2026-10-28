"""Synthetic corporate-banking data: 30 corporate accounts with 36 months of transactions.

Writes data/accounts.csv and data/transactions.csv. Fully made up, seeded so reruns match.
Each company has a sector seasonality, a growth trend, payroll/rent/loan/tax/capex patterns,
and a few are deliberately deteriorating so the forecast has liquidity risks to find.
"""
import csv
from datetime import date, timedelta
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data"
START, MONTHS = date(2023, 10, 1), 36          # history Oct 2023 .. Sep 2026
rng = np.random.default_rng(42)

# sector -> 12 monthly multipliers on revenue (Jan..Dec)
SEASONALITY = {
    "Retail":        [0.85, 0.80, 0.90, 0.95, 1.00, 0.95, 0.95, 1.00, 0.95, 1.05, 1.25, 1.45],
    "Manufacturing": [0.95, 1.00, 1.05, 1.05, 1.00, 1.00, 0.85, 0.80, 1.05, 1.05, 1.05, 1.05],
    "Construction":  [0.70, 0.75, 0.95, 1.10, 1.20, 1.20, 1.15, 1.10, 1.10, 1.00, 0.90, 0.75],
    "Hospitality":   [0.70, 0.70, 0.85, 0.95, 1.05, 1.30, 1.45, 1.45, 1.10, 0.90, 0.75, 0.90],
    "Agriculture":   [0.60, 0.60, 0.75, 0.85, 0.95, 1.00, 1.10, 1.30, 1.55, 1.45, 0.95, 0.65],
    "Technology":    [0.95, 0.95, 1.10, 0.95, 0.95, 1.10, 0.90, 0.85, 1.05, 1.00, 1.00, 1.20],
    "Healthcare":    [1.05, 1.00, 1.00, 1.00, 1.00, 0.95, 0.95, 0.95, 1.00, 1.05, 1.05, 1.05],
    "Logistics":     [0.90, 0.90, 1.00, 1.00, 1.00, 1.00, 0.95, 0.95, 1.05, 1.10, 1.15, 1.25],
}
NAMES = [
    "Cedar Steelworks", "Phoenicia Retail Group", "Byblos Construction", "Levant Logistics",
    "Mount Lebanon Hotels", "Bekaa Agro Holdings", "Beirut Digital Systems", "Orient Medical Supplies",
    "Sidon Port Services", "Tripoli Textiles", "Zahle Wineries", "Mediterranean Foods Co",
    "Atlas Engineering", "Horizon Pharma", "Coastal Resorts SAL", "Northern Cement",
    "Union Electronics", "Prime Freight Lines", "Golden Harvest Farms", "Skyline Developers",
    "Nova Software House", "Crescent Hospitals", "Delta Packaging", "Summit Retail Brands",
    "Blue Bay Hospitality", "Vertex Machinery", "Green Valley Dairy", "Metro Couriers",
    "Apex Clinics Group", "Silverline Construction",
]
SECTORS = list(SEASONALITY)
SUPPLIERS = ["Global Supply Ltd", "Prime Materials", "Euro Components", "Gulf Trading", "Local Wholesale"]
CUSTOMERS = ["Client A", "Client B", "Client C", "Client D", "Client E", "Client F", "Distributor X"]
DETERIORATING = {3, 9, 16, 23, 29}               # account indexes whose business slides


def month_start(i):
    y, m = divmod(START.month - 1 + i, 12)
    return date(START.year + y, m + 1, 1)


def day_in(month, d):
    next_month = (month.replace(day=28) + timedelta(days=4)).replace(day=1)
    return min(month + timedelta(days=d - 1), next_month - timedelta(days=1))


def main():
    OUT.mkdir(exist_ok=True)
    accounts, txns = [], []
    for i, name in enumerate(NAMES):
        sector = SECTORS[i % len(SECTORS)]
        segment = ["Large Corporate", "Mid-Market", "SME"][i % 3]
        scale = {"Large Corporate": 2_500_000, "Mid-Market": 650_000, "SME": 140_000}[segment]
        revenue0 = scale * rng.uniform(0.7, 1.3)
        growth = rng.uniform(-0.02, 0.15) if i not in DETERIORATING else rng.uniform(-0.35, -0.20)
        cost_ratio = rng.uniform(0.68, 0.82) if i not in DETERIORATING else rng.uniform(0.90, 0.98)
        payroll_share, rent = rng.uniform(0.25, 0.40), revenue0 * rng.uniform(0.02, 0.05)
        loan = revenue0 * rng.uniform(0.03, 0.08) if rng.random() < 0.7 else 0.0
        acc = f"CA{100001 + i}"
        accounts.append({
            "account_id": acc, "company_name": name, "sector": sector, "segment": segment,
            "country": "Lebanon", "currency": "USD", "relationship_manager": ["R. Haddad", "M. Khoury", "L. Saad", "N. Aoun"][i % 4],
            "opening_date": f"{2008 + i % 14}-0{1 + i % 9}-15",
            "opening_balance": round(revenue0 * rng.uniform(0.6, 1.5), 2),
            "overdraft_limit": round(revenue0 * rng.uniform(0.2, 0.5), 2),
        })
        n = 0
        for mi in range(MONTHS):
            month = month_start(mi)
            level = revenue0 * (1 + growth) ** (mi / 12) * SEASONALITY[sector][month.month - 1]
            revenue = level * rng.lognormal(0, 0.08)

            def add(day, amount, direction, category, cp):
                nonlocal n
                n += 1
                txns.append({"txn_id": f"{acc}-{n:05d}", "account_id": acc, "booking_date": day_in(month, day).isoformat(),
                             "amount": round(abs(amount), 2), "direction": direction, "category": category, "counterparty": cp})

            k = int(rng.integers(5, 11))                         # customer receipts
            for share, d in zip(rng.dirichlet(np.ones(k)), rng.integers(1, 29, k)):
                add(int(d), revenue * share, "CREDIT", "CUSTOMER_RECEIPT", str(rng.choice(CUSTOMERS)))
            opex = revenue * cost_ratio
            add(25, opex * payroll_share, "DEBIT", "PAYROLL", "Staff salaries")
            add(1, rent, "DEBIT", "RENT", "Landlord")
            k = int(rng.integers(3, 8))                          # supplier payments
            for share, d in zip(rng.dirichlet(np.ones(k)), rng.integers(1, 29, k)):
                add(int(d), opex * (1 - payroll_share) * share, "DEBIT", "SUPPLIER_PAYMENT", str(rng.choice(SUPPLIERS)))
            if loan:
                add(10, loan, "DEBIT", "LOAN_REPAYMENT", "Bank loan")
            if month.month in (1, 4, 7, 10):
                add(20, revenue * 3 * rng.uniform(0.03, 0.06), "DEBIT", "TAX", "Tax authority")
            if rng.random() < 0.08:
                add(int(rng.integers(5, 25)), revenue * rng.uniform(0.3, 1.0), "DEBIT", "CAPEX", "Equipment vendor")
            if rng.random() < 0.05:
                add(int(rng.integers(5, 25)), revenue * rng.uniform(0.5, 1.5), "CREDIT", "LOAN_DISBURSEMENT", "Bank loan")
            if rng.random() < 0.15:
                add(int(rng.integers(1, 29)), revenue * rng.uniform(0.01, 0.05), "CREDIT", "INTEREST_OTHER", "Other income")

    txns.sort(key=lambda t: (t["account_id"], t["booking_date"], t["txn_id"]))
    for fname, rows in (("accounts.csv", accounts), ("transactions.csv", txns)):
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"{len(accounts)} accounts, {len(txns)} transactions -> {OUT}")


if __name__ == "__main__":
    main()
