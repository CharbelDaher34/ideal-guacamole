# Category from the bank description (first matching rule wins), flow from the sign of the amount, and
# excluded = 1 for what is not repeatable cash flow (internal transfers, one-off deals).
import dataiku

RULES = [
    ("INTERNAL TRF", "internal_transfer"),
    ("SALE OF", "one_off"), ("LOAN DRAWDOWN", "one_off"), ("CAPITAL INJECTION", "one_off"),
    ("SETTLEMENT PAYMENT", "one_off"),
    ("TRF FROM", "customer_receipt"), ("POS SETTLEMENT", "card_sales"),
    ("PAYMENT TO", "supplier"), ("SALARY", "payroll"), ("RENT", "rent"), ("LOAN INSTALLMENT", "loan"),
    ("BILL", "utilities"), ("VAT PAYMENT", "tax"), ("FEE", "bank_fee"),
]


def category(description):
    return next((cat for text, cat in RULES if text in description), "other")


tx = dataiku.Dataset("transactions").get_dataframe()
tx["category"] = tx["description"].map(category)
tx["flow"] = (tx["amount"] > 0).map({True: "IN", False: "OUT"})
tx["excluded"] = tx["category"].isin(["internal_transfer", "one_off"]).astype(int)
tx = tx.rename(columns={"booking_date": "date"})

dataiku.Dataset("transactions_clean").write_with_schema(tx)
