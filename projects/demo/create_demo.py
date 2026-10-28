"""Demo flow: orders.csv -> [Python] -> orders_clean (Postgres) -> [SQL] -> customer_totals (Postgres)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dssproj import Project, col  # noqa: E402

p = Project("DEV_DEMO", "Dev demo (created by code)", "Orders flow built from projects/demo/create_demo.py")

p.csv_dataset("orders_raw", HERE / "orders.csv", [
    col("order_id", "bigint"), col("customer"), col("country"),
    col("order_date"), col("amount", "double"), col("status"),
])

p.pg_dataset("orders_clean", [
    col("order_id", "bigint"), col("customer"), col("country"),
    col("order_date", "date"), col("amount", "double"), col("status"), col("is_paid", "boolean"),
])
p.python_recipe("compute_orders_clean", ["orders_raw"], ["orders_clean"], """
import dataiku
import pandas as pd

df = dataiku.Dataset("orders_raw").get_dataframe()
df = df[df["status"] != "cancelled"].copy()
df["order_date"] = pd.to_datetime(df["order_date"])
df["is_paid"] = df["status"] == "paid"

dataiku.Dataset("orders_clean").write_with_schema(df)
""")

p.pg_dataset("customer_totals", [
    col("customer"), col("country"), col("orders", "bigint"), col("paid_amount", "double"),
])
p.sql_recipe("compute_customer_totals", ["orders_clean"], ["customer_totals"], """
SELECT "customer",
       "country",
       COUNT(*)                                       AS "orders",
       SUM(CASE WHEN "is_paid" THEN "amount" ELSE 0 END) AS "paid_amount"
FROM "${projectKey}_orders_clean"
GROUP BY "customer", "country"
ORDER BY "paid_amount" DESC
""")

if __name__ == "__main__":
    p.deploy()
