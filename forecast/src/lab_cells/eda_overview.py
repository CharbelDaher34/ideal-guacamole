# Money in, money out and net cash flow per client and month, by segment
monthly = (flows.pivot_table(index=["client_no", "month"], columns="direction", values="amount", aggfunc="sum")
           .reindex(columns=["IN", "OUT"]).fillna(0.0).reset_index())
monthly["NET"] = monthly["IN"] - monthly["OUT"]
monthly = monthly.merge(rows.drop_duplicates("client_no")[["client_no", "segment", "sole_proprietor"]], on="client_no")
overview = monthly.groupby("segment").agg(
    clients=("client_no", "nunique"), months=("month", "size"), money_in=("IN", "mean"), money_out=("OUT", "mean"),
    net=("NET", "mean"), negative_months=("NET", lambda s: (s < 0).mean())).round({"money_in": 0, "money_out": 0,
                                                                                  "net": 0, "negative_months": 3})
display(overview)
print("Sole proprietors:", int(rows.drop_duplicates("client_no")["sole_proprietor"].astype(str).str.lower()
                               .eq("true").sum()), "of", rows["client_no"].nunique(), "clients")
