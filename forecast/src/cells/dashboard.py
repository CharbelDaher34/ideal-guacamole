# Dashboard "Try the model" (Dashboards menu): how to upload, the uploaded files, a button that runs the scenario
# Try the model, the chart and the forecast table. Created once: when the setup runs again, it is kept as it is
# (with any changes made to it).
TRY_DASHBOARD = "Try the model"
HOW_TO = """### Try the model on your own clients
1. **Upload** one or more CSV files to the folder `try_model_uploads`: Flow, zone *Try the model*, open the folder,
   **Upload** (or drag the files there). Delete `example_clients.csv` once you have your own.
2. **Run** with the button on the right (scenario *Try the model*, about 30 seconds), then refresh this page.
3. **Read** the chart and the table below: the production model's P10, EV and P90 for the next 3 months.

**File format**, one row per client, month and category:

- Columns: `client_no`, `month` (2026-10), `category`, `amount` (SAR, positive)
- Optional: `segment`, `sector`, `region`, `last_annual_sales`, `number_of_employees`, `balance_end`
- Categories: PAYROLL, RENT, FINANCING, SADAD_GOV, SUPPLIERS, CORP_CARD, OTHER_OUT, RECEIPTS, CARD_SALES, CHEQUES_IN, DEPOSIT_MAT, OTHER_IN
- A client needs 12 months up to the latest month in the files. Scheduled payments are not uploaded: every amount counts as not scheduled.
- Nothing else in the project changes."""

if TRY_DASHBOARD in [d.name for d in project.list_dashboards()]:
    print(f"Dashboard {TRY_DASHBOARD}: already there, kept")
else:
    existing_insights = {(i.type, i.name): i.id for i in project.list_insights()}

    def insight(kind, name, params):
        """The insight (a tile's content): reused if there, else created."""
        if (kind, name) in existing_insights:
            return existing_insights[(kind, name)]
        return project.create_insight({"type": kind, "name": name, "params": params}).insight_id

    def tile(kind, box, insight_id=None, title="YES", params=None, **more):
        base = {"tileType": "INSIGHT", "insightId": insight_id, "insightType": kind, "displayMode": "INSIGHT",
                "clickAction": "DO_NOTHING", "resizeImageMode": "FIT_SIZE", "autoLoad": True}
        if kind == "text":
            base = {"tileType": "TEXT"}
        return {**base, "tileParams": params or {}, "titleOptions": {"showTitle": title},
                "box": dict(zip(("left", "top", "width", "height"), box)), **more}

    scenario_id = next(s["id"] for s in project.list_scenarios() if s["name"] == "Try the model")
    tiles = [
        tile("text", (0, 0, 24, 9), title="NO", params={"text": HOW_TO, "textAlign": "LEFT", "verticalAlign": "TOP"}),
        tile("scenario_run_button", (24, 0, 12, 4),
             insight("scenario_run_button", "Run Try the model", {"scenarioSmartId": scenario_id})),
        tile("managed-folder_content", (24, 4, 12, 5),
             insight("managed-folder_content", "Uploaded files", {"folderSmartId": folders["try_model_uploads"].id})),
        tile("managed-folder_content", (0, 9, 36, 14),
             insight("managed-folder_content", "Forecast chart (net cash flow)",
                     {"folderSmartId": folders["try_model_charts"].id, "filePath": "/forecast_chart.png",
                      "isDirectory": False, "singleFile": True})),
        tile("dataset_table", (0, 23, 36, 12),
             insight("dataset_table", "Forecast of the uploaded clients", {"datasetSmartName": "try_model_forecast"}),
             params={"viewKind": "EXPLORE", "showName": True, "showDescription": False, "showCustomFields": False,
                     "showMeaning": False, "showProgressBar": False}),
    ]
    project.create_dashboard(TRY_DASHBOARD, {"pages": [{"title": TRY_DASHBOARD, "grid": {"tiles": tiles}}],
                                             "listed": True})
    print(f"Dashboard {TRY_DASHBOARD} created (Dashboards menu)")
