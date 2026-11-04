started = time.time()
steps = [  # first run: what the scenario "Retrain and test the models" does, without redoing the data prep
    ("NON_RECURSIVE_FORCED_BUILD", [(models.id, "MANAGED_FOLDER")]),
    ("RECURSIVE_BUILD", [("test_results", None), ("predictions", None), ("predictions_visual", None)]),
    ("NON_RECURSIVE_FORCED_BUILD", [(s.mes_id, "MODEL_EVALUATION_STORE") for s in stores]),
    ("NON_RECURSIVE_FORCED_BUILD", [("model_comparison", None), ("final_forecast", None)]),
]
for mode, items in steps:
    job = project.new_job(mode).with_auto_update_schema_before_each_recipe_run(True)
    for item, kind in items:
        job.with_output(item, object_type=kind)
    job.start_and_wait()
print(f"Models trained and tested, forecast built in {(time.time() - started) / 60:.1f} minutes")
