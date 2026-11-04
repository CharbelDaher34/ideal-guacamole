from dataikuapi.dss.utils import DSSDatasetSelectionBuilder

QUANTILE_MODEL = """
from lightgbm import LGBMRegressor
clf = LGBMRegressor(objective="quantile", alpha={alpha}, n_estimators=500, learning_rate=0.05, num_leaves=31,
                    min_child_samples=50, verbose=-1)
"""
PINBALL = """
import numpy as np
def score(y_valid, y_pred, eval_df=None, output_df=None, ref_sample_df=None, **kwargs):
    diff = np.asarray(y_valid) - np.asarray(y_pred)
    return float(np.mean(np.maximum({alpha} * diff, ({alpha} - 1) * diff)))
"""
SHARE_BELOW = """
import numpy as np
def score(y_valid, y_pred, eval_df=None, output_df=None, ref_sample_df=None, **kwargs):
    return float(np.mean(np.asarray(y_valid) < np.asarray(y_pred)))
"""
# saved model: (Lab analysis name, quantile or None for the expected value)
VISUAL_MODELS = {"ev_visual": ("EV (visual)", None), "p10_visual": ("P10 (visual)", 0.1),
                 "p90_visual": ("P90 (visual)", 0.9)}
full = DSSDatasetSelectionBuilder().with_all_data_sampling().build()


def lab_model(analysis_name, alpha):
    """Visual ML task on ml_train (tested on ml_test), trained in the Lab; returns the task and its best model."""
    task = project.create_prediction_ml_task(input_dataset="ml_train", target_variable="target_scaled",
                                             prediction_type="REGRESSION", wait_guess_complete=True)
    definition = project.get_analysis(task.analysis_id).get_definition()
    definition.get_raw()["name"] = analysis_name
    definition.save()
    settings = task.get_settings()
    settings.split_params.set_split_explicit(full, full, dataset_name="ml_train", test_dataset_name="ml_test")
    for column in list(settings.get_raw()["preprocessing"]["per_feature"]):
        if column in lib["FEATURES"]:
            settings.use_feature(column)
            handling = settings.get_feature_preprocessing(column)
            if column in lib["CAT_FEATURES"]:       # one column per value
                handling.update(type="CATEGORY", category_handling="DUMMIFY", missing_handling="NONE",
                                missing_impute_with="MODE", dummy_clip="MAX_NB_CATEGORIES", max_nb_categories=100,
                                dummy_drop="AUTO")
            else:                                   # numbers: missing values = the average, then standardised
                handling.update(type="NUMERIC", numerical_handling="REGULAR", missing_handling="IMPUTE",
                                missing_impute_with="MEAN", rescaling="AVGSTD")
        elif column != "target_scaled":
            settings.reject_feature(column)
    settings.disable_all_algorithms()
    if alpha is None:
        settings.set_algorithm_enabled("LIGHTGBM_REGRESSION", True)
        settings.set_algorithm_enabled("XGBOOST_REGRESSION", True)
        settings.set_metric("RMSE")
    else:
        settings.add_custom_python_model(f"LightGBM quantile {alpha}", QUANTILE_MODEL.format(alpha=alpha))
        settings.set_metric(custom_metric=PINBALL.format(alpha=alpha), custom_metric_greater_is_better=False,
                            custom_metric_name=f"Pinball loss {alpha}")
        settings.get_raw()["modeling"]["metrics"]["customMetrics"].append({
            "name": "Share below prediction", "metricCode": SHARE_BELOW, "greaterIsBetter": False,
            "needsProbability": False, "description": f"Should be close to {alpha}"})
    settings.save()
    task.train(session_name="First training")
    ids = task.get_trained_models_ids()
    if len(ids) == 1:
        return task, ids[0]
    rmse = {}
    for model_id in ids:
        metrics = task.get_trained_model_details(model_id).get_performance_metrics()
        rmse[model_id] = next(v for k, v in metrics.items() if k.lower() == "rmse")
    return task, min(rmse, key=rmse.get)


saved_models = {}
for name, (analysis_name, alpha) in VISUAL_MODELS.items():
    started = time.time()
    task, best = lab_model(analysis_name, alpha)
    deployed = task.deploy_to_flow(best, name, "ml_train", test_dataset="ml_test")
    saved_models[name] = deployed["savedModelId"]
    print(f"{analysis_name}: trained in the Lab and deployed as saved model {name} ({time.time() - started:.0f}s)")

# Evaluate recipes (test on the last 6 months, kept in a model evaluation store) and Score recipes
for name, model_id in saved_models.items():
    store = project.create_model_evaluation_store(f"{name}_evaluations")
    for out in (f"eval_{name}_scored", f"eval_{name}_metrics", f"scored_{name}"):
        new_dataset(out)
    builder = project.new_recipe("evaluation", f"evaluate_{name}")
    builder.with_input_model(model_id)
    builder.with_input("ml_test")
    builder.with_output(f"eval_{name}_scored")
    builder.with_output_metrics(f"eval_{name}_metrics")
    builder.with_output_evaluation_store(store.mes_id)
    evaluate = builder.create()
    builder = project.new_recipe("prediction_scoring", f"score_{name}")
    builder.with_input_model(model_id)
    builder.with_input("ml_scoring")
    builder.with_existing_output(f"scored_{name}")
    score = builder.create()
    for created in (evaluate, score):
        created.compute_schema_updates().apply()
    score.rename(f"score_{name}")         # Dataiku names a Score recipe after its input
    print("recipes", f"evaluate_{name}", f"score_{name}")

for name, (inputs, outputs, _) in FLOW_AFTER_VISUAL.items():
    recipe("python", name, inputs, outputs, RECIPES[name])

zones = {}
def to_zone(zone_name, item):
    if zone_name not in zones:
        zones[zone_name] = flow.create_zone(zone_name)
    zones[zone_name].add_item(item)

for name in RAW:
    to_zone("Raw data", project.get_dataset(name))
for table in lib["TABLES"]:
    to_zone("Staging", project.get_dataset(table))
to_zone("Data prep", project.get_dataset("category_rules"))
for _, outputs, zone_name in list(FLOW.values()) + list(FLOW_AFTER_VISUAL.values()):
    for out in outputs:
        to_zone(zone_name, models if out == "models" else project.get_dataset(out))
for name, model_id in saved_models.items():
    to_zone("Training (visual)", project.get_saved_model(model_id))
    for out in (f"eval_{name}_scored", f"eval_{name}_metrics"):
        to_zone("Training (visual)", project.get_dataset(out))
    to_zone("Inference", project.get_dataset(f"scored_{name}"))
stores = project.list_model_evaluation_stores()
for store in stores:
    to_zone("Training (visual)", store)
print("Flow ready:", ", ".join(zones))


# Production runs. Code models (the models folder), visual saved models and evaluation stores are only rebuilt
# when built by name, so the monthly forecast never retrains by accident.
def build_step(n, label, items, mode):
    return {"id": f"step_{n}", "type": "build_flowitem", "name": label, "enabled": True,
            "runConditionType": "RUN_IF_STATUS_MATCH", "runConditionStatuses": ["SUCCESS", "WARNING"],
            "params": {"builds": [{"type": t, "itemId": i, "partitionsSpec": ""} for t, i in items], "jobType": mode,
                       "proceedOnFailure": False, "handleWarningsAs": "WARNING"}}


datasets = lambda *names: [("DATASET", n) for n in names]
SCENARIOS = {
    "Monthly forecast": (
        "After each warehouse load: check it, then forecast the next 3 months with the current models",
        dict(day=2, hour=7), [
            ("Check the latest warehouse load", datasets("load_status"), "NON_RECURSIVE_FORCED_BUILD"),
            ("Stage, prepare and forecast", datasets("predictions", "predictions_visual", "final_forecast"),
             "RECURSIVE_FORCED_BUILD"),
        ]),
    "Retrain and test the models": (
        "Retrain both ways (code and visual), test them on the last 6 months, compare, and forecast with them",
        dict(day=5, hour=7, repeat_every=3), [
            ("Check the latest warehouse load", datasets("load_status"), "NON_RECURSIVE_FORCED_BUILD"),
            ("Stage and prepare the training data",
             datasets("training_rows", "scoring_rows", "ml_train", "ml_test", "ml_scoring"), "RECURSIVE_FORCED_BUILD"),
            ("Retrain the code models", [("MANAGED_FOLDER", models.id)], "NON_RECURSIVE_FORCED_BUILD"),
            ("Test the code models", datasets("test_results"), "NON_RECURSIVE_FORCED_BUILD"),
            ("Retrain the visual models", [("SAVED_MODEL", m) for m in saved_models.values()],
             "NON_RECURSIVE_FORCED_BUILD"),
            ("Test the visual models", [("MODEL_EVALUATION_STORE", s.mes_id) for s in stores],
             "NON_RECURSIVE_FORCED_BUILD"),
            ("Compare and forecast with the new models", datasets("model_comparison", "final_forecast"),
             "RECURSIVE_BUILD"),
        ]),
}
for name, (description, schedule, steps) in SCENARIOS.items():
    scenario = project.create_scenario(name, "step_based", {
        "active": False, "description": description,
        "params": {"steps": [build_step(n, *step) for n, step in enumerate(steps)]}})
    settings = scenario.get_settings()
    settings.add_monthly_trigger(timezone="Asia/Riyadh", **schedule)
    settings.raw_triggers[-1]["active"] = False    # switch on in production (Automation node)
    try:
        settings.save()
        note = "monthly trigger added, switched off"
    except Exception as e:                         # licences without time-based triggers (Free Edition)
        if "License" not in str(e):
            raise
        note = f"no trigger (not in this licence): add a monthly trigger ({schedule}) on the Automation node"
    print(f"Scenario ready: {name} ({note})")
