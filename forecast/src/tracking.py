"""Model tracking and promotion (project library: python/tracking.py), for training and the lab.

Every training run is one model with its own model ID, recorded in Dataiku's Experiment Tracking (MLflow; artifacts in
the managed folder "experiments"): its forecaster and settings, the cut-off, test scores on the last 6 periods of
30 days, the backtest forecaster (trained without those periods, for champion / challenger tests) and one MLflow model (the final
forecaster, which returns P10, EV and P90). It is deployed as a new version (version ID = model ID) of the saved model
cash_forecast (interface 3, cashflow.SAVED_MODEL). The production model is the active version; promoting a model
activates it, and the nightly forecast always uses the active version.
Lab tries (notebook forecast_lab) go to their own experiment, cashflow_lab, in the folder lab_experiments. They are
never deployed.
"""
import inspect
import os
import pickle
import sys
import tempfile

import pandas as pd

from cashflow import SAVED_MODEL

EXPERIMENT, FOLDER = "cashflow_forecast", "experiments"
LAB_EXPERIMENT, LAB_FOLDER = "cashflow_lab", "lab_experiments"
# The promotion rules: project variables (change them in the project's variables, no code change), with these
# defaults. A candidate is promoted only if all hold:
PROMOTION_RULES = {
    "promotion_min_coverage": 0.70,      # its P10-P90 coverage is at least this...
    "promotion_max_coverage": 0.90,      # ...and at most this (80% is the aim)
    "promotion_beat_simple_guess": True,  # its EV error is lower than the simple guess's (same as the month before)
    "promotion_min_gain": 0.0,           # its EV error is lower than the production model's by more than this share
                                         # (0.02: at least 2% lower). A tie keeps the production model.
}


def _project():
    import dataiku
    return dataiku.api_client().get_default_project()


def _mlflow(folder):
    """Experiment Tracking in the folder. By its ID: outside a recipe (the lab notebook), setup_mlflow does not resolve
    a folder name."""
    import dataiku
    return _project().setup_mlflow(managed_folder=dataiku.Folder(folder).get_id())


def _experiment_name(mlflow, experiment, folder):
    """The experiment, kept in the folder. If an experiment of that name points to another folder (the folder was
    deleted and created again, for example when the flow was deleted by hand), its files are gone: use
    <experiment>_<folder id> instead."""
    import dataiku
    folder_id = dataiku.Folder(folder).get_id()
    found = mlflow.get_experiment_by_name(experiment)
    if found is None or f"//{folder_id}/" in (found.artifact_location or ""):
        return experiment
    return f"{experiment}_{folder_id}"


def new_model_id(cutoff):
    """CFF_<cut-off day>_<training time>, for example CFF_20261031_20261104T1215 (also the saved model version)."""
    return f"CFF_{cutoff:%Y%m%d}_{pd.Timestamp.now():%Y%m%dT%H%M}"


def promotion_rules():
    """The promotion rules: the project variables PROMOTION_RULES names, or their defaults."""
    import dataiku
    variables = dataiku.get_custom_variables()
    rules = {}
    for name, default in PROMOTION_RULES.items():
        value = variables.get(name, default)
        rules[name] = (str(value).strip().lower() in ("true", "1", "yes")) if isinstance(default, bool) else float(value)
    return rules


def saved_model(name=SAVED_MODEL):
    project = _project()
    return project.get_saved_model(next(m["id"] for m in project.list_saved_models() if m["name"] == name))


def forecast_model(forecaster):
    """An MLflow model around a trained forecaster. The nightly forecast unwraps it and calls the forecaster
    (forecast.production_forecaster); called as an MLflow model, it takes daily flows (interface 1) and returns the
    forecast of the 3 windows after the last day it was given, without client details."""
    import mlflow.pyfunc
    from models import forecast

    class ForecastModel(mlflow.pyfunc.PythonModel):
        def __init__(self, forecaster):
            self.forecaster = forecaster

        def predict(self, context, model_input, params=None):
            cutoff = pd.to_datetime(model_input["day"]).max().normalize()
            return forecast(self.forecaster, model_input, None, cutoff)

    return ForecastModel(forecaster)


def _carry_code(forecaster):
    """The models carry their own copy of the library code (pickled by value: cashflow, models, tracking and the
    forecaster's own module, forecasters.py for example), so they keep working when the library changes."""
    import cloudpickle
    import cashflow
    import models
    own = sys.modules.get(type(forecaster).__module__)
    for module in (cashflow, models, sys.modules[__name__], own):
        if module is not None and module.__name__ != "__main__":
            cloudpickle.register_pickle_by_value(module)


def _record(mlflow, params, metrics, backtest):
    """A run's settings, test scores and backtest forecaster."""
    import cloudpickle
    mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "forecaster.pkl"), "wb") as f:
            cloudpickle.dump(backtest, f)
        mlflow.log_artifacts(tmp, artifact_path="backtest")


def log_model(model_id, params, metrics, backtest, final):
    """Record one training run in Experiment Tracking, then deploy its final forecaster as version model_id of the
    saved model cash_forecast (not activated: the promotion does that)."""
    _carry_code(final)
    with _mlflow(FOLDER) as mlflow:
        mlflow.set_experiment(_experiment_name(mlflow, EXPERIMENT, FOLDER))
        with mlflow.start_run(run_name=model_id) as run:
            mlflow.set_tag("model_id", model_id)
            _record(mlflow, params, metrics, backtest)
            where = "name" if int(mlflow.__version__.split(".")[0]) >= 3 else "artifact_path"   # MLflow 3 or 2
            mlflow.pyfunc.log_model(python_model=forecast_model(final), **{where: "forecaster"})
            run_id = run.info.run_id
    extension = _project().get_mlflow_extension()
    code_env = _project().get_recipe("train_and_test").get_settings().get_recipe_params()["envSelection"]["envName"]
    extension.set_run_inference_info(run_id=run_id, prediction_type="REGRESSION", code_env_name=code_env)
    # which model of the run: model_name in DSS 15 (MLflow 3 models), model_sub_folder in DSS 14 (the artifact path)
    which = "model_name" if "model_name" in inspect.signature(extension.deploy_run_model).parameters else "model_sub_folder"
    model = saved_model()
    before = (model.get_active_version() or {}).get("id")
    extension.deploy_run_model(run_id, model.id, version_id=model_id, activate=False, skip_expensive_reports=True,
                               **{which: "forecaster"})
    if before and (model.get_active_version() or {}).get("id") != before:
        model.set_active_version(before)       # DSS 14 activates every deployed version: only a promotion may


def log_lab_run(run_name, params, metrics, backtest):
    """Record a lab try in Experiment Tracking (experiment cashflow_lab, folder lab_experiments): its settings, test
    scores and forecaster. Never deployed. Returns the run ID."""
    _carry_code(backtest)
    with _mlflow(LAB_FOLDER) as mlflow:
        mlflow.set_experiment(_experiment_name(mlflow, LAB_EXPERIMENT, LAB_FOLDER))
        with mlflow.start_run(run_name=run_name) as run:
            _record(mlflow, params, metrics, backtest)
            return run.info.run_id


def runs():
    """Every training run, newest first: model_id, run_id, trained_at, its settings (params.*) and metrics."""
    with _mlflow(FOLDER) as mlflow:
        # sorted here: DSS 14 cannot sort runs by start time in the search
        found = mlflow.search_runs(experiment_names=[_experiment_name(mlflow, EXPERIMENT, FOLDER)])
    if found.empty:
        return pd.DataFrame(columns=["model_id", "run_id", "trained_at"])
    found = found.sort_values("start_time", ascending=False).reset_index(drop=True)
    return found.rename(columns={"tags.model_id": "model_id", "start_time": "trained_at"})


def latest_model_id():
    """The most recently trained model."""
    found = runs()
    if found.empty:
        raise ValueError("No model trained yet: run the scenario Retrain, test and promote")
    return found.iloc[0]["model_id"]


def production_model_id():
    """The production model: the active version of the saved models, if it was ever promoted (a version Dataiku
    activated on its own when the first one was deployed does not count until the promotion)."""
    model_id = (saved_model().get_active_version() or {}).get("id")
    return model_id if model_id in promoted_model_ids() else ""


def promoted_model_ids():
    """Every model ID promoted so far (read through the API: also works in a recipe that writes promotion_log)."""
    try:
        log = _project().get_dataset("promotion_log")
        columns = [c["name"] for c in log.get_schema()["columns"]]
        model, decision = columns.index("candidate_model_id"), columns.index("decision")
        return {row[model] for row in log.iter_rows() if row[decision] == "PROMOTE"}
    except Exception:            # no promotion yet
        return set()


def load_backtest(model_id):
    """The backtest forecaster of a training run (trained without the last 6 periods, for champion / challenger)."""
    found = runs()
    match = found[found["model_id"] == model_id]
    if match.empty:
        raise ValueError(f"Model {model_id} not found in Experiment Tracking ({EXPERIMENT})")
    with _mlflow(FOLDER) as mlflow:
        path = mlflow.artifacts.download_artifacts(run_id=match.iloc[0]["run_id"], artifact_path="backtest")
    with open(os.path.join(path, "forecaster.pkl"), "rb") as f:
        return pickle.load(f)


def decide(candidate, production, rules=None):
    """Champion / challenger: the decision (PROMOTE or KEEP) and its reason, from the test scores of the candidate and
    of the production model (models.scores), with the promotion rules (promotion_rules: project variables). production
    None: no production model to compare with, the candidate is promoted (the forecast needs one) with its failed
    checks as a warning."""
    rules = rules or promotion_rules()
    low, high = rules["promotion_min_coverage"], rules["promotion_max_coverage"]
    checks = []
    if not low <= candidate["coverage_p10_p90"] <= high:
        checks.append(f"coverage {candidate['coverage_p10_p90']:.1%} outside {low:.0%}-{high:.0%}")
    if rules["promotion_beat_simple_guess"] and candidate["wape_model"] >= candidate["wape_baseline"]:
        checks.append(f"does not beat the simple guess ({candidate['wape_model']:.1%} vs "
                      f"{candidate['wape_baseline']:.1%})")
    if production is None:
        return "PROMOTE", (f"no production model to compare with, EV error {candidate['wape_model']:.2%}, coverage "
                           f"{candidate['coverage_p10_p90']:.1%}" + (f" (warning: {'; '.join(checks)})" if checks else ""))
    if checks:
        return "KEEP", f"candidate {checks[0]}"
    needed = production["wape_model"] * (1 - rules["promotion_min_gain"])
    if candidate["wape_model"] >= needed:
        gain = f" by more than {rules['promotion_min_gain']:.0%}" if rules["promotion_min_gain"] else ""
        return "KEEP", (f"candidate EV error {candidate['wape_model']:.2%} not lower than production "
                        f"{production['wape_model']:.2%}{gain}")
    return "PROMOTE", (f"candidate EV error {candidate['wape_model']:.2%} vs production {production['wape_model']:.2%}, "
                       f"coverage {candidate['coverage_p10_p90']:.1%}")


def apply_promotion():
    """Apply the latest decision in promotion_log: on PROMOTE, activate the candidate's version of the saved model.
    Run by a scenario step after the champion / challenger test."""
    import dataiku
    log = dataiku.Dataset("promotion_log").get_dataframe()
    last = log.sort_values("decided_at").iloc[-1]
    if last["decision"] == "PROMOTE":
        saved_model().set_active_version(last["candidate_model_id"])
        print(f"Promoted {last['candidate_model_id']}: {last['reason']}")
    else:
        print(f"Kept the production model: {last['reason']}")
    return production_model_id()
