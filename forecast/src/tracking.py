"""Model tracking and promotion (project library: python/tracking.py), for training and the lab.

Every training run is one model with its own model ID, recorded in Dataiku's Experiment Tracking (MLflow; artifacts in
the managed folder "experiments"): its forecaster and settings, the cut-off, test scores on the last 6 months, the
backtest forecaster (trained without those months, for champion / challenger tests) and three MLflow models (EV, P10,
P90). Those three are deployed as a new version (version ID = model ID) of the saved models forecast_ev, forecast_p10
and forecast_p90 (interface 3, cashflow.SAVED_MODELS). The production model is the active version; promoting a model
activates it in all three, and the Score recipes always use the active version.
Lab tries (notebook forecast_lab) go to their own experiment, cashflow_lab, in the folder lab_experiments. They are
never deployed.
"""
import inspect
import os
import pickle
import sys
import tempfile

import pandas as pd

from cashflow import SAVED_MODELS

EXPERIMENT, FOLDER = "cashflow_forecast", "experiments"
LAB_EXPERIMENT, LAB_FOLDER = "cashflow_lab", "lab_experiments"
MIN_COVERAGE, MAX_COVERAGE = 0.70, 0.90     # a model is only promoted if its P10-P90 coverage is in this range


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
    """CFF_<cut-off month>_<training time>, for example CFF_202610_20261104T1215 (also the saved model version)."""
    return f"CFF_{cutoff:%Y%m}_{pd.Timestamp.now():%Y%m%dT%H%M}"


def saved_model(name):
    project = _project()
    return project.get_saved_model(next(m["id"] for m in project.list_saved_models() if m["name"] == name))


def forecast_model(forecaster, column):
    """An MLflow model around a trained forecaster: takes model rows, returns one forecast column (ev, p10 or p90)
    in SAR."""
    import mlflow.pyfunc
    from models import forecast

    class ForecastModel(mlflow.pyfunc.PythonModel):
        def __init__(self, forecaster, column):
            self.forecaster, self.column = forecaster, column

        def predict(self, context, model_input, params=None):
            return forecast(self.forecaster, model_input)[self.column].values

    return ForecastModel(forecaster, column)


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


def log_model(model_id, params, metrics, backtest, final, reference_dataset):
    """Record one training run in Experiment Tracking, then deploy its final forecaster as version model_id of the
    three saved models (not activated: the promotion does that)."""
    _carry_code(final)
    with _mlflow(FOLDER) as mlflow:
        mlflow.set_experiment(_experiment_name(mlflow, EXPERIMENT, FOLDER))
        with mlflow.start_run(run_name=model_id) as run:
            mlflow.set_tag("model_id", model_id)
            _record(mlflow, params, metrics, backtest)
            where = "name" if int(mlflow.__version__.split(".")[0]) >= 3 else "artifact_path"   # MLflow 3 or 2
            for column in SAVED_MODELS:
                mlflow.pyfunc.log_model(python_model=forecast_model(final, column), **{where: column})
            run_id = run.info.run_id
    extension = _project().get_mlflow_extension()
    code_env = _project().get_recipe("train_and_test").get_settings().get_recipe_params()["envSelection"]["envName"]
    extension.set_run_inference_info(run_id=run_id, prediction_type="REGRESSION", code_env_name=code_env,
                                     target="target")
    # which model of the run: model_name in DSS 15 (MLflow 3 models), model_sub_folder in DSS 14 (the artifact path)
    which = "model_name" if "model_name" in inspect.signature(extension.deploy_run_model).parameters else "model_sub_folder"
    for column, saved_model_name in SAVED_MODELS.items():
        model = saved_model(saved_model_name)
        before = (model.get_active_version() or {}).get("id")
        # evaluated on the reference sample: gives the version its input schema (the Score recipes need it)
        extension.deploy_run_model(run_id, model.id, version_id=model_id,
                                   evaluation_dataset=reference_dataset, target_column_name="target",
                                   activate=False, skip_expensive_reports=True, **{which: column})
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
    active = {name: (saved_model(sm).get_active_version() or {}).get("id") for name, sm in SAVED_MODELS.items()}
    if len(set(active.values())) != 1:
        raise ValueError(f"The saved models do not have the same active version: {active}. Activate the same model "
                         "ID in forecast_ev, forecast_p10 and forecast_p90.")
    model_id = active["ev"]
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
    """The backtest forecaster of a training run (trained without the last 6 months, for champion / challenger)."""
    found = runs()
    match = found[found["model_id"] == model_id]
    if match.empty:
        raise ValueError(f"Model {model_id} not found in Experiment Tracking ({EXPERIMENT})")
    with _mlflow(FOLDER) as mlflow:
        path = mlflow.artifacts.download_artifacts(run_id=match.iloc[0]["run_id"], artifact_path="backtest")
    with open(os.path.join(path, "forecaster.pkl"), "rb") as f:
        return pickle.load(f)


def decide(candidate, production):
    """Champion / challenger: the decision (PROMOTE or KEEP) and its reason, from the test scores of the candidate and
    of the production model (models.scores). The candidate is promoted if its P10-P90 coverage is between 70% and 90%,
    it beats the simple guess, and its EV error is lower than the production model's (a tie keeps the production
    model). production None: no production model to compare with, the candidate is promoted (the forecast needs one)
    with its failed checks as a warning."""
    checks = []
    if not MIN_COVERAGE <= candidate["coverage_p10_p90"] <= MAX_COVERAGE:
        checks.append(f"coverage {candidate['coverage_p10_p90']:.1%} outside {MIN_COVERAGE:.0%}-{MAX_COVERAGE:.0%}")
    if candidate["wape_model"] >= candidate["wape_baseline"]:
        checks.append(f"does not beat the simple guess ({candidate['wape_model']:.1%} vs "
                      f"{candidate['wape_baseline']:.1%})")
    if production is None:
        return "PROMOTE", (f"no production model to compare with, EV error {candidate['wape_model']:.2%}, coverage "
                           f"{candidate['coverage_p10_p90']:.1%}" + (f" (warning: {'; '.join(checks)})" if checks else ""))
    if checks:
        return "KEEP", f"candidate {checks[0]}"
    if candidate["wape_model"] >= production["wape_model"]:
        return "KEEP", (f"candidate EV error {candidate['wape_model']:.2%} not lower than production "
                        f"{production['wape_model']:.2%}")
    return "PROMOTE", (f"candidate EV error {candidate['wape_model']:.2%} vs production {production['wape_model']:.2%}, "
                       f"coverage {candidate['coverage_p10_p90']:.1%}")


def apply_promotion():
    """Apply the latest decision in promotion_log: on PROMOTE, activate the candidate's version in the three saved
    models. Run by a scenario step after the champion / challenger test."""
    import dataiku
    log = dataiku.Dataset("promotion_log").get_dataframe()
    last = log.sort_values("decided_at").iloc[-1]
    if last["decision"] == "PROMOTE":
        for saved_model_name in SAVED_MODELS.values():
            saved_model(saved_model_name).set_active_version(last["candidate_model_id"])
        print(f"Promoted {last['candidate_model_id']}: {last['reason']}")
    else:
        print(f"Kept the production model: {last['reason']}")
    return production_model_id()
