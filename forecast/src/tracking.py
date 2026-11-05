"""Model tracking and promotion (project library: python/tracking.py).

Every training run is one model with its own model ID, recorded in Dataiku's Experiment Tracking (MLflow;
artifacts in the managed folder "experiments"): settings, cut-off, test scores on the last 6 months, the backtest
models (trained without those months, for champion / challenger tests) and three MLflow models (EV, P10, P90).
Those three are deployed as a new version (version ID = model ID) of the saved models forecast_ev, forecast_p10
and forecast_p90. The production model is the active version; promoting a model activates it in all three, and
the Score recipes always use the active version.
"""
import inspect
import os
import pickle
import tempfile

import pandas as pd

EXPERIMENT = "cashflow_forecast"
FOLDER = "experiments"
SAVED_MODELS = {"ev": "forecast_ev", "p10": "forecast_p10", "p90": "forecast_p90"}   # MLflow model name: saved model
MIN_COVERAGE, MAX_COVERAGE = 0.70, 0.90     # a model is only promoted if its P10-P90 coverage is in this range


def _project():
    import dataiku
    return dataiku.api_client().get_default_project()


def _mlflow():
    import dataiku
    return _project().setup_mlflow(managed_folder=dataiku.Folder(FOLDER))


def _experiment_name(mlflow):
    """EXPERIMENT, kept in the folder experiments. If an experiment of that name points to another folder (the
    experiments folder was deleted and created again, for example when the flow was deleted by hand), its files are
    gone: use EXPERIMENT_<folder id> instead."""
    import dataiku
    folder_id = dataiku.Folder(FOLDER).get_id()
    found = mlflow.get_experiment_by_name(EXPERIMENT)
    if found is None or f"//{folder_id}/" in (found.artifact_location or ""):
        return EXPERIMENT
    return f"{EXPERIMENT}_{folder_id}"


def new_model_id(cutoff):
    """CFF_<cut-off month>_<training time>, for example CFF_202610_20261104T1215 (also the saved model version)."""
    return f"CFF_{cutoff:%Y%m}_{pd.Timestamp.now():%Y%m%dT%H%M}"


def saved_model(name):
    project = _project()
    return project.get_saved_model(next(m["id"] for m in project.list_saved_models() if m["name"] == name))


def forecast_model(model, column=None):
    """An MLflow model around a CatBoost model: takes model rows, returns amounts in SAR (column: which quantile)."""
    import mlflow.pyfunc
    from cashflow import model_matrix

    class ForecastModel(mlflow.pyfunc.PythonModel):
        def __init__(self, model, column):
            self.model, self.column = model, column

        def predict(self, context, model_input, params=None):
            raw = self.model.predict(model_matrix(model_input))
            if self.column is not None:
                raw = raw[:, self.column]
            return raw * model_input["series_scale"].values

    return ForecastModel(model, column)


def log_model(model_id, params, metrics, backtest, final, reference_dataset):
    """Record one training run in Experiment Tracking, then deploy its EV, P10 and P90 models as version model_id
    of the three saved models (not activated: the promotion does that)."""
    import cloudpickle
    import cashflow
    import tracking
    cloudpickle.register_pickle_by_value(cashflow)     # the MLflow models carry their own copy of the code
    cloudpickle.register_pickle_by_value(tracking)
    with _mlflow() as mlflow:
        mlflow.set_experiment(_experiment_name(mlflow))
        with mlflow.start_run(run_name=model_id) as run:
            mlflow.set_tag("model_id", model_id)
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
            with tempfile.TemporaryDirectory() as tmp:
                for kind, model in backtest.items():
                    with open(os.path.join(tmp, f"backtest_{kind}.pkl"), "wb") as f:
                        pickle.dump(model, f)
                mlflow.log_artifacts(tmp, artifact_path="backtest")
            where = "name" if int(mlflow.__version__.split(".")[0]) >= 3 else "artifact_path"   # MLflow 3 or 2
            for name, model, column in (("ev", final["ev"], None), ("p10", final["p10_p90"], 0),
                                        ("p90", final["p10_p90"], 1)):
                mlflow.pyfunc.log_model(python_model=forecast_model(model, column), **{where: name})
            run_id = run.info.run_id
    extension = _project().get_mlflow_extension()
    code_env = _project().get_recipe("train_and_test").get_settings().get_recipe_params()["envSelection"]["envName"]
    extension.set_run_inference_info(run_id=run_id, prediction_type="REGRESSION", code_env_name=code_env,
                                     target="target")
    # which model of the run: model_name in DSS 15 (MLflow 3 models), model_sub_folder in DSS 14 (the artifact path)
    which = "model_name" if "model_name" in inspect.signature(extension.deploy_run_model).parameters else "model_sub_folder"
    for name, saved_model_name in SAVED_MODELS.items():
        model = saved_model(saved_model_name)
        before = (model.get_active_version() or {}).get("id")
        # evaluated on the reference sample: gives the version its input schema (the Score recipes need it)
        extension.deploy_run_model(run_id, model.id, version_id=model_id,
                                   evaluation_dataset=reference_dataset, target_column_name="target",
                                   activate=False, skip_expensive_reports=True, **{which: name})
        if before and (model.get_active_version() or {}).get("id") != before:
            model.set_active_version(before)       # DSS 14 activates every deployed version: only a promotion may


def runs():
    """Every training run, newest first: model_id, run_id, trained_at, and its metrics."""
    with _mlflow() as mlflow:
        # sorted here: DSS 14 cannot sort runs by start time in the search
        found = mlflow.search_runs(experiment_names=[_experiment_name(mlflow)])
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


def load_models(model_id, name="backtest"):
    """The backtest EV and P10/P90 models of a training run (for champion / challenger tests)."""
    found = runs()
    match = found[found["model_id"] == model_id]
    if match.empty:
        raise ValueError(f"Model {model_id} not found in Experiment Tracking ({EXPERIMENT})")
    with _mlflow() as mlflow:
        path = mlflow.artifacts.download_artifacts(run_id=match.iloc[0]["run_id"], artifact_path=name)
    models = {}
    for kind in ("ev", "p10_p90"):
        with open(os.path.join(path, f"{name}_{kind}.pkl"), "rb") as f:
            models[kind] = pickle.load(f)
    return models


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
