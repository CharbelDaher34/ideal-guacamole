"""Model tracking and promotion (project library: python/tracking.py), for training and the lab.

Every training run is one model with its own model ID, recorded in Dataiku's Experiment Tracking (MLflow; artifacts in
the managed folder "experiments"): its forecaster and settings, the cut-off, test scores on the last 6 periods of
30 days, the backtest forecaster (trained without those periods, for champion / challenger tests) and one MLflow model (the final
forecaster, which returns P10, EV and P90). It is deployed as a new version (version ID = model ID) of the saved model
cash_forecast (interface 3, cashflow.SAVED_MODEL). The production model is the active version; promoting a model
activates it, and the nightly forecast always uses the active version.
Lab tries (notebook forecast_lab) go to their own experiment, cashflow_lab, in the same folder. They are never
deployed: lab_compare records each forecaster it tries, and lab_run records the data scientist's own experiments.

Every run, lab and training, records the same fingerprint (describe):
    data         params daily_rows, clients, categories, data_first_day, data_last_day, load_id
    code         tags forecaster_version (its classes' code and settings: the same in the lab and in training when the
                 code is the same), code_library (the library files), and the forecaster's code as a file (code/)
    environment  tags env.python, env.catboost, env.mlflow, ...
    where        tags ran_in, user, job
    test detail  files test/summary.csv (per window, TOTAL and categories) and test/by_category.csv
A training run also gets the tag status (candidate, then production, rejected or retired: set at each promotion,
sync_status) and lab_runs: the lab runs of the same forecaster_version, the experiments behind it.
"""
import hashlib
import inspect
import json
import os
import pickle
import platform
import sys
import tempfile
from contextlib import contextmanager

import pandas as pd

from cashflow import SAVED_MODEL

EXPERIMENT, FOLDER = "cashflow_forecast", "experiments"
LAB_EXPERIMENT, LAB_FOLDER = "cashflow_lab", FOLDER        # lab tries: their own experiment, same folder
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


def _hash(text):
    return hashlib.sha1(text.encode()).hexdigest()[:10]


def _code(obj):
    """The code of an object (class or module), or "" when Python cannot read it (a class typed in a notebook)."""
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def _class_code(cls):
    """A class's own code: each of its methods' code and its other attributes (name, params, ...). Readable both in a
    file and in a notebook cell (where Python cannot read a whole class), so the same code gives the same text."""
    parts = []
    for key, value in sorted(vars(cls).items()):
        if key.startswith("__"):
            continue
        function = value.__func__ if isinstance(value, (staticmethod, classmethod)) else (
            value.fget if isinstance(value, property) else value)
        parts.append(f"{key}:\n{_code(function)}" if inspect.isfunction(function) else f"{key} = {value!r}")
    return "\n".join(parts)


def forecaster_version(forecaster):
    """A short ID of a forecaster's code and settings: its name, its params and the code of its classes (its own and
    the ones it builds on). The same in the lab and in training when the code and settings are the same, whether the
    class is typed in the lab notebook or saved in forecasters.py."""
    classes = [c for c in type(forecaster).__mro__ if c is not object]
    return _hash(json.dumps({"name": getattr(forecaster, "name", ""), "params": getattr(forecaster, "params", {}),
                             "code": [_class_code(c) for c in classes]}, sort_keys=True, default=str))


def _versions():
    """The versions of Python and of the packages the models use."""
    from importlib import metadata
    found = {"env.python": platform.python_version()}
    for package in ("catboost", "mlflow", "pandas", "numpy", "polars", "scikit-learn", "cloudpickle"):
        try:
            found[f"env.{package}"] = metadata.version(package)
        except metadata.PackageNotFoundError:
            found[f"env.{package}"] = "not installed"
    return found


def _who():
    """The user, and the job when in a recipe."""
    import dataiku
    try:
        user = dataiku.api_client().get_auth_info().get("authIdentifier", "")
    except Exception:
        user = ""
    return {"user": user, "job": os.environ.get("DKU_JOB_ACTIVITY_ID", "")}


def describe(forecaster, daily, clients, cutoff, tested, ran_in):
    """The fingerprint every run records, lab and training (see the module's text): {"params", "tags", "files"}.
    forecaster, daily, clients, cutoff, tested (the test rows, models.test_forecasts) may be None (lab_run)."""
    import dataiku
    import cashflow
    import models
    params, tags, files = {}, {"ran_in": ran_in, **_who(), **_versions()}, {}
    if daily is not None:
        days = pd.to_datetime(daily["day"])
        params.update(daily_rows=len(daily), clients=daily["client_no"].nunique(),
                      categories=daily["category"].nunique(),
                      data_first_day=f"{days.min():%Y-%m-%d}" if len(days) else "",
                      data_last_day=f"{days.max():%Y-%m-%d}" if len(days) else "",
                      load_id=dataiku.get_custom_variables().get("load_id", ""))
    if clients is not None:
        params["clients_table_rows"] = len(clients)
    if cutoff is not None:
        params["cutoff_date"] = f"{pd.Timestamp(cutoff):%Y-%m-%d}"
    library = [cashflow, models, sys.modules[__name__], sys.modules.get("forecasters")]
    tags["code_library"] = _hash("".join(_code(m) for m in library if m is not None))
    if forecaster is not None:
        params.update(forecaster=forecaster.name, **getattr(forecaster, "params", {}))
        tags.update(forecaster_version=forecaster_version(forecaster), forecaster_class=type(forecaster).__qualname__,
                    forecaster_module=type(forecaster).__module__)
        module = sys.modules.get(type(forecaster).__module__)
        own = _code(module) if module is not None and module.__name__ != "__main__" else ""
        files[f"code/{module.__name__ if own else 'forecaster'}.py"] = own or "\n\n".join(
            f"# class {c.__qualname__}\n{_code(c) or _class_code(c)}"
            for c in type(forecaster).__mro__ if c is not object)
    if tested is not None and len(tested):
        files["test/summary.csv"] = models.test_results(tested).to_csv(index=False)
        scored = ["target", "p10", "ev", "p90", "baseline"]
        by_category = tested.groupby(["category", "window"])[scored].apply(models.scores).reset_index()
        files["test/by_category.csv"] = by_category.to_csv(index=False)
    return {"params": params, "tags": tags, "files": files}


def _record(mlflow, params, metrics, backtest, about=None, tags=None):
    """A run's settings, test scores and backtest forecaster, with its fingerprint (describe) and other tags."""
    import cloudpickle
    about = about or {"params": {}, "tags": {}, "files": {}}
    mlflow.log_params({k: str(v)[:250] for k, v in {**about["params"], **params}.items()})
    mlflow.log_metrics({k: float(v) for k, v in metrics.items() if pd.notna(v)})
    mlflow.set_tags({k: str(v)[:5000] for k, v in {**about["tags"], **(tags or {})}.items()})
    with tempfile.TemporaryDirectory() as tmp:
        for name, text in about["files"].items():
            os.makedirs(os.path.join(tmp, os.path.dirname(name)), exist_ok=True)
            with open(os.path.join(tmp, name), "w") as f:
                f.write(text)
        if backtest is not None:
            os.makedirs(os.path.join(tmp, "backtest"))
            with open(os.path.join(tmp, "backtest", "forecaster.pkl"), "wb") as f:
                cloudpickle.dump(backtest, f)
        mlflow.log_artifacts(tmp)


def lab_runs_of(version):
    """The lab runs (cashflow_lab) of a forecaster_version, newest first: their run names."""
    with _mlflow(LAB_FOLDER) as mlflow:
        name = _experiment_name(mlflow, LAB_EXPERIMENT, LAB_FOLDER)
        if mlflow.get_experiment_by_name(name) is None:
            return []
        found = mlflow.search_runs(experiment_names=[name])
    if found.empty or "tags.forecaster_version" not in found:
        return []
    found = found[found["tags.forecaster_version"] == version].sort_values("start_time", ascending=False)
    return list(found.get("tags.mlflow.runName", found["run_id"]))


def log_model(model_id, params, metrics, backtest, final, about=None):
    """Record one training run in Experiment Tracking (status candidate, its fingerprint `about`: describe, and the
    lab runs of the same forecaster_version), then deploy its final forecaster as version model_id of the saved model
    cash_forecast (not activated: the promotion does that)."""
    _carry_code(final)
    version = forecaster_version(final)
    try:
        lab = lab_runs_of(version)
    except Exception as e:                   # the lab was never used, or its experiment cannot be read
        print(f"Warning: the lab runs could not be read ({e})")
        lab = []
    with _mlflow(FOLDER) as mlflow:
        mlflow.set_experiment(_experiment_name(mlflow, EXPERIMENT, FOLDER))
        with mlflow.start_run(run_name=model_id) as run:
            _record(mlflow, params, metrics, backtest, about, {
                "model_id": model_id, "status": "candidate", "status_reason": "trained, not judged yet",
                "lab_runs": ", ".join(lab[:5]) or "none (never tried in the lab with this code and settings)"})
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


def log_lab_run(run_name, params, metrics, backtest, about=None, purpose=""):
    """Record a lab try of lab_compare in Experiment Tracking (experiment cashflow_lab, folder experiments): its
    settings, test scores, fingerprint (describe) and forecaster. Never deployed. Returns the run ID."""
    _carry_code(backtest)
    with _mlflow(LAB_FOLDER) as mlflow:
        mlflow.set_experiment(_experiment_name(mlflow, LAB_EXPERIMENT, LAB_FOLDER))
        with mlflow.start_run(run_name=run_name) as run:
            _record(mlflow, params, metrics, backtest, about, {"kind": "lab_compare", "purpose": purpose})
            return run.info.run_id


@contextmanager
def lab_run(name, purpose="", forecaster=None, daily=None, clients=None, cutoff=None, tested=None):
    """A lab experiment of your own, recorded in Experiment Tracking (experiment cashflow_lab) with the same
    fingerprint as every run (describe: give what you have of forecaster, daily, clients, cutoff, tested). Inside,
    record your own settings, scores and files with the mlflow it gives. Never deployed. For example:
        with tracking.lab_run("weekly_sums", purpose="do weekly sums help?", forecaster=f, daily=daily) as mlflow:
            f.fit(daily, clients, cutoff)
            mlflow.log_metric("my_error", 0.31)
    """
    about = describe(forecaster, daily, clients, cutoff, tested, "lab notebook (tracking.lab_run)")
    with _mlflow(LAB_FOLDER) as mlflow:
        mlflow.set_experiment(_experiment_name(mlflow, LAB_EXPERIMENT, LAB_FOLDER))
        with mlflow.start_run(run_name=f"{name}_{pd.Timestamp.now():%Y%m%dT%H%M}"):
            _record(mlflow, {}, {}, None, about, {"kind": "own experiment", "purpose": purpose})
            yield mlflow


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


def _note(decided):
    """A promotion_log row, short: the candidate's forecaster, scores and the decision."""
    def share(value):
        return f"{float(value):.1%}" if pd.notna(value) else "?"
    return (f"{decided['candidate_forecaster']}, cut-off {decided['cutoff_date']}: EV error "
            f"{share(decided['candidate_wape'])} (simple guess {share(decided['baseline_wape'])}), coverage "
            f"{share(decided['candidate_coverage'])}. {decided['decision']} on {str(decided['decided_at'])[:16]}: "
            f"{decided['reason']}")


def sync_status(decided=None):
    """The tag status of the training runs, and the description of the saved model versions, made to match the saved
    model: the active version is production, a run tagged production that is no longer active is retired, and the
    candidate of `decided` (a promotion_log row) is rejected when it was kept out. Run by apply_promotion; run it by
    hand after activating a version by hand (a rollback). Never stops the promotion: a failure is a warning."""
    try:
        active = (saved_model().get_active_version() or {}).get("id")
        found = runs()
        status = found.get("tags.status", pd.Series("", index=found.index)).fillna("")
        today = f"{pd.Timestamp.now():%Y-%m-%d %H:%M}"
        updates = {}
        for model_id, run_id, was in zip(found["model_id"], found["run_id"], status):
            if model_id == active and was != "production":
                updates[model_id] = (run_id, "production", f"active since {today}")
            elif model_id != active and was == "production":
                updates[model_id] = (run_id, "retired", f"replaced by {active} on {today}")
        if decided is not None and decided["decision"] != "PROMOTE" and decided["candidate_model_id"] != active:
            match = found[found["model_id"] == decided["candidate_model_id"]]
            if len(match):
                updates[decided["candidate_model_id"]] = (match.iloc[0]["run_id"], "rejected", decided["reason"])
        with _mlflow(FOLDER) as mlflow:
            client = mlflow.tracking.MlflowClient()
            for model_id, (run_id, new, reason) in updates.items():
                client.set_tag(run_id, "status", new)
                client.set_tag(run_id, "status_reason", str(reason)[:5000])
        model = saved_model()
        for model_id, (_, new, reason) in updates.items():
            details = model.get_version_details(model_id)
            meta = details.get_user_meta()
            text = _note(decided) if decided is not None and model_id == decided["candidate_model_id"] else reason
            meta["description"] = f"{new.upper()}: {text}"[:2000]
            details.save_user_meta()
        for model_id, (_, new, _) in updates.items():
            print(f"{model_id}: {new}")
    except Exception as e:
        print(f"Warning: the status of the runs could not be updated ({type(e).__name__}: {e})")


def apply_promotion():
    """Apply the latest decision in promotion_log: on PROMOTE, activate the candidate's version of the saved model.
    Then the runs' status and the versions' description (sync_status). Run by a scenario step after the champion /
    challenger test."""
    import dataiku
    log = dataiku.Dataset("promotion_log").get_dataframe()
    last = log.sort_values("decided_at").iloc[-1]
    if last["decision"] == "PROMOTE":
        saved_model().set_active_version(last["candidate_model_id"])
        print(f"Promoted {last['candidate_model_id']}: {last['reason']}")
    else:
        print(f"Kept the production model: {last['reason']}")
    sync_status(last)
    return production_model_id()
