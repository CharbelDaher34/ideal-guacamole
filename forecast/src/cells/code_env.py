# polars: the daily flows and the nightly forecast; catboost: the placeholder model (daily_catboost); scikit-learn: for
# the data scientist's models; mlflow: Experiment
# Tracking and the saved model. Dataiku tested MLflow models with mlflow 2.22.0 and 3.10.1 (DSS 14 and 15 docs, "MLflow
# models: limitations"); mlflow 3 needs Python 3.10 or later. Either works here; the request below asks for 2.22.0,
# which also runs on Python 3.9. An Oracle output also needs oracledb: checked when the flow is built (section 3).
import re

PACKAGES = ("polars", "catboost", "scikit-learn", "mlflow==2.22.0")
placeholders = [name for name, value in (("EDW_CONNECTION", EDW_CONNECTION), ("FORECAST_CONNECTION",
                                                                           FORECAST_CONNECTION), ("CODE_ENV", CODE_ENV))
                if not str(value).strip() or str(value).startswith("YOUR_")]
if placeholders:
    raise RuntimeError(f"Fill in {' and '.join(placeholders)} in the settings cell above, then run again.")


def check_table_settings():
    """The tables' settings: each table named; the input names may be anything but a double quote (the code quotes
    them); the forecast's schema and table are created with plain SQL, so letters, digits and _ only."""
    problems = []
    for name, value, plain, required in (
            ("CLIENTS_SCHEMA", CLIENTS_SCHEMA, False, False), ("CLIENTS_TABLE", CLIENTS_TABLE, False, True),
            ("POSTINGS_SCHEMA", POSTINGS_SCHEMA, False, False), ("POSTINGS_TABLE", POSTINGS_TABLE, False, True),
            ("FORECAST_SCHEMA", FORECAST_SCHEMA, True, False), ("FORECAST_TABLE", FORECAST_TABLE, True, True)):
        if not isinstance(value, str):
            problems.append(f"{name} must be text (a name in quotes)")
        elif not value.strip():
            if required:
                problems.append(f"{name} is empty: give the table's name")
        elif value != value.strip() or '"' in value:
            problems.append(f"{name} = {value!r}: no spaces around it and no double quote")
        elif plain and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,127}", value):
            problems.append(f"{name} = {value!r}: letters, digits and _ only, starting with a letter")
    if problems:
        raise RuntimeError("Fix in the settings cell above, then run again:\n- " + "\n- ".join(problems))


check_table_settings()
check_code_env(CODE_ENV, PACKAGES)
try:                                    # this notebook's own Python: the first run tracks and promotes from here
    import mlflow  # noqa: F401
except ImportError:
    raise RuntimeError(f"This notebook runs on a Python without MLflow (Dataiku's built-in one): the first run tracks "
                       f"and promotes the first model from here. Switch it to the code env: Kernel > Change kernel > "
                       f"Python (env {CODE_ENV}), then run all the cells again.") from None
print(f"This notebook runs on a Python with MLflow {mlflow.__version__}")
