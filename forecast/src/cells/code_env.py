# polars: the daily flows and the nightly forecast; catboost: the placeholder model (daily_catboost); scikit-learn: for
# the data scientist's models; mlflow: Experiment
# Tracking and the saved model. Dataiku tested MLflow models with mlflow 2.22.0 and 3.10.1 (DSS 14 and 15 docs, "MLflow
# models: limitations"); mlflow 3 needs Python 3.10 or later. Either works here; the request below asks for 2.22.0,
# which also runs on Python 3.9.
PACKAGES = ("polars", "catboost", "scikit-learn", "mlflow==2.22.0")
placeholders = [name for name, value in (("EDW_CONNECTION", EDW_CONNECTION), ("OUTPUT_SQL_CONNECTION",
                                                                           OUTPUT_SQL_CONNECTION), ("CODE_ENV", CODE_ENV))
                if str(value).startswith("YOUR_")]
if placeholders:
    raise RuntimeError(f"Fill in {' and '.join(placeholders)} in the settings cell above, then run again.")
check_code_env(CODE_ENV, PACKAGES)
