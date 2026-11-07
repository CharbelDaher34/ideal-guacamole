import dataiku

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())
# scikit-learn: for the data scientist's models; mlflow: Experiment Tracking and the saved model. Dataiku tested MLflow models with
# mlflow 2.22.0 and 3.10.1 (DSS 14 and 15 docs, "MLflow models: limitations"); mlflow 3 needs Python 3.10 or later.
# Either works here; the request below asks for 2.22.0, which also runs on Python 3.9.
PACKAGES = ("scikit-learn", "mlflow==2.22.0")


def check_code_env(name, packages=PACKAGES):
    """Check the code env exists and lists the packages. It is not created here: an admin creates it once."""
    request = (f"Ask a Dataiku admin to create the Python code env {name} (Python 3.9 or later) with these "
               f"packages: {', '.join(packages)}. Or set CODE_ENV to an existing env that has them.")
    if name not in [e["envName"] for e in client.list_code_envs()]:
        raise RuntimeError(f"Code env {name} not found. {request}")
    try:
        listed = " ".join(client.get_code_env("PYTHON", name).get_settings().get_required_packages(as_list=True))
    except Exception:          # no right to read the env settings: it is checked when the first recipe runs
        print(f"Code env {name} found (could not read its packages)")
        return
    missing = [p for p in packages if p.split("=")[0].split(">")[0] not in listed]
    if missing:
        raise RuntimeError(f"Code env {name} lacks {', '.join(missing)}. {request}")
    print(f"Code env {name} has {', '.join(packages)}")


check_code_env(CODE_ENV)
