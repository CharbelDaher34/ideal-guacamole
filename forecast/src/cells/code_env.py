import dataiku

client = dataiku.api_client()
project = client.get_project(dataiku.default_project_key())


def ensure_code_env(name, packages=("catboost>=1.2", "scikit-learn")):
    """Create a Python code env with CatBoost, unless one with this name already exists (takes a few minutes)."""
    if name in [e["envName"] for e in client.list_code_envs()]:
        print(f"Code env {name} already exists")
        return
    error = None
    for interpreter in ("PYTHON312", "PYTHON311", "PYTHON310", "PYTHON39"):
        try:
            client.create_code_env("PYTHON", name, "DESIGN_MANAGED", {
                "pythonInterpreter": interpreter, "installCorePackages": True, "installJupyterSupport": False})
            break
        except Exception as e:          # interpreter not installed on this DSS: try the next one
            error = e
    else:
        raise RuntimeError(f"Could not create code env {name}: {error}")
    env = client.get_code_env("PYTHON", name)
    settings = env.get_settings()
    settings.set_required_packages(*packages)
    settings.save()
    env.update_packages()
    print(f"Code env {name} created ({interpreter}) with {', '.join(packages)}")


ensure_code_env(CODE_ENV)
