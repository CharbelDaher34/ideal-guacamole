# An experiment of your own, recorded in Experiment Tracking (experiment cashflow_lab) with the same fingerprint as
# every run (the data, the code, the package versions, who ran it). Inside, record what you want with `mlflow`.
OWN_EXPERIMENT = False      # True: run this example
if OWN_EXPERIMENT:
    from models import backtest, scores
    forecaster = RecentAverage6()
    with tracking.lab_run("my_experiment", purpose="an example: the average of 6 periods instead of 3",
                          forecaster=forecaster, daily=daily, clients=clients, cutoff=cutoff) as mlflow:
        tested = backtest(forecaster, daily, clients, cutoff)
        mlflow.log_metrics({k: float(v) for k, v in scores(tested).items() if pd.notna(v)})
        mlflow.log_param("note", "anything you want to keep with the run")
    print("Recorded in Experiment Tracking, experiment cashflow_lab")
