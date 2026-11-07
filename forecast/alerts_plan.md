# Plan: alerts (not built)

Status: **plan only.** Today nobody is told when a run fails. The failure is in Dataiku's scenario history, and
`staging_log` / `inference_log` get a row with status `FAILED` and the error.

## What to add

A **reporter** on each scenario (Scenario → Settings → Reporters), sending a message when the scenario ends:

| Scenario | When to send | To |
|---|---|---|
| Nightly batch | on failure | the inference owner |
| Retrain, test and promote | always (says if the new model was promoted or kept) | the inference owner and the data scientist |
| Full rebuild of the daily flows | always | the inference owner |

The message holds the scenario, the outcome, the failed step and its error, and a link to the run.

## Email or Teams

- **Email:** a Dataiku mail channel (SMTP) set up once by an admin. Each reporter then just lists the addresses.
- **Teams:** a Dataiku Microsoft Teams channel (a Teams webhook) set up once by an admin, then picked in the reporter.
- Both can be used together. Either way the channel needs an admin once; after that the reporters are ours.

## What we need from the bank

1. Email, Teams, or both?
2. The name of the Dataiku channel the admin creates (for example `bank-mail` or `cash-forecast-teams`).
3. Who receives what.

The setup notebook can then add the reporters to the scenarios (switched off until the channel exists).
