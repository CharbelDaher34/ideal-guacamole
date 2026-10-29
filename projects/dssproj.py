"""Define a Dataiku project (datasets + recipes) in Python and push it into the local DSS.

The Free Edition has no public REST API, so this writes a project export archive
and loads it with `dsscli project-import` inside the dataiku-dss container,
then builds the flow with `dsscli build`.
"""
import http.cookiejar
import io
import json
import re
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

CONTAINER = "dataiku-dss"
DSS_URL = "http://127.0.0.1:10000"
DSSCLI = "/home/dataiku/dss/bin/dsscli"
DSS_VERSION = "15.2.0"
DSS_CONF_VERSION = "15200"
PG_CONNECTION = "postgres_local"
TAG_COLORS = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#d62728", "#17becf", "#8c564b"]


def col(name, type_="string"):
    return {"name": name, "type": type_}


def _tag():
    return {"versionNumber": 0, "lastModifiedBy": {"login": "admin"}, "lastModifiedOn": int(time.time() * 1000)}


class Project:
    def __init__(self, key, name, description="", variables=None):
        self.key, self.name, self.description = key, name, description
        self.variables = variables or {}   # project variables, read in recipes via dataiku.get_custom_variables()
        self.files = {}           # path inside archive -> bytes
        self.uploads = []         # uploaded dataset names
        self.connections = set()
        self.tags = []            # tag names in first-use order (declared in tags.json, else DSS drops them)
        self.recipes = []         # (name, outputs, kind, spec) in flow order; kind: code, visual or ml
        self.models = {}          # saved model name -> id, filled at deploy

    # ---- datasets -------------------------------------------------------
    def csv_dataset(self, name, csv_path, columns, description="", tags=()):
        """Uploaded CSV file (shows as an input dataset in the flow)."""
        data = Path(csv_path).read_bytes()
        self.files[f"uploads/{name}/{Path(csv_path).name}"] = data
        self._dataset(name, {
            "type": "UploadedFiles", "managed": False, "formatType": "csv",
            "formatParams": {"style": "excel", "charset": "utf-8", "separator": ",", "quoteChar": '"',
                             "escapeChar": "\\", "parseHeaderRow": True, "skipRowsBeforeHeader": 0,
                             "skipRowsAfterHeader": 0, "compress": ""},
            "params": {"uploadConnection": "Default (in DSS data dir.)", "notReadyIfEmpty": False,
                       "filesSelectionRules": {"mode": "ALL", "excludeRules": [], "includeRules": [],
                                               "explicitFiles": []}},
        }, columns, description, tags)
        self.uploads.append((name, len(data)))
        self.connections.add(("filesystem_managed", "Filesystem"))

    def pg_dataset(self, name, columns, description="", tags=(), connection=PG_CONNECTION):
        """Managed PostgreSQL table, named <PROJECTKEY>_<name>."""
        self._dataset(name, {
            "type": "PostgreSQL", "managed": True,
            "params": {"connection": connection, "mode": "table", "table": "${projectKey}_" + name,
                       "notReadyIfEmpty": False},
        }, columns, description, tags)
        self.connections.add((connection, "PostgreSQL"))

    def _use_tags(self, tags):
        self.tags += [t for t in tags if t not in self.tags]
        return list(tags)

    def _dataset(self, name, body, columns, description, tags):
        body.update({"schema": {"columns": columns, "userModified": True}, "tags": self._use_tags(tags),
                     "description": description, "shortDesc": description.split("\n")[0],
                     "creationTag": _tag(), "flowOptions": {"rebuildBehavior": "NORMAL"}})
        self._json(f"project_config/datasets/{name}.json", body)

    # ---- recipes --------------------------------------------------------
    def python_recipe(self, name, inputs, outputs, code, description="", tags=()):
        self._recipe(name, "python", inputs, outputs, {"envSelection": {"envMode": "INHERIT"}}, "py", code,
                     description, tags)

    def sql_recipe(self, name, inputs, outputs, sql, description="", tags=()):
        self._recipe(name, "sql_query", inputs, outputs, {}, "sql", sql, description, tags)

    def _recipe(self, name, type_, inputs, outputs, params, ext, code, description, tags):
        self._json(f"project_config/recipes/{name}.json", {
            "type": type_, "params": params, "tags": self._use_tags(tags), "creationTag": _tag(),
            "description": description, "shortDesc": description.split("\n")[0],
            "inputs": {"main": {"items": [{"ref": i, "deps": []} for i in inputs]}},
            "outputs": {"main": {"items": [{"ref": o, "appendMode": False} for o in outputs]}},
        })
        if isinstance(code, dict):               # visual recipes are configured by a JSON payload
            self._json(f"project_config/recipes/{name}.{ext}", code)
            self.recipes.append((name, outputs, "visual", None))
        else:
            self.files[f"project_config/recipes/{name}.{ext}"] = code.strip().encode() + b"\n"
            self.recipes.append((name, outputs, "code", None))

    # ---- visual recipes (output schemas are computed by DSS at deploy) ----
    def prepare_recipe(self, name, input_, output, steps, description="", tags=()):
        """Prepare recipe: a list of processor steps, built with the step helpers below (formula, ...)."""
        self._recipe(name, "shaker", [input_], [output], {}, "shaker",
                     {"columnsSelection": {"mode": "ALL"}, "steps": steps, "maxProcessedMemTableBytes": -1,
                      "previewMode": "ALL_ROWS"}, description, tags)

    def group_recipe(self, name, input_, output, keys, aggregations, count=True, pre_filter=None,
                     computed=None, names=None, description="", tags=()):
        """Group recipe. aggregations: {column: ["sum", "avg", "min", "max", "count", "concat", ...]}
        (concat joins the values with new lines).
        Output columns are named <column>_<aggregation> (+ `count`), unless renamed in names={old: new}.
        keys=[] aggregates the whole dataset into one row. computed={column: (type, GREL)} adds columns
        before aggregating."""
        values = [{"column": c, "concatSeparator": "\n", **{a: True for a in aggs}} for c, aggs in aggregations.items()]
        self._recipe(name, "grouping", [input_], [output], {}, "grouping", {
            "keys": [{"column": k} for k in keys], "values": values, "globalCount": count,
            "preFilter": _filter(pre_filter), "postFilter": _filter(None), "computedColumns": _computed(computed),
            "outputColumnNameOverrides": names or {}, "enlargeYourBits": True}, description, tags)

    def window_recipe(self, name, input_, output, partition_by, order_by, aggregations, frame=None,
                      names=None, description="", tags=()):
        """Window recipe over rows of the same partition, sorted by order_by.
        aggregations: {column: {"lag": [1, 2], "lead": [1], "avg": True, "sum": True, ...}}; every input
        column is kept. frame=(preceding, following) rows; None = from the first row up to the current one.
        Output columns are named <column>_lag<n>, <column>_lead<n> (a single lead of 1: <column>_lead),
        <column>_avg, ..., unless renamed in names."""
        values = []
        for c, aggs in aggregations.items():
            v = {"column": c, "value": True}
            for a, x in aggs.items():
                v[a] = bool(x)
                if a in ("lag", "lead"):
                    v[f"{a}Values"] = ",".join(str(n) for n in x)
            values.append(v)
        window = {"enablePartitioning": bool(partition_by), "partitioningColumns": list(partition_by),
                  "enableOrdering": bool(order_by), "orders": [{"column": c, "desc": False} for c in order_by],
                  "enableLimits": frame is not None, "windowLimitMode": "ROWS", "limitPreceding": frame is not None,
                  "precedingRows": frame[0] if frame else 0, "limitFollowing": frame is not None,
                  "followingRows": frame[1] if frame else 0}
        self._recipe(name, "window", [input_], [output], {}, "window", {
            "windows": [window], "values": values, "retrievedColumnsSelectionMode": "ALL",
            "preFilter": _filter(None), "postFilter": _filter(None), "computedColumns": [],
            "outputColumnNameOverrides": names or {}}, description, tags)

    def join_recipe(self, name, inputs, output, on, how="LEFT", columns=None, pre_filters=None,
                    description="", tags=()):
        """Join inputs[1:] to inputs[0] on the key columns `on` (how="CROSS" with on=[]: every pair).
        on: same-named keys for every input, or {dataset: keys} per joined input; a key is a column name or
        (column in inputs[0], column in the joined input) when the names differ.
        columns: {dataset: [columns to keep]} (default: all non-conflicting columns);
        pre_filters: {dataset: GREL condition} to filter an input before the join."""
        columns, pre_filters = columns or {}, pre_filters or {}
        keys = on if isinstance(on, dict) else {d: on for d in inputs[1:]}
        pairs = {d: [k if isinstance(k, tuple) else (k, k) for k in ks] for d, ks in keys.items()}
        vinputs = [{"index": i, "originLabel": d, "preFilter": _filter(pre_filters.get(d)), "computedColumns": [],
                    "outputColumnsSelectionMode": "MANUAL" if d in columns else "AUTO_NON_CONFLICTING"}
                   for i, d in enumerate(inputs)]
        joins = [{"table1": 0, "table2": i, "type": how, "conditionsMode": "AND", "outerJoinOnTheLeft": True,
                  "on": [{"column1": {"table": 0, "name": a}, "column2": {"table": i, "name": b}, "type": "EQ"}
                         for a, b in pairs[inputs[i]]]} for i in range(1, len(inputs))]
        selected = [{"name": c, "table": i} for i, d in enumerate(inputs) for c in columns.get(d, [])]
        self._recipe(name, "join", inputs, [output], {}, "join", {
            "virtualInputs": vinputs, "joins": joins, "selectedColumns": selected, "resolvedSelectedColumns": [],
            "postFilter": _filter(None), "computedColumns": [], "enableAutoCastInJoinConditions": False},
            description, tags)

    def stack_recipe(self, name, inputs, output, description="", tags=()):
        """Stack recipe: rows of all inputs, union of their columns."""
        self._recipe(name, "vstack", inputs, [output], {}, "vstack", {
            "mode": "UNION", "addOriginColumn": False, "originColumnName": "original_dataset",
            "virtualInputs": [{"index": i, "originLabel": d, "preFilter": _filter(None)} for i, d in enumerate(inputs)],
            "postFilter": _filter(None)}, description, tags)

    def split_recipe(self, name, input_, column, outputs, description="", tags=()):
        """Split recipe on the values of a column. outputs: {value: dataset}; other values are dropped."""
        datasets = list(dict.fromkeys(outputs.values()))
        self._recipe(name, "split", [input_], datasets, {}, "split", {
            "mode": "VALUES", "column": column, "defaultOutputIndex": -1, "preFilter": _filter(None),
            "valueSplits": [{"value": v, "outputIndex": datasets.index(d), "caseSensitive": True}
                            for v, d in outputs.items()],
            "computedColumns": [], "writeComputedColumnsInOutput": False}, description, tags)

    def prompt_recipe(self, name, input_, output, llm_id, template, system="", temperature=None,
                      max_tokens=None, description="", tags=()):
        """LLM Prompt recipe: one LLM call per row. {{column}} in template/system is replaced by that
        column's value. The answer lands in the output column `llm_output`."""
        cols = list(dict.fromkeys(re.findall(r"{{([^{}]+)}}", system + template)))
        self._recipe(name, "prompt", [input_], [output], {"containerSelection": {"containerMode": "INHERIT"}},
                     "prompt", {
                         "llmId": llm_id,
                         "prompt": {"promptMode": "PROMPT_TEMPLATE_TEXT", "promptTemplateQueriesSource": "DATASET",
                                    "textPromptTemplate": template, "textPromptSystemTemplate": system,
                                    "textPromptTemplateInputs": [{"name": c, "type": "TEXT", "datasetColumnName": c}
                                                                 for c in cols],
                                    "promptTemplateInputs": [], "structuredPromptExamples": [],
                                    "resultValidation": {"expectedFormat": "NONE", "requiredJSONObjectKeys": [],
                                                         "forbiddenTerms": []},
                                    "guardrailsPipelineSettings": {"guardrails": []}},
                         "completionSettings": {"temperature": temperature, "maxOutputTokens": max_tokens,
                                                "stopSequences": []},
                         "rawQueryOutputMode": "NONE", "rawResponseOutputMode": "NONE",
                         "passContext": False, "performFiltering": False,
                         "context": {"mode": "KEY_VALUE", "jsonField": {"fromColumn": False, "value": "{}"},
                                     "fields": []}},
                     description, tags)

    # ---- Visual ML (created at deploy through the same calls as the UI: Lab analysis -> deploy) ----
    def ml_model(self, name, train, test, target, reject=(), algorithm="lightgbm_regression", params=None,
                 interval=None, metric="MAE", description="", tags=()):
        """Regression model trained in a Lab analysis on `train`, tested on `test`, then deployed to the
        flow as saved model `name` with its training recipe train_<name>. params: hyperparameters of the
        algorithm (one value each); interval: prediction interval coverage, e.g. 0.8 => 10%-90%."""
        self.recipes.append((f"train_{name}", [], "ml", {
            "model": name, "train": train, "test": test, "target": target, "reject": list(reject),
            "algorithm": algorithm, "params": params or {}, "interval": interval, "metric": metric,
            "description": description, "tags": self._use_tags(tags)}))

    def timeseries_model(self, name, input_, target, time, ids, horizon, algorithms, env, external=(),
                         metric="MASE", description="", tags=()):
        """Time series forecasting model trained in a Lab analysis on input_ (one row per series and time
        step), then the best algorithm on `metric` is deployed to the flow as saved model `name`.
        ids: columns identifying each series; horizon: time steps to forecast (the last `horizon` steps of
        each series are the test set); algorithms: {algorithm: {param: value}}, a list value is a grid of
        values or, for option sets (ETS trend, ...), the options to enable; env: code env with the time
        series packages (ensure_code_env); external: columns known in advance (calendar, ...) used as
        external features by the algorithms that support them; any other column is ignored. With external
        features, the dataset to score must also hold the `horizon` future rows (target empty)."""
        self.recipes.append((f"train_{name}", [], "ml", {
            "model": name, "train": input_, "test": "", "target": target, "time": time, "ids": list(ids),
            "horizon": horizon, "algorithms": algorithms, "env": env, "external": list(external),
            "metric": metric, "description": description, "tags": self._use_tags(tags)}))

    def score_recipe(self, name, model, input_, output, description="", tags=()):
        """Score recipe: applies saved model `model` to input_ (adds the prediction columns)."""
        self.recipes.append((name, [output], "score", {
            "model": model, "input": input_, "output": output, "description": description, "tags": self._use_tags(tags)}))

    def evaluate_recipe(self, name, model, input_, scored, metrics, description="", tags=()):
        """Evaluate recipe: scores input_ (which has the true target) and computes the model's metrics."""
        self.recipes.append((name, [scored, metrics], "evaluate", {
            "model": model, "input": input_, "scored": scored, "metrics": metrics, "description": description,
            "tags": self._use_tags(tags)}))

    # ---- wiki -----------------------------------------------------------
    def wiki_home(self, title, markdown):
        """Project wiki home article (Wiki tab in the project)."""
        self._json("project_config/wiki/taxonomy.json", {"homeArticleId": "1", "taxonomy": [{"id": "1", "children": []}]})
        self._json("project_config/wiki/articles/1.json", {"name": title, "layout": "WIKI_ARTICLE", "attachments": [],
                                                           "creationTag": _tag(), "tags": []})
        self.files["project_config/wiki/articles/1.md"] = markdown.strip().encode() + b"\n"

    # ---- packaging & deploy ---------------------------------------------
    def _json(self, path, obj):
        self.files[path] = json.dumps(obj, indent=2).encode()

    def archive(self):
        self._json("project_config/params.json", {
            "name": self.name, "shortDesc": self.description, "owner": "admin",
            "projectType": "REGULAR", "permissions": [], "imgPattern": 2})
        tags = {t: {"color": TAG_COLORS[i % len(TAG_COLORS)]} for i, t in enumerate(self.tags)}
        for f, v in {"variables.json": self.variables, "localvariables.json": {}, "tags.json": {"tags": tags}}.items():
            self._json(f"project_config/{f}", v)
        self._json("export-manifest.json", {
            "exportedWithOptions": {"exportUploads": True},
            "actualContent": {"includedDatasets": [
                {"name": n, "type": "UploadedFiles", "usedTransferMethod": "UPLOAD", "size": s}
                for n, s in self.uploads]},
            "originalProjectKey": self.key, "manifestVersion": 3,
            "generatedWithDSSVersion": DSS_VERSION, "generatedWithDSSConfVersion": DSS_CONF_VERSION,
            "requiredConnections": {c: {"name": c, "type": t, "isMissing": False} for c, t in self.connections},
            "exportedByAdmin": True,
        })
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for path, data in sorted(self.files.items()):
                z.writestr(path, data)
        return buf.getvalue()

    def deploy(self, replace=True, build=True):
        """Import into DSS (replacing an existing project with the same key), then run the recipes one by
        one in flow order. A visual recipe's output schema is computed by DSS from its (built) input data,
        like the UI does when you save a recipe, so each recipe's inputs are built before it."""
        existing = _dsscli("projects-list", "--output", "json", capture=True)
        if any(p.get("projectKey") == self.key for p in json.loads(existing)):
            if not replace:
                raise SystemExit(f"{self.key} already exists")
            _dsscli("project-delete", self.key)
        remote = f"/tmp/{self.key}.zip"
        subprocess.run(["docker", "exec", "-i", CONTAINER, "sh", "-c", f"cat > {remote}"],
                       input=self.archive(), check=True)
        _dsscli("project-import", remote)
        subprocess.run(["docker", "exec", CONTAINER, "rm", "-f", remote], check=True)
        _login()
        for name, outputs, kind, spec in self.recipes:
            if kind == "ml":
                print(f"  training {spec['model']} (Lab analysis, then deploy to the flow)")
                self._train_and_deploy(name, spec)
                continue
            if kind in ("score", "evaluate"):
                self._create_model_recipe(name, kind, spec)
            if kind != "code":
                self._update_schema(name)
            if build:
                print(f"  running {name} -> {', '.join(outputs)}")
                _dsscli("build", self.key, "--wait", "--mode", "NON_RECURSIVE_FORCED_BUILD",
                        *[a for o in outputs for a in ("--dataset", o)], capture=True)
        print(f"{self.key} ready: http://127.0.0.1:10000/projects/{self.key}/flow/")

    def _train_and_deploy(self, recipe, spec):
        k = self.key
        timeseries = "horizon" in spec
        aid = _api("POST", "analysis/create", {"projectKey": k, "inputDatasetSmartName": spec["train"],
                                               "name": spec["model"]})["id"]
        guess = {"projectKey": k, "analysisId": aid, "targetVariable": spec["target"], "mlBackendType": "PY_MEMORY",
                 "predictionType": "REGRESSION", "timeseriesIdentifiers": "[]", "guessPolicy": "DEFAULT",
                 "mlBackendName": ""}
        if timeseries:
            guess.update({"predictionType": "TIMESERIES_FORECAST", "timeVariable": spec["time"],
                          "timeseriesIdentifiers": json.dumps(spec["ids"]), "guessPolicy": "TIMESERIES_DEFAULT"})
        tid = _api("POST", "analysis/pml/create-and-guess", guess)["id"]
        task = {"projectKey": k, "analysisId": aid, "mlTaskId": tid}
        _wait(lambda: not _api("GET", "analysis/pml/get-mltask-status", task).get("guessing"))
        s = _api("POST", "analysis/pml/get-updated-settings", task)
        algorithms = spec["algorithms"] if timeseries else {spec["algorithm"]: spec["params"]}
        for algo, conf in s["modeling"].items():
            if isinstance(conf, dict) and "enabled" in conf:
                conf["enabled"] = algo in algorithms
        for algo, params in algorithms.items():
            _set_params(s["modeling"][algo], params)
        s["modeling"]["metrics"]["evaluationMetric"] = spec["metric"]
        if timeseries:
            for c, f in s["preprocessing"]["per_feature"].items():
                if f["role"] not in ("TARGET", "TIME", "TIMESERIES_IDENTIFIER"):
                    f["role"] = "INPUT" if c in spec["external"] else "REJECT"
            for algo in algorithms:
                if "_use_external_features" in s["modeling"][algo]:
                    s["modeling"][algo]["_use_external_features"] = bool(spec["external"])
            # ML algorithms (LightGBM, ...) use past values shifted by at least the horizon
            shifts = s["preprocessing"]["feature_generation"]["auto_shifts_params"]
            shifts["max_horizon_shift_past_only"] = min(shifts["max_horizon_shift_past_only"], -spec["horizon"])
            shifts["min_horizon_shift_past_only"] = min(shifts["min_horizon_shift_past_only"], -2 * spec["horizon"])
            s["predictionLength"] = spec["horizon"]
            s["evaluationParams"] = {"gapSize": 0, "testSize": spec["horizon"]}
            s["envSelection"] = {"envMode": "EXPLICIT_ENV", "envName": spec["env"]}
        else:
            for c in spec["reject"]:
                s["preprocessing"]["per_feature"][c]["role"] = "REJECT"
            full = {"samplingMethod": "FULL", "partitionSelectionMethod": "ALL", "maxRecords": -1,
                    "filter": {"enabled": False, "distinct": False}}
            s["splitParams"].update({"ttPolicy": "EXPLICIT_FILTERING_TWO_DATASETS",
                                     "eftdTrain": {"datasetSmartName": spec["train"], "selection": full},
                                     "eftdTest": {"datasetSmartName": spec["test"], "selection": full}})
            if spec["interval"]:
                s["uncertainty"].update({"predictionIntervalsEnabled": True,
                                         "predictionIntervalsCoverage": spec["interval"]})
        _api("POST", "analysis/pml/save-settings", {"projectKey": k, "analysisId": aid, "mlTask": json.dumps(s)})
        _api("POST", "analysis/pml/train-start", {**task, "userSessionName": "", "userSessionDescription": "",
                                                 "forceRefresh": "true", "runQueue": "false"})
        status = _wait(lambda: (lambda st: st if not st.get("training") and st.get("fullModelIds") else None)(
            _api("GET", "analysis/pml/get-mltask-status", task)), timeout=1800)
        ids = [m["id"] for m in status["fullModelIds"]]
        snippets = _api("GET", "analysis/pml/get-model-snippets", {**task, "fullModelIds": json.dumps(ids)})
        for i in ids:
            sn = snippets[i]
            if sn.get("trainInfo", {}).get("state") != "DONE":
                raise SystemExit(f"training {spec['model']} ({sn.get('algorithm')}) failed: "
                                 f"{json.dumps(sn.get('trainInfo'))[:800]}")
            print(f"    {sn.get('algorithm')}: {spec['metric']} {sn.get(spec['metric'].lower()):.3f}")
        model_id = min(ids, key=lambda i: snippets[i][spec["metric"].lower()])   # error metrics: lower is better
        r = _api("POST", "analysis/pml/flow/deploy-train", {
            "fullModelId": model_id, "trainDatasetSmartName": spec["train"], "testDatasetSmartName": spec["test"],
            "modelName": spec["model"], "managedFolderSmartId": "", "options": json.dumps({"redoOptimization": False})})
        self.models[spec["model"]] = r["second"]
        self._set_recipe_meta(r["first"], recipe, spec)

    def _create_model_recipe(self, recipe, kind, spec):
        opts = {"inputDatasetSmartName": spec["input"], "savedModelSmartName": self.models[spec["model"]],
                "managedFolderSmartId": ""}
        if kind == "score":
            opts.update({"createOutput": False, "outputDatasetSmartName": spec["output"]})
        else:
            opts.update({"scoredDatasetSmartName": spec["scored"], "metricsDatasetSmartName": spec["metrics"]})
        created = _api("POST", f"savedmodels/prediction/deploy-{'scoring' if kind == 'score' else 'evaluation'}",
                       {"projectKey": self.key, "options": json.dumps(opts)})["id"]
        self._set_recipe_meta(created, recipe, spec)

    def _set_recipe_meta(self, current, name, spec):
        """Rename an API-created recipe and give it the description and tags of the code."""
        if current != name:
            _api("POST", "flow/recipes/rename", {"projectKey": self.key, "origName": current, "newName": name})
        recipe = _api("GET", "flow/recipes/get", {"projectKey": self.key, "name": name})
        recipe.update({"tags": spec["tags"], "description": spec["description"],
                       "shortDesc": spec["description"].split("\n")[0]})
        _api("POST", "flow/recipes/save", {"projectKey": self.key, "recipe": json.dumps(recipe),
                                           "commitMessage": "created by dssproj"})

    def _update_schema(self, recipe):
        result = _api("GET", "flow/recipes/get-schema-update-result", {"projectKey": self.key, "recipeName": recipe})
        for c in result.get("computables", []):
            if c.get("newSchema"):
                _api("POST", "flow/recipes/save-output-schema", {
                    "projectKey": self.key, "computableType": "DATASET", "computableId": c["datasetName"],
                    "newSchema": json.dumps(c["newSchema"]), "dropAndRecreate": "true", "synchronizeMetastore": "false"})


def ensure_code_env(name, preset="DOCTOR", interpreter="PYTHON312", core="PANDAS22"):
    """Create a DSS-managed Python code env with one of DSS's package presets (Administration > Code envs >
    Add sets of packages), unless it already exists. DOCTOR = Visual ML + time series forecasting (CPU)."""
    _login()
    if any(e.get("envName") == name for e in _api("GET", "code-envs/design/list")):
        return
    print(f"  creating code env {name} ({preset}), a few minutes")
    _future(_api("POST", "code-envs/design/create", {"envLang": "PYTHON", "envSpec": json.dumps({
        "envLang": "PYTHON", "envName": name, "deploymentMode": "DESIGN_MANAGED", "pythonInterpreter": interpreter,
        "conda": False, "installCorePackages": True, "corePackagesSet": core, "installJupyterSupport": False})}))
    presets = _api("GET", "code-envs/design/package-presets/python",
                   {"interpreter": interpreter, "useConda": "false", "corePackageSet": core})
    env = _api("GET", "code-envs/design/get", {"envLang": "PYTHON", "envName": name})
    env["specPackageList"] = "\n".join(next(p for p in presets if p["id"] == preset)["packages"])
    _api("POST", "code-envs/design/save", {"envLang": "PYTHON", "envName": name, "data": json.dumps(env)})
    _future(_api("POST", "code-envs/design/update", {"envLang": "PYTHON", "envName": name, "updateSettings": json.dumps(
        {"upgradeAllPackages": False, "updateResources": False, "forceRebuildEnv": False})}))


def _future(started, timeout=3600):
    """Wait for a DSS long-running task (code env creation, ...) and fail on error."""
    fid = started["jobId"]
    done = _wait(lambda: (lambda u: u if not u.get("alive") else None)(
        _api("GET", "futures/get-update", {"futureId": fid})), timeout=timeout, every=10)
    messages = (done.get("result") or {}).get("messages", {})
    if done.get("aborted") or done.get("unexpectedError") or messages.get("error"):
        raise SystemExit(f"DSS task failed: {json.dumps(done)[:1500]}")
    return done


def _wait(check, timeout=600, every=3):
    """Poll check() until it returns something truthy."""
    end = time.time() + timeout
    while time.time() < end:
        result = check()
        if result:
            return result
        time.sleep(every)
    raise SystemExit("timed out waiting for DSS")


def _set_params(algo, params):
    """Algorithm hyperparameters as the UI stores them: a grid of values, a set of options to enable, or a
    plain value."""
    for p, v in params.items():
        values = algo.get(p, {}).get("values") if isinstance(algo.get(p), dict) else None
        if isinstance(values, dict):                       # option set, e.g. ETS trend: add / mul / none
            for option, conf in values.items():
                conf["enabled"] = option in v
        elif values is not None:                           # grid of values
            values[:] = v if isinstance(v, list) else [v]
        else:
            algo[p] = v


def _computed(columns):
    return [{"name": n, "type": t, "expr": e, "mode": "GREL"} for n, (t, e) in (columns or {}).items()]


def _filter(expression):
    """Recipe pre/post filter from a GREL condition (None = no filter)."""
    if not expression:
        return {"enabled": False, "distinct": False}
    return {"enabled": True, "distinct": False, "uiData": {"mode": "CUSTOM"}, "expression": expression}


# ---- Prepare recipe steps ---------------------------------------------------
def step(type_, **params):
    return {"type": type_, "params": params, "metaType": "PROCESSOR", "preview": False, "disabled": False,
            "alwaysShowComment": False}


def formula(column, expression):
    """New (or overwritten) column from a GREL formula, e.g. formula("net", "inflow - outflow")."""
    return step("CreateColumnWithGREL", column=column, expression=expression)


def parse_date(column, fmt="yyyy-MM-dd"):
    return step("DateParser", appliesTo="SINGLE_COLUMN", columns=[column], formats=[fmt], lang="en_US",
                timezone_id="UTC", outCol="")


def truncate_date(column, part, out_column):
    return step("DateTruncate", inCol=column, datePart=part, outCol=out_column, timezone_id="UTC")


def remove_empty(*columns):
    """Drop rows where any of these columns is empty."""
    return step("RemoveRowsOnEmpty", appliesTo="COLUMNS", columns=list(columns), keep=False)


def keep_rows(condition):
    """Keep only rows matching a GREL condition."""
    return step("FilterOnCustomFormula", expression=condition, action="KEEP_ROW")


def fold(columns, name_column, value_column):
    """Turn columns into rows: one row per column, its name in name_column and its value in value_column."""
    return step("MultiColumnFold", columns=list(columns), foldNameColumn=name_column, foldValueColumn=value_column)


def rename(mapping):
    return step("ColumnRenamer", renamings=[{"from": a, "to": b} for a, b in mapping.items()])


def keep_columns(*columns):
    return step("ColumnsSelector", appliesTo="COLUMNS", columns=list(columns), keep=True)


def delete_columns(*columns):
    return step("ColumnsSelector", appliesTo="COLUMNS", columns=list(columns), keep=False)


# ---- internal DSS UI API (admin/admin), used to compute visual recipe output schemas ----------
_jar = http.cookiejar.CookieJar()
_http = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar))


def _api(method, path, fields=None):
    data = urllib.parse.urlencode(fields or {})
    url = f"{DSS_URL}/dip/api/{path}"
    if method == "GET":
        url, body = f"{url}?{data}", None
    else:
        body = data.encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}
    headers.update({"X-XSRF-TOKEN": c.value for c in _jar if c.name.startswith("dss_xsrf_token")})
    try:
        with _http.open(urllib.request.Request(url, data=body, headers=headers, method=method)) as r:
            text = r.read().decode()
    except urllib.error.HTTPError as e:
        raise SystemExit(f"DSS API {path} failed ({e.code}): {e.read().decode()[:500]}")
    return json.loads(text) if text[:1] in ("{", "[") else text


def _login():
    _api("POST", "login", {"login": "admin", "password": "admin"})
    _api("GET", "get-configuration")      # sets the XSRF cookie


def _dsscli(*args, capture=False):
    r = subprocess.run(["docker", "exec", CONTAINER, DSSCLI, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"dsscli {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout if capture else print(r.stdout.strip())
