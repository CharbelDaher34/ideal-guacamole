"""Talk to the local sandbox DSS from the host, like the browser does (the Free Edition has no public API keys):
log in, create a project, import a notebook and run its cells in a DSS kernel, run a scenario. Test machine only.

    python3 sandbox/dss.py run-notebook PROJECT path/to/notebook.ipynb [KERNEL]
    python3 sandbox/dss.py import-notebook PROJECT path/to/notebook.ipynb ...
    python3 sandbox/dss.py run-scenario PROJECT SCENARIO_ID
"""
import http.cookiejar
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

DSS = os.environ.get("DSS_URL", "http://127.0.0.1:10000")
_jar = http.cookiejar.CookieJar()
_open = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(_jar)).open


def _xsrf():
    h = {"X-XSRF-TOKEN": c.value for c in _jar if c.name.startswith("dss_xsrf_token")}
    h.update({"X-XSRFToken": c.value for c in _jar if c.name == "_xsrf"})
    return h


def _send(request):
    try:
        with _open(request) as r:
            text = r.read().decode()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{request.full_url} {e.code}: {e.read().decode()[:800]}") from None
    return json.loads(text) if text[:1] in ("{", "[") else text


def api(method, path, fields=None):
    """The DSS web API (/dip/api/...), with form fields."""
    data = urllib.parse.urlencode(fields or {})
    url, body = f"{DSS}/dip/api/{path}", None
    if method == "GET":
        url = f"{url}?{data}"
    else:
        body = data.encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8", **_xsrf()}
    return _send(urllib.request.Request(url, data=body, headers=headers, method=method))


def raw(method, path, payload=None):
    """Any DSS path (Jupyter's own API), with a JSON body."""
    body = json.dumps(payload).encode() if payload is not None else None
    return _send(urllib.request.Request(DSS + path, data=body, headers={"Content-Type": "application/json", **_xsrf()},
                                        method=method))


def login(user="admin", password="admin"):
    api("POST", "login", {"login": user, "password": password})
    api("GET", "get-configuration")
    try:
        _open(f"{DSS}/jupyter/tree").read()          # sets Jupyter's own cookie
    except Exception:
        pass


def create_project(key, name):
    if key not in api("GET", "projects/list-all-keys"):
        api("POST", "projects/create", {"projectKey": key, "name": name})


def import_notebook(project, path, name=None):
    """Put a notebook in a project (like Notebooks > New notebook > Import), replacing one of the same name."""
    name = name or os.path.basename(path).removesuffix(".ipynb")
    try:
        raw("DELETE", f"/jupyter/api/contents/{project}/{name}.ipynb")
    except RuntimeError:
        pass
    boundary = uuid.uuid4().hex
    parts = [f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
             for k, v in {"projectKey": project, "baseName": name, "language": "python"}.items()]
    with open(path, "rb") as f:
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}.ipynb"\r\n'
                     f'Content-Type: application/octet-stream\r\n\r\n'.encode() + f.read() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}", **_xsrf()}
    _send(urllib.request.Request(f"{DSS}/dip/api/jupyter/new-notebook-from-file", data=b"".join(parts),
                                 headers=headers, method="POST"))
    return name


def delete_notebook(project, name):
    raw("DELETE", f"/jupyter/api/contents/{project}/{name}.ipynb")


def run_notebook(project, path, kernel="python3", name=None, quiet=False):
    """Import the notebook, then run its code cells one by one in a DSS kernel, printing their output. Stops at the
    first error. Returns True if every cell ran."""
    import websocket
    name = import_notebook(project, path, name)
    session = raw("POST", "/jupyter/api/sessions", {"path": f"{project}/{name}.ipynb", "type": "notebook",
                                                     "name": f"{name}.ipynb", "kernel": {"name": kernel}})
    cookie = "; ".join(f"{c.name}={c.value}" for c in _jar)
    ws = websocket.create_connection(
        f"{DSS.replace('http', 'ws', 1)}/jupyter/api/kernels/{session['kernel']['id']}/channels?session_id={session['id']}",
        header=[f"Cookie: {cookie}", *[f"{k}: {v}" for k, v in _xsrf().items()]], timeout=7200)
    ok = True
    with open(path) as f:
        cells = [c for c in json.load(f)["cells"] if c["cell_type"] == "code"]
    for i, cell in enumerate(cells):
        code, msg_id, started = "".join(cell["source"]), uuid.uuid4().hex, time.time()
        ws.send(json.dumps({"header": {"msg_id": msg_id, "username": "admin", "session": session["id"],
                                       "msg_type": "execute_request", "version": "5.3"},
                            "parent_header": {}, "metadata": {}, "channel": "shell",
                            "content": {"code": code, "silent": False, "store_history": True,
                                        "user_expressions": {}, "allow_stdin": False, "stop_on_error": True}}))
        if not quiet:
            print(f"\n=== {name}, cell {i}: {code.splitlines()[0][:80] if code else ''}", flush=True)
        while True:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            kind, content = m["msg_type"], m["content"]
            if kind == "stream" and not quiet:
                sys.stdout.write(content["text"])
            elif kind in ("execute_result", "display_data") and not quiet:
                print(content["data"].get("text/plain", "")[:3000])
            elif kind == "error":
                print(f"ERROR in {name}, cell {i}:", content["ename"], content["evalue"])
                print("\n".join(content["traceback"])[-4000:])
                ok = False
            elif kind == "status" and content["execution_state"] == "idle":
                break
        if not quiet:
            print(f"--- {time.time() - started:.0f}s", flush=True)
        if not ok:
            break
    ws.close()
    raw("DELETE", f"/jupyter/api/sessions/{session['id']}")
    return ok


def run_scenario(project, scenario_id):
    """Run a scenario like its Run button, wait for it, print each step's outcome. Returns the outcome."""
    api("POST", "scenarios/manual-run", {"projectKey": project, "smartScenarioId": scenario_id, "params": "{}",
                                         "waitForStart": "true", "waitForCompletion": "false"})
    started = time.time()
    while True:
        time.sleep(10)
        last = api("GET", "scenarios/get-last-scenario-runs", {"projectKey": project, "smartScenarioId": scenario_id,
                                                              "withFullScenario": "false", "limit": "1",
                                                              "onlyFinishedRuns": "false"})[0]
        if last.get("result"):
            details = api("GET", "scenarios/get-scenario-run-details",
                          {"projectKey": project, "scenarioId": scenario_id, "runId": last["runId"]})
            outcome = last["result"].get("outcome")
            print(f"{scenario_id}: {outcome} ({time.time() - started:.0f}s)")
            for s in details.get("stepRuns", []):
                print("  ", s["step"]["name"], "|", s.get("result", {}).get("outcome"))
            return outcome


if __name__ == "__main__":
    command, args = sys.argv[1], sys.argv[2:]
    login()
    if command == "run-notebook":
        sys.exit(0 if run_notebook(args[0], args[1], *(args[2:3] or ["python3"])) else 1)
    elif command == "import-notebook":
        for p in args[1:]:
            print("imported", import_notebook(args[0], p))
    elif command == "run-scenario":
        sys.exit(0 if run_scenario(args[0], args[1]) == "SUCCESS" else 1)
    else:
        sys.exit(__doc__)
