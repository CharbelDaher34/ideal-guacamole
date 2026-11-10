import time


def build(step, items, mode):
    """Runs a Dataiku job and waits for it, showing its PROGRESS lines here (also in the job's log) every 30 s."""
    print(f"\n{step} ...", flush=True)
    builder = project.new_job(mode).with_auto_update_schema_before_each_recipe_run(True)
    for item in items:
        if item in folders:
            builder.with_output(folders[item].id, object_type="MANAGED_FOLDER")
        else:
            builder.with_output(item)
    job, shown, began = builder.start(), 0, time.time()
    while True:
        state = job.get_status().get("baseStatus", {}).get("state", "")
        try:
            lines = [l[l.index("PROGRESS ") + 9:] for l in job.get_log().splitlines() if "PROGRESS " in l]
        except Exception:
            lines = []
        for line in lines[shown:]:
            print("   ", line, flush=True)
        shown = max(shown, len(lines))
        if state in ("DONE", "FAILED", "ABORTED"):
            break
        time.sleep(30 if time.time() - began > 60 else 5)
    if state != "DONE":
        raise RuntimeError(f"{step}: the job {state}. Open it in Jobs (id {job.id}) to read its log.")
    print(f"{step}: done in {(time.time() - began) / 60:.1f} min", flush=True)

