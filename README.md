# Overbooking Analyse

Cancellation prediction + overbooking dashboards.
Built on uv + Python 3.12.

---

## Setup (one-time)

From the project root:

```bash
# 1. Make sure Python 3.12 is available to uv (only needed once on a fresh machine).
uv python install 3.12

# 2. Create the virtualenv and install ALL dependencies from pyproject.toml.
uv sync

# 3. (One-time per machine, only for live BigQuery pulls) authenticate gcloud.
gcloud auth application-default login
```

### BigQuery credentials (order of precedence)

`src/data_loader.py::get_bigquery_client()` builds the one shared client (plain
google-cloud-bigquery SDK, no custom REST calls):

1. `GCP_SERVICE_ACCOUNT_JSON_FILE` — path to a service-account key file.
2. `GOOGLE_APPLICATION_CREDENTIALS` — the standard Google env var, handled the same way.
3. gcloud ADC (`gcloud auth application-default login`) — local dev.

**Job project ≠ data project.** Queries reference the fully-qualified
`stayery-analytics.reporting.*` tables, so your account only needs *read* access
there (`roles/bigquery.dataViewer`). The query **jobs** run in your OWN project —
whatever the credential resolves to (SA key: its project; ADC: the gcloud
default/quota project, overridable via `GOOGLE_CLOUD_PROJECT`). Never point the
quota/config project at `stayery-analytics` unless you actually have
`serviceusage.services.use` there — the resulting 403 is the classic failure.

Diagnose everything with `uv run python main.py bqcheck` (or the "Test connection"
button on the Update page). Downloads use the plain SDK path by default; set
`BQ_USE_STORAGE_API=1` to opt into the faster Storage API where reachable.

That's it. `uv sync` reads `pyproject.toml`, creates `.venv/`, and installs everything pinned in `uv.lock`. Re-run `uv sync` any time pyproject.toml changes.

---

## Running things

**Do not** call `python somefile.py` directly with your shell's bare Python - that's not the project's venv and won't have the dependencies. Always go through `uv run`, or activate the venv first.

### Option A - `uv run` (recommended, no activation needed)

```bash
uv run python main.py status
uv run python main.py score --model xgb --refresh
uv run jupyter lab          # opens the notebooks in a browser
```

`uv run` automatically uses `.venv/bin/python` and makes sure the venv is in sync with `pyproject.toml` first.

### Option B - activate the venv

```bash
source .venv/bin/activate
python main.py status
jupyter lab
deactivate                   # when you're done
```

While activated, `python` and `pip` and `jupyter` point at the venv until you `deactivate`.

---

## Day-to-day commands

```bash
uv add <package>             # add a new dependency
uv add --dev <package>       # add a dev-only dependency
uv remove <package>          # uninstall + remove from pyproject
uv lock --upgrade            # refresh uv.lock (rebuild later via uv sync)
uv tree                      # show the full dep tree
uv run python -c "import src; print(src.__all__)"   # smoke-test the package
```
---

## Health checks

```bash
uv run python diagnostics/verify_pipeline.py       # ~28 consistency + leakage checks
uv run python diagnostics/diagnose_calibration.py  # eval artifact vs notebook reference
```

Run `verify_pipeline.py` from the repo root after any refactor or data refresh — it
fails loudly (non-zero exit) if the feature roster, leakage guards or walk-forward
splits drift.

---

## Troubleshooting

- **`ModuleNotFoundError` / "pandas not found":** you're on the wrong interpreter.
  Always go through `uv run python …` (or activate `.venv` first) — the bare system
  Python doesn't see the project's dependencies.
- **`ImportError: attempted relative import`:** files in `src/` are package modules,
  not scripts. Import them (`from src import load_reservations`) or use the CLI
  (`uv run python main.py status`) instead of running them directly.
- **BigQuery errors:** see the credentials section above; the classic failure is a
  gcloud login without a default project (fixed by `get_bigquery_client`, which pins
  the project) or a blocked BigQuery-Storage endpoint (default download avoids it).
