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

1. `GCP_SERVICE_ACCOUNT_JSON_FILE` — path to a service-account key file (recommended
   for servers/CI; requests the BigQuery + Drive-readonly scopes).
2. `GOOGLE_APPLICATION_CREDENTIALS` — the standard Google env var, handled the same way.
3. gcloud ADC (`gcloud auth application-default login`) — local dev; no scopes enforced.

All paths pin the GCP project (`stayery-analytics`), so an ADC login without a default
project still works. Result downloads use the plain SDK path by default; set
`BQ_USE_STORAGE_API=1` to opt into the faster BigQuery Storage API where reachable.

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

## Why the `python data_loader.py` call failed

Two things, neither of them about your install:

1. **Wrong interpreter.** You ran `/Users/.../uv/python/cpython-3.12.../bin/python`, which is the base Python uv keeps in its cache, *not* the project's `.venv/bin/python`. Pandas (and everything else) is only installed inside `.venv/`, so that bare interpreter sees nothing.
2. **`src/data_loader.py` is not runnable as a script.** It starts with `from .paths import data_dir`, a *relative* import - Python only allows that when the file is loaded as part of a package (i.e. imported via `from src import load_reservations`). Even with the right Python you'd see `ImportError: attempted relative import with no known parent package`.

Right way to call into the module from the shell:

```bash
uv run python -c "from src import load_reservations; print(load_reservations(limit=5).head())"
```

Or use the CLI:

```bash
uv run python main.py status
```
