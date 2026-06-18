# Overbooking Analyse

Cancellation prediction + overbooking dashboards for Stayery.
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

That's it. `uv sync` reads `pyproject.toml`, creates `.venv/`, and installs everything pinned in `uv.lock`. Re-run `uv sync` any time pyproject.toml changes.

---

## Running things

**Never** call `python somefile.py` directly with your shell's bare Python — that's not the project's venv and won't have the dependencies. Always go through `uv run`, or activate the venv first.

### Option A — `uv run` (recommended, no activation needed)

```bash
uv run python main.py status
uv run python main.py score --model xgb --refresh
uv run jupyter lab          # opens the notebooks in a browser
```

`uv run` automatically uses `.venv/bin/python` and makes sure the venv is in sync with `pyproject.toml` first.

### Option B — activate the venv (more traditional)

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

## Pipeline order

Model lineup decided 2026-06-11 (RF + MLP are out — see reports/open_decisions.md):
3 static classifiers + 3 survival models, then a fair comparison.

1. `notebooks/00_data_audit.ipynb`   — load + audit + clean + temporal_split (60/15/25) → `Data/reservations_clean.parquet`
2. `notebooks/01_logreg.ipynb`       — logistic regression (gold-standard template, fully commented)
3. `notebooks/02_xgboost.ipynb`      — XGBoost (clone of 01, swap §5)        [to rebuild]
4. `notebooks/03_histgb.ipynb`       — HistGradientBoosting (clone of 01)    [to rebuild]
5. `notebooks/08_hazard.ipynb`       — discrete-time hazard (daily rescoring) [to finish]
6. `notebooks/09_xgb_aft.ipynb`      — XGBoost AFT survival                   [to build]
7. `notebooks/10_rsf.ipynb`          — Random Survival Forest                 [to build]
8. `notebooks/05_model_comparison.ipynb` — fair bake-off + tuning of finalists [to rebuild]
9. `uv run python main.py score`     — daily scoring with the chosen model

Each notebook is self-contained: cleaning + features + modeling live inline; loading + styling + paths + scoring are imported from `src/`.

---

## Why the `python data_loader.py` call failed

Two things, neither of them about your install:

1. **Wrong interpreter.** You ran `/Users/.../uv/python/cpython-3.12.../bin/python`, which is the base Python uv keeps in its cache, *not* the project's `.venv/bin/python`. Pandas (and everything else) is only installed inside `.venv/`, so that bare interpreter sees nothing.
2. **`src/data_loader.py` is not runnable as a script.** It starts with `from .paths import data_dir`, a *relative* import — Python only allows that when the file is loaded as part of a package (i.e. imported via `from src import load_reservations`). Even with the right Python you'd see `ImportError: attempted relative import with no known parent package`.

Right way to call into the module from the shell:

```bash
uv run python -c "from src import load_reservations; print(load_reservations(limit=5).head())"
```

Or use the CLI:

```bash
uv run python main.py status
```
