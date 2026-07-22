# dash_app/pages/data_update.py
# PAGE 5 - Current model info (READ-ONLY). Shows the served model's tiles, current
# hyperparameters, last walk-forward metrics and the cleaned-history freshness.
# Retraining is deliberately NOT in the app — it is heavy and OVERWRITES the serving
# model, so it runs from the CLI (`uv run python main.py retrain …`). Data update lives
# on Cancellation History and 14-day scoring on Occupancy & Predictions. Logic in
# model_ops.py.

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, callback, dcc, html

from dash_app.backend import model_ops as mo
from dash_app.components import ui

dash.register_page(__name__, path="/data-update", name="Model info",
                   order=4, title="STAYERY · Model info")



# ---------------------------------------------------------------------------
# Small builders
# ---------------------------------------------------------------------------
def _tiles(status: dict):
    n_train = (f"{status['n_train_deploy']:,} bookings" if status.get("n_train_deploy")
               else (f"{status['n_train_person_period']:,} person-periods"
                     if status.get("n_train_person_period") else "unavailable"))
    days_ago = status.get("retrained_days_ago")
    sub_days = f"{days_ago} days ago" if days_ago is not None else None
    return ui.kpi_strip([
        ui.kpi_card("Model status", status.get("status_label") or "-",
                    sub=status.get("label"), accent=True,
                    tooltip="Whether this model's artifact is on disk and how it is served."),
        ui.kpi_card("Last retrained", status.get("retrained_at") or "unavailable",
                    sub=sub_days, tooltip="From the model card (shown in your local time)."),
        ui.kpi_card("Model version", status.get("version") or "-",
                    sub=f"kind: {status.get('kind') or '-'}",
                    tooltip="Retrain date + feature-set fingerprint (roster hash)."),
        ui.kpi_card("Training set", n_train, sub=f"as-of {status.get('asof') or '-'}",
                    tooltip="Static models count bookings; the hazard model counts "
                            "person-period rows (a different unit)."),
    ])


# Plain-language explanations surfaced as ⓘ tooltips (no unexplained numbers).
_HP_HELP = {
    "max_depth": "Maximum tree depth. Higher = more complex trees, more overfitting risk.",
    "learning_rate": "Boosting step size. Lower = slower learning but more robust.",
    "eta": "Boosting step size (same as learning_rate). Lower = more robust.",
    "n_estimators": "Number of boosting rounds (trees).",
    "min_child_weight": "Min summed instance weight per leaf. Higher = more conservative splits.",
    "reg_lambda": "L2 regularisation on leaf weights. Higher = smoother, less overfit.",
    "reg_alpha": "L1 regularisation on leaf weights. Higher = sparser, less overfit.",
    "subsample": "Fraction of rows sampled per tree (<1 adds regularising randomness).",
    "colsample_bytree": "Fraction of features sampled per tree.",
    "gamma": "Minimum loss reduction required to make a split. Higher = more conservative.",
    "C": "Inverse regularisation strength (logistic reg). Lower = stronger regularisation.",
    "max_iter": "Maximum optimisation iterations.",
    "l2_regularization": "L2 penalty (HistGB). Higher = smoother, less overfit.",
    "max_leaf_nodes": "Maximum leaves per tree (HistGB).",
    "learning_rate_init": "Initial learning rate.",
}
_METRIC_HELP = {
    "auc": "ROC-AUC on the leak-free walk-forward  ranking quality (0.5 = coin flip, 1 = perfect).",
    "ap": "Average precision (area under precision–recall). At a ~20% cancel base rate, "
          "values well below 1 are normal.",
    "brier": "Mean squared error of the predicted probabilities. Lower = better calibrated.",
    "cost": "Total € cost at the cost-optimal threshold on the validation fold.",
    "val_ap_person_period": "Average precision for the HAZARD model, computed on person-period "
        "rows (one row per booking per day-until-arrival). On any single day a booking cancels "
        "on only a few of its many days, so the base rate per row is tiny and AP here is "
        "naturally small (≈0.05–0.10). Don't read it as 'bad'  judge the hazard model by its "
        "calibration and the aggregate per-night cancellation forecast, not this number alone.",
}


def _train_rows_panel(rows: list[dict]):
    """Bar list of bookings-per-property in the cleaned training set (a quick comparison)."""
    if not rows:
        return dmc.Text("No cleaned training data yet  build it on the Cancellation "
                        "History page.", size="sm", c="dimmed")
    mx = max((r["rows"] for r in rows), default=1) or 1
    items = []
    for r in rows:
        items.append(dmc.Stack([
            dmc.Group([dmc.Text(r["property"], size="sm"),
                       dmc.Text(f"{r['rows']:,}", size="xs", c="dimmed",
                                style={"fontFamily": "monospace"})],
                      justify="space-between", wrap="nowrap"),
            dmc.Progress(value=r["rows"] / mx * 100, size="sm", radius="sm", color="yellow"),
        ], gap=2))
    return dmc.Stack(items, gap=8)


def _hp_rows(model: str):
    """Hyperparameter rows with a per-parameter ⓘ explanation."""
    pairs = mo.hyperparams_rows(model)
    rows = []
    for i, p in enumerate(pairs):
        name = str(p["param"])
        help_txt = _HP_HELP.get(name)
        label = dmc.Group(
            [dmc.Text(name, size="sm", c="dimmed")]
            + ([ui.info_icon(help_txt)] if help_txt else []),
            gap=4, wrap="nowrap")
        rows.append(dmc.Group([label, dmc.Text(str(p["value"]), size="sm", fw=600,
                                               style={"fontFamily": "monospace"})],
                              justify="space-between", wrap="nowrap"))
        if i < len(pairs) - 1:
            rows.append(dmc.Divider(variant="dotted"))
    return dmc.Stack(rows, gap=6)


def _wf_panel(model: str):
    wf = mo.latest_walkforward(model)
    if not wf:
        return dmc.Text("No walk-forward metrics stored for this model yet.",
                        size="sm", c="dimmed")
    label = {"auc": "ROC-AUC", "ap": "Avg. precision", "brier": "Brier",
             "cost": "Cost @ cost-optimal thr", "val_ap_person_period": "Val AP (person-period)"}
    rows = []
    for k, cell in wf.items():
        mean, std = cell.get("mean"), cell.get("std")
        val = "-" if mean is None else (f"{mean:.4g}" + (f" ± {std:.2g}" if std is not None else ""))
        name_row = dmc.Group(
            [dmc.Text(label.get(k, k), size="sm", c="dimmed")]
            + ([ui.info_icon(_METRIC_HELP[k])] if k in _METRIC_HELP else []),
            gap=4, wrap="nowrap")
        rows.append(dmc.Group([name_row, dmc.Text(val, size="sm", fw=600,
                                                  style={"fontFamily": "monospace"})],
                              justify="space-between", wrap="nowrap"))
    return dmc.Stack(rows, gap=6)


def _clean_history_line():
    """One-line freshness note for the cleaned history that the served model was trained on
    (read straight from the parquet, so it stays accurate after a history rebuild)."""
    s = mo.clean_history_status()
    if not s.get("exists"):
        return dmc.Text("Cleaned history not built yet — rebuild it on the Cancellation "
                        "History page.", size="xs", c="orange")
    rows = f"{s['rows']:,} bookings" if s.get("rows") else "—"
    return dmc.Text(f"Training data: {rows} · through {s.get('data_through') or '?'} · "
                    f"cleaned history last rebuilt {s.get('rebuilt_at') or '?'}.",
                    size="xs", c="dimmed")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def layout(**_kwargs):
    opts = mo.scoring_model_options()
    default_model = opts[0]["value"] if opts else "hazard"

    header = dmc.Group([
        dmc.Group([
            dmc.Title("Model info", order=3),
            dmc.Badge("Current served model · read-only", color="gray",
                      variant="light", radius="sm"),
        ], gap="sm", align="center"),
        dmc.Text("View the served model; retraining runs from the CLI (see below). "
                 "Data updates and scoring live on their own pages.", size="sm", c="dimmed"),
    ], justify="space-between", align="center", wrap="wrap", mb="xs")

    stores = html.Div([
        dcc.Store(id="du-info-version", data=0),
    ])

    # 3) Model info (tiles, hyperparams, metrics).
    info_card = dmc.Card([
        dmc.Text("Current model", fw=600, size="sm", mb=6),
        html.Div(id="du-tiles", children=dmc.Skeleton(height=110, radius="lg")),
        html.Div(id="du-cadence", className="mt-2"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    hp_card = dmc.Card([
        dmc.Group([dmc.Text("Current hyperparameters", fw=600, size="sm"),
                   ui.info_icon("Frozen hyperparameters from the model card. A plain retrain "
                                "reuses them; 'Re-estimate' re-searches them.")], gap=6),
        dmc.Space(h=8),
        html.Div(id="du-hp", children=dmc.Skeleton(height=140, radius="md")),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    metrics_card = dmc.Card([
        dmc.Group([dmc.Text("Last walk-forward metrics", fw=600, size="sm"),
                   ui.info_icon("Honest out-of-time metrics stored in the model card. Hover "
                                "each metric's ⓘ for what it means.")], gap=6),
        dmc.Space(h=8),
        html.Div(id="du-metrics"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    trainrows_card = dmc.Card([
        dmc.Group([dmc.Text("Training set by location", fw=600, size="sm"),
                   ui.info_icon("Bookings per location in the cleaned training set  a quick "
                                "comparison of how much history each location contributes. "
                                "Rebuild it on the Cancellation History page.")], gap=6),
        dmc.Space(h=8),
        html.Div(id="du-trainrows", children=dmc.Skeleton(height=200, radius="md")),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    # 4) RETRAIN - CLI only. Retraining is heavy and OVERWRITES the serving model, so it
    # is deliberately NOT a button here. This card just documents the CLI commands.
    retrain_card = dmc.Card([
        dmc.Group([dmc.Text("Retraining (command line)", fw=600, size="sm"),
                   ui.info_icon("Retraining refits on all resolved data and OVERWRITES the "
                                "serving artifact. It is heavy, so it runs from the CLI — not "
                                "from this page. Scoring and data updates stay in the app.")],
                  gap=6),
        dmc.Text("Retraining is not available as a button (on purpose). Run it from a "
                 "terminal in the project root:", size="xs", c="dimmed"),
        html.Div(_clean_history_line(), id="du-clean-freshness", className="mt-1"),
        html.Pre(
            "# refit — reuse the frozen hyperparameters (fast)\n"
            "uv run python main.py retrain --model hazard\n\n"
            "# retune — larger hyperparameter search (slower, more thorough)\n"
            "uv run python main.py retrain --model hazard --retune\n\n"
            "# train on fresh data — rebuild the cleaned history first, then retrain\n"
            "uv run python main.py update          # BigQuery pull + rescore\n"
            "uv run python main.py retrain --model hazard\n\n"
            "# dry run — fit + metrics, writes nothing\n"
            "uv run python main.py retrain --model hazard --dry-run",
            style={"background": "#f4f4f5", "padding": "12px", "borderRadius": "8px",
                   "fontSize": "12px", "overflowX": "auto", "whiteSpace": "pre-wrap",
                   "fontFamily": "monospace", "margin": "8px 0 0 0"}),
        dmc.Text("Models: hazard (served), xgboost (fallback), logreg / histgb (baselines). "
                 "After retraining, re-score the next 14 days on Occupancy & Predictions to "
                 "serve the new model.", size="xs", c="dimmed", mt=8),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    # Page-level model selector — drives the info cards below (view any model's card,
    # hyperparameters and metrics). Not tied to retraining, which is CLI-only.
    controls = dmc.Paper(dmc.Group([
        dmc.Select(id="du-model", label="Model", data=opts, value=default_model,
                   clearable=False, style={"width": "220px"},
                   leftSection=html.I(className="bi bi-cpu")),
        dmc.Text("Pick a model to inspect its card, hyperparameters and walk-forward "
                 "metrics.", size="xs", c="dimmed"),
    ], align="flex-end", gap="md", wrap="wrap"), p="md", radius="lg", withBorder=True)

    top_grid = dmc.Grid([
        dmc.GridCol(info_card, span={"base": 12, "md": 12}),
        dmc.GridCol(hp_card, span={"base": 12, "md": 6}),
        dmc.GridCol(metrics_card, span={"base": 12, "md": 6}),
        dmc.GridCol(trainrows_card, span={"base": 12, "md": 12}),
    ], gutter="md")

    return dmc.Stack([header, stores, controls, top_grid, retrain_card], gap="md")


# ---------------------------------------------------------------------------
# Model info (instant, card-read only)
# ---------------------------------------------------------------------------
@callback(
    Output("du-tiles", "children"),
    Output("du-cadence", "children"),
    Output("du-hp", "children"),
    Output("du-metrics", "children"),
    Output("du-trainrows", "children"),
    Output("du-clean-freshness", "children"),
    Input("du-model", "value"),
    Input("du-info-version", "data"),
)
def _fill_info(model, _version):
    model = model or "hazard"
    status = mo.model_status(model)
    hint = mo.cadence_hint(model)
    color = {"due": "yellow"}.get(hint["level"], "gray")
    cadence = dmc.Alert(hint["text"], color=color, variant="light", radius="md",
                        icon=html.I(className="bi bi-clock-history"))
    return (_tiles(status), cadence, _hp_rows(model), _wf_panel(model),
            _train_rows_panel(mo.training_rows_by_property()), _clean_history_line())


# Retraining is intentionally CLI-only (see the "Retraining (command line)" card) — the
# heavy, model-overwriting action does not belong in a shared web app, so there is no
# retrain button and no job poller here anymore.
