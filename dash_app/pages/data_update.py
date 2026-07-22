# dash_app/pages/data_update.py
# PAGE 5 - Retraining & hyperparameters ONLY. Data update moved to Cancellation
# History (rebuild the cleaned history) and 14-day scoring to Occupancy & Predictions;
# building the Model-Performance eval/SHAP artifacts lives on the Model Performance
# page. What remains here: the current-model tiles + hyperparameters + walk-forward
# metrics, and the on-demand Retrain action (confirm-modal, overwrites the serving
# artifact). Retrain runs through dash_app.backend.jobs (file-backed thread): a
# dcc.Interval polls the job file, so progress SURVIVES page changes and a dead
# worker shows a loud error instead of an eternal loading bar. Logic in model_ops.py.

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, html, no_update

from dash_app.backend import jobs
from dash_app.backend import model_ops as mo
from dash_app.components import ui

dash.register_page(__name__, path="/data-update", name="Update & Retraining",
                   order=4, title="STAYERY · Update & Retraining")



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


def _err_alert(text: str):
    return dmc.Alert(dmc.Text(str(text), size="sm"), color="red", variant="light",
                     title="Failed", icon=html.I(className="bi bi-exclamation-triangle"))


# ---------------------------------------------------------------------------
# Per-job SUCCESS renderers
# ---------------------------------------------------------------------------
def _retrain_result(res: dict):
    agg = res.get("walk_forward_aggregate") or {}
    auc = agg.get("auc", {}).get("mean") if isinstance(agg.get("auc"), dict) else None
    change = res.get("feature_change") or {}
    n_books = res.get("n_train_deploy")
    books_txt = f"{n_books:,}" if isinstance(n_books, (int, float)) and n_books else "?"
    n_pp = res.get("n_train_person_period")
    pp_txt = f" ({n_pp:,} person-periods)" if isinstance(n_pp, (int, float)) and n_pp else ""
    return dmc.Alert(dmc.Stack([
        dmc.Text(f"Retrained {mo.model_label(res.get('model', '?'))} ({res.get('mode')}) "
                 f"on {books_txt} bookings{pp_txt}.", fw=600, size="sm"),
        dmc.Text((f"Walk-forward AUC ≈ {auc:.3f}. " if auc is not None else "")
                 + (f"Feature set changed: +{change.get('added')} −{change.get('removed')}. "
                    if change.get("changed") else "Feature set unchanged. ")
                 + f"Finished {res.get('retrained_at') or '?'}.", size="xs", c="dimmed"),
        dmc.Text("Re-score the next 14 days on Occupancy & Predictions to use the new model.",
                 size="xs", c="dimmed"),
    ], gap=6), color="green", variant="light", icon=html.I(className="bi bi-check-circle"))


def _clean_history_line():
    """One-line freshness note for the cleaned history that Retrain trains on, so the user
    can see at a glance whether to tick 'update history first'."""
    s = mo.clean_history_status()
    if not s.get("exists"):
        return dmc.Text("Cleaned history not built yet — tick 'update history first' below, "
                        "or rebuild it on the Cancellation History page.",
                        size="xs", c="orange")
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
            dmc.Title("Update & retraining", order=3),
            dmc.Badge("Data & model · jobs survive page changes", color="gray",
                      variant="light", radius="sm"),
        ], gap="sm", align="center"),
        dmc.Text("Retrain the serving model on demand; data updates live on their "
                 "own pages now.", size="sm", c="dimmed"),
    ], justify="space-between", align="center", wrap="wrap", mb="xs")

    stores = html.Div([
        dcc.Store(id="du-info-version", data=0),
        dcc.Store(id="du-jobs-seen", data={}),
        dcc.Store(id="du-kick", data=0),
        # THE page heartbeat: polls Data/jobs/*.json so progress/result render
        # from server truth - page changes and app restarts included.
        dcc.Interval(id="du-poll", interval=1200, n_intervals=0),
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

    # 4) RETRAIN - controls, progress and result in ONE card + confirm modal.
    retrain_card = dmc.Card([
        dmc.Group([dmc.Text("Retraining (on demand)", fw=600, size="sm"),
                   ui.info_icon("Refits the selected model on all resolved data and OVERWRITES "
                                "the serving artifact - hence the confirmation. Cannot be "
                                "cancelled once started; it runs to completion (or error) and "
                                "survives page changes.")], gap=6),
        dmc.Text("Default keeps the frozen hyperparameters; re-estimating searches them "
                 "again (slower). Trains on the cleaned history — tick 'update history "
                 "first' to pull fresh data from BigQuery before retraining.",
                 size="xs", c="dimmed"),
        html.Div(_clean_history_line(), id="du-clean-freshness", className="mt-1"),
        dmc.Group([
            dmc.Select(id="du-model", label="Model", data=opts, value=default_model,
                       clearable=False, style={"width": "220px"},
                       leftSection=html.I(className="bi bi-cpu")),
            dmc.Button("Retrain model…", id="du-retrain-btn", size="sm", variant="light",
                       leftSection=html.I(className="bi bi-gear-wide-connected"), mt=22),
            dmc.Stack([
                dmc.Checkbox(id="du-retune", checked=False, size="sm",
                             label="Re-estimate hyperparameters (slower)"),
                dmc.Checkbox(id="du-update-first", checked=False, size="sm",
                             label="Update history from BigQuery first (slower)"),
            ], gap=6, mt=22),
        ], gap="md", align="flex-end", mt=8),
        ui.two_stage_loader("du-retrain", "Retrain + evaluate", "Explanations"),
        html.Div(id="du-retrain-result", className="mt-2"),
        dmc.Modal(id="du-retrain-modal", opened=False, centered=True,
                  title=dmc.Text("Confirm retraining", fw=700),
                  children=[
                      html.Div(id="du-retrain-modal-body"),
                      dmc.Group([
                          dmc.Button("Cancel", id="du-retrain-abort", variant="subtle",
                                     color="gray"),
                          dmc.Button("Yes, retrain now", id="du-retrain-confirm",
                                     color="red", variant="filled"),
                      ], justify="flex-end", mt="md"),
                  ]),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    top_grid = dmc.Grid([
        dmc.GridCol(info_card, span={"base": 12, "md": 12}),
        dmc.GridCol(hp_card, span={"base": 12, "md": 6}),
        dmc.GridCol(metrics_card, span={"base": 12, "md": 6}),
        dmc.GridCol(trainrows_card, span={"base": 12, "md": 12}),
    ], gutter="md")

    return dmc.Stack([header, stores, top_grid, retrain_card], gap="md")


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


# ---------------------------------------------------------------------------
# Retrain (confirm modal -> start -> poller renders progress). This page is now
# retrain + hyperparameters only; data update lives on Cancellation History and
# scoring on Occupancy & Predictions.
# ---------------------------------------------------------------------------
@callback(
    Output("du-retrain-modal", "opened"),
    Output("du-retrain-modal-body", "children"),
    Input("du-retrain-btn", "n_clicks"),
    State("du-model", "value"),
    State("du-retune", "checked"),
    State("du-update-first", "checked"),
    prevent_initial_call=True,
)
def _confirm_retrain(_n, model, retune, update_first):
    st = mo.model_status(model or "hazard")
    mode = "retune (hyperparameter search)" if retune else "refit (frozen hyperparameters)"
    data_note = ("Will FIRST pull fresh reservations from BigQuery and rebuild the cleaned "
                 "history, then retrain on it."
                 if update_first else
                 "Trains on the current cleaned history (no BigQuery pull).")
    body = dmc.Stack([
        dmc.Text(f"This will retrain {st.get('label')} and OVERWRITE the serving artifact.",
                 size="sm"),
        dmc.Text(f"Mode: {mode}. Last retrained: {st.get('retrained_at') or 'never'}.",
                 size="sm", c="dimmed"),
        dmc.Text(data_note, size="sm", c="dimmed"),
        dmc.Text("You can cancel it  it stops at the next stage and the previous model "
                 "stays in place (a fit already running finishes that step first). "
                 "It also survives page changes.", size="xs", c="dimmed"),
    ], gap=6)
    return True, body


@callback(
    Output("du-retrain-modal", "opened", allow_duplicate=True),
    Output("du-kick", "data", allow_duplicate=True),
    Input("du-retrain-confirm", "n_clicks"),
    State("du-model", "value"),
    State("du-retune", "checked"),
    State("du-update-first", "checked"),
    prevent_initial_call=True,
)
def _start_retrain(n, model, retune, update_first):
    jobs.start("retrain", mo.retrain_job, model or "hazard", bool(retune), bool(update_first))
    return False, n


@callback(
    Output("du-retrain-modal", "opened", allow_duplicate=True),
    Input("du-retrain-abort", "n_clicks"),
    prevent_initial_call=True,
)
def _abort_retrain(_n):
    return False


# ---------------------------------------------------------------------------
# THE poller - renders the two-stage retrain loader from its status file, bumps the
# model-info version once on completion, and disables the retrain button while running.
# ---------------------------------------------------------------------------
@callback(
    Output("du-retrain-ring1", "sections"), Output("du-retrain-pct1", "children"),
    Output("du-retrain-ring2", "sections"), Output("du-retrain-pct2", "children"),
    Output("du-retrain-msg", "children"), Output("du-retrain-wrap", "style"),
    Output("du-retrain-cancel", "children"),
    Output("du-retrain-result", "children"), Output("du-retrain-btn", "disabled"),
    Output("du-info-version", "data"),
    Output("du-jobs-seen", "data"),
    Input("du-poll", "n_intervals"),
    Input("du-kick", "data"),
    State("du-jobs-seen", "data"),
    State("du-info-version", "data"),
)
def _poll(_n, _kick, seen, info_v):
    seen = dict(seen or {})
    st = jobs.read("retrain")
    status = st.get("status", "idle")
    if status == "running":
        r1, p1, r2, p2, msg, wrap = ui.two_stage_view(float(st.get("progress", 0)),
                                                      st.get("message", ""), show=True)
        return r1, p1, r2, p2, msg, wrap, "Cancel", no_update, True, no_update, no_update
    r1, p1, r2, p2, msg, wrap = ui.two_stage_view(0, "", show=False)
    fin = st.get("finished")
    new_info = no_update
    if fin and seen.get("retrain") != fin:
        seen["retrain"] = fin
        if status == "done":
            new_info = (info_v or 0) + 1     # refresh the model tiles/metrics once
    if status == "error":
        return r1, p1, r2, p2, msg, wrap, no_update, _err_alert(st.get("error", "unknown error")), False, new_info, seen
    if status == "cancelled":
        return (r1, p1, r2, p2, msg, wrap, no_update,
                dmc.Alert("Retrain cancelled  the previous model is still in place.",
                          color="gray", variant="light", icon=html.I(className="bi bi-x-circle")),
                False, new_info, seen)
    if status == "done":
        return r1, p1, r2, p2, msg, wrap, no_update, _retrain_result(st.get("result") or {}), False, new_info, seen
    return r1, p1, r2, p2, msg, wrap, no_update, no_update, False, no_update, seen


@callback(
    Output("du-retrain-cancel", "children", allow_duplicate=True),
    Input("du-retrain-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def _cancel_retrain(_n):
    jobs.cancel("retrain")
    return "Cancelling…"
