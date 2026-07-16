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

import time

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, html, no_update

import src
from dash_app.backend import jobs
from dash_app.backend import model_ops as mo
from dash_app.components import ui

dash.register_page(__name__, path="/data-update", name="Update & Retraining",
                   order=4, title="STAYERY · Update & Retraining")

_HIDDEN = {"display": "none"}
_SHOWN = {"display": "block"}


# ---------------------------------------------------------------------------
# Small builders
# ---------------------------------------------------------------------------
def _progress_row(prefix: str):
    return html.Div(
        dmc.Stack([
            html.Progress(id=f"{prefix}-bar", value="0", max="100",
                          style={"width": "100%", "height": "8px"}),
            dmc.Text(id=f"{prefix}-msg", size="xs", c="dimmed"),
        ], gap=4),
        id=f"{prefix}-wrap", style=_HIDDEN, className="mt-2",
    )


def _kv_rows(pairs: list[dict], key: str = "param", val: str = "value"):
    rows = []
    for i, p in enumerate(pairs):
        rows.append(dmc.Group([
            dmc.Text(str(p[key]), size="sm", c="dimmed"),
            dmc.Text(str(p[val]), size="sm", fw=600, style={"fontFamily": "monospace"}),
        ], justify="space-between", wrap="nowrap"))
        if i < len(pairs) - 1:
            rows.append(dmc.Divider(variant="dotted"))
    return dmc.Stack(rows, gap=6)


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
        rows.append({"param": label.get(k, k), "value": val})
    return _kv_rows(rows)


def _err_alert(text: str):
    return dmc.Alert(dmc.Text(str(text), size="sm"), color="red", variant="light",
                     title="Failed", icon=html.I(className="bi bi-exclamation-triangle"))


def _running_note(state: dict) -> str:
    started = state.get("started")
    since = src.fmt_ts_local(started * 1_000_000_000) if started else None  # epoch s -> ns
    mins = f" · running {int((time.time() - started) // 60)} min" if started else ""
    return ((f"Started {since}" if since else "Running") + mins
            + " - keeps running if you switch pages; the bar picks up again here.")


# ---------------------------------------------------------------------------
# Per-job SUCCESS renderers
# ---------------------------------------------------------------------------
def _retrain_result(res: dict):
    agg = res.get("walk_forward_aggregate") or {}
    auc = agg.get("auc", {}).get("mean") if isinstance(agg.get("auc"), dict) else None
    change = res.get("feature_change") or {}
    return dmc.Alert(dmc.Stack([
        dmc.Text(f"Retrained {mo.model_label(res.get('model', '?'))} ({res.get('mode')}) "
                 f"on {res.get('n_train_deploy') or '?'} bookings.", fw=600, size="sm"),
        dmc.Text((f"Walk-forward AUC ≈ {auc:.3f}. " if auc is not None else "")
                 + (f"Feature set changed: +{change.get('added')} −{change.get('removed')}. "
                    if change.get("changed") else "Feature set unchanged. ")
                 + f"Finished {res.get('retrained_at') or '?'}.", size="xs", c="dimmed"),
        dmc.Text("Re-score the next 14 days on Occupancy & Predictions to use the new model.",
                 size="xs", c="dimmed"),
    ], gap=6), color="green", variant="light", icon=html.I(className="bi bi-check-circle"))


_RESULT_RENDERERS = {"retrain": _retrain_result}


def _job_view(name: str, idle_text: str):
    """(bar_value, msg, wrap_style, result_children, is_running) for one job."""
    st = jobs.read(name)
    status = st.get("status", "idle")
    if status == "running":
        bar = str(int(float(st.get("progress", 0)) * 100))
        return bar, st.get("message", ""), _SHOWN, dmc.Text(_running_note(st), size="xs",
                                                            c="dimmed"), True
    if status == "error":
        return "0", "", _HIDDEN, _err_alert(st.get("error", "unknown error")), False
    if status == "done":
        render = _RESULT_RENDERERS.get(name, lambda r: dmc.Text(str(r), size="xs"))
        return "0", "", _HIDDEN, render(st.get("result") or {}), False
    return "0", "", _HIDDEN, dmc.Text(idle_text, size="xs", c="dimmed"), False


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
                   ui.info_icon("Honest out-of-time metrics stored in the model card.")], gap=6),
        dmc.Space(h=8),
        html.Div(id="du-metrics"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    # 4) RETRAIN - controls, progress and result in ONE card + confirm modal.
    retrain_card = dmc.Card([
        dmc.Group([dmc.Text("Retraining (on demand)", fw=600, size="sm"),
                   ui.info_icon("Refits the selected model on all resolved data and OVERWRITES "
                                "the serving artifact - hence the confirmation. Cannot be "
                                "cancelled once started; it runs to completion (or error) and "
                                "survives page changes.")], gap=6),
        dmc.Text("Default keeps the frozen hyperparameters; re-estimating searches them "
                 "again (slower). Trains on the cleaned history — rebuild that on the "
                 "Cancellation History page first if you need fresh data.",
                 size="xs", c="dimmed"),
        dmc.Group([
            dmc.Select(id="du-model", label="Model", data=opts, value=default_model,
                       clearable=False, style={"width": "220px"},
                       leftSection=html.I(className="bi bi-cpu")),
            dmc.Button("Retrain model…", id="du-retrain-btn", size="sm", variant="light",
                       leftSection=html.I(className="bi bi-gear-wide-connected"), mt=22),
            dmc.Checkbox(id="du-retune", checked=False, size="sm",
                         label="Re-estimate hyperparameters (slower)", mt=26),
        ], gap="md", align="flex-end", mt=8),
        _progress_row("du-retrain"),
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
    return _tiles(status), cadence, _kv_rows(mo.hyperparams_rows(model)), _wf_panel(model)


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
    prevent_initial_call=True,
)
def _confirm_retrain(_n, model, retune):
    st = mo.model_status(model or "hazard")
    mode = "retune (hyperparameter search)" if retune else "refit (frozen hyperparameters)"
    body = dmc.Stack([
        dmc.Text(f"This will retrain {st.get('label')} and OVERWRITE the serving artifact.",
                 size="sm"),
        dmc.Text(f"Mode: {mode}. Last retrained: {st.get('retrained_at') or 'never'}.",
                 size="sm", c="dimmed"),
        dmc.Text("The job cannot be cancelled once started; it survives page changes.",
                 size="xs", c="dimmed"),
    ], gap=6)
    return True, body


@callback(
    Output("du-retrain-modal", "opened", allow_duplicate=True),
    Output("du-kick", "data", allow_duplicate=True),
    Input("du-retrain-confirm", "n_clicks"),
    State("du-model", "value"),
    State("du-retune", "checked"),
    prevent_initial_call=True,
)
def _start_retrain(n, model, retune):
    jobs.start("retrain", mo.retrain_job, model or "hazard", bool(retune))
    return False, n


@callback(
    Output("du-retrain-modal", "opened", allow_duplicate=True),
    Input("du-retrain-abort", "n_clicks"),
    prevent_initial_call=True,
)
def _abort_retrain(_n):
    return False


# ---------------------------------------------------------------------------
# THE poller - renders the retrain job from its status file, bumps the model-info
# version once on completion, and disables the retrain button while it runs.
# ---------------------------------------------------------------------------
@callback(
    Output("du-retrain-bar", "value"), Output("du-retrain-msg", "children"),
    Output("du-retrain-wrap", "style"), Output("du-retrain-result", "children"),
    Output("du-retrain-btn", "disabled"),
    Output("du-info-version", "data"),
    Output("du-jobs-seen", "data"),
    Input("du-poll", "n_intervals"),
    Input("du-kick", "data"),
    State("du-jobs-seen", "data"),
    State("du-info-version", "data"),
)
def _poll(_n, _kick, seen, info_v):
    seen = dict(seen or {})
    r = _job_view("retrain", "Idle.")
    new_info = no_update
    st = jobs.read("retrain")
    fin = st.get("finished")
    if fin and seen.get("retrain") != fin:
        seen["retrain"] = fin
        if st.get("status") == "done":
            new_info = (info_v or 0) + 1     # refresh the model tiles/metrics once
    return (r[0], r[1], r[2], r[3], r[4], new_info, seen)
