# dash_app/pages/data_update.py
# PAGE 5 - Update & Retraining. Three long-running actions, each in ITS OWN card
# with button, progress bar and result TOGETHER (no more "button on top, bar at
# the bottom"):
#   1. Update data & scores - ONE strict BigQuery pull per table + scoring
#   2. Retrain (confirm-modal, since it overwrites the serving artifact)
#   3. Model-Performance artifacts (eval/SHAP) - explicit buttons, no auto-start
#
# All three run through dash_app.backend.jobs (file-backed threads): a dcc.Interval
# polls the job files, so progress SURVIVES page changes, a dead worker shows a
# loud error instead of an eternal loading bar, and the rest of the app stays
# responsive. Business logic lives in backend/model_ops.py.

from __future__ import annotations

import time

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, html, no_update

import src
from dash_app.backend import jobs
from dash_app.backend import model_ops as mo
from dash_app.backend import model_performance as mp   # read_cost_params (shared cost-store)
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
def _update_result(res: dict):
    b = res.get("buckets") or {}
    return dmc.Alert(dmc.Stack([
        dmc.Text(f"Fresh data pulled and {res.get('scored_rows', 0):,} bookings scored "
                 f"with {res.get('model_label') or res.get('model_used')} "
                 f"in {res.get('elapsed_s', '?')}s.", fw=600, size="sm"),
        dmc.Group([
            dmc.Badge(f"High: {b.get('high', 0):,}", color="red", variant="light"),
            dmc.Badge(f"Medium: {b.get('medium', 0):,}", color="yellow", variant="light"),
            dmc.Badge(f"Low: {b.get('low', 0):,}", color="green", variant="light"),
        ], gap="xs"),
        dmc.Text(f"BigQuery: {res.get('reservations_rows', 0):,} reservations + "
                 f"{res.get('perf_rows', 0):,} performance rows · data as of "
                 f"{res.get('data_as_of') or '?'} · finished {res.get('finished') or '?'}",
                 size="xs", c="dimmed"),
    ], gap=6), color="green", variant="light", icon=html.I(className="bi bi-check-circle"))


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
        dmc.Text("Click 'Update data & scores' to re-score with the new model.",
                 size="xs", c="dimmed"),
    ], gap=6), color="green", variant="light", icon=html.I(className="bi bi-check-circle"))


def _artifacts_result(res: dict):
    cov = mo.eval_coverage()
    parts = [f"eval ready {len(cov['have'])}/{len(cov['all'])}"]
    if res.get("built_eval"):
        parts.append("built eval: " + ", ".join(res["built_eval"]))
    if res.get("built_shap"):
        parts.append("built SHAP: " + ", ".join(res["built_shap"]))
    if res.get("rebuilt"):
        parts.append("rebuilt: " + ", ".join(res["rebuilt"]))
    errs = res.get("errors") or []
    if errs:
        return _err_alert(" · ".join(parts) + " - errors: " + "; ".join(errs))
    return dmc.Alert(" · ".join(parts), color="green", variant="light",
                     icon=html.I(className="bi bi-layers"))


_RESULT_RENDERERS = {"update": _update_result, "retrain": _retrain_result,
                     "artifacts": _artifacts_result}


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
        dmc.Text("One BigQuery update for everything; retrain on demand.",
                 size="sm", c="dimmed"),
    ], justify="space-between", align="center", wrap="wrap", mb="xs")

    stores = html.Div([
        dcc.Store(id="du-info-version", data=0),
        dcc.Store(id="du-scored-version", data=0),
        dcc.Store(id="du-jobs-seen", data={}),
        dcc.Store(id="du-kick", data=0),
        # THE page heartbeat: polls Data/jobs/*.json so progress/result render
        # from server truth - page changes and app restarts included.
        dcc.Interval(id="du-poll", interval=1200, n_intervals=0),
        dcc.Download(id="du-scored-download"),
    ])

    # 1) DATA UPDATE - button, progress and result together.
    update_card = dmc.Card([
        dmc.Group([
            dmc.Group([dmc.Text("Update data & scores", fw=600, size="sm"),
                       ui.info_icon("ONE strict BigQuery pull per table (full reservations "
                                    "history + property performance), then the next 14 days "
                                    "are scored from the fresh data. No silent cache fallback: "
                                    "if BigQuery fails, this fails loudly and the data is "
                                    "explicitly not fresh.")], gap=6),
            dmc.Group([
                dmc.Button("Test connection", id="du-bqtest-btn", size="xs", variant="subtle",
                           leftSection=html.I(className="bi bi-plug")),
                dmc.Text(id="du-bq-status", size="xs", c="dimmed"),
            ], gap="xs", align="center"),
        ], justify="space-between", align="center", wrap="wrap"),
        dmc.Space(h=8),
        dmc.Group([
            dmc.Select(id="du-model", label="Scoring model", data=opts, value=default_model,
                       clearable=False, style={"width": "230px"},
                       leftSection=html.I(className="bi bi-cpu")),
            dmc.Button("Update data & scores", id="du-update-btn", size="sm", variant="filled",
                       leftSection=html.I(className="bi bi-arrow-clockwise"), mt=22),
        ], align="flex-start", gap="md", wrap="wrap"),
        _progress_row("du-upd"),
        html.Div(id="du-upd-result", className="mt-2"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    # 2) Scored set - table + compact/full export.
    scored_card = dmc.Card([
        dmc.Group([
            dmc.Group([dmc.Text("Scored bookings - next 14 days", fw=600, size="sm"),
                       ui.info_icon("The current scored set (highest cancel risk first). "
                                    "'CSV (compact)' has the columns a revenue manager reads; "
                                    "'CSV (all columns)' includes every engineered feature.")],
                      gap=6),
            dmc.Group([dmc.Text(id="du-scored-summary", size="xs", c="dimmed"),
                       dmc.Button("CSV (compact)", id="du-dl-slim-btn", size="xs", variant="light",
                                  leftSection=html.I(className="bi bi-download")),
                       dmc.Button("CSV (all columns)", id="du-dl-full-btn", size="xs",
                                  variant="subtle")],
                      gap="sm", align="center"),
        ], justify="space-between", align="center", wrap="wrap"),
        dmc.Space(h=8),
        dag.AgGrid(
            id="du-scored-grid", rowData=[],
            columnDefs=[
                {"headerName": "Location", "field": "property_name", "flex": 2},
                {"headerName": "Arrival", "field": "arrival", "flex": 1},
                {"headerName": "P(cancel) %", "field": "cancel_pct", "flex": 1,
                 "type": "numericColumn"},
                {"headerName": "Risk", "field": "risk_bucket", "flex": 1},
            ],
            defaultColDef={"sortable": True, "filter": True, "resizable": True},
            columnSize="responsiveSizeToFit", style={"height": "340px"},
            dashGridOptions={"animateRows": True, "pagination": True,
                             "paginationAutoPageSize": True}),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

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
                 "again (slower).", size="xs", c="dimmed"),
        dmc.Group([
            dmc.Button("Retrain model…", id="du-retrain-btn", size="sm", variant="light",
                       leftSection=html.I(className="bi bi-gear-wide-connected")),
            dmc.Checkbox(id="du-retune", checked=False, size="sm",
                         label="Re-estimate hyperparameters (slower)"),
        ], gap="md", align="center", mt=8),
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

    # 5) ARTIFACTS - explicit builds only (no auto-start on page load: the hazard
    # eval takes minutes and used to wedge the page).
    artifacts_card = dmc.Card([
        dmc.Group([
            dmc.Group([dmc.Text("Model-Performance artifacts", fw=600, size="sm"),
                       ui.info_icon("The XAI page reads pre-built eval/SHAP artifacts. Build "
                                    "them here explicitly - nothing starts automatically. The "
                                    "hazard eval takes several minutes; the job keeps running "
                                    "if you leave the page.")], gap=6),
            dmc.Group([
                dmc.Button("Build missing (eval)", id="du-art-missing-btn", size="xs",
                           variant="light", leftSection=html.I(className="bi bi-layers")),
                dmc.Button("Build all (eval + SHAP)", id="du-art-all-btn", size="xs",
                           variant="subtle"),
            ], gap="xs"),
        ], justify="space-between", align="center", wrap="wrap"),
        _progress_row("du-art"),
        html.Div(id="du-art-result", className="mt-2"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    top_grid = dmc.Grid([
        dmc.GridCol(info_card, span={"base": 12, "md": 12}),
        dmc.GridCol(hp_card, span={"base": 12, "md": 6}),
        dmc.GridCol(metrics_card, span={"base": 12, "md": 6}),
    ], gutter="md")

    return dmc.Stack([header, stores, update_card, scored_card, top_grid,
                      retrain_card, artifacts_card], gap="md")


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
# Job starters (plain, fast callbacks - the poller renders the progress)
# ---------------------------------------------------------------------------
@callback(
    Output("du-kick", "data", allow_duplicate=True),
    Input("du-update-btn", "n_clicks"),
    State("du-model", "value"),
    State("cost-store", "data"),
    prevent_initial_call=True,
)
def _start_update(n, model, cost_store):
    walk, empty, high, mult = mp.read_cost_full(cost_store)
    eff_walk = src.effective_walk_cost(walk, high, mult)   # apply high-demand multiplier
    jobs.start("update", mo.update_all, model, eff_walk, empty)
    return n


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


@callback(
    Output("du-kick", "data", allow_duplicate=True),
    Input("du-art-missing-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _start_art_missing(n):
    jobs.start("artifacts", mo.artifacts_job, False)
    return n


@callback(
    Output("du-kick", "data", allow_duplicate=True),
    Input("du-art-all-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _start_art_all(n):
    jobs.start("artifacts", mo.artifacts_job, True)
    return n


# ---------------------------------------------------------------------------
# THE poller - renders all three jobs from the status files, bumps versions on
# completion, and disables buttons while (conflicting) jobs run.
# ---------------------------------------------------------------------------
@callback(
    Output("du-upd-bar", "value"), Output("du-upd-msg", "children"),
    Output("du-upd-wrap", "style"), Output("du-upd-result", "children"),
    Output("du-retrain-bar", "value"), Output("du-retrain-msg", "children"),
    Output("du-retrain-wrap", "style"), Output("du-retrain-result", "children"),
    Output("du-art-bar", "value"), Output("du-art-msg", "children"),
    Output("du-art-wrap", "style"), Output("du-art-result", "children"),
    Output("du-update-btn", "disabled"), Output("du-retrain-btn", "disabled"),
    Output("du-art-missing-btn", "disabled"), Output("du-art-all-btn", "disabled"),
    Output("du-scored-version", "data"), Output("du-info-version", "data"),
    Output("du-jobs-seen", "data"),
    Input("du-poll", "n_intervals"),
    Input("du-kick", "data"),
    State("du-jobs-seen", "data"),
    State("du-scored-version", "data"),
    State("du-info-version", "data"),
)
def _poll(_n, _kick, seen, scored_v, info_v):
    seen = dict(seen or {})
    u = _job_view("update", "Idle - pulls BigQuery and re-scores on demand.")
    r = _job_view("retrain", "Idle.")
    a = _job_view("artifacts", _artifact_idle_text())

    # Version bumps exactly once per finished job (finished timestamp as marker).
    new_scored, new_info = no_update, no_update
    for name, key in (("update", "scored"), ("retrain", "info")):
        st = jobs.read(name)
        fin = st.get("finished")
        if st.get("status") == "done" and fin and seen.get(name) != fin:
            seen[name] = fin
            if key == "scored":
                new_scored = (scored_v or 0) + 1
            else:
                new_info = (info_v or 0) + 1
        elif fin and seen.get(name) != fin:
            seen[name] = fin           # errors: remember, but nothing to refresh

    upd_running, ret_running, art_running = u[4], r[4], a[4]
    busy = upd_running or ret_running
    return (u[0], u[1], u[2], u[3],
            r[0], r[1], r[2], r[3],
            a[0], a[1], a[2], a[3],
            busy, busy or art_running,
            art_running or ret_running, art_running or ret_running,
            new_scored, new_info, seen)


def _artifact_idle_text() -> str:
    cov = mo.eval_coverage()
    missing = [m for m in cov["all"] if m not in cov["have"]]
    if not missing:
        return f"All evaluation artifacts present ({len(cov['have'])}/{len(cov['all'])})."
    return ("Missing eval artifacts: " + ", ".join(missing)
            + " - the XAI page shows empty charts for these until built.")


# ---------------------------------------------------------------------------
# BigQuery connection test (synchronous - a COUNT(*) probe, seconds)
# ---------------------------------------------------------------------------
@callback(
    Output("du-bq-status", "children"),
    Input("du-bqtest-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _bq_test(_n):
    res = src.bigquery_healthcheck()
    if res["ok"]:
        return dmc.Text("✓ " + res["detail"], size="xs", c="green")
    return dmc.Text("✗ " + res["detail"], size="xs", c="red")


# ---------------------------------------------------------------------------
# Scored table + exports
# ---------------------------------------------------------------------------
@callback(
    Output("du-scored-grid", "rowData"),
    Output("du-scored-summary", "children"),
    Input("du-scored-version", "data"),
)
def _fill_scored(_v):
    ov = mo.scored_overview()
    if ov["n"] == 0:
        return [], "No scored set yet - click 'Update data & scores'."
    summ = (f"{ov['n']:,} bookings · {mo.model_label(ov['model_used']) if ov['model_used'] else '-'} "
            f"· high {ov['high']:,} / medium {ov['medium']:,} / low {ov['low']:,}"
            + (f" · scored {ov['scored_at']}" if ov["scored_at"] else ""))
    return ov["rows"], summ


@callback(
    Output("du-scored-download", "data"),
    Input("du-dl-slim-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _download_slim(_n):
    df = mo.scored_export_frame(slim=True)
    if df is None or df.empty:
        return no_update
    return dcc.send_data_frame(df.to_csv, "scored_upcoming_compact.csv", index=False)


@callback(
    Output("du-scored-download", "data", allow_duplicate=True),
    Input("du-dl-full-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _download_full(_n):
    df = mo.scored_export_frame(slim=False)
    if df is None or df.empty:
        return no_update
    return dcc.send_data_frame(df.to_csv, "scored_upcoming_full.csv", index=False)
