# dash_app/pages/data_update.py
# PAGE 5 — Update & Retraining ("under the hood"). Two jobs: (a) pull the next 14 days of
# arrivals and score them immediately with the chosen model, refreshing the full history in
# the background; (b) retrain a model on demand. The page paints instantly from the model
# card (info tiles), then heavier work streams in via SEPARATE background callbacks so the
# fast scoring never waits on the slow full refresh (progressive-loading requirement).
#
# All real logic lives in dash_app/backend/model_ops.py (import-only, UI-independent — the
# prerequisite for later automation). This file is layout + callback orchestration only.
# Design follows DESIGN_NOTES.md 1:1 (dmc, ui.kpi_card/kpi_strip, theme, skeleton loaders).

from __future__ import annotations

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from dash_app.backend import model_ops as mo
from dash_app.backend import model_performance as mp   # read_cost_params (shared cost-store)
from dash_app.components import ui

dash.register_page(__name__, path="/data-update", name="Update & Retraining",
                   order=4, title="STAYERY · Update & Retraining")


# ---------------------------------------------------------------------------
# Small presentational helpers
# ---------------------------------------------------------------------------
def _report(set_progress):
    """Adapt model_ops' report(msg, frac) to Dash's set_progress(value, max, message)."""
    def report(msg: str, frac: float) -> None:
        set_progress((str(int(max(0.0, min(1.0, frac)) * 100)), "100", msg))
    return report


def _progress_block(bar_id: str, msg_id: str, wrap_id: str):
    """A hidden-by-default progress row (html.Progress + message). `running` reveals it."""
    return html.Div(
        dmc.Stack([
            html.Progress(id=bar_id, value="0", max="100",
                          style={"width": "100%", "height": "8px"}),
            dmc.Text(id=msg_id, size="xs", c="dimmed"),
        ], gap=4),
        id=wrap_id, style={"display": "none"}, className="mt-2",
    )


def _kv_rows(pairs: list[dict], key: str = "param", val: str = "value"):
    """Definition-list style rows (avoids any Table API assumptions)."""
    rows = []
    for i, p in enumerate(pairs):
        rows.append(dmc.Group([
            dmc.Text(str(p[key]), size="sm", c="dimmed"),
            dmc.Text(str(p[val]), size="sm", fw=600,
                     style={"fontFamily": "monospace"}),
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
        ui.kpi_card("Model status", status.get("status_label") or "—",
                    sub=status.get("label"), accent=True,
                    tooltip="Whether this model's artifact is on disk and how it is served "
                            "(default scorer, fallback, or comparison baseline)."),
        ui.kpi_card("Last retrained", status.get("retrained_at") or "unavailable",
                    sub=sub_days,
                    tooltip="From the model card's retrained_at. 'unavailable' means no card "
                            "exists yet (model not trained in this repo)."),
        ui.kpi_card("Model version", status.get("version") or "—",
                    sub=f"kind: {status.get('kind') or '—'}",
                    tooltip="Retrain date + the feature-set fingerprint (roster hash) the "
                            "deployed model was trained on."),
        ui.kpi_card("Training set", n_train, sub=f"as-of {status.get('asof') or '—'}",
                    tooltip="Size of the resolved training set the deployed model was fit on. "
                            "Static models report bookings; the hazard model reports "
                            "person-period rows (a different unit)."),
    ])


def _wf_panel(model: str):
    wf = mo.latest_walkforward(model)
    if not wf:
        return dmc.Text("No walk-forward metrics stored for this model yet.",
                        size="sm", c="dimmed")
    label = {"auc": "ROC-AUC", "ap": "Avg. precision", "brier": "Brier",
             "cost": "Cost @ threshold", "val_ap_person_period": "Val AP (person-period)"}
    rows = []
    for k, cell in wf.items():
        mean = cell.get("mean")
        std = cell.get("std")
        val = "—" if mean is None else (f"{mean:.4g}" + (f" ± {std:.2g}" if std is not None else ""))
        rows.append({"param": label.get(k, k), "value": val})
    return _kv_rows(rows)


# ---------------------------------------------------------------------------
# Layout (callable => model list re-read on each navigation)
# ---------------------------------------------------------------------------
def layout(**_kwargs):
    opts = mo.scoring_model_options()
    default_model = opts[0]["value"] if opts else "hazard"

    header = dmc.Group([
        dmc.Group([
            dmc.Title("Update & retraining", order=3),
            dmc.Badge("Under the hood · data & model", color="gray", variant="light", radius="sm"),
        ], gap="sm", align="center"),
        dmc.Text("Score the next 14 days now; refresh history and retrain in the background.",
                 size="sm", c="dimmed"),
    ], justify="space-between", align="center", wrap="wrap", mb="xs")

    controls = dmc.Paper(dmc.Group([
        dmc.Select(id="du-model", label="Model", data=opts, value=default_model,
                   clearable=False, style={"width": "230px"},
                   leftSection=html.I(className="bi bi-cpu")),
        dmc.Stack([
            dmc.Text("Data", size="sm", fw=600),
            dmc.Button("Update scores (next 14 days)", id="du-update-btn", size="sm",
                       variant="filled", leftSection=html.I(className="bi bi-arrow-clockwise")),
        ], gap=4),
        dmc.Stack([
            dmc.Text("Model", size="sm", fw=600),
            dmc.Group([
                dmc.Button("Retrain model", id="du-retrain-btn", size="sm", variant="light",
                           leftSection=html.I(className="bi bi-gear-wide-connected")),
                dmc.Button("Cancel", id="du-cancel-btn", size="sm", variant="subtle",
                           color="gray", style={"display": "none"}),
            ], gap="xs"),
            dmc.Checkbox(id="du-retune", checked=False, size="sm",
                         label="Re-estimate hyperparameters (slower)"),
        ], gap=4),
    ], align="flex-end", gap="lg", wrap="wrap"), p="md", radius="lg", withBorder=True)

    stores = html.Div([
        dcc.Store(id="du-info-version", data=0),     # bumped after a retrain -> tiles refresh
        dcc.Store(id="du-scored-version", data=0),   # bumped after fast score -> triggers slow path
        dcc.Interval(id="du-metrics-timer", interval=400, n_intervals=0, max_intervals=1),
        dcc.Interval(id="du-warm-timer", interval=900, n_intervals=0, max_intervals=1),
        dcc.Download(id="du-scored-download"),
    ])

    # PRIORITY: the scored set, directly retrievable (table + export). Painted on load from
    # the cached parquet, refreshed after every fast-path score.
    scored_card = dmc.Card([
        dmc.Group([
            dmc.Group([dmc.Text("Scored bookings — next 14 days", fw=600, size="sm"),
                       ui.info_icon("The current scored set the app produced (highest cancel "
                                    "risk first). Directly viewable here and exportable to CSV.")],
                      gap=6),
            dmc.Group([dmc.Text(id="du-scored-summary", size="xs", c="dimmed"),
                       dmc.Button("Download CSV", id="du-download-btn", size="xs", variant="light",
                                  leftSection=html.I(className="bi bi-download"))],
                      gap="sm", align="center"),
        ], justify="space-between", align="center", wrap="wrap"),
        dmc.Space(h=8),
        dcc.Loading(
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
            custom_spinner=dmc.Skeleton(height=340, radius="md")),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    # Auto-warm the Model-Performance eval artifacts (background, missing only) + a manual
    # "build everything" (eval + SHAP) button.
    artifacts_card = dmc.Card([
        dmc.Group([
            dmc.Group([dmc.Text("Model-Performance artifacts", fw=600, size="sm"),
                       ui.info_icon("The XAI page reads pre-built eval (and SHAP) artifacts. "
                                    "Missing eval is warmed automatically in the background; "
                                    "use the button to (re)build everything incl. SHAP.")],
                      gap=6),
            dmc.Button("Build all (eval + SHAP)", id="du-buildall-btn", size="xs", variant="light",
                       leftSection=html.I(className="bi bi-layers")),
        ], justify="space-between", align="center", wrap="wrap"),
        _progress_block("du-warm-bar", "du-warm-msg", "du-warm-wrap"),
        html.Div(id="du-warm-status", className="mt-2",
                 children=dmc.Text("Checking artifacts…", size="xs", c="dimmed")),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    # Info + hyperparameters (instant, from the card)
    info_card = dmc.Card([
        dmc.Text("Current model", fw=600, size="sm", mb=6),
        html.Div(id="du-tiles", children=dmc.Skeleton(height=110, radius="lg")),
        html.Div(id="du-cadence", className="mt-2"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    hp_card = dmc.Card([
        dmc.Group([dmc.Text("Current hyperparameters", fw=600, size="sm"),
                   ui.info_icon("The model's frozen hyperparameters from its card. A plain "
                                "'Retrain' reuses these; tick 'Re-estimate' to re-search them.")],
                  gap=6),
        dmc.Space(h=8),
        html.Div(id="du-hp", children=dmc.Skeleton(height=140, radius="md")),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    metrics_card = dmc.Card([
        dmc.Group([dmc.Text("Last walk-forward metrics", fw=600, size="sm"),
                   ui.info_icon("Loaded lazily after the page paints (non-blocking). Honest "
                                "one-step-ahead metrics stored in the model card.")],
                  gap=6),
        dmc.Space(h=8),
        dcc.Loading(html.Div(id="du-metrics"),
                    custom_spinner=dmc.Skeleton(height=120, radius="md")),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    top_grid = dmc.Grid([
        dmc.GridCol(info_card, span={"base": 12, "md": 12}),
        dmc.GridCol(hp_card, span={"base": 12, "md": 6}),
        dmc.GridCol(metrics_card, span={"base": 12, "md": 6}),
    ], gutter="md")

    # Fast path (14-day scoring) card
    fast_card = dmc.Card([
        dmc.Text("Scoring — next 14 days (fast path)", fw=600, size="sm"),
        dmc.Text("Pulls only bookings arriving in the next 14 days and scores them "
                 "immediately. Writes the shared scored set the Occupancy page reads.",
                 size="xs", c="dimmed"),
        _progress_block("du-fast-bar", "du-fast-msg", "du-fast-wrap"),
        html.Div(id="du-scored-panel", className="mt-2"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    # Slow path (full history refresh) card
    slow_card = dmc.Card([
        dmc.Text("History refresh (background)", fw=600, size="sm"),
        dmc.Text("Re-pulls the full reservations + property-performance history the "
                 "cancellation-rate views rely on. Runs after the fast path; does not block it.",
                 size="xs", c="dimmed"),
        _progress_block("du-slow-bar", "du-slow-msg", "du-slow-wrap"),
        html.Div(id="du-slow-panel", className="mt-2",
                 children=dmc.Text("Idle — starts automatically after an update.",
                                   size="xs", c="dimmed")),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    retrain_card = dmc.Card([
        dmc.Text("Retraining (on demand)", fw=600, size="sm"),
        dmc.Text("Refits the selected model on all resolved data. Default keeps the existing "
                 "hyperparameters; anything unchanged (feature roster) is reused, not recomputed.",
                 size="xs", c="dimmed"),
        _progress_block("du-retrain-bar", "du-retrain-msg", "du-retrain-wrap"),
        html.Div(id="du-retrain-panel", className="mt-2"),
    ], withBorder=True, radius="lg", p="md", shadow="xs")

    action_grid = dmc.Grid([
        dmc.GridCol(fast_card, span={"base": 12, "md": 6}),
        dmc.GridCol(slow_card, span={"base": 12, "md": 6}),
        dmc.GridCol(retrain_card, span={"base": 12, "md": 12}),
    ], gutter="md")

    return dmc.Stack([header, stores, controls, scored_card, top_grid, action_grid,
                      artifacts_card], gap="md")


# ---------------------------------------------------------------------------
# Instant: info tiles + cadence hint + hyperparameters (plain callback, card only)
# ---------------------------------------------------------------------------
@callback(
    Output("du-tiles", "children"),
    Output("du-cadence", "children"),
    Output("du-hp", "children"),
    Input("du-model", "value"),
    Input("du-info-version", "data"),
)
def _fill_info(model, _version):
    model = model or "hazard"
    status = mo.model_status(model)
    hint = mo.cadence_hint(model)
    color = {"due": "yellow", "unknown": "gray", "ok": "gray"}.get(hint["level"], "gray")
    cadence = dmc.Alert(hint["text"], color=color, variant="light", radius="md",
                        icon=html.I(className="bi bi-clock-history"))
    return _tiles(status), cadence, _kv_rows(mo.hyperparams_rows(model))


# ---------------------------------------------------------------------------
# Lazy (non-blocking): last walk-forward metrics — loads after the page paints
# ---------------------------------------------------------------------------
@callback(
    Output("du-metrics", "children"),
    Input("du-metrics-timer", "n_intervals"),
    Input("du-model", "value"),
    Input("du-info-version", "data"),
)
def _fill_metrics(_n, model, _version):
    return _wf_panel(model or "hazard")


# ---------------------------------------------------------------------------
# FAST PATH (background): pull next 14 days + score. Bumps scored-version -> slow path.
# ---------------------------------------------------------------------------
@callback(
    Output("du-scored-panel", "children"),
    Output("du-scored-version", "data"),
    Input("du-update-btn", "n_clicks"),
    State("du-model", "value"),
    State("cost-store", "data"),
    State("du-scored-version", "data"),
    background=True,
    running=[
        (Output("du-update-btn", "disabled"), True, False),
        (Output("du-fast-wrap", "style"), {"display": "block"}, {"display": "none"}),
    ],
    progress=[Output("du-fast-bar", "value"), Output("du-fast-bar", "max"),
              Output("du-fast-msg", "children")],
    progress_default=("0", "100", ""),
    prevent_initial_call=True,
)
def _fast_update(set_progress, _n, model, cost_store, version):
    walk, empty = mp.read_cost_params(cost_store)
    try:
        res = mo.fast_score_next_14d(model, walk=walk, empty=empty,
                                     progress=_report(set_progress))
    except Exception as e:  # noqa: BLE001 — surface the reason, never crash the page
        return dmc.Alert(f"Update failed: {e}", color="red", variant="light",
                         icon=html.I(className="bi bi-exclamation-triangle")), no_update

    if res.get("empty"):
        return dmc.Alert("No upcoming arrivals in the next 14 days to score.",
                         color="gray", variant="light"), (version or 0) + 1

    body = dmc.Stack([
        dmc.Text(f"Scored {res['rows']:,} bookings with "
                 f"{mo.model_label(res['model_used'])} in {res['elapsed_s']}s.", fw=600, size="sm"),
        dmc.Group([
            dmc.Badge(f"High: {res['high']:,}", color="red", variant="light"),
            dmc.Badge(f"Uncertain: {res['uncertain']:,}", color="yellow", variant="light"),
            dmc.Badge(f"Low: {res['low']:,}", color="green", variant="light"),
        ], gap="xs"),
        dmc.Text(f"Decision threshold: "
                 + (f"{res['threshold']:.3f} (cost-optimal for current costs)"
                    if res.get("threshold") is not None else "model default")
                 + f" · scored at {res['scored_at']}", size="xs", c="dimmed"),
    ], gap=6)
    return dmc.Alert(body, color="green", variant="light",
                     icon=html.I(className="bi bi-check-circle")), (version or 0) + 1


# ---------------------------------------------------------------------------
# SLOW PATH (background): chained after the fast path via scored-version.
# ---------------------------------------------------------------------------
@callback(
    Output("du-slow-panel", "children"),
    Input("du-scored-version", "data"),
    background=True,
    running=[(Output("du-slow-wrap", "style"), {"display": "block"}, {"display": "none"})],
    progress=[Output("du-slow-bar", "value"), Output("du-slow-bar", "max"),
              Output("du-slow-msg", "children")],
    progress_default=("0", "100", ""),
    prevent_initial_call=True,
)
def _slow_refresh(set_progress, _version):
    try:
        res = mo.slow_refresh_history(progress=_report(set_progress))
    except Exception as e:  # noqa: BLE001
        return dmc.Alert(f"History refresh failed: {e}", color="red", variant="light",
                         icon=html.I(className="bi bi-exclamation-triangle"))
    return dmc.Alert(
        f"History refreshed: {res['reservations_rows']:,} reservations, "
        f"{res['perf_rows']:,} performance rows in {res['elapsed_s']}s "
        f"(at {res['refreshed_at']}).",
        color="gray", variant="light", icon=html.I(className="bi bi-database-check"))


# ---------------------------------------------------------------------------
# RETRAIN (background): refit the selected model; bump info-version -> tiles refresh.
# ---------------------------------------------------------------------------
@callback(
    Output("du-retrain-panel", "children"),
    Output("du-info-version", "data"),
    Input("du-retrain-btn", "n_clicks"),
    State("du-model", "value"),
    State("du-retune", "checked"),
    State("du-info-version", "data"),
    background=True,
    running=[
        (Output("du-retrain-btn", "disabled"), True, False),
        (Output("du-cancel-btn", "style"), {"display": "inline-block"}, {"display": "none"}),
        (Output("du-retrain-wrap", "style"), {"display": "block"}, {"display": "none"}),
    ],
    cancel=[Input("du-cancel-btn", "n_clicks")],
    progress=[Output("du-retrain-bar", "value"), Output("du-retrain-bar", "max"),
              Output("du-retrain-msg", "children")],
    progress_default=("0", "100", ""),
    prevent_initial_call=True,
)
def _retrain(set_progress, _n, model, retune, info_version):
    try:
        res = mo.run_retrain(model, retune=bool(retune), progress=_report(set_progress))
    except Exception as e:  # noqa: BLE001
        return dmc.Alert(f"Retraining failed: {e}", color="red", variant="light",
                         icon=html.I(className="bi bi-exclamation-triangle")), no_update

    agg = res.get("walk_forward_aggregate") or {}
    auc = agg.get("auc", {}).get("mean") if isinstance(agg.get("auc"), dict) else None
    change = res.get("feature_change") or {}
    changed = change.get("changed")
    body = dmc.Stack([
        dmc.Text(f"Retrained {mo.model_label(res['model'])} "
                 f"({res.get('mode')}) on {res.get('n_train_deploy') or '?'} bookings.",
                 fw=600, size="sm"),
        dmc.Text((f"Walk-forward AUC ≈ {auc:.3f}. " if auc is not None else "")
                 + (f"Feature set changed: added={change.get('added')} "
                    f"removed={change.get('removed')}." if changed else "Feature set unchanged.")
                 + f" Retrained at {res.get('retrained_at')}.", size="xs", c="dimmed"),
        dmc.Text("Model updated — click 'Update scores' to re-score the next 14 days with "
                 "the new model.", size="xs", c="dimmed"),
    ], gap=6)
    return (dmc.Alert(body, color="green", variant="light",
                      icon=html.I(className="bi bi-check-circle")),
            (info_version or 0) + 1)


# ---------------------------------------------------------------------------
# Scored set: directly-retrievable table (paints on load, refreshes after scoring)
# ---------------------------------------------------------------------------
@callback(
    Output("du-scored-grid", "rowData"),
    Output("du-scored-summary", "children"),
    Input("du-scored-version", "data"),
    Input("du-metrics-timer", "n_intervals"),
)
def _fill_scored(_v, _n):
    ov = mo.scored_overview()
    if ov["n"] == 0:
        return [], "No scored set yet — click ‘Update scores (next 14 days)’."
    summ = (f"{ov['n']:,} bookings · {mo.model_label(ov['model_used']) if ov['model_used'] else '—'} "
            f"· high {ov['high']:,} / uncertain {ov['uncertain']:,} / low {ov['low']:,}"
            + (f" · scored {ov['scored_at']}" if ov["scored_at"] else ""))
    return ov["rows"], summ


@callback(
    Output("du-scored-download", "data"),
    Input("du-download-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _download_scored(_n):
    df = mo.scored_export_frame()
    if df is None or df.empty:
        return no_update
    return dcc.send_data_frame(df.to_csv, "scored_upcoming.csv", index=False)


# ---------------------------------------------------------------------------
# Auto-warm the Model-Performance eval artifacts (background, missing only).
# Same callback handles the manual "Build all (eval + SHAP)" button.
# ---------------------------------------------------------------------------
@callback(
    Output("du-warm-status", "children"),
    Input("du-warm-timer", "n_intervals"),
    Input("du-buildall-btn", "n_clicks"),
    background=True,
    running=[
        (Output("du-buildall-btn", "disabled"), True, False),
        (Output("du-warm-wrap", "style"), {"display": "block"}, {"display": "none"}),
    ],
    progress=[Output("du-warm-bar", "value"), Output("du-warm-bar", "max"),
              Output("du-warm-msg", "children")],
    progress_default=("0", "100", ""),
    prevent_initial_call=True,
)
def _warm_artifacts(set_progress, _n_timer, _n_click):
    include_shap = ctx.triggered_id == "du-buildall-btn"
    try:
        res = mo.ensure_all_eval(progress=_report(set_progress), include_shap=include_shap)
    except Exception as e:  # noqa: BLE001
        return dmc.Alert(f"Artifact build failed: {e}", color="red", variant="light",
                         icon=html.I(className="bi bi-exclamation-triangle"))
    cov = mo.eval_coverage()
    built = res.get("built_eval") or []
    built_shap = res.get("built_shap") or []
    errs = res.get("errors") or []
    if cov["complete"] and not built and not built_shap and not errs:
        return dmc.Text(f"All evaluation artifacts present ({len(cov['have'])}/"
                        f"{len(cov['all'])}).", size="xs", c="dimmed")
    parts = [f"eval ready {len(cov['have'])}/{len(cov['all'])}"]
    if built:
        parts.append("built eval: " + ", ".join(built))
    if built_shap:
        parts.append("built SHAP: " + ", ".join(built_shap))
    color = "gray"
    if errs:
        parts.append("errors: " + "; ".join(errs))
        color = "yellow"
    return dmc.Alert(" · ".join(parts), color=color, variant="light", radius="md",
                     icon=html.I(className="bi bi-layers"))
