# dash_app/pages/model_performance.py
# PAGE 4 - XAI & Model Performance. Lets a team member FAIRLY compare the four cancellation
# models against the naive historical-average baseline (same estimand, same rows, same
# label - see src.model_eval), inspect where each model beats the baseline, and click into
# a single booking's explanation. Read-only: every figure reads the pre-computed eval /
# SHAP artifacts (Data/model_eval_*.parquet, Data/shap_*.parquet); the heavy compute is
# pre-warmed offline (`python main.py eval` / `python main.py explain`).
#
# Design: same shared system as the other pages (components/ui, theme.brand_figure, dmc
# Stack, chart_card, kpi_strip, location_filter, right-side Drawer). The overbooking COST
# parameter is read from the GLOBAL shared cost-store (single source of truth).

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

import src
from dash_app.backend import data_access as da
from dash_app.backend import explain as ex          # iteration curve only (XAI moved to Occupancy)
from dash_app.backend import jobs
from dash_app.backend import model_ops as mo
from dash_app.backend import model_performance as mp
from dash_app.components import performance_charts as pc
from dash_app.components import ui

dash.register_page(__name__, path="/model-performance", name="Model Performance",
                   order=3, title="STAYERY · Model Performance")
_METRIC_DATA = [{"label": "AUC", "value": "auc"}, {"label": "Brier", "value": "brier"}]

_INFO = {
    "kpi": "All metrics come from one leak-free, decision-time walk-forward per model "
           "(each booking scored once at d = min(lead, 14) days before arrival, on the same "
           "rows + label). The baseline is the historical average cancel rate.",
    "roc": "ROC on the pooled out-of-time predictions. No baseline line here on purpose: a "
           "constant historical-average predictor sits on the diagonal (AUC 0.5) by "
           "construction, so an AUC-vs-baseline comparison is not meaningful.",
    "roc_loc": "ROC-AUC per location (pooled predictions). Locations with fewer than "
               f"{mp.LOC_MIN_N} bookings or only one class are hidden as unreliable.",
    "pr": "Precision / recall / F1 across the decision threshold. The dashed line is the "
          "cost-optimal threshold for the current walk/empty costs; the orange diamond is "
          "the constant-baseline F1 at that threshold (here the comparison IS valid).",
    "rel": "Reliability diagram: predicted vs observed cancel frequency. On-diagonal = well "
           "calibrated. The orange diamond is the historical-average baseline (a good sanity "
           "check - it should sit on the diagonal at the base rate).",
    "tt": "Train vs test metric, averaged over the walk-forward folds (± std). A large "
          "train-test gap signals overfitting.",
    "iter": "Boosting train/validation loss per iteration - only defined for the boosting "
            "models (XGBoost, HistGB); other model types show nothing rather than a faked curve.",
    "imp": "Mean |SHAP| per feature - model-agnostic, on the same P(cancel-by-arrival) scale "
           "for every model, so importances are comparable across the four.",
    "bee": "SHAP beeswarm: each dot is a booking; colour is the feature value (blue low, red "
           "high). Model-agnostic on the scalar output, so the hazard model is explained on "
           "the same estimand as the classifiers (no SurvSHAP needed).",
    "pdp": "Partial dependence: average P(cancel) as one feature is swept across its range, "
           "with faint per-booking ICE lines. Computed through the same adapter for every model.",
    "table": "Upcoming scored bookings (highest risk first). Click a row for its SHAP "
             "explanation on the selected model.",
}


def _sel(value) -> list[str] | None:
    return list(value) if value else None


# ---------------------------------------------------------------------------
# Layout (callable => artifact availability re-checked on each navigation)
# ---------------------------------------------------------------------------
def layout(**_kwargs):
    props = da.property_list()
    serving = mo.available_serving_models()
    model0 = serving[0] if serving else "hazard"

    controls = dmc.Paper(dmc.Group([
        # Served models only (hazard / xgboost) — no 4-model comparison. XAI lives on
        # Occupancy & Predictions now; this page is the served model's training performance.
        dmc.Select(id="mp-model", label="Served model", value=model0,
                   data=mo.scoring_model_options(), allowDeselect=False,
                   style={"width": "220px"}),
        # No min: negative costs allowed. Shared globally with Occupancy & Predictions.
        dmc.NumberInput(id="mp-cost-walk", label="Walk cost (€)", value=mp.DEFAULT_WALK,
                        step=10, style={"width": "150px"}),
        dmc.NumberInput(id="mp-cost-empty", label="Empty-room cost (€)", value=mp.DEFAULT_EMPTY,
                        step=10, style={"width": "170px"}),
        dmc.Stack([dmc.Text("Cost-optimal threshold", size="xs", c="dimmed", fw=600),
                   dmc.Text(id="mp-thr-badge", size="sm", fw=700)], gap=0),
    ], align="flex-end", gap="md", wrap="wrap"), p="md", radius="lg", withBorder=True)

    tt_extra = dmc.SegmentedControl(id="mp-tt-metric", data=_METRIC_DATA, value="auc",
                                    size="xs", radius="md")

    return dmc.Stack([
        dmc.Group([
            dmc.Group([dmc.Title("Model performance", order=3),
                       dmc.Badge("Leak-free · training walk-forward", color="gray",
                                 variant="light", radius="sm")], gap="sm", align="center"),
            dmc.Text("Training/walk-forward performance of the served model. SHAP & PDP "
                     "explanations live on Occupancy & Predictions (XAI section).",
                     size="sm", c="dimmed"),
        ], justify="space-between", align="center", wrap="wrap", mb="xs"),

        controls,
        ui.location_filter(props, "mp-location-filter"),

        dcc.Store(id="mp-eval-version", data=0),
        dcc.Store(id="mp-jobs-seen", data={}),
        # Poller for the shared 'artifacts' job - progress survives page changes.
        dcc.Interval(id="mp-poll", interval=1500, n_intervals=0),
        dmc.Paper(dmc.Group([
            dmc.Stack([dmc.Text("Evaluation artifact", size="sm", fw=600),
                       dmc.Text("Build / refresh the leak-free eval this page reads. Runs as a "
                                "background job (hazard takes minutes) and keeps running if "
                                "you switch pages.",
                                id="mp-rebuild-status", size="xs", c="dimmed")], gap=2),
            dmc.Group([
                dmc.Checkbox(id="mp-rebuild-all", label="all models", checked=False),
                dmc.Button("Rebuild evaluation", id="mp-rebuild-btn", size="sm", variant="filled",
                           leftSection=html.I(className="bi bi-arrow-clockwise")),
            ], gap="md", align="center"),
        ], justify="space-between", align="center", wrap="wrap"),
            p="md", radius="lg", withBorder=True),

        html.Div(id="mp-status"),
        html.Div(id="mp-kpi", children=dmc.Skeleton(height=96, radius="lg")),

        dmc.SimpleGrid([
            ui.chart_card("ROC curve", "mp-roc", info=_INFO["roc"], height=360),
            ui.chart_card("ROC-AUC per location", "mp-roc-loc", info=_INFO["roc_loc"], height=360),
        ], cols={"base": 1, "md": 2}, spacing="md"),

        ui.chart_card("Precision / Recall / F1 · cost-optimal threshold", "mp-pr",
                      info=_INFO["pr"], height=380),
        ui.chart_card("Calibration (reliability)", "mp-reliability", info=_INFO["rel"], height=380),

        dmc.SimpleGrid([
            ui.chart_card("Train vs test", "mp-traintest", info=_INFO["tt"], height=340,
                          header_extra=tt_extra),
            ui.chart_card("Iteration curve", "mp-itercurve", info=_INFO["iter"], height=340),
        ], cols={"base": 1, "md": 2}, spacing="md"),
    ], gap="md")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _pct(x, d=1):
    return "unavailable" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x * 100:.{d}f}%"


def _num(x, d=3):
    return "unavailable" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}f}"


def _status_alert(model: str):
    st = mp.eval_status(model)
    if not st["available"]:
        return dmc.Alert(
            dmc.Text(f"No evaluation artifact for '{model}' yet - click 'Rebuild evaluation' "
                     "above. The job runs in the background and survives page changes.",
                     size="sm"),
            title="Evaluation not built", color="yellow", variant="light",
            icon=html.I(className="bi bi-exclamation-triangle"), radius="md")
    if not ex.shap_available(model):
        return dmc.Alert(
            dmc.Text(f"Metrics are ready. SHAP/importance for '{model}' is not built yet - "
                     "use 'Build all (eval + SHAP)' on the Update & Retraining page.",
                     size="sm"),
            title="SHAP not built", color="blue", variant="light",
            icon=html.I(className="bi bi-info-circle"), radius="md", withCloseButton=True)
    return None


def _kpi_cards(k: dict) -> list:
    beats = (k["bss"] is not None and k["bss"] > 0)
    return [
        ui.kpi_card("Discrimination (AUC)", _num(k["auc"]),
                    sub=f"{k['n']:,} pooled bookings" if k["n"] else "no data",
                    accent=True, tooltip="Pooled ROC-AUC on the leak-free walk-forward."),
        ui.kpi_card("Brier skill vs baseline", _num(k["bss"]),
                    sub=("beats baseline" if beats else "not beating baseline"),
                    tooltip="Brier Skill Score vs the historical-average predictor. >0 = better "
                            "than always guessing the base rate."),
        ui.kpi_card("Best location", k["best_loc"] or "unavailable",
                    sub=f"BSS {_num(k['best_bss'])}" if k["best_loc"] else "below sample threshold",
                    tooltip="Location where the model most beats its baseline (Brier Skill)."),
        ui.kpi_card("Weakest location", k["worst_loc"] or "unavailable",
                    sub=f"BSS {_num(k['worst_bss'])}" if k["worst_loc"] else "below sample threshold",
                    tooltip="Location where the model least beats (or trails) its baseline."),
    ]


# ---------------------------------------------------------------------------
# Prefill cost inputs from the shared store (single source of truth); resync on model change
# ---------------------------------------------------------------------------
@callback(
    Output("mp-cost-walk", "value"),
    Output("mp-cost-empty", "value"),
    Input("mp-model", "value"),
    State("cost-store", "data"),
)
def _prefill_cost(_model, store):
    walk, empty = mp.read_cost_params(store)
    return walk, empty


@callback(
    Output("cost-store", "data", allow_duplicate=True),
    Input("mp-cost-walk", "value"),
    Input("mp-cost-empty", "value"),
    State("cost-store", "data"),
    prevent_initial_call=True,
)
def _save_cost(walk, empty, store):
    store = dict(store or {})
    cur = dict(store.get(mp.GLOBAL_COST_KEY) or {})
    cur.update({"walk": walk, "empty": empty})
    store[mp.GLOBAL_COST_KEY] = cur
    return store


# ---------------------------------------------------------------------------
# In-app rebuild of the eval artifact(s) - file-backed job (survives page changes)
# ---------------------------------------------------------------------------
@callback(
    Output("mp-rebuild-status", "children", allow_duplicate=True),
    Input("mp-rebuild-btn", "n_clicks"),
    State("mp-model", "value"),
    State("mp-rebuild-all", "checked"),
    prevent_initial_call=True,
)
def _start_rebuild(_n, model, do_all):
    started = jobs.start("artifacts", mo.rebuild_eval_job, model, bool(do_all))
    return ("Rebuild started…" if started
            else "An artifact job is already running - see progress below.")


@callback(
    Output("mp-rebuild-status", "children"),
    Output("mp-rebuild-btn", "disabled"),
    Output("mp-eval-version", "data"),
    Output("mp-jobs-seen", "data"),
    Input("mp-poll", "n_intervals"),
    State("mp-jobs-seen", "data"),
    State("mp-eval-version", "data"),
)
def _poll_rebuild(_n, seen, version):
    seen = dict(seen or {})
    st = jobs.read("artifacts")
    status = st.get("status", "idle")
    if status == "running":
        pct = int(float(st.get("progress", 0)) * 100)
        return (f"⏳ {pct}% - {st.get('message', '')} (keeps running across pages)",
                True, no_update, no_update)
    fin = st.get("finished")
    bump = no_update
    if fin and seen.get("artifacts") != fin:
        seen["artifacts"] = fin
        if status == "done":
            bump = (version or 0) + 1        # re-read the fresh artifacts once
    # Auto-refresh when a RETRAIN finishes elsewhere — it rebuilds this page's eval, so the
    # performance charts must not lag the freshly deployed model.
    rt = jobs.read("retrain")
    rfin = rt.get("finished")
    if rfin and seen.get("retrain") != rfin:
        seen["retrain"] = rfin
        if rt.get("status") == "done":
            bump = (version or 0) + 1
    if status == "error":
        return "✗ " + str(st.get("error", "failed")), False, bump, seen
    if status == "done":
        res = st.get("result") or {}
        parts = res.get("rebuilt") or res.get("built_eval") or []
        errs = res.get("errors") or []
        txt = ("Rebuilt: " + ", ".join(parts) if parts else "Nothing to rebuild.")
        if errs:
            txt += " - errors: " + "; ".join(errs)
        return txt, False, bump, seen
    return no_update, False, bump, seen


# ---------------------------------------------------------------------------
# Core metrics: KPI + ROC + PR + reliability (react to model / location / cost)
# ---------------------------------------------------------------------------
@callback(
    Output("mp-status", "children"),
    Output("mp-kpi", "children"),
    Output("mp-roc", "figure"),
    Output("mp-roc-loc", "figure"),
    Output("mp-pr", "figure"),
    Output("mp-reliability", "figure"),
    Output("mp-thr-badge", "children"),
    Input("mp-model", "value"),
    Input("mp-location-filter", "value"),
    Input("mp-cost-walk", "value"),
    Input("mp-cost-empty", "value"),
    Input("mp-eval-version", "data"),
    State("cost-store", "data"),
)
def _update_core(model, sel_value, walk, empty, _version, cost_store):
    props = _sel(sel_value)
    walk = float(walk) if walk not in (None, "") else mp.DEFAULT_WALK
    empty = float(empty) if empty not in (None, "") else mp.DEFAULT_EMPTY
    # Apply the shared high-demand multiplier so the threshold matches the app-wide setting.
    _, _, high, mult = mp.read_cost_full(cost_store)
    eff_walk = src.effective_walk_cost(walk, high, mult)

    status = _status_alert(model)
    k = mp.kpis(model, props, eff_walk, empty)
    kpi = ui.kpi_strip(_kpi_cards(k)) if k.get("n") else dmc.Alert(
        "No evaluation data for this selection yet.", color="gray", variant="light")

    pr = mp.pr_threshold(model, props, eff_walk, empty)
    thr_txt = (f"{pr['t_cost']:.0%}" + (" · high-demand" if high else "")) if pr else "-"
    return (status, kpi,
            pc.fig_roc(mp.roc_global(model, props)),
            pc.fig_roc_by_location(mp.roc_by_location(model, props)),
            pc.fig_pr_threshold(pr),
            pc.fig_reliability(mp.reliability(model, props)),
            thr_txt)


# ---------------------------------------------------------------------------
# Train vs test + iteration curve
# ---------------------------------------------------------------------------
@callback(
    Output("mp-traintest", "figure"),
    Output("mp-itercurve", "figure"),
    Input("mp-model", "value"),
    Input("mp-tt-metric", "value"),
    Input("mp-eval-version", "data"),
)
def _update_traintest(model, metric, _version):
    tt = mp.train_test(model)
    try:
        curve = ex.iteration_curve(model)
    except Exception:  # noqa: BLE001
        curve = {}
    return pc.fig_train_test(tt, metric=metric or "auc"), pc.fig_iteration_curve(curve)
