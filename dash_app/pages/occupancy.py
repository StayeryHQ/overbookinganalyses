# dash_app/pages/occupancy.py
# MAIN PAGE — Occupancy & Overbooking. Fixed 14-day forward window; property filter
# only (no date filter). Layout top→bottom: KPI tiles → heatmap → composition charts
# → booking table → cost panel → room-type sub-view. Shell + KPIs render immediately
# from the local cache; scoring runs in a BACKGROUND callback. Already-cancelled
# bookings are excluded centrally in the data layer. No filter/table interaction ever
# hits BigQuery.

from __future__ import annotations

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from dash_app import theme
from dash_app.backend import data_access as da
from dash_app.components import panels
from dash_app.components import ui
from src import overbooking as ob

dash.register_page(__name__, path="/occupancy", name="Occupancy & Overbooking",
                   order=1, title="STAYERY · Occupancy")


def _iso_week(today: pd.Timestamp | None = None) -> str:
    t = today or pd.Timestamp.now("UTC")
    iso = t.isocalendar()
    return f"{iso[0]}-W{int(iso[1]):02d}"


def _selected_props(value) -> list[str]:
    """Normalise the multiselect value; empty selection is treated as 'all'."""
    if value is None:
        return []
    return list(value)


# ---------------------------------------------------------------------------
# Layout (callable => re-read on each navigation, so property list stays fresh)
# ---------------------------------------------------------------------------
def layout(**_kwargs):
    props = da.property_list()
    prop_opts = [{"label": p, "value": p} for p in props]

    header = dmc.Group([
        dmc.Group([
            dmc.Title("Occupancy & overbooking", order=3),
            dmc.Badge("Next 14 days · live scoring", color="gray", variant="light", radius="sm"),
        ], gap="sm", align="center"),
        dmc.Text("Click a heatmap tile to drill into one property + day.",
                 size="sm", c="dimmed"),
    ], justify="space-between", align="center", wrap="wrap", mb="xs")

    controls = dmc.Paper(dmc.Group([
        dmc.MultiSelect(
            id="occ-property-filter", data=prop_opts, value=props, label="Properties",
            placeholder="All properties", clearable=True, searchable=True,
            leftSection=html.I(className="bi bi-geo-alt"),
            comboboxProps={"withinPortal": True},
            style={"flex": "1 1 320px", "minWidth": "260px"}),
        dmc.NumberInput(id="occ-risk-threshold", label="High-risk threshold", min=0, max=1,
                        step=0.05, value=da.DEFAULT_RISK_THRESHOLD, style={"width": "170px"}),
        dmc.Stack([
            dmc.Text("Scores", size="sm", fw=600),
            dmc.Group([
                dmc.Button("Refresh scores", id="occ-refresh-btn", size="sm", variant="filled",
                           leftSection=html.I(className="bi bi-arrow-clockwise")),
                dmc.Text(id="occ-refresh-status", size="xs", c="dimmed"),
            ], gap="xs", align="center"),
        ], gap=4),
    ], align="flex-end", gap="md", wrap="wrap"), p="md", radius="lg", withBorder=True)

    stores = html.Div([
        dcc.Store(id="occ-scored-version", data=0),
        dcc.Store(id="occ-pernight-store"),
        dcc.Store(id="occ-selection"),                    # {"property":..,"day":..} or None
        # NOTE: "cost-store" now lives in the GLOBAL app.layout (single shared source of
        # truth across pages). Declaring it here too would duplicate the component id.
    ])

    grid = dag.AgGrid(
        id="occ-grid",
        columnDefs=panels.booking_column_defs(),
        rowData=[],
        getRowId="params.data.id",
        columnSize="responsiveSizeToFit",
        defaultColDef={"sortable": True, "filter": True, "resizable": True},
        dashGridOptions={"rowSelection": {"mode": "singleRow", "enableClickSelection": True},
                         "animateRows": True},
        style={"height": "460px"},
    )

    # Heatmap card: the clear button + live selection label live in the card header.
    heatmap_extra = dmc.Group([
        dmc.Text(id="occ-selection-label", size="xs", c="dimmed"),
        dmc.Button("Clear selection", id="occ-clear-btn", size="xs", variant="subtle",
                   color="gray"),
    ], gap="sm", align="center", wrap="nowrap")

    heatmap_card = ui.chart_card(
        "Occupancy heatmap", "occ-heatmap", height=None, header_extra=heatmap_extra,
        subtitle="Colour = occupancy % (occupied units if capacity unset). "
                 "Click a tile to filter the views below.")

    booking_row = dmc.Grid([
        dmc.GridCol([dmc.Text("Bookings", fw=600, size="sm", mb=6), grid],
                    span={"base": 12, "md": 8}),
        dmc.GridCol(
            dmc.Card([dmc.Text("Booking detail", fw=600, size="sm", mb=6),
                      html.Div(id="occ-side-panel",
                               children=panels.side_panel_content(None))],
                     withBorder=True, radius="lg", p="md", style={"height": "100%"}),
            span={"base": 12, "md": 4}),
    ], gutter="md")

    cost_row = dmc.Grid([
        dmc.GridCol(panels.cost_panel(prop_opts), span={"base": 12, "md": 6}),
        dmc.GridCol(html.Div(id="occ-reco"), span={"base": 12, "md": 6}),
    ], gutter="md")

    return dmc.Stack([
        header,
        stores,
        controls,
        dbc.Alert(id="occ-scored-warning", color="warning", is_open=False, class_name="py-2"),

        # 1) KPI tiles (filled by callback; skeleton until then)
        html.Div(id="occ-kpi-row", children=dmc.Skeleton(height=110, radius="lg")),

        # 2) Heatmap + selection controls
        heatmap_card,

        # 3) Composition charts (filtered by heatmap selection)
        html.Div(id="occ-composition"),

        # 4) Booking table (filtered by heatmap selection) + detail side panel
        booking_row,

        # 5) Cost-parameter panel + recommendation
        cost_row,

        # 6) Room-type sub-view (single property only)
        html.Div(id="occ-roomtype-section"),
    ], gap="md")


# ---------------------------------------------------------------------------
# Selection: a heatmap click selects (property, day); clear button / filter /
# threshold reset it. Single callback (ctx decides) => one owner of the store.
# ---------------------------------------------------------------------------
@callback(
    Output("occ-selection", "data"),
    Output("occ-selection-label", "children"),
    Input("occ-heatmap", "clickData"),
    Input("occ-clear-btn", "n_clicks"),
    Input("occ-property-filter", "value"),
    Input("occ-risk-threshold", "value"),
)
def _selection(click, _clear, _props, _thr):
    if ctx.triggered_id == "occ-heatmap" and click:
        p = click["points"][0]
        sel = {"property": p["y"], "day": p["x"]}
        return sel, f"Filtered to {sel['property']} · {sel['day']} (click ‘Clear selection’ to reset)"
    return None, "Showing all selected properties across 14 days"


# ---------------------------------------------------------------------------
# Cost panel's active-property selector: follows the heatmap selection, else the
# property filter.
# ---------------------------------------------------------------------------
@callback(
    Output("cost-active-property", "data"),   # dmc.Select uses `data` (was dbc `options`)
    Output("cost-active-property", "value"),
    Input("occ-property-filter", "value"),
    Input("occ-selection", "data"),
    State("cost-active-property", "value"),
)
def _sync_active_property(selected, selection, current):
    props = _selected_props(selected) or da.property_list()
    opts = [{"label": p, "value": p} for p in props]
    if selection and selection.get("property") in props:
        return opts, selection["property"]
    value = current if current in props else (props[0] if props else None)
    return opts, value


# ---------------------------------------------------------------------------
# KPI tiles + heatmap + per-night store + scored warning
# ---------------------------------------------------------------------------
@callback(
    Output("occ-kpi-row", "children"),
    Output("occ-heatmap", "figure"),
    Output("occ-pernight-store", "data"),
    Output("occ-scored-warning", "children"),
    Output("occ-scored-warning", "is_open"),
    Input("occ-property-filter", "value"),
    Input("occ-risk-threshold", "value"),
    Input("occ-scored-version", "data"),
)
def _update_main(selected, threshold, _version):
    props = _selected_props(selected)
    freshness = da.data_freshness()
    mm = da.model_meta()
    thr = da.DEFAULT_RISK_THRESHOLD if threshold is None else float(threshold)

    scored = da.load_scored()
    grid_fig = panels.heatmap_figure(da.heatmap_grid(props or None, thr))

    if scored.empty:
        kpis = panels.kpi_tiles(freshness, mm, None, thr)
        warn = ("No scored bookings yet. Click 'Refresh scores' to run the model on the "
                "cached reservations (no BigQuery query is triggered).")
        return kpis, grid_fig, None, warn, True

    win = da.in_window(scored, props)
    high_risk = int((pd.to_numeric(win["cancel_proba"], errors="coerce") >= thr).sum()) \
        if not win.empty else 0
    kpis = panels.kpi_tiles(freshness, mm, high_risk, thr)

    pernight = da.per_night_expected_freed(win, hotel_col="property_name")
    if not pernight.empty:
        pernight = pernight.assign(arrival_date=pernight["arrival_date"].astype(str))
    return kpis, grid_fig, pernight.to_dict("records"), "", False


# ---------------------------------------------------------------------------
# Composition charts + booking table (both follow the heatmap selection)
# ---------------------------------------------------------------------------
@callback(
    Output("occ-composition", "children"),
    Output("occ-grid", "rowData"),
    Input("occ-selection", "data"),
    Input("occ-property-filter", "value"),
    Input("occ-scored-version", "data"),
)
def _update_views(selection, selected, _version):
    scored = da.add_display_columns(da.load_scored())
    if scored.empty:
        return panels.composition_row(pd.DataFrame(), "no scored data"), []

    if selection:
        props = [selection["property"]]
        day = selection["day"]
        label = f"{selection['property']} · {selection['day']}"
    else:
        props = _selected_props(selected)
        day = None
        label = "all selected properties · 14 days"

    arrivals = da.arrivals_window(scored, props, day)
    return panels.composition_row(arrivals, label), panels.booking_row_data(arrivals)


# ---------------------------------------------------------------------------
# Booking detail side panel (raw record — no SHAP/XAI, that's Phase 4)
# ---------------------------------------------------------------------------
@callback(
    Output("occ-side-panel", "children"),
    Input("occ-grid", "selectedRows"),
)
def _update_side_panel(selected_rows):
    if not selected_rows:
        return panels.side_panel_content(None)
    booking_id = selected_rows[0].get("id")
    scored = da.load_scored()
    if scored.empty or booking_id is None:
        return panels.side_panel_content(selected_rows[0])
    match = scored[scored["id"] == booking_id]
    record = match.iloc[0].to_dict() if not match.empty else selected_rows[0]
    return panels.side_panel_content(record)


# ---------------------------------------------------------------------------
# Overbooking recommendation — follows the active property; if a heatmap DAY is
# selected, the recommendation is for that specific night.
# ---------------------------------------------------------------------------
@callback(
    Output("occ-reco", "children"),
    Input("cost-walk", "value"),
    Input("cost-empty", "value"),
    Input("cost-high-demand", "value"),
    Input("cost-multiplier", "value"),
    Input("cost-active-property", "value"),
    Input("occ-pernight-store", "data"),
    Input("occ-selection", "data"),
)
def _update_recommendation(cost_walk, cost_empty, high_demand, multiplier, prop,
                           pernight_data, selection):
    costs_ready = cost_walk is not None and cost_empty is not None
    if not prop or not pernight_data:
        return panels.recommendation_card(None, costs_ready, prop)

    pernight = pd.DataFrame(pernight_data)
    if "hotel" in pernight.columns:
        pernight = pernight[pernight["hotel"] == prop]
    # If a specific day is selected for this property, recommend for that night only.
    if selection and selection.get("property") == prop and "arrival_date" in pernight.columns:
        pernight = pernight[pernight["arrival_date"] == selection.get("day")]
    if pernight.empty:
        return panels.recommendation_card(None, costs_ready, prop)
    if not costs_ready:
        return panels.recommendation_card(None, False, prop)

    reco = ob.recommend_from_per_night(
        pernight, cost_empty=float(cost_empty), cost_walk=float(cost_walk),
        high_demand=bool(high_demand),
        high_demand_multiplier=float(multiplier or ob.DEFAULT_HIGH_DEMAND_MULTIPLIER),
    )
    summary = ob.summarize_property(reco)
    return panels.recommendation_card(summary, True, prop)


# ---------------------------------------------------------------------------
# Room-type occupancy sub-view (single property only)
# ---------------------------------------------------------------------------
@callback(
    Output("occ-roomtype-section", "children"),
    Input("occ-property-filter", "value"),
    Input("occ-scored-version", "data"),
)
def _update_roomtype(selected, _version):
    props = _selected_props(selected)
    if len(props) != 1:                       # visible only for a single property
        return html.Div()
    prop = props[0]
    from src import load_room_type_capacity
    occ = da.room_type_occupancy(prop)
    caps = load_room_type_capacity().get(prop, {})
    fig = panels.room_type_figure(occ, caps, prop)
    hint = None if caps else dmc.Alert(
        "No room-type capacities set for this property yet — fill in "
        "configs/room_type_capacity.yaml to show the dashed capacity lines.",
        color="gray", variant="light", radius="md",
        icon=html.I(className="bi bi-info-circle"))
    return dmc.Card([
        dmc.Text("Room-type occupancy", fw=600, size="sm", mb=6),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        hint,
    ], withBorder=True, radius="lg", p="md")


# ---------------------------------------------------------------------------
# Cost-parameter persistence (per property, per ISO week) + visible pre-fill
# ---------------------------------------------------------------------------
@callback(
    Output("cost-walk", "value"),
    Output("cost-empty", "value"),
    Output("cost-high-demand", "value"),
    Output("cost-multiplier", "value"),
    Output("cost-empty-help", "children"),
    Input("cost-active-property", "value"),
    State("cost-store", "data"),
)
def _load_cost_params(prop, store):
    store = store or {}
    key = f"{prop}|{_iso_week()}"
    if key in store:
        s = store[key]
        help_txt = "Loaded your saved values for this property/week."
        return (s.get("walk"), s.get("empty"), bool(s.get("high", False)),
                s.get("mult", ob.DEFAULT_HIGH_DEMAND_MULTIPLIER), help_txt)
    # New property/week: walk cost has no default (RM sets it). Empty-room cost is
    # PRE-FILLED and shown in the field from the best available source.
    prefill, source = da.empty_room_cost_prefill()
    empty_val = prefill.get(prop)
    help_txt = (f"Empty-room cost pre-filled from {source} = {empty_val}. "
                "Adjust as needed." if empty_val is not None
                else "No pre-fill available yet — enter the empty-room cost.")
    return None, empty_val, False, ob.DEFAULT_HIGH_DEMAND_MULTIPLIER, help_txt


@callback(
    Output("cost-store", "data"),
    Input("cost-walk", "value"),
    Input("cost-empty", "value"),
    Input("cost-high-demand", "value"),
    Input("cost-multiplier", "value"),
    State("cost-active-property", "value"),
    State("cost-store", "data"),
    prevent_initial_call=True,
)
def _save_cost_params(walk, empty, high, mult, prop, store):
    if not prop:
        return no_update
    store = dict(store or {})
    store[f"{prop}|{_iso_week()}"] = {"walk": walk, "empty": empty,
                                      "high": bool(high), "mult": mult}
    return store


# ---------------------------------------------------------------------------
# Background: (re)score upcoming bookings from the cache (no BigQuery)
# ---------------------------------------------------------------------------
@callback(
    Output("occ-scored-version", "data"),
    Output("occ-refresh-status", "children"),
    Input("occ-refresh-btn", "n_clicks"),
    State("occ-scored-version", "data"),
    background=True,
    running=[(Output("occ-refresh-btn", "disabled"), True, False)],
    prevent_initial_call=True,
)
def _refresh_scores(_n, version):
    try:
        count = da.refresh_scored()
        return (version or 0) + 1, f"Scored {count:,} bookings."
    except Exception as e:  # noqa: BLE001 — surface the reason, don't crash
        return no_update, f"Refresh failed: {e}"
