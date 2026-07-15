# dash_app/pages/occupancy.py
# MAIN PAGE - Occupancy & Overbooking. Fixed 14-day forward window; property filter
# only (no date filter). Layout top→bottom: KPI tiles → heatmap → composition charts
# → booking table → cost panel. Shell + KPIs render immediately
# from the local cache; scoring runs in a BACKGROUND callback. Already-cancelled
# bookings are excluded centrally in the data layer. No filter/table interaction ever
# hits BigQuery.

from __future__ import annotations

import dash
import dash_ag_grid as dag
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from dash_app import theme
from dash_app.backend import data_access as da
from dash_app.backend import model_performance as mp   # shared global cost-store
from dash_app.components import panels
from dash_app.components import ui

dash.register_page(__name__, path="/occupancy", name="Occupancy & Predictions",
                   order=1, title="STAYERY · Occupancy")


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
            dmc.Title("Occupancy & Predictions", order=3),
            dmc.Badge("Next 14 days · scoring", color="gray", variant="light", radius="sm"),
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
        # Fires once on page load so the cost inputs get pre-filled from the shared store.
        dcc.Interval(id="occ-costs-init", interval=200, n_intervals=0, max_intervals=1),
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

    return dmc.Stack([
        header,
        stores,
        controls,

        # Global overbooking costs + derived cost-optimal threshold (single entry point)
        panels.cost_controls(),

        html.Div(id="occ-scored-warning"),

        # 1) KPI tiles (filled by callback; skeleton until then)
        html.Div(id="occ-kpi-row", children=dmc.Skeleton(height=110, radius="lg")),

        # 2) Heatmap + selection controls
        heatmap_card,

        # 3) Composition charts (filtered by heatmap selection)
        html.Div(id="occ-composition"),

        # 4) Booking table (filtered by heatmap selection) + detail side panel
        booking_row,
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
)
def _selection(click, _clear, _props):
    if ctx.triggered_id == "occ-heatmap" and click:
        p = click["points"][0]
        sel = {"property": p["y"], "day": p["x"]}
        return sel, f"Filtered to {sel['property']} · {sel['day']} (click ‘Clear selection’ to reset)"
    return None, "Showing all selected properties across 14 days"


# ---------------------------------------------------------------------------
# Cost-optimal threshold badge (derived from the global costs; drives everything)
# ---------------------------------------------------------------------------
@callback(
    Output("occ-thr-badge", "children"),
    Input("cost-store", "data"),
)
def _threshold_badge(store):
    walk, empty, high, mult = mp.read_cost_full(store)
    return f"{da.cost_optimal_threshold(walk, empty, high, mult):.0%}"


# ---------------------------------------------------------------------------
# KPI tiles + heatmap + per-night store + scored warning
# ---------------------------------------------------------------------------
@callback(
    Output("occ-kpi-row", "children"),
    Output("occ-heatmap", "figure"),
    Output("occ-pernight-store", "data"),
    Output("occ-scored-warning", "children"),
    Input("occ-property-filter", "value"),
    Input("cost-store", "data"),
    Input("occ-scored-version", "data"),
)
def _update_main(selected, cost_store, _version):
    props = _selected_props(selected)
    freshness = da.data_freshness()
    mm = da.model_meta()
    walk, empty, high, mult = mp.read_cost_full(cost_store)
    thr = da.cost_optimal_threshold(walk, empty, high, mult)

    scored = da.load_scored()
    grid_fig = panels.heatmap_figure(da.heatmap_grid(props or None, thr))

    if scored.empty:
        kpis = panels.kpi_tiles(freshness, mm, None, thr)
        warn = dmc.Alert(
            "No scored bookings yet. Click 'Refresh scores' to run the model on the "
            "cached reservations (no BigQuery query is triggered).",
            color="yellow", variant="light", radius="md",
            icon=html.I(className="bi bi-exclamation-triangle"))
        return kpis, grid_fig, None, warn

    win = da.in_window(scored, props)
    high_risk = int((pd.to_numeric(win["cancel_proba"], errors="coerce") >= thr).sum()) \
        if not win.empty else 0
    kpis = panels.kpi_tiles(freshness, mm, high_risk, thr)

    pernight = da.per_night_expected_freed(win, hotel_col="property_name")
    if not pernight.empty:
        pernight = pernight.assign(arrival_date=pernight["arrival_date"].astype(str))
    return kpis, grid_fig, pernight.to_dict("records"), None


# ---------------------------------------------------------------------------
# Composition charts + booking table (both follow the heatmap selection)
# ---------------------------------------------------------------------------
@callback(
    Output("occ-composition", "children"),
    Output("occ-grid", "rowData"),
    Input("occ-selection", "data"),
    Input("occ-property-filter", "value"),
    Input("cost-store", "data"),
    Input("occ-scored-version", "data"),
)
def _update_views(selection, selected, cost_store, _version):
    walk, empty, high, mult = mp.read_cost_full(cost_store)
    thr = da.cost_optimal_threshold(walk, empty, high, mult)
    scored = da.add_display_columns(da.load_scored(), thr)   # risk labels from the cost threshold
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
# Booking detail side panel (raw record - no SHAP/XAI, that's Phase 4)
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
# Global cost persistence (one shared entry) + visible empty-room pre-fill.
# Loads once on page open from the shared store (so values set on Model Performance
# show here too); saves on every edit back to the same global key.
# ---------------------------------------------------------------------------
@callback(
    Output("cost-walk", "value"),
    Output("cost-empty", "value"),
    Output("cost-high-demand", "checked"),
    Output("cost-multiplier", "value"),
    Output("cost-empty-help", "children"),
    Input("occ-costs-init", "n_intervals"),
    State("cost-store", "data"),
)
def _load_cost_params(_tick, store):
    s = (store or {}).get(mp.GLOBAL_COST_KEY) or {}
    walk = s.get("walk")          # no default: the RM sets the walk cost
    empty = s.get("empty")
    high = bool(s.get("high", False))
    mult = s.get("mult") or mp.sc_default_multiplier()
    if empty in (None, ""):
        # PRE-FILL the empty-room cost from the best available (global) source.
        empty, source = da.empty_room_cost_prefill_global()
        help_txt = (f"Empty-room cost pre-filled from {source} = {empty}. Adjust as needed."
                    if empty is not None
                    else "No pre-fill available yet - enter the empty-room cost.")
    else:
        help_txt = "Costs are global - shared with the Model Performance page."
    return walk, empty, high, mult, help_txt


@callback(
    # allow_duplicate: Model Performance writes the SAME global cost entry; each page
    # declares allow_duplicate so registration is independent of page import order.
    Output("cost-store", "data", allow_duplicate=True),
    Input("cost-walk", "value"),
    Input("cost-empty", "value"),
    Input("cost-high-demand", "checked"),
    Input("cost-multiplier", "value"),
    State("cost-store", "data"),
    prevent_initial_call=True,
)
def _save_cost_params(walk, empty, high, mult, store):
    store = dict(store or {})
    cur = dict(store.get(mp.GLOBAL_COST_KEY) or {})
    cur.update({"walk": walk, "empty": empty, "high": bool(high), "mult": mult})
    store[mp.GLOBAL_COST_KEY] = cur
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
    except Exception as e:  # noqa: BLE001 - surface the reason, don't crash
        return no_update, f"Refresh failed: {e}"
