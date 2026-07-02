# dash_app/pages/occupancy.py
# MAIN PAGE — Occupancy & Overbooking. Fixed 14-day forward window; only a property
# filter (no date filter). Shell + KPI tiles render immediately from the local cache;
# scoring runs in a BACKGROUND callback (Refresh). No filter/table interaction ever
# hits BigQuery — everything reads the Phase-1 caches.

from __future__ import annotations

import dash
import dash_ag_grid as dag
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from dash_app import theme
from dash_app.backend import data_access as da
from dash_app.components import panels
from src import overbooking as ob

dash.register_page(__name__, path="/occupancy", name="Occupancy & Overbooking",
                   order=1, title="STAYERY · Occupancy")


def _iso_week(today: pd.Timestamp | None = None) -> str:
    t = today or pd.Timestamp.now("UTC")
    iso = t.isocalendar()
    return f"{iso[0]}-W{int(iso[1]):02d}"


# ---------------------------------------------------------------------------
# Layout (callable => re-read on each navigation, so property list stays fresh)
# ---------------------------------------------------------------------------
def layout(**_kwargs):
    props = da.property_list()
    prop_opts = [{"label": p, "value": p} for p in props]

    controls = dbc.Row([
        dbc.Col([
            dbc.Label("Properties", class_name="mb-1"),
            dcc.Dropdown(id="occ-property-filter", options=prop_opts, value=props,
                         multi=True, placeholder="All properties"),
        ], md=6),
        dbc.Col([
            dbc.Label("High-risk threshold", class_name="mb-1"),
            dbc.Input(id="occ-risk-threshold", type="number", min=0, max=1, step=0.05,
                      value=da.DEFAULT_RISK_THRESHOLD),
        ], md=2),
        dbc.Col([
            dbc.Label("Scores", class_name="mb-1 d-block"),
            dbc.Button("Refresh scores", id="occ-refresh-btn", color="dark", size="sm"),
            html.Span(id="occ-refresh-status", className="text-muted ms-2",
                      style={"fontSize": "0.8rem"}),
        ], md=4),
    ], class_name="g-2 align-items-end mb-2")

    stores = html.Div([
        dcc.Store(id="occ-scored-version", data=0),
        dcc.Store(id="occ-pernight-store"),
        dcc.Store(id="cost-store", storage_type="local"),  # persists across reloads
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
        style={"height": "520px"},
    )

    return dbc.Container([
        html.H3("Occupancy & Overbooking"),
        html.P("Next 14 days · predicted cancellations drive the overbooking "
               "recommendation per property.", className="text-muted"),
        stores,
        controls,
        dbc.Alert(id="occ-scored-warning", color="warning", is_open=False, class_name="py-2"),
        dbc.Row(id="occ-kpi-row", class_name="g-2 mb-2"),
        dbc.Row([
            dbc.Col([html.H6("Bookings in the next 14 days"), grid], md=8),
            dbc.Col(dbc.Card(dbc.CardBody([html.H6("Booking detail"),
                                           html.Div(id="occ-side-panel",
                                                    children=panels.side_panel_content(None))]),
                             style=theme.CARD_STYLE), md=4),
        ], class_name="g-2 mb-2"),
        dbc.Row([
            dbc.Col(panels.cost_panel(prop_opts), md=6),
            dbc.Col(html.Div(id="occ-reco"), md=6),
        ], class_name="g-2 mb-3"),
        html.Div(id="occ-roomtype-section"),
    ], fluid=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
def _selected_props(value) -> list[str]:
    """Normalise the multiselect value; empty selection is treated as 'all'."""
    if value is None:
        return []
    return list(value)


@callback(
    Output("cost-active-property", "options"),
    Output("cost-active-property", "value"),
    Input("occ-property-filter", "value"),
    State("cost-active-property", "value"),
)
def _sync_active_property(selected, current):
    """Cost panel's per-property selector tracks the filtered properties."""
    props = _selected_props(selected) or da.property_list()
    opts = [{"label": p, "value": p} for p in props]
    value = current if current in props else (props[0] if props else None)
    return opts, value


@callback(
    Output("occ-kpi-row", "children"),
    Output("occ-grid", "rowData"),
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
    if scored.empty:
        kpis = panels.kpi_tiles(freshness, mm, None, thr)
        warn = "No scored bookings yet. Click 'Refresh scores' to run the model on the " \
               "cached reservations (no BigQuery query is triggered)."
        return kpis, [], None, warn, True

    win = da.in_window(scored, props)
    high_risk = int((pd.to_numeric(win["cancel_proba"], errors="coerce") >= thr).sum()) \
        if not win.empty else 0
    kpis = panels.kpi_tiles(freshness, mm, high_risk, thr)
    row_data = panels.booking_row_data(win)

    pernight = da.per_night_expected_freed(win, hotel_col="property_name")
    if not pernight.empty:
        pernight = pernight.assign(arrival_date=pernight["arrival_date"].astype(str))
    return kpis, row_data, pernight.to_dict("records"), "", False


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


@callback(
    Output("occ-reco", "children"),
    Input("cost-walk", "value"),
    Input("cost-empty", "value"),
    Input("cost-high-demand", "value"),
    Input("cost-multiplier", "value"),
    Input("cost-active-property", "value"),
    Input("occ-pernight-store", "data"),
)
def _update_recommendation(cost_walk, cost_empty, high_demand, multiplier, prop, pernight_data):
    costs_ready = cost_walk is not None and cost_empty is not None
    if not prop or not pernight_data:
        return panels.recommendation_card(None, costs_ready, prop)

    pernight = pd.DataFrame(pernight_data)
    if "hotel" in pernight.columns:
        pernight = pernight[pernight["hotel"] == prop]
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
    hint = None if caps else dbc.Alert(
        "No room-type capacities set for this property yet — fill in "
        "configs/room_type_capacity.yaml to show the dashed capacity lines.",
        color="secondary", class_name="py-2")
    return html.Div([html.H6("Room-type occupancy"),
                     dcc.Graph(figure=fig, config={"displayModeBar": False}),
                     hint])


# ---- Cost-parameter persistence (per property, per ISO week) ---------------
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
    help_txt = ("Empty-room cost pre-fills from ADR once the property_performance cache "
                "is built and the propertyId↔property mapping is set; until then, enter it.")
    if key in store:
        s = store[key]
        return (s.get("walk"), s.get("empty"), bool(s.get("high", False)),
                s.get("mult", ob.DEFAULT_HIGH_DEMAND_MULTIPLIER), help_txt)
    # Defaults for a property/week not seen before. Walk cost has no default (RM sets
    # it). Empty-room ADR pre-fill needs the propertyId mapping (open) — blank for now.
    return None, None, False, ob.DEFAULT_HIGH_DEMAND_MULTIPLIER, help_txt


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


# ---- Background: (re)score upcoming bookings from the cache ----------------
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
    except Exception as e:  # noqa: BLE001 — surface the reason in the UI, don't crash
        return no_update, f"Refresh failed: {e}"
