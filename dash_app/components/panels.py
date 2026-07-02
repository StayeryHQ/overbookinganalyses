# dash_app/components/panels.py
# Pure builder functions for the Occupancy page: KPI tiles, the ag-grid column defs
# + row data, the booking side panel, the cost-parameter panel, the overbooking
# recommendation card, and the room-type occupancy figure. Kept separate from the
# page so occupancy.py holds only layout + callbacks.

from __future__ import annotations

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import dcc, html

from dash_app import theme

# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------
def _kpi_card(label: str, value, sub: str | None = None) -> dbc.Col:
    body = [html.Div(label, style=theme.KPI_LABEL_STYLE),
            html.Div(value, style=theme.KPI_VALUE_STYLE)]
    if sub:
        body.append(html.Div(sub, className="text-muted", style={"fontSize": "0.75rem"}))
    return dbc.Col(dbc.Card(dbc.CardBody(body), style=theme.CARD_STYLE), md=3, className="mb-2")


def kpi_tiles(freshness: dict, model_meta: dict, high_risk_count: int | None,
              risk_threshold: float) -> list:
    """Four KPI tiles. Values that aren't backed by real metadata are shown as
    'unavailable' rather than fabricated."""
    data_ts = freshness.get("reservations") or "never — run `main.py refresh`"

    m_ts = model_meta.get("retrained_at") or "unavailable"
    m_sub = f"model: {model_meta['model']}" if model_meta.get("model") else "no model on disk"

    n_train = model_meta.get("trained_on_bookings")
    train_val = f"~{n_train:,}" if n_train is not None else "unavailable"
    train_sub = model_meta.get("trained_on_note") or "no training metadata"

    hr_val = "—" if high_risk_count is None else f"{high_risk_count:,}"
    hr_sub = f"≥ {risk_threshold:.0%} cancel risk · next 14 days · selected"

    return [
        _kpi_card("Data last updated", data_ts),
        _kpi_card("Model last retrained", m_ts, m_sub),
        _kpi_card("Training set size", train_val, train_sub),
        _kpi_card("High-risk bookings", hr_val, hr_sub),
    ]


# ---------------------------------------------------------------------------
# Booking risk table (dash-ag-grid)
# ---------------------------------------------------------------------------
# cancel_proba is kept numeric (0..1) for correct sorting; a valueFormatter renders
# it as a percentage. Risk bucket is colour-coded via styleConditions.
def booking_column_defs() -> list[dict]:
    return [
        {"headerName": "Booking", "field": "id", "pinned": "left", "width": 130},
        {"headerName": "Property", "field": "property_name", "width": 170},
        {"headerName": "Arrival", "field": "arrival", "width": 120, "sort": "asc"},
        {"headerName": "LoS", "field": "los_nights", "width": 80, "type": "numericColumn"},
        {"headerName": "Channel", "field": "channelCode", "width": 130},
        {"headerName": "Cancel risk", "field": "cancel_proba", "width": 120,
         "type": "numericColumn",
         "valueFormatter": {"function": "(params.value == null ? '' : (params.value*100).toFixed(0) + '%')"}},
        {"headerName": "Risk", "field": "risk_bucket", "width": 110,
         "cellStyle": {"styleConditions": [
             {"condition": "params.value == 'high'", "style": {"color": theme.RED, "fontWeight": "bold"}},
             {"condition": "params.value == 'uncertain'", "style": {"color": theme.ORANGE}},
             {"condition": "params.value == 'low'", "style": {"color": theme.GREEN}},
         ]}},
        {"headerName": "Status", "field": "status", "width": 120},
    ]


_TABLE_FIELDS = ["id", "property_name", "arrival", "los_nights", "channelCode",
                 "cancel_proba", "risk_bucket", "status"]


def booking_row_data(df_window: pd.DataFrame) -> list[dict]:
    """ag-grid rowData from the scored window frame. Formats arrival as a date and
    keeps only the display fields (the full record still feeds the side panel)."""
    if df_window.empty:
        return []
    d = df_window.copy()
    d["arrival"] = pd.to_datetime(d["arrival"], utc=True).dt.strftime("%Y-%m-%d")
    for f in _TABLE_FIELDS:
        if f not in d.columns:
            d[f] = None
    return d[_TABLE_FIELDS].to_dict("records")


# ---------------------------------------------------------------------------
# Booking detail side panel (raw record — NO SHAP/XAI, that's Phase 4)
# ---------------------------------------------------------------------------
# Fields shown in the side panel, in order (only those present are rendered).
_DETAIL_FIELDS = [
    ("id", "Booking reference"), ("bookingId", "Booking ID"), ("property_name", "Property"),
    ("status", "Status"), ("arrival", "Arrival"), ("departure", "Departure"),
    ("created", "Booked on"), ("los_nights", "Length of stay (nights)"),
    ("channelCode", "Channel"), ("ratePlan_name", "Rate plan"),
    ("unitGroup_name", "Room type"), ("adults", "Adults"),
    ("totalGrossAmount_amount", "Gross amount"), ("cancellationFee_fee_amount", "Cancellation fee"),
    ("guaranteeType", "Guarantee"), ("cancel_proba", "Predicted cancel risk"),
    ("risk_bucket", "Risk bucket"), ("model_used", "Scored by model"),
]


def _fmt(field: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if field == "cancel_proba":
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return str(value)
    if field in ("arrival", "departure", "created"):
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        return "—" if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    return str(value)


def side_panel_content(record: dict | None) -> list:
    """Definition-list of the raw booking record for the selected row."""
    if not record:
        return [html.P("Select a booking in the table to see its full record.",
                       className="text-muted")]
    rows = []
    for field, label in _DETAIL_FIELDS:
        if field in record:
            rows.append(html.Tr([html.Th(label, style={"width": "45%", "fontWeight": 600}),
                                 html.Td(_fmt(field, record.get(field)))]))
    return [dbc.Table(html.Tbody(rows), bordered=False, size="sm", striped=True)]


# ---------------------------------------------------------------------------
# Cost-parameter panel (static layout; values are set/persisted via callbacks)
# ---------------------------------------------------------------------------
def cost_panel(active_property_options: list[dict]) -> dbc.Card:
    return dbc.Card(dbc.CardBody([
        html.H6("Overbooking cost parameters", className="card-title"),
        html.Small("Per property, per week. Saved in your browser so they survive a "
                   "reload.", className="text-muted"),
        html.Div([
            dbc.Label("Property", html_for="cost-active-property", class_name="mt-2 mb-1"),
            dcc.Dropdown(id="cost-active-property", options=active_property_options,
                         value=(active_property_options[0]["value"] if active_property_options else None),
                         clearable=False),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Cost of walking a guest", html_for="cost-walk", class_name="mt-2 mb-1"),
                dbc.Input(id="cost-walk", type="number", min=0, step=1, placeholder="set your own"),
            ], md=6),
            dbc.Col([
                dbc.Label("Cost of an empty room", html_for="cost-empty", class_name="mt-2 mb-1"),
                dbc.Input(id="cost-empty", type="number", min=0, step=1, placeholder="ADR pre-fill (if available)"),
            ], md=6),
        ]),
        dbc.Row([
            dbc.Col(dbc.Switch(id="cost-high-demand", label="High-demand period", value=False,
                               class_name="mt-3"), md=6),
            dbc.Col([
                dbc.Label("Walk-cost multiplier", html_for="cost-multiplier", class_name="mt-2 mb-1"),
                dbc.Input(id="cost-multiplier", type="number", min=1, step=0.1, value=1.5),
            ], md=6),
        ]),
        html.Div(id="cost-empty-help", className="text-muted mt-2", style={"fontSize": "0.75rem"}),
    ]), style=theme.CARD_STYLE)


# ---------------------------------------------------------------------------
# Overbooking recommendation card
# ---------------------------------------------------------------------------
_RECO_TOOLTIP = (
    "The tool treats freed rooms per night as a Poisson-binomial random variable "
    "(mean = sum of cancel probabilities, i.e. expected cancellations). It then picks "
    "the overbooking level that minimises expected cost: overbooking too little leaves "
    "rooms empty (cost of an empty room); overbooking too much means walking guests "
    "(cost of a walked guest). Because walking is usually far more expensive, the "
    "recommendation stays below the expected number of cancellations. 'High-demand "
    "period' raises the walk cost, making the recommendation more conservative."
)


def recommendation_card(summary: dict | None, costs_ready: bool,
                        property_name: str | None) -> dbc.Card:
    if not property_name:
        inner = [html.P("Select a property in the cost panel to see its recommendation.",
                        className="text-muted")]
    elif not costs_ready:
        inner = [html.P("Enter the walk cost (and empty-room cost) to get a recommendation.",
                        className="text-muted")]
    elif not summary or summary.get("median_reco") is None:
        inner = [html.P("No upcoming bookings for this property in the next 14 days.",
                        className="text-muted")]
    else:
        inner = [
            html.Div([
                html.Span("Recommended overbooking allowance ", className="text-muted"),
                html.I(className="bi bi-info-circle", id="reco-info",
                       style={"cursor": "help"}),
                dbc.Tooltip(_RECO_TOOLTIP, target="reco-info", placement="right",
                            style={"maxWidth": "420px"}),
            ]),
            html.Div(f"{summary['median_reco']} rooms",
                     style={**theme.KPI_VALUE_STYLE, "fontSize": "2.4rem", "color": theme.BLACK}),
            html.Small(f"typical per night · peak night {summary['max_reco']} · "
                       f"avg expected cancellations {summary['mean_exp_freed']:.1f}/night "
                       f"across {summary['nights']} nights", className="text-muted"),
        ]
    return dbc.Card(dbc.CardBody([html.H6("Recommendation", className="card-title"), *inner]),
                    style={**theme.CARD_STYLE, "backgroundColor": "#FFFDF0"})


# ---------------------------------------------------------------------------
# Room-type occupancy figure (single property)
# ---------------------------------------------------------------------------
def room_type_figure(occ_df: pd.DataFrame, capacities: dict, property_name: str) -> go.Figure:
    fig = go.Figure()
    if occ_df.empty:
        fig.update_layout(title=f"No upcoming occupancy data for {property_name}")
        return theme.brand_figure(fig)
    groups = sorted(occ_df["unitGroup"].unique())
    for i, g in enumerate(groups):
        col = theme.CATEGORICAL[i % len(theme.CATEGORICAL)]
        gd = occ_df[occ_df["unitGroup"] == g].sort_values("date")
        fig.add_trace(go.Scatter(x=gd["date"], y=gd["occupied"], mode="lines+markers",
                                 name=str(g), line=dict(color=col, width=2)))
        cap = capacities.get(str(g))
        if cap is not None:
            # dashed capacity reference line in the same colour as the room type
            fig.add_trace(go.Scatter(
                x=[occ_df["date"].min(), occ_df["date"].max()], y=[cap, cap],
                mode="lines", line=dict(color=col, width=1, dash="dash"),
                name=f"{g} capacity", showlegend=False, hoverinfo="skip"))
    fig.update_layout(title=f"Room-type occupancy · {property_name} · next 14 days",
                      yaxis_title="Occupied units", xaxis_title=None, height=380)
    return theme.brand_figure(fig)
