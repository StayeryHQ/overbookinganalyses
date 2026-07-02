# dash_app/pages/home.py — landing page: short intro + navigation.
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/", name="Home", order=0, title="STAYERY · Home")


def _nav_card(title: str, body: str, href: str, disabled: bool = False) -> dbc.Col:
    btn = dbc.Button("Open" if not disabled else "Coming soon",
                     href=href if not disabled else None, color="dark",
                     outline=True, disabled=disabled, size="sm")
    return dbc.Col(dbc.Card(dbc.CardBody([
        html.H5(title, className="card-title"),
        html.P(body, className="text-muted", style={"minHeight": "3.2rem"}),
        btn,
    ])), md=4, className="mb-3")


layout = dbc.Container([
    html.H2("Overbooking & Cancellation Toolkit"),
    html.P("Weekly overbooking support for the 11 STAYERY properties. The model "
           "predicts which of the next 14 days' bookings are likely to cancel, so "
           "each property can be overbooked just enough to offset expected no-shows "
           "without walking guests.", className="text-muted"),
    dbc.Row([
        _nav_card("Occupancy & Overbooking",
                  "The main weekly workflow: 14-day risk table, KPI tiles and a "
                  "cost-optimal overbooking recommendation per property.",
                  "/occupancy"),
        _nav_card("Cancellation History",
                  "Historical cancellation rates by property, channel and season.",
                  "/cancellation-history", disabled=True),
        _nav_card("Model Performance",
                  "Model quality, calibration and explainability (XAI).",
                  "/model-performance", disabled=True),
    ]),
], fluid=True)
