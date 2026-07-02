# dash_app/pages/cancellation_history.py — placeholder (built in Phase 4).
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/cancellation-history", name="Cancellation History",
                   order=2, title="STAYERY · Cancellation History")

layout = dbc.Container([
    html.H3("Cancellation History"),
    dbc.Alert("This page is planned for a later phase and is intentionally empty for now.",
              color="secondary"),
], fluid=True)
