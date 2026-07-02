# dash_app/pages/model_performance.py — placeholder (built in Phase 4).
from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
from dash import html

dash.register_page(__name__, path="/model-performance", name="Model Performance",
                   order=3, title="STAYERY · Model Performance")

layout = dbc.Container([
    html.H3("Model Performance & Explainability"),
    dbc.Alert("This page is planned for a later phase and is intentionally empty for now.",
              color="secondary"),
], fluid=True)
