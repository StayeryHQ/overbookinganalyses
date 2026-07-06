# dash_app/components/shap_explain.py
# Reusable single-booking SHAP explanation unit (spec 4.8): "booking + model in, SHAP plot
# out." Built as ONE component with two sizes so the SAME logic is used full-size on the
# Model-Performance page AND mini in the Overbooking page's booking sidebar later — no
# duplicate explanation logic on the Overbooking side.
#
# The heavy lifting (model-agnostic SHAP over the scalar adapter) lives in
# dash_app.backend.explain.single_contribution; this module only turns one booking row into
# a figure + a small dmc panel.

from __future__ import annotations

import dash_mantine_components as dmc
import pandas as pd
from dash import dcc

from dash_app.backend import explain as ex
from dash_app.components import performance_charts as pc


def waterfall_figure(model: str, booking: pd.Series | pd.DataFrame, *, mini: bool = False):
    """go.Figure waterfall explaining ONE booking's P(cancel) for `model`. Empty-state safe."""
    contrib = ex.single_contribution(model, booking)
    return pc.fig_waterfall(contrib, height=300 if mini else 460)


def explanation_panel(model: str, booking: pd.Series | pd.DataFrame, *,
                      mini: bool = False, title: str | None = None):
    """Reusable dmc panel: title + waterfall. `mini=True` renders the compact sidebar form.

    Use full on the Model-Performance page (drawer) and mini in the Overbooking sidebar:
        shap_explain.explanation_panel(model, booking_row, mini=True)
    """
    fig = waterfall_figure(model, booking, mini=mini)
    graph = dcc.Graph(figure=fig, config={"displayModeBar": False},
                      style={"height": f"{300 if mini else 460}px"})
    header = dmc.Text(title or f"Why this booking · {model}", fw=700,
                      size="sm" if mini else "md")
    body = [header, dmc.Space(h=4), graph]
    if not mini:
        body.append(dmc.Text("Green pushes P(cancel) down, red pushes it up. Bars are the "
                             "model-agnostic SHAP contributions on the same P(cancel-by-arrival) "
                             "scale used everywhere on this page.", size="xs", c="dimmed"))
    return dmc.Stack(body, gap=2)
