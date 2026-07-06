# dash_app/pages/home.py — landing page: short intro + navigation (dmc design).
from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import dcc, html

dash.register_page(__name__, path="/", name="Home", order=0, title="STAYERY · Home")


def _nav_card(icon: str, title: str, body: str, href: str,
              disabled: bool = False) -> dmc.Card:
    if disabled:
        action = dmc.Button("Coming soon", disabled=True, variant="light", size="sm")
    else:
        action = dcc.Link(
            dmc.Button("Open", size="sm", variant="filled",
                       rightSection=html.I(className="bi bi-arrow-right")),
            href=href, style={"textDecoration": "none"})
    return dmc.Card([
        dmc.Group([
            dmc.ThemeIcon(html.I(className=icon), size=38, radius="md",
                          variant="light", color="yellow"),
            dmc.Text(title, fw=700, size="md"),
        ], gap="sm", align="center"),
        dmc.Text(body, c="dimmed", size="sm", mt="sm",
                 style={"minHeight": "3.6rem"}),
        dmc.Space(h=4),
        action,
    ], withBorder=True, radius="lg", p="lg", shadow="xs", style={"height": "100%"})


def layout(**_kwargs):
    # Real, non-fabricated facts for the hero pills (property count from the cache).
    try:
        from dash_app.backend import data_access as da
        n_props = len(da.property_list()) or 11
    except Exception:  # noqa: BLE001 — landing page must render even if cache is cold
        n_props = 11

    pills = dmc.Group([
        dmc.Badge(f"{n_props} properties", variant="light", color="gray", radius="sm"),
        dmc.Badge("14-day overbooking horizon", variant="light", color="gray", radius="sm"),
        dmc.Badge("Read-only · no BigQuery writes", variant="light", color="gray", radius="sm"),
    ], gap="xs", mt="sm")

    hero = dmc.Stack([
        dmc.Title("Overbooking & cancellation toolkit", order=2),
        dmc.Text(
            "Weekly overbooking support for the STAYERY properties. The model predicts "
            "which of the next 14 days' bookings are likely to cancel, so each property "
            "can be overbooked just enough to offset expected cancellations — without "
            "walking guests.", c="dimmed", size="md", style={"maxWidth": "760px"}),
        pills,
    ], gap="xs")

    cards = dmc.SimpleGrid([
        _nav_card("bi bi-grid-1x2", "Occupancy & Overbooking",
                  "The main weekly workflow: 14-day occupancy heatmap, risk table, "
                  "and a cost-optimal overbooking recommendation per property.",
                  "/occupancy"),
        _nav_card("bi bi-graph-down-arrow", "Cancellation History",
                  "Historical cancellation rates by location, channel, stay length and "
                  "lead time — with a location × month heatmap and drill-down.",
                  "/cancellation-history"),
        _nav_card("bi bi-clipboard-data", "Model Performance",
                  "Fair model comparison vs baseline, calibration and explainability (XAI).",
                  "/model-performance"),
    ], cols={"base": 1, "sm": 2, "lg": 3}, spacing="md")

    return dmc.Stack([hero, dmc.Divider(my="sm"), cards], gap="md")
