# dash_app/pages/home.py
# ---------------------------------------------------------------------------
# Landing / overview page. Mirrors streamlit_app/Home.py: hero, data-status
# metrics, navigation tiles to the four work areas, and the locations table.
#
# Dash multi-page mechanics:
#   * dash.register_page(__name__, ...) registers THIS module as a route. With
#     path="/" it is the home route; `name` is the sidebar label; `order` sets
#     its position; `title` is the browser tab title.
#   * `layout` may be a value OR a function. We use a function so the metrics are
#     recomputed each visit (the snapshot can change after a refresh).
# ---------------------------------------------------------------------------

from __future__ import annotations

import dash
import pandas as pd
from dash import html, dash_table

# Backend facade (mode-agnostic data access) + UI helpers + config.
from dash_app import backend as B
from dash_app.backend import schema as S
from dash_app.components import alert, hero, metric_row, nav_card, nav_card_grid
from dash_app import config as CFG

# Reuse the project's locations table + the overbooking benchmark rule from src.
from src.utils import load_locations, benchmark_overbooking_allowance

# Register this module as the "/" route, labelled "Übersicht" in the sidebar.
dash.register_page(
    __name__,
    path="/",                       # home route
    name="Übersicht",               # sidebar label
    title="Stayery Overbooking Analytics",  # browser tab title
    order=CFG.PAGE_ORDER["home"],   # nav ordering from the central config
)


def _datastand_metrics() -> html.Div:
    """Build the 'Datenstand' metric row from backend metadata (with a fallback)."""
    try:
        # Pull the data-status summary (rows, cancel rate, high-risk count, ...).
        meta = B.get_metadata()
    except Exception as e:  # noqa: BLE001 — never let the landing page crash
        # On any backend error, show a warning instead of metrics.
        return alert(f"Datenstand konnte nicht geladen werden: {type(e).__name__}: {e}",
                     kind="warning")
    # Format the refresh timestamp like the Streamlit page ("YYYY-MM-DD HH:MM:SS").
    refreshed = str(meta.get("refreshed_at", "?"))[:19].replace("T", " ")
    # Compose the four headline metrics.
    return metric_row([
        {"label": "Letzter Refresh", "value": refreshed},
        {"label": "Künftige Anreisen", "value": f"{meta['upcoming']['rows']:,}".replace(",", ".")},
        {"label": "Ø Storno-Quote", "value": f"{meta.get('cancel_rate', 0) * 100:.0f} %",
         "help": "Durchschnittliche prognostizierte Storno-Wahrscheinlichkeit der künftigen Anreisen."},
        {"label": "High-Risk-Buchungen", "value": meta.get("high_risk", 0),
         "help": "Buchungen mit Storno-Wahrscheinlichkeit ≥ 75 %."},
    ])


def _locations_table() -> html.Div:
    """Build the locations reference table (from configs/locations.yaml via src)."""
    loc = load_locations()
    if loc.empty:
        return alert("Keine Standorte in `configs/locations.yaml` gefunden.", kind="warning")
    # Build a display frame mirroring the Streamlit columns.
    table = pd.DataFrame({
        "Code": loc["hotel_code"],
        "Stadt": loc.get("city", ""),
        "Bundesland": loc.get("bundesland", ""),
        "Units": loc["units_total"],
        # The benchmark overbooking allowance per property size.
        "Overbooking-Limit": loc["units_total"].apply(benchmark_overbooking_allowance),
    })
    # dash_table.DataTable renders an interactive table. We pass:
    #   * columns: list of {name, id} dicts (header label + data key);
    #   * data: list-of-dicts (one per row) via to_dict("records").
    return html.Div(
        dash_table.DataTable(
            columns=[{"name": c, "id": c} for c in table.columns],
            data=table.to_dict("records"),
            # Light styling consistent with the brand (CSS handles the rest).
            style_cell={"fontFamily": "inherit", "fontSize": "0.85rem",
                        "padding": "0.5rem", "textAlign": "left", "border": "none"},
            style_header={"fontWeight": "600"},
            page_size=20,
        ),
        className="stayery-table",
    )


def layout(**kwargs) -> html.Div:
    """Page layout factory — called by Dash each time the home route renders."""
    return html.Div([
        # Editorial hero.
        hero(
            eyebrow="Stayery · Overbooking Analytics",
            title="Auslastung verstehen, Stornos vorhersagen, Overbooking steuern",
            subtitle="Wähl links einen Bereich. Die Predictions-Seite zeigt die "
            "prognostizierten Stornos je Standort, die Modell-Seite erklärt, wie das "
            "Storno-Modell rechnet.",
        ),
        # Data-status section.
        html.H2("Datenstand"),
        _datastand_metrics(),
        html.Hr(),
        # Navigation tiles to the four areas (whole card is a link).
        html.H2("Bereiche"),
        nav_card_grid([
            nav_card(href="/auslastung", icon_name="calendar", kicker="Dashboard",
                     title="Auslastung & Anreisen",
                     desc="Belegung, An- und Abreisen je Standort und Zimmerkategorie. "
                          "Stub — kommt mit der BigQuery-Occupancy-Tabelle.",
                     status="soon"),
            nav_card(href="/overbooking-predictions", icon_name="trend", kicker="Vorhersage",
                     title="Overbooking Predictions",
                     desc="Erwartete Stornos je Standort und Tag, Empfehlung gegen das "
                          "Overbooking-Limit und die Cancel-Wahrscheinlichkeit pro Buchung.",
                     status="ready"),
            nav_card(href="/modell-performance", icon_name="bolt", kicker="Modell",
                     title="Modell & Performance",
                     desc="Wie das Storno-Modell rechnet: Confusion-Matrix, ROC/PR, "
                          "Feature-Einfluss, Kalibrierung.",
                     status="ready"),
            nav_card(href="/datenaktualisierung", icon_name="refresh", kicker="Daten",
                     title="Datenaktualisierung",
                     desc="Snapshot-Status, Refresh per Knopfdruck und der Umschalter "
                          "zwischen Platzhalter-Backend und echtem Modell.",
                     status="soon"),
        ]),
        html.Hr(),
        # Locations reference.
        html.H2("Standorte"),
        html.P("Hotel-Stammdaten aus configs/locations.yaml. Overbooking-Limit nach "
               "Benchmark: unter 50 Units → 2 erlaubt, ab 50 Units → 4 erlaubt.",
               className="stayery-section-desc"),
        _locations_table(),
        html.Hr(),
        html.Div("Stayery Overbooking Analytics · Dash-App, Design gespiegelt vom Streamlit-Frontend.",
                 className="stayery-caption"),
    ])
