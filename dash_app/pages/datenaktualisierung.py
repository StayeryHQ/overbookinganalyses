# dash_app/pages/datenaktualisierung.py
# ---------------------------------------------------------------------------
# "Datenaktualisierung" — STUB page (data-refresh trigger). Mirrors
# streamlit_app/pages/3_Datenaktualisierung.py at a minimal level: it shows the
# snapshot status, lets you flip the backend mode (dummy <-> real), and trigger
# a refresh. The heavy lifting (BigQuery pull) is the backend's job; the page is
# intentionally light with a clear TODO for the eventual richer UI.
# ---------------------------------------------------------------------------

from __future__ import annotations

import dash
from dash import callback, dcc, html, Input, Output, State

# Backend facade (mode toggle + refresh + metadata) + UI helpers + config.
from dash_app import backend as B
from dash_app.components import alert, hero, metric_row, placeholder_panel
from dash_app import config as CFG

# Register at /datenaktualisierung.
dash.register_page(
    __name__,
    path="/datenaktualisierung",
    name="Datenaktualisierung",
    title="Datenaktualisierung · Stayery",
    order=CFG.PAGE_ORDER["datenaktualisierung"],
)

# Element ids.
_MODE_RADIO = "data-mode-radio"
_REFRESH_BTN = "data-refresh-btn"
_STATUS = "data-status-row"
_RESULT = "data-refresh-result"


def _status_row() -> html.Div:
    """Build the snapshot-status metric row from backend metadata."""
    meta = B.get_metadata()
    refreshed = str(meta.get("refreshed_at", "?"))[:19].replace("T", " ")
    win = meta.get("window", {})
    return html.Div([
        metric_row([
            {"label": "Letzter Refresh", "value": refreshed},
            {"label": "Bestätigt", "value": f"{meta.get('confirmed', {}).get('rows', 0):,}".replace(",", ".")},
            {"label": "Storniert (raus)", "value": f"{meta.get('canceled', {}).get('rows', 0):,}".replace(",", ".")},
            {"label": "Anreisen (künftig)", "value": f"{meta['upcoming']['rows']:,}".replace(",", ".")},
        ]),
        html.Div(
            f"Anreise-Range: {win.get('earliest', '?')} bis {win.get('latest', '?')} · "
            f"{len(meta.get('properties', []))} Standorte · Modus: {meta.get('mode')} · "
            f"Modell: {meta.get('model')}",
            className="stayery-caption",
        ),
    ])


def layout(**kwargs) -> html.Div:
    """Stub layout: status + working mode toggle + refresh button + planned TODO."""
    return html.Div([
        hero(
            eyebrow="Daten",
            title="Datenaktualisierung",
            subtitle="Snapshot-Status ansehen, Backend-Modus umschalten und Daten refreshen.",
        ),

        # Live snapshot status (replaced by the callback after a refresh).
        html.H2("Aktueller Snapshot"),
        html.Div(_status_row(), id=_STATUS),

        html.Hr(),

        # Backend mode toggle. This drives the SAME B.set_mode() the whole app
        # reads, so flipping it here changes the data source for every page.
        html.H2("Backend-Modus"),
        html.P("Die Seiten kennen nur ein einheitliches Schema — der Wechsel zwischen "
               "synthetischem Platzhalter und echtem Modell ist ein reiner Backend-Tausch.",
               className="stayery-section-desc"),
        dcc.RadioItems(
            id=_MODE_RADIO,
            options=[
                {"label": " Dummy-Modell (synthetisch, Standard)", "value": "dummy"},
                {"label": " Echtes Modell (src.score_upcoming)", "value": "real"},
            ],
            value=B.mode(),                # current mode preselected
            labelStyle={"display": "block", "marginBottom": "0.3rem"},
        ),

        html.Hr(),

        # Refresh trigger. n_clicks increments on each click and fires the callback.
        html.H2("Aktualisieren"),
        html.Button("Daten aktualisieren", id=_REFRESH_BTN, n_clicks=0,
                    className="stayery-btn stayery-btn--primary"),
        # Where the refresh result message is rendered.
        html.Div(id=_RESULT, style={"marginTop": "0.8rem"}),

        html.Hr(),

        # Placeholder describing the richer UI still to come.
        placeholder_panel(
            title="Erweiterung geplant",
            intro="Diese Seite ist bewusst schlank gehalten. Geplant für die Vollversion:",
            planned=[
                "Detaillierter Refresh-Log mit Schritt-für-Schritt-Status",
                "Real-Modus: Google-Cloud-Auth-Check vor dem BigQuery-Pull",
                "Anzeige des verwendeten Modells + Score-Zeitstempel",
            ],
            badge="Stub · TODO",
        ),
    ])


# ---------------------------------------------------------------------------
# Callback: clicking "Daten aktualisieren" applies the selected mode, runs the
# backend refresh, then re-renders the status row + a result message.
#   * Input : the button's n_clicks (the trigger).
#   * State : the selected mode (read, not a trigger).
#   * Output: the result message + the refreshed status row.
# prevent_initial_call=True stops it firing on page load (only on real clicks).
# ---------------------------------------------------------------------------
@callback(
    Output(_RESULT, "children"),
    Output(_STATUS, "children"),
    Input(_REFRESH_BTN, "n_clicks"),
    State(_MODE_RADIO, "value"),
    prevent_initial_call=True,
)
def _on_refresh(n_clicks: int, selected_mode: str):
    """Apply the chosen mode, refresh, and report back."""
    # Apply the selected backend mode for the whole process.
    B.set_mode(selected_mode)
    try:
        # Run the refresh (dummy: new seed; real: BigQuery pull + score).
        meta = B.refresh()
        msg = alert(
            f"Fertig. {meta['reservations']['rows']:,} Buchungen, "
            f"{meta['upcoming']['rows']:,} künftige Anreisen, "
            f"{len(meta['properties'])} Standorte.".replace(",", "."),
            kind="success",
        )
    except Exception as e:  # noqa: BLE001 — surface refresh failures cleanly
        # Most likely in real mode without a trained model / GCP auth.
        msg = alert(f"Refresh fehlgeschlagen: {type(e).__name__}: {e}", kind="warning")
    # Always re-render the status row so it reflects the latest state.
    return msg, _status_row()
