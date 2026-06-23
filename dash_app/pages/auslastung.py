# dash_app/pages/auslastung.py
# ---------------------------------------------------------------------------
# "Auslastung & Anreisen" — the deep arrivals-analysis page.
#
# Layout: a LEFT filter panel (sidebar-style) + a RIGHT content column with:
#   * an in-page table of contents (TOC) that jump-links to each section;
#   * KPI tiles (filtered);
#   * the 14-day operations RASTER heatmap (↑arrivals ↓departures %occupancy
#     ⚠expected-cancels per cell) — click a cell to drill the reservation table;
#   * arrival SEGMENT breakdowns (room category / channel / travel purpose) so we
#     see WHO these arrivals are and how they behave;
#   * a RESERVATION-LEVEL table (cancel probability + risk) with CSV download.
#
# Filters are GLOBAL (left panel) and the heatmap click is a per-cell OVERRIDE on
# the reservation table. Occupancy is still a booking-derived proxy (backend),
# everything else is real data via the backend facade (dummy or real).
#
# NOTE: the interactive callbacks below are import-verified here; their live
# behaviour needs the running server (python -m dash_app.app).
# ---------------------------------------------------------------------------

from __future__ import annotations

# pandas for the date window, frames, and the CSV download.
import pandas as pd

# Dash primitives. `callback`/Input/Output/State wire interactivity; `ctx` tells
# us which control fired; `dcc.Download` streams the CSV; `dash_table` is the grid.
import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, ctx

# Backend facade + derivations + config + brand UI + Plotly factories.
from dash_app import backend as B
from dash_app.backend import derive
from dash_app.backend import occupancy as occ      # property_performance_daily adapter
from dash_app.backend import schema as S           # canonical column names + labels
from dash_app import config as CFG
from dash_app.components import hero, section, explain, metric_row
from dash_app.filters_registry import register     # per-page sidebar filters
from src import plotting as P

# German weekday abbreviations (Mon=0 .. Sun=6) for the heatmap column labels.
_WD = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]

# Register this module as the /auslastung page.
dash.register_page(
    __name__,
    path="/auslastung",
    name="Auslastung & Anreisen",
    title="Auslastung & Anreisen · Stayery",
    order=CFG.PAGE_ORDER["auslastung"],
)

# ---- element ids (kept together so the callbacks below read clearly) --------
F_HOTELS, F_CHANNELS, F_PURPOSES = "aus-f-hotels", "aus-f-channels", "aus-f-purposes"
F_ROOMS, F_RISK, F_THR, F_WIN = "aus-f-rooms", "aus-f-risk", "aus-f-thr", "aus-f-win"
KPIS, HEATMAP = "aus-kpis", "aus-heatmap"
SEG_ROOM, SEG_CHANNEL, SEG_PURPOSE = "aus-seg-room", "aus-seg-channel", "aus-seg-purpose"
TABLE, DRILL_INFO = "aus-table", "aus-drill-info"
DL_BTN, DL = "aus-dl-btn", "aus-dl"
REC = "aus-rec-table"                                       # day-level recommendation table


# =============================================================================
# Small helpers
# =============================================================================
def _window_dates(n_days: int) -> list[pd.Timestamp]:
    """Return the next `n_days` calendar days (normalised to midnight)."""
    today = pd.Timestamp.today().normalize()                # midnight today
    return [today + pd.Timedelta(days=i) for i in range(int(n_days))]


def _date_label(d: pd.Timestamp) -> str:
    """Format a date as 'Mo 23.06.' for the heatmap column header."""
    return f"{_WD[d.weekday()]} {d.day:02d}.{d.month:02d}."


def _opts(values) -> list[dict]:
    """Turn a sorted unique iterable into Dash dropdown {label,value} options."""
    return [{"label": str(v), "value": v} for v in values]


def _seg_fig(df, by: str, title: str):
    """Build a segment bar chart for column `by`: bars = #bookings, with the
    average cancel rate annotated. Returns a brand Plotly figure."""
    seg = derive.segment_breakdown(df, by)                  # label / bookings / cancel_rate / ...
    # P.bars draws the booking counts; we add the cancel-rate as the bar text so
    # one chart answers "how many?" AND "how cancel-prone?".
    fig = P.bars(seg["label"].tolist(), seg["bookings"].tolist(),
                 title=title, yaxis_title="Buchungen", text_fmt=None)
    # Overwrite the bar text with "<count> · <cancel-rate>%" for richer context.
    fig.data[0].text = [f"{int(b)} · {r*100:.0f}%"
                        for b, r in zip(seg["bookings"], seg["cancel_rate"])]
    fig.data[0].textposition = "auto"
    return fig


def _kpis(df, threshold: float):
    """KPI tiles for the filtered set: bookings, expected cancels, high-risk, Ø proba."""
    n = len(df)
    exp = float(df[S.CANCEL_PROBA].sum()) if n else 0.0                 # Σ proba = expected count
    high = int((df[S.CANCEL_PROBA] >= threshold).sum()) if n else 0     # ≥ threshold
    mean = float(df[S.CANCEL_PROBA].mean()) if n else 0.0               # average probability
    return metric_row([
        {"label": "Anreisen (gefiltert)", "value": f"{n:,}".replace(",", ".")},
        {"label": "Erwartete Stornos", "value": f"{exp:.1f}",
         "help": "Summe der Storno-Wahrscheinlichkeiten (Erwartungswert; unabhängig vom Schwellwert)."},
        {"label": f"High-Risk (≥ {threshold:.0%})", "value": f"{high:,}".replace(",", "."),
         "help": "Buchungen mit Storno-Wahrscheinlichkeit ≥ Schwellwert."},
        {"label": "Ø Storno-Wahrsch.", "value": f"{mean*100:.0f} %"},
    ])


def _toc() -> html.Div:
    """In-page table of contents — anchor links that jump to each section id."""
    # html.A href="#<id>" scrolls to the element with that id (needs the running browser).
    links = [
        ("ts-kpis", "Kennzahlen"),
        ("ts-heatmap", "Prognose-Raster (14 Tage)"),
        ("ts-segments", "Wer reist an? (Segmente)"),
        ("ts-rec", "Overbooking-Empfehlung je Tag"),
        ("ts-table", "Reservierungen + CSV"),
    ]
    return html.Div(
        [html.Span("Inhalt: ", className="stayery-caption")]
        + [html.A(label, href=f"#{anchor}", className="stayery-toc-link") for anchor, label in links],
        className="stayery-toc",
    )


# =============================================================================
# Layout
# =============================================================================
def filters_layout() -> list:
    """Filter controls rendered into the SIDEBAR slot (registered for /auslastung).

    These use the SAME ids the page callback listens to — they just live in the
    sidebar now (per page). persistence=True keeps each value across navigation.
    """
    df = derive.confirmed(B.get_scored_bookings())
    units = B.units_by_hotel(); labels = B.hotel_labels()
    hotel_opts = [{"label": labels.get(h, h), "value": h} for h in units]
    chan_opts = _opts(sorted(df[S.CHANNEL].dropna().unique()))
    purp_opts = _opts(sorted(df[S.TRAVEL_PURPOSE].dropna().unique()))
    room_opts = _opts(sorted(df[S.UNIT_GROUP].dropna().unique()))
    risk_opts = [{"label": S.RISK_LABELS_DE[b], "value": b} for b in S.RISK_BUCKETS]
    return [
        html.Div("Filter", className="stayery-sidebar-heading"),
        html.Div("Standort", className="stayery-control-label"),
        dcc.Dropdown(id=F_HOTELS, options=hotel_opts, multi=True, placeholder="alle", persistence=True),
        html.Div("Kanal", className="stayery-control-label"),
        dcc.Dropdown(id=F_CHANNELS, options=chan_opts, multi=True, placeholder="alle", persistence=True),
        html.Div("Reisezweck", className="stayery-control-label"),
        dcc.Dropdown(id=F_PURPOSES, options=purp_opts, multi=True, placeholder="alle", persistence=True),
        html.Div("Zimmerkategorie", className="stayery-control-label"),
        dcc.Dropdown(id=F_ROOMS, options=room_opts, multi=True, placeholder="alle", persistence=True),
        html.Div("Risiko", className="stayery-control-label"),
        dcc.Dropdown(id=F_RISK, options=risk_opts, multi=True, placeholder="alle", persistence=True),
        html.Div("High-Risk-Schwelle", className="stayery-control-label"),
        dcc.Slider(id=F_THR, min=0.50, max=0.90, step=0.05, value=float(S.HIGH_THR),
                   marks={i/100: f"{i/100:.0%}" for i in range(50, 91, 10)}, persistence=True),
        html.Div("Zeithorizont (Tage)", className="stayery-control-label"),
        dcc.Slider(id=F_WIN, min=7, max=35, step=7, value=14,
                   marks={d: str(d) for d in (7, 14, 21, 28, 35)}, persistence=True),
    ]


# Register so these controls render in the sidebar slot when the user is on /auslastung.
register("/auslastung", filters_layout)


def layout(**kwargs) -> html.Div:
    """Page content. The filters live in the SIDEBAR (see filters_layout())."""
    content = html.Div([
        _toc(),
        # KPIs (anchor target ts-kpis).
        html.Div(section(1, "Kennzahlen", children=[html.Div(id=KPIS)]), id="ts-kpis"),

        # Heatmap (anchor ts-heatmap).
        html.Div(section(2, "Prognose-Raster — nächste Tage",
                 description="Jede Zelle: ↑ Anreisen · ↓ Abreisen · % Auslastung · ⚠ erwartete "
                             "Stornos. Farbe = Auslastung. Klick auf eine Zelle → Reservierungen "
                             "dieses Tags/Standorts unten.",
                 children=[
                     dcc.Graph(id=HEATMAP, config={"displayModeBar": False}),
                 ]), id="ts-heatmap"),

        # Segments (anchor ts-segments).
        html.Div(section(3, "Wer reist an? — Segmente",
                 description="Anzahl Buchungen je Segment, Balkenbeschriftung = '#Buchungen · Ø Storno-Quote'.",
                 children=[
                     html.Div([
                         html.Div(dcc.Graph(id=SEG_ROOM, config={"displayModeBar": False}),
                                  style={"flex": "1 1 300px", "minWidth": "0"}),
                         html.Div(dcc.Graph(id=SEG_CHANNEL, config={"displayModeBar": False}),
                                  style={"flex": "1 1 300px", "minWidth": "0"}),
                         html.Div(dcc.Graph(id=SEG_PURPOSE, config={"displayModeBar": False}),
                                  style={"flex": "1 1 300px", "minWidth": "0"}),
                     ], style={"display": "flex", "flexWrap": "wrap", "gap": "1rem"}),
                 ]), id="ts-segments"),

        # Day-level overbooking recommendation (anchor ts-rec).
        html.Div(section(4, "Overbooking-Empfehlung je Tag & Standort",
                 description="Pro Tag, nicht über das Fenster gemittelt: berücksichtigt die "
                             "wechselnde Auslastung UND die erwarteten Stornos des Tages. "
                             "Empfehlung = 0 bei Slack (Auslastung < 85 %), sonst die erwarteten "
                             "Stornos des Tages, gedeckelt aufs Limit (2 / 4).",
                 children=[
                     dash_table.DataTable(
                         id=REC, page_size=12, sort_action="native", filter_action="native",
                         style_as_list_view=True,
                         style_header={"fontWeight": "700", "backgroundColor": "#FFF7CC"},
                         style_cell={"fontFamily": "Inter, Helvetica, Arial, sans-serif",
                                     "padding": "6px 10px", "textAlign": "left", "fontSize": "13px"},
                         style_table={"overflowX": "auto"},
                     ),
                 ]), id="ts-rec"),

        # Reservation table + CSV (anchor ts-table).
        html.Div(section(5, "Reservierungen (Storno-Wahrscheinlichkeit & Risiko)",
                 description="Buchungsebene, höchste Storno-Wahrscheinlichkeit zuerst. "
                             "Globale Filter links; Heatmap-Klick überschreibt mit einem Tag/Standort.",
                 children=[
                     html.Div(id=DRILL_INFO, className="stayery-caption"),
                     # Download button + the (hidden) Download component that streams the CSV.
                     html.Button("Als CSV herunterladen", id=DL_BTN, n_clicks=0,
                                 className="stayery-btn"),
                     dcc.Download(id=DL),
                     dash_table.DataTable(
                         id=TABLE, page_size=15, sort_action="native", filter_action="native",
                         style_as_list_view=True,
                         style_header={"fontWeight": "700", "backgroundColor": "#FFF7CC"},
                         style_cell={"fontFamily": "Inter, Helvetica, Arial, sans-serif",
                                     "padding": "6px 10px", "textAlign": "left", "fontSize": "13px"},
                         style_table={"overflowX": "auto"},
                     ),
                 ]), id="ts-table"),
    ], style={"flex": "1 1 auto", "minWidth": "0"})

    return html.Div([
        hero(eyebrow="Dashboard", title="Auslastung & Anreisen",
             subtitle="Tiefe Anreise-Analyse: Segmente, Zimmerkategorien, Risiko — 14 Tage voraus."),
        content,
    ])


# =============================================================================
# Callbacks
# =============================================================================
@callback(
    Output(KPIS, "children"), Output(HEATMAP, "figure"),
    Output(SEG_ROOM, "figure"), Output(SEG_CHANNEL, "figure"), Output(SEG_PURPOSE, "figure"),
    Output(TABLE, "data"), Output(TABLE, "columns"), Output(DRILL_INFO, "children"),
    Output(REC, "data"), Output(REC, "columns"),
    Input(F_HOTELS, "value"), Input(F_CHANNELS, "value"), Input(F_PURPOSES, "value"),
    Input(F_ROOMS, "value"), Input(F_RISK, "value"), Input(F_THR, "value"),
    Input(F_WIN, "value"), Input(HEATMAP, "clickData"),
)
def _update(hotels, channels, purposes, rooms, risk, threshold, win_days, click):
    """Recompute KPIs, heatmap, segment charts and the reservation table.

    The reservation table reflects the GLOBAL filters, UNLESS the user just
    clicked a heatmap cell (ctx tells us), in which case it drills into that
    single (location, day). Changing any filter resets the drill (intuitive).
    """
    threshold = float(threshold or S.HIGH_THR)
    win_days = int(win_days or 14)
    dates = _window_dates(win_days)
    units = B.units_by_hotel()
    labels = B.hotel_labels()

    # GLOBAL-filtered, window-restricted bookings.
    df_all = B.get_scored_bookings()
    filt = derive.apply_filters(df_all, hotels=hotels, channels=channels,
                                purposes=purposes, rooms=rooms, risk=risk, dates=dates)

    # Heatmap always shows the full filtered grid (context to click into).
    # Occupancy + departures come from property_performance_daily (real) where a
    # (property, date) row exists; daily_grid falls back to the booking proxy
    # otherwise (e.g. future days the actuals table doesn't cover yet).
    perf = occ.get_perf()
    grids = derive.grid_matrices(filt, dates, units, labels=labels, perf=perf)
    nice_cols = [_date_label(d) for d in dates]
    for m in grids.values():
        m.columns = nice_cols                                # friendly date headers
    heat = P.raster_grid_heatmap(grids, color_by="occupancy",
                                 title="Operations-Raster", colorbar_title="Auslastung")
    heat.update_layout(height=max(360, 52 * len(grids["occupancy"].index) + 130))

    # A heatmap click filters the WHOLE lower view (KPIs + segments + table) to
    # that single (Standort, Tag); otherwise the global-filtered set is shown.
    drilled = (ctx.triggered_id == HEATMAP) and bool(click)
    view = filt
    info = (f"{len(filt):,} gefilterte Reservierungen ({win_days} Tage). "
            f"Klick auf eine Heatmap-Zelle filtert KPIs, Segmente und Tabelle auf "
            f"einen Tag/Standort.").replace(",", ".")
    if drilled:
        ylab = click["points"][0]["y"]                       # clicked location label
        code = next((h for h, lab in labels.items() if lab == ylab), None)  # -> hotel_code
        xlab = click["points"][0]["x"]                       # clicked date label
        date = next((d for d in dates if _date_label(d) == xlab), None)     # -> date
        if code is not None and date is not None:
            view = filt[(filt[S.HOTEL_CODE] == code) & (filt[S.ARRIVAL_DATE] == date)]
            info = (f"Drilldown: {ylab} · {xlab} — {len(view)} Reservierungen. "
                    f"Andere Zelle wechselt, Filter ändern setzt zurück.")

    # KPIs + segments + table all reflect `view` (cell-narrowed if a cell was clicked).
    kpis = _kpis(view, threshold)
    seg_room = _seg_fig(view, S.UNIT_GROUP, "Zimmerkategorie")
    seg_chan = _seg_fig(view, S.CHANNEL, "Kanal")
    seg_purp = _seg_fig(view, S.TRAVEL_PURPOSE, "Reisezweck")
    tbl = derive.reservation_view(view)
    columns = [{"name": c, "id": c} for c in tbl.columns]

    # Day-level overbooking recommendation reflects the GLOBAL filter (not the
    # single clicked cell) — it's a per-(Standort, Tag) plan over the window.
    rec = derive.recommendation_by_day(filt, dates, units, perf=perf, labels=labels)
    rec_cols = [{"name": c, "id": c} for c in rec.columns]
    return (kpis, heat, seg_room, seg_chan, seg_purp, tbl.to_dict("records"), columns, info,
            rec.to_dict("records"), rec_cols)


@callback(
    Output(DL, "data"),
    Input(DL_BTN, "n_clicks"),
    State(TABLE, "data"), State(TABLE, "columns"),
    prevent_initial_call=True,
)
def _download_csv(n_clicks, data, columns):
    """Stream the CURRENT reservation table (filtered/drilled) as a CSV download."""
    if not data:                                            # nothing to download
        return dash.no_update
    # Rebuild a DataFrame in the displayed column order, then send as CSV.
    cols = [c["id"] for c in (columns or [])]
    frame = pd.DataFrame(data)[cols] if cols else pd.DataFrame(data)
    # dcc.send_data_frame wraps a to_csv call into a browser download payload.
    return dcc.send_data_frame(frame.to_csv, "reservierungen.csv", index=False)
