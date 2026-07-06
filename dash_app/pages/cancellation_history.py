# dash_app/pages/cancellation_history.py
# PAGE 3 — Cancellation History (historical, read-only). Everything reads the cleaned
# reservations cache via dash_app.backend.cancellation_history; no live BigQuery, ever.
# Cancellations are the measured quantity here, so (unlike Occupancy) they are NOT
# excluded. One global location filter drives every chart; clicking a heatmap cell or a
# month opens a right-side detail Drawer. Built with dash-mantine-components as the new
# shared design system (see components/ui.py) + Plotly figures (components/history_charts).

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from dash_app import theme
from dash_app.backend import cancellation_history as ch
from dash_app.components import history_charts as hc
from dash_app.components import ui

dash.register_page(__name__, path="/cancellation-history", name="Cancellation History",
                   order=2, title="STAYERY · Cancellation History")

_WINDOW_DATA = [{"label": "6 mo", "value": "6"}, {"label": "12 mo", "value": "12"}]
_MODE_DATA = [{"label": "Aggregate", "value": "agg"},
              {"label": "Per location", "value": "per"}]

# Metric explanations surfaced via ⓘ tooltips (UX requirement: no unexplained numbers).
_INFO = {
    "monthly": "Share of bookings (per arrival month) that were cancelled before "
               "arrival. The dashed line is the overall base rate across all history. "
               "Months with fewer than 50 bookings are hidden as too noisy to trust.",
    "heatmap": "Average cancellation rate per location and arrival month. Green = low, "
               "red = high. Blank cells have too few bookings (<30) to show a reliable "
               "rate. Click a cell for its detail.",
    "channel": "How each booking channel's cancellation rate compares to the current "
               "selection's base rate. Red = cancels more than average, green = less. "
               "Channels with fewer than 200 bookings are omitted.",
    "stay": "Cancellation rate by length of stay. Short = 1–2 nights, Mid = 3–6, "
            "Long = 7+ nights.",
    "lead": "Cancellation rate by lead time — the gap between when a booking was made "
            "and the arrival date.",
    "timing": "Of the bookings that cancelled, the cumulative share that had cancelled "
              "by a given number of days before arrival. Shows how late rooms typically "
              "free up.",
}


def _sel(value) -> list[str] | None:
    """Normalise the MultiSelect value; an empty selection means 'all locations'."""
    return list(value) if value else None


# ---------------------------------------------------------------------------
# Layout (callable => property list re-read on each navigation)
# ---------------------------------------------------------------------------
def layout(**_kwargs):
    props = ch.property_list()
    span = ch.date_span()
    span_label = f"History {span[0]} – {span[1]}" if span else "No cached history"

    header = dmc.Group([
        dmc.Group([
            dmc.Title("Cancellation history", order=3),
            dmc.Badge("Historical · read-only", color="gray", variant="light", radius="sm"),
        ], gap="sm", align="center"),
        dmc.Text("How cancellations behave over time, by channel, stay length and lead "
                 "time — globally or per location.", size="sm", c="dimmed"),
    ], justify="space-between", align="center", wrap="wrap", mb="xs")

    monthly_extra = dmc.SegmentedControl(id="cxl-per-prop", data=_MODE_DATA, value="agg",
                                         size="xs", radius="md")
    heatmap_extra = dmc.SegmentedControl(id="cxl-window", data=_WINDOW_DATA, value="12",
                                         size="xs", radius="md")

    breakdowns = dmc.SimpleGrid(
        [ui.chart_card("Channel risk", "cxl-channel", info=_INFO["channel"], height=300),
         ui.chart_card("Length of stay", "cxl-stay", info=_INFO["stay"], height=300),
         ui.chart_card("Lead time", "cxl-lead", info=_INFO["lead"], height=300)],
        cols={"base": 1, "md": 3}, spacing="md")

    drawer = dmc.Drawer(
        id="cxl-drawer", position="right", size="md", padding="lg", opened=False,
        title=dmc.Text("Detail", fw=700), withCloseButton=True,
        closeOnClickOutside=True, closeOnEscape=True,
        children=html.Div(id="cxl-drawer-body"))

    return dmc.Stack([
        header,
        ui.location_filter(props, "cxl-property-filter", span_label=span_label),
        dcc.Store(id="cxl-drawer-store"),

        html.Div(id="cxl-kpi", children=dmc.Skeleton(height=96, radius="lg")),
        html.Div(id="cxl-anomaly"),

        ui.chart_card("Cancellation rate over time", "cxl-monthly", info=_INFO["monthly"],
                      height=340, header_extra=monthly_extra,
                      subtitle="Monthly · click a point for that month's location breakdown."),

        ui.chart_card("Cancellation rate · location × month", "cxl-heatmap",
                      info=_INFO["heatmap"], height=430, header_extra=heatmap_extra,
                      subtitle="Click a cell to drill into that location × month."),

        breakdowns,

        ui.chart_card("Cancel-timing curve", "cxl-timing", info=_INFO["timing"], height=320,
                      subtitle="How late before arrival do cancellations land?"),

        drawer,
    ], gap="md")


# ---------------------------------------------------------------------------
# KPI + anomaly + filter-only charts (channel / stay / lead / timing)
# ---------------------------------------------------------------------------
def _pct(x, digits=1):
    return "unavailable" if x is None else f"{x * 100:.{digits}f}%"


def _kpi_cards(k: dict) -> list:
    overall = _pct(k["overall_rate"])
    n_txt = f"{k['n_bookings']:,}" if k["n_bookings"] is not None else "unavailable"
    base_txt = _pct(k["base_rate"])

    if k["latest_rate"] is not None and k["delta_vs_base"] is not None:
        latest_val = _pct(k["latest_rate"])
        latest_sub = f"{k['latest_month']} · {k['delta_vs_base'] * 100:+.1f} pp vs base"
    else:
        latest_val, latest_sub = "unavailable", "no complete month"

    if k["top_property"] is not None:
        top_val, top_sub = k["top_property"], f"{_pct(k['top_rate'])} cancel rate"
    else:
        top_val, top_sub = "unavailable", "below sample threshold"

    return [
        ui.kpi_card("Cancel rate · selection", overall, sub=f"{n_txt} bookings · base {base_txt}",
                    accent=True, tooltip="Cancellation rate across the selected locations "
                                         "and the whole available history."),
        ui.kpi_card("Latest month", latest_val, sub=latest_sub,
                    tooltip="Most recent month with enough bookings, vs the overall base rate."),
        ui.kpi_card("Highest-rate location", top_val, sub=top_sub,
                    tooltip="Selected location with the highest cancellation rate "
                            "(needs ≥100 bookings to qualify)."),
        ui.kpi_card("Bookings in view", n_txt, sub="resolved bookings in the selection"),
    ]


def _anomaly_alert(props):
    hot = ch.flag_anomalies(props)
    if hot.empty:
        return None
    top = hot.head(3)
    parts = ", ".join(
        f"{r.property_name} {pd.Timestamp(r.month):%b %Y} ({r.cancel_rate * 100:.0f}%)"
        for r in top.itertuples())
    base = hot["base_rate"].iloc[0]
    return dmc.Alert(
        dmc.Text(f"{len(hot)} location-month(s) run ≥1.5× the base rate "
                 f"({base * 100:.1f}%). Highest: {parts}.", size="sm"),
        title="Unusually high cancellation months", color="red", variant="light",
        icon=html.I(className="bi bi-exclamation-triangle"), radius="md",
        withCloseButton=True)


@callback(
    Output("cxl-kpi", "children"),
    Output("cxl-anomaly", "children"),
    Output("cxl-channel", "figure"),
    Output("cxl-stay", "figure"),
    Output("cxl-lead", "figure"),
    Output("cxl-timing", "figure"),
    Input("cxl-property-filter", "value"),
)
def _update_filter(sel_value):
    props = _sel(sel_value)
    k = ch.kpis(props)
    kpis = ui.kpi_strip(_kpi_cards(k))
    anomaly = _anomaly_alert(props)

    dev, base = ch.channel_deviation(props)
    stay = ch.stay_segment_rate(props)
    lead = ch.leadtime_bucket_rate(props)
    curve, n = ch.cancel_timing_curve(props)
    overall = k["overall_rate"]

    return (kpis, anomaly,
            hc.fig_channel(dev, base),
            hc.fig_rate_bars(stay, "label", base=overall, color=theme.GREEN),
            hc.fig_rate_bars(lead, "bucket", base=overall, color=theme.ORANGE),
            hc.fig_timing(curve, n))


# ---------------------------------------------------------------------------
# Monthly line (filter + aggregate/per-location mode)
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-monthly", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-per-prop", "value"),
)
def _update_monthly(sel_value, mode):
    props = _sel(sel_value)
    monthly = ch.monthly_rate(props)
    per = ch.monthly_rate(props, per_property=True) if mode == "per" else None
    return hc.fig_monthly(monthly, ch.base_rate(), per)


# ---------------------------------------------------------------------------
# Heatmap (filter + 6/12-month window)
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-heatmap", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-window", "value"),
)
def _update_heatmap(sel_value, window):
    props = _sel(sel_value)
    months = int(window or "12")
    return hc.fig_heatmap(ch.property_month_matrix(props, months_back=months))


# ---------------------------------------------------------------------------
# Drill-down Drawer: heatmap-cell click OR month-point click
# ---------------------------------------------------------------------------
def _stat(label, value):
    return dmc.Stack([dmc.Text(label, size="xs", c="dimmed"),
                      dmc.Text(value, fw=700, size="lg")], gap=0)


def _track(rate: float):
    pct = min(100.0, max(0.0, rate * 100))
    color = theme.RED if rate >= 0.30 else theme.ORANGE if rate >= 0.20 else theme.GREEN
    return html.Div(
        html.Div(style={"height": "6px", "width": f"{pct}%", "backgroundColor": color,
                        "borderRadius": "3px"}),
        style={"height": "6px", "backgroundColor": "#EEEEEE", "borderRadius": "3px",
               "overflow": "hidden"})


def _rate_row(name, rate, n):
    return dmc.Stack([
        dmc.Group([dmc.Text(str(name), size="sm"),
                   dmc.Text(f"{rate * 100:.0f}% · {n:,}", size="xs", c="dimmed")],
                  justify="space-between", wrap="nowrap"),
        _track(rate)], gap=2)


def _section(text):
    return dmc.Text(text, fw=600, size="sm", mt="sm", mb=2)


def _cell_body(d):
    if not d:
        return dmc.Text("No bookings for this location × month.", c="dimmed")
    return dmc.Stack([
        dmc.Text(d["month"], fw=700, size="lg"),
        dmc.Group([_stat("Cancel rate", f"{d['rate'] * 100:.1f}%"),
                   _stat("Bookings", f"{d['n']:,}")], gap="xl"),
        _section("By channel"),
        *[_rate_row(r["channel"], r["cancel_rate"], r["n"]) for r in d["channels"]],
        _section("By stay segment"),
        *[_rate_row(r["label"], r["cancel_rate"], r["n"]) for r in d["stays"]],
    ], gap=6)


def _month_body(d):
    if not d:
        return dmc.Text("No bookings this month for the current selection.", c="dimmed")
    return dmc.Stack([
        dmc.Group([_stat("Cancel rate", f"{d['rate'] * 100:.1f}%"),
                   _stat("Bookings", f"{d['n']:,}")], gap="xl"),
        _section("By location"),
        *[_rate_row(r["property_name"], r["cancel_rate"], r["n"]) for r in d["properties"]],
    ], gap=6)


@callback(
    Output("cxl-drawer", "opened"),
    Output("cxl-drawer", "title"),
    Output("cxl-drawer-body", "children"),
    Input("cxl-heatmap", "clickData"),
    Input("cxl-monthly", "clickData"),
    State("cxl-property-filter", "value"),
    prevent_initial_call=True,
)
def _drawer(hm_click, ml_click, sel_value):
    props = _sel(sel_value)
    trig = ctx.triggered_id
    if trig == "cxl-heatmap" and hm_click:
        p = hm_click["points"][0]
        prop, month_iso = p["y"], p["x"]
        detail = ch.drill_property_month(prop, month_iso)
        return True, dmc.Text(prop, fw=700), _cell_body(detail)
    if trig == "cxl-monthly" and ml_click:
        p = ml_click["points"][0]
        month_iso = pd.to_datetime(p["x"]).strftime("%Y-%m")
        detail = ch.drill_month(props, month_iso)
        title = detail["month"] if detail else "Month detail"
        return True, dmc.Text(title, fw=700), _month_body(detail)
    return no_update, no_update, no_update
