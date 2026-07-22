# dash_app/pages/cancellation_history.py
# PAGE 3 - Cancellation History (historical, read-only). Everything reads the cleaned
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
from dash_app.backend import jobs                       # file-backed job runner
from dash_app.backend import model_ops as mo
from dash_app.components import history_charts as hc
from dash_app.components import ui

_HIDDEN = {"display": "none"}
_SHOWN = {"display": "block"}


def _history_update_card() -> dmc.Paper:
    """Green 'update history' card (mirrors the Occupancy scoring card): pulls the full
    reservations history and rebuilds the cleaned cache on the file-backed job runner."""
    return dmc.Paper(dmc.Stack([
        dmc.Group([
            dmc.Stack([
                dmc.Text("Update cancellation history", fw=600, size="sm"),
                dmc.Text("Pulls the full reservations history from BigQuery and rebuilds "
                         "the cleaned dataset behind these charts. Runs in the background; "
                         "progress survives page changes.", size="xs", c="dimmed"),
            ], gap=2),
            dmc.Button("Update history", id="cxl-upd-btn", size="sm", variant="filled",
                       leftSection=html.I(className="bi bi-arrow-clockwise")),
        ], justify="space-between", align="center", wrap="wrap"),
        ui.job_loader("cxl-upd"),
        html.Div(id="cxl-upd-result"),
    ], gap="xs"), p="md", radius="lg", withBorder=True)


def _history_result(res: dict) -> dmc.Alert:
    return dmc.Alert(dmc.Stack([
        dmc.Text(f"History rebuilt: {res.get('clean_rows', 0):,} cleaned bookings from "
                 f"{res.get('raw_rows', 0):,} raw rows in {res.get('elapsed_s', '?')}s.",
                 fw=600, size="sm"),
        dmc.Text(f"Cancel base rate {res.get('base_rate', 0) * 100:.1f}% · history "
                 f"{res.get('span_start', '?')} → {res.get('span_end', '?')}.",
                 size="xs", c="dimmed"),
    ], gap=6), color="green", variant="light", icon=html.I(className="bi bi-check-circle"))


def _err_alert(text: str) -> dmc.Alert:
    return dmc.Alert(dmc.Text(str(text), size="sm"), color="red", variant="light",
                     title="Update failed", icon=html.I(className="bi bi-exclamation-triangle"))


def _cancelled_alert() -> dmc.Alert:
    return dmc.Alert("History update cancelled  previous data kept.", color="gray",
                     variant="light", icon=html.I(className="bi bi-x-circle"))

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
    "stay": "Cancellation rate by exact length of stay, night by night (stays of 14+ "
            "nights are pooled into one bar). Bar colour marks the segment: green = "
            "short (1–2 nights), amber = mid (3–6), red = long (7+). Bars over fewer "
            "than 50 bookings are hidden as too noisy.",
    "lead": "Cancellation rate by lead time  the gap in days between when a booking was "
            "made and the arrival date  shown day by day (longer lead almost always "
            "cancels more). Use 'by length of stay' to split the curve into short / mid "
            "/ long, and the day toggle to extend the window. Days over fewer than 50 "
            "bookings are hidden.",
    "timing": "Of the bookings that cancelled, the cumulative share that had cancelled "
              "by a given number of days before arrival. Shows how late rooms typically "
              "free up.",
    "timewin": "Filters every chart on this page to the last 6 / 12 / 24 months of "
               "arrivals (or the full history). The dashed base-rate reference always "
               "stays the full-history average, so a window is easy to compare against it.",
    "blend": "Cancellation rate for each combination of lead time and length of stay  "
             "the two biggest drivers blended into one grid. Green = low, red = high. "
             "Top-right (long stays booked far ahead) is the riskiest mix. Cells under "
             "30 bookings are hidden.",
    "ct_grid": "Cancellations in the last 14 days before arrival, day by day (0 = arrival "
               "day). In 'Cancel rate' mode each cell is a per-day rate: cancellations "
               "that day ÷ the bookings that were still due to arrive that day (booked at "
               "least that far ahead and not yet cancelled)  so it answers 'of the "
               "bookings still due to arrive d days out, what share cancel that day', not "
               "a share of all bookings. A row therefore does NOT sum to the segment's "
               "overall cancel rate, and a 0–7 day lead bucket is simply empty past day 7. "
               "'Count' mode shows the raw number of cancellations. Rows switch between "
               "length of stay and lead time; both count and base are always in the hover.",
    "ns_monthly": "Share of resolved arrivals (guests who did NOT cancel before arrival) "
                  "that were no-shows, per arrival month. Based on the raw reservations "
                  "history. Months with fewer than 50 arrivals are hidden. Hover shows the "
                  "no-show count and the arrivals base.",
    "ns_heatmap": "No-show rate per location and arrival month. Blank cells have fewer "
                  "than 30 resolved arrivals. Hover shows the no-show count and arrivals.",
    "ns_stay": "No-show rate by length of stay (short 1–2, mid 3–6, long 7+ nights). "
               "Each bar is labelled with the absolute number of no-shows.",
    "ct_hist": "How many cancellations land at each whole day before arrival, in exact "
               "daily bins (bin '0–1' = cancelled on the arrival day, '1–2' = the day "
               "before, and so on). Everything 30+ days out pools into the last bar. This "
               "is a raw count of cancellations, not a rate. Toggle 'by length of stay' to "
               "split the bars into short / mid / long.",
}


def _sel(value) -> list[str] | None:
    """Normalise the MultiSelect value; an empty selection means 'all locations'."""
    return list(value) if value else None


def _win(value) -> int | None:
    """Normalise the time-window control; 'all'/empty means the full history."""
    return None if not value or value == "all" else int(value)


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
                 "time - globally or per location.", size="sm", c="dimmed"),
    ], justify="space-between", align="center", wrap="wrap", mb="xs")

    monthly_extra = dmc.SegmentedControl(id="cxl-per-prop", data=_MODE_DATA, value="agg",
                                         size="xs", radius="md")
    heatmap_extra = dmc.SegmentedControl(id="cxl-window", data=_WINDOW_DATA, value="12",
                                         size="xs", radius="md")

    # Channel + Length-of-stay stay side by side; Lead time gets its own full-width row
    # (below) so the day-by-day curve has room to breathe.
    breakdowns = dmc.SimpleGrid(
        [ui.chart_card("Channel risk", "cxl-channel", info=_INFO["channel"], height=300),
         ui.chart_card("Length of stay", "cxl-stay", info=_INFO["stay"], height=300)],
        cols={"base": 1, "md": 2}, spacing="md")

    # Lead-time card controls: window (focus first 30 days, extend to 60) + split toggle.
    lead_extra = dmc.Group([
        dmc.SegmentedControl(id="cxl-lead-range", value="30", size="xs", radius="md",
                             data=[{"label": "30 d", "value": "30"},
                                   {"label": "60 d", "value": "60"}]),
        dmc.SegmentedControl(id="cxl-lead-split", value="all", size="xs", radius="md",
                             data=[{"label": "Overall", "value": "all"},
                                   {"label": "By length of stay", "value": "stay"}]),
    ], gap="xs", wrap="nowrap")
    lead_card = ui.chart_card(
        "Lead time", "cxl-lead", info=_INFO["lead"], height=360, header_extra=lead_extra,
        subtitle="Cancellation rate day by day before arrival · toggle the split by "
                 "length of stay to see how short / mid / long stays differ.")

    # Standmixer: lead × stay cancel-rate heatmap (the two drivers blended).
    blend_card = ui.chart_card(
        "Lead time × length of stay", "cxl-blend", info=_INFO["blend"], height=300,
        subtitle="Where the two drivers combine  the darkest cell is the riskiest mix.")

    # Cancel-timing near-window heatmap: days-before × (stay or lead), rate or count.
    ct_extra = dmc.Group([
        dmc.SegmentedControl(id="cxl-ct-dim", value="stay", size="xs", radius="md",
                             data=[{"label": "By length of stay", "value": "stay"},
                                   {"label": "By lead time", "value": "lead"}]),
        dmc.SegmentedControl(id="cxl-ct-metric", value="rate", size="xs", radius="md",
                             data=[{"label": "Cancel rate", "value": "rate"},
                                   {"label": "Count", "value": "count"}]),
    ], gap="xs", wrap="nowrap")
    ct_card = ui.chart_card(
        "Cancel timing · last 14 days before arrival", "cxl-ct-grid", info=_INFO["ct_grid"],
        height=320, header_extra=ct_extra,
        subtitle="Per-day cancel rate near arrival (0 = arrival day) · rate = of the "
                 "bookings still due to arrive that day, the share that cancel. Switch "
                 "rows (stay / lead) and colour (rate / count).")

    # "When do cancellations happen" — daily count histogram (+ by-LOS split).
    cth_extra = dmc.SegmentedControl(id="cxl-cth-split", value="all", size="xs", radius="md",
                                     data=[{"label": "All", "value": "all"},
                                           {"label": "By length of stay", "value": "stay"}])
    cth_card = ui.chart_card(
        "When do cancellations happen?", "cxl-ct-hist", info=_INFO["ct_hist"], height=320,
        header_extra=cth_extra,
        subtitle="Number of cancellations by day before arrival · exact daily bins "
                 "(0–1 = arrival day) · 30+ days pooled.")

    # ---- No-show section (new) --------------------------------------------
    noshow_header = dmc.Stack([
        dmc.Divider(mt="lg", mb=2),
        dmc.Group([
            dmc.Title("No-shows", order=2),
            dmc.Badge("Resolved arrivals · didn't cancel, didn't show", color="gray",
                      variant="light", radius="sm"),
        ], gap="sm", align="center"),
        dmc.Text("How often confirmed arrivals turn into no-shows  over time, by location "
                 "and by length of stay. Based on the raw reservations history; the same "
                 "location and time-window filters apply.", size="sm", c="dimmed"),
    ], gap=4)
    noshow_stay_card = ui.chart_card(
        "No-show rate by length of stay", "cxl-ns-stay", info=_INFO["ns_stay"], height=300,
        subtitle="Short 1–2 · Mid 3–6 · Long 7+ nights · bars labelled with the no-show count.")

    drawer = dmc.Drawer(
        id="cxl-drawer", position="right", size="md", padding="lg", opened=False,
        title=dmc.Text("Detail", fw=700), withCloseButton=True,
        closeOnClickOutside=True, closeOnEscape=True,
        children=html.Div(id="cxl-drawer-body"))

    return dmc.Stack([
        header,

        # Update-history card (top)  rebuilds the cleaned dataset behind these charts.
        _history_update_card(),

        ui.sticky_filter_bar(props, "cxl-property-filter", "cxl-timewin",
                             span_label=span_label, timewin_info=_INFO["timewin"]),
        dcc.Store(id="cxl-drawer-store"),
        dcc.Store(id="cxl-upd-kick", data=0),
        dcc.Store(id="cxl-upd-seen", data={}),
        dcc.Store(id="cxl-data-version", data=0),         # bumped on update -> charts refresh
        dcc.Interval(id="cxl-upd-poll", interval=1500, n_intervals=0),

        html.Div(id="cxl-kpi", children=dmc.Skeleton(height=96, radius="lg")),
        html.Div(id="cxl-anomaly"),

        ui.chart_card("Cancellation rate over time", "cxl-monthly", info=_INFO["monthly"],
                      height=340, header_extra=monthly_extra,
                      subtitle="Monthly · click a point for that month's location breakdown."),

        ui.chart_card("Cancellation rate · location × month", "cxl-heatmap",
                      info=_INFO["heatmap"], height=430, header_extra=heatmap_extra,
                      subtitle="Click a cell to drill into that location × month."),

        breakdowns,

        lead_card,

        blend_card,

        ui.chart_card("Cancel-timing curve", "cxl-timing", info=_INFO["timing"], height=320,
                      subtitle="How late before arrival do cancellations land?"),

        cth_card,

        ct_card,

        noshow_header,

        ui.chart_card("No-show rate over time", "cxl-ns-monthly", info=_INFO["ns_monthly"],
                      height=320,
                      subtitle="Monthly · share of resolved arrivals that were no-shows."),

        ui.chart_card("No-show rate · location × month", "cxl-ns-heatmap",
                      info=_INFO["ns_heatmap"], height=None,
                      subtitle="Click-free overview · blank cells have too few arrivals."),

        noshow_stay_card,

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


def _anomaly_alert(props, window_months=None):
    hot = ch.flag_anomalies(props, window_months=window_months)
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


# ---------------------------------------------------------------------------
# Update-history job (file-backed runner): start on click, poll the job file,
# bump cxl-data-version on success so every chart re-reads the fresh clean cache.
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-upd-kick", "data"),
    Input("cxl-upd-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _start_history_update(n):
    jobs.start("history", mo.update_history_job)
    return (n or 0)


@callback(
    Output("cxl-upd-ring", "sections"),
    Output("cxl-upd-pct", "children"),
    Output("cxl-upd-msg", "children"),
    Output("cxl-upd-wrap", "style"),
    Output("cxl-upd-cancel", "children"),
    Output("cxl-upd-result", "children"),
    Output("cxl-upd-btn", "disabled"),
    Output("cxl-data-version", "data"),
    Output("cxl-upd-seen", "data"),
    Input("cxl-upd-poll", "n_intervals"),
    Input("cxl-upd-kick", "data"),
    State("cxl-data-version", "data"),
    State("cxl-upd-seen", "data"),
)
def _poll_history_update(_n, _kick, version, seen):
    st = jobs.read("history")
    status = st.get("status", "idle")
    seen = dict(seen or {})
    if status == "running":
        sec, pct, msg, wrap = ui.loader_view(float(st.get("progress", 0)) * 100,
                                             st.get("message", ""), show=True)
        return sec, pct, msg, wrap, "Cancel", no_update, True, no_update, no_update
    sec, pct, msg, wrap = ui.loader_view(0, "", show=False)
    fin = st.get("finished")
    bump = no_update
    if fin and seen.get("history") != fin:
        seen["history"] = fin
        if status == "done":
            bump = (version or 0) + 1
    if status == "error":
        return sec, pct, msg, wrap, no_update, _err_alert(st.get("error", "unknown error")), False, bump, seen
    if status == "cancelled":
        return sec, pct, msg, wrap, no_update, _cancelled_alert(), False, bump, seen
    if status == "done":
        return sec, pct, msg, wrap, no_update, _history_result(st.get("result") or {}), False, bump, seen
    return sec, pct, msg, wrap, no_update, no_update, False, no_update, seen


@callback(
    Output("cxl-upd-cancel", "children", allow_duplicate=True),
    Input("cxl-upd-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def _cancel_history_update(_n):
    jobs.cancel("history")
    return "Cancelling…"


@callback(
    Output("cxl-kpi", "children"),
    Output("cxl-anomaly", "children"),
    Output("cxl-channel", "figure"),
    Output("cxl-stay", "figure"),
    Output("cxl-timing", "figure"),
    Output("cxl-blend", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-timewin", "value"),
    Input("cxl-data-version", "data"),
)
def _update_filter(sel_value, timewin, _version):
    props = _sel(sel_value)
    win = _win(timewin)
    k = ch.kpis(props, window_months=win)
    kpis = ui.kpi_strip(_kpi_cards(k))
    anomaly = _anomaly_alert(props, win)

    dev, base = ch.channel_deviation(props, window_months=win)
    stay = ch.stay_daily_rate(props, window_months=win)      # per-night, coloured by segment
    curve, n = ch.cancel_timing_curve(props, window_months=win)
    grid = ch.leadtime_stay_grid(props, window_months=win)   # lead × stay heatmap
    overall = k["overall_rate"]

    return (kpis, anomaly,
            hc.fig_channel(dev, base),
            hc.fig_stay_daily(stay, base=overall),
            hc.fig_timing(curve, n),
            hc.fig_leadtime_stay_heatmap(grid))


# ---------------------------------------------------------------------------
# Cancel-timing near-window heatmap: filter + time-window + dim (stay/lead) +
# metric (rate/count). Own callback because of its two extra toggles.
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-ct-grid", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-ct-dim", "value"),
    Input("cxl-ct-metric", "value"),
    Input("cxl-timewin", "value"),
    Input("cxl-data-version", "data"),
)
def _update_ct_grid(sel_value, dim, metric, timewin, _version):
    props = _sel(sel_value)
    dim = dim if dim in ("stay", "lead") else "stay"
    grid = ch.cancel_timing_grid(props, dim=dim, window_months=_win(timewin))
    return hc.fig_cancel_timing_heatmap(grid, dim=dim, metric=(metric or "rate"))


@callback(
    Output("cxl-ct-hist", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-cth-split", "value"),
    Input("cxl-timewin", "value"),
    Input("cxl-data-version", "data"),
)
def _update_ct_hist(sel_value, split, timewin, _version):
    props = _sel(sel_value)
    by_stay = split == "stay"
    hist = ch.cancel_timing_histogram(props, by_stay=by_stay, window_months=_win(timewin))
    return hc.fig_cancel_timing_hist(hist, by_stay=by_stay)


# ---------------------------------------------------------------------------
# No-show section: monthly rate + location×month heatmap + by length of stay.
# Same location + time-window filters as the cancellation charts.
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-ns-monthly", "figure"),
    Output("cxl-ns-heatmap", "figure"),
    Output("cxl-ns-stay", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-timewin", "value"),
    Input("cxl-data-version", "data"),
)
def _update_noshow(sel_value, timewin, _version):
    props = _sel(sel_value)
    win = _win(timewin)
    base = ch.noshow_overall_rate(props, window_months=win)
    monthly = ch.noshow_monthly_rate(props, window_months=win)
    matrix = ch.noshow_property_month_matrix(props, months_back=12, window_months=win)
    stay = ch.noshow_stay_rate(props, window_months=win)
    return (hc.fig_noshow_monthly(monthly, base),
            hc.fig_noshow_heatmap(matrix),
            hc.fig_noshow_stay(stay, base=base))


# ---------------------------------------------------------------------------
# Lead-time curve (own row): filter + window (30/60 d) + optional stay split
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-lead", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-lead-split", "value"),
    Input("cxl-lead-range", "value"),
    Input("cxl-timewin", "value"),
    Input("cxl-data-version", "data"),
)
def _update_leadtime(sel_value, split, rng, timewin, _version):
    props = _sel(sel_value)
    win = _win(timewin)
    by_stay = split == "stay"
    daily = ch.leadtime_daily_rate(props, by_stay=by_stay, max_day=int(rng or "30"),
                                   window_months=win)
    return hc.fig_leadtime_daily(daily, by_stay=by_stay,
                                 base=ch.selection_rate(props, window_months=win))


# ---------------------------------------------------------------------------
# Monthly line (filter + aggregate/per-location mode)
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-monthly", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-per-prop", "value"),
    Input("cxl-timewin", "value"),
    Input("cxl-data-version", "data"),
)
def _update_monthly(sel_value, mode, timewin, _version):
    props = _sel(sel_value)
    win = _win(timewin)
    monthly = ch.monthly_rate(props, window_months=win)
    per = ch.monthly_rate(props, per_property=True, window_months=win) if mode == "per" else None
    return hc.fig_monthly(monthly, ch.base_rate(), per)


# ---------------------------------------------------------------------------
# Heatmap (filter + 6/12-month window)
# ---------------------------------------------------------------------------
@callback(
    Output("cxl-heatmap", "figure"),
    Input("cxl-property-filter", "value"),
    Input("cxl-window", "value"),
    Input("cxl-timewin", "value"),
    Input("cxl-data-version", "data"),
)
def _update_heatmap(sel_value, window, timewin, _version):
    props = _sel(sel_value)
    months = int(window or "12")
    return hc.fig_heatmap(ch.property_month_matrix(props, months_back=months,
                                                   window_months=_win(timewin)))


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
