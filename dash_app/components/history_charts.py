# dash_app/components/history_charts.py
# Plotly figure builders for the Cancellation History page. Each takes an already-
# aggregated frame from dash_app.backend.cancellation_history and returns a brand-styled
# go.Figure (theme.brand_figure applies the STAYERY font/colours). Kept separate from the
# page so the page module holds only layout + callbacks. Plotly-first throughout.

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash_app import theme

# Stay-segment colour + label vocabulary, shared by the per-night LOS bars and the
# length-of-stay split on the lead-time curve. Traffic-light reading: short = low/green,
# mid = amber, long = high/red (matches the "higher LOS cancels more" signal in the data).
_SEG_COLOR = {"short": theme.GREEN, "mid": theme.ORANGE, "long": theme.RED}
_SEG_NAME = {"short": "Short · 1–2 nights", "mid": "Mid · 3–6 nights",
             "long": "Long · 7+ nights"}
_SEG_ORDER = ["short", "mid", "long"]


def _empty(msg: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#9AA0A6", size=13))
    fig.update_layout(height=height, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return theme.brand_figure(fig)


# ---- 1) Monthly cancellation rate ------------------------------------------
def fig_monthly(monthly: pd.DataFrame, base_rate: float,
                per_property: pd.DataFrame | None = None,
                height: int = 340) -> go.Figure:
    """Bold aggregate line for the current selection; optional faint per-location lines.
    A dashed line marks the overall base rate for reference."""
    if monthly is None or monthly.empty:
        return _empty("No bookings in the current selection", height)
    fig = go.Figure()

    show_per = per_property is not None and not per_property.empty
    if show_per:
        for i, (name, grp) in enumerate(per_property.groupby("property_name")):
            grp = grp.dropna(subset=["cancel_rate"]).sort_values("month")
            if grp.empty:
                continue
            fig.add_trace(go.Scatter(
                x=grp["month"], y=grp["cancel_rate"] * 100, mode="lines",
                line=dict(width=1, color=theme.CATEGORICAL[i % len(theme.CATEGORICAL)]),
                opacity=0.5, name=str(name),
                hovertemplate="%{fullData.name}<br>%{x|%b %Y}: %{y:.1f}%<extra></extra>"))

    agg = monthly.dropna(subset=["cancel_rate"]).sort_values("month")
    fig.add_trace(go.Scatter(
        x=agg["month"], y=agg["cancel_rate"] * 100, mode="lines+markers",
        line=dict(width=3, color=theme.BLUE), marker=dict(size=5, color=theme.BLUE),
        name="Selection (all)", customdata=agg["n"],
        hovertemplate="<b>%{x|%b %Y}</b><br>Cancel rate: %{y:.1f}%"
                      "<br>Bookings: %{customdata:,}<extra></extra>"))

    if base_rate is not None and np.isfinite(base_rate):
        fig.add_hline(y=base_rate * 100, line=dict(color="#9AA0A6", width=1, dash="dash"),
                      annotation_text=f"overall base {base_rate * 100:.1f}%",
                      annotation_position="top left", annotation_font_size=11)

    fig.update_layout(
        height=height, xaxis_title=None, yaxis_title="Cancel rate",
        yaxis=dict(ticksuffix="%"), showlegend=show_per,
        hovermode="closest" if show_per else "x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
    return theme.brand_figure(fig)


# ---- 2) Property × month heatmap -------------------------------------------
def fig_heatmap(matrix: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Rows = locations, cols = months, colour = cancel rate. Thin cells (masked to NaN
    upstream) render as blanks. x-values are ISO 'YYYY-MM' so a click maps cleanly back
    to a month for the drill-down drawer."""
    if matrix is None or matrix.empty:
        return _empty("No data for the selected locations / window", height or 320)
    m = matrix.copy()
    months = sorted(m["month"].unique())
    month_iso = [pd.to_datetime(x).strftime("%Y-%m") for x in months]
    props = sorted(m["property_name"].unique())

    piv_r = (m.pivot(index="property_name", columns="month", values="cancel_rate")
             .reindex(index=props, columns=months))
    piv_n = (m.pivot(index="property_name", columns="month", values="n")
             .reindex(index=props, columns=months))
    z = piv_r.to_numpy(dtype="float64") * 100
    n = piv_n.to_numpy(dtype="float64")

    fig = go.Figure(go.Heatmap(
        z=z, x=month_iso, y=props, customdata=n,
        colorscale=theme.CANCEL_SCALE, zmin=5, zmax=40,
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(title="Cancel %", thickness=12),
        hovertemplate="<b>%{y}</b> · %{x}<br>Cancel rate: %{z:.1f}%"
                      "<br>Bookings: %{customdata:,.0f}<extra></extra>"))
    h = height or max(320, 90 + 30 * len(props))
    fig.update_layout(height=h, xaxis_title=None, yaxis_title=None,
                      yaxis_autorange="reversed", xaxis_side="top",
                      margin=dict(l=140, r=20, t=50, b=20))
    return theme.brand_figure(fig)


# ---- 3) Channel: deviation from base ---------------------------------------
def fig_channel(dev: pd.DataFrame, base: float, height: int = 300) -> go.Figure:
    """Horizontal diverging bars: how far each channel's cancel rate sits above (red) or
    below (green) the selection's base rate. Long-tail channels (< min sample) excluded
    upstream, not fabricated."""
    if dev is None or dev.empty:
        return _empty("No channel meets the sample threshold", height)
    d = dev.copy()
    colors = [theme.RED if v > 0 else theme.GREEN for v in d["deviation"]]
    fig = go.Figure(go.Bar(
        x=d["deviation"] * 100, y=d["channel"], orientation="h", marker_color=colors,
        customdata=np.stack([d["cancel_rate"] * 100, d["n"]], axis=-1),
        hovertemplate="<b>%{y}</b><br>Cancel rate: %{customdata[0]:.1f}%"
                      "<br>vs base: %{x:+.1f} pp<br>Bookings: %{customdata[1]:,}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    fig.update_layout(height=height, xaxis_title="Deviation from base (pp)",
                      yaxis_title=None, bargap=0.25, margin=dict(l=10, r=20, t=20, b=40))
    return theme.brand_figure(fig)


# ---- 4) Generic rate bars (stay segment, lead-time bucket) -----------------
def fig_rate_bars(df: pd.DataFrame, cat_col: str, *, base: float | None = None,
                  color: str | None = None, height: int = 300) -> go.Figure:
    if df is None or df.empty:
        return _empty("No data", height)
    d = df.copy()
    fig = go.Figure(go.Bar(
        x=d[cat_col].astype(str), y=d["cancel_rate"] * 100,
        marker_color=color or theme.BLUE, customdata=d["n"],
        hovertemplate="<b>%{x}</b><br>Cancel rate: %{y:.1f}%"
                      "<br>Bookings: %{customdata:,}<extra></extra>"))
    if base is not None and np.isfinite(base):
        fig.add_hline(y=base * 100, line=dict(color="#9AA0A6", width=1, dash="dash"),
                      annotation_text=f"base {base * 100:.1f}%",
                      annotation_position="top right", annotation_font_size=10)
    fig.update_layout(height=height, xaxis_title=None, yaxis_title="Cancel rate",
                      yaxis=dict(ticksuffix="%"), margin=dict(l=45, r=15, t=25, b=35))
    return theme.brand_figure(fig)


# ---- 4b) Length of stay, per NIGHT, coloured by segment --------------------
def fig_stay_daily(df: pd.DataFrame, base: float | None = None,
                   height: int = 300) -> go.Figure:
    """One bar per exact stay length in nights (…, 13, 14+), coloured short/mid/long so
    the segment boundaries stay readable while the day-by-day shape shows through. Dashed
    line marks the current selection's overall rate."""
    if df is None or df.empty:
        return _empty("No data", height)
    d = df.sort_values("night")
    labels = d["label"].tolist()
    fig = go.Figure()
    for seg in _SEG_ORDER:
        s = d[d["segment"] == seg]
        if s.empty:
            continue
        fig.add_trace(go.Bar(
            x=s["label"], y=s["cancel_rate"] * 100, name=_SEG_NAME[seg],
            marker_color=_SEG_COLOR[seg], customdata=s["n"],
            hovertemplate="<b>%{x} nights</b><br>Cancel rate: %{y:.1f}%"
                          "<br>Bookings: %{customdata:,}<extra></extra>"))
    if base is not None and np.isfinite(base):
        fig.add_hline(y=base * 100, line=dict(color="#9AA0A6", width=1, dash="dash"),
                      annotation_text=f"base {base * 100:.1f}%",
                      annotation_position="top right", annotation_font_size=10)
    fig.update_layout(
        height=height, barmode="group", xaxis_title="Nights",
        yaxis_title="Cancel rate", yaxis=dict(ticksuffix="%"),
        xaxis=dict(categoryorder="array", categoryarray=labels),
        margin=dict(l=45, r=15, t=25, b=38),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)))
    return theme.brand_figure(fig)


# ---- 4c) Lead time, per DAY, optional length-of-stay split -----------------
def fig_leadtime_daily(df: pd.DataFrame, by_stay: bool = False,
                       base: float | None = None, height: int = 340) -> go.Figure:
    """Cancel rate day-by-day against lead time (days between booking and arrival). A
    single blue line by default; when by_stay is set, three coloured lines (short/mid/
    long) reveal how stay length shifts the whole curve. Dashed line = selection base."""
    if df is None or df.empty:
        return _empty("No bookings in the current selection", height)
    d = df.copy()
    fig = go.Figure()
    if by_stay and "stay_bucket" in d.columns:
        for seg in _SEG_ORDER:
            s = d[d["stay_bucket"] == seg].dropna(subset=["cancel_rate"]).sort_values("lead_day")
            if s.empty:
                continue
            fig.add_trace(go.Scatter(
                x=s["lead_day"], y=s["cancel_rate"] * 100, mode="lines",
                name=_SEG_NAME[seg], line=dict(width=2.5, color=_SEG_COLOR[seg]),
                customdata=s["n"],
                hovertemplate=_SEG_NAME[seg] + "<br>Lead %{x} d: %{y:.1f}%"
                              "<br>Bookings: %{customdata:,}<extra></extra>"))
    else:
        s = d.dropna(subset=["cancel_rate"]).sort_values("lead_day")
        fig.add_trace(go.Scatter(
            x=s["lead_day"], y=s["cancel_rate"] * 100, mode="lines+markers",
            line=dict(width=3, color=theme.BLUE), marker=dict(size=4, color=theme.BLUE),
            name="All stays", customdata=s["n"],
            hovertemplate="<b>Lead %{x} days</b><br>Cancel rate: %{y:.1f}%"
                          "<br>Bookings: %{customdata:,}<extra></extra>"))
    if base is not None and np.isfinite(base):
        fig.add_hline(y=base * 100, line=dict(color="#9AA0A6", width=1, dash="dash"),
                      annotation_text=f"base {base * 100:.1f}%",
                      annotation_position="bottom right", annotation_font_size=10)
    fig.update_layout(
        height=height, xaxis_title="Lead time (days before arrival)",
        yaxis_title="Cancel rate", yaxis=dict(ticksuffix="%"),
        xaxis=dict(rangemode="tozero"), showlegend=by_stay, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        margin=dict(l=50, r=20, t=30, b=40))
    return theme.brand_figure(fig)


# ---- 5) Cancel-timing curve (the "storno-curve") ---------------------------
def fig_timing(curve: pd.DataFrame, n: int, height: int = 320) -> go.Figure:
    """Cumulative share of cancellations occurring within X days before arrival. Answers
    'how late can rooms still free up?'. A marker shows where 50% is reached."""
    if curve is None or curve.empty:
        return _empty("No cancellations in the current selection", height)
    c = curve.copy()
    y = c["cum_share_within"] * 100
    fig = go.Figure(go.Scatter(
        x=c["days_before"], y=y, mode="lines", fill="tozeroy",
        line=dict(color=theme.PURPLE, width=2.5), fillcolor="rgba(110,50,200,0.12)",
        hovertemplate="Within %{x} days before arrival:<br>%{y:.1f}% of cancellations<extra></extra>"))

    hit = c.loc[c["cum_share_within"] >= 0.5, "days_before"]
    if not hit.empty:
        med = int(hit.min())
        fig.add_vline(x=med, line=dict(color=theme.ORANGE, width=1.5, dash="dash"),
                      annotation_text=f"50% within {med} days",
                      annotation_position="top right", annotation_font_size=11)

    fig.update_layout(
        height=height, xaxis_title="Days before arrival (0 = arrival day)",
        yaxis_title="Cumulative % of cancellations",
        yaxis=dict(ticksuffix="%", range=[0, 100]),
        title=dict(text=f"Based on {n:,} cancellations", font=dict(size=11, color="#9AA0A6"),
                   x=0, xanchor="left", y=0.98))
    return theme.brand_figure(fig)


# ---- 6) Lead × stay "blend" heatmap (Standmixer) ---------------------------
_STAY_ROWS = ["long", "mid", "short"]          # long on top (worst first)
_STAY_ROW_LABEL = {"short": "Short · 1–2 N", "mid": "Mid · 3–6 N", "long": "Long · 7+ N"}
_LEAD_COLS = ["0–7 d", "8–30 d", "31–90 d", "90 d+"]


def fig_leadtime_stay_heatmap(grid: pd.DataFrame, height: int = 300) -> go.Figure:
    """Cancel rate for every lead-bucket × stay-bucket cell  the two drivers blended into
    one view. Green→red traffic-light scale; the % is printed in each cell."""
    if grid is None or grid.empty:
        return _empty("No data for the current selection", height)
    g = grid.copy()
    piv_r = (g.pivot(index="stay_bucket", columns="lead_bucket", values="cancel_rate")
             .reindex(index=_STAY_ROWS, columns=_LEAD_COLS))
    piv_n = (g.pivot(index="stay_bucket", columns="lead_bucket", values="n")
             .reindex(index=_STAY_ROWS, columns=_LEAD_COLS))
    z = piv_r.to_numpy(dtype="float64") * 100
    n = piv_n.to_numpy(dtype="float64")
    text = [["" if np.isnan(v) else f"{v:.0f}%" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(
        z=z, x=_LEAD_COLS, y=[_STAY_ROW_LABEL[s] for s in _STAY_ROWS], customdata=n,
        colorscale=theme.CANCEL_SCALE, zmin=5, zmax=55, xgap=3, ygap=3, hoverongaps=False,
        text=text, texttemplate="%{text}", textfont=dict(size=13, color="#20140E"),
        colorbar=dict(title="Cancel %", thickness=12),
        hovertemplate="<b>%{y}</b> · lead %{x}<br>Cancel rate: %{z:.1f}%"
                      "<br>Bookings: %{customdata:,.0f}<extra></extra>"))
    fig.update_layout(height=height, xaxis_title="Lead time (booking → arrival)",
                      yaxis_title=None, xaxis_side="top",
                      margin=dict(l=110, r=20, t=50, b=20))
    return theme.brand_figure(fig)


# ---- 7) Cancel-timing near-window hazard heatmap (days-before × stay/lead) --
def fig_cancel_timing_heatmap(g: pd.DataFrame, dim: str = "stay", metric: str = "rate",
                              max_day: int = 14, height: int = 320) -> go.Figure:
    """Grid of near-arrival cancellation timing. Columns = whole days before arrival
    (0 = arrival day … max_day). Rows = stay segments or lead buckets. `metric='rate'`
    colours by the per-day cancel HAZARD (cancellations that day ÷ bookings still due to
    arrive that day), `metric='count'` by the raw number of cancellations. Count AND the
    at-risk base are always in the hover. Cell numbers turn white above 4,000
    cancellations so they stay readable on the darkest cells."""
    if g is None or g.empty:
        return _empty("No cancellations in the current selection", height)
    day_labels = [str(i) for i in range(max_day + 1)]
    if dim == "lead":
        rows = ["0–7 d", "8–30 d", "31–90 d", "90 d+"]
    else:
        rows = ["Short · 1–2 nights", "Mid · 3–6 nights", "Long · 7+ nights"]
    rows = [r for r in rows if r in set(g["row"])]
    piv_rate = g.pivot(index="row", columns="day", values="rate").reindex(index=rows, columns=day_labels)
    piv_nc = g.pivot(index="row", columns="day", values="n_cancel").reindex(index=rows, columns=day_labels)
    piv_na = g.pivot(index="row", columns="day", values="n_atrisk").reindex(index=rows, columns=day_labels)
    rate = piv_rate.to_numpy(dtype="float64") * 100
    nc = piv_nc.to_numpy(dtype="float64")
    na = piv_na.to_numpy(dtype="float64")

    if metric == "count":
        z, colorscale, cbar = nc, [[0.0, "#F3F1EA"], [1.0, theme.PURPLE]], "Cancellations"
    else:
        z, colorscale, cbar = rate, theme.CANCEL_SCALE, "Cancel %/day"
    custom = np.dstack([np.nan_to_num(nc), np.nan_to_num(na), np.nan_to_num(rate)])
    fig = go.Figure(go.Heatmap(
        z=z, x=day_labels, y=rows, customdata=custom, colorscale=colorscale,
        xgap=2, ygap=2, hoverongaps=False, colorbar=dict(title=cbar, thickness=12),
        hovertemplate="<b>%{y}</b> · %{x} days before arrival"
                      "<br>Cancellations: %{customdata[0]:,.0f} of %{customdata[1]:,.0f} still due"
                      "<br>Cancel rate that day: %{customdata[2]:.2f}%<extra></extra>"))
    # Per-cell labels via annotations so the number can turn white above 4,000 cancels.
    for i, rname in enumerate(rows):
        for j, dlab in enumerate(day_labels):
            v, cnt = z[i][j], nc[i][j]
            if np.isnan(v):
                continue
            if metric == "count":
                if cnt <= 0:
                    continue
                txt = f"{int(cnt):,}"
            else:
                txt = f"{v:.1f}%"
            color = "#FFFFFF" if (not np.isnan(cnt) and cnt > 4000) else "#20140E"
            fig.add_annotation(x=dlab, y=rname, text=txt, showarrow=False,
                               font=dict(size=10, color=color))
    fig.update_layout(height=height, xaxis_title="Days before arrival (0 = arrival day)",
                      yaxis_title=None, xaxis_side="top", yaxis_autorange="reversed",
                      margin=dict(l=150, r=20, t=50, b=30))
    return theme.brand_figure(fig)


# ---- 8) "When do cancellations happen" — daily count histogram -------------
def fig_cancel_timing_hist(df: pd.DataFrame, by_stay: bool = False,
                           height: int = 320) -> go.Figure:
    """Number of cancellations by whole day before arrival (exact daily bins, 0–1 = the
    arrival day). Counts, not shares. When by_stay is set, grouped bars split the count by
    length of stay (short/mid/long)."""
    if df is None or df.empty:
        return _empty("No cancellations in the current selection", height)
    d = df.copy().sort_values("day_order")
    order = d.drop_duplicates("day_order").sort_values("day_order")["day"].tolist()
    fig = go.Figure()
    if by_stay and "stay_bucket" in d.columns:
        for seg in _SEG_ORDER:
            s = d[d["stay_bucket"] == seg]
            if s.empty:
                continue
            fig.add_trace(go.Bar(
                x=s["day"], y=s["n_cancel"], name=_SEG_NAME[seg], marker_color=_SEG_COLOR[seg],
                hovertemplate=_SEG_NAME[seg] + "<br>%{x} days before arrival"
                              "<br>%{y:,} cancellations<extra></extra>"))
        fig.update_layout(barmode="group")
    else:
        fig.add_trace(go.Bar(
            x=d["day"], y=d["n_cancel"], marker_color=theme.BLUE,
            hovertemplate="%{x} days before arrival<br>%{y:,} cancellations<extra></extra>"))
    fig.update_layout(
        height=height, xaxis_title="Days before arrival (0–1 = arrival day)",
        yaxis_title="Cancellations",
        xaxis=dict(categoryorder="array", categoryarray=order),
        showlegend=by_stay,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        margin=dict(l=55, r=20, t=30, b=45))
    return theme.brand_figure(fig)


# ---- 9) No-shows: monthly rate, location×month heatmap, by length of stay ---
def fig_noshow_monthly(monthly: pd.DataFrame, base_rate: float | None = None,
                       height: int = 320) -> go.Figure:
    """No-show rate per arrival month (orange), with the overall rate as a dashed line.
    Hover carries the absolute no-show count and the arrivals base."""
    if monthly is None or monthly.empty:
        return _empty("No resolved arrivals in the current selection", height)
    d = monthly.dropna(subset=["rate"]).sort_values("month")
    if d.empty:
        return _empty("Too few arrivals per month to show a stable rate", height)
    fig = go.Figure(go.Scatter(
        x=d["month"], y=d["rate"] * 100, mode="lines+markers",
        line=dict(width=3, color=theme.ORANGE), marker=dict(size=5, color=theme.ORANGE),
        customdata=np.stack([d["n_noshow"], d["n"]], axis=-1), name="No-show rate",
        hovertemplate="<b>%{x|%b %Y}</b><br>No-show rate: %{y:.2f}%"
                      "<br>No-shows: %{customdata[0]:,} of %{customdata[1]:,} arrivals<extra></extra>"))
    if base_rate is not None and np.isfinite(base_rate):
        fig.add_hline(y=base_rate * 100, line=dict(color="#9AA0A6", width=1, dash="dash"),
                      annotation_text=f"overall {base_rate * 100:.2f}%",
                      annotation_position="top left", annotation_font_size=11)
    fig.update_layout(height=height, xaxis_title=None, yaxis_title="No-show rate",
                      yaxis=dict(ticksuffix="%"), hovermode="x unified",
                      margin=dict(l=50, r=20, t=25, b=30))
    return theme.brand_figure(fig)


def fig_noshow_heatmap(matrix: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Rows = locations, cols = months, colour = no-show rate (orange ramp). Hover carries
    the no-show count and the arrivals base. Thin cells (masked upstream) render blank."""
    if matrix is None or matrix.empty:
        return _empty("No data for the selected locations / window", height or 320)
    m = matrix.copy()
    months = sorted(m["month"].unique())
    month_iso = [pd.to_datetime(x).strftime("%Y-%m") for x in months]
    props = sorted(m["property_name"].unique())
    piv_r = m.pivot(index="property_name", columns="month", values="rate").reindex(index=props, columns=months)
    piv_ns = m.pivot(index="property_name", columns="month", values="n_noshow").reindex(index=props, columns=months)
    piv_n = m.pivot(index="property_name", columns="month", values="n").reindex(index=props, columns=months)
    z = piv_r.to_numpy(dtype="float64") * 100
    custom = np.dstack([np.nan_to_num(piv_ns.to_numpy(dtype="float64")),
                        np.nan_to_num(piv_n.to_numpy(dtype="float64"))])
    fig = go.Figure(go.Heatmap(
        z=z, x=month_iso, y=props, customdata=custom,
        colorscale=[[0.0, "#F3F1EA"], [1.0, theme.ORANGE]], zmin=0, zmax=5,
        xgap=2, ygap=2, hoverongaps=False, colorbar=dict(title="No-show %", thickness=12),
        hovertemplate="<b>%{y}</b> · %{x}<br>No-show rate: %{z:.2f}%"
                      "<br>No-shows: %{customdata[0]:,.0f} of %{customdata[1]:,.0f} arrivals<extra></extra>"))
    h = height or max(320, 90 + 30 * len(props))
    fig.update_layout(height=h, xaxis_title=None, yaxis_title=None,
                      yaxis_autorange="reversed", xaxis_side="top",
                      margin=dict(l=140, r=20, t=50, b=20))
    return theme.brand_figure(fig)


def fig_noshow_stay(df: pd.DataFrame, base: float | None = None,
                    height: int = 300) -> go.Figure:
    """No-show rate by length-of-stay segment. Each bar is labelled with the absolute
    number of no-shows so the rate is never read without its base."""
    if df is None or df.empty:
        return _empty("No data", height)
    d = df.copy()
    fig = go.Figure(go.Bar(
        x=d["label"].astype(str), y=d["rate"] * 100, marker_color=theme.ORANGE,
        customdata=np.stack([d["n_noshow"], d["n"]], axis=-1),
        text=[f"{int(x):,} no-shows" for x in d["n_noshow"]], textposition="outside",
        hovertemplate="<b>%{x}</b><br>No-show rate: %{y:.2f}%"
                      "<br>No-shows: %{customdata[0]:,} of %{customdata[1]:,} arrivals<extra></extra>"))
    if base is not None and np.isfinite(base):
        fig.add_hline(y=base * 100, line=dict(color="#9AA0A6", width=1, dash="dash"),
                      annotation_text=f"overall {base * 100:.2f}%",
                      annotation_position="top right", annotation_font_size=10)
    fig.update_layout(height=height, xaxis_title=None, yaxis_title="No-show rate",
                      yaxis=dict(ticksuffix="%"), margin=dict(l=45, r=15, t=25, b=35))
    return theme.brand_figure(fig)


