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
