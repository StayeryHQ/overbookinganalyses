# dash_app/components/panels.py
# Pure builder functions for the Occupancy page: KPI tiles, the ag-grid column defs
# + row data, the booking side panel, the cost-parameter panel, and the occupancy
# heatmap. Kept separate from the page so occupancy.py holds only layout + callbacks.

from __future__ import annotations

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import dcc

from dash_app import theme
from dash_app.components import ui


# ---------------------------------------------------------------------------
# KPI tiles
# ---------------------------------------------------------------------------
def kpi_tiles(
    freshness: dict,
    model_meta: dict,
    high_risk_count: int | None,
    risk_threshold: float,
) -> list:
    """Four KPI tiles. Values that aren't backed by real metadata are shown as
    'unavailable' rather than fabricated."""
    data_ts = freshness.get("reservations") or "never - run `main.py refresh`"

    m_ts = model_meta.get("retrained_at") or "unavailable"
    m_sub = (
        f"model: {model_meta['model']}"
        if model_meta.get("model")
        else "no model on disk"
    )

    n_train = model_meta.get("trained_on_bookings")
    train_val = f"~{n_train:,}" if n_train is not None else "unavailable"
    train_sub = model_meta.get("trained_on_note") or "no training metadata"

    hr_val = "-" if high_risk_count is None else f"{high_risk_count:,}"
    hr_sub = f"≥ {risk_threshold:.0%} cancel risk · next 14 days · selected"

    return ui.kpi_strip(
        [
            ui.kpi_card(
                "Data last updated",
                data_ts,
                tooltip="When the local reservations cache was last refreshed.",
            ),
            ui.kpi_card(
                "Model last retrained",
                m_ts,
                sub=m_sub,
                tooltip="Rebuild everything locally: BigQuery history "
                "→ retrain all models (with HP search) → comparison → evaluation → "
                "SHAP/PDP → score: `uv run python main.py refresh-all`"
                "needs to be run offline by an Admin (Ayo/Luca)",
            ),
            ui.kpi_card("Training set size", train_val, sub=train_sub),
            ui.kpi_card(
                "High-risk bookings",
                hr_val,
                sub=hr_sub,
                accent=True,
                tooltip="Upcoming bookings scored at or above the high-risk "
                "threshold, for the selected properties over the next 14 days.",
            ),
        ]
    )


# ---------------------------------------------------------------------------
# Booking risk table (dash-ag-grid)
# ---------------------------------------------------------------------------
# cancel_proba is kept numeric (0..1) for correct sorting; a valueFormatter renders
# it as a percentage. Risk bucket is colour-coded via styleConditions.
def booking_column_defs() -> list[dict]:
    return [
        {"headerName": "Booking", "field": "id", "pinned": "left", "width": 130},
        {"headerName": "Property", "field": "property_name", "width": 170},
        {"headerName": "Arrival", "field": "arrival", "width": 120, "sort": "asc"},
        {
            "headerName": "LoS",
            "field": "los_nights",
            "width": 80,
            "type": "numericColumn",
        },
        {"headerName": "Channel", "field": "channelCode", "width": 130},
        {
            "headerName": "Cancel risk",
            "field": "cancel_proba",
            "width": 120,
            "type": "numericColumn",
            "valueFormatter": {
                "function": "(params.value == null ? '' : (params.value*100).toFixed(0) + '%')"
            },
        },
        {"headerName": "Status", "field": "status", "width": 110},
    ]


_TABLE_FIELDS = [
    "id",
    "property_name",
    "arrival",
    "los_nights",
    "channelCode",
    "cancel_proba",
    "status",
]


def booking_row_data(df_window: pd.DataFrame) -> list[dict]:
    """ag-grid rowData from the scored window frame. Formats arrival as a date and keeps
    only the display fields (the full record still feeds the side panel)."""
    if df_window.empty:
        return []
    d = df_window.copy()
    d["arrival"] = pd.to_datetime(d["arrival"], utc=True).dt.strftime("%Y-%m-%d")
    for f in _TABLE_FIELDS:
        if f not in d.columns:
            d[f] = None
    return d[_TABLE_FIELDS].to_dict("records")


# ---------------------------------------------------------------------------
# Booking detail side panel (raw record - NO SHAP/XAI, that's Phase 4)
# ---------------------------------------------------------------------------
# Fields shown in the side panel, in order (only those present are rendered).
_DETAIL_FIELDS = [
    ("id", "Booking reference"),
    ("bookingId", "Booking ID"),
    ("property_name", "Property"),
    ("status", "Status"),
    ("arrival", "Arrival"),
    ("departure", "Departure"),
    ("created", "Booked on"),
    ("los_nights", "Length of stay (nights)"),
    ("channelCode", "Channel"),
    ("ratePlan_name", "Rate plan"),
    ("unitGroup_name", "Room type"),
    ("adults", "Adults"),
    ("totalGrossAmount_amount", "Gross amount"),
    ("cancellationFee_fee_amount", "Cancellation fee"),
    ("guaranteeType", "Guarantee"),
    ("cancel_proba", "Predicted cancel risk"),
    ("model_used", "Scored by model"),
]


def _fmt(field: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "-"
    if field == "cancel_proba":
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return str(value)
    if field in ("arrival", "departure", "created"):
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        return "-" if pd.isna(ts) else ts.strftime("%Y-%m-%d")
    return str(value)


def side_panel_content(record: dict | None) -> list:
    """Definition-list of the raw booking record for the selected row (dmc)."""
    if not record:
        return [
            dmc.Text(
                "Select a booking in the table to see its full record.",
                c="dimmed",
                size="sm",
            )
        ]
    rows = []
    for field, label in _DETAIL_FIELDS:
        if field in record:
            rows.append(
                dmc.Group(
                    [
                        dmc.Text(label, size="xs", c="dimmed"),
                        dmc.Text(
                            _fmt(field, record.get(field)),
                            size="sm",
                            fw=500,
                            ta="right",
                        ),
                    ],
                    justify="space-between",
                    wrap="nowrap",
                    gap="md",
                )
            )
    return [dmc.Stack(rows, gap=6)]


# ---------------------------------------------------------------------------
# Global overbooking cost controls (single entry point) + derived threshold.
# Placed at the TOP of the Occupancy & Predictions page; the same values are
# shared app-wide via the global cost-store (Model Performance reads/writes them
# too). Values are set/persisted via callbacks.
# ---------------------------------------------------------------------------
_MULT_HELP = (
    "High-demand period: walking a guest hurts more, so the walk cost is multiplied "
    "by this factor. That raises the EFFECTIVE walk cost and therefore the cost-optimal "
    "threshold  the model flags fewer bookings, only the clearest cancellations."
)


def cost_controls() -> dmc.Paper:
    """One global set of overbooking costs + the derived cost-optimal threshold.

    Negative values are allowed on both cost inputs (a walk can be net-positive if
    resold at a premium; an empty-room cost can be negative in edge cases). The
    threshold badge is filled by a callback and drives the heatmap count, the KPI and
    the table's Low/Medium boundary. High risk stays fixed at ≥ 85%.
    """
    return dmc.Paper(
        dmc.Stack(
            [
                dmc.Group(
                    [
                        dmc.Text("Overbooking costs", fw=600, size="sm"),
                        dmc.Text(
                            "One global setting  shared with Model Performance.",
                            size="xs",
                            c="dimmed",
                        ),
                    ],
                    gap="sm",
                    align="center",
                ),
                dmc.Group(
                    [
                        # No min on the cost inputs: negative values are allowed.
                        dmc.NumberInput(
                            id="cost-walk",
                            label="Cost of walking a guest (€)",
                            step=1,
                            placeholder="set your own",
                            style={"width": "185px"},
                        ),
                        dmc.NumberInput(
                            id="cost-empty",
                            label="Cost of an empty room (€)",
                            step=1,
                            placeholder="pre-filled from room revenue",
                            style={"width": "205px"},
                        ),
                        dmc.Switch(
                            id="cost-high-demand",
                            label="High-demand period",
                            checked=False,
                        ),
                        dmc.Group(
                            [
                                dmc.NumberInput(
                                    id="cost-multiplier",
                                    label="Walk-cost multiplier",
                                    min=1,
                                    step=0.1,
                                    value=1.5,
                                    style={"width": "150px"},
                                ),
                                ui.info_icon(_MULT_HELP),
                            ],
                            gap=4,
                            align="flex-end",
                            wrap="nowrap",
                        ),
                        dmc.Stack(
                            [
                                dmc.Group(
                                    [
                                        dmc.Text(
                                            "Cost-optimal threshold",
                                            size="xs",
                                            c="dimmed",
                                            fw=600,
                                        ),
                                        ui.info_icon(
                                            "Derived from the costs above (with the high-demand "
                                            "multiplier applied). Flags a booking as a likely "
                                            "cancellation at/above this probability; also the "
                                            "Low/Medium boundary in the table below."
                                        ),
                                    ],
                                    gap=4,
                                    align="center",
                                    wrap="nowrap",
                                ),
                                dmc.Text(id="occ-thr-badge", size="lg", fw=700),
                            ],
                            gap=0,
                        ),
                    ],
                    align="flex-end",
                    gap="md",
                    wrap="wrap",
                ),
                dmc.Text(id="cost-empty-help", c="dimmed", size="xs"),
            ],
            gap="xs",
        ),
        p="md",
        radius="lg",
        withBorder=True,
    )


# ---------------------------------------------------------------------------
# Heatmap: property (rows) × 14 days (cols). Colour = occupancy %; the other
# metrics (occupied, arrivals, departures, expected + over-threshold
# cancellations) live in the hover  too many numbers to print in each tile.
# ---------------------------------------------------------------------------
_OCC_COLORSCALE = [[0.0, "#FFFFFF"], [0.5, theme.YELLOW], [1.0, theme.ORANGE]]


def heatmap_figure(
    grid: pd.DataFrame,
    *,
    metric: str = "occupancy",
    pct_on: bool = True,
    pct_thr: float | None = 90,
    free_on: bool = True,
    free_thr: float | None = 5,
    show_hover: bool = True,
) -> go.Figure:
    """Occupancy heatmap (property rows × next 14 days). Colour is ALWAYS occupancy %;
    the in-tile number is chosen by `metric`:
        'occupancy' -> occupancy %                    (tab 1, default)
        'expected'  -> expected cancellations Σp      (tab 2)
        'threshold' -> bookings above the cost threshold (tab 3)
    An overbooking-relevance filter greys out tiles that are not tight: a tile stays
    visible only if occupancy ≥ pct_thr (when pct_on) OR free rooms ≤ free_thr (when
    free_on; negative = overbooked, always shown). Greyed tiles carry no colour, no
    number and no hover."""
    if grid.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No data for the selected properties / window", height=320
        )
        return theme.brand_figure(fig)
    props = list(dict.fromkeys(grid["property_name"]))
    days = sorted(grid["day"].unique())

    def piv(col):
        return grid.pivot(index="property_name", columns="day", values=col).reindex(
            index=props, columns=days
        )

    occ_pct = piv("occupancy_pct")
    occupied = piv("occupied_units")
    arr, dep, pred = piv("arrivals"), piv("departures"), piv("pred_cancels")
    expc = piv("exp_cancels") if "exp_cancels" in grid.columns else occupied * 0.0
    canc = piv("cancellations") if "cancellations" in grid.columns else occupied * 0.0
    ooo = piv("out_of_order") if "out_of_order" in grid.columns else occupied * 0.0
    free = piv("free_rooms") if "free_rooms" in grid.columns else occupied * np.nan

    has_pct = bool(np.isfinite(occ_pct.to_numpy(dtype="float64")).any())
    color = (
        occ_pct.to_numpy(dtype="float64")
        if has_pct
        else occupied.to_numpy(dtype="float64")
    )
    occ_np = occ_pct.to_numpy(dtype="float64")
    free_np = free.to_numpy(dtype="float64")

    # --- overbooking-relevance mask (which tiles stay visible) ----------
    if not has_pct or (not pct_on and not free_on):
        rel = np.ones(color.shape, dtype=bool)  # can't assess / no filter -> show all
    else:
        rel = np.zeros(color.shape, dtype=bool)
        if pct_on and pct_thr is not None:
            rel |= occ_np >= float(pct_thr)
        if free_on and free_thr is not None:
            rel |= free_np <= float(free_thr)
    z = np.where(rel, color, np.nan)  # greyed tiles fall back to plot bg

    # --- in-tile number depends on the chosen metric -------------------
    if metric == "expected":
        mv, fmt = expc.to_numpy(dtype="float64"), (lambda v: f"{v:.1f}")
    elif metric == "threshold":
        mv, fmt = pred.to_numpy(dtype="float64"), (lambda v: f"{int(round(v))}")
    else:
        mv = color
        fmt = (lambda v: f"{v:.0f}%") if has_pct else (lambda v: f"{v:.0f}")
    text = [
        [
            fmt(mv[i][j]) if (rel[i][j] and np.isfinite(mv[i][j])) else ""
            for j in range(mv.shape[1])
        ]
        for i in range(mv.shape[0])
    ]

    cd = np.dstack(
        [
            occupied.to_numpy(),
            arr.to_numpy(),
            dep.to_numpy(),
            pred.to_numpy(),
            expc.to_numpy(),
            canc.to_numpy(),
            ooo.to_numpy(),
            free_np,
        ]
    )
    hover = (
        "<b>%{y}</b> · %{x}"
        "<br>Occupancy: %{z:.0f}%<br>Occupied: %{customdata[0]:.0f}"
        "<br>Free rooms: %{customdata[7]:.0f}<br>Out of order: %{customdata[6]:.0f}"
        "<br>Arrivals: %{customdata[1]:.0f}<br>Departures: %{customdata[2]:.0f}"
        "<br>Cancellations (recorded): %{customdata[5]:.0f}"
        "<br>Expected cancellations (Σp): %{customdata[4]:.1f}"
        "<br>Above cost threshold: %{customdata[3]:.0f}<extra></extra>"
    )
    heat = go.Heatmap(
        z=z,
        x=days,
        y=props,
        customdata=cd,
        colorscale=_OCC_COLORSCALE,
        xgap=2,
        ygap=2,
        hoverongaps=False,
        zmin=0 if has_pct else None,
        zmax=100 if has_pct else None,
        text=text,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="Occ%" if has_pct else "Occ", thickness=12),
    )
    if show_hover:
        heat.hovertemplate = hover
    else:
        heat.hoverinfo = "skip"  # tab 2/3: no tooltips at all
    fig = go.Figure(heat)
    fig.update_layout(
        height=max(300, 60 + 26 * len(props)),
        xaxis_title=None,
        yaxis_title=None,
        yaxis_autorange="reversed",
    )
    fig = theme.brand_figure(fig)
    fig.update_layout(plot_bgcolor="#E9E9E9")  # greyed-out tiles read as grey
    return fig


# ---------------------------------------------------------------------------
# Five composition charts (computed from ARRIVALS of the current selection)
# ---------------------------------------------------------------------------
def _empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text="No arrivals", showarrow=False, font={"color": "#999"})
    fig.update_layout(title=title, height=260)
    return theme.brand_figure(fig)


def _hist(values: pd.Series, title: str, color: str, xtitle: str) -> go.Figure:
    vals = (
        pd.to_numeric(values, errors="coerce").dropna()
        if values is not None
        else pd.Series(dtype=float)
    )
    if len(vals) == 0:
        return _empty_fig(title)
    fig = go.Figure(go.Histogram(x=vals, marker_color=color, nbinsx=20))
    fig.update_layout(
        title=title,
        height=260,
        xaxis_title=xtitle,
        yaxis_title="Bookings",
        margin=dict(l=45, r=15, t=40, b=35),
    )
    return theme.brand_figure(fig)


def _hbar(
    counts: pd.Series, title: str, color: str, height: int = 260, left_margin: int = 100
) -> go.Figure:
    """Readable horizontal bar (categories on the y-axis, count on x, % label on bar).
    Replaces the old pies, which were unreadable once a category had many levels."""
    if counts is None or counts.empty:
        return _empty_fig(title)
    total = int(counts.sum())
    c = counts.sort_values()  # ascending => largest bar on top after plot
    pct = [f"{v / total * 100:.0f}%" for v in c.values]
    fig = go.Figure(
        go.Bar(
            x=c.values,
            y=c.index.astype(str),
            orientation="h",
            marker_color=color,
            text=pct,
            textposition="auto",
            cliponaxis=False,
            hovertemplate="%{y}: %{x} bookings<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Bookings",
        yaxis_title=None,
        margin=dict(l=left_margin, r=20, t=40, b=30),
    )
    return theme.brand_figure(fig)


def _granular_channel(df: pd.DataFrame) -> pd.Series:
    """OTA-level channel: the `source` field (Booking.com, Expedia, HRS, Airbnb, …),
    filled with `channelCode` for direct/IBE bookings that have no external source.
    This is the granular breakdown, NOT the coarse 'ChannelManager' bucket."""
    if df is None or df.empty:
        return pd.Series(dtype="object")
    source = (
        df["source"].astype("string")
        if "source" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )
    channel = (
        df["channelCode"].astype("string")
        if "channelCode" in df.columns
        else pd.Series("-", index=df.index, dtype="string")
    )
    return source.fillna(channel).fillna("-")


def _rateplan_table(df: pd.DataFrame, top: int = 8) -> "dmc.Table":
    """Top rate plans as a compact table (name / bookings / share). Rate-plan names are
    whitespace-normalised so trailing-space duplicates collapse into one row."""
    if df is None or df.empty or "ratePlan_name" not in df.columns:
        return dmc.Text("No arrivals", c="dimmed", size="sm")
    names = (
        df["ratePlan_name"].astype("string").str.strip().replace("", pd.NA).fillna("-")
    )
    vc = names.value_counts().head(top)
    total = int(names.notna().sum()) or 1
    body = [
        [str(name), f"{int(cnt):,}", f"{cnt / total * 100:.0f}%"]
        for name, cnt in vc.items()
    ]
    return dmc.Table(
        data={"head": ["Rate plan", "Bookings", "Share"], "body": body},
        striped=True,
        highlightOnHover=True,
        withRowBorders=False,
        fz="sm",
        verticalSpacing=4,
        horizontalSpacing="sm",
    )


def _tile(child, *, pad: str = "sm") -> "dmc.Card":
    return dmc.Card(
        child, withBorder=True, radius="lg", p=pad, style={"height": "100%"}
    )


def _graph_tile(fig: go.Figure) -> "dmc.Card":
    return _tile(dcc.Graph(figure=fig, config={"displayModeBar": False}))


def composition_row(df: pd.DataFrame, context_label: str) -> list:
    """Titled row of the five composition tiles for an arrivals frame:
    business/leisure (bar), length of stay (hist), lead time (hist), rate plan (table),
    channel/source (granular bar). Empty-safe."""
    n = 0 if df is None or df.empty else len(df)
    empty = df is None or df.empty

    # 1) business vs leisure - horizontal bar (was a pie)
    if empty or "travelPurpose" not in df.columns:
        tp_counts = pd.Series(dtype=int)
    else:
        tp = df["travelPurpose"].astype("string").fillna("").replace("", "Unknown")
        tp_counts = tp.value_counts()
    tile_purpose = _graph_tile(
        _hbar(tp_counts, "Business vs. leisure", theme.BLUE, left_margin=80)
    )

    # 2) length of stay + 3) lead time - histograms (kept, they read well)
    tile_los = _graph_tile(
        _hist(
            None if empty else df.get("los_nights"),
            "Length of stay",
            theme.GREEN,
            "Nights",
        )
    )
    tile_lead = _graph_tile(
        _hist(
            None if empty else df.get("lead_time_days"),
            "Lead time",
            theme.PURPLE,
            "Lead time (days from booking to arrival)",
        )
    )

    # 4) rate plan - TABLE (was an unreadable pie)
    tile_rate = _tile(
        [dmc.Text("Rate plan (top 8)", fw=600, size="sm", mb=6), _rateplan_table(df)],
        pad="md",
    )

    # 5) channel - granular OTA source as a horizontal bar
    ch_counts = _granular_channel(df).value_counts()
    if len(ch_counts) > 8:
        ch_counts = pd.concat(
            [ch_counts.head(8), pd.Series({"Other": int(ch_counts.iloc[8:].sum())})]
        )
    tile_channel = _graph_tile(
        _hbar(
            ch_counts,
            "Channel · source (granular)",
            theme.ORANGE,
            height=280,
            left_margin=115,
        )
    )

    cards = [tile_purpose, tile_los, tile_lead, tile_rate, tile_channel]
    return [
        dmc.Text(
            f"Composition · {context_label} · {n} arrivals", fw=600, size="sm", mb=6
        ),
        dmc.SimpleGrid(cards, cols={"base": 1, "sm": 2, "lg": 3}, spacing="md"),
    ]
