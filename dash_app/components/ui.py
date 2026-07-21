# dash_app/components/ui.py
# Reusable dash-mantine-components building blocks shared across all pages. These are
# intentionally generic (no page-specific data) so the Occupancy / XAI / Home pages can
# adopt the same look, filter bar, KPI cards, chart cards, skeleton loaders and drawer
# once the design is signed off here on the Cancellation History page.
#
# Every prop used here was verified against the installed dash-mantine-components 2.8.0
# (MultiSelect / Skeleton.visible / Tooltip / Card / Paper / SimpleGrid) and Dash 4.3.0
# (dcc.Loading.custom_spinner) - no signatures are assumed from memory.

from __future__ import annotations

import dash_mantine_components as dmc
from dash import dcc, html

from dash_app import theme


# ---------------------------------------------------------------------------
# Info icon: a small ⓘ that reveals an explanation on hover. Used to demystify
# metrics ("lead time", "stay segment") without cluttering the chart.
# ---------------------------------------------------------------------------
def info_icon(text: str):
    return dmc.Tooltip(
        html.I(className="bi bi-info-circle",
               style={"cursor": "help", "color": "#9AA0A6", "fontSize": "0.85rem"}),
        label=text, multiline=True, w=290, withArrow=True, position="top",
        transitionProps={"transition": "fade", "duration": 150},
    )


# ---------------------------------------------------------------------------
# KPI metric card. `accent=True` adds the brand-yellow left bar for the headline
# figure. Values not backed by real data should be passed as "unavailable".
# ---------------------------------------------------------------------------
def kpi_card(label: str, value: str, sub: str | None = None,
             accent: bool = False, tooltip: str | None = None):
    head = dmc.Group(
        [dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=600),
         info_icon(tooltip) if tooltip else None],
        gap=4, wrap="nowrap",
    )
    body = [head,
            dmc.Text(value, fw=700, style={"fontSize": "1.7rem", "lineHeight": 1.15}),
            dmc.Text(sub, size="xs", c="dimmed") if sub else None]
    style = {"borderLeft": f"4px solid {theme.YELLOW}"} if accent else {}
    return dmc.Paper(body, p="md", radius="lg", withBorder=True, style=style)


def kpi_strip(cards: list, min_width: int = 210):
    """Responsive row of KPI cards (wraps on narrow viewports)."""
    return dmc.SimpleGrid(cards, cols={"base": 1, "xs": 2, "md": 4}, spacing="md",
                          style={"minWidth": min_width})


# ---------------------------------------------------------------------------
# Chart card: titled surface with an optional info tooltip and header control
# (e.g. a SegmentedControl), wrapping a Graph in a skeleton-style loader so long
# aggregations show a shimmer instead of a bare spinner.
# ---------------------------------------------------------------------------
def chart_card(title: str, graph_id: str, *, info: str | None = None,
               height: int | None = 340, subtitle: str | None = None,
               header_extra=None):
    # height=None => let the figure's own layout height drive (e.g. a heatmap whose
    # height scales with the number of rows); the skeleton uses a sensible fallback.
    graph_style = {"height": f"{height}px"} if height else {"width": "100%"}
    skeleton_h = height or 380
    header = dmc.Group(
        [dmc.Group([dmc.Text(title, fw=600, size="sm"),
                    info_icon(info) if info else None], gap=6, wrap="nowrap"),
         header_extra if header_extra is not None else html.Span()],
        justify="space-between", align="center", wrap="nowrap",
    )
    children = [header]
    if subtitle:
        children.append(dmc.Text(subtitle, size="xs", c="dimmed", mt=2, mb=2))
    children.append(
        dcc.Loading(
            dcc.Graph(id=graph_id, config={"displayModeBar": False}, style=graph_style),
            custom_spinner=dmc.Skeleton(height=skeleton_h, radius="md", animate=True),
            # `custom_spinner` verified present in dcc.Loading (Dash 4.3.0).
        )
    )
    return dmc.Card(children, withBorder=True, radius="lg", p="md", shadow="xs",
                    style={"height": "100%"})


# ---------------------------------------------------------------------------
# Location filter bar. MultiSelect defaults to ALL locations selected; clearing it
# is treated by callbacks as "all" too, so charts are never empty by accident.
# ---------------------------------------------------------------------------
def location_filter(options: list[str], filter_id: str, *,
                    span_label: str | None = None):
    data = [{"label": o, "value": o} for o in options]
    control = dmc.MultiSelect(
        id=filter_id, data=data, value=list(options),
        placeholder="All locations", clearable=True, searchable=True,
        hidePickedOptions=False, maxDropdownHeight=320,
        leftSection=html.I(className="bi bi-geo-alt"),
        comboboxProps={"withinPortal": True},
        styles={"input": {"minHeight": "40px"}},
    )
    label_row = dmc.Group(
        [dmc.Text("Locations", size="sm", fw=600),
         dmc.Text(span_label, size="xs", c="dimmed") if span_label else None],
        justify="space-between", align="baseline",
    )
    return dmc.Paper([label_row, dmc.Space(h=6), control],
                     p="md", radius="lg", withBorder=True)


# ---------------------------------------------------------------------------
# Sticky filter bar: location MultiSelect + a global time-window control, pinned
# to the top so it stays reachable after scrolling. `position: sticky` degrades
# gracefully (just scrolls with the page) if an ancestor prevents pinning.
# ---------------------------------------------------------------------------
_TIMEWIN_DATA = [{"label": "All time", "value": "all"}, {"label": "24 mo", "value": "24"},
                 {"label": "12 mo", "value": "12"}, {"label": "6 mo", "value": "6"}]


def sticky_filter_bar(options: list[str], filter_id: str, timewin_id: str, *,
                      span_label: str | None = None, timewin_info: str | None = None):
    location = dmc.MultiSelect(
        id=filter_id, data=[{"label": o, "value": o} for o in options], value=list(options),
        placeholder="All locations", clearable=True, searchable=True,
        hidePickedOptions=False, maxDropdownHeight=320,
        leftSection=html.I(className="bi bi-geo-alt"),
        comboboxProps={"withinPortal": True},
        styles={"input": {"minHeight": "40px"}})
    left = dmc.Stack([
        dmc.Group([dmc.Text("Locations", size="sm", fw=600),
                   dmc.Text(span_label, size="xs", c="dimmed") if span_label else None],
                  justify="space-between", align="baseline"),
        location,
    ], gap=6, style={"flex": "1 1 340px", "minWidth": "260px"})
    right = dmc.Stack([
        dmc.Group([dmc.Text("Time window", size="sm", fw=600),
                   info_icon(timewin_info) if timewin_info else None], gap=4, wrap="nowrap"),
        dmc.SegmentedControl(id=timewin_id, data=_TIMEWIN_DATA, value="all",
                             size="xs", radius="md"),
    ], gap=6)
    bar = dmc.Group([left, right], justify="space-between", align="flex-end",
                    gap="lg", wrap="wrap")
    return dmc.Paper(bar, p="md", radius="lg", withBorder=True,
                     style={"position": "sticky", "top": "8px", "zIndex": 200,
                            "backgroundColor": theme.WHITE,
                            "boxShadow": "0 2px 12px rgba(0,0,0,0.06)"})


# ---------------------------------------------------------------------------
# Job loader: a beautiful RingProgress % + a gently spinning hourglass + the live
# status message + an optional Cancel button. Replaces the raw <progress> bar.
# IDs: {prefix}-wrap / -ring / -pct / -msg / -cancel. Hidden until a job runs.
# ---------------------------------------------------------------------------
def job_loader(prefix: str, *, with_cancel: bool = True) -> html.Div:
    right = [
        dmc.Group([
            html.I(className="bi bi-hourglass-split stayery-hourglass",
                   style={"color": theme.ORANGE, "fontSize": "1.15rem"}),
            dmc.Text(id=f"{prefix}-msg", size="sm", fw=500),
        ], gap=8, align="center", wrap="nowrap"),
    ]
    if with_cancel:
        right.append(dmc.Button("Cancel", id=f"{prefix}-cancel", size="xs", variant="subtle",
                                color="gray", leftSection=html.I(className="bi bi-x-circle")))
    inner = dmc.Group([
        dmc.RingProgress(
            id=f"{prefix}-ring", size=64, thickness=6, roundCaps=True,
            sections=[{"value": 0, "color": "yellow"}],
            label=dmc.Center(dmc.Text("0%", id=f"{prefix}-pct", fw=700, size="sm")),
        ),
        dmc.Stack(right, gap=6),
    ], gap="lg", align="center")
    return html.Div(inner, id=f"{prefix}-wrap", style={"display": "none"})


def loader_view(pct: float, message: str, *, show: bool):
    """(ring_sections, pct_text, message, wrap_style) for one job loader  feed these to
    the {prefix}-ring/-pct/-msg/-wrap outputs from a poll callback."""
    p = max(0, min(100, int(round(float(pct)))))
    return ([{"value": p, "color": "yellow"}], f"{p}%", message,
            {"display": "block"} if show else {"display": "none"})


def two_stage_loader(prefix: str, s1_label: str, s2_label: str, *,
                     with_cancel: bool = True) -> html.Div:
    """Two RingProgress rings side by side (one per stage) + spinning hourglass + message
    + Cancel. For jobs with two clear phases (e.g. retrain: fit -> rebuild evaluation).
    IDs: {prefix}-wrap / -ring1 / -pct1 / -ring2 / -pct2 / -msg / -cancel."""
    def _ring(rid, pid, label):
        return dmc.Stack([
            dmc.RingProgress(id=rid, size=58, thickness=5, roundCaps=True,
                             sections=[{"value": 0, "color": "yellow"}],
                             label=dmc.Center(dmc.Text("0%", id=pid, fw=700, size="xs"))),
            dmc.Text(label, size="xs", c="dimmed", ta="center"),
        ], gap=2, align="center")
    right = [dmc.Group([
        html.I(className="bi bi-hourglass-split stayery-hourglass",
               style={"color": theme.ORANGE, "fontSize": "1.15rem"}),
        dmc.Text(id=f"{prefix}-msg", size="sm", fw=500),
    ], gap=8, align="center", wrap="nowrap")]
    if with_cancel:
        right.append(dmc.Button("Cancel", id=f"{prefix}-cancel", size="xs", variant="subtle",
                                color="gray", leftSection=html.I(className="bi bi-x-circle")))
    inner = dmc.Group([
        _ring(f"{prefix}-ring1", f"{prefix}-pct1", s1_label),
        _ring(f"{prefix}-ring2", f"{prefix}-pct2", s2_label),
        dmc.Stack(right, gap=6),
    ], gap="lg", align="center")
    return html.Div(inner, id=f"{prefix}-wrap", style={"display": "none"})


def two_stage_view(frac: float, message: str, *, show: bool):
    """(ring1_sections, pct1, ring2_sections, pct2, message, wrap_style) for a two-stage
    loader. Stage 1 fills over frac 0→0.5, stage 2 over 0.5→1."""
    f = max(0.0, min(1.0, float(frac)))
    p1 = int(round(min(f / 0.5, 1.0) * 100))
    p2 = int(round(max(0.0, (f - 0.5) / 0.5) * 100))
    return ([{"value": p1, "color": "yellow"}], f"{p1}%",
            [{"value": p2, "color": "yellow"}], f"{p2}%",
            message, {"display": "block"} if show else {"display": "none"})
