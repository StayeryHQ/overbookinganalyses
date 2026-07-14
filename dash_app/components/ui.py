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
