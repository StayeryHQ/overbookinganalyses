# ---------------------------------------------------------------------------
# src/plotting.py - Stayery-branded PLOTLY helpers (single styling source).
#
# WHY THIS MODULE EXISTS
#   The notebooks used matplotlib; we are migrating to Plotly so the *exact same*
#   figure objects can be dropped into the Dash app (Plotly is Dash-native).
#   To avoid style drift, the brand theme + every chart factory live HERE and are
#   imported by both the notebooks and the Dash app. This mirrors what
#   `utils.apply_stayery_style()` does for matplotlib.
#
#   It is deliberately NOT imported by `src/__init__.py`, so `import src` never
#   requires plotly to be installed. Consumers do `from src.plotting import ...`.
#   (Add the dependency once: `uv add plotly` - and `uv add dash` for the app.)
#
# EVERY public function returns a `plotly.graph_objects.Figure`, which:
#   * renders inline in notebooks (fig.show()),
#   * serialises to HTML/JSON for reports,
#   * plugs straight into Dash via `dcc.Graph(figure=fig)`.
# ---------------------------------------------------------------------------

from __future__ import annotations

# Standard typing import; `Sequence` lets us accept lists/tuples/np arrays alike.
from typing import Sequence

# numpy/pandas for the light numeric work the factories do (curves, bins).
import numpy as np
import pandas as pd

# Plotly's two layers:
#   * graph_objects (go) = the explicit, low-level figure API (full control).
#   * io (pio)           = template registry + (optional) export helpers.
# We use go everywhere for predictability; pio only to register the theme.
import plotly.graph_objects as go
import plotly.io as pio

# Reuse the brand palette from utils so colours have ONE source (the YAML).
from .utils import categorical_palette, color, diverging_triplet

# Name under which we register the Stayery template in Plotly's global registry.
TEMPLATE_NAME = "stayery"


# =============================================================================
# Theme
# =============================================================================
def stayery_template() -> go.layout.Template:
    """Build the Stayery Plotly template (fonts, colours, white bg, subtle grid).

    A Plotly *template* is a reusable `layout`/`data` defaults object; setting it
    once means every figure inherits the brand look without repeating kwargs.
    """
    # The brand font chain - Plotly takes a single comma-separated string and
    # falls back left→right, exactly like CSS font-family.
    font_family = ("Neue Haas Grotesk Display Pro, Neue Haas Grotesk, Inter, "
                   "Helvetica Neue, Arial, sans-serif")

    # Build a Template object; `layout` holds the global defaults.
    tmpl = go.layout.Template()

    # --- global layout defaults ------------------------------------------------
    tmpl.layout = go.Layout(
        # Categorical series cycle through the brand palette (yellow, blue, ...).
        colorway=categorical_palette(),
        # Base font: family + size + brand black for text.
        font=dict(family=font_family, size=13, color=color("black")),
        # Titles slightly larger; left-aligned (x=0) like the Streamlit app.
        title=dict(font=dict(size=17, color=color("black")), x=0.0, xanchor="left"),
        # Transparent paper/plot backgrounds so charts sit cleanly on app cards.
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        # Horizontal legend along the top, like the brand dashboards.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0,
                    bgcolor="rgba(0,0,0,0)"),
        # Comfortable margins (l/r/t/b in px); small top leaves room for legend.
        margin=dict(l=60, r=20, t=50, b=50),
        # X axis: thin baseline, no vertical grid (matches matplotlib style).
        xaxis=dict(showgrid=False, zeroline=False, linecolor=color("black"),
                   linewidth=1, ticks="outside", tickcolor=color("black")),
        # Y axis: light horizontal grid only (the brand "grid.axis = y").
        yaxis=dict(showgrid=True, gridcolor="#E5E5E5", gridwidth=1, zeroline=False,
                   linecolor=color("black"), linewidth=1),
    )
    return tmpl


def apply_plotly_theme() -> None:
    """Register the Stayery template and make it the default for all figures.

    Call once at the top of a notebook / the Dash app. After this, every
    `go.Figure()` inherits the brand look unless explicitly overridden.
    """
    pio.templates[TEMPLATE_NAME] = stayery_template()
    pio.templates.default = TEMPLATE_NAME


RISK_COLORSCALE = [
    [0.00, "#FFFFFF"],   # 0% risk → white
    [0.33, "#FFF1A8"],   # low     → pale yellow
    [0.66, "#EB6E14"],   # mid     → brand orange
    [1.00, "#E62828"],   # high    → brand red
]


# =============================================================================
# Generic factories
# =============================================================================
def bars(
    categories: Sequence,
    values: Sequence,
    *,
    title: str = "",
    yaxis_title: str = "",
    xaxis_title: str = "",
    color_name: str = "yellow",
    text_fmt: str | None = ".3f",
    horizontal: bool = False,
) -> go.Figure:
    """A single-series bar chart (vertical by default, horizontal if asked).

    Used for metric comparisons and risk-bucket counts.
    """
    # Optionally format the value labels printed on the bars (e.g. ".3f").
    text = [format(v, text_fmt) for v in values] if text_fmt else None
    # `go.Bar` is one trace; orientation "h" swaps x/y for a horizontal bar.
    bar = go.Bar(
        x=values if horizontal else list(categories),       # x = numbers if horizontal
        y=list(categories) if horizontal else values,       # y = numbers if vertical
        orientation="h" if horizontal else "v",
        marker_color=color(color_name),                      # brand fill colour
        marker_line=dict(color=color("black"), width=0.6),   # thin black outline
        text=text, textposition="auto",                      # value labels on bars
    )
    fig = go.Figure(bar)                                     # wrap the trace in a figure
    # Apply titles; `template="stayery"` is implicit once apply_plotly_theme() ran.
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


def grouped_bars(
    categories: Sequence,
    series: dict[str, Sequence],
    *,
    title: str = "",
    yaxis_title: str = "",
    text_fmt: str | None = ".3f",
) -> go.Figure:
    """Grouped bar chart: one bar group per category, one bar per series.

    `series` maps series-name → values (aligned to `categories`). Used by the
    model comparison (e.g. AUC/AP/Brier across models).
    """
    fig = go.Figure()                                        # empty figure; add traces below
    palette = categorical_palette(len(series))               # one brand colour per series
    # Iterate series in insertion order; enumerate gives us a colour index.
    for i, (name, values) in enumerate(series.items()):
        text = [format(v, text_fmt) for v in values] if text_fmt else None
        # Each series is its own `go.Bar`; Plotly groups them by shared x.
        fig.add_bar(x=list(categories), y=list(values), name=name,
                    marker_color=palette[i], text=text, textposition="auto")
    # `barmode="group"` places series side-by-side (vs "stack").
    fig.update_layout(title=title, yaxis_title=yaxis_title, barmode="group")
    return fig


def horizontal_importance(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str = "",
    xaxis_title: str = "Contribution",
) -> go.Figure:
    """Signed horizontal bars (orange = increases risk, blue = decreases).

    Mirrors the Streamlit `importance_fig`: positive → brand orange, negative
    → brand blue. Used for logreg coefficients / SHAP-style contributions.
    """
    # Per-bar colour by sign of the value.
    colors = [color("orange") if v >= 0 else color("blue") for v in values]
    bar = go.Bar(
        x=list(values), y=list(labels), orientation="h",     # horizontal bars
        marker_color=colors, marker_line=dict(color=color("black"), width=0.6),
    )
    fig = go.Figure(bar)
    fig.update_layout(title=title, xaxis_title=xaxis_title)
    # A vertical reference line at x=0 (where effect flips sign).
    fig.add_vline(x=0, line_color=color("black"), line_width=1)
    return fig


# =============================================================================
# Model-evaluation factories (take arrays, stay framework-agnostic)
# =============================================================================
def roc_curve_fig(curves: dict[str, tuple[Sequence, Sequence, float]], *,
                  title: str = "ROC") -> go.Figure:
    """Overlay one or more ROC curves.

    `curves` maps model-name → (fpr, tpr, auc). Keeping the math outside means
    this factory has no sklearn dependency and is trivially testable.
    """
    fig = go.Figure()
    palette = categorical_palette(len(curves))
    for i, (name, (fpr, tpr, auc)) in enumerate(curves.items()):
        # One line trace per model; the AUC goes into the legend label.
        fig.add_scatter(x=list(fpr), y=list(tpr), mode="lines",
                        name=f"{name} (AUC={auc:.3f})", line=dict(color=palette[i], width=2))
    # Diagonal "coin-flip" baseline (dashed grey) from (0,0) to (1,1).
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                    line=dict(color="#B8B6A8", width=1.2, dash="dash"))
    fig.update_layout(title=title, xaxis_title="False positive rate",
                      yaxis_title="True positive rate")
    return fig


def pr_curve_fig(curves: dict[str, tuple[Sequence, Sequence, float]], *,
                 base_rate: float | None = None, title: str = "Precision-Recall") -> go.Figure:
    """Overlay precision-recall curves; optional base-rate baseline (no-skill).

    `curves` maps model-name → (recall, precision, ap).
    """
    fig = go.Figure()
    palette = categorical_palette(len(curves))
    for i, (name, (rec, prec, ap)) in enumerate(curves.items()):
        fig.add_scatter(x=list(rec), y=list(prec), mode="lines",
                        name=f"{name} (AP={ap:.3f})", line=dict(color=palette[i], width=2))
    # The base rate is the precision of a random classifier → horizontal line.
    if base_rate is not None:
        fig.add_hline(y=base_rate, line_color="#B8B6A8", line_width=1.2, line_dash="dash",
                      annotation_text=f"Base rate {base_rate:.2%}", annotation_position="top left")
    fig.update_layout(title=title, xaxis_title="Recall", yaxis_title="Precision")
    return fig


def calibration_fig(curves: dict[str, tuple[Sequence, Sequence]], *,
                    title: str = "Calibration") -> go.Figure:
    """Reliability diagram: predicted vs observed frequency, with the y=x ideal.

    `curves` maps model-name → (mean_predicted, fraction_positive).
    """
    fig = go.Figure()
    palette = categorical_palette(len(curves))
    # Perfect-calibration diagonal first (so model lines draw on top).
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", showlegend=False,
                    line=dict(color="#B8B6A8", width=1.2, dash="dash"))
    for i, (name, (mean_pred, frac_pos)) in enumerate(curves.items()):
        fig.add_scatter(x=list(mean_pred), y=list(frac_pos), mode="lines+markers",
                        name=name, line=dict(color=palette[i], width=2))
    fig.update_layout(title=title, xaxis_title="Predicted probability",
                      yaxis_title="Observed cancellation rate")
    return fig


def confusion_fig(cm: np.ndarray, *, labels: Sequence[str] = ("no cancel", "cancel"),
                  title: str = "Confusion matrix") -> go.Figure:
    """2×2 confusion matrix as an annotated heatmap (brand risk ramp)."""
    cm = np.asarray(cm)
    # `go.Heatmap` colours each cell; we reverse y so row 0 sits at the top
    # (reading order: truth on rows top→bottom).
    fig = go.Figure(go.Heatmap(
        z=cm, x=[f"predicted: {l}" for l in labels],
        y=[f"actual: {l}" for l in labels],
        colorscale=RISK_COLORSCALE, showscale=False,
    ))
    # Write the integer counts into each cell as annotations.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            fig.add_annotation(x=j, y=i, text=f"{int(cm[i, j]):,}",
                               showarrow=False, font=dict(size=16, color=color("black")))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")   # row 0 on top
    return fig


def lines_by_x(
    x: Sequence,
    series: dict[str, Sequence],
    *,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    reverse_x: bool = False,
    markers: bool = True,
) -> go.Figure:
    """Generic multi-line chart over a shared x axis.

    Used by the hazard notebook (AUC per snapshot day, hazard curves). Set
    `reverse_x=True` to show days-until-arrival counting down toward arrival.
    """
    fig = go.Figure()
    palette = categorical_palette(len(series))
    mode = "lines+markers" if markers else "lines"
    for i, (name, y) in enumerate(series.items()):
        fig.add_scatter(x=list(x), y=list(y), mode=mode, name=name,
                        line=dict(color=palette[i], width=2))
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    if reverse_x:
        fig.update_xaxes(autorange="reversed")   # large x (far from arrival) on the left
    return fig


def risk_heatmap(matrix: pd.DataFrame, *, title: str = "",
                 colorbar_title: str = "") -> go.Figure:
    """Location × date heatmap on the brand risk ramp (for the cancellation dashboard).

    `matrix`: index = location, columns = dates, values = expected cancellations
    or risk. This is the Plotly twin of `charts.cancellation_heatmap_fig`.
    """
    fig = go.Figure(go.Heatmap(
        z=matrix.values,                                  # 2-D value grid
        x=[str(c) for c in matrix.columns],               # column (date) labels
        y=[str(r) for r in matrix.index],                 # row (location) labels
        colorscale=RISK_COLORSCALE,
        colorbar=dict(title=colorbar_title),
    ))
    fig.update_layout(title=title)
    fig.update_yaxes(autorange="reversed")                # first location on top
    return fig


# =============================================================================
# Export helper
# =============================================================================
def fig_to_html(fig: go.Figure, path: str) -> str:
    """Write a standalone interactive HTML file (handy for reports/audit trail).

    `include_plotlyjs="cdn"` keeps the file small by loading plotly.js from a CDN
    instead of inlining ~3 MB. Returns the path for convenience.
    """
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


# =============================================================================
# Big multi-metric raster (occupancy & arrivals dashboard)
# =============================================================================
# Occupancy colour ramp (creme → brand yellow → orange), matching the Streamlit
# `charts._occupancy_cmap()` so the look is identical across both frontends.
OCCUPANCY_COLORSCALE = [
    [0.00, "#FFFDF2"],   # ~empty → creme
    [0.45, "#FFF3A0"],   # filling → pale yellow
    [0.70, "#FFE650"],   # busy    → brand yellow
    [0.85, "#F4B53F"],   # high    → amber
    [1.00, "#EB6E14"],   # full    → brand orange
]


def raster_grid_heatmap(
    grids: dict[str, "pd.DataFrame"],
    *,
    color_by: str = "occupancy",
    title: str = "",
    colorbar_title: str = "Auslastung",
    over_threshold: float = 1.0,
) -> go.Figure:
    """Big location × date raster where EACH cell shows four numbers at once.

    The cell is COLOURED by one metric (`color_by`, default occupancy on the
    occupancy ramp) and ANNOTATED with all four:
        ↑ arrivals   ↓ departures   <occupancy %>   ⚠ expected cancellations.
    Over-capacity cells (occupancy > `over_threshold`) get red text so they pop.

    `grids` must hold aligned DataFrames (same index = location, same columns =
    dates) under the keys: 'arrivals', 'departures', 'occupancy', 'expected_cancels'.
    This is the Plotly twin of the Streamlit occupancy/cancellation heatmaps,
    merged into one dense 14-day operations view.
    """
    occ  = grids["occupancy"]                              # occupancy matrix (0..1+)
    arr  = grids["arrivals"]                               # arrivals matrix (counts)
    dep  = grids["departures"]                             # departures matrix (counts)
    canc = grids["expected_cancels"]                       # expected cancellations (Σ proba)
    color_m = grids[color_by]                              # which matrix drives the colour

    # Axis labels as plain strings (dates → "YYYY-MM-DD", locations as given).
    x = [str(c) for c in occ.columns]
    y = [str(r) for r in occ.index]
    # Occupancy → occupancy ramp; anything else → the risk ramp.
    scale = OCCUPANCY_COLORSCALE if color_by == "occupancy" else RISK_COLORSCALE
    # Colour scale upper bound: 1.0 for occupancy, else the data max (min 1.0).
    zmax = 1.0 if color_by == "occupancy" else max(
        1.0, float(np.nanmax(color_m.values)) if color_m.size else 1.0)

    # One Heatmap trace provides the cell COLOURS (z) + the colour bar.
    # NB: do NOT use hoverinfo="skip" - that ALSO disables click events in Plotly
    # (the trace is excluded from hover/click entirely). We want clickData, so we
    # set a hovertemplate instead (hover works AND cells are clickable).
    fig = go.Figure(go.Heatmap(
        z=color_m.values, x=x, y=y,
        colorscale=scale, zmin=0, zmax=zmax,
        colorbar=dict(title=colorbar_title),
        hovertemplate="%{y} · %{x}<br>" + (color_by or "Value") + ": %{z:.2f}<extra></extra>",
    ))

    # Write the four metrics into every cell as a compact two-line annotation.
    for i in range(len(occ.index)):                        # rows = locations
        for j in range(len(occ.columns)):                  # cols = dates
            a = arr.iat[i, j]; d = dep.iat[i, j]           # arrivals / departures
            o = occ.iat[i, j]; c = canc.iat[i, j]          # occupancy / expected cancels
            o = 0.0 if pd.isna(o) else float(o)            # guard NaN occupancy
            # Line 1: arrivals + departures; line 2: occupancy % + expected cancels.
            txt = f"↑{int(a)} ↓{int(d)}<br>{o*100:.0f}% ⚠{c:.1f}"
            over = o > over_threshold                      # over-capacity?
            fig.add_annotation(x=x[j], y=y[i], text=txt, showarrow=False,
                               font=dict(size=9, color="#E62828" if over else "#1a1a1a"))

    # Dates on TOP (calendar reading), x labels angled, location order top→down.
    fig.update_layout(title=title, xaxis=dict(side="top", tickangle=-30))
    fig.update_yaxes(autorange="reversed")
    return fig
