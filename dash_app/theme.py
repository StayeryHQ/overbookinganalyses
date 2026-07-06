# dash_app/theme.py
# Central brand styling for the Dash app — colours, fonts, Plotly defaults.
# Pulled from configs/stayery_brand.yaml via src.load_brand_config() so the app and
# the notebooks stay visually consistent.

from __future__ import annotations

import dash_bootstrap_components as dbc

from src import load_brand_config

_brand = load_brand_config()
_core = _brand["colors"]["core"]
_sup = _brand["colors"]["supporting"]

# ---- Colours ---------------------------------------------------------------
YELLOW = _core["yellow"]
BLACK = _core["black"]
WHITE = _core["white"]
GREEN = _sup["green"]
ORANGE = _sup["orange"]
RED = _sup["red"]
BLUE = _sup["blue"]
PINK = _sup["pink"]
PURPLE = _sup["purple"]

# Risk-bucket -> colour (matches src.scoring bucket names).
RISK_COLORS = {"low": GREEN, "uncertain": ORANGE, "high": RED}

# Categorical series colours (brand order) for room-type lines etc.
CATEGORICAL = [YELLOW, BLUE, GREEN, ORANGE, PINK, PURPLE, RED]

# ---- Typography ------------------------------------------------------------
_typ = _brand["typography"]
FONT_FAMILY = ", ".join([_typ["primary"], *_typ["primary_fallback"]])
# Display font stack (Topol) for headings / callouts.
HEADING_FONT_FAMILY = ", ".join([_typ["display"], *_typ["display_fallback"]])

# ---- Cancel-rate heatmap colourscale (green = low/good → red = high/bad) ----
# Intuitive traffic-light reading for a "higher is worse" metric, built from the
# brand's supporting green/yellow/red so it stays on-brand.
CANCEL_SCALE = [[0.0, GREEN], [0.5, YELLOW], [1.0, RED]]

# ---- dash-mantine-components theme -----------------------------------------
# Seeded from the same brand config so dmc components match the notebooks/charts.
# primaryColor "dark" => controls render near-black (STAYERY is black + yellow);
# yellow is used as an explicit accent (KPI bar, highlights) rather than as the UI
# primary, which keeps text-on-accent contrast readable. forceColorScheme="light"
# is applied at the provider so the brand's white canvas is stable regardless of the
# viewer's OS dark-mode setting.
DMC_THEME = {
    "primaryColor": "dark",
    "defaultRadius": "md",
    "fontFamily": FONT_FAMILY,
    "fontFamilyMonospace": "SFMono-Regular, Menlo, monospace",
    "headings": {"fontFamily": HEADING_FONT_FAMILY, "fontWeight": "700"},
}

# ---- Stylesheets -----------------------------------------------------------
# FLATLY = clean, light Bootstrap theme; brand accents are applied via inline
# styles so we don't fight the theme. dbc icons for small UI affordances.
EXTERNAL_STYLESHEETS = [dbc.themes.FLATLY, dbc.icons.BOOTSTRAP]

# ---- Plotly defaults -------------------------------------------------------
def brand_figure(fig):
    """Apply the brand font + a clean white layout to a Plotly figure in place."""
    fig.update_layout(
        font=dict(family=FONT_FAMILY, color=BLACK, size=13),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        margin=dict(l=50, r=20, t=40, b=40),
        colorway=CATEGORICAL,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EEEEEE", zeroline=False)
    return fig

# ---- Reusable style dicts --------------------------------------------------
CARD_STYLE = {"borderRadius": "10px", "border": "1px solid #E6E6E6",
              "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"}
KPI_VALUE_STYLE = {"fontSize": "1.9rem", "fontWeight": 700, "lineHeight": 1.1,
                   "fontFamily": FONT_FAMILY}
KPI_LABEL_STYLE = {"fontSize": "0.8rem", "color": "#666", "textTransform": "uppercase",
                   "letterSpacing": "0.04em"}
