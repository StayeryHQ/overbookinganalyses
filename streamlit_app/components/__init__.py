"""Wiederverwendbare Streamlit-Komponenten für die Overbooking-Analytics-App.

Design 1:1 gespiegelt vom RevenueBlindSpots-Projekt, damit beide Stayery-Apps
konsistent aussehen.
"""

from .alerts import alert_card, alert_cards
from .app_data import (
    benchmark_overbooking_allowance,
    get_data_status,
    has_scored_snapshot,
    load_locations,
)
from .brand import hero, inject_brand_css, load_brand_config
from .icons import icon
from .nav import nav_card, placeholder_panel
from .section import (
    lazy_section,
    preload_all_button,
    render_toc,
    section,
    section_loaded,
)

__all__ = [
    # brand / layout
    "inject_brand_css",
    "hero",
    "load_brand_config",
    "icon",
    # alerts
    "alert_card",
    "alert_cards",
    # navigation / landing
    "nav_card",
    "placeholder_panel",
    # sections / TOC
    "section",
    "lazy_section",
    "render_toc",
    "section_loaded",
    "preload_all_button",
    # data helpers
    "load_locations",
    "get_data_status",
    "has_scored_snapshot",
    "benchmark_overbooking_allowance",
]
