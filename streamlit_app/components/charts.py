"""Brand-Charts + Tabellen-Helfer für die Overbooking-App.

Pattern wie in der RevenueBlindSpots-App: matplotlib im Stayery-Style →
als PNG gerendert und in ``st.session_state`` gecached (schnelle Reruns).
Die **Figur-Builder** sind reine Funktionen (kein Streamlit) und damit testbar;
nur Cache/Render/Tabellen hängen an Streamlit.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

from .brand import load_brand_config  # noqa: E402

_WD = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
_DEFAULT_DPI = 140


# =============================================================================
# Style
# =============================================================================
def apply_style_once() -> None:
    """Stayery-matplotlib-Style einmal je Session setzen."""
    if st.session_state.get("_ob_style_applied"):
        return
    cfg = load_brand_config()
    chain = [cfg["typography"]["primary"], *cfg["typography"]["primary_fallback"], "DejaVu Sans"]
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": chain,
        "font.size": 10.5,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor": "#111111",
        "savefig.dpi": _DEFAULT_DPI,
        "figure.dpi": _DEFAULT_DPI,
    })
    st.session_state["_ob_style_applied"] = True


def _occupancy_cmap() -> LinearSegmentedColormap:
    # Creme → Brand-Gelb → Orange (Belegung steigend).
    return LinearSegmentedColormap.from_list(
        "stayery_occ", ["#FFFDF2", "#FFF3A0", "#FFE650", "#F4B53F", "#EB6E14"]
    )


def _risk_cmap() -> LinearSegmentedColormap:
    # Weiß → Gelb → Orange → Rot (Storno-Druck steigend).
    return LinearSegmentedColormap.from_list(
        "stayery_risk", ["#FFFFFF", "#FFF1A8", "#EB6E14", "#E62828"]
    )


def _date_labels(cols) -> list[str]:
    out = []
    for c in cols:
        d = pd.Timestamp(c)
        out.append(f"{_WD[d.weekday()]}\n{d.day:02d}.{d.month:02d}")
    return out


# =============================================================================
# Figur-Builder (rein matplotlib)
# =============================================================================
def occupancy_heatmap_fig(matrix: pd.DataFrame, row_labels: list[str] | None = None):
    """Belegungs-Heatmap. ``matrix``: Index = Standort, Spalten = Datum (Quote 0..1+)."""
    rows = row_labels or list(matrix.index)
    data = matrix.values.astype(float)
    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=(max(7.5, n_cols * 0.66), max(3.2, n_rows * 0.44)))
    cmap = _occupancy_cmap()
    norm = Normalize(vmin=0.0, vmax=1.0)
    ax.imshow(np.clip(data, 0, 1), aspect="auto", cmap=cmap, norm=norm)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                continue
            over = v > 1.0
            ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                    fontsize=8.0, fontweight="bold" if over else "normal",
                    color="#E62828" if over else "#1a1a1a")
            if over:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                           edgecolor="#E62828", linewidth=1.8))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(_date_labels(matrix.columns), fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows, fontsize=9)
    for j, c in enumerate(matrix.columns):
        if pd.Timestamp(c).weekday() >= 5:
            ax.get_xticklabels()[j].set_color("#B8860B")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=1.4)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def cancellation_heatmap_fig(matrix: pd.DataFrame, row_labels: list[str] | None = None,
                             integer: bool = True):
    """Heatmap erwarteter/wahrscheinlicher Stornos je Standort/Tag."""
    rows = row_labels or list(matrix.index)
    data = matrix.values.astype(float)
    n_rows, n_cols = data.shape
    vmax = max(1.0, np.nanmax(data)) if data.size else 1.0
    fig, ax = plt.subplots(figsize=(max(7.5, n_cols * 0.66), max(3.2, n_rows * 0.44)))
    cmap = _risk_cmap()
    norm = Normalize(vmin=0.0, vmax=vmax)
    ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)

    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.0f}" if integer else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.0,
                    color="#FFFFFF" if v > 0.62 * vmax else "#1a1a1a")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(_date_labels(matrix.columns), fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#ECEAE0", linewidth=1.0)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def hist_fig(series: pd.Series, *, bins: int = 24, xlabel: str = "",
             vline: float | None = None, color: str = "#FFE650"):
    """Schlankes Histogramm im Brand-Style."""
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    vals = pd.to_numeric(series, errors="coerce").dropna().values
    if vals.size:
        ax.hist(vals, bins=bins, color=color, edgecolor="#111111", linewidth=0.6)
    if vline is not None:
        ax.axvline(vline, color="#E62828", linewidth=1.6, linestyle="--")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Buchungen", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    fig.tight_layout()
    return fig


def _mono_cmap(end_hex: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("mono", ["#FFFFFF", end_hex])


def value_heatmap_fig(matrix: pd.DataFrame, row_labels=None, *, cmap=None,
                      integer=True, fmt=None, vmax=None, x_labels=None):
    rows = row_labels or list(matrix.index)
    data = matrix.values.astype(float)
    n_rows, n_cols = data.shape
    vmax = vmax or (max(1.0, np.nanmax(data)) if data.size else 1.0)
    cmap = cmap or _risk_cmap()
    fig, ax = plt.subplots(figsize=(max(7.5, n_cols * 0.66), max(3.0, n_rows * 0.44)))
    ax.imshow(data, aspect="auto", cmap=cmap, norm=Normalize(0.0, vmax))
    for i in range(n_rows):
        for j in range(n_cols):
            v = data[i, j]
            if np.isnan(v):
                continue
            txt = (fmt % v) if fmt else (f"{v:.0f}" if integer else f"{v:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.0,
                    color="#FFFFFF" if v > 0.62 * vmax else "#1a1a1a")
    xl = x_labels if x_labels is not None else _date_labels(matrix.columns)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(xl, fontsize=8)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(rows, fontsize=9)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#ECEAE0", linewidth=1.0)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def arrivals_cmap():
    return _mono_cmap("#1E4BA1")


def departures_cmap():
    return _mono_cmap("#6E32C8")


def confusion_fig(cm, labels=("kein Storno", "Storno")):
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    vmax = cm.max() if cm.size else 1
    ax.imshow(cm, cmap=_occupancy_cmap(), norm=Normalize(0, vmax))
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(cm[i, j]):,}".replace(",", "."), ha="center", va="center",
                    fontsize=14, fontweight="bold", color="#1a1a1a")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"vorhergesagt:\n{labels[0]}", f"vorhergesagt:\n{labels[1]}"], fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"tatsächlich:\n{labels[0]}", f"tatsächlich:\n{labels[1]}"], fontsize=9)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linewidth=2.0)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def importance_fig(df: pd.DataFrame):
    d = df.iloc[::-1]
    colors = ["#EB6E14" if c >= 0 else "#1E4BA1" for c in d["coef"]]
    fig, ax = plt.subplots(figsize=(6.6, max(3.0, 0.42 * len(d))))
    ax.barh(d["Feature"], d["coef"], color=colors, edgecolor="#111111", linewidth=0.6)
    ax.axvline(0, color="#111111", linewidth=1.0)
    ax.set_xlabel("Koeffizient (standardisiert) — orange erhöht, blau senkt Storno-Risiko", fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    fig.tight_layout()
    return fig


def calibration_fig(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], color="#B8B6A8", linestyle="--", linewidth=1.2)
    ax.plot(df["predicted"], df["actual"], marker="o", color="#EB6E14", linewidth=2.0)
    ax.set_xlabel("vorhergesagte Wahrscheinlichkeit", fontsize=10)
    ax.set_ylabel("tatsächliche Storno-Rate", fontsize=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(color="#E5E5E5", linewidth=0.6)
    fig.tight_layout()
    return fig


# =============================================================================
# Cache + Render
# =============================================================================
def _sig(*parts) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(str(p).encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def chart_png(cache_key: str, fig_fn: Callable, *args, dpi: int = _DEFAULT_DPI, **kwargs) -> bytes:
    """Figur rendern → PNG-Bytes, gecached in session_state."""
    bucket = st.session_state.setdefault("_ob_chart_cache", {})
    if cache_key in bucket:
        return bucket[cache_key]
    fig = fig_fn(*args, **kwargs)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    png = buf.getvalue()
    bucket[cache_key] = png
    if len(bucket) > 48:
        for k in list(bucket.keys())[: len(bucket) - 48]:
            bucket.pop(k, None)
    return png


def render(cache_key: str, fig_fn: Callable, *args, **kwargs) -> None:
    """chart_png + st.image (container-breit)."""
    png = chart_png(cache_key, fig_fn, *args, **kwargs)
    st.image(png, use_container_width=True)


def heatmap_signature(matrix: pd.DataFrame, *extra) -> str:
    """Cache-Key aus Matrix-Inhalt + Extras."""
    return _sig(matrix.to_numpy().tobytes() if matrix.size else b"", list(matrix.columns), *extra)


# =============================================================================
# Tabellen
# =============================================================================
def style_collect() -> None:
    """plt.close('all') + gc — am Seitenende aufrufen."""
    import gc
    plt.close("all")
    gc.collect()


def data_table_expander(df: pd.DataFrame, *, title: str = "Datentabelle",
                        filename: str = "tabelle", expanded: bool = False,
                        max_rows: int = 300, column_config: dict | None = None,
                        height: int | None = None) -> None:
    """Expander mit DataFrame + CSV-Download (alle Zeilen)."""
    if df is None or len(df) == 0:
        return
    with st.expander(title, expanded=expanded):
        disp = df.head(max_rows) if len(df) > max_rows else df
        kwargs = {"hide_index": True, "use_container_width": True}
        if column_config:
            kwargs["column_config"] = column_config
        if height is not None:
            kwargs["height"] = height
        st.dataframe(disp, **kwargs)
        if len(df) > max_rows:
            st.caption(f"Anzeige auf {max_rows} Zeilen begrenzt "
                       f"({len(df):,} gesamt) — CSV enthält alle.".replace(",", "."))
        st.download_button(
            "Als CSV herunterladen",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{filename}.csv", mime="text/csv",
            key=f"dl_{filename}_{len(df)}",
        )
