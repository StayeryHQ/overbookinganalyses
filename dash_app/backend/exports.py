# dash_app/backend/exports.py
# Excel export for the Occupancy & Predictions page. Two sheets, both read-only from the
# local caches (no BigQuery, no re-scoring):
#   1) "Predictions (shown)"  — exactly the bookings currently shown in the page table
#      (same rows as panels.booking_row_data, honouring the property filter + heatmap cell).
#   2) "Backtest"             — the served model's leak-free walk-forward predictions vs the
#      REAL outcome (model_performance eval artifact), so predictions can be reconciled
#      against what actually happened. Falls back to a short note sheet if that artifact
#      has not been built yet (Model Performance → Rebuild evaluation / `python main.py eval`).
#
# openpyxl is a project dependency (see pyproject); the workbook is streamed straight into
# the buffer dcc.Download hands us — nothing is written to disk.

from __future__ import annotations

import pandas as pd

from dash_app.backend import data_access as da
from dash_app.backend import model_performance as mp

# Friendly headers for the "shown" sheet — mirror the on-screen table columns.
_SHOWN_RENAME = {
    "id": "Booking", "property_name": "Property", "arrival": "Arrival",
    "los_nights": "LoS (nights)", "channelCode": "Channel",
    "cancel_proba": "Cancel risk (predicted)", "risk_label": "Risk",
    "flag": "Flag", "status": "Status",
}
_BACKTEST_RENAME = {
    "property_name": "Property", "days_until_arrival": "Days until arrival",
    "y_prob": "Predicted cancel prob.", "y_true": "Actually cancelled (1/0)",
}


def _shown_frame(properties: list[str] | None, day: str | None,
                 threshold: float) -> pd.DataFrame:
    """The exact rows the page table shows: scored window, enriched with the cost-based
    risk label, filtered to the current property selection (and heatmap day, if any)."""
    from dash_app.components import panels          # lazy: avoid import cycle at module load
    scored = da.add_display_columns(da.load_scored(), threshold)
    if scored.empty:
        return pd.DataFrame()
    arrivals = da.arrivals_window(scored, properties, day)
    rows = panels.booking_row_data(arrivals)
    df = pd.DataFrame(rows)
    return df.rename(columns=_SHOWN_RENAME) if not df.empty else df


def _backtest_frame(model: str, threshold: float) -> pd.DataFrame:
    """Per-booking backtest for the served model: predicted probability + the actual
    outcome, plus a decision flag at the current cost threshold and a 'correct?' column.
    Empty frame if the model's eval artifact has not been built."""
    ev = mp.load_eval(model)
    if ev is None or ev.empty or "y_true" not in ev.columns or "y_prob" not in ev.columns:
        return pd.DataFrame()
    keep = [c for c in ["property_name", "days_until_arrival", "y_prob", "y_true"]
            if c in ev.columns]
    out = ev[keep].copy()
    prob = pd.to_numeric(ev["y_prob"], errors="coerce")
    actual = pd.to_numeric(ev["y_true"], errors="coerce").astype("Int64")
    out = out.rename(columns=_BACKTEST_RENAME)
    out["Predicted cancel @ threshold"] = (prob >= threshold).astype(int)
    out["Correct"] = (out["Predicted cancel @ threshold"] == actual).astype("Int64")
    return out


def _autosize(writer, sheet_name: str, df: pd.DataFrame, cap: int = 42) -> None:
    """Roughly fit each column to its header/content width (nice-to-have, not critical)."""
    ws = writer.sheets[sheet_name]
    from openpyxl.utils import get_column_letter
    for i, col in enumerate(df.columns, start=1):
        body = df[col].astype(str)
        width = min(cap, max(len(str(col)), int(body.str.len().max() or 0)) + 2)
        ws.column_dimensions[get_column_letter(i)].width = width


def write_predictions_workbook(buf, properties: list[str] | None, day: str | None,
                               threshold: float, model: str) -> None:
    """Write the two-sheet workbook into `buf` (a binary buffer from dcc.send_bytes)."""
    shown = _shown_frame(properties, day, threshold)
    back = _backtest_frame(model, threshold)

    if shown.empty:
        shown = pd.DataFrame({"Note": [
            "No scored bookings in the current view. Run scoring on the page first, "
            "then download again."]})
    if back.empty:
        back = pd.DataFrame({"Note": [
            f"No backtest artifact for model '{model}' yet. Build it on the Model "
            "Performance page (Rebuild evaluation) or run `python main.py eval`."]})

    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        shown.to_excel(xw, index=False, sheet_name="Predictions (shown)")
        back.to_excel(xw, index=False, sheet_name="Backtest")
        _autosize(xw, "Predictions (shown)", shown)
        _autosize(xw, "Backtest", back)
