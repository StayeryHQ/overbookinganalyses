"""Leichte Daten-Helfer für die UI.

Bewusst minimal und unabhängig vom ``src``-ML-Package: liest die
Standort-Stammdaten aus ``configs/locations.yaml`` und stellt einen
Einstiegspunkt für den Daten-/Vorhersage-Status bereit.

``get_data_status()`` ist der Plug-in-Punkt für die nächsten Schritte:
Sobald das Platzhalter-Backend (Dummy-Modell) bzw. der echte BigQuery-Snapshot
existiert, liefert diese Funktion echte Metadaten. Bis dahin gibt sie sauber
``None`` zurück, damit die Pages einen freundlichen Empty-State zeigen.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCATIONS_YAML = _REPO_ROOT / "configs" / "locations.yaml"

# Kandidaten-Pfade, an denen das (künftige) Dummy-/Echt-Backend seinen
# Snapshot ablegen wird. Wird in einem späteren Schritt befüllt.
_DATA_DIR = _REPO_ROOT / "Data"
_SCORED_CANDIDATES = (
    _DATA_DIR / "scored_upcoming.parquet",
    _DATA_DIR / "dummy_scored_upcoming.parquet",
)
_METADATA_CANDIDATES = (
    _DATA_DIR / "app_metadata.json",
    _DATA_DIR / "dummy_metadata.json",
)


@lru_cache(maxsize=1)
def load_locations() -> pd.DataFrame:
    """Standort-Stammdaten als DataFrame.

    Spalten: hotel_code, city, neighborhood, bundesland, opening_date,
    units_total, notes. ``units_total`` ist int.
    """
    with _LOCATIONS_YAML.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    rows = data.get("locations", [])
    df = pd.DataFrame(rows)
    if not df.empty and "units_total" in df.columns:
        df["units_total"] = pd.to_numeric(df["units_total"], errors="coerce").fillna(0).astype(int)
    return df


def benchmark_overbooking_allowance(units_total: int) -> int:
    """Business-Regel: erlaubte Overbookings je Standortgröße.

    < 50 Units → 2 erlaubt, ≥ 50 Units → 4 erlaubt.
    """
    return 4 if units_total >= 50 else 2


def get_data_status() -> dict[str, Any] | None:
    """Status des aktuellen Daten-/Vorhersage-Snapshots.

    Returns:
        Dict mit Metadaten, sobald ein Snapshot existiert — sonst ``None``.
        Aktuell (vor Anbindung des Platzhalter-Backends) immer ``None``.
    """
    import json

    meta_path = next((p for p in _METADATA_CANDIDATES if p.exists()), None)
    if meta_path is None:
        return None
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def has_scored_snapshot() -> bool:
    """True, wenn bereits ein (Dummy- oder echter) Score-Snapshot auf Disk liegt."""
    return any(p.exists() for p in _SCORED_CANDIDATES)
