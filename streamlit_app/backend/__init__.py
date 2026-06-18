"""Backend-Fassade: **eine** Schnittstelle für alle Pages, ein Schalter.

Die Seiten rufen ausschließlich Funktionen aus diesem Modul. Welche
Implementierung dahinterliegt — synthetisches ``dummy`` oder echtes ``real``
(``src``-Scoring) — entscheidet ``mode()``:

  * Default: ``dummy`` (kein Modell/keine Spalte nötig).
  * Umstellen: Umgebungsvariable ``OVERBOOKING_BACKEND=real`` ODER zur Laufzeit
    ``set_mode("real")`` (Schalter auf der Seite „Datenaktualisierung").

So ist der spätere Wechsel ein Ein-Zeilen-Eingriff — keine Page muss angefasst
werden, weil beide Backends exakt das Schema aus ``schema.py`` liefern.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from . import derive, schema  # noqa: F401  (re-export für Pages)
from . import schema as S

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DATA_DIR = _REPO_ROOT / "Data"
_SNAPSHOT_JSON = _DATA_DIR / "dummy_snapshot.json"
_METADATA_JSON = _DATA_DIR / "dummy_metadata.json"

GENERATION_HORIZON_DAYS = 35   # wie weit der Snapshot generiert wird
_DEFAULT_SEED = 42

# Laufzeit-Override des Modus (None = aus Env/Default ableiten).
_MODE_OVERRIDE: str | None = None


# =============================================================================
# Modus
# =============================================================================
def mode() -> str:
    """Aktiver Backend-Modus: 'dummy' oder 'real'."""
    if _MODE_OVERRIDE in ("dummy", "real"):
        return _MODE_OVERRIDE
    return os.environ.get("OVERBOOKING_BACKEND", "dummy").strip().lower() or "dummy"


def set_mode(new_mode: str) -> None:
    """Modus zur Laufzeit setzen ('dummy' | 'real')."""
    global _MODE_OVERRIDE
    if new_mode not in ("dummy", "real"):
        raise ValueError("mode muss 'dummy' oder 'real' sein")
    _MODE_OVERRIDE = new_mode


# =============================================================================
# Standort-Helfer
# =============================================================================
@lru_cache(maxsize=1)
def units_by_hotel() -> dict[str, int]:
    """{hotel_code: units_total} aus configs/locations.yaml."""
    from .dummy import _load_locations
    return {hc: units for hc, _city, units in _load_locations()}


@lru_cache(maxsize=1)
def city_by_hotel() -> dict[str, str]:
    from .dummy import _load_locations
    return {hc: city for hc, city, _units in _load_locations()}


@lru_cache(maxsize=1)
def hotel_labels() -> dict[str, str]:
    """{hotel_code: Anzeige-Label}. Eindeutige Stadt → nur Stadt, sonst Stadt (CODE)."""
    cities = city_by_hotel()
    counts: dict[str, int] = {}
    for city in cities.values():
        counts[city] = counts.get(city, 0) + 1
    return {hc: (city if counts[city] == 1 else f"{city} ({hc})") for hc, city in cities.items()}


@lru_cache(maxsize=1)
def capacity_by_category() -> dict[str, dict[str, int]]:
    """{hotel_code: {Zimmerkategorie: Kapazität}}."""
    return {hc: S.category_capacity(units) for hc, units in units_by_hotel().items()}


# =============================================================================
# Snapshot-Status (nur Dummy: persistiert Seed/Refresh-Zeit)
# =============================================================================
def _read_snapshot() -> dict:
    if _SNAPSHOT_JSON.exists():
        try:
            return json.loads(_SNAPSHOT_JSON.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {}


def _write_snapshot(snap: dict) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SNAPSHOT_JSON.write_text(json.dumps(snap, indent=2), encoding="utf-8")


def _ensure_snapshot() -> dict:
    snap = _read_snapshot()
    if not snap.get("seed"):
        snap = {
            "seed": _DEFAULT_SEED,
            "refreshed_at": datetime.now().astimezone().isoformat(),
            "horizon_days": GENERATION_HORIZON_DAYS,
        }
        _write_snapshot(snap)
    return snap


# =============================================================================
# Hauptschnittstelle
# =============================================================================
def get_scored_bookings(force_refresh: bool = False) -> pd.DataFrame:
    """Bewertete Upcoming-Buchungen im kanonischen Schema (gecached)."""
    if mode() == "real":
        from . import real
        df = real.get_scored_bookings(
            horizon_days=GENERATION_HORIZON_DAYS, force_refresh=force_refresh
        )
    else:
        from . import dummy
        snap = _ensure_snapshot()
        df = dummy.generate(seed=int(snap["seed"]),
                            horizon_days=int(snap.get("horizon_days", GENERATION_HORIZON_DAYS)))
    return df


def get_metadata() -> dict:
    """Datenstand für Home + Datenaktualisierung. Erzeugt/aktualisiert den Cache."""
    df = get_scored_bookings()
    today = pd.Timestamp.today().normalize()
    conf = derive.confirmed(df)
    canceled = df[df[S.STATUS] == S.STATUS_CANCELED] if S.STATUS in df else df.iloc[0:0]
    upcoming = conf[conf[S.ARRIVAL_DATE] >= today]
    snap = _read_snapshot()
    meta = {
        "mode": mode(),
        "refreshed_at": snap.get("refreshed_at", datetime.now().astimezone().isoformat()),
        "reservations": {"rows": int(len(df))},
        "confirmed": {"rows": int(len(conf))},
        "canceled": {"rows": int(len(canceled))},
        "upcoming": {"rows": int(len(upcoming))},
        "properties": sorted(df[S.HOTEL_CODE].unique().tolist()),
        "window": {
            "earliest": str(conf[S.ARRIVAL_DATE].min().date()) if len(conf) else None,
            "latest": str(conf[S.ARRIVAL_DATE].max().date()) if len(conf) else None,
        },
        "high_risk": int((upcoming[S.CANCEL_PROBA] >= S.HIGH_THR).sum()) if len(upcoming) else 0,
        "cancel_rate": round(float(upcoming[S.CANCEL_PROBA].mean()), 3) if len(upcoming) else 0.0,
    }
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        _METADATA_JSON.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass
    return meta


def refresh(**kwargs) -> dict:
    """Daten aktualisieren.

    * dummy: neuer Seed → frischer, anderer Snapshot (Caches geleert).
    * real:  BigQuery future-only neu ziehen + neu bewerten.

    Returns die aktualisierten Metadaten.
    """
    if mode() == "real":
        from . import real
        real.get_scored_bookings(force_refresh=True)
        _write_snapshot({
            "seed": "real",
            "refreshed_at": datetime.now().astimezone().isoformat(),
            "horizon_days": GENERATION_HORIZON_DAYS,
        })
    else:
        from . import dummy
        dummy.generate.cache_clear()
        _write_snapshot({
            "seed": int(time.time()) % 1_000_000,
            "refreshed_at": datetime.now().astimezone().isoformat(),
            "horizon_days": GENERATION_HORIZON_DAYS,
        })
    return get_metadata()


def has_snapshot() -> bool:
    return _SNAPSHOT_JSON.exists() or mode() == "real"
