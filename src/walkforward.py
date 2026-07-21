# ---------------------------------------------------------------------------
# src/walkforward.py
# Point-in-time temporal helpers  the SINGLE evaluation regime for every
# cancellation model (static classifiers AND the hazard model).
#
# Three timestamps, kept distinct (this is the thing that was previously muddled):
#   * created            = when a booking's FEATURES first exist (information time).
#                          NOT the overbooking decision  that is made near arrival.
#   * outcome_known_date = when the LABEL became known (cancellationTime for a
#                          pre-arrival cancel, else arrival). The only no-leakage
#                          rule is: train on outcomes known before the scoring date.
#   * arrival            = resolution.
#
# What the folds here measure  DECISION-TIME, production-faithful:
#   Production scores a booking on every day it is open AND within H days of
#   arrival. We test each booking EXACTLY ONCE, at its DECISION DATE
#       S* = max(created, arrival - H)
#   (long-lead -> arrival-H; short-lead -> its creation day), if it is still open
#   then. Its evaluation horizon is d = min(lead, H). This is the fix for a short
#   median lead: a non-overlapping arrival grid silently dropped ~half the
#   decision population (bookings booked < H days out); anchoring on S* keeps them.
#       train = outcome_known_date <= O_k              # resolved before the window
#       test  = O_k < S* <= O_k+step  AND  known > S*  # decided here, still open
#       deferred (embargo_idx) = decided later, still open (info only)
#   Each booking falls in exactly one step-wide S* window -> no double-count.
#   Graded on cancel-by-arrival  the same estimand for every model, so the
#   hazard-vs-static comparison is apples-to-apples. POOL the per-fold test
#   predictions into one large decision-aligned sample for AUC / AP / Brier / cost
#   (per-fold spread = stability). Procedure metric; the deployed model is refit
#   on ALL resolved data (training != production).
#
# `add_outcome_known_date` is also used by retraining to fit the deployment model
# on all data resolved "now" (train = outcome_known_date <= asof). PURE
# pandas/numpy (unit-testable).
# ---------------------------------------------------------------------------

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

# Canonical column names used across the project.
ARRIVAL: Final[str] = "arrival"
CREATED: Final[str] = "created"
STATUS: Final[str] = "status"
CANCEL_DAYS: Final[str] = "cancel_days_before_arrival"
CANCEL_TIME: Final[str] = "cancellationTime"
KNOWN_COL: Final[str] = "outcome_known_date"

# Target column, by preference. `is_canceled_by_arrival` is the readable alias
# notebook 00 writes alongside the historically-named `status` (which is a STRING
# in the raw cache but the encoded 0/1 target in the clean parquet  the
# project's most common stumbling block, hence the alias).
TARGET_CANDIDATES: Final[tuple[str, ...]] = ("is_canceled_by_arrival", "is_cancelled", "status")


def target_series(df: pd.DataFrame) -> pd.Series:
    """The 0/1 cancel-by-arrival target from CLEAN data, whatever it is called.
    THE one target accessor  training, hazard and eval all use it."""
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    raise KeyError(f"no target column found (looked for {TARGET_CANDIDATES})")


# =============================================================================
# 1. The point-in-time primitive: when did each booking's label become known?
# =============================================================================
def add_outcome_known_date(df: pd.DataFrame, *, out_col: str = KNOWN_COL) -> pd.DataFrame:
    """Add `outcome_known_date` = the date each booking's label became KNOWN.

        positive (cancelled before arrival): known at cancellationTime
        otherwise (arrived / no-show / post-arrival cancel): known at arrival

    Uses `cancellationTime` if present, else reconstructs it from
    `cancel_days_before_arrival` (= (arrival - cancellationTime) in days). This
    column is METADATA for splitting only - it is NEVER a model feature.
    """
    out = df.copy()
    arr = pd.to_datetime(out[ARRIVAL], utc=True, errors="coerce")
    known = arr.copy()

    if CANCEL_TIME in out.columns:
        ct = pd.to_datetime(out[CANCEL_TIME], utc=True, errors="coerce")
        is_pos = ct.notna() & (ct < arr)
        known = known.mask(is_pos, ct)
    elif CANCEL_DAYS in out.columns:
        cd = pd.to_numeric(out[CANCEL_DAYS], errors="coerce")
        is_pos = cd.notna() & (cd > 0)
        ct = arr - pd.to_timedelta(cd.clip(lower=0), unit="D")
        known = known.mask(is_pos, ct)
    # else: no cancel-timing available -> outcome_known_date == arrival for all.

    out[out_col] = known
    return out


# =============================================================================
# 2. Fold container
# =============================================================================
@dataclass
class Fold:
    """One walk-forward fold. Indices are POSITIONAL into the frame passed in."""
    k: int
    origin: pd.Timestamp           # decision-window start O_k
    next_origin: pd.Timestamp      # decision-window end O_k + step
    train_idx: np.ndarray
    test_idx: np.ndarray
    embargo_idx: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_train(self) -> int: return int(self.train_idx.size)
    @property
    def n_test(self) -> int:  return int(self.test_idx.size)
    @property
    def n_embargo(self) -> int: return int(self.embargo_idx.size)


# =============================================================================
# 3. Walk-forward fold generator  DECISION-TIME (each booking tested once)
# =============================================================================
def decision_date(df: pd.DataFrame, *, horizon_days: int = 14,
                  created_col: str = CREATED, arrival_col: str = ARRIVAL) -> pd.Series:
    """Per-booking DECISION date S* = max(created_day, arrival - H): the first day
    the booking is decidable (long-lead -> arrival-H; short-lead -> creation)."""
    created = pd.to_datetime(df[created_col], utc=True, errors="coerce").dt.normalize()
    arrival = pd.to_datetime(df[arrival_col], utc=True, errors="coerce")
    horizon_start = arrival - pd.Timedelta(days=horizon_days)
    return created.where(created >= horizon_start, horizon_start)


def make_folds(df: pd.DataFrame, *, n_folds: int = 6, horizon_days: int = 14,
               step_days: int = 14, scheme: str = "expanding",
               window_days: int | None = None, asof: str | pd.Timestamp | None = None,
               created_col: str = CREATED, arrival_col: str = ARRIVAL,
               known_col: str = KNOWN_COL) -> list[Fold]:
    """Decision-time walk-forward folds  each booking tested EXACTLY ONCE.

    Production scores a booking on every day it is open and within `horizon_days`
    of arrival. We test it ONCE, at its DECISION date S* = max(created, arrival-H)
    (see `decision_date`), if it is still open then (outcome_known_date > S*). Its
    evaluation horizon is d = min(lead, H). This keeps SHORT-LEAD bookings (booked
    < H days out) that a non-overlapping arrival grid would drop.

    Folds tile the S* axis in non-overlapping `step_days` windows (last ending at
    `asof`), so each booking lands in exactly one -> no double-count:

        train = outcome_known_date <= O_k                  # resolved before the window
        test  = O_k < S* <= O_k+step  AND  known > S*      # decided here, still open
        deferred (embargo_idx) = decided later, still open at O_k+step (info only)

    Graded on cancel-by-arrival. scheme="expanding" trains on all history <= O_k;
    "sliding" uses a fixed `window_days` look-back. Positional indices into `df`.
    """
    if known_col not in df.columns:
        raise KeyError(f"{known_col!r} missing - call add_outcome_known_date(df) first.")
    arrival = pd.to_datetime(df[arrival_col], utc=True, errors="coerce")
    created = pd.to_datetime(df[created_col], utc=True, errors="coerce")
    known = pd.to_datetime(df[known_col], utc=True, errors="coerce")
    asof_ts = pd.Timestamp(asof, tz="UTC") if asof is not None else pd.Timestamp(known.max())
    if scheme == "sliding" and window_days is None:
        window_days = step_days * n_folds

    s_star = decision_date(df, horizon_days=horizon_days,
                           created_col=created_col, arrival_col=arrival_col)
    step = pd.Timedelta(days=step_days)
    # Non-overlapping S* windows (O_k, O_k+step], the last ending at asof.
    origins = sorted(asof_ts - step * (i + 1) for i in range(n_folds))

    folds: list[Fold] = []
    for k, O in enumerate(origins):
        O_next = O + step
        train_mask = known <= O                                    # resolved before the window
        if scheme == "sliding":
            train_mask = train_mask & (known > O - pd.Timedelta(days=window_days))
        decided_here = (s_star > O) & (s_star <= O_next)
        test_mask = decided_here & (known > s_star)                # still open at its decision date
        deferred = (s_star > O_next) & (known > O_next) & (created <= O_next)
        folds.append(Fold(
            k=k, origin=O, next_origin=O_next,
            train_idx=np.where(train_mask.to_numpy())[0],
            test_idx=np.where(test_mask.to_numpy())[0],
            embargo_idx=np.where(deferred.to_numpy())[0],
            meta={"horizon_days": horizon_days},
        ))
    return folds


# =============================================================================
# 4. Summaries + point-in-time invariants (used in 00 and by the unit test)
# =============================================================================
def fold_summary(df: pd.DataFrame, folds: list[Fold], *, target_col: str = STATUS,
                 arrival_col: str = ARRIVAL, created_col: str = CREATED,
                 known_col: str = KNOWN_COL) -> pd.DataFrame:
    """Per-fold sizes, base rates, decision-window + test-horizon ranges (00 prints)."""
    y = pd.to_numeric(df[target_col], errors="coerce") if target_col in df else None
    arrival = pd.to_datetime(df[arrival_col], utc=True, errors="coerce")
    created = pd.to_datetime(df[created_col], utc=True, errors="coerce")
    lead = (arrival - created) / pd.Timedelta(days=1)
    rows = []
    for f in folds:
        H = int(f.meta.get("horizon_days", 14))
        def rate(idx):
            return float(y.iloc[idx].mean()) if (y is not None and idx.size) else float("nan")
        d = np.minimum(lead.iloc[f.test_idx], H) if f.n_test else None
        rows.append({
            "fold": f.k,
            "decision_win_start": pd.Timestamp(f.origin).date(),
            "decision_win_end": pd.Timestamp(f.next_origin).date(),
            "n_train": f.n_train, "n_test": f.n_test, "n_deferred": f.n_embargo,
            "train_pos": rate(f.train_idx), "test_pos": rate(f.test_idx),
            "test_horizon_med": float(d.median()) if f.n_test else None,
            "test_arr_min": arrival.iloc[f.test_idx].min().date() if f.n_test else None,
            "test_arr_max": arrival.iloc[f.test_idx].max().date() if f.n_test else None,
        })
    return pd.DataFrame(rows)


def assert_point_in_time(df: pd.DataFrame, folds: list[Fold], *,
                         created_col: str = CREATED, arrival_col: str = ARRIVAL,
                         known_col: str = KNOWN_COL) -> None:
    """Raise if any fold leaks the future (decision-time). Per fold window (O, O_next]:
        (1) every TRAIN label was known by O (outcome_known_date <= O),
        (2) every TEST booking was DECIDED inside (O, O_next] and still OPEN then
            (outcome_known_date > O -> disjoint from train),
        (3) train ∩ test = ∅.
    """
    known = pd.to_datetime(df[known_col], utc=True, errors="coerce")
    for f in folds:
        O = pd.Timestamp(f.origin); O_next = pd.Timestamp(f.next_origin)
        H = int(f.meta.get("horizon_days", 14))
        s_star = decision_date(df, horizon_days=H, created_col=created_col, arrival_col=arrival_col)
        if f.n_train:
            assert known.iloc[f.train_idx].max() <= O, f"fold {f.k}: train label known after O"
        if f.n_test:
            ss = s_star.iloc[f.test_idx]
            assert ss.min() > O and ss.max() <= O_next, f"fold {f.k}: test decided outside its window"
            assert known.iloc[f.test_idx].min() > O, f"fold {f.k}: test resolved by O (leak into train)"
        assert np.intersect1d(f.train_idx, f.test_idx).size == 0, f"fold {f.k}: train/test overlap"


# =============================================================================
# 5. Run-to-run data diff (00: "last run vs this run")
# =============================================================================
def run_summary(df: pd.DataFrame, *, target_col: str = STATUS, created_col: str = CREATED,
                arrival_col: str = ARRIVAL, label: str | None = None) -> dict:
    """Compact snapshot of the dataset for run-to-run comparison after a refresh."""
    created = pd.to_datetime(df[created_col], utc=True, errors="coerce")
    arrival = pd.to_datetime(df[arrival_col], utc=True, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce") if target_col in df else None
    return {
        "label": label,
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "rows": int(len(df)), "cols": int(df.shape[1]),
        "base_rate": (float(y.mean()) if y is not None else None),
        "positives": (int(y.sum()) if y is not None else None),
        "created_min": str(created.min()), "created_max": str(created.max()),
        "arrival_min": str(arrival.min()), "arrival_max": str(arrival.max()),
    }


def diff_summaries(prev: dict | None, curr: dict) -> pd.DataFrame:
    """Side-by-side last-run vs this-run table with deltas for numeric keys."""
    keys = ["rows", "cols", "positives", "base_rate",
            "created_min", "created_max", "arrival_min", "arrival_max"]
    rows = []
    for k in keys:
        pv = (prev or {}).get(k)
        cv = curr.get(k)
        delta = ""
        if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
            d = cv - pv
            delta = f"{d:+.4f}" if isinstance(cv, float) else f"{d:+d}"
        rows.append({"metric": k, "last_run": pv, "this_run": cv, "delta": delta})
    return pd.DataFrame(rows)
