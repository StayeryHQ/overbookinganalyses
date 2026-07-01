# ---------------------------------------------------------------------------
# src/walkforward.py
# Point-in-time temporal helpers — the SINGLE evaluation regime for every
# cancellation model (static classifiers AND the hazard model).
#
# Three timestamps, kept distinct (this is the thing that was previously muddled):
#   * created            = when a booking's FEATURES first exist (information time).
#                          NOT the overbooking decision — that is made near arrival.
#   * outcome_known_date = when the LABEL became known (cancellationTime for a
#                          pre-arrival cancel, else arrival). The only no-leakage
#                          rule is: train on outcomes known before the scoring date.
#   * arrival            = resolution.
#
# What the folds here measure — ARRIVAL-ANCHORED, production-faithful:
#   Production scores, on each date S, the bookings that ARRIVE within the next
#   H days. The folds mirror exactly that, at a sequence of scoring dates S_k:
#       train = outcome_known_date <= S_k          # everything resolved by S_k
#       test  = created<=S_k & S_k<arrival<=S_k+H  # the population scored at S_k
#               & outcome_known_date>S_k           # (still open at S_k)
#       deferred (embargo_idx) = active at S_k but arriving beyond H (info only)
#   Graded on cancel-before-arrival — the same estimand for every model, so the
#   hazard-vs-static comparison is apples-to-apples. Set step_days == H to tile
#   the timeline contiguously, then POOL the per-fold test predictions into one
#   large decision-aligned sample for AUC / AP / Brier / cost (per-fold spread =
#   stability). This is the procedure metric; the deployed model is refit on ALL
#   resolved data (training != production).
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
    origin: pd.Timestamp           # decision cutoff O_k
    next_origin: pd.Timestamp      # end of this fold's test block O_{k+1}
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
# 3. Walk-forward fold generator — ARRIVAL-ANCHORED (production-faithful)
# =============================================================================
def make_folds(df: pd.DataFrame, *, n_folds: int = 6, horizon_days: int = 14,
               step_days: int = 30, scheme: str = "expanding",
               window_days: int | None = None, asof: str | pd.Timestamp | None = None,
               created_col: str = CREATED, arrival_col: str = ARRIVAL,
               known_col: str = KNOWN_COL) -> list[Fold]:
    """Arrival-anchored walk-forward folds that mirror the production decision.

    We do NOT decide at booking creation; we decide overbooking for bookings
    ARRIVING in the next `horizon_days`. So at each scoring date S:

        train = outcome_known_date <= S                  # ALL bookings resolved by S
        test  = active at S AND arriving within horizon:
                created <= S  &  arrival > S  &  arrival <= S+H  &  known > S
        deferred (embargo_idx) = active at S but arriving beyond the horizon (info)

    `test` is exactly the population the desk would score at S (known to exist,
    not yet resolved, arriving inside the decision window); graded on
    cancel-before-arrival. Origins S are spaced `step_days` apart so the last
    horizon ends at `asof`. scheme="expanding" trains on all history <= S;
    "sliding" uses a fixed `window_days` look-back.

    Returns folds with POSITIONAL indices into `df` (use df.iloc[idx]).
    """
    if known_col not in df.columns:
        raise KeyError(f"{known_col!r} missing - call add_outcome_known_date(df) first.")
    created = pd.to_datetime(df[created_col], utc=True, errors="coerce")
    arrival = pd.to_datetime(df[arrival_col], utc=True, errors="coerce")
    known = pd.to_datetime(df[known_col], utc=True, errors="coerce")
    asof_ts = pd.Timestamp(asof, tz="UTC") if asof is not None else pd.Timestamp(known.max())
    H = pd.Timedelta(days=horizon_days)
    if scheme == "sliding" and window_days is None:
        window_days = step_days * n_folds

    # Scoring dates S, spaced step_days apart, last one so S+H == asof (gradeable).
    s_last = asof_ts - H
    origins = sorted(s_last - pd.Timedelta(days=step_days * i) for i in range(n_folds))

    folds: list[Fold] = []
    for k, S in enumerate(origins):
        train_known = known <= S                                   # ALL resolved by S
        if scheme == "sliding":
            train_mask = train_known & (known > S - pd.Timedelta(days=window_days))
        else:
            train_mask = train_known
        active = (created <= S) & (arrival > S) & (known > S)      # exists, unresolved at S
        test_mask = active & (arrival <= S + H)                    # arrives within the horizon
        deferred = active & (arrival > S + H)                      # active but beyond horizon
        folds.append(Fold(
            k=k, origin=S, next_origin=S + H,
            train_idx=np.where(train_mask.to_numpy())[0],
            test_idx=np.where(test_mask.to_numpy())[0],
            embargo_idx=np.where(deferred.to_numpy())[0],
        ))
    return folds


# =============================================================================
# 4. Summaries + point-in-time invariants (used in 00 and by the unit test)
# =============================================================================
def fold_summary(df: pd.DataFrame, folds: list[Fold], *, target_col: str = STATUS,
                 arrival_col: str = ARRIVAL, known_col: str = KNOWN_COL) -> pd.DataFrame:
    """Per-fold sizes, base rates, and arrival ranges - the table 00 prints/persists."""
    y = pd.to_numeric(df[target_col], errors="coerce") if target_col in df else None
    arrival = pd.to_datetime(df[arrival_col], utc=True, errors="coerce")
    rows = []
    for f in folds:
        def rate(idx):
            return float(y.iloc[idx].mean()) if (y is not None and idx.size) else float("nan")
        rows.append({
            "fold": f.k,
            "scoring_date_S": pd.Timestamp(f.origin).date(),
            "horizon_end": pd.Timestamp(f.next_origin).date(),
            "n_train": f.n_train, "n_test": f.n_test, "n_deferred": f.n_embargo,
            "train_pos": rate(f.train_idx), "test_pos": rate(f.test_idx),
            "test_arr_min": arrival.iloc[f.test_idx].min().date() if f.n_test else None,
            "test_arr_max": arrival.iloc[f.test_idx].max().date() if f.n_test else None,
        })
    return pd.DataFrame(rows)


def assert_point_in_time(df: pd.DataFrame, folds: list[Fold], *,
                         created_col: str = CREATED, arrival_col: str = ARRIVAL,
                         known_col: str = KNOWN_COL) -> None:
    """Raise if any fold leaks the future (arrival-anchored). Per fold at S:
        (1) every TRAIN label was known by S (outcome_known_date <= S),
        (2) every TEST booking was visible by S (created <= S),
        (3) every TEST booking was still future at S (arrival > S) and unresolved
            at S (outcome_known_date > S),
        (4) train ∩ test = ∅.
    """
    created = pd.to_datetime(df[created_col], utc=True, errors="coerce")
    arrival = pd.to_datetime(df[arrival_col], utc=True, errors="coerce")
    known = pd.to_datetime(df[known_col], utc=True, errors="coerce")
    for f in folds:
        S = pd.Timestamp(f.origin)
        if f.n_train:
            assert known.iloc[f.train_idx].max() <= S, f"fold {f.k}: train label known after S"
        if f.n_test:
            assert created.iloc[f.test_idx].max() <= S, f"fold {f.k}: test created after S"
            assert arrival.iloc[f.test_idx].min() > S, f"fold {f.k}: test arrival on/before S"
            assert known.iloc[f.test_idx].min() > S, f"fold {f.k}: test resolved before S (leak)"
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
