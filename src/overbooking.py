# ---------------------------------------------------------------------------
# src/overbooking.py
# Cost-optimal overbooking allowance from the model's per-booking cancel
# probabilities + the Revenue Manager's cost parameters.
#
# The decision is a classic NEWSVENDOR / critical-fractile problem, per arrival
# night, for one property:
#
#   * Let X = number of bookings that free up (cancel at/before arrival) for a
#     given arrival night. Under independence, X is Poisson-binomial with
#         mean  mu    = Σ p_i            (expected freed rooms)
#         var   sigma^2 = Σ p_i(1 - p_i) (see src.hazard.per_night_table)
#     We use the Normal approximation X ~ N(mu, sigma^2) (fine for the room counts
#     per night here; the app also surfaces the raw mu so the RM sees the basis).
#   * We choose an overbooking allowance b (rooms sold ABOVE physical capacity).
#       - If X > b  -> we under-overbooked: (X - b) rooms sit empty. Cost per room
#         = COST_EMPTY  (the "underage" cost Cu).
#       - If X < b  -> we over-overbooked: (b - X) guests get walked. Cost per
#         guest = COST_WALK (the "overage" cost Co).
#   * Newsvendor optimum: pick the smallest b with P(X <= b) >= critical ratio
#         cr = Cu / (Cu + Co) = COST_EMPTY / (COST_EMPTY + COST_WALK)
#     Normal approx: b* = mu + z * sigma,  z = Phi^-1(cr),  then round, floor at 0.
#
# Because walking a guest is normally MUCH more expensive than an empty room,
# cr is small, z is negative, and b* < mu -> the tool recommends overbooking
# CONSERVATIVELY (well below the expected number of cancellations). The
# "high-demand period" toggle scales COST_WALK up by a configurable multiplier,
# which pushes b* down further (walk even more cautiously when rooms are scarce).
#
# This module is pure numpy/pandas + math and is unit-testable without the model.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

# Default multiplier applied to the walk cost during a flagged high-demand period.
# Exposed in the UI as an editable value — this is only the pre-filled default.
DEFAULT_HIGH_DEMAND_MULTIPLIER: Final[float] = 1.5


# ---------------------------------------------------------------------------
# Inverse standard-normal CDF (Phi^-1). Uses scipy when present (it is, via
# scikit-learn), else Peter Acklam's rational approximation so this module never
# hard-depends on scipy.
# ---------------------------------------------------------------------------
def _norm_ppf(q: float) -> float:
    q = min(max(float(q), 1e-9), 1 - 1e-9)
    try:
        from scipy.special import ndtri  # type: ignore
        return float(ndtri(q))
    except Exception:  # noqa: BLE001 — scipy missing: fall back to Acklam (1985)
        a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
             1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
        b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
             6.680131188771972e+01, -1.328068155288572e+01]
        c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
             -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
        d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
             3.754408661907416e+00]
        plow, phigh = 0.02425, 1 - 0.02425
        if q < plow:
            t = math.sqrt(-2 * math.log(q))
            return (((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
                   ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
        if q > phigh:
            t = math.sqrt(-2 * math.log(1 - q))
            return -(((((c[0]*t+c[1])*t+c[2])*t+c[3])*t+c[4])*t+c[5]) / \
                    ((((d[0]*t+d[1])*t+d[2])*t+d[3])*t+1)
        t = q - 0.5
        r = t * t
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * t / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def critical_ratio(cost_empty: float, cost_walk: float) -> float:
    """Newsvendor critical ratio cr = Cu/(Cu+Co) = cost_empty/(cost_empty+cost_walk).

    Returns 0.0 if the costs are non-positive/degenerate (=> recommend no overbooking).
    """
    denom = float(cost_empty) + float(cost_walk)
    if denom <= 0:
        return 0.0
    return float(cost_empty) / denom


def effective_walk_cost(cost_walk: float, high_demand: bool,
                        multiplier: float = DEFAULT_HIGH_DEMAND_MULTIPLIER) -> float:
    """Walk cost after the optional high-demand scaling (multiplier is user-editable)."""
    return float(cost_walk) * (float(multiplier) if high_demand else 1.0)


def recommend_allowance(exp_freed: float | None, var_freed: float | None,
                        cost_empty: float | None, cost_walk: float | None) -> int | None:
    """Cost-optimal overbooking allowance b* for ONE arrival night.

    exp_freed / var_freed: mean & variance of freed rooms (Σp, Σp(1-p)).
    Returns an int >= 0, or None if any required input is missing (so the UI shows
    a blank recommendation rather than a fabricated number — e.g. before the RM
    enters the walk cost).
    """
    if exp_freed is None or cost_empty is None or cost_walk is None:
        return None
    cr = critical_ratio(cost_empty, cost_walk)
    mu = float(exp_freed)
    sigma = math.sqrt(max(float(var_freed or 0.0), 0.0))
    if cr <= 0:
        return 0
    b = mu if sigma == 0 else mu + _norm_ppf(cr) * sigma
    return int(max(0, round(b)))


def recommend_from_per_night(
    per_night: pd.DataFrame,
    cost_empty: float | None,
    cost_walk: float | None,
    *,
    high_demand: bool = False,
    high_demand_multiplier: float = DEFAULT_HIGH_DEMAND_MULTIPLIER,
    exp_col: str = "exp",
    var_col: str = "var",
) -> pd.DataFrame:
    """Add a `recommended_allowance` column to a per-arrival-night table.

    `per_night` must have expected-freed (`exp_col` = Σp) and variance
    (`var_col` = Σp(1-p)) columns — e.g. the output of src.hazard.per_night_table.
    Also records the effective walk cost used, so the UI can show what was applied.
    """
    out = per_night.copy()
    c_walk = None if cost_walk is None else effective_walk_cost(
        cost_walk, high_demand, high_demand_multiplier)
    out["cost_walk_effective"] = c_walk
    out["recommended_allowance"] = [
        recommend_allowance(e, v, cost_empty, c_walk)
        for e, v in zip(out[exp_col], out[var_col])
    ]
    return out


def summarize_property(per_night: pd.DataFrame,
                       reco_col: str = "recommended_allowance",
                       exp_col: str = "exp") -> dict:
    """Headline numbers for one property over the window: night count, mean expected
    freed rooms, and the median / max recommended allowance across nights.

    We surface the MEDIAN as the single "recommended allowance for this property"
    (a stable typical value the RM can apply per night) alongside the per-night
    detail, and the MAX as the peak-night figure. NOTE: aggregating a per-night
    decision into one property-level number is a presentation choice — confirm the
    preferred summary (median vs. max vs. per-night only) once you see it.
    """
    if per_night.empty:
        return {"nights": 0, "mean_exp_freed": 0.0, "median_reco": None, "max_reco": None}
    recos = per_night[reco_col].dropna()
    return {
        "nights": int(len(per_night)),
        "mean_exp_freed": float(per_night[exp_col].mean()),
        "median_reco": None if recos.empty else int(round(float(recos.median()))),
        "max_reco": None if recos.empty else int(recos.max()),
    }
