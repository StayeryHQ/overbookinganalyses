# ---------------------------------------------------------------------------
# Cost-optimal overbooking allowance — the classic NEWSVENDOR problem, per
# arrival night and property.
#
#   X = rooms freed by cancellations that night. With per-booking cancel
#   probabilities p_i (independence assumed): mean mu = Σ p_i, variance
#   sigma² = Σ p_i(1-p_i) — Poisson-binomial, Normal approximation is fine at
#   these counts. Choose the allowance b (rooms sold above capacity):
#     X > b -> rooms sit empty,  cost COST_EMPTY each ("underage" Cu)
#     X < b -> guests get walked, cost COST_WALK each ("overage"  Co)
#   Optimum: smallest b with P(X <= b) >= cr, where cr = Cu / (Cu + Co).
#   Normal approx: b* = mu + z·sigma, z = Phi^-1(cr); round, floor at 0.
#
# Walking costs far more than an empty room, so cr is small, z negative and b*
# sits WELL BELOW the expected cancellations — deliberately conservative. The
# high-demand toggle scales COST_WALK up, pushing b* down further.
# Pure numpy/math — unit-testable without the models.
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pandas as pd

# Default multiplier applied to the walk cost during a flagged high-demand period.
# Exposed in the UI as an editable value — this is only the pre-filled default.
DEFAULT_HIGH_DEMAND_MULTIPLIER: Final[float] = 1.5


def _norm_ppf(q: float) -> float:
    """Inverse standard-normal CDF (Phi^-1), clamped away from 0/1.
    scipy is always installed (scikit-learn depends on it), so no fallback."""
    from scipy.special import ndtri
    q = min(max(float(q), 1e-9), 1 - 1e-9)
    return float(ndtri(q))


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
