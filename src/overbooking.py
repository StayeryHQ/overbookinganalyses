# ---------------------------------------------------------------------------
# High-demand walk-cost scaling.
#
# The overbooking decision threshold is COST-BASED: flag a booking as a likely
# cancellation when its probability clears the cost-optimal threshold
# (src.scoring.operating_threshold / analytic_threshold). During a flagged
# high-demand period, walking a guest hurts more, so the walk cost is scaled up
# by a user-editable multiplier — this raises the effective walk cost, which in
# turn RAISES the threshold, making the flags more conservative.
#
# Pure math, no model dependency — trivially unit-testable.
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Final

# Default multiplier applied to the walk cost during a flagged high-demand period.
# Exposed in the UI as an editable value — this is only the pre-filled default.
DEFAULT_HIGH_DEMAND_MULTIPLIER: Final[float] = 1.5


def effective_walk_cost(cost_walk: float, high_demand: bool,
                        multiplier: float = DEFAULT_HIGH_DEMAND_MULTIPLIER) -> float:
    """Walk cost after the optional high-demand scaling (multiplier is user-editable).

    high_demand off -> walk cost unchanged. high_demand on -> walk cost * multiplier.
    """
    return float(cost_walk) * (float(multiplier) if high_demand else 1.0)
