"""Stake sizing. A non-positive edge or a swallowed interval stakes zero."""

from __future__ import annotations

import math

import numpy as np

from nba_sim.config import SETTINGS
from nba_sim.markets.devig import american_to_decimal


def interval_contains(model_p: float, market_p: float, n_sims: int, z: float | None = None) -> bool:
    z = SETTINGS.ci_z if z is None else z
    prob = float(np.clip(model_p, 1e-6, 1 - 1e-6))
    se = math.sqrt(prob * (1.0 - prob) / max(int(n_sims), 1))
    return (prob - z * se) <= market_p <= (prob + z * se)


def kelly_stake(
    prob: float,
    american: float,
    fraction: float | None = None,
    cap: float | None = None,
) -> float:
    fraction = SETTINGS.kelly_fraction if fraction is None else fraction
    cap = SETTINGS.kelly_cap if cap is None else cap
    prob = float(np.clip(prob, 1e-6, 1.0 - 1e-6))
    b = american_to_decimal(american) - 1.0
    if b <= 0:
        return 0.0
    full = (b * prob - (1.0 - prob)) / b
    if full <= 0:
        return 0.0
    return float(min(cap, full * fraction))
