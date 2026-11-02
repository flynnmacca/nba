"""American odds, power-method devig, and fractional Kelly."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


def american_to_implied(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def american_to_decimal(odds: float) -> float:
    odds = float(odds)
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def power_devig(implied: np.ndarray) -> np.ndarray:
    """Find k such that the implied probabilities raised to k sum to 1."""
    prices = np.asarray(implied, dtype=float)
    if prices.ndim != 1 or prices.size < 2:
        raise ValueError("devig needs at least two prices")
    if np.any(prices <= 0) or np.any(prices >= 1):
        raise ValueError("implied probabilities must be inside (0, 1)")

    def gap(exponent: float) -> float:
        return float(np.sum(prices**exponent) - 1.0)

    if abs(gap(1.0)) < 1e-10:
        return prices.copy()
    if gap(1.0) > 0:
        exponent = brentq(gap, 1.0, 40.0)
    else:
        exponent = brentq(gap, 1e-3, 1.0)
    fair = prices**exponent
    return fair / fair.sum()


def devig_american(odds: list[float] | np.ndarray) -> np.ndarray:
    implied = np.array([american_to_implied(value) for value in odds], dtype=float)
    return power_devig(implied)
