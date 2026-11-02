"""Position priors for play types when a synergy sample is missing.

Shares are how often that spot runs the action. Rates are points per possession
before the team rating scales them.
"""

from __future__ import annotations

from nba_sim.constants import PLAY_TYPES

# Isolation, Transition, PRBallHandler, PRRollman, Postup, Spotup,
# Handoff, Cut, OffScreen, OffRebound, Misc
_FREQ = {
    "G": (0.14, 0.16, 0.30, 0.01, 0.01, 0.16, 0.08, 0.05, 0.05, 0.01, 0.03),
    "F": (0.10, 0.14, 0.08, 0.12, 0.08, 0.20, 0.06, 0.08, 0.06, 0.04, 0.04),
    "C": (0.03, 0.08, 0.02, 0.26, 0.18, 0.06, 0.02, 0.12, 0.02, 0.15, 0.06),
}
_OFF = {
    "G": (0.98, 1.20, 1.00, 0.90, 0.78, 1.08, 1.02, 1.15, 1.04, 0.90, 0.92),
    "F": (0.96, 1.16, 0.92, 1.10, 0.98, 1.04, 0.98, 1.20, 1.02, 1.10, 0.94),
    "C": (0.88, 1.10, 0.80, 1.16, 1.08, 0.96, 0.90, 1.24, 0.94, 1.22, 0.96),
}
# Points allowed. A lower number is a better defense of that action.
_DEF = {
    "G": (0.96, 1.12, 0.94, 1.16, 1.12, 1.00, 0.98, 1.18, 1.00, 1.20, 0.96),
    "F": (0.98, 1.14, 1.00, 1.08, 1.02, 1.02, 1.00, 1.16, 1.02, 1.10, 0.96),
    "C": (1.06, 1.18, 1.08, 1.00, 0.90, 1.08, 1.04, 1.10, 1.06, 1.00, 0.98),
}


def position_key(position: str) -> str:
    text = str(position or "").strip().upper()
    if text.startswith("C"):
        return "C"
    if text.startswith("G"):
        return "G"
    return "F"


def position_rates(position: str) -> dict[str, tuple[float, float, float]]:
    """Play type to (possession share, offensive PPP, points allowed)."""
    key = position_key(position)
    return {
        play: (_FREQ[key][index], _OFF[key][index], _DEF[key][index])
        for index, play in enumerate(PLAY_TYPES)
    }
