"""Hierarchical shrinkage. League numbers are the last prior."""

from __future__ import annotations

import math


# Possessions of a broader number that sit beside a more specific one.
# The league is last. A team file, a player's own actions, and a head-to-head
# sample each pass it once they have a real look.
LEAGUE_PRIOR = 16.0
FILE_PRIOR = 24.0
PLAYER_PRIOR = 18.0
MATCHUP_PRIOR = 8.0


def shrink(value: float, sample_size: float, prior: float, prior_strength: float) -> float:
    """Precision-weighted mean. A non-positive sample returns the prior."""
    if sample_size is None or not math.isfinite(sample_size) or sample_size <= 0:
        return float(prior)
    if value is None or not math.isfinite(value):
        return float(prior)
    weight = float(sample_size)
    strength = max(0.0, float(prior_strength))
    if strength == 0:
        return float(value)
    return (weight * float(value) + strength * float(prior)) / (weight + strength)


def blend_ppp(
    player_ppp: float,
    player_poss: float,
    team_ppp: float,
    team_poss: float,
    position_ppp: float,
    league_ppp: float,
    recent_mult: float = 1.0,
    recent_games: float = 0.0,
    prev_ppp: float | None = None,
    prev_poss: float = 0.0,
) -> float:
    """This season, last season, the team, the position, then the league.

    ``recent_games`` of 0 leaves the form multiplier at 1. A missing previous
    season falls through to the team. The league is only the position's prior,
    and a short team file still outweighs it. A new season is a change of role,
    so this year's possessions and the recent games count more than the prior.
    """
    position = shrink(position_ppp, 60.0, league_ppp, LEAGUE_PRIOR)
    team = shrink(team_ppp, team_poss, position, FILE_PRIOR)
    if prev_ppp is not None and prev_poss > 0:
        anchor = shrink(prev_ppp, prev_poss, team, FILE_PRIOR)
    else:
        anchor = team
    season = shrink(player_ppp, player_poss, anchor, PLAYER_PRIOR)
    # Early-season form samples are thin; pull the recent multiplier harder
    # toward 1.0 until eight games are on the board.
    recent_prior = 16.0 if recent_games < 8.0 else 8.0
    multiplier = shrink(recent_mult, recent_games, 1.0, recent_prior)
    return season * multiplier
