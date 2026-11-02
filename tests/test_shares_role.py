"""Role-flip drops prior-year Synergy from the measured FG book."""

from __future__ import annotations

import pandas as pd
import pytest

from nba_sim.features.shares import _is_role_flip, _measured_with_prior
from nba_sim.sim.specs import PlayerModel


def _play_frame(season: str, name: str, fg: float, poss: float, gp: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "entity_type": "P",
                "entity_id": 9,
                "player_name": name,
                "side": "offensive",
                "play_type": "Spotup",
                "poss": poss,
                "ppp": 1.05,
                "tov_poss_pct": 0.1,
                "ft_poss_pct": 0.08,
                "fg_pct": fg,
                "gp": gp,
            }
        ]
    )


def test_role_flip_keeps_current_season_fg_only():
    plays = pd.concat(
        [
            _play_frame("2025-26", "Role Flip", 0.50, 40.0, 10.0),
            _play_frame("2024-25", "Role Flip", 0.30, 200.0, 70.0),
        ],
        ignore_index=True,
    )
    pooled = _measured_with_prior(plays, "Role Flip", "2025-26", role_flip=False)
    flipped = _measured_with_prior(plays, "Role Flip", "2025-26", role_flip=True)
    assert pooled["Spotup"]["fg"] < 0.45
    assert flipped["Spotup"]["fg"] == pytest.approx(0.50)


def test_role_flip_detects_depth_gap_and_thin_starters():
    depth = PlayerModel(
        player_id=1,
        name="A",
        team_id=1,
        position="G",
        minutes_mean=34.0,
        minutes_depth=22.0,
        minutes_sample=20,
    )
    thin = PlayerModel(
        player_id=2,
        name="B",
        team_id=1,
        position="G",
        minutes_mean=30.0,
        minutes_depth=30.0,
        minutes_sample=3,
    )
    steady = PlayerModel(
        player_id=3,
        name="C",
        team_id=1,
        position="G",
        minutes_mean=28.0,
        minutes_depth=27.0,
        minutes_sample=40,
    )
    assert _is_role_flip(depth)
    assert _is_role_flip(thin)
    assert not _is_role_flip(steady)
