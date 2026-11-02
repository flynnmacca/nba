from nba_sim.features.shrink import blend_ppp


def _blend(player_poss: float, league_ppp: float) -> float:
    return blend_ppp(
        player_ppp=1.30,
        player_poss=player_poss,
        team_ppp=1.00,
        team_poss=200,
        position_ppp=0.98,
        league_ppp=league_ppp,
        recent_mult=1.0,
        recent_games=0,
    )


def test_large_sample_stays_near_the_player_and_a_tiny_sample_does_not():
    large = _blend(500, 0.95)
    small = _blend(3, 0.95)
    assert large > 1.22
    assert small < 1.08


def test_league_prior_moves_a_small_sample_more_than_a_large_one():
    large_gap = abs(_blend(500, 1.10) - _blend(500, 0.95))
    small_gap = abs(_blend(3, 1.10) - _blend(3, 0.95))
    assert small_gap > large_gap
