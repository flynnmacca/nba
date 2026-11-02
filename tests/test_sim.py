import numpy as np

from nba_sim.sim.adjust import oreb_probability
from nba_sim.sim.demo import build_demo_slate, make_player, oreb_matchup, pnr_matchup
from nba_sim.sim.engine import simulate_game


def test_same_seed_replays_the_game():
    spec = build_demo_slate()[0]
    first = simulate_game(spec, n_sims=200, seed=42)
    second = simulate_game(spec, n_sims=200, seed=42)
    other = simulate_game(spec, n_sims=200, seed=43)
    np.testing.assert_array_equal(first.home_points, second.home_points)
    assert not np.array_equal(first.home_points, other.home_points)


def test_team_points_are_the_sum_of_player_points_and_minutes_fill_the_game():
    spec = build_demo_slate()[0]
    result = simulate_game(spec, n_sims=250, seed=5)
    home = np.vstack([result.player_stats[pid]["pts"] for pid in result.home_ids]).sum(axis=0)
    away = np.vstack([result.player_stats[pid]["pts"] for pid in result.away_ids]).sum(axis=0)
    np.testing.assert_allclose(home, result.home_points)
    np.testing.assert_allclose(away, result.away_points)
    for pid in result.home_ids:
        minutes = result.player_stats[pid]["minutes"]
        assert minutes.shape == (250,)
    home_minutes = np.vstack(
        [result.player_stats[pid]["minutes"] for pid in result.home_ids]
    ).sum(axis=0)
    np.testing.assert_allclose(home_minutes, 240.0, atol=1e-6)
    assert 95 < float(result.home_points.mean()) < 145
    assert 95 < float(result.away_points.mean()) < 145


def test_handler_scores_more_against_a_weak_pick_and_roll_defense():
    weak = simulate_game(pnr_matchup("weak"), n_sims=700, seed=11)
    strong = simulate_game(pnr_matchup("strong"), n_sims=700, seed=11)
    handler = 1000
    weak_points = float(weak.player_stats[handler]["pts"].mean())
    strong_points = float(strong.player_stats[handler]["pts"].mean())
    assert weak_points > strong_points + 2.0


def test_box_score_counts_add_up_to_the_points():
    spec = build_demo_slate()[0]
    result = simulate_game(spec, n_sims=60, seed=4)
    for pid in result.home_ids + result.away_ids:
        stats = result.player_stats[pid]
        assert np.all(stats["fgm"] <= stats["fga"] + 1e-6)
        assert np.all(stats["tpm"] <= stats["tpa"] + 1e-6)
        assert np.all(stats["tpm"] <= stats["fgm"] + 1e-6)
        assert np.all(stats["ftm"] <= stats["fta"] + 1e-6)
        assert np.all(stats["ast"] <= stats["pot_ast"] + 1e-6)
        np.testing.assert_allclose(stats["reb"], stats["oreb"] + stats["dreb"])
        points = 2.0 * (stats["fgm"] - stats["tpm"]) + 3.0 * stats["tpm"] + stats["ftm"]
        np.testing.assert_allclose(points, stats["pts"])
    home_pf = sum(float(result.player_stats[pid]["pf"].mean()) for pid in result.home_ids)
    assert home_pf > 8.0


def test_a_thin_defensive_rebound_sample_does_not_invent_putbacks():
    offense = [
        make_player(index, f"O{index}", 1, "F", 24, "Spotup", 8, 1.02, oreb_pct=0.04, dreb_pct=0.14)
        for index in range(5)
    ]
    thin = [
        make_player(10 + index, f"D{index}", 2, "F", 24, "Spotup", 8, 1.02, oreb_pct=0.02, dreb_pct=0.04)
        for index in range(5)
    ]
    solid = [
        make_player(20 + index, f"S{index}", 2, "F", 24, "Spotup", 8, 1.02, oreb_pct=0.02, dreb_pct=0.16)
        for index in range(5)
    ]
    leaked = oreb_probability(offense, thin)
    normal = oreb_probability(offense, solid)
    assert leaked < 0.34
    assert normal < leaked


def test_high_oreb_unit_creates_more_second_chance_points():
    high = simulate_game(oreb_matchup("high"), n_sims=700, seed=13)
    low = simulate_game(oreb_matchup("low"), n_sims=700, seed=13)
    assert float(high.second_chance_home.mean()) > float(low.second_chance_home.mean()) + 3.0


def test_team_rates_move_the_glass_and_the_players_on_the_floor_still_lead():
    crashers = [
        make_player(index, f"C{index}", 1, "F", 24, "Spotup", 8, 1.02, oreb_pct=0.08, dreb_pct=0.14)
        for index in range(5)
    ]
    guards = [
        make_player(10 + index, f"G{index}", 1, "G", 24, "Spotup", 8, 1.02, oreb_pct=0.02, dreb_pct=0.14)
        for index in range(5)
    ]
    glass = [
        make_player(20 + index, f"S{index}", 2, "C", 24, "Spotup", 8, 1.02, oreb_pct=0.04, dreb_pct=0.16)
        for index in range(5)
    ]
    soft = [
        make_player(30 + index, f"W{index}", 2, "G", 24, "Spotup", 8, 1.02, oreb_pct=0.04, dreb_pct=0.08)
        for index in range(5)
    ]
    players = oreb_probability(crashers, glass)
    higher_team = oreb_probability(crashers, glass, off_team=0.48, def_team=0.80, def_allowed=0.20)
    lower_team = oreb_probability(crashers, glass, off_team=0.22, def_team=0.80, def_allowed=0.20)
    assert higher_team > players
    assert lower_team < players
    same_team_crash = oreb_probability(crashers, glass, 0.30, 0.70, 0.30)
    same_team_guard = oreb_probability(guards, glass, 0.30, 0.70, 0.30)
    assert same_team_crash > same_team_guard + 0.04
    soft_floor = oreb_probability(crashers, soft, 0.28, 0.72, 0.28)
    solid_floor = oreb_probability(crashers, glass, 0.28, 0.72, 0.28)
    loose = oreb_probability(crashers, glass, 0.28, 0.72, 0.34)
    stingy = oreb_probability(crashers, glass, 0.28, 0.72, 0.24)
    assert soft_floor > solid_floor
    assert loose > stingy
    assert (soft_floor - solid_floor) > (loose - stingy)


def test_a_short_opponent_rebound_sample_leans_on_last_season():
    import pandas as pd

    from nba_sim.features.prepare import _oreb_allowed

    season = _oreb_allowed(pd.Series({"opp_oreb_pct": 0.33, "gp": 40}), None)
    assert abs(season - 0.33) < 1e-6
    short = _oreb_allowed(
        pd.Series({"opp_oreb_pct": 0.48, "gp": 2}),
        pd.Series({"opp_oreb_pct": 0.30, "gp": 82}),
    )
    assert 0.30 < short < 0.36
