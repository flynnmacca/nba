"""Early foul trouble sits a player, and the minute sheet follows the floor."""

import numpy as np

from nba_sim.features.coaches import THIBODEAU, coach_for
from nba_sim.sim.clock import _replacement, _side, should_sit
from nba_sim.sim.demo import make_team, rotation
from nba_sim.sim.engine import simulate_game
from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors, PlayerModel


def _row(period, seconds, fouls, margin, patience, foul_rate, is_star):
    return bool(
        should_sit(
            period,
            np.array([seconds], dtype=float),
            np.array([fouls], dtype=float),
            np.array([margin], dtype=float),
            patience,
            np.array([foul_rate], dtype=float),
            np.array([is_star]),
        )[0]
    )


def test_the_sit_line_is_q_plus_one_and_the_coach_moves_it():
    assert _row(1, 8 * 60, 2, 0, 0.30, 1.0, False)
    assert not _row(1, 8 * 60, 2, 0, 0.45, 1.0, True)
    assert _row(1, 8 * 60, 3, 0, 0.45, 1.0, True)
    assert _row(1, 8 * 60, 2, 0, 0.22, 1.0, True)
    assert not _row(1, 8 * 60, 2, 0, 0.72, 1.0, True)
    assert not _row(1, 8 * 60, 1, 0, 0.30, 1.0, False)
    assert _row(2, 6 * 60, 3, 0, 0.45, 1.0, False)
    assert not _row(4, 8 * 60, 5, 3, 0.30, 1.0, True)
    assert _row(4, 8 * 60, 5, 15, 0.30, 1.0, False)
    assert not _row(4, 8 * 60, 5, 15, 0.72, 1.0, True)


def test_a_foul_prone_player_sits_deeper_into_the_quarter():
    # Standard window is 309 seconds. A non-star who already fouls a lot uses
    # 65% of that, so he is still out at 3:40 while a cleaner teammate is back in.
    assert _row(1, 220, 2, 0, 0.45, 1.40, False)
    assert not _row(1, 220, 2, 0, 0.45, 0.80, False)


def test_short_staffs_wait_longer_than_deep_ones():
    assert coach_for("uta").foul_patience < coach_for("gsw").foul_patience < 0.30
    assert coach_for("phi").foul_patience < coach_for("nyk").foul_patience
    assert coach_for("nyk").foul_patience < THIBODEAU.foul_patience


def _game(patience: float, foul_rate: float) -> GameSpec:
    home = make_team("phi")
    away = make_team("cle")
    home.foul_patience = patience
    away.foul_patience = 0.45
    home_players = rotation(home.team_id, "PHI", 100, handler_ppp=1.05, handler_poss=40.0)
    away_players = rotation(away.team_id, "CLE", 200, handler_ppp=1.05, handler_poss=40.0)
    home_players[0].foul_rate = foul_rate
    home_players[0].minutes_std = 0.3
    return GameSpec(
        game_id=f"foul-{patience}-{foul_rate}",
        date="2026-10-22",
        home=home,
        away=away,
        home_players=home_players,
        away_players=away_players,
        context=GameContext(),
        league=LeaguePriors.flat(1.02),
        source="scenario",
    )


def _minutes(patience: float, foul_rate: float, seed: int = 7, n_sims: int = 280) -> np.ndarray:
    result = simulate_game(_game(patience, foul_rate), n_sims=n_sims, seed=seed)
    home = np.vstack([result.player_stats[pid]["minutes"] for pid in result.home_ids])
    np.testing.assert_allclose(home.sum(axis=0), 240.0, atol=1e-6)
    return home.mean(axis=1)


def test_early_fouls_take_minutes_off_the_player_who_picks_them_up():
    clean = _minutes(0.40, 0.40)
    prone = _minutes(0.40, 4.0)
    assert prone[0] < clean[0] - 1.5
    assert prone[0] > 8.0


def test_a_loose_staff_plays_a_star_through_foul_trouble():
    # Around four or five fouls, so early sits still change the night. A rate
    # that fouls him out every game takes the coach's choice away at the sixth.
    strict = _minutes(0.22, 1.4, seed=11, n_sims=450)
    loose = _minutes(0.85, 1.4, seed=11, n_sims=450)
    assert loose[0] > strict[0] + 0.5


def test_foul_trouble_brings_the_same_position():
    players = [
        PlayerModel(player_id=0, name="Guard", team_id=1, position="G", minutes_mean=34.0, available=True),
        PlayerModel(player_id=1, name="Big", team_id=1, position="C", minutes_mean=22.0, available=True),
        PlayerModel(player_id=2, name="Backup", team_id=1, position="G", minutes_mean=16.0, available=True),
    ]
    side = _side(players, np.ones((3, 1)))
    fouls = np.zeros(3)
    assert _replacement(side, {0}, fouls, 0, 1, 400.0, 0.0, 0.45, False) == 2
    same = [
        PlayerModel(player_id=index, name=f"G{index}", team_id=1, position="G", minutes_mean=minutes, available=True)
        for index, minutes in enumerate((34.0, 22.0, 16.0))
    ]
    same_side = _side(same, np.ones((3, 1)))
    assert _replacement(same_side, {0}, fouls, 0, 1, 400.0, 0.0, 0.45, False) == 1


def test_a_bench_foul_does_not_bring_the_star_off_the_bench():
    clean = _game(0.30, 0.40)
    dirty = _game(0.30, 0.40)
    dirty.home_players[-1].foul_rate = 4.0
    clean_star = simulate_game(clean, n_sims=220, seed=19).player_stats[100]["minutes"].mean()
    dirty_star = simulate_game(dirty, n_sims=220, seed=19).player_stats[100]["minutes"].mean()
    # A foul-out still needs a body, so the lead guard can take a short emergency
    # stint. He does not inherit the bench role.
    assert dirty_star < clean_star + 3.0
