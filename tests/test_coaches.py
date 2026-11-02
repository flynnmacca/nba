import numpy as np

from nba_sim.features.coaches import THIBODEAU, coach_for, depth_minutes, finish_window
from nba_sim.features.minutes import MinutePlayer, allocate_minutes, rotation_player_ids
from nba_sim.sim.clock import _fourth_marks, _reserves_enter, _script_cap, _scripted_lineup
from nba_sim.sim.engine import _minute_spread, _unit
from nba_sim.sim.specs import PlayerModel


def _regulars(minutes: list[float]) -> int:
    return sum(value >= 6.0 for value in minutes)


def test_every_coach_curve_sums_to_a_game_and_matches_his_depth():
    staff = [THIBODEAU, *[coach_for(abbr) for abbr in ("phi", "cle", "gsw", "nyk", "uta", "hou")]]
    for coach in staff:
        minutes = depth_minutes(coach, 15)
        assert abs(sum(minutes) - 240.0) < 1e-6
        assert all(value >= -1e-9 for value in minutes)
        assert _regulars(minutes) == coach.players
        assert minutes[coach.players] < 6.0


def test_thibodeau_plays_starters_and_kerr_plays_the_eleventh_man():
    short = depth_minutes(THIBODEAU, 15)
    long = depth_minutes(coach_for("gsw"), 15)
    assert float(np.mean(short[:5])) > float(np.mean(long[:5])) + 4.0
    assert short[10] < 3.0
    assert long[10] > 6.0
    assert _regulars(short) < _regulars(long)


def test_nurse_stops_before_atkinson_does():
    philadelphia = depth_minutes(coach_for("phi"), 15)
    cleveland = depth_minutes(coach_for("cle"), 15)
    assert philadelphia[9] < 6.0
    assert cleveland[9] >= 6.0
    assert float(np.mean(philadelphia[:5])) > float(np.mean(cleveland[:5]))


def test_an_eleventh_man_keeps_measured_minutes():
    players = []
    for index in range(12):
        minutes = 14.0 if index == 10 else (20.0 if index < 10 else 0.0)
        players.append(
            PlayerModel(
                player_id=index,
                name=f"P{index}",
                team_id=1,
                position="G",
                minutes_mean=minutes,
            )
        )
    chosen = rotation_player_ids(players)
    assert 10 in chosen
    assert 11 not in chosen
    rows = [
        MinutePlayer(player.player_id, player.minutes_mean if player.player_id in chosen else 0.0, 0.0, True)
        for player in players
    ]
    drawn = allocate_minutes(rows, n_sims=4, seed=1)
    assert float(drawn[:, 10].mean()) > 8.0
    assert np.all(drawn[:, 11] == 0)


def test_a_long_rotation_puts_the_eleventh_man_on_the_floor():
    players = [
        PlayerModel(player_id=index, name=f"P{index}", team_id=1, position="G", minutes_mean=20.0)
        for index in range(11)
    ]
    minutes = np.full(11, 20.0)
    bench = _unit(players, minutes, "bench")
    deep = _unit(players, minutes, "bench_deep")
    assert 10 in deep
    assert deep != bench

    short = players[:8]
    assert _unit(short, minutes[:8], "bench") == _unit(short, minutes[:8], "bench_deep")


def test_spot_minutes_stay_off_the_planned_lineup():
    players = [
        PlayerModel(player_id=index, name=f"P{index}", team_id=1, position="G", minutes_mean=30.0)
        for index in range(8)
    ]
    players.append(PlayerModel(player_id=8, name="Spot", team_id=1, position="G", minutes_mean=2.0))
    minutes = np.array([30.0] * 8 + [2.0])
    bench = _unit(players, minutes, "bench")
    assert 8 not in bench


def test_the_blowout_finish_is_the_lowest_minute_players():
    players = [
        PlayerModel(
            player_id=index,
            name=f"P{index}",
            team_id=1,
            position="G",
            minutes_mean=float(34 - index * 2),
        )
        for index in range(12)
    ]
    minutes = np.array([player.minutes_mean for player in players])
    garbage = _unit(players, minutes, "garbage")
    starters = _unit(players, minutes, "starters")
    assert garbage == [11, 10, 9, 8, 7]
    assert set(starters).isdisjoint(garbage)
    assert 0 not in garbage


def test_a_short_staff_hooks_his_closers_later_than_a_long_one():
    short_span, short_hook = finish_window(THIBODEAU)
    long_span, long_hook = finish_window(coach_for("gsw"))
    assert short_span < long_span
    assert short_hook > long_hook
    marks = _fourth_marks(long_span, short_span)
    assert marks[0] == 0.0
    assert marks[-1] == 720.0
    assert marks[1] == 720.0 - long_span
    margin = np.array([24.0, 10.0, -24.0])
    assert not _reserves_enter(0.0, long_span, margin, long_hook).any()
    opened = 720.0 - long_span
    long_sit = _reserves_enter(opened, long_span, margin, long_hook)
    short_sit = _reserves_enter(opened, short_span, margin, short_hook)
    assert bool(long_sit[0]) and bool(long_sit[2]) and not bool(long_sit[1])
    assert not short_sit.any()
    later = _reserves_enter(720.0 - short_span, short_span, margin, short_hook)
    assert bool(later[0]) and bool(later[2]) and not bool(later[1])


def _roster(rows: list[tuple[int, str, float]]) -> tuple[list[PlayerModel], np.ndarray]:
    players = [
        PlayerModel(player_id=pid, name=f"P{pid}", team_id=1, position=pos, minutes_mean=minutes, available=True)
        for pid, pos, minutes in rows
    ]
    return players, np.array([minutes for _pid, _pos, minutes in rows])


def test_a_center_keeps_a_starting_spot_when_the_guards_have_the_minutes():
    players, minutes = _roster(
        [
            (0, "G", 34.0),
            (1, "G", 32.0),
            (2, "G", 30.0),
            (3, "G", 28.0),
            (4, "G", 26.0),
            (5, "F", 20.0),
            (6, "C", 18.0),
        ]
    )
    starters = _unit(players, minutes, "starters")
    assert set(starters) == {0, 1, 2, 5, 6}


def test_the_group_that_played_together_starts_and_the_second_unit_is_apart():
    players, minutes = _roster(
        [(index, "G", 30.0) for index in range(10)]
    )
    groups = {
        (0, 1, 2, 3, 5): 800.0,
        (0, 1, 2, 3, 4): 100.0,
        (0, 1, 2, 6, 7): 350.0,
        (4, 6, 7, 8, 9): 280.0,
        (5, 6, 7, 8, 9): 40.0,
    }
    assert set(_unit(players, minutes, "starters", groups)) == {0, 1, 2, 3, 5}
    assert set(_unit(players, minutes, "bench", groups)) == {4, 6, 7, 8, 9}
    assert set(_unit(players, minutes, "mixed_early", groups)) == {0, 1, 2, 6, 7}
    assert 9 not in _unit(players, minutes, "starters", groups)


def test_a_short_unit_does_not_take_the_tip_from_the_lead_man():
    players, minutes = _roster(
        [
            (0, "G", 36.0),
            (1, "G", 30.0),
            (2, "F", 28.0),
            (3, "F", 24.0),
            (4, "C", 22.0),
            (5, "G", 16.0),
            (6, "F", 14.0),
            (7, "C", 12.0),
        ]
    )
    groups = {
        (5, 6, 7, 2, 3): 400.0,
        (0, 1, 2, 3, 4): 180.0,
    }
    starters = _unit(players, minutes, "starters", groups)
    assert 0 in starters
    assert 1 in starters
    assert set(starters) == {0, 1, 2, 3, 4}


def _closer_roster() -> tuple[list[PlayerModel], np.ndarray, np.ndarray]:
    minutes = [34.0, 32.0, 30.0, 28.0, 26.0, 16.0]
    players = [
        PlayerModel(player_id=index, name=f"P{index}", team_id=1, position="G", minutes_mean=value, available=True)
        for index, value in enumerate(minutes)
    ]
    means = np.array(minutes, dtype=float)
    return players, means, np.array([means])


def test_a_close_fourth_lets_a_closer_finish_the_stint():
    _players, means, sheet = _closer_roster()
    played = np.array([[34.0, 20.0, 18.0, 16.0, 14.0, 8.0]])
    cap = _script_cap(sheet, means, played, np.array([4.0]), 4, 16.0, 8.0, 12.0, 8.0)
    assert cap[0, 0] == 40.0
    assert cap[0, 5] == 16.0


def test_the_leader_sits_and_the_team_that_is_chasing_keeps_its_closer():
    _players, means, _sheet = _closer_roster()
    sheet = np.repeat(means[None, :], 2, axis=0)
    played = np.repeat(np.array([[30.0, 18.0, 16.0, 14.0, 12.0, 8.0]]), 2, axis=0)
    cap = _script_cap(sheet, means, played, np.array([20.0, -18.0]), 3, 15.0, 8.0, 12.0, 7.0)
    assert cap[0, 0] == 30.0
    assert cap[1, 0] == 37.0
    both = _script_cap(sheet, means, played, np.array([-30.0]), 4, 15.0, 8.0, 12.0, 8.0)
    assert both[0, 0] == 30.0


def test_the_first_half_keeps_the_draw_when_the_lead_is_already_safe():
    _players, means, sheet = _closer_roster()
    played = np.array([[20.0, 16.0, 14.0, 12.0, 10.0, 6.0]])
    cap = _script_cap(sheet, means, played, np.array([28.0]), 2, 15.0, 8.0, 30.0, 8.0)
    assert cap[0, 0] == 34.0


def test_a_long_staff_sits_late_in_a_blowout_and_a_short_staff_does_not():
    players, _means, sheet = _closer_roster()
    plan = np.zeros((1, 6))
    plan[0, 0] = 20.0 * 60.0
    unit = [0, 1, 2, 3, 4]
    # Twenty points at the start of the third is still early. The bar is the
    # hook plus three-quarters of a point for each minute before the window.
    early = np.array([20.0])
    still_in = _scripted_lineup(unit, players, plan, sheet, early, 3, "gsw", 420.0, False, 24.0)
    assert 0 in still_in[0]
    # Late in the third, 22 points clears a long staff and stays under a short one.
    lead = np.array([22.0])
    kept = _scripted_lineup(unit, players, plan, sheet, lead, 3, "nyk", 180.0, False, 15.0)
    sat = _scripted_lineup(unit, players, plan, sheet, lead, 3, "gsw", 180.0, False, 15.0)
    assert 0 in kept[0]
    assert 0 not in sat[0]
    assert 5 in sat[0]
    stuck = _scripted_lineup(unit, players, plan, sheet, lead, 4, "gsw", 240.0, True, 8.0)
    assert list(stuck[0]) == unit


def test_a_logged_minute_spread_is_his_and_a_chart_closer_stays_narrow():
    closer = PlayerModel(player_id=1, name="Star", team_id=1, position="G", minutes_mean=34.0, minutes_std=3.0)
    reserve = PlayerModel(player_id=2, name="End", team_id=1, position="G", minutes_mean=4.0, minutes_std=3.0)
    logged = PlayerModel(
        player_id=3,
        name="Logged",
        team_id=1,
        position="G",
        minutes_mean=34.0,
        minutes_std=6.0,
        minutes_sample=8,
    )
    assert _minute_spread(closer) < _minute_spread(reserve)
    assert 3.5 < _minute_spread(closer) < 5.0
    assert _minute_spread(logged) == 6.0
