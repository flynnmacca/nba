"""The court at that moment. Synthetic histories, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nba_sim.features.court import (
    PlayerCourt,
    SquadFace,
    _events_before,
    _nearest,
    absorb,
    availability_mask,
    fit_court,
    regulation_minutes,
    repaired_level,
    select_lineup,
    squad_level,
    squad_mix,
    trip_price,
)
from nba_sim.sim.demo import make_team, rotation
from nba_sim.sim.engine import simulate_game
from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors, PlayerModel


def _guy(pid: int, pos: str = "G"):
    return PlayerModel(player_id=pid, name=f"P{pid}", team_id=1, position=pos, minutes_mean=20.0)


def _games(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _panel() -> tuple[list[PlayerModel], pd.DataFrame, pd.DataFrame]:
    roster = [
        _guy(1, "G"),
        _guy(2, "G"),
        _guy(3, "G"),
        _guy(4, "G"),
    ]
    rows = []
    ages = [
        {"player_id": 1, "age": 38},
        {"player_id": 2, "age": 24},
        {"player_id": 3, "age": 38},
        {"player_id": 4, "age": 38},
    ]
    for day in range(1, 31):
        rows.append({"date": f"2025-11-{day:02d}", "player_id": 1, "team_id": 1, "min": 34, "game_id": f"a{day}", "margin": 2})
        rows.append({"date": f"2025-11-{day:02d}", "player_id": 2, "team_id": 1, "min": 36, "game_id": f"a{day}", "margin": 2})
        rows.append({"date": f"2025-11-{day:02d}", "player_id": 3, "team_id": 1, "min": 22, "game_id": f"a{day}", "margin": 2})
    for day in range(1, 21):
        rows.append({"date": f"2024-11-{day:02d}", "player_id": 4, "team_id": 9, "min": 34, "game_id": f"b{day}", "margin": 2})
    rows.append({"date": "2025-11-01", "player_id": 4, "team_id": 1, "min": 10, "game_id": "c1", "margin": 2})
    rows.append({"date": "2025-11-03", "player_id": 4, "team_id": 1, "min": 10, "game_id": "c2", "margin": 2})
    rows.append({"date": "2025-12-01", "player_id": 1, "team_id": 1, "min": 4, "game_id": "future", "margin": 2})
    return roster, _games(rows), _games(ages)


def test_his_minutes_stay_his_and_a_future_game_does_not_move_them():
    roster, logs, ages = _panel()
    profiles, _whistle, _squads, _matrix = fit_court(logs, roster, on="2025-11-20", ages=ages, team_id=1, coach="A")
    assert profiles[1].minutes == pytest_approx(34)
    assert profiles[2].minutes > profiles[3].minutes
    assert profiles[2].minutes == pytest_approx(36)
    assert profiles[3].minutes == pytest_approx(22)
    assert profiles[1].position == profiles[3].position
    # Two games at 10 after a 34-minute other-team history is a role shift.
    assert profiles[4].minutes == pytest_approx(10)
    assert profiles[4].source == "His games"
    as_of, _w, _s, _m = fit_court(logs, roster, on="2025-11-20", ages=ages, team_id=1, coach="A")
    held_out = logs[logs["date"] < "2025-11-20"]
    without, _w, _s, _m = fit_court(held_out, roster, on="2025-11-20", ages=ages, team_id=1, coach="A")
    assert as_of[1].minutes == without[1].minutes
    assert as_of[1].minutes == pytest_approx(34)


def pytest_approx(value: float):
    return __import__("pytest").approx(value)


def test_a_short_sample_moves_toward_his_own_age_and_the_coach_keeps_his_split():
    roster, logs, ages = _panel()
    profiles, _w, _s, _m = fit_court(logs, roster, on="2025-11-20", ages=ages, team_id=1, coach="A")
    assert profiles[1].sample >= 19
    assert profiles[3].minutes < profiles[2].minutes
    assert profiles[4].source == "His games"
    assert profiles[1].source == "His games"


def test_a_star_with_a_short_sample_keeps_his_own_minutes():
    # Six games at 36 should not pool into the high twenties against a low peer prior.
    roster = [_guy(1, "C"), _guy(2, "G"), _guy(3, "G"), _guy(4, "G"), _guy(5, "G")]
    rows = []
    for day in range(1, 7):
        rows.append({"date": f"2025-11-{day:02d}", "player_id": 1, "team_id": 1, "min": 36, "game_id": f"s{day}", "margin": 2})
        for pid, minutes in ((2, 22), (3, 20), (4, 18), (5, 16)):
            rows.append(
                {"date": f"2025-11-{day:02d}", "player_id": pid, "team_id": 1, "min": minutes, "game_id": f"s{day}", "margin": 2}
            )
    ages = _games([{"player_id": pid, "age": 28} for pid in range(1, 6)])
    profiles, _w, _s, _m = fit_court(_games(rows), roster, on="2025-11-20", ages=ages, team_id=1, coach="A", panel_trust=40.0)
    assert profiles[1].minutes >= 32.0
    assert abs(profiles[1].minutes - 36.0) < abs(profiles[1].minutes - 22.0)


def test_a_one_minute_night_on_a_new_team_is_not_last_years_role():
    # Brooklyn end-of-bench minutes must not become a Wizards projection after one spot.
    roster = [_guy(1, "G"), _guy(2, "G"), _guy(3, "G")]
    rows = []
    for day in range(1, 21):
        rows.append({"date": f"2025-03-{day:02d}", "player_id": 1, "team_id": 1, "min": 18, "game_id": f"old{day}", "margin": 4})
        rows.append({"date": f"2025-03-{day:02d}", "player_id": 2, "team_id": 1, "min": 28, "game_id": f"old{day}", "margin": 4})
    rows.append({"date": "2025-10-23", "player_id": 1, "team_id": 2, "min": 1, "game_id": "new1", "margin": 12})
    rows.append({"date": "2025-10-23", "player_id": 3, "team_id": 2, "min": 30, "game_id": "new1", "margin": 12})
    ages = _games([{"player_id": 1, "age": 23}, {"player_id": 2, "age": 27}, {"player_id": 3, "age": 25}])
    profiles, _w, _s, _m = fit_court(_games(rows), roster, on="2025-10-25", ages=ages, team_id=2, coach="B")
    assert profiles[1].minutes == pytest_approx(1.0)
    assert profiles[1].sample == 1


def test_a_mid_band_demotion_on_a_new_team_keeps_this_teams_minutes():
    roster = [_guy(1, "G"), _guy(2, "G"), _guy(3, "G")]
    rows = []
    for day in range(1, 21):
        rows.append({"date": f"2025-03-{day:02d}", "player_id": 1, "team_id": 1, "min": 28, "game_id": f"old{day}", "margin": 4})
        rows.append({"date": f"2025-03-{day:02d}", "player_id": 2, "team_id": 1, "min": 24, "game_id": f"old{day}", "margin": 4})
    rows.append({"date": "2025-10-22", "player_id": 1, "team_id": 2, "min": 16, "game_id": "new1", "margin": 6})
    rows.append({"date": "2025-10-24", "player_id": 1, "team_id": 2, "min": 16, "game_id": "new2", "margin": 4})
    rows.append({"date": "2025-10-22", "player_id": 3, "team_id": 2, "min": 30, "game_id": "new1", "margin": 6})
    rows.append({"date": "2025-10-24", "player_id": 3, "team_id": 2, "min": 30, "game_id": "new2", "margin": 4})
    ages = _games([{"player_id": 1, "age": 26}, {"player_id": 2, "age": 27}, {"player_id": 3, "age": 25}])
    profiles, _w, _s, _m = fit_court(_games(rows), roster, on="2025-10-25", ages=ages, team_id=2, coach="B")
    assert profiles[1].minutes == pytest_approx(16.0)
    assert profiles[1].source == "His games"


def test_a_mid_band_promotion_on_a_new_team_keeps_this_teams_minutes():
    roster = [_guy(1, "G"), _guy(2, "G"), _guy(3, "G")]
    rows = []
    for day in range(1, 21):
        rows.append({"date": f"2025-03-{day:02d}", "player_id": 1, "team_id": 1, "min": 12, "game_id": f"old{day}", "margin": 4})
        rows.append({"date": f"2025-03-{day:02d}", "player_id": 2, "team_id": 1, "min": 28, "game_id": f"old{day}", "margin": 4})
    rows.append({"date": "2025-10-22", "player_id": 1, "team_id": 2, "min": 30, "game_id": "new1", "margin": 6})
    rows.append({"date": "2025-10-24", "player_id": 1, "team_id": 2, "min": 30, "game_id": "new2", "margin": 4})
    rows.append({"date": "2025-10-22", "player_id": 3, "team_id": 2, "min": 22, "game_id": "new1", "margin": 6})
    rows.append({"date": "2025-10-24", "player_id": 3, "team_id": 2, "min": 22, "game_id": "new2", "margin": 4})
    ages = _games([{"player_id": 1, "age": 24}, {"player_id": 2, "age": 27}, {"player_id": 3, "age": 25}])
    profiles, _w, _s, _m = fit_court(_games(rows), roster, on="2025-10-25", ages=ages, team_id=2, coach="B")
    assert profiles[1].minutes == pytest_approx(30.0)
    assert profiles[1].source == "His games"


def test_whistle_size_and_the_set_that_leaves_together():
    roster = [_guy(index, "G") for index in range(1, 8)]
    subs = []
    for game in range(12):
        for pid in (1, 2, 3):
            subs.append(
                {
                    "game_id": f"g{game}",
                    "period": 1,
                    "clock": "PT6M00.00S",
                    "player_out": pid,
                    "player_in": pid + 3,
                    "margin": 2,
                }
            )
    logs = _games(
        [{"date": "2025-11-01", "player_id": pid, "team_id": 1, "min": 30, "game_id": "g0", "margin": 2} for pid in range(1, 8)]
    )
    profiles, whistle, _squads, _matrix = fit_court(logs, roster, subs=_games(subs), coach="A", team_id=1)
    assert whistle.size_share.get(3, 0) > 0.9
    assert set(profiles[1].leaves_with) == {2, 3}
    assert 4 in profiles[1].replaced_by
    assert 1 in profiles[4].comes_in_for
    assert profiles[4].enter_share[1] == pytest_approx(1 / 3)


def test_the_same_player_keeps_a_different_clock_for_a_different_coach():
    roster = [_guy(1), _guy(2)]
    logs = _games(
        [
            {"date": "2025-11-01", "player_id": 1, "team_id": 1, "min": 32, "game_id": "g", "margin": 2},
            {"date": "2025-11-01", "player_id": 2, "team_id": 1, "min": 20, "game_id": "g", "margin": 2},
        ]
    )

    def subs(clock: str) -> pd.DataFrame:
        return _games(
            [
                {"game_id": "g", "period": 1, "clock": clock, "player_out": 1, "player_in": 2, "margin": 2},
            ]
        )

    first, _w, _s, _m = fit_court(logs, roster, subs=subs("PT8M00.00S"), coach="A", team_id=1)
    second, _w, _s, _m = fit_court(logs, roster, subs=subs("PT3M00.00S"), coach="B", team_id=1)
    assert first[1].sit_remaining == pytest_approx(8 * 60)
    assert second[1].sit_remaining == pytest_approx(3 * 60)
    assert first[1].clock_from_subs and second[1].clock_from_subs
    # Equal minutes owed. At 5:00 the earlier exit is the one preferred off.
    tied = []
    for pid, sit in ((1, 8 * 60), (2, 3 * 60), (3, 3 * 60), (4, 3 * 60), (5, 3 * 60), (6, 3 * 60)):
        player = _guy(pid)
        player.minutes_mean = 30
        player.available = True
        player.court = PlayerCourt(
            player_id=pid,
            minutes=30,
            minutes_close=30,
            sit_remaining=sit,
            clock_from_subs=True,
            starts=True,
            sample=20,
        )
        tied.append(player)
    five = select_lineup(tied, lead=2, period=1, seconds_left=5 * 60)
    assert 0 not in five
    assert 1 in five


def test_a_missing_center_lands_on_the_teammate_who_absorbed_him():
    roster = [_guy(1, "C"), _guy(2, "F"), _guy(3, "G")]
    rows = []
    for day in range(10):
        rows.append({"date": f"2025-11-{day+1:02d}", "player_id": 1, "team_id": 1, "min": 30, "game_id": f"in{day}", "margin": 2, "off_rating": 110})
        rows.append({"date": f"2025-11-{day+1:02d}", "player_id": 2, "team_id": 1, "min": 24, "game_id": f"in{day}", "margin": 2, "off_rating": 110})
        rows.append({"date": f"2025-11-{day+1:02d}", "player_id": 3, "team_id": 1, "min": 30, "game_id": f"in{day}", "margin": 2, "off_rating": 110})
    for day in range(10):
        rows.append({"date": f"2025-12-{day+1:02d}", "player_id": 1, "team_id": 1, "min": 0, "game_id": f"out{day}", "margin": 2, "off_rating": 118})
        rows.append({"date": f"2025-12-{day+1:02d}", "player_id": 2, "team_id": 1, "min": 36, "game_id": f"out{day}", "margin": 2, "off_rating": 118})
        rows.append({"date": f"2025-12-{day+1:02d}", "player_id": 3, "team_id": 1, "min": 30, "game_id": f"out{day}", "margin": 2, "off_rating": 118})
    profiles, whistle, _squads, matrix = fit_court(_games(rows), roster, team_id=1, coach="A")
    assert matrix[("F", "C")] > matrix[("G", "C")]
    assert whistle.absence_off["C"] > 4
    before = profiles[2].minutes
    absorb(profiles, [1], matrix)
    assert profiles[1].minutes == 0
    assert profiles[2].minutes > before


def test_usage_rises_when_the_high_usage_teammate_sits_and_the_mix_is_the_fives():
    ids = [1, 2, 3, 4, 5, 6]
    roster = [_guy(pid) for pid in ids]
    trips = []
    for _index in range(40):
        for pid, _share in ((1, 8), (2, 24), (3, 4), (4, 2), (5, 2)):
            for _copy in range(_share):
                trips.append({"player_id": pid, "lineup": "1-2-3-4-5", "action": "spot"})
    for _index in range(30):
        for pid, _share in ((1, 18), (3, 4), (4, 4), (5, 2), (6, 2)):
            for _copy in range(_share):
                trips.append({"player_id": pid, "lineup": "1-3-4-5-6", "action": "post" if pid == 1 else "spot"})
    thin = []
    for _index in range(3):
        thin.append({"player_id": 1, "lineup": "1-2-3-4-6", "action": "transition"})
    _profiles, _w, squads, _m = fit_court(pd.DataFrame(), roster, possessions=_games(trips + thin))
    with_him = trip_price([1, 2, 3, 4, 5], squads, {pid: 0.2 for pid in ids}, {"spot": 1.0})
    without = trip_price([1, 3, 4, 5, 6], squads, {pid: 0.2 for pid in ids}, {"spot": 1.0})
    assert without["usage"][1] > with_him["usage"][1]
    assert without["mix"].get("post", 0) > with_him["mix"].get("post", 0)
    borrowed = squad_mix([1, 2, 3, 4, 6], squads, {"spot": 0.5, "post": 0.5})
    # The thin five shares four with the posted five, so it borrows that mix.
    assert borrowed.get("post", 0) > 0.2
    face_ids = [1, 2, 3, 4, 5]
    lineups = _games([{"group_id": "1-2-3-4-5", "min": 80, "off_rating": 121, "def_rating": 108, "pace": 103, "oreb_pct": 0.31}])
    _p, _w, rated, _m = fit_court(pd.DataFrame(), roster, possessions=_games(trips), lineups=lineups)
    assert squad_level(face_ids, rated)[0] == pytest_approx(121)


def test_nearest_four_keeps_the_longer_five_and_the_earlier_tie():
    longer = SquadFace(minutes=40, mix={"post": 1.0}, possessions=40)
    shorter = SquadFace(minutes=12, mix={"spot": 1.0}, possessions=20)
    tied_first = SquadFace(minutes=12, mix={"cut": 1.0}, possessions=10)
    tied_later = SquadFace(minutes=12, mix={"handoff": 1.0}, possessions=10)
    squads = {
        frozenset({1, 2, 3, 4, 9}): shorter,
        frozenset({1, 2, 3, 4, 8}): longer,
        frozenset({9, 8, 7, 6, 5}): tied_first,
        frozenset({9, 8, 7, 6, 4}): tied_later,
    }
    face, overlap = _nearest(frozenset({1, 2, 3, 4, 6}), squads, 4)
    assert overlap == 4
    assert face is longer
    tied, tied_overlap = _nearest(frozenset({9, 8, 7, 6, 1}), squads, 4)
    assert tied_overlap == 4
    assert tied is tied_first


def test_a_repaired_five_moves_by_the_sub_and_a_foul_sit_reprices_the_next_trip():
    players = []
    for pid, impact in ((1, 1.0), (2, 1.0), (3, 1.0), (4, 1.0), (5, 1.25), (6, 0.8)):
        player = _guy(pid)
        player.off_impact = impact
        players.append(player)
    entered = repaired_level(players, [1, 2, 3, 4, 5], {})
    departed = repaired_level(players, [1, 2, 3, 4, 6], {})
    assert entered > departed
    squads = {}
    from nba_sim.features.court import SquadFace

    squads[frozenset({1, 2, 3, 4, 5})] = SquadFace(minutes=40, off_rating=120, def_rating=110, pace=100, possessions=40)
    squads[frozenset({1, 2, 3, 4, 6})] = SquadFace(minutes=40, off_rating=102, def_rating=110, pace=100, possessions=40)
    before = trip_price([1, 2, 3, 4, 5], squads)
    after = trip_price([1, 2, 3, 4, 6], squads)
    assert before["off"] == pytest_approx(120)
    assert after["off"] == pytest_approx(102)


def test_a_back_to_back_draw_removes_him_from_those_paths_only():
    player = _guy(7)
    player.court = PlayerCourt(player_id=7, minutes=32, rest_b2b=0.5, sample=20)
    player.available = True
    mask = availability_mask([player], 400, np.random.default_rng(3), b2b=True)
    assert mask.any() and not mask.all()
    home = availability_mask([player], 50, np.random.default_rng(3), b2b=False)
    assert home.all()
    rested = _guy(8)
    rested.court = PlayerCourt(player_id=8, minutes=28, play_rate=0.5, sample=20)
    rested.available = True
    drawn = availability_mask([rested], 400, np.random.default_rng(4))
    assert drawn.any() and not drawn.all()


def test_a_blowout_uses_the_decided_whistle_and_regulation_is_240():
    players = []
    for pid, minutes, starts in (
        (1, 34, True), (2, 32, True), (3, 30, True), (4, 28, True), (5, 26, True),
        (6, 18, False), (7, 16, False), (8, 14, False), (9, 12, False), (10, 10, False),
    ):
        player = _guy(pid, "G" if pid <= 3 else "F" if pid <= 7 else "C")
        player.minutes_mean = minutes
        player.available = True
        player.court = PlayerCourt(
            player_id=pid,
            minutes=minutes,
            minutes_close=minutes,
            minutes_blowout=minutes - 8 if starts else minutes,
            starts=starts,
            sit_remaining=180 if starts else 0,
            blowout_remaining=500 if starts else 720,
            comes_in_for=(pid - 5,) if not starts else (),
            enter_share={pid - 5: 1.0} if not starts else {},
            sample=20,
        )
        players.append(player)
    close = select_lineup(players, lead=2, period=1, seconds_left=600)
    assert 0 in close
    close_game = regulation_minutes(players, lead=2)
    blow_game = regulation_minutes(players, lead=20)
    assert sum(close_game.values()) == pytest_approx(240)
    assert sum(blow_game.values()) == pytest_approx(240)
    star = players[0].player_id
    target = players[0].court.minutes_blowout
    assert blow_game[star] < close_game[star] - 2
    assert abs(blow_game[star] - target) < abs(close_game[star] - target)
    fields = set(PlayerCourt.__dataclass_fields__)
    assert fields == set(players[0].court.__dataclass_fields__)
    assert fields == set(players[-1].court.__dataclass_fields__)


def test_no_player_is_named_and_a_profiled_path_sums_to_240():
    source = open(__import__("pathlib").Path(__file__).resolve().parents[1] / "src" / "nba_sim" / "features" / "court.py", encoding="utf-8").read().lower()
    assert "curry" not in source
    home = rotation(1, "Home", 100)
    away = rotation(2, "Away", 200)
    for side in (home, away):
        for player in side:
            minutes = float(player.minutes_mean)
            player.court = PlayerCourt(
                player_id=int(player.player_id),
                minutes=minutes,
                starts=minutes >= 24,
                sit_remaining=0 if minutes >= 36 else 720 - minutes / 4 * 60,
                sample=30,
                position="G",
            )
            player.available = True
    spec = GameSpec(
        game_id="court",
        date="2026-01-02",
        home=make_team("den"),
        away=make_team("cle"),
        home_players=home,
        away_players=away,
        context=GameContext(),
        league=LeaguePriors.flat(),
    )
    result = simulate_game(spec, n_sims=2, seed=1)
    home_sum = np.array([result.player_stats[player.player_id]["minutes"] for player in home]).sum(axis=0)
    assert np.all(np.abs(home_sum - 240) < 2) or np.all(np.abs(home_sum - 265) < 8)


def test_status_rates_shrink_a_thin_tag_toward_the_pooled_rate():
    from nba_sim.features.court import status_play_rates

    joined = _games(
        [{"status": "questionable", "played": 1 if index < 8 else 0} for index in range(10)]
        + [{"status": "doubtful", "played": 0}]
    )
    rates = status_play_rates(joined)
    assert 0.7 < rates["questionable"] < 0.85
    assert 0 < rates["doubtful"] < rates["questionable"]


def test_a_later_game_is_not_in_the_whistle_history():
    logs = pd.DataFrame({"game_id": ["0022500001"], "date": ["2025-10-21"]})
    events = pd.DataFrame(
        {
            "game_id": ["0022500001", "0022500999"],
            "player_out": [1, 2],
            "player_in": [6, 7],
        }
    )
    kept = _events_before(events, logs)
    assert list(kept["game_id"]) == ["0022500001"]
    assert list(kept["date"]) == ["2025-10-21"]


def test_an_injury_does_not_change_his_healthy_night_and_a_coach_rest_does():
    roster = [_guy(1), _guy(2)]
    dates = [f"2025-11-{day:02d}" for day in range(1, 13)]
    played = [
        {"date": day, "player_id": 1, "team_id": 1, "min": 34, "game_id": f"g{index}", "margin": 2}
        for index, day in enumerate(dates)
    ]
    played += [
        {"date": day, "player_id": 2, "team_id": 1, "min": 20, "game_id": f"g{index}", "margin": 2}
        for index, day in enumerate(dates)
    ]
    hurt = played + [
        {"date": "2025-11-13", "player_id": 1, "team_id": 1, "min": 0, "game_id": "inj", "reason": "Left knee", "margin": 2},
        {"date": "2025-11-14", "player_id": 1, "team_id": 1, "min": 0, "game_id": "inj2", "reason": "Left knee", "margin": 2},
    ]
    rested = played + [
        {"date": "2025-11-13", "player_id": 1, "team_id": 1, "min": 0, "game_id": "rest", "reason": "Coach's Decision", "margin": 2},
    ]
    healthy, _w, _s, _m = fit_court(_games(played), roster, participation=_games(hurt), team_id=1, coach="A")
    assert healthy[1].play_rate == pytest_approx(1)
    sat, _w, _s, _m = fit_court(_games(played), roster, participation=_games(rested), team_id=1, coach="A")
    assert 0.9 < sat[1].play_rate < 1
    named = played + [
        {"date": "2025-11-13", "player_id": 1, "team_id": 1, "min": 0, "game_id": "rest", "reason": "DND_REST", "margin": 2},
        {"date": "2025-11-14", "player_id": 1, "team_id": 1, "min": 0, "game_id": "off", "reason": "INACTIVE_COACH", "margin": 2},
    ]
    spared, _w, _s, _m = fit_court(_games(played), roster, participation=_games(named), team_id=1, coach="A")
    assert spared[1].play_rate == pytest_approx(1)
    scratches = played + [
        {"date": f"2025-11-{day:02d}", "player_id": 1, "team_id": 1, "min": 0, "game_id": f"cd{day}", "reason": "Coach's Decision", "margin": 2}
        for day in range(13, 19)
    ]
    often, _w, _s, _m = fit_court(_games(played), roster, participation=_games(scratches), team_id=1, coach="A")
    assert often[1].play_rate == pytest_approx(12 / 18)
    player = _guy(1)
    player.court = sat[1]
    player.available = True
    mask = availability_mask([player], 400, np.random.default_rng(5))
    assert mask.any() and not mask.all()


def test_an_unlabeled_zero_beside_an_out_is_the_same_absence():
    roster = [_guy(1), _guy(2)]
    days = [f"2025-11-{day:02d}" for day in range(1, 13)]
    played = []
    for index, day in enumerate(days):
        if day in {"2025-11-06", "2025-11-07", "2025-11-08", "2025-11-09", "2025-11-11"}:
            continue
        played.append({"date": day, "player_id": 1, "team_id": 1, "min": 30, "game_id": f"g{index}", "margin": 2})
        played.append({"date": day, "player_id": 2, "team_id": 1, "min": 20, "game_id": f"g{index}", "margin": 2})
    absence = played + [
        {"date": "2025-11-06", "player_id": 1, "team_id": 1, "min": 0, "game_id": "a", "reason": "Coach's Decision", "margin": 2},
        {"date": "2025-11-07", "player_id": 1, "team_id": 1, "min": 0, "game_id": "b", "reason": "Left knee", "margin": 2},
        {"date": "2025-11-08", "player_id": 1, "team_id": 1, "min": 0, "game_id": "c", "reason": "Coach's Decision", "margin": 2},
        {"date": "2025-11-09", "player_id": 1, "team_id": 1, "min": 0, "game_id": "d", "reason": "", "margin": 2},
        {"date": "2025-11-11", "player_id": 1, "team_id": 1, "min": 0, "game_id": "e", "reason": "Coach's Decision", "margin": 2},
    ]
    profiles, _w, _s, _m = fit_court(_games(played), roster, participation=_games(absence), team_id=1, coach="A")
    assert profiles[1].play_rate == pytest_approx(7 / 8)


def test_a_return_clock_keeps_the_exit_and_the_check_in():
    roster = [_guy(pid) for pid in (1, 6)]
    logs = _games(
        [
            {"date": "2025-11-01", "player_id": 1, "team_id": 1, "min": 34, "game_id": "g", "margin": 2},
            {"date": "2025-11-01", "player_id": 6, "team_id": 1, "min": 14, "game_id": "g", "margin": 2},
        ]
    )
    subs = _games(
        [
            {"game_id": "g", "period": 1, "clock": "PT8M00.00S", "player_out": 1, "player_in": 6, "margin": 2},
            {"game_id": "g", "period": 1, "clock": "PT5M00.00S", "player_out": 6, "player_in": 1, "margin": 2},
            {"game_id": "g", "period": 4, "clock": "PT2M00.00S", "player_out": 1, "player_in": 6, "margin": 20},
        ]
    )
    profiles, _w, _s, _m = fit_court(logs, roster, subs=subs, team_id=1, coach="A")
    assert profiles[1].sit_remaining == pytest_approx(8 * 60)
    assert profiles[1].return_remaining == pytest_approx(5 * 60)
    assert profiles[1].blowout_remaining == pytest_approx(2 * 60)


def test_the_sixth_man_plays_the_gap_and_the_starters_keep_theirs():
    """Planted minutes hold when the whistle gap is a different length."""
    planted = {1: 34, 2: 36, 3: 32, 4: 30, 5: 28, 6: 26, 7: 14, 8: 16, 9: 12, 10: 12}
    players = []
    for pid, minutes in planted.items():
        player = _guy(pid)
        player.minutes_mean = minutes
        player.available = True
        # His usual exit would leave him eight minutes if the clock were the lock.
        sit = 600 if pid == 1 else 0
        share = {}
        if pid == 6:
            share = {1: 0.7}
        elif pid == 7:
            share = {1: 0.3}
        player.court = PlayerCourt(
            player_id=pid,
            minutes=minutes,
            minutes_close=minutes,
            starts=pid <= 5,
            sit_remaining=sit,
            clock_from_subs=pid == 1,
            comes_in_for=(1,) if pid in (6, 7) else (),
            enter_share=share,
            leaves_with=(2, 3) if pid == 1 else (),
            sample=20,
        )
        players.append(player)
    got = regulation_minutes(players, lead=2)
    assert abs(got[1] - 34) < 3
    assert abs(got[6] - 26) < 3
    assert abs(got[7] - 14) < 3
    assert 4 < got[7] < 22
    assert abs(sum(got.values()) - 240) < 1
