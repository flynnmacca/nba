import numpy as np
import pytest

from nba_sim.constants import ASSIST_RATE, PLAY_TYPES
from nba_sim.sim.clock import (
    BONUS_FOULS,
    COMMON,
    INTENTIONAL,
    OFFENSIVE,
    OFFREB,
    OT_BONUS_FOULS,
    SHOT,
    FLAGRANT,
    TAKE,
    _draw_side,
    _eject,
    _fit_pace_impacts,
    _live_board,
    _pace,
    _unit_pace,
    _pass_chance,
    _second_chance_poss,
    _on_floor_oreb,
    _side,
    bonus_on,
    free_throw_award,
    trailing_mask,
)
from nba_sim.sim.specs import PlaySkill
from nba_sim.sim.demo import make_player, make_team, rotation
from nba_sim.sim.engine import simulate_game
from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors


def _award(kind, *, three=False, made=False, fouls=0, bonus=BONUS_FOULS, seconds=40.0):
    fta, team_foul, keep, live = free_throw_award(
        np.array([kind]),
        np.array([three]),
        np.array([made]),
        np.array([fouls], dtype=float),
        bonus,
        np.array([seconds], dtype=float),
    )
    return int(fta[0]), bool(team_foul[0]), bool(keep[0]), bool(live[0])


def test_bonus_and_free_throw_counts_follow_the_whistle():
    assert bonus_on(1) == BONUS_FOULS
    assert bonus_on(4) == BONUS_FOULS
    assert bonus_on(5) == OT_BONUS_FOULS
    assert _award(SHOT, made=False) == (2, True, False, True)
    assert _award(SHOT, three=True, made=False) == (3, True, False, True)
    assert _award(SHOT, made=True) == (1, True, False, True)
    assert _award(SHOT, made=True, seconds=0.0) == (1, True, False, False)
    assert _award(COMMON, fouls=3) == (0, True, True, False)
    assert _award(COMMON, fouls=4) == (2, True, False, True)
    assert _award(TAKE, fouls=1) == (1, True, True, False)
    assert _award(TAKE, fouls=4) == (1, True, True, False)
    # A bonus reach-in that is not a late take ends the trip: two shots, no keep.
    assert _award(COMMON, fouls=4) == (2, True, False, True)
    assert _award(OFFENSIVE, fouls=6) == (0, False, False, False)
    assert _award(INTENTIONAL, fouls=2, seconds=8) == (1, True, True, False)
    assert _award(INTENTIONAL, fouls=4, seconds=8) == (2, True, False, True)
    assert _award(COMMON, fouls=2, bonus=OT_BONUS_FOULS) == (0, True, True, False)
    assert _award(COMMON, fouls=3, bonus=OT_BONUS_FOULS) == (2, True, False, True)
    assert _award(FLAGRANT, fouls=1) == (2, True, True, False)
    assert _award(FLAGRANT, three=True, made=True) == (2, True, True, False)


def test_trailing_team_fouls_only_late_and_only_when_it_helps():
    def mask(period, seconds, deficit, fouls, bonus=BONUS_FOULS):
        return bool(
            trailing_mask(
                period,
                np.array([seconds], dtype=float),
                np.array([deficit], dtype=float),
                np.array([fouls], dtype=float),
                bonus,
            )[0]
        )

    assert mask(4, 10, 3, 4)
    assert mask(4, 10, 3, 0)
    assert not mask(4, 30, 3, 2)
    assert not mask(3, 10, 3, 6)
    assert not mask(4, 20, 15, 5)
    assert mask(5, 8, 2, 3, bonus=OT_BONUS_FOULS)


def _five(team_id: int, start: int, positions: list[str], foul: float, play: str) -> list:
    players = []
    for index, position in enumerate(positions):
        player = make_player(
            start + index,
            f"{position}{index}",
            team_id,
            position,
            32,
            play,
            80 if index == 0 else 4,
            1.05,
        )
        player.foul_rate = foul
        players.append(player)
    return players


def _spec(home_players, away_players) -> GameSpec:
    home = make_team("den", pace=100)
    away = make_team("mia", pace=100)
    return GameSpec(
        game_id="clock",
        date="2026-10-22",
        home=home,
        away=away,
        home_players=home_players,
        away_players=away_players,
        context=GameContext(),
        league=LeaguePriors.flat(),
    )


def test_post_fouls_are_charged_to_the_center():
    home = make_team("den", pace=100)
    away = make_team("mia", pace=100)
    offense = _five(home.team_id, 1, ["C", "G", "G", "F", "F"], 1.0, "Postup")
    defense = _five(away.team_id, 20, ["C", "G", "G", "G", "G"], 1.0, "Spotup")
    result = simulate_game(_spec(offense, defense), n_sims=180, seed=3)
    center = float(result.player_stats[20]["pf"].mean())
    guard = float(result.player_stats[21]["pf"].mean())
    # The post still belongs to the center. The foul budget is his rate, so the
    # gap is his share of that budget rather than a second whistle.
    assert center > guard + 0.6


def _drawn(ft_draw: float, foul: float, three: float, fouls: float = 0.0, ft: float = 0.10, follow: bool = False, kept: bool = False):
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    defense = _five(2, 20, ["C", "G", "G", "F", "F"], foul, "Spotup")
    for player in offense:
        player.ft_draw = ft_draw
        player.zone_freq = {"ra": max(0.0, 1.0 - three), "paint": 0.0, "mid": 0.0, "corner3": 0.0, "atb3": three}
        for skill in player.offense.values():
            skill.ppp = 1.05
            skill.tov_poss_pct = 0.10
            skill.ft_poss_pct = ft
            skill.fg_pct = 0.46
    n = 40
    off = _side(offense, np.full((n, 5, len(PLAY_TYPES)), 1.05))
    deff = _side(defense, np.ones((n, 5, len(PLAY_TYPES))))
    lineup = np.tile(np.arange(5), (n, 1))
    zeros = np.zeros(n)
    return _draw_side(
        np.random.default_rng(4),
        off,
        lineup,
        deff,
        lineup,
        np.zeros((n, 5)),
        np.ones(len(PLAY_TYPES)),
        np.ones(5),
        np.ones(5),
        np.zeros(n, dtype=bool),
        zeros,
        1.0,
        1.0,
        1,
        zeros,
        zeros,
        np.zeros(n, dtype=bool),
        team_fouls=np.full(n, fouls),
        follow=np.full(n, follow),
        continuation=np.full(n, kept),
    )


def test_his_free_throw_rate_and_the_defender_move_the_whistle():
    drawn = _drawn(1.70, 1.0, 0.20)
    shy = _drawn(0.55, 1.0, 0.20)
    assert float(drawn["p_sfoil"].mean()) > float(shy["p_sfoil"].mean()) * 1.8
    rough = _drawn(1.0, 2.2, 0.20)
    clean = _drawn(1.0, 0.50, 0.20)
    assert float(rough["p_sfoil"].mean()) > float(clean["p_sfoil"].mean()) * 1.4


def test_the_free_throw_residual_is_applied_once():
    high = _drawn(1.60, 1.0, 0.20)
    low = _drawn(0.80, 1.0, 0.20)
    ratio = float(high["p_sfoil"].mean()) / float(low["p_sfoil"].mean())
    assert ratio == pytest.approx(1.60 / 0.80, rel=0.08)
    assert ratio < 3.0


def _shares(draw: dict) -> float:
    total = draw["p_off"] + draw["p_tov"] + draw["p_sfoil"] + draw["p_nsf"] + draw["p_shot"]
    return float(np.max(np.abs(total - 1.0)))


def test_a_transition_reach_in_is_priced_like_a_common_foul():
    # Ordinary transition NSF is no longer a take: pre-bonus has no free throws,
    # and the bonus is two shots. A true late take still awards one plus the ball.
    pre = _drawn(1.0, 1.0, 0.20, fouls=0.0, ft=0.06)
    bonus = _drawn(1.0, 1.0, 0.20, fouls=4.0, ft=0.06)
    assert float(pre["p_nsf"].mean()) > 0.04
    # Expected NSF points come from the common formula, not ft_pct for every transition.
    assert float(bonus["foul_pts"].mean()) > float(pre["foul_pts"].mean())
    assert _award(COMMON, fouls=4) == (2, True, False, True)
    assert _award(TAKE, fouls=4) == (1, True, True, False)


def test_a_continuation_does_not_open_a_second_foul_budget():
    # Continuations keep the budget-capped shooting foul. Putbacks still close
    # the reach-in; a kept whistle already in the bonus can end as two FTA.
    opening = _drawn(1.5, 1.0, 0.20, ft=0.25)
    follow = _drawn(1.5, 1.0, 0.20, ft=0.25, follow=True)
    side_out = _drawn(1.5, 1.0, 0.20, ft=0.25, kept=True)
    # Soft FT draw leaves leftover reach-in mass; bonus keep can spend it as COMMON.
    reach_open = _drawn(1.0, 1.0, 0.20, ft=0.06)
    bonus_keep = _drawn(1.0, 1.0, 0.20, fouls=4.0, ft=0.06, kept=True)
    assert float(follow["p_nsf"].mean()) == 0.0
    assert float(side_out["p_nsf"].mean()) == 0.0
    assert float(reach_open["p_nsf"].mean()) > 0.04
    assert float(bonus_keep["p_nsf"].mean()) > 0.0
    # Continuations use the capped shooting foul, not the uncapped play rate.
    assert float(follow["p_sfoil"].mean()) == pytest.approx(float(opening["p_sfoil"].mean()), abs=0.02)
    assert float(side_out["p_sfoil"].mean()) == pytest.approx(float(opening["p_sfoil"].mean()), abs=0.02)
    assert float(follow["p_shot"].mean()) < 1.0 - float(follow["p_tov"].mean()) - float(follow["p_off"].mean()) - 0.05
    reach = _drawn(1.0, 1.0, 0.20, ft=0.06)
    putback = _drawn(1.0, 1.0, 0.20, ft=0.06, follow=True)
    assert float(reach["p_nsf"].mean()) > 0.04
    # Putbacks drop the reach-in without pouring it into turnovers.
    assert float(putback["p_nsf"].mean()) == 0.0
    assert float(putback["p_tov"].mean()) <= float(reach["p_tov"].mean()) + 0.01
    assert float(putback["p_shot"].mean()) > float(reach["p_shot"].mean())
    assert _shares(opening) < 1e-6
    assert _shares(follow) < 1e-6
    assert _shares(side_out) < 1e-6
    assert _shares(bonus_keep) < 1e-6


def _count(result, ids: list[int], key: str) -> float:
    total = 0.0
    for player_id in ids:
        box = result.player_stats.get(player_id)
        if box is not None and key in box:
            total += float(box[key].mean())
    return total


def test_the_glass_does_not_pick_up_a_foul_per_rebound():
    def played(glass: float, dreb: float):
        home = make_team("den", pace=100)
        away = make_team("mia", pace=100)
        home.oreb_pct = 0.40 if glass > 0.1 else 0.18
        away.dreb_pct = 0.60 if glass > 0.1 else 0.80
        away.oreb_allowed = 0.40 if glass > 0.1 else 0.20
        offense = _five(home.team_id, 1, ["C", "F", "F", "G", "G"], 1.0, "Postup")
        defense = _five(away.team_id, 20, ["C", "C", "F", "F", "G"], 2.0, "Spotup")
        for player in offense:
            player.oreb_pct = glass
            for skill in player.offense.values():
                skill.ft_poss_pct = 0.04
                skill.tov_poss_pct = 0.10
        for player in defense:
            player.dreb_pct = dreb
        result = simulate_game(_spec(offense, defense), n_sims=40, seed=11)
        return _count(result, [1, 2, 3, 4, 5], "oreb"), _count(result, [20, 21, 22, 23, 24], "pf")

    glass_boards, glass_fouls = played(0.18, 0.08)
    dry_boards, dry_fouls = played(0.005, 0.20)
    extra_boards = glass_boards - dry_boards
    extra_fouls = glass_fouls - dry_fouls
    assert extra_boards > 4.0
    # A second five-man budget on each rebound would add about one foul for every few boards.
    assert extra_fouls < 0.15 * extra_boards


def test_a_block_is_a_two_point_miss():
    def blocked(play: str, zone: dict[str, float], blk: float) -> float:
        home = make_team("den", pace=100)
        away = make_team("mia", pace=100)
        offense = _five(home.team_id, 1, ["C", "F", "F", "G", "G"], 1.0, play)
        defense = _five(away.team_id, 20, ["C", "C", "F", "F", "G"], 1.0, "Spotup")
        for player in offense:
            player.zone_freq = dict(zone)
            for skill in player.offense.values():
                skill.fg_pct = 0.40
                skill.ft_poss_pct = 0.04
        for player in defense:
            player.blk_weight = blk
        result = simulate_game(_spec(offense, defense), n_sims=30, seed=5)
        return _count(result, [20, 21, 22, 23, 24], "blk")

    twos = {"ra": 1.0, "paint": 0.0, "mid": 0.0, "corner3": 0.0, "atb3": 0.0}
    threes = {"ra": 0.0, "paint": 0.0, "mid": 0.0, "corner3": 0.0, "atb3": 1.0}
    walls = blocked("Postup", twos, 3.5)
    soft = blocked("Postup", twos, 0.15)
    from_deep = blocked("Spotup", threes, 3.5)
    # The old clamp stopped a shot-blocker at 1.6 times a flat rate of every miss.
    assert walls > soft + 4.0
    assert walls > 8.0
    assert from_deep < walls / 3.0


def test_one_efficiency_leaves_the_three_point_percentage():
    # The chart's three stays the three. A matching defensive edge does not
    # multiply zone, head-to-head, and play type into three separate cuts.
    neutral = _draw_defense(np.ones(len(PLAY_TYPES)), np.ones(5), np.ones(5))
    edged = _draw_defense(np.full(len(PLAY_TYPES), 0.90), np.full(5, 0.90), np.full(5, 0.90))
    ratio = float(edged["three_fg"].mean()) / float(neutral["three_fg"].mean())
    assert 0.84 < ratio < 0.96


def test_a_smaller_shot_read_does_not_replace_the_defense():
    level = _draw_defense(np.full(len(PLAY_TYPES), 0.90), np.ones(5), np.ones(5))
    softer = _draw_defense(np.full(len(PLAY_TYPES), 0.90), np.full(5, 0.96), np.ones(5))
    other_way = _draw_defense(np.full(len(PLAY_TYPES), 0.90), np.full(5, 1.03), np.ones(5))
    same = float(softer["three_fg"].mean()) / float(level["three_fg"].mean())
    opposite = float(other_way["three_fg"].mean()) / float(level["three_fg"].mean())
    # 0.96 is a smaller cut in the same direction, so the 0.90 level stands.
    assert 0.98 < same < 1.02
    # 1.03 points the other way. It moves the level by about 3 percent.
    assert 1.01 < opposite < 1.06


def test_the_five_is_a_residual_on_the_team_defense():
    base = _draw_defense(np.full(len(PLAY_TYPES), 0.90), np.ones(5), np.ones(5))
    five = _draw_defense(np.full(len(PLAY_TYPES), 0.90), np.ones(5), np.ones(5), def_impact=1.06)
    ratio = float(five["three_fg"].mean()) / float(base["three_fg"].mean())
    # 6 percent worse than his own team, on top of a team that allows 10 percent less.
    assert 1.03 < ratio < 1.09


def _draw_defense(def_mult: np.ndarray, zone: np.ndarray, pair: np.ndarray, off_impact: float = 1.0, def_impact: float = 1.0):
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    defense = _five(2, 20, ["C", "G", "G", "F", "F"], 1.0, "Spotup")
    for player in offense:
        player.ft_draw = 1.0
        player.zone_freq = {"ra": 0.20, "paint": 0.0, "mid": 0.0, "corner3": 0.0, "atb3": 0.80}
        for skill in player.offense.values():
            skill.ppp = 1.05
            skill.tov_poss_pct = 0.10
            skill.ft_poss_pct = 0.10
            skill.fg_pct = 0.46
            skill.poss = 400.0
        player.off_impact = off_impact
    for player in defense:
        player.def_impact = def_impact
    n = 40
    off = _side(offense, np.full((n, 5, len(PLAY_TYPES)), 1.05))
    deff = _side(defense, np.ones((n, 5, len(PLAY_TYPES))))
    lineup = np.tile(np.arange(5), (n, 1))
    zeros = np.zeros(n)
    return _draw_side(
        np.random.default_rng(4),
        off,
        lineup,
        deff,
        lineup,
        np.zeros((n, 5)),
        def_mult,
        zone,
        pair,
        np.zeros(n, dtype=bool),
        zeros,
        1.0,
        1.0,
        1,
        zeros,
        zeros,
        np.zeros(n, dtype=bool),
    )


def test_the_shot_bucket_is_a_field_goal():
    quiet = _drawn(1.0, 1.0, 0.20, fouls=0)
    np.testing.assert_allclose(quiet["p_volume"], quiet["p_shot"])
    assert float(quiet["p_shot"].mean()) > 0.4


def test_a_bonus_non_shooting_foul_is_a_foul_not_a_deleted_shot():
    quiet = _drawn(1.0, 1.0, 0.20, fouls=0)
    bonus = _drawn(1.0, 1.0, 0.20, fouls=6)
    assert float(bonus["foul_pts"].mean()) > float(quiet["foul_pts"].mean()) + 0.04
    np.testing.assert_allclose(bonus["p_volume"], bonus["p_shot"])


def test_makes_and_misses_share_one_three_rate():
    home = make_team("den", pace=100)
    away = make_team("mia", pace=100)
    offense = _five(home.team_id, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    defense = _five(away.team_id, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    for player in offense:
        player.zone_freq = {"ra": 0.15, "paint": 0.0, "mid": 0.05, "corner3": 0.0, "atb3": 0.80}
        player.zone_fg = {"ra": 0.62, "paint": 0.42, "mid": 0.40, "corner3": 0.36, "atb3": 0.34}
        for skill in player.offense.values():
            skill.ppp = 1.05
            skill.fg_pct = 0.46
            skill.ft_poss_pct = 0.08
            skill.tov_poss_pct = 0.11
    result = simulate_game(_spec(offense, defense), n_sims=200, seed=11)
    fga = tpa = tpm = 0.0
    for player_id in result.home_ids:
        box = result.player_stats[player_id]
        fga += float(box["fga"].mean())
        tpa += float(box["tpa"].mean())
        tpm += float(box["tpm"].mean())
    assert fga > 40
    share = tpa / fga
    made = tpm / tpa
    assert share > 0.55
    assert 0.22 < made < 0.46


def test_a_three_point_shot_is_a_three_point_foul():
    bomber = _drawn(1.0, 1.0, 0.78)
    big = _drawn(1.0, 1.0, 0.04)
    assert float(bomber["three_p"].mean()) > float(big["three_p"].mean()) + 0.35


def test_a_flagrant_two_ends_his_night():
    pf = np.zeros((2, 3))
    pf[0, 1] = 2
    _eject(pf, np.array([1, 1]), np.array([True, False]))
    assert pf[0, 1] == 6
    assert pf[1, 1] == 0


def test_a_foul_prone_defense_sends_the_offense_to_the_line():
    home = make_team("den", pace=100)
    offense = _five(home.team_id, 1, ["G", "G", "F", "F", "C"], 1.0, "Isolation")
    tough = _five(home.team_id + 1, 30, ["G", "G", "F", "F", "C"], 2.4, "Spotup")
    timid = _five(home.team_id + 2, 40, ["G", "G", "F", "F", "C"], 0.45, "Spotup")
    high = simulate_game(_spec(offense, tough), n_sims=160, seed=5)
    low = simulate_game(_spec(offense, timid), n_sims=160, seed=5)
    assert float(high.home_fta.mean()) > float(low.home_fta.mean()) + 3.0


def test_every_free_throw_trip_has_a_defensive_foul_and_the_clock_tracks_pace():
    spec = _spec(
        _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "PRBallHandler"),
        _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup"),
    )
    result = simulate_game(spec, n_sims=120, seed=9)
    away_pf = np.vstack([result.player_stats[pid]["pf"] for pid in result.away_ids]).sum(axis=0)
    home_pf = np.vstack([result.player_stats[pid]["pf"] for pid in result.home_ids]).sum(axis=0)
    assert np.all(away_pf + 1e-6 >= result.home_ft_trips)
    assert np.all(home_pf + 1e-6 >= result.away_ft_trips)
    trips = result.home_ft_trips + result.away_ft_trips
    attempts = result.home_fta + result.away_fta
    assert float(trips.sum()) > 0
    ratio = float(attempts.sum() / trips.sum())
    assert 1.45 < ratio < 2.45
    assert not np.all(np.isclose(attempts, 2.0 * trips))
    assert 80 < float(result.home_possessions.mean()) < 125
    assert float(result.home_team_fouls.mean()) > float(result.home_max_quarter_fouls.mean()) + 3
    assert 95 < float(result.home_points.mean()) < 145
    assert float(result.intentional_fouls.mean()) > 0.05
    # Points, attempts, free throws, and turnovers are the same draw.
    fgm = np.vstack([result.player_stats[pid]["fgm"] for pid in result.home_ids]).sum(axis=0)
    tpm = np.vstack([result.player_stats[pid]["tpm"] for pid in result.home_ids]).sum(axis=0)
    fga = np.vstack([result.player_stats[pid]["fga"] for pid in result.home_ids]).sum(axis=0)
    tov = np.vstack([result.player_stats[pid]["tov"] for pid in result.home_ids]).sum(axis=0)
    built = 2.0 * (fgm - tpm) + 3.0 * tpm + result.home_ftm
    np.testing.assert_allclose(result.home_points, built)
    assert float(fga.mean()) > 40
    assert float(result.home_fta.mean()) > 5
    assert float(tov.mean()) > 5


def _raw_draw(play: str | None, *, live: bool = False, n: int = 700):
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, play or "Transition")
    defense = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    for player in offense:
        for name, skill in player.offense.items():
            if play is None:
                skill.poss = 30.0 if name in ("Transition", "Postup") else 0.0
            else:
                skill.poss = 40.0 if name == play else 0.0
    off = _side(offense, np.ones((n, 5, len(PLAY_TYPES))))
    deff = _side(defense, np.ones((n, 5, len(PLAY_TYPES))))
    lineup = np.tile(np.arange(5), (n, 1))
    zeros = np.zeros(n)
    return _draw_side(
        np.random.default_rng(11),
        off,
        lineup,
        deff,
        lineup,
        np.zeros((n, 5)),
        np.ones(len(PLAY_TYPES)),
        np.ones(5),
        np.ones(5),
        np.full(n, live),
        zeros,
        1.0,
        1.0,
        1,
        zeros,
        zeros,
        np.zeros(n, dtype=bool),
    )


def test_the_lineup_sets_the_average_length_and_the_action_moves_it():
    running = _raw_draw("Transition")
    posting = _raw_draw("Postup")
    assert abs(float(running["raw"].mean()) - float(posting["raw"].mean())) < 1.0
    assert 13.2 < float(running["raw"].mean()) < 15.6
    mixed = _raw_draw(None)
    transition = mixed["type_idx"] == PLAY_TYPES.index("Transition")
    post = mixed["type_idx"] == PLAY_TYPES.index("Postup")
    assert int(transition.sum()) > 80 and int(post.sum()) > 80
    assert float(mixed["raw"][transition].mean()) < 11.0
    assert float(mixed["raw"][post].mean()) > 16.0
    stolen = _raw_draw("Postup", live=True)
    live_transition = stolen["type_idx"] == PLAY_TYPES.index("Transition")
    assert int(live_transition.sum()) > 80
    assert float(stolen["raw"][live_transition].mean()) < 10.0


def test_a_faster_team_uses_a_shorter_clock():
    def possessions(pace: float) -> float:
        home_players = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Transition")
        away_players = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
        for player in home_players:
            for name, skill in player.offense.items():
                skill.poss = 40.0 if name == "Transition" else 0.0
        spec = GameSpec(
            game_id=f"pace-{pace}",
            date="2026-10-22",
            home=make_team("den", pace=pace),
            away=make_team("mia", pace=pace),
            home_players=home_players,
            away_players=away_players,
            context=GameContext(),
            league=LeaguePriors.flat(),
        )
        result = simulate_game(spec, n_sims=60, seed=4)
        return float(result.home_possessions.mean())

    level = possessions(100)
    assert 90 < level < 115
    assert possessions(110) > possessions(90) + 8


def test_two_fast_teams_play_faster_than_a_fast_team_against_an_average_one():
    def possessions(home_pace: float, away_pace: float) -> float:
        home_players = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
        away_players = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
        spec = GameSpec(
            game_id=f"pace-{home_pace}-{away_pace}",
            date="2026-10-22",
            home=make_team("den", pace=home_pace),
            away=make_team("mia", pace=away_pace),
            home_players=home_players,
            away_players=away_players,
            context=GameContext(),
            league=LeaguePriors.flat(),
        )
        result = simulate_game(spec, n_sims=100, seed=4)
        return float(result.home_possessions.mean())

    assert possessions(110, 110) > possessions(110, 100) + 6


def test_the_two_fives_set_the_game_pace_as_a_product():
    home = make_team("den", pace=108)
    away = make_team("mia", pace=108)
    home_ids = [1, 2, 3, 4, 5]
    away_ids = [6, 7, 8, 9, 10]
    assert abs(_pace(home, away, home_ids, away_ids, 1.0) - 116.64) < 1e-6
    away.pace = 94
    assert abs(_pace(home, away, home_ids, away_ids, 1.0) - 101.52) < 1e-6
    home.pace = 100
    away.pace = 100
    assert _pace(home, away, home_ids, away_ids, 1.0) == 100.0


def test_a_players_pace_travels_into_a_five_that_has_not_played():
    home = make_team("den", pace=100)
    away = make_team("mia", pace=100)
    home.lineup_pace[(1, 2, 3, 4, 5)] = 110.0
    home.lineup_minutes[(1, 2, 3, 4, 5)] = 500.0
    home.lineup_pace[(1, 2, 3, 6, 7)] = 100.0
    home.lineup_minutes[(1, 2, 3, 6, 7)] = 500.0
    impacts = _fit_pace_impacts(home)
    assert impacts[4] > impacts[1] + 1.0
    assert impacts[6] < impacts[1]
    home.pace_impact = impacts
    kept = _unit_pace(home, [1, 2, 3, 4, 5])
    swapped = _unit_pace(home, [1, 2, 3, 6, 7])
    fresh = _unit_pace(home, [1, 2, 3, 8, 9])
    assert kept > fresh > swapped
    assert abs(_pace(home, away, [1, 2, 3, 8, 9], [20, 21, 22, 23, 24], 1.0) - fresh) < 1e-6

    played = make_team("den", pace=100)
    played.lineup_pace[(1, 2, 3, 90, 91)] = 120.0
    played.lineup_minutes[(1, 2, 3, 90, 91)] = 800.0
    home_players = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    away_players = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    fast = GameSpec(
        game_id="pace-profile",
        date="2026-10-22",
        home=played,
        away=make_team("mia", pace=100),
        home_players=home_players,
        away_players=_five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup"),
        context=GameContext(),
        league=LeaguePriors.flat(),
    )
    plain = GameSpec(
        game_id="pace-plain",
        date="2026-10-22",
        home=make_team("den", pace=100),
        away=make_team("mia", pace=100),
        home_players=home_players,
        away_players=away_players,
        context=GameContext(),
        league=LeaguePriors.flat(),
    )
    fast_trips = float(simulate_game(fast, n_sims=40, seed=6).home_possessions.mean())
    plain_trips = float(simulate_game(plain, n_sims=40, seed=6).home_possessions.mean())
    assert fast_trips > plain_trips + 6


def test_a_rebound_stays_with_the_rebounder_or_a_kick_out():
    players = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    players[4].offense["OffRebound"].poss = 9
    players[0].offense["Spotup"].poss = 7
    for player in players:
        player.offense["Postup"].poss = 0
        player.offense["Isolation"].poss = 0
    side = _side(players, np.ones((1, 5, len(PLAY_TYPES))))
    lineup = np.arange(5)[None, :]
    on_floor = _second_chance_poss(side, lineup, np.array([4]), side.poss[lineup], np.array([True]))
    assert on_floor[0, 4, OFFREB] == 9
    assert on_floor[0, 4, PLAY_TYPES.index("Spotup")] == 0
    assert on_floor[0, 0, PLAY_TYPES.index("Spotup")] == 7
    assert on_floor[0, 0, OFFREB] == 0
    assert on_floor[0, 0, PLAY_TYPES.index("Postup")] == 0
    gone = _second_chance_poss(side, lineup, np.array([99]), side.poss[lineup], np.array([True]))
    np.testing.assert_array_equal(gone, side.poss[lineup])


def test_a_rebound_does_not_change_the_first_shot():
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    defense = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    for player in offense:
        for skill in player.offense.values():
            skill.ppp = 1.08
            skill.tov_poss_pct = 0.11
            skill.ft_poss_pct = 0.08
            skill.fg_pct = 0.48
    n = 50
    ppp = np.full((n, len(offense), len(PLAY_TYPES)), 1.08)
    off = _side(offense, ppp)
    deff = _side(defense, np.ones((n, len(defense), len(PLAY_TYPES))))
    lineup = np.tile(np.arange(5), (n, 1))
    zeros = np.zeros(n)
    flags = np.zeros(n, dtype=bool)

    def draw(prob: float) -> np.ndarray:
        return _draw_side(
            np.random.default_rng(11),
            off,
            lineup,
            deff,
            lineup,
            np.zeros((n, len(defense))),
            np.ones(len(PLAY_TYPES)),
            np.ones(len(offense)),
            np.ones(len(offense)),
            flags,
            zeros,
            1.0,
            1.0,
            1,
            zeros,
            zeros,
            flags,
            oreb_prob=prob,
        )["make_p"]

    bare = draw(0.0)
    glass = draw(0.40)
    np.testing.assert_allclose(glass, bare)
    assert float(bare.mean()) > 0.38


def test_a_high_percentage_play_does_not_fill_the_rest_with_misses():
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Cut")
    defense = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Cut")
    for player in offense:
        player.zone_fg = {zone: 0.66 for zone in player.zone_fg}
        for skill in player.offense.values():
            skill.ppp = 1.10
            skill.tov_poss_pct = 0.08
            skill.ft_poss_pct = 0.06
            skill.fg_pct = 0.66
    n = 80
    ppp = np.full((n, len(offense), len(PLAY_TYPES)), 1.10)
    off = _side(offense, ppp)
    deff = _side(defense, np.ones((n, len(defense), len(PLAY_TYPES))))
    lineup = np.tile(np.arange(5), (n, 1))
    zeros = np.zeros(n)
    drawn = _draw_side(
        np.random.default_rng(11),
        off,
        lineup,
        deff,
        lineup,
        np.zeros((n, len(defense))),
        np.ones(len(PLAY_TYPES)),
        np.ones(len(offense)),
        np.ones(len(offense)),
        np.zeros(n, dtype=bool),
        zeros,
        1.0,
        1.0,
        1,
        zeros,
        zeros,
        np.zeros(n, dtype=bool),
    )
    np.testing.assert_allclose(drawn["p_volume"], drawn["p_shot"])
    # His own percentage stays the make. A short sample may lean on his points
    # per possession; the player with the possessions does not.
    star = drawn["actor"] == 0
    assert float(drawn["make_p"][star].mean()) > 0.60
    assert float(drawn["p_shot"].mean()) > 0.55


def test_a_second_chance_does_not_rebound_again():
    chance = np.array([0.30, 0.22, 0.18])
    follow = np.array([False, True, False])
    boarded = _live_board(chance, follow)
    assert boarded[0] == pytest.approx(0.30)
    assert boarded[1] == pytest.approx(0.0)
    assert boarded[2] == pytest.approx(0.18)


def test_the_lineup_on_the_floor_moves_the_price():
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    defense = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    n = 300
    lineup = np.tile(np.arange(5), (n, 1))
    ppp = np.full((n, 5, len(PLAY_TYPES)), 1.05)
    zeros = np.zeros(n)

    def draw(off_impact: float, def_impact: float):
        for player in offense:
            player.off_impact = off_impact
        for player in defense:
            player.def_impact = def_impact
        return _draw_side(
            np.random.default_rng(5),
            _side(offense, ppp),
            lineup,
            _side(defense, ppp),
            lineup,
            np.zeros((n, 5)),
            np.ones(len(PLAY_TYPES)),
            np.ones(5),
            np.ones(5),
            np.zeros(n, dtype=bool),
            zeros,
            1.0,
            1.0,
            1,
            zeros,
            zeros,
            np.zeros(n, dtype=bool),
        )

    boosted = float(draw(1.06, 1.06)["make_p"].mean())
    quiet = float(draw(0.94, 0.94)["make_p"].mean())
    assert boosted > quiet + 0.02


def test_soft_pnr_defense_draws_more_handler_share():
    """Damped def_mult mix tilt: soft PnR D raises PRBallHandler volume vs stingy."""
    n = 5000
    soft = np.ones(len(PLAY_TYPES))
    soft[PLAY_TYPES.index("PRBallHandler")] = 1.18
    soft[PLAY_TYPES.index("PRRollman")] = 1.12
    stingy = np.ones(len(PLAY_TYPES))
    stingy[PLAY_TYPES.index("PRBallHandler")] = 0.85
    stingy[PLAY_TYPES.index("PRRollman")] = 0.88
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    defense = _five(2, 20, ["C", "G", "G", "F", "F"], 1.0, "Spotup")
    for player in offense:
        for play in PLAY_TYPES:
            skill = player.offense[play]
            skill.ppp = 1.05
            skill.tov_poss_pct = 0.10
            skill.ft_poss_pct = 0.10
            skill.fg_pct = 0.46
            skill.poss = 400.0
            skill.poss_pct = 1.0 / len(PLAY_TYPES)

    def share(def_mult: np.ndarray) -> float:
        off = _side(offense, np.full((n, 5, len(PLAY_TYPES)), 1.05))
        # Flat possession weights so only the hunt tilt moves PnR share.
        off.poss[:] = 1.0
        deff = _side(defense, np.ones((n, 5, len(PLAY_TYPES))))
        lineup = np.tile(np.arange(5), (n, 1))
        zeros = np.zeros(n)
        drawn = _draw_side(
            np.random.default_rng(11),
            off,
            lineup,
            deff,
            lineup,
            np.zeros((n, 5)),
            def_mult,
            np.ones(5),
            np.ones(5),
            np.zeros(n, dtype=bool),
            zeros,
            1.0,
            1.0,
            1,
            zeros,
            zeros,
            np.zeros(n, dtype=bool),
        )
        return float(np.mean(drawn["type_idx"] == PLAY_TYPES.index("PRBallHandler")))

    soft_share = share(soft)
    stingy_share = share(stingy)
    assert soft_share > stingy_share
    assert soft_share / max(stingy_share, 1e-6) > 1.05


def test_spot_mass_follows_the_rate_card_not_one_action():
    from nba_sim.sim.clock import _crosswalk_mix

    synergy = np.zeros(len(PLAY_TYPES))
    shares = {
        "Isolation": 0.20,
        "Spotup": 0.10,
        "Handoff": 0.05,
        "Cut": 0.05,
        "OffScreen": 0.10,
        "PRBallHandler": 0.20,
        "PRRollman": 0.10,
        "Transition": 0.10,
        "Postup": 0.10,
    }
    for name, share in shares.items():
        synergy[PLAY_TYPES.index(name)] = share
    out = _crosswalk_mix({"spot": 0.50, "pick": 0.30, "transition": 0.10, "post": 0.10}, synergy)
    assert out[PLAY_TYPES.index("Spotup")] == pytest.approx(0.10, abs=0.02)
    assert out[PLAY_TYPES.index("Isolation")] == pytest.approx(0.20, abs=0.02)
    assert out[PLAY_TYPES.index("PRRollman")] == pytest.approx(0.10, abs=0.02)
    assert out[PLAY_TYPES.index("PRBallHandler")] > out[PLAY_TYPES.index("PRRollman")]


def test_a_thin_play_by_play_sample_leaves_the_rate_card():
    from nba_sim.features.court import SquadFace
    from nba_sim.sim.clock import _apply_squad_mix

    players = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    side = _side(players, np.ones((1, 5, len(PLAY_TYPES))))
    side.court_on = True
    ids = [player.player_id for player in players]
    side.squads = {frozenset(ids): SquadFace(mix={"spot": 1.0}, possessions=4.0)}
    lineup = np.arange(5)[None, :]
    weights = side.poss[lineup]
    out = _apply_squad_mix(side, lineup, weights.copy())
    np.testing.assert_allclose(out, weights)


def test_a_profiled_game_scores_without_the_script():
    from nba_sim.features.court import PlayerCourt
    from nba_sim.sim.clock import play_game

    home = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    away = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    for player in home + away:
        player.court = PlayerCourt(player_id=player.player_id, minutes=28.0, minutes_close=28.0, minutes_blowout=10.0)
    spec = _spec(home, away)
    n = 6
    played = play_game(
        np.random.default_rng(2),
        spec,
        None,
        None,
        np.ones((n, len(home), len(PLAY_TYPES))),
        np.ones((n, len(away), len(PLAY_TYPES))),
        1.0,
        1.0,
        None,
        None,
    )
    assert float(played.home.pts.sum()) > 0
    assert float(played.home.fga.sum()) > 0
    assert float(played.home.tov.sum()) >= 0


def test_a_profiled_game_does_not_build_scripted_units(monkeypatch):
    from nba_sim.features.court import PlayerCourt
    import nba_sim.sim.engine as engine

    calls = {"n": 0}
    real = engine._unit

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(engine, "_unit", wrapped)
    home = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    away = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    simulate_game(_spec(home, away), n_sims=1, seed=1)
    assert calls["n"] == 12
    for player in home + away:
        player.court = PlayerCourt(player_id=player.player_id, minutes=28.0, minutes_close=28.0, sample=30)
    calls["n"] = 0
    simulate_game(_spec(home, away), n_sims=1, seed=1)
    assert calls["n"] == 0


def test_the_matched_defender_moves_the_action():
    from nba_sim.sim.clock import _matchup_tilt

    stops = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    for player in stops:
        ppp = 0.80 if player.position == "G" else 1.16
        player.defense["Isolation"] = PlaySkill(poss=400, ppp=ppp, tov_poss_pct=0.0)
    side = _side(stops, np.ones((1, 5, len(PLAY_TYPES))))
    tilt = _matchup_tilt(side, np.array([[0, 1, 2, 3, 4]]), np.array([PLAY_TYPES.index("Isolation")]))
    assert float(tilt[0]) < 0.98


def test_a_guard_shot_follows_the_guard_defending_him():
    from nba_sim.sim.adjust import zone_matchup_mult

    shooter = make_player(1, "Shooter", 1, "G", 32, "Spotup", 40, 1.05)
    shooter.zone_freq = {"ra": 0.05, "paint": 0.05, "mid": 0.05, "corner3": 0.15, "atb3": 0.70}
    tight = make_player(2, "Guard", 2, "G", 32, "Spotup", 8, 1.0)
    leak = make_player(3, "Big", 2, "C", 32, "Spotup", 8, 1.0)
    normal = {"ra": 0.66, "paint": 0.42, "mid": 0.41, "corner3": 0.38, "atb3": 0.36}
    volume = {"ra": 80, "paint": 40, "mid": 30, "corner3": 40, "atb3": 160}

    def zones(player, three: float) -> None:
        player.zone_normal_fg = dict(normal)
        player.zone_d_fg = {"ra": 0.66, "paint": 0.42, "mid": 0.41, "corner3": three, "atb3": three}
        player.zone_d_fga = dict(volume)

    zones(tight, 0.30)
    zones(leak, 0.42)
    team = make_team("mia")
    league = LeaguePriors.flat()
    guarded = zone_matchup_mult(shooter, team, league, [tight, leak])
    zones(tight, 0.42)
    zones(leak, 0.30)
    swapped = zone_matchup_mult(shooter, team, league, [tight, leak])
    assert guarded < swapped


def test_turnover_chance_is_the_handler_against_this_defense():
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Isolation")
    for player in offense:
        for skill in player.offense.values():
            skill.tov_poss_pct = 0.10
    pressure = _five(2, 20, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    soft = _five(3, 40, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    for player in pressure:
        player.defense["Isolation"] = PlaySkill(poss=500, ppp=1.0, tov_poss_pct=0.24)
    for player in soft:
        player.defense["Isolation"] = PlaySkill(poss=500, ppp=1.0, tov_poss_pct=0.06)
    n = 400
    lineup = np.tile(np.arange(5), (n, 1))
    ppp = np.full((n, 5, len(PLAY_TYPES)), 1.05)
    off = _side(offense, ppp)
    zeros = np.zeros(n)

    def draw(defense, rate: float):
        return _draw_side(
            np.random.default_rng(1),
            off,
            lineup,
            defense,
            lineup,
            np.zeros((n, 5)),
            np.ones(len(PLAY_TYPES)),
            np.ones(5),
            np.ones(5),
            np.zeros(n, dtype=bool),
            zeros,
            1.0,
            1.0,
            1,
            zeros,
            zeros,
            np.zeros(n, dtype=bool),
            np.zeros(n, dtype=bool),
            np.full(n, -1),
            np.full(len(PLAY_TYPES), rate),
            0.25,
        )

    hard = draw(_side(pressure, ppp), 0.24)
    easy = draw(_side(soft, ppp), 0.06)
    assert float(hard["p_tov"].mean()) > float(easy["p_tov"].mean()) + 0.03


def test_a_reserve_plays_the_end_of_a_blowout_and_a_closer_keeps_his_minutes():
    home = make_team("uta")
    away = make_team("was")
    home_players = rotation(home.team_id, "UTA", 100)
    away_players = rotation(away.team_id, "WAS", 200)
    home_players.append(make_player(199, "UTA End", home.team_id, "G", 2.0, "Spotup", 4.0, 0.9, other_poss=1.0))
    result = simulate_game(
        GameSpec(
            game_id="finish",
            date="2026-10-22",
            home=home,
            away=away,
            home_players=home_players,
            away_players=away_players,
            context=GameContext(),
            league=LeaguePriors.flat(),
        ),
        n_sims=100,
        seed=4,
    )
    star = result.player_stats[100]["minutes"]
    reserve = result.player_stats[199]["minutes"]
    # He misses some nights. The exact share moves with the possession path.
    # The nights he plays are a finish rather than a trickle.
    assert np.mean(reserve < 1.0) >= 0.10
    assert np.mean(reserve > 4.0) > 0.2
    assert float(np.percentile(reserve, 90)) > 6.0
    assert float(np.percentile(star, 10)) > 22.0
    assert float(np.median(star)) > float(np.percentile(reserve, 90))
    np.testing.assert_allclose(
        np.vstack([result.player_stats[pid]["minutes"] for pid in result.home_ids]).sum(axis=0),
        240.0,
        atol=1e-6,
    )


def test_a_creator_assists_the_makes_and_the_shooter_does_not():
    players = _five(1, 10, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    for player, pct in zip(players, (0.45, 0.08, 0.08, 0.08, 0.06)):
        player.ast_pct = pct
    side = _side(players, np.zeros(1))
    n = 6000
    rng = np.random.default_rng(7)
    lineup = np.tile(np.arange(5), (n, 1))
    rates = np.array([ASSIST_RATE[name] for name in PLAY_TYPES])
    spot = np.full(n, PLAY_TYPES.index("Spotup"))
    iso = np.full(n, PLAY_TYPES.index("Isolation"))
    shooter = np.full(n, 4)
    quiet = np.zeros(n, dtype=bool)
    none = np.full(n, -1)

    def draw(type_idx, follow, glass):
        return _pass_chance(
            rng,
            np.repeat(side.ast_pct[None, :], n, axis=0),
            lineup,
            shooter,
            np.full(n, 4),
            type_idx,
            rates,
            follow,
            glass,
        )

    spot_pass, spot_who = draw(spot, quiet, none)
    iso_pass, iso_who = draw(iso, quiet, none)
    assert int(np.sum(spot_who[spot_pass] == 4)) == 0
    assert float(np.mean(spot_pass & (spot_who == 0))) > 0.50
    assert float(np.mean(spot_pass)) > float(np.mean(iso_pass)) * 3
    kick_pass, kick_who = draw(spot, np.ones(n, dtype=bool), np.full(n, 2))
    assert int(np.sum(kick_pass & (kick_who != 2))) == 0
    assert float(np.mean(kick_pass)) > 0.75


def test_the_five_on_the_floor_and_the_team_both_move_the_glass():
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    offense.append(make_player(9, "C5", 1, "C", 24, "Spotup", 8, 1.05, oreb_pct=0.16, dreb_pct=0.18))
    for player in offense[:5]:
        player.oreb_pct = 0.02
    defense = _five(2, 20, ["C", "G", "G", "F", "F"], 1.0, "Spotup")
    for player in defense:
        player.dreb_pct = 0.14
    off = _side(offense, np.ones((1, 6, len(PLAY_TYPES))))
    deff = _side(defense, np.ones((1, 5, len(PLAY_TYPES))))
    off.team_oreb = 0.30
    deff.team_dreb = 0.72
    deff.team_oreb_allowed = 0.28
    defenders = np.array([[0, 1, 2, 3, 4]])
    small = _on_floor_oreb(off, deff, np.array([[0, 1, 2, 3, 4]]), defenders)
    with_big = _on_floor_oreb(off, deff, np.array([[0, 1, 2, 3, 5]]), defenders)
    assert float(with_big[0]) > float(small[0]) + 0.04
    deff.team_dreb = 0.64
    deff.team_oreb_allowed = 0.36
    loose = _on_floor_oreb(off, deff, np.array([[0, 1, 2, 3, 5]]), defenders)
    deff.team_dreb = 0.76
    deff.team_oreb_allowed = 0.22
    stingy = _on_floor_oreb(off, deff, np.array([[0, 1, 2, 3, 5]]), defenders)
    assert float(loose[0]) > float(stingy[0])


def _make_draw(appearances: float, make_form: float = 1.0, make_form_games: float = 0.0, three: float = 0.35):
    offense = _five(1, 1, ["G", "G", "F", "F", "C"], 1.0, "Spotup")
    defense = _five(2, 20, ["C", "G", "G", "F", "F"], 1.0, "Spotup")
    for player in offense:
        player.minutes_appearances = appearances
        player.make_form = make_form
        player.make_form_games = make_form_games
        player.zone_freq = {
            "ra": max(0.0, 1.0 - three),
            "paint": 0.0,
            "mid": 0.0,
            "corner3": 0.0,
            "atb3": three,
        }
        for skill in player.offense.values():
            skill.ppp = 1.05
            skill.tov_poss_pct = 0.10
            skill.ft_poss_pct = 0.08
            skill.fg_pct = 0.46
    n = 80
    off = _side(offense, np.full((n, 5, len(PLAY_TYPES)), 1.05))
    deff = _side(defense, np.ones((n, 5, len(PLAY_TYPES))))
    lineup = np.tile(np.arange(5), (n, 1))
    zeros = np.zeros(n)
    return _draw_side(
        np.random.default_rng(4),
        off,
        lineup,
        deff,
        lineup,
        np.zeros((n, 5)),
        np.ones(len(PLAY_TYPES)),
        np.ones(5),
        np.ones(5),
        np.zeros(n, dtype=bool),
        zeros,
        1.0,
        1.0,
        1,
        zeros,
        zeros,
        np.zeros(n, dtype=bool),
    )


def test_thin_appearances_cool_makes_below_a_full_slate():
    cold = _make_draw(appearances=0.0)
    warm = _make_draw(appearances=20.0)
    assert float(cold["make_p"].mean()) < float(warm["make_p"].mean()) - 0.005
    assert float(cold["three_fg"].mean()) < float(warm["three_fg"].mean())


def test_cold_make_form_shrinks_the_make_path():
    hot = _make_draw(appearances=20.0, make_form=1.0, make_form_games=12.0)
    cold = _make_draw(appearances=20.0, make_form=0.85, make_form_games=12.0)
    assert float(cold["make_p"].mean()) < float(hot["make_p"].mean()) - 0.01
