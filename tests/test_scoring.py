from datetime import date

import numpy as np
import pandas as pd
import pytest

from nba_sim.constants import PLAY_TYPES
from nba_sim.features.factors import _play_profile
from nba_sim.features.prepare import _template_player
from nba_sim.features.scoring import apply_player_scoring, apply_season_minutes, scoring_table
from nba_sim.names import norm_person
from nba_sim.sim.clock import _assign, _touch_weights, _within_minutes
from nba_sim.sim.demo import make_player, make_team, rotation
from nba_sim.sim.engine import simulate_game
from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors, PlayerModel, PlaySkill, TeamModel


def _team() -> TeamModel:
    return TeamModel(
        team_id=1,
        abbr="cle",
        name="Cleveland",
        off_ppp={play: 1.02 for play in PLAY_TYPES},
        off_poss={play: 0.0 for play in PLAY_TYPES},
    )


def _box(**overrides) -> dict:
    row = {
        "usg": 0.20,
        "ft_pct": 0.78,
        "oreb_pct": 0.04,
        "dreb_pct": 0.14,
        "ts": 0.57,
        "ast": 3.0,
        "stl": 0.8,
        "blk": 0.4,
        "pf": 2.2,
        "min": 30.0,
        "gp": 60.0,
        "fga": 12.0,
        "fg3a": 4.0,
        "fg_pct": 0.46,
        "fg3_pct": 0.36,
        "fta": 3.0,
        "three_fga": 0.33,
        "paint_pts": 0.35,
        "mid_pts": 0.12,
    }
    row.update(overrides)
    return row


def _share(player: PlayerModel, plays: tuple[str, ...]) -> float:
    total = sum(skill.poss for skill in player.offense.values())
    return sum(player.offense[play].poss for play in plays) / total


def test_assist_percentage_is_his_share_of_teammate_makes():
    team = _team()
    league = LeaguePriors.flat()
    creator = _template_player({"position": "G", "espn_id": "1", "player_name": "Creator"}, team, league, 34)
    apply_player_scoring([creator], {norm_person("Creator"): _box(ast=10.0, ast_pct=0.44, min=35.0)})
    assert abs(creator.ast_pct - 0.44) < 1e-6
    missing = _template_player({"position": "G", "espn_id": "2", "player_name": "Missing"}, team, league, 30)
    apply_player_scoring([missing], {norm_person("Missing"): _box(ast=6.0, min=30.0)})
    assert 0.30 < missing.ast_pct < 0.36


def test_usage_splits_the_trip_and_a_flat_rate_keeps_the_weights():
    poss = np.array([[[8.0, 4.0, 4.0], [8.0, 4.0, 4.0]]])
    flat = _touch_weights(poss, np.array([[0.20, 0.20]]))
    np.testing.assert_array_equal(flat, poss)
    touch = _touch_weights(poss, np.array([[0.32, 0.14]]))
    assert touch[0, 0].sum() > touch[0, 1].sum() * 2
    assert abs(touch[0, 0, 0] / touch[0, 0].sum() - 0.5) < 1e-6
    assert abs(touch[0, 1, 0] / touch[0, 1].sum() - 0.5) < 1e-6


def _keep_the_mix(player: PlayerModel) -> None:
    for skill in player.offense.values():
        skill.poss_pct = skill.poss


def test_a_play_mix_that_already_matches_his_free_throws_is_left_alone():
    matched = make_player(1, "Matched", 1, "G", 32, "Isolation", 80, 1.05)
    heavier = make_player(2, "Heavier", 1, "G", 32, "Isolation", 80, 1.05)
    short = make_player(3, "Short", 1, "G", 32, "Isolation", 80, 1.05)
    for player in (matched, heavier, short):
        _keep_the_mix(player)
    # The synthetic mix implies 0.20 free throws per shot. His line is that rate.
    apply_player_scoring([matched], {norm_person("Matched"): _box(fga=10.0, fta=2.0, gp=60)})
    assert matched.ft_draw == pytest.approx(1.0, abs=0.02)
    apply_player_scoring([heavier], {norm_person("Heavier"): _box(fga=10.0, fta=4.0, gp=60)})
    assert heavier.ft_draw > matched.ft_draw + 0.5
    apply_player_scoring([short], {norm_person("Short"): _box(fga=10.0, fta=4.0, gp=2)})
    assert abs(short.ft_draw - 1.0) < abs(heavier.ft_draw - 1.0)


def test_a_measured_line_still_carries_his_free_throw_rate():
    player = make_player(1, "Drawer", 1, "G", 32, "Isolation", 80, 1.05)
    player.offense["Isolation"].poss_pct = 0.20
    apply_player_scoring([player], {norm_person("Drawer"): _box(usg=0.28, fga=16.0, fta=8.0, ft_pct=0.80)})
    assert player.ft_draw == pytest.approx(1.85)
    shy = make_player(2, "Shy", 1, "G", 32, "Isolation", 80, 1.05)
    shy.offense["Isolation"].poss_pct = 0.20
    apply_player_scoring([shy], {norm_person("Shy"): _box(usg=0.20, fga=16.0, fta=2.0, ft_pct=0.85)})
    assert shy.ft_draw < 0.70
    assert player.offense["Isolation"].ft_poss_pct == shy.offense["Isolation"].ft_poss_pct


def test_a_scoring_line_sets_usage_free_throws_glass_and_threes():
    team = _team()
    league = LeaguePriors.flat()
    handler = _template_player({"position": "G", "espn_id": "1", "player_name": "Handler"}, team, league, 34)
    shooter = _template_player({"position": "G", "espn_id": "2", "player_name": "Shooter"}, team, league, 26)
    big = _template_player({"position": "C", "espn_id": "3", "player_name": "Big"}, team, league, 30)
    for player in (handler, shooter, big):
        for skill in player.offense.values():
            skill.poss_pct = skill.poss
    apply_player_scoring(
        [handler, shooter, big],
        {
            norm_person("Handler"): _box(usg=0.31, ft_pct=0.86, oreb_pct=0.02, three_fga=0.44, fg3a=8.8, fga=20),
            norm_person("Shooter"): _box(usg=0.16, ft_pct=0.85, oreb_pct=0.02, three_fga=0.78, fg3a=7.2, fga=9.3, fg3_pct=0.42),
            norm_person("Big"): _box(usg=0.19, ft_pct=0.61, oreb_pct=0.10, three_fga=0.02, fg3a=0.2, fga=9.4, fg_pct=0.64),
        },
    )
    assert handler.usg > big.usg + 0.08
    assert shooter.usg < handler.usg - 0.10
    assert big.oreb_pct > handler.oreb_pct * 3
    assert big.ft_pct < handler.ft_pct - 0.15
    assert shooter.zone_freq["atb3"] + shooter.zone_freq["corner3"] > 0.70
    assert big.zone_freq["atb3"] + big.zone_freq["corner3"] < 0.08
    assert abs(handler.usg - 0.20) > 0.05


def test_a_stretch_big_without_a_sample_spots_up():
    team = _team()
    player = _template_player(
        {"position": "C", "espn_id": "9", "player_name": "Stretch"},
        team,
        LeaguePriors.flat(),
        28,
    )
    rim = ("PRRollman", "Postup", "Cut", "OffRebound")
    before = _share(player, rim)
    apply_player_scoring([player], {norm_person("Stretch"): _box(usg=0.18, three_fga=0.62, fg3a=7.0, fga=11.0)})
    after = _share(player, rim)
    spot = _share(player, ("Spotup", "OffScreen", "Handoff"))
    assert before > 0.60
    assert after < before - 0.25
    assert spot > 0.30
    assert player.usg == 0.18


def test_a_measured_mix_stays_with_his_own_shot_profile():
    team = _team()
    player = _template_player(
        {"position": "C", "espn_id": "4", "player_name": "Finisher"},
        team,
        LeaguePriors.flat(),
        32,
    )
    player.offense["PRRollman"].poss = 6.0
    player.offense["PRRollman"].poss_pct = 6.0
    player.offense["Postup"].poss = 4.0
    player.offense["Postup"].poss_pct = 4.0
    apply_player_scoring([player], {norm_person("Finisher"): _box(usg=0.27, three_fga=0.55, ft_pct=0.84)})
    assert player.offense["PRRollman"].poss == 6.0
    assert player.offense["Postup"].poss == 4.0
    assert player.usg == 0.27
    assert player.zone_freq["atb3"] + player.zone_freq["corner3"] > 0.45


def test_the_chart_follows_usage_when_the_mixes_differ():
    def guy(player_id: int, name: str, usg: float, handler: float, spot: float) -> PlayerModel:
        offense = {play: PlaySkill(poss=1.0, ppp=1.0) for play in PLAY_TYPES}
        offense["PRBallHandler"] = PlaySkill(poss=handler, ppp=1.1)
        offense["Spotup"] = PlaySkill(poss=spot, ppp=1.05)
        return PlayerModel(
            player_id=player_id,
            name=name,
            team_id=1,
            position="G",
            minutes_mean=32,
            usg=usg,
            offense=offense,
        )

    profile = _play_profile(
        [guy(1, "Handler", 0.34, 30, 2), guy(2, "Spot", 0.12, 2, 30)],
        "offense",
    )
    assert profile["PRBallHandler"][1] > profile["Spotup"][1]


def test_the_higher_usage_player_scores_from_the_same_actions():
    home = make_team("cle")
    away = make_team("phi")
    high = make_player(1, "High", home.team_id, "G", 34, "PRBallHandler", 20, 1.05, other_poss=4)
    low = make_player(2, "Low", home.team_id, "G", 34, "PRBallHandler", 20, 1.05, other_poss=4)
    high.usg = 0.36
    low.usg = 0.12
    home_players = [high, low]
    for offset, minutes in enumerate((30, 28, 26, 18, 14, 10)):
        player = make_player(
            10 + offset,
            f"Mate{offset}",
            home.team_id,
            "F",
            minutes,
            "Spotup",
            12,
            1.02,
            other_poss=3,
        )
        player.usg = 0.16
        home_players.append(player)
    result = simulate_game(
        GameSpec(
            game_id="usage",
            date="2026-10-22",
            home=home,
            away=away,
            home_players=home_players,
            away_players=rotation(away.team_id, "PHI", 200),
            context=GameContext(),
            league=LeaguePriors.flat(),
        ),
        n_sims=120,
        seed=9,
    )
    high_pts = float(result.player_stats[1]["pts"].mean())
    low_pts = float(result.player_stats[2]["pts"].mean())
    high_fga = float(result.player_stats[1]["fga"].mean())
    low_fga = float(result.player_stats[2]["fga"].mean())
    assert high_pts > low_pts * 1.7
    assert high_fga > low_fga * 1.7


def test_a_share_of_the_schedule_moves_the_depth_slot():
    def guy(
        player_id: int,
        name: str,
        depth: float,
        average: float,
        games: float,
        team_games: float,
    ) -> PlayerModel:
        player = PlayerModel(
            player_id=player_id,
            name=name,
            team_id=1,
            position="G",
            minutes_mean=depth,
            minutes_base=depth,
            minutes_depth=depth,
            minutes_slot=player_id,
            minutes_source="Depth chart",
            available=True,
        )
        player.minutes_appearance_avg = average
        player.minutes_appearances = games
        player.minutes_team_games = team_games
        return player

    short = guy(1, "Short", 35.0, 22.0, 6, 76)
    regular = guy(2, "Regular", 20.0, 31.0, 60, 70)
    rookie = guy(3, "Rookie", 30.0, 0.0, 0.0, 10)
    partial = guy(4, "Partial", 28.0, 18.0, 4, 10)
    apply_season_minutes([short, regular, rookie, partial])
    assert short.minutes_mean == pytest.approx((6 / 76) * 22.0 + (70 / 76) * 35.0)
    assert short.minutes_source == "Depth chart"
    assert regular.minutes_mean == pytest.approx((60 / 70) * 31.0 + (10 / 70) * 20.0)
    assert regular.minutes_source == "Season average"
    assert short.minutes_mean > regular.minutes_mean
    assert rookie.minutes_mean == pytest.approx(30.0)
    assert rookie.minutes_source == "Depth chart"
    assert partial.minutes_mean == pytest.approx(0.4 * 18.0 + 0.6 * 28.0)
    assert partial.minutes_source == "Depth chart"


def test_a_long_night_stays_and_a_short_night_sits():
    players = [
        PlayerModel(player_id=index, name=f"P{index}", team_id=1, position="G", minutes_mean=minutes, available=True)
        for index, minutes in enumerate((32.0, 30.0, 28.0, 24.0, 18.0, 14.0))
    ]
    plan = np.zeros((2, len(players)))
    plan[:, 0] = 30.0 * 60.0
    target = np.tile(np.array([32.0, 30.0, 28.0, 24.0, 18.0, 14.0]), (2, 1))
    target[0, 0] = 28.0
    target[1, 0] = 38.0
    lines = _assign([0, 1, 2, 3, 4], players, plan, target, stick=False)
    assert 0 not in lines[0]
    assert 5 in lines[0]
    assert 0 in lines[1]
    assert 5 not in lines[1]


def test_a_guard_is_replaced_by_a_guard_who_has_played_with_the_group():
    players = [
        PlayerModel(player_id=0, name="Out", team_id=1, position="G", minutes_mean=32.0, available=True),
        PlayerModel(player_id=1, name="Mate", team_id=1, position="G", minutes_mean=30.0, available=True),
        PlayerModel(player_id=2, name="Wing", team_id=1, position="F", minutes_mean=28.0, available=True),
        PlayerModel(player_id=3, name="Four", team_id=1, position="F", minutes_mean=24.0, available=True),
        PlayerModel(player_id=4, name="Big", team_id=1, position="C", minutes_mean=22.0, available=True),
        PlayerModel(player_id=5, name="Center", team_id=1, position="C", minutes_mean=20.0, available=True),
        PlayerModel(player_id=6, name="Stranger", team_id=1, position="G", minutes_mean=18.0, available=True),
        PlayerModel(
            player_id=7,
            name="Backup",
            team_id=1,
            position="G",
            minutes_mean=14.0,
            available=True,
            shared_minutes={1: 220.0, 2: 220.0, 3: 220.0, 4: 220.0},
        ),
        PlayerModel(player_id=8, name="Spot", team_id=1, position="G", minutes_mean=3.0, available=True),
    ]
    plan = np.zeros((1, len(players)))
    plan[0, 0] = 34.0 * 60.0
    target = np.array([[32.0, 30.0, 28.0, 24.0, 22.0, 20.0, 18.0, 14.0, 3.0]])
    lines = _assign([0, 1, 2, 3, 4], players, plan, target, stick=False)
    assert lines[0].tolist()[0] == 7 or 7 in lines[0].tolist()
    assert 5 not in lines[0]
    assert 6 not in lines[0]
    assert 8 not in lines[0]


def test_a_player_leaves_once_the_script_reaches_his_minutes():
    players = [
        PlayerModel(player_id=index, name=f"P{index}", team_id=1, position="G", minutes_mean=minutes, available=True)
        for index, minutes in enumerate((22.0, 34.0, 32.0, 18.0, 16.0, 14.0))
    ]
    plan = np.zeros((1, len(players)))
    plan[0, 0] = 24.0 * 60.0
    unit = _within_minutes([0, 1, 2, 3, 4], players, plan)
    assert 0 not in unit
    assert 5 in unit
    assert _within_minutes([0, 1, 2, 3, 4], players, np.zeros((1, len(players)))) == [0, 1, 2, 3, 4]


def test_a_short_season_leans_on_last_year():
    from nba_sim.features.scoring import _blend_scoring

    older = {"usg": 0.30, "gp": 70.0, "min": 34.0, "ast_pct": 0.40}
    newer = {"usg": 0.10, "gp": 2.0, "min": 20.0, "ast_pct": 0.10}
    mixed = _blend_scoring(older, newer)
    trust = 2.0 / 10.0
    assert mixed["usg"] == pytest.approx((1.0 - trust) * 0.30 + trust * 0.10)
    assert mixed["usg"] < 0.28
    assert mixed["gp"] == 2.0
    full = _blend_scoring(older, {"usg": 0.22, "gp": 70.0, "min": 34.0})
    assert full["usg"] == pytest.approx(0.22)


def test_a_later_season_stays_out_of_an_earlier_slate(tmp_path):
    from nba_sim.warehouse import Warehouse

    warehouse = Warehouse(tmp_path)
    warehouse.write(
        "raw_player_Base_2025-26",
        pd.DataFrame([{"PLAYER_ID": 1, "TEAM_ID": 1, "PLAYER_NAME": "Tyrese Haliburton", "MIN": 34.0, "GP": 70, "AST": 9.0}]),
    )
    warehouse.write(
        "raw_player_Advanced_2025-26",
        pd.DataFrame([{"PLAYER_ID": 1, "TEAM_ID": 1, "USG_PCT": 0.21, "AST_PCT": 0.40}]),
    )
    warehouse.write(
        "raw_player_Base_2026-27",
        pd.DataFrame([{"PLAYER_ID": 1, "TEAM_ID": 1, "PLAYER_NAME": "Tyrese Haliburton", "MIN": 32.0, "GP": 2, "AST": 8.0}]),
    )
    warehouse.write(
        "raw_player_Advanced_2026-27",
        pd.DataFrame([{"PLAYER_ID": 1, "TEAM_ID": 1, "USG_PCT": 0.05, "AST_PCT": 0.10}]),
    )
    key = norm_person("Tyrese Haliburton")
    locked = scoring_table(warehouse, through="2025-26")
    live = scoring_table(warehouse, through="2026-27")
    assert locked[key]["usg"] == pytest.approx(0.21)
    trust = 2.0 / 10.0
    assert live[key]["usg"] == pytest.approx((1.0 - trust) * 0.21 + trust * 0.05)


def test_a_new_final_asks_for_a_refresh(tmp_path):
    from nba_sim.ingest.nba_stats import slate_refresh_due
    from nba_sim.warehouse import Warehouse

    warehouse = Warehouse(tmp_path)
    assert slate_refresh_due(warehouse, "2026-27", []) is True
    for name in ("raw_player_Base_2026-27", "raw_team_Advanced_2026-27", "raw_play_offensive_T_Spotup_2026-27"):
        warehouse.write(name, pd.DataFrame({"x": [1]}))
    warehouse.write("game_logs", pd.DataFrame({"entity_type": ["T", "T"], "date": ["2026-10-22", "2026-10-22"]}))
    assert slate_refresh_due(warehouse, "2026-27", [date(2026, 10, 22)]) is False
    assert slate_refresh_due(warehouse, "2026-27", [date(2026, 10, 22), date(2026, 10, 23)]) is True
    assert slate_refresh_due(warehouse, "2026-27", [date(2026, 10, 22), date(2026, 10, 22)]) is True


def test_the_warehouse_scoring_table_is_player_specific():
    table = scoring_table()
    if not table:
        return
    mitchell = table[norm_person("Donovan Mitchell")]
    merrill = table[norm_person("Sam Merrill")]
    allen = table[norm_person("Jarrett Allen")]
    embiid = table[norm_person("Joel Embiid")]
    assert mitchell["usg"] > merrill["usg"] + 0.08
    assert mitchell["usg"] > allen["usg"] + 0.08
    assert embiid["usg"] > 0.30
    assert allen["three_fga"] < 0.08
    assert merrill["three_fga"] > 0.60
    assert allen["oreb_pct"] > mitchell["oreb_pct"] * 2
    assert allen["ft_pct"] < mitchell["ft_pct"] - 0.10
