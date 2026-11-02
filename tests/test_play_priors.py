import pytest

from nba_sim.constants import PLAY_TYPES
from nba_sim.features.factors import play_type_gaps
from nba_sim.features.play_priors import position_rates
from nba_sim.features.prepare import _template_player
from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors, TeamModel


def _team(abbr: str, team_id: int, off_ppp: float = 1.02, off_poss: float = 0.0) -> TeamModel:
    return TeamModel(
        team_id=team_id,
        abbr=abbr,
        name=abbr,
        off_ppp={play: off_ppp for play in PLAY_TYPES},
        off_poss={play: off_poss for play in PLAY_TYPES},
        def_ppp={play: 1.14 for play in PLAY_TYPES},
        def_poss={play: 0.0 for play in PLAY_TYPES},
    )


def test_position_shares_sum_to_one_and_split_guards_from_bigs():
    for position in ("G", "F", "C"):
        shares = [share for share, _ppp, _allowed in position_rates(position).values()]
        assert abs(sum(shares) - 1.0) < 1e-9
    guard = position_rates("G")
    center = position_rates("C")
    assert guard["PRBallHandler"][0] > center["PRBallHandler"][0] * 4
    assert center["Postup"][0] > guard["Postup"][0] * 4
    assert center["OffRebound"][0] > guard["OffRebound"][0] * 4


def test_lineup_chart_does_not_use_one_rate_for_every_play():
    league = LeaguePriors.flat()
    home = _team("PHI", 1)
    away = _team("CLE", 2)
    guards = [
        _template_player({"position": "G", "espn_id": str(i), "name": f"G{i}"}, home, league, 32)
        for i in range(5)
    ]
    bigs = [
        _template_player({"position": "C", "espn_id": str(10 + i), "name": f"C{i}"}, away, league, 32)
        for i in range(5)
    ]
    spec = GameSpec(
        game_id="g",
        date="2026-10-22",
        home=home,
        away=away,
        home_players=guards,
        away_players=bigs,
        context=GameContext(),
        league=league,
    )
    rows = play_type_gaps(spec)
    by = {(row["team"], row["play_type"]): row for row in rows}
    assert rows[0]["source"] == "position prior"
    assert by[("PHI", "PRBallHandler")]["off_share"] > by[("CLE", "PRBallHandler")]["off_share"] + 0.15
    assert by[("CLE", "Postup")]["expected"] > by[("PHI", "Postup")]["expected"] + 0.15
    assert by[("CLE", "OffRebound")]["off_share"] > by[("PHI", "OffRebound")]["off_share"] + 0.08
    phi = [by[("PHI", play)]["expected"] for play in PLAY_TYPES]
    assert max(phi) - min(phi) > 0.15


def test_a_missing_action_keeps_his_defensive_level():
    from nba_sim.features.prepare import _complete_defense
    from nba_sim.sim.specs import PlaySkill

    defense = {"Isolation": PlaySkill(poss=200, ppp=0.90, tov_poss_pct=0.14)}
    team = _team("PHI", 1)
    _complete_defense(defense, team, {}, {}, {}, 1.0, league=LeaguePriors.flat())
    assert defense["Transition"].poss > 0
    assert defense["Transition"].ppp == pytest.approx(0.90)
    assert defense["Transition"].tov_poss_pct == 0.0
    assert defense["Isolation"].ppp == pytest.approx(0.90)
    assert defense["Isolation"].tov_poss_pct == pytest.approx(0.14)


def test_zones_split_the_rim_from_the_three():
    from nba_sim.features.prepare import _complete_defense
    from nba_sim.sim.specs import PlaySkill

    defense = {"Isolation": PlaySkill(poss=300, ppp=1.00, tov_poss_pct=0.10)}
    team = _team("PHI", 1)
    allowed = {"ra": 0.46, "paint": 0.40, "mid": 0.40, "corner3": 0.40, "atb3": 0.40}
    normal = {"ra": 0.66, "paint": 0.42, "mid": 0.41, "corner3": 0.38, "atb3": 0.36}
    fga = {"ra": 400, "paint": 80, "mid": 40, "corner3": 30, "atb3": 50}
    _complete_defense(defense, team, allowed, normal, fga, 1.0, league=LeaguePriors.flat())
    assert defense["Postup"].ppp < defense["Spotup"].ppp


def test_a_thin_defensive_year_keeps_the_season_before_it():
    import pandas as pd

    from nba_sim.features.prepare import _defensive_profile

    rows = []
    for season, poss, ppp in (("2024-25", 200, 0.90), ("2025-26", 30, 1.20)):
        rows.append(
            {
                "season": season,
                "entity_type": "P",
                "entity_id": 7,
                "side": "defensive",
                "play_type": "Isolation",
                "poss": poss,
                "ppp": ppp,
                "fg_pct": 0.45,
                "ft_poss_pct": 0.08,
                "tov_poss_pct": 0.12,
            }
        )
    profile = _defensive_profile(pd.DataFrame(rows), "2026-27", 7, _team("PHI", 1))
    assert profile["Isolation"].poss == pytest.approx(230)
    assert profile["Isolation"].ppp == pytest.approx((1.20 * 30 + 0.90 * 200) / 230)


def test_a_measured_team_rate_holds_over_the_position_prior():
    league = LeaguePriors.flat()
    team = _team("PHI", 1, off_ppp=1.20, off_poss=400)
    team.def_poss = {play: 200.0 for play in PLAY_TYPES}
    player = _template_player({"position": "C", "espn_id": "1", "name": "Big"}, team, league, 32)
    assert player.offense["Postup"].ppp > 1.15
    assert player.defense["Postup"].poss == 0.0
