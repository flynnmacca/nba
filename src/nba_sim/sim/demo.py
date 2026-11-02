"""Synthetic slates and the matchup scenarios the tests lock in."""

from __future__ import annotations

from datetime import date, timedelta

from nba_sim.constants import DEFAULT_ZONE_FG, DEFAULT_ZONE_FREQ, PLAY_TYPES
from nba_sim.features.context import TeamTrip, build_context
from nba_sim.sim.specs import (
    GameContext,
    GameSpec,
    LeaguePriors,
    MarketQuote,
    PlayerModel,
    PlaySkill,
    TeamModel,
)
from nba_sim.teams import TEAM_ID_BY_ABBR, team_name


def skill(poss: float, ppp: float, tov: float = 0.12, ft: float = 0.08) -> PlaySkill:
    return PlaySkill(
        poss=poss,
        ppp=ppp,
        ft_poss_pct=ft,
        tov_poss_pct=tov,
        fg_pct=0.47,
        score_poss_pct=0.50,
        efg_pct=0.53,
    )


def offense_profile(
    main: str,
    main_poss: float,
    main_ppp: float,
    other_poss: float = 8.0,
    other_ppp: float = 1.02,
) -> dict[str, PlaySkill]:
    profile: dict[str, PlaySkill] = {}
    for play in PLAY_TYPES:
        if play == main:
            profile[play] = skill(main_poss, main_ppp)
        else:
            profile[play] = skill(other_poss, other_ppp)
    return profile


def make_player(
    player_id: int,
    name: str,
    team_id: int,
    position: str,
    minutes: float,
    main: str,
    main_poss: float,
    main_ppp: float,
    oreb_pct: float = 0.04,
    dreb_pct: float = 0.14,
    other_poss: float = 8.0,
    ast_share: float = 1.0,
    stl_weight: float = 1.0,
    blk_weight: float = 0.4,
) -> PlayerModel:
    return PlayerModel(
        player_id=player_id,
        name=name,
        team_id=team_id,
        position=position,
        minutes_mean=minutes,
        minutes_std=1.5,
        offense=offense_profile(main, main_poss, main_ppp, other_poss=other_poss),
        oreb_pct=oreb_pct,
        dreb_pct=dreb_pct,
        ast_share=ast_share,
        stl_weight=stl_weight,
        blk_weight=blk_weight,
        zone_freq=dict(DEFAULT_ZONE_FREQ),
        zone_fg=dict(DEFAULT_ZONE_FG),
    )


def make_team(
    abbr: str,
    def_ppp: dict[str, float] | None = None,
    off_ppp: dict[str, float] | None = None,
    off_rating: float = 114.0,
    def_rating: float = 114.0,
    pace: float = 99.5,
) -> TeamModel:
    league = 1.02
    defense = {play: league for play in PLAY_TYPES}
    offense = {play: league for play in PLAY_TYPES}
    if def_ppp:
        defense.update(def_ppp)
    if off_ppp:
        offense.update(off_ppp)
    poss = {play: 800.0 for play in PLAY_TYPES}
    return TeamModel(
        team_id=TEAM_ID_BY_ABBR[abbr],
        abbr=abbr,
        name=team_name(abbr),
        pace=pace,
        off_rating=off_rating,
        def_rating=def_rating,
        games=60,
        off_ppp=offense,
        off_poss=dict(poss),
        def_ppp=defense,
        def_poss=dict(poss),
        zone_d_fg=dict(DEFAULT_ZONE_FG),
        zone_normal_fg=dict(DEFAULT_ZONE_FG),
    )


def rotation(
    team_id: int,
    prefix: str,
    id_start: int,
    handler_ppp: float = 1.05,
    handler_poss: float = 80.0,
    oreb: float = 0.045,
) -> list[PlayerModel]:
    specs = [
        ("Handler", "G", 36, "PRBallHandler", handler_poss, handler_ppp, 0.02, 0.08, 2.2, 1.4, 0.2),
        ("Creator", "G", 34, "Isolation", 70, 1.04, 0.02, 0.09, 1.6, 1.2, 0.2),
        ("Wing", "F", 32, "Spotup", 60, 1.08, 0.03, 0.12, 1.0, 1.0, 0.4),
        ("Forward", "F", 30, "PRRollman", 55, 1.06, 0.07, 0.16, 0.8, 0.8, 0.7),
        ("Big", "C", 28, "Postup", 45, 1.04, oreb, 0.22, 0.5, 0.6, 1.3),
        ("Guard", "G", 26, "Handoff", 30, 1.0, 0.02, 0.08, 1.1, 0.9, 0.2),
        ("Wing2", "F", 28, "Cut", 28, 1.02, 0.05, 0.14, 0.6, 0.7, 0.5),
        ("Big2", "C", 26, "OffRebound", 24, 1.0, oreb, 0.18, 0.4, 0.5, 0.9),
    ]
    players = []
    for offset, row in enumerate(specs):
        name, pos, minutes, play, poss, ppp, oreb_pct, dreb_pct, ast, stl, blk = row
        players.append(
            make_player(
                id_start + offset,
                f"{prefix} {name}",
                team_id,
                pos,
                minutes,
                play,
                poss,
                ppp,
                oreb_pct=oreb_pct,
                dreb_pct=dreb_pct,
                other_poss=6.0,
                ast_share=ast,
                stl_weight=stl,
                blk_weight=blk,
            )
        )
    return players


def pnr_matchup(defense: str, n_label: str = "den") -> GameSpec:
    """Same offensive handler. Only the opponent's ball-screen defense changes.

    League PPP is identical in the weak and strong versions, so a model that
    ignores the opponent and uses the league rate cannot separate them.
    """
    allowed = 1.32 if defense == "weak" else 0.74
    offense = make_team("den", off_rating=112.0, def_rating=114.0)
    defense_team = make_team(
        "mia",
        def_ppp={"PRBallHandler": allowed},
        off_rating=112.0,
        def_rating=114.0,
    )
    players = rotation(offense.team_id, "DEN", 1000, handler_ppp=1.16, handler_poss=900.0)
    # The handler's other actions are a small share of the lineup's possessions.
    for play, skill_row in players[0].offense.items():
        if play != "PRBallHandler":
            skill_row.poss = 4.0
    opponents = rotation(defense_team.team_id, "MIA", 2000, handler_ppp=1.02, handler_poss=40.0)
    return GameSpec(
        game_id=f"pnr-{defense}-{n_label}",
        date="2026-10-22",
        home=offense,
        away=defense_team,
        home_players=players,
        away_players=opponents,
        context=GameContext(),
        league=LeaguePriors.flat(1.02),
        source="scenario",
    )


def oreb_matchup(glass: str) -> GameSpec:
    """Same shooting profile. Only on-court offensive rebound rates change."""
    oreb = 0.09 if glass == "high" else 0.008
    home = make_team("den", off_rating=114.0, def_rating=114.0)
    away = make_team("mia", off_rating=114.0, def_rating=114.0)
    home_players = rotation(home.team_id, "DEN", 3000, oreb=oreb)
    for player in home_players:
        player.oreb_pct = oreb
        player.box_outs = 6.0 if glass == "high" else 1.0
    away_players = rotation(away.team_id, "MIA", 4000, oreb=0.03)
    for player in away_players:
        player.dreb_pct = 0.15
    return GameSpec(
        game_id=f"oreb-{glass}",
        date="2026-10-22",
        home=home,
        away=away,
        home_players=home_players,
        away_players=away_players,
        context=GameContext(),
        league=LeaguePriors.flat(1.02),
        source="scenario",
    )


def build_demo_slate(on: date | None = None) -> list[GameSpec]:
    """Two games with real schedule effects and a labeled demo price board."""
    on = on or date(2026, 10, 22)
    denver = make_team(
        "den",
        off_ppp={"PRBallHandler": 1.12, "PRRollman": 1.14, "Spotup": 1.08, "Postup": 1.10},
        def_ppp={"PRBallHandler": 0.98, "Isolation": 0.96, "Spotup": 1.00},
        off_rating=116.0,
        def_rating=112.0,
        pace=100.5,
    )
    miami = make_team(
        "mia",
        off_ppp={"Isolation": 1.08, "Spotup": 1.06, "Handoff": 1.05},
        def_ppp={"PRBallHandler": 1.16, "PRRollman": 1.12, "Isolation": 0.94, "Spotup": 1.04},
        off_rating=112.0,
        def_rating=113.0,
        pace=97.5,
    )
    boston = make_team(
        "bos",
        off_ppp={"Spotup": 1.14, "OffScreen": 1.10, "Handoff": 1.08, "PRBallHandler": 1.06},
        def_ppp={"PRBallHandler": 0.92, "Spotup": 0.95, "Isolation": 0.93, "Postup": 0.97},
        off_rating=118.0,
        def_rating=110.0,
        pace=98.0,
    )
    knicks = make_team(
        "nyk",
        off_ppp={"Isolation": 1.10, "Postup": 1.08, "PRBallHandler": 1.04},
        def_ppp={"Isolation": 1.08, "PRBallHandler": 1.02, "Spotup": 1.06, "Postup": 0.99},
        off_rating=114.0,
        def_rating=112.5,
        pace=96.5,
    )
    den_players = rotation(denver.team_id, "DEN", 5100, handler_ppp=1.12, handler_poss=95, oreb=0.07)
    mia_players = rotation(miami.team_id, "MIA", 5200, handler_ppp=1.02, handler_poss=75, oreb=0.035)
    bos_players = rotation(boston.team_id, "BOS", 5300, handler_ppp=1.08, handler_poss=85, oreb=0.04)
    nyk_players = rotation(knicks.team_id, "NYK", 5400, handler_ppp=1.05, handler_poss=80, oreb=0.05)
    den_players[0].clutch_mult = 1.06
    bos_players[0].clutch_mult = 1.04

    altitude_game = GameSpec(
        game_id=f"demo-{on.isoformat()}-den-mia",
        date=on.isoformat(),
        home=denver,
        away=miami,
        home_players=den_players,
        away_players=mia_players,
        context=build_context(
            denver,
            miami,
            on,
            TeamTrip(
                prev_date=on - timedelta(days=3),
                prev_lat=39.75,
                prev_lon=-105.01,
                prev_tz=-6,
                games_in_last_4=1,
                measured_home_court=3.1,
                home_court_games=50,
                form_mult=1.03,
                form_games=6,
            ),
            TeamTrip(
                prev_date=on - timedelta(days=1),
                prev_lat=25.78,
                prev_lon=-80.19,
                prev_tz=-4,
                games_in_last_4=3,
                form_mult=0.98,
                form_games=6,
            ),
        ),
        league=LeaguePriors.flat(1.02),
        quotes=[
            MarketQuote("demo", "moneyline", "home", -150),
            MarketQuote("demo", "moneyline", "away", 130),
            MarketQuote("demo", "spread", "home", -110, line=-4.5),
            MarketQuote("demo", "spread", "away", -110, line=4.5),
            MarketQuote("demo", "total", "over", -110, line=224.5),
            MarketQuote("demo", "total", "under", -110, line=224.5),
            MarketQuote(
                "demo",
                "player_points",
                "over",
                -115,
                line=26.5,
                player_id=den_players[0].player_id,
                player_name=den_players[0].name,
            ),
            MarketQuote(
                "demo",
                "player_points",
                "under",
                -105,
                line=26.5,
                player_id=den_players[0].player_id,
                player_name=den_players[0].name,
            ),
        ],
        source="demo",
    )
    rested = GameSpec(
        game_id=f"demo-{on.isoformat()}-bos-nyk",
        date=on.isoformat(),
        home=boston,
        away=knicks,
        home_players=bos_players,
        away_players=nyk_players,
        context=build_context(
            boston,
            knicks,
            on,
            TeamTrip(
                prev_date=on - timedelta(days=2),
                prev_lat=42.37,
                prev_lon=-71.06,
                prev_tz=-4,
                games_in_last_4=1,
                measured_home_court=2.4,
                home_court_games=40,
            ),
            TeamTrip(
                prev_date=on - timedelta(days=2),
                prev_lat=40.75,
                prev_lon=-73.99,
                prev_tz=-4,
                games_in_last_4=1,
            ),
        ),
        league=LeaguePriors.flat(1.02),
        quotes=[
            MarketQuote("demo", "moneyline", "home", -175),
            MarketQuote("demo", "moneyline", "away", 150),
            MarketQuote("demo", "spread", "home", -108, line=-5.5),
            MarketQuote("demo", "spread", "away", -112, line=5.5),
            MarketQuote("demo", "total", "over", -105, line=219.5),
            MarketQuote("demo", "total", "under", -115, line=219.5),
        ],
        source="demo",
    )
    return [altitude_game, rested]
