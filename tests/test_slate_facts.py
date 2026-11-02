"""Schedule dates, tip clocks, roster membership, injury rules, and sim histograms."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from nba_sim.features.coaches import coach_for, depth_minutes
from nba_sim.features.prepare import _games_on, apply_injuries, build_players
from nba_sim.geo import local_clock
from nba_sim.ingest.espn import espn_slug, injury_decision, parse_event, parse_injuries, parse_roster
from nba_sim.sim.specs import LeaguePriors, PlayerModel, TeamModel


def test_game_spread_counts_every_simulation_and_keeps_the_posted_line():
    from nba_sim.report import spread_view, summarize
    from nba_sim.sim.engine import SimResult
    from nba_sim.sim.specs import GameContext, GameSpec, LeaguePriors, MarketQuote, TeamModel

    home = np.array([110, 100, 108, 120], dtype=float)
    away = np.array([100, 106, 100, 110], dtype=float)
    result = SimResult(
        home_points=home,
        away_points=away,
        player_stats={},
        home_ids=[],
        away_ids=[],
        second_chance_home=np.zeros(4),
        second_chance_away=np.zeros(4),
        n_sims=4,
        seed=1,
    )
    spec = GameSpec(
        game_id="1",
        date="2026-10-22",
        home=TeamModel(team_id=1, abbr="phi", name="Philadelphia 76ers"),
        away=TeamModel(team_id=2, abbr="mia", name="Miami Heat"),
        home_players=[],
        away_players=[],
        context=GameContext(),
        league=LeaguePriors.flat(),
        quotes=[
            MarketQuote("demo", "spread", "home", -110, line=-4.5),
            MarketQuote("demo", "spread", "away", -110, line=4.5),
        ],
    )
    summary = summarize(spec, result)
    counts = {row["value"]: row["sims"] for row in summary["team"]["spread_counts"]}
    assert sum(counts.values()) == 4
    assert counts[-6] == 1
    assert counts[8] == 1
    assert counts[10] == 2
    assert counts[0] == 0
    view = spread_view(summary)
    assert view["source"] == "line"
    assert view["home_text"] == "-4.5"
    assert view["away_text"] == "+4.5"
    assert view["line_text"] == "PHI -4.5"
    assert view["threshold"] == 4.5
    assert view["sim_text"] == "PHI by 5.5"


def test_spread_view_names_an_away_favorite_and_falls_back_to_the_sim():
    from nba_sim.report import spread_view

    summary = {
        "home": "PHI",
        "away": "MIA",
        "team": {"spread_mean": 2.2},
        "prices": [{"market": "spread", "side": "home", "line": 6.5}],
    }
    view = spread_view(summary)
    assert view["home_text"] == "+6.5"
    assert view["away_text"] == "-6.5"
    assert view["line_text"] == "MIA -6.5"
    assert view["threshold"] == -6.5
    assert view["sim_text"] == "PHI by 2.2"

    summary["prices"] = [{"market": "spread", "side": "home", "line": 0.0}]
    pick = spread_view(summary)
    assert pick["home_text"] == "PK"
    assert pick["away_text"] == "PK"
    assert pick["line_text"] == "Pick"

    summary["prices"] = []
    summary["team"]["spread_mean"] = -3.24
    sim = spread_view(summary)
    assert sim["source"] == "sim"
    assert sim["threshold"] is None
    assert sim["home_text"] == "+3.2"
    assert sim["away_text"] == "-3.2"
    assert sim["sim_text"] == "MIA by 3.2"


def test_prop_columns_sort_by_the_number_not_the_first_digit():
    from nba_sim.app.slate import _prop_table

    summary = {
        "player_bins": [
            {"espn_id": "1", "player": "Low", "team": "PHI", "stat": "pts", "value": 10, "sims": 95},
            {"espn_id": "1", "player": "Low", "team": "PHI", "stat": "pts", "value": 30, "sims": 5},
            {"espn_id": "2", "player": "High", "team": "PHI", "stat": "pts", "value": 0, "sims": 55},
            {"espn_id": "2", "player": "High", "team": "PHI", "stat": "pts", "value": 10, "sims": 45},
        ]
    }
    props = pd.DataFrame(
        [
            {
                "espn_id": "1",
                "stat": "pts",
                "market": "Points",
                "line": 20.5,
                "fair_over": 0.05,
                "over_american": -110,
                "under_american": -110,
                "main": True,
            },
            {
                "espn_id": "2",
                "stat": "pts",
                "market": "Points",
                "line": 4.5,
                "fair_over": 0.90,
                "over_american": 145,
                "under_american": -170,
                "main": True,
            },
        ]
    )
    shown = _prop_table(summary, props)
    low = shown[shown["Player"] == "Low"].iloc[0]
    assert low["Side"] == "Over"
    assert low["Sim"] == 5.0
    assert low["Fair"] == 5.0
    assert low["Edge"] == 0.0
    assert low["Price"] == -110.0
    high = shown[shown["Player"] == "High"].iloc[0]
    assert high["Side"] == "Under"
    assert high["Sim"] == 55.0
    assert high["Fair"] == 10.0
    assert high["Edge"] == 45.0
    assert high["Price"] == -170.0
    assert list(shown.sort_values("Sim")["Sim"]) == [5.0, 55.0]
    assert list(shown.sort_values("Line")["Line"]) == [4.5, 20.5]
    assert list(shown.sort_values("Price")["Price"]) == [-170.0, -110.0]


def test_each_simulation_lands_in_its_own_bin():
    from nba_sim.report import _integer_counts

    rows = _integer_counts(np.array([0, 0, 2, 2, 2]), start_at_zero=True)
    assert [(row["value"], row["sims"]) for row in rows] == [(0, 2), (1, 0), (2, 3)]
    assert sum(row["sims"] for row in rows) == 5


def test_melbourne_today_is_the_day_after_a_late_utc_afternoon():
    from datetime import datetime, timezone

    from nba_sim.geo import melbourne_tip_date, melbourne_today

    assert melbourne_today(datetime(2026, 10, 22, 13, 17, tzinfo=timezone.utc)) == date(2026, 10, 23)
    assert melbourne_today(datetime(2026, 10, 22, 12, 0, tzinfo=timezone.utc)) == date(2026, 10, 22)
    assert melbourne_tip_date("2026-10-22T23:00:00Z") == date(2026, 10, 23)
    assert melbourne_tip_date("2026-10-23T20:00:00Z") == date(2026, 10, 24)


def test_tip_clocks_use_the_arena_and_melbourne():
    slate, local, melbourne = local_clock("2026-10-22T23:00:00Z", "phi")
    assert slate == "2026-10-22"
    assert local == "Thu 22 Oct, 7:00 PM EDT"
    assert melbourne == "Fri 23 Oct, 10:00 AM AEDT"

    slate, local, melbourne = local_clock("2026-10-23T01:30:00Z", "okc")
    assert slate == "2026-10-22"
    assert local == "Thu 22 Oct, 8:30 PM CDT"
    assert melbourne == "Fri 23 Oct, 12:30 PM AEDT"

    _, phoenix, _ = local_clock("2027-01-15T03:00:00Z", "phx")
    assert phoenix == "Thu 14 Jan, 8:00 PM MST"


def test_espn_slugs_for_new_orleans_and_utah():
    assert espn_slug("nop") == "no"
    assert espn_slug("uta") == "utah"
    assert espn_slug("phi") == "phi"


def test_late_tip_stays_on_the_venue_date():
    parsed = parse_event(
        {
            "id": "401",
            "competitions": [
                {
                    "id": "401",
                    "date": "2026-10-23T01:30:00Z",
                    "neutralSite": False,
                    "venue": {
                        "fullName": "Paycom Center",
                        "address": {"city": "Oklahoma City", "state": "OK"},
                    },
                    "competitors": [
                        {"homeAway": "away", "team": {"abbreviation": "DEN"}},
                        {"homeAway": "home", "team": {"abbreviation": "OKC"}},
                    ],
                    "status": {"type": {"description": "Scheduled", "shortDetail": "10/22 - 9:30 PM EDT"}},
                }
            ],
        }
    )
    assert parsed is not None
    assert parsed["date"] == "2026-10-22"
    assert parsed["home"] == "okc"
    assert parsed["away"] == "den"
    assert parsed["venue"] == "Paycom Center"
    assert "8:30 PM CDT" in parsed["tip_local"]
    assert "12:30 PM AEDT" in parsed["tip_melbourne"]

    schedule = pd.DataFrame([parsed])
    assert len(_games_on(date(2026, 10, 22), schedule, None)) == 1
    assert _games_on(date(2026, 10, 23), schedule, None) == []


def test_injury_report_uses_status_return_date_and_roster_names():
    frame = parse_injuries(
        {
            "injuries": [
                {
                    "displayName": "Atlanta Hawks",
                    "injuries": [
                        {
                            "status": "Out",
                            "shortComment": "Gueye is out.",
                            "date": "2026-10-21T12:00:00Z",
                            "athlete": {"id": "1", "displayName": "Mouhamed Gueye"},
                            "details": {
                                "type": "Foot",
                                "detail": "Fracture",
                                "returnDate": "2026-11-20",
                                "fantasyStatus": {"abbreviation": "OUT", "description": "Out"},
                            },
                        },
                        {
                            "status": "Day-To-Day",
                            "shortComment": "Bilodeau will miss the opener.",
                            "date": "2026-10-21T15:00:00Z",
                            "athlete": {"id": "2", "displayName": "Someone Bilodeau"},
                            "details": {
                                "type": "Shoulder",
                                "returnDate": "2026-10-24",
                                "fantasyStatus": {"abbreviation": "GTD"},
                            },
                        },
                        {
                            "status": "Out",
                            "date": "2026-10-21T15:00:00Z",
                            "athlete": {"id": "3", "displayName": "Season Veesaar"},
                            "details": {
                                "type": "Knee",
                                "returnDate": "2027-07-01",
                                "fantasyStatus": {"abbreviation": "OFS"},
                            },
                        },
                        {
                            "status": "Day-To-Day",
                            "shortComment": "Knee soreness.",
                            "date": "2026-10-21T15:00:00Z",
                            "athlete": {"id": "4", "displayName": "Nikola Jokić"},
                            "details": {"type": "Knee", "fantasyStatus": {"abbreviation": "GTD"}},
                        },
                    ],
                }
            ]
        }
    )
    assert frame.iloc[0]["body_part"] == "Foot"
    assert frame.iloc[0]["status"] == "Out"
    assert injury_decision(frame.iloc[0].to_dict(), date(2026, 10, 22)) == ("Out", 0.0)
    label, multiplier = injury_decision(frame.iloc[1].to_dict(), date(2026, 10, 22))
    assert multiplier == 0.0
    assert label == "Out until Oct 24"
    assert injury_decision(frame.iloc[2].to_dict(), date(2026, 10, 22)) == ("Out for season", 0.0)
    assert injury_decision({"status": "Rest"}, date(2026, 10, 22)) == ("Out", 0.0)
    assert injury_decision({"status": "Inactive"}, date(2026, 10, 22)) == ("Out", 0.0)

    team = TeamModel(team_id=1, abbr="atl", name="Atlanta Hawks")
    roster = pd.DataFrame(
        [
            {
                "team_abbr": "atl",
                "espn_id": "4",
                "player_name": "Nikola Jokic",
                "position": "C",
                "experience": 9,
                "roster_status": "Active",
            },
            {
                "team_abbr": "atl",
                "espn_id": "9",
                "player_name": "Roster Only",
                "position": "G",
                "experience": 1,
                "roster_status": "Active",
            },
        ]
    )
    players = build_players(
        "atl",
        "2026-27",
        team,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        LeaguePriors.flat(),
        roster=roster,
    )
    assert [player.name for player in players] == ["Nikola Jokic", "Roster Only"]
    flags, display = apply_injuries(players, frame, "atl", date(2026, 10, 22))
    jokic = players[0]
    other = players[1]
    assert other.minutes_source == "Depth chart"
    assert jokic.foul_rate > 0
    assert jokic.injury_factor == 0.65
    assert jokic.minutes_mean == pytest.approx(jokic.minutes_base * 0.65)
    assert jokic.minutes_slot == 1
    assert jokic.injury_label == "Day-to-day"
    assert jokic.injury_factor == 0.65
    assert any("Nikola Jokic" in flag and "Day-to-day" in flag for flag in flags)
    assert any(row["player"] == "Season Veesaar" and row["label"] == "Out for season" and not row["matched"] for row in display)
    assert any(row["player"] == "Someone Bilodeau" and row["label"] == "Out until Oct 24" for row in display)


def test_injury_id_comes_from_the_player_link():
    frame = parse_injuries(
        {
            "injuries": [
                {
                    "displayName": "Atlanta Hawks",
                    "injuries": [
                        {
                            "status": "Out",
                            "athlete": {
                                "displayName": "Mouhamed Gueye",
                                "links": [
                                    {"href": "https://www.espn.com/nba/player/_/id/4712863/mouhamed-gueye"}
                                ],
                            },
                            "details": {"type": "Foot", "fantasyStatus": {"abbreviation": "OUT"}},
                        }
                    ],
                }
            ]
        }
    )
    assert frame.iloc[0]["athlete_id"] == "4712863"
    assert frame.iloc[0]["body_part"] == "Foot"


def test_shared_last_name_stays_on_the_listed_player():
    team = TeamModel(team_id=1, abbr="den", name="Denver Nuggets")
    roster = pd.DataFrame(
        [
            {
                "team_abbr": "den",
                "espn_id": "1",
                "player_name": "Tyus Jones",
                "position": "G",
                "experience": 10,
                "salary": 1,
                "depth_rank": 2,
                "roster_status": "Active",
            },
            {
                "team_abbr": "den",
                "espn_id": "2",
                "player_name": "Spencer Jones",
                "position": "F",
                "experience": 1,
                "salary": 1,
                "depth_rank": 1,
                "roster_status": "Active",
            },
        ]
    )
    players = build_players(
        "den",
        "2026-27",
        team,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        LeaguePriors.flat(),
        roster=roster,
    )
    injuries = pd.DataFrame(
        [
            {
                "team_abbr": "den",
                "player_name": "Spencer Jones",
                "athlete_id": "2",
                "status": "Day-To-Day",
                "fantasy": "GTD",
                "body_part": "Hamstring",
                "comment": "Jones (hamstring) is questionable.",
                "return_date": "2026-10-22",
                "report_date": "2026-10-21",
                "detail": "Jones (hamstring) is questionable.",
            }
        ]
    )
    apply_injuries(players, injuries, "den", date(2026, 10, 22))
    by_name = {player.name: player for player in players}
    spencer = by_name["Spencer Jones"]
    assert spencer.injury_factor == 0.65
    assert spencer.minutes_mean == pytest.approx(spencer.minutes_base * 0.65)
    assert by_name["Tyus Jones"].injury_label == ""


def test_a_pre_tip_report_matches_the_chart_team():
    player = PlayerModel(
        player_id=7,
        name="Fred Van",
        team_id=1,
        position="G",
        minutes_mean=32.0,
        minutes_base=32.0,
        available=True,
    )

    def report(team: str, reported: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "team_abbr": team,
                    "player_name": "Fred Van",
                    "athlete_id": "",
                    "status": "Out",
                    "fantasy": "",
                    "comment": "",
                    "return_date": "",
                    "report_date": reported,
                }
            ]
        )

    apply_injuries([player], report("hou", "2026-10-22"), "hou", date(2026, 10, 22))
    assert player.available
    assert player.injury_label == ""
    apply_injuries([player], report("lal", "2026-10-21"), "hou", date(2026, 10, 22))
    assert player.available
    assert player.injury_label == ""
    apply_injuries([player], report("HOU", "2026-10-21"), "hou", date(2026, 10, 22))
    assert player.available is False
    assert player.minutes_mean == pytest.approx(0.0)
    # An upcoming night keeps the same-day report. A finished night still drops it.
    player.available = True
    player.minutes_mean = 32.0
    player.injury_label = ""
    player.injury_factor = 1.0
    apply_injuries(
        [player],
        report("hou", "2026-10-22"),
        "hou",
        date(2026, 10, 22),
        include_game_day=True,
    )
    assert player.available is False
    assert player.minutes_mean == pytest.approx(0.0)


def test_depth_chart_keeps_the_best_slot():
    from nba_sim.ingest.espn import parse_depth

    ranks = parse_depth(
        {
            "depthchart": [
                {
                    "positions": {
                        "pg": {"athletes": [{"id": "10", "displayName": "Star"}, {"id": "11"}]},
                        "sg": {"athletes": [{"id": "11", "displayName": "Backup"}]},
                    }
                }
            ]
        }
    )
    assert ranks == {"10": 0, "11": 0}


def test_roster_keeps_the_listed_injury_status():
    rows = parse_roster(
        {
            "athletes": [
                {
                    "id": "10",
                    "displayName": "Joel Embiid",
                    "jersey": "21",
                    "position": {"abbreviation": "C"},
                    "status": {"name": "Active", "abbreviation": "Active"},
                    "experience": {"years": 10},
                    "injuries": [{"status": "Out"}],
                }
            ]
        },
        "phi",
    )
    assert rows[0]["player_name"] == "Joel Embiid"
    assert rows[0]["roster_status"] == "Active"
    assert rows[0]["roster_injury"] == "Out"
    assert rows[0]["team_abbr"] == "phi"


def test_on_and_off_last_one_run_and_the_sheet_returns():
    from nba_sim.app.slate import _forced_availability, _parse_availability, _parse_prompt
    from nba_sim.features.court import PlayerCourt, availability_mask
    from nba_sim.sim.specs import GameContext, GameSpec, PlayerModel, TeamModel

    assert _parse_availability("12:on, 18:off, bad, 3:maybe") == {12: "on", 18: "off"}
    modes, minutes = _parse_prompt("12:on,7:min=34,18:off,9:min=0,bad:min=12")
    assert modes == {12: "on", 18: "off"}
    assert minutes == {7: 34.0, 9: 0.0}
    sitting = PlayerCourt(
        player_id=1,
        minutes=0.0,
        minutes_close=32.0,
        play_rate=0.4,
        rest_b2b=0.5,
        rest_3in4=0.2,
        rest_heavy=0.1,
        starts=False,
    )
    held = PlayerModel(
        player_id=1,
        name="Held",
        team_id=10,
        position="G",
        minutes_mean=0.0,
        minutes_base=32.0,
        available=False,
        injury_label="Left knee",
        injury_factor=0.0,
        court=sitting,
    )
    playing = PlayerCourt(player_id=2, minutes=28.0, minutes_close=28.0, play_rate=1.0, starts=True)
    dressed = PlayerModel(
        player_id=2,
        name="Dressed",
        team_id=20,
        position="F",
        minutes_mean=28.0,
        available=True,
        injury_label="",
        injury_factor=1.0,
        court=playing,
    )
    spec = GameSpec(
        game_id="g",
        date="2026-10-26",
        home=TeamModel(team_id=10, abbr="HOM", name="Home"),
        away=TeamModel(team_id=20, abbr="AWY", name="Away"),
        home_players=[held],
        away_players=[dressed],
        context=GameContext(),
        league=LeaguePriors.flat(),
    )
    with _forced_availability(spec, ""):
        assert held.available is False
        assert dressed.available is True
    with _forced_availability(spec, "1:on,2:off"):
        assert held.available is True
        assert held.injury_label == ""
        assert held.injury_factor == 1.0
        assert held.minutes_mean == 32.0
        assert sitting.minutes == 32.0
        assert sitting.minutes_close == 32.0
        assert sitting.play_rate == 1.0
        assert sitting.rest_b2b == 0.0
        assert sitting.rest_3in4 == 0.0
        assert sitting.rest_heavy == 0.0
        assert dressed.available is False
        assert dressed.injury_label == "Out"
        assert dressed.injury_factor == 0.0
        mask = availability_mask([held, dressed], 12, np.random.default_rng(0))
        assert mask[:, 0].all()
        assert not mask[:, 1].any()
    assert held.available is False
    assert held.injury_label == "Left knee"
    assert held.injury_factor == 0.0
    assert held.minutes_mean == 0.0
    assert sitting.minutes == 0.0
    assert sitting.minutes_close == 32.0
    assert sitting.play_rate == 0.4
    assert sitting.rest_b2b == 0.5
    assert sitting.rest_3in4 == 0.2
    assert sitting.rest_heavy == 0.1
    assert sitting.starts is False
    assert dressed.available is True
    assert dressed.injury_label == ""
    assert dressed.minutes_mean == 28.0
    assert playing.minutes == 28.0
    assert playing.play_rate == 1.0
    try:
        with _forced_availability(spec, "2:off"):
            assert dressed.available is False
            raise RuntimeError("stop")
    except RuntimeError:
        pass
    assert dressed.available is True
    assert playing.play_rate == 1.0
    with _forced_availability(spec, "2:min=36"):
        assert dressed.available is True
        assert dressed.minutes_mean == 36.0
        assert playing.minutes == 36.0
        assert playing.minutes_close == 36.0
        assert playing.minutes_blowout == 36.0
    assert dressed.minutes_mean == 28.0
    assert playing.minutes == 28.0
    with _forced_availability(spec, "1:min=30"):
        assert held.available is True
        assert held.minutes_mean == 30.0
        assert sitting.minutes == 30.0
    assert held.available is False
    assert held.minutes_mean == 0.0
    assert sitting.minutes == 0.0


def test_one_game_sits_the_player_set_off_and_plays_the_player_set_on():
    from nba_sim.app.slate import _forced_availability
    from nba_sim.features.court import PlayerCourt
    from nba_sim.sim.demo import make_team, rotation
    from nba_sim.sim.engine import simulate_game
    from nba_sim.sim.specs import GameContext, GameSpec

    home = rotation(1, "Home", 100)
    away = rotation(2, "Away", 200)
    for side in (home, away):
        for player in side:
            minutes = float(player.minutes_mean)
            player.court = PlayerCourt(
                player_id=int(player.player_id),
                minutes=minutes,
                minutes_close=minutes,
                starts=minutes >= 24,
                sample=30,
            )
            player.available = True
    held = home[0]
    held.available = False
    held.injury_label = "Out"
    held.injury_factor = 0.0
    held.minutes_base = float(held.minutes_mean)
    held.minutes_mean = 0.0
    held.court.minutes = 0.0
    held.court.starts = False
    dressed = away[0]
    spec = GameSpec(
        game_id="one",
        date="2026-10-26",
        home=make_team("den"),
        away=make_team("cle"),
        home_players=home,
        away_players=away,
        context=GameContext(),
        league=LeaguePriors.flat(),
    )
    token = f"{int(held.player_id)}:on,{int(dressed.player_id)}:off"
    with _forced_availability(spec, token):
        result = simulate_game(spec, n_sims=2, seed=3)
    held_minutes = result.player_stats[int(held.player_id)]["minutes"]
    sat_minutes = result.player_stats[int(dressed.player_id)]["minutes"]
    assert np.all(held_minutes > 0)
    assert np.all(sat_minutes == 0)
    assert held.available is False
    assert held.minutes_mean == 0.0
    assert held.court.minutes == 0.0
    assert dressed.available is True
    assert dressed.injury_label == ""
