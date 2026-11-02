"""Settled prop picks and the past-game list."""

from datetime import date

import pandas as pd

from nba_sim.markets.results import box_actuals, grade_pick, list_future_games, list_past_games, pick_record
from nba_sim.teams import TEAM_ID_BY_ABBR


def test_a_settled_edge_pick_shows_the_result_and_a_missing_player_does_not_play():
    from nba_sim.app.slate import _prop_table

    summary = {
        "player_bins": [
            {"espn_id": "1", "player": "Low", "team": "PHI", "stat": "pts", "value": 30, "sims": 100},
            {"espn_id": "2", "player": "High", "team": "PHI", "stat": "pts", "value": 0, "sims": 100},
            {"espn_id": "3", "player": "Out", "team": "PHI", "stat": "pts", "value": 20, "sims": 100},
        ]
    }
    props = pd.DataFrame(
        [
            {"espn_id": "1", "stat": "pts", "market": "Points", "line": 21.5, "fair_over": 0.40, "over_american": -110, "under_american": -110, "main": True},
            {"espn_id": "2", "stat": "pts", "market": "Points", "line": 4.5, "fair_over": 0.90, "over_american": 145, "under_american": -170, "main": True},
            {"espn_id": "3", "stat": "pts", "market": "Points", "line": 15.5, "fair_over": 0.40, "over_american": -110, "under_american": -110, "main": True},
        ]
    )
    shown = _prop_table(summary, props, {"1": {"pts": 28}, "2": {"pts": 10}})
    assert shown.loc[shown["Player"] == "Low", "Side"].iloc[0] == "Over"
    assert shown.loc[shown["Player"] == "Low", "Result"].iloc[0] == "W"
    assert shown.loc[shown["Player"] == "Low", "Actual"].iloc[0] == 28
    assert shown.loc[shown["Player"] == "High", "Side"].iloc[0] == "Under"
    assert shown.loc[shown["Player"] == "High", "Result"].iloc[0] == "L"
    assert shown.loc[shown["Player"] == "Out", "Result"].iloc[0] == "DNP"
    upcoming = _prop_table(summary, props)
    assert "Result" not in upcoming.columns


def test_an_over_wins_above_the_line_and_an_under_wins_below_it():
    assert grade_pick(22, 21.5, "Over") == "W"
    assert grade_pick(18, 21.5, "Over") == "L"
    assert grade_pick(4, 4.5, "Under") == "W"
    assert grade_pick(8, 4.5, "Under") == "L"
    assert grade_pick(10, 10, "Over") == "Push"
    assert grade_pick(10, 9.5, "Side") is None


def test_the_record_keeps_pushes_and_absences_out_of_the_tally():
    assert pick_record(["W", "W", "L", "Push", "DNP"]) == "Edge picks 2-1, 1 push, 1 did not play"
    assert pick_record([]) is None
    assert pick_record(["", None]) is None


def test_past_games_include_a_logged_game_before_the_status_says_final():
    schedule = pd.DataFrame(
        [
            {
                "game_id": "future",
                "date": "2026-10-25",
                "start_utc": "2026-10-25T23:00:00Z",
                "home": "nyk",
                "away": "bos",
                "status": "Scheduled",
            },
            {
                "game_id": "logged",
                "date": "2026-10-22",
                "start_utc": "2026-10-23T00:00:00Z",
                "home": "det",
                "away": "bos",
                "status": "Scheduled",
            },
            {
                "game_id": "final",
                "date": "2026-10-20",
                "start_utc": "2026-10-20T23:30:00Z",
                "home": "sas",
                "away": "okc",
                "status": "Final",
            },
        ]
    )
    logs = pd.DataFrame(
        [
            {
                "entity_type": "P",
                "game_id": "0022600001",
                "date": "2026-10-22",
                "player_id": 7,
                "team_id": TEAM_ID_BY_ABBR["det"],
                "pts": 20,
                "reb": 5,
                "ast": 5,
                "fg3m": 2,
            }
        ]
    )
    found = list_past_games(schedule, logs, limit=5)
    assert [row["game_id"] for row in found] == ["logged", "final"]
    assert found[0]["label"] == "BOS at DET · 23 Oct"
    assert found[0]["melbourne"] == "2026-10-23"


def test_future_games_skip_a_final_and_a_game_already_in_the_log():
    schedule = pd.DataFrame(
        [
            {
                "game_id": "yesterday",
                "date": "2026-10-23",
                "start_utc": "2026-10-23T00:00:00Z",
                "home": "det",
                "away": "bos",
                "status": "Scheduled",
            },
            {
                "game_id": "played",
                "date": "2026-10-25",
                "start_utc": "2026-10-25T02:00:00Z",
                "home": "nyk",
                "away": "phi",
                "status": "Scheduled",
            },
            {
                "game_id": "final",
                "date": "2026-10-25",
                "start_utc": "2026-10-25T17:00:00Z",
                "home": "mia",
                "away": "orl",
                "status": "Final",
            },
            {
                "game_id": "tonight",
                "date": "2026-10-25",
                "start_utc": "2026-10-26T00:00:00Z",
                "home": "lal",
                "away": "den",
                "status": "Scheduled",
            },
        ]
    )
    logs = pd.DataFrame(
        [
            {
                "entity_type": "P",
                "game_id": "0022600009",
                "date": "2026-10-25",
                "player_id": 7,
                "team_id": TEAM_ID_BY_ABBR["nyk"],
                "pts": 20,
                "reb": 5,
                "ast": 5,
                "fg3m": 2,
            }
        ]
    )
    found = list_future_games(schedule, logs, date(2026, 10, 25))
    assert [row["game_id"] for row in found] == ["tonight"]


def test_box_actuals_follow_the_home_teams_game_and_skip_a_did_not_play():
    det = TEAM_ID_BY_ABBR["det"]
    bos = TEAM_ID_BY_ABBR["bos"]
    other = TEAM_ID_BY_ABBR["nyk"]
    logs = pd.DataFrame(
        [
            {"entity_type": "T", "game_id": "0022600001", "date": "2026-10-22", "player_id": None, "team_id": det, "pts": 110, "reb": 40, "ast": 25, "fg3m": 12},
            {"entity_type": "P", "game_id": "0022600001", "date": "2026-10-22", "player_id": 101, "team_id": det, "pts": 22, "reb": 5, "ast": 4, "fg3m": 3},
            {"entity_type": "P", "game_id": "0022600001", "date": "2026-10-22", "player_id": 202, "team_id": bos, "pts": 11, "reb": 2, "ast": 9, "fg3m": 1},
            {"entity_type": "P", "game_id": "0022600099", "date": "2026-10-22", "player_id": 303, "team_id": other, "pts": 40, "reb": 10, "ast": 10, "fg3m": 6},
        ]
    )
    actuals = box_actuals(logs, date(2026, 10, 22), det, bos, [(101, "espn-a"), (202, "espn-b"), (404, "espn-out")])
    assert actuals is not None
    assert actuals["espn-a"] == {"pts": 22.0, "reb": 5.0, "ast": 4.0, "tpm": 3.0, "pra": 31.0}
    assert actuals["espn-b"]["ast"] == 9.0
    assert "espn-out" not in actuals
    assert box_actuals(logs, date(2026, 10, 22), other, bos, [(303, "espn-c")])["espn-c"]["pts"] == 40.0
    assert box_actuals(logs, date(2026, 10, 23), det, bos, [(101, "espn-a")]) is None
