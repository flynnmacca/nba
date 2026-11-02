"""Profile parsers and the prop-line comparison."""

import pandas as pd

from nba_sim.profiles import parse_athlete, parse_gamelog, parse_props, parse_team_stats, sim_over_rate


def test_props_pair_over_then_under_and_keep_the_even_line():
    items = []
    for line, over, under in ((22.5, "-114", "-112"), (28.5, "-400", "+280")):
        for price in (over, under):
            items.append(
                {
                    "type": {"name": "Total Points"},
                    "athlete": {"$ref": "http://sports.core.api.espn.com/v2/athletes/3059318"},
                    "odds": {"american": {"value": price}, "total": {"value": line}},
                }
            )
    frame = parse_props(items, "401")
    main = frame[frame["main"]].iloc[0]
    assert main["espn_id"] == "3059318"
    assert main["stat"] == "pts"
    assert main["line"] == 22.5
    assert main["over_american"] == -114
    assert main["under_american"] == -112
    assert main["fair_over"] > 0.45


def test_sim_over_rate_splits_a_push():
    rows = [
        {"value": 22, "sims": 10},
        {"value": 23, "sims": 30},
        {"value": 24, "sims": 10},
    ]
    assert sim_over_rate(rows, 22.5) == 0.8
    assert sim_over_rate(rows, 23) == 0.5


def test_gamelog_uses_the_arena_date_and_box_line():
    payload = {
        "labels": ["MIN", "FG", "FG%", "3PT", "3P%", "FT", "FT%", "REB", "AST", "BLK", "STL", "PF", "TO", "PTS"],
        "events": {
            "401": {
                "gameDate": "2026-10-20T23:00:00.000+00:00",
                "atVs": "@",
                "gameResult": "W",
                "score": "106-100",
                "opponent": {"abbreviation": "NY", "displayName": "New York Knicks"},
            }
        },
        "seasonTypes": [
            {
                "displayName": "2026-27 Regular Season",
                "categories": [{"events": [{"eventId": "401", "stats": ["33", "4-15", "26.7", "2-7", "28.6", "3-4", "75.0", "11", "6", "3", "2", "3", "4", "13"]}]}],
            }
        ],
    }
    rows = parse_gamelog(payload, "phi")
    assert rows[0]["date"] == "2026-10-20"
    assert rows[0]["opponent"] == "NY"
    assert rows[0]["home_away"] == "Away"
    assert rows[0]["points"] == "13"
    assert "EDT" in rows[0]["tip_local"]
    assert "AEDT" in rows[0]["tip_melbourne"]


def test_athlete_and_team_stats_keep_the_display_fields():
    bio = parse_athlete(
        {
            "athlete": {
                "id": "3059318",
                "displayName": "Joel Embiid",
                "age": 32,
                "displayHeight": "7' 0\"",
                "displayWeight": "270 lbs",
                "displayDraft": "2014: Rd 1, Pk 3 (PHI)",
                "position": {"displayName": "Center"},
                "team": {"abbreviation": "PHI"},
            }
        }
    )
    assert bio["height"] == "7' 0\""
    assert bio["age"] == 32
    assert bio["team"] == "phi"
    stats = parse_team_stats(
        {"results": {"stats": {"categories": [{"name": "offensive", "stats": [{"name": "avgPoints", "displayValue": "106.0"}]}]}}}
    )
    assert stats == [{"category": "Offensive", "stat": "Avg Points", "value": "106.0"}]
    assert not pd.DataFrame(stats).empty
