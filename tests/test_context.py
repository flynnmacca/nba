from dataclasses import replace
from datetime import date, timedelta

import pandas as pd
import pytest

from nba_sim.config import SETTINGS
from nba_sim.features.context import TeamTrip, build_context
from nba_sim.features.prepare import _trip
from nba_sim.sim.demo import make_team
from nba_sim.teams import TEAM_ID_BY_ABBR


def test_altitude_and_a_back_to_back_show_up_on_the_context():
    home = make_team("den")
    away = make_team("mia")
    on = date(2026, 1, 15)
    context = build_context(
        home,
        away,
        on,
        TeamTrip(
            prev_date=on - timedelta(days=2),
            prev_lat=39.75,
            prev_lon=-105.0,
            prev_tz=-7,
            measured_home_court=3.0,
            home_court_games=40,
        ),
        TeamTrip(
            prev_date=on - timedelta(days=1),
            prev_lat=25.78,
            prev_lon=-80.19,
            prev_tz=-5,
            games_in_last_4=3,
        ),
    )
    text = " ".join(context.flags).lower()
    assert "altitude" in text
    assert "back-to-back" in text
    assert "three-in-four" in text
    assert context.away_ppp_mult < context.home_ppp_mult
    names = {row["name"] for row in context.factors}
    assert "referee" in names
    assert "home_court" in names


def test_two_games_of_a_new_season_move_form_off_last_year():
    team_id = int(TEAM_ID_BY_ABBR["phi"])
    rows = []
    for offset in range(12):
        rows.append(
            {
                "entity_type": "T",
                "team_id": team_id,
                "date": date(2026, 3, 1) + timedelta(days=2 * offset),
                "matchup": "PHI vs BOS",
                "pts": 100,
                "plus_minus": 1,
            }
        )
    rows.append(
        {
            "entity_type": "T",
            "team_id": team_id,
            "date": date(2026, 10, 20),
            "matchup": "PHI vs BOS",
            "pts": 120,
            "plus_minus": 8,
        }
    )
    rows.append(
        {
            "entity_type": "T",
            "team_id": team_id,
            "date": date(2026, 10, 22),
            "matchup": "PHI @ BOS",
            "pts": 124,
            "plus_minus": -4,
        }
    )
    trip = _trip("phi", date(2026, 10, 23), pd.DataFrame(rows))
    assert trip.form_games == 2
    assert trip.form_mult > 1.15
    on = date(2026, 10, 23)
    context = build_context(make_team("phi"), make_team("bos"), on, trip, TeamTrip())
    home_form = next(row["value"] for row in context.factors if row["name"] == "form" and team_id in row["team_ids"])
    assert 1.02 < home_form < trip.form_mult
    bare = build_context(make_team("phi"), make_team("bos"), on, replace(trip, form_mult=1.0, form_games=0), TeamTrip())
    # Pts form adds a light residual on the home point edge; rating trends are
    # separate. Early-season progress scales the residual down (form_games / 20).
    progress = min(1.0, float(trip.form_games) / 20.0)
    assert context.home_ppp_mult > bare.home_ppp_mult
    assert context.home_ppp_mult - bare.home_ppp_mult == pytest.approx(
        (home_form - 1.0) * 0.18 * progress, abs=1e-6
    )
    assert context.away_ppp_mult == pytest.approx(bare.away_ppp_mult)
    # Division mates share a pace bump; a cross-conference night does not.
    assert any(row["name"] == "division" for row in context.factors)
    cross = build_context(make_team("phi"), make_team("lal"), on, TeamTrip(), TeamTrip())
    assert not any(row["name"] == "division" for row in cross.factors)
    assert context.pace_mult > cross.pace_mult


def test_rating_trends_move_points_and_pace_once():
    on = date(2026, 11, 15)
    hot = TeamTrip(off_form=1.12, def_form=1.08, pace_form=1.06, rating_games=8, form_mult=1.0, form_games=0)
    cold = TeamTrip(off_form=0.92, def_form=0.94, pace_form=0.95, rating_games=8, form_mult=1.0, form_games=0)
    context = build_context(make_team("phi"), make_team("bos"), on, hot, cold)
    bare = build_context(make_team("phi"), make_team("bos"), on, TeamTrip(), TeamTrip())
    assert context.home_ppp_mult > bare.home_ppp_mult
    assert context.away_ppp_mult < bare.away_ppp_mult or context.away_ppp_mult != bare.away_ppp_mult
    assert context.pace_mult != bare.pace_mult
    assert any(row["name"] == "rating_trend" for row in context.factors)


def test_three_in_four_counts_tonight():
    # Two games in the prior three nights plus tonight is a classic three-in-four.
    team_id = int(TEAM_ID_BY_ABBR["mia"])
    rows = [
        {
            "entity_type": "T",
            "team_id": team_id,
            "date": date(2026, 1, 10),
            "matchup": "MIA vs BOS",
            "pts": 110,
            "plus_minus": 2,
        },
        {
            "entity_type": "T",
            "team_id": team_id,
            "date": date(2026, 1, 12),
            "matchup": "MIA @ NYK",
            "pts": 108,
            "plus_minus": -4,
        },
    ]
    trip = _trip("mia", date(2026, 1, 13), pd.DataFrame(rows))
    assert trip.games_in_last_4 == 3
    context = build_context(make_team("mia"), make_team("bos"), date(2026, 1, 13), TeamTrip(), trip)
    assert "away three-in-four" in " ".join(context.flags).lower()


def test_home_court_is_half_the_split_applied_once():
    home = make_team("phi")
    away = make_team("bos")
    context = build_context(
        home,
        away,
        date(2026, 1, 15),
        TeamTrip(measured_home_court=4.0, home_court_games=80),
        TeamTrip(),
    )
    hca = next(row["value"] for row in context.factors if row["name"] == "home_court")
    assert context.home_ppp_mult == pytest.approx(1.0 + hca / SETTINGS.points_per_ppp_unit)
    assert context.away_ppp_mult == pytest.approx(1.0)
