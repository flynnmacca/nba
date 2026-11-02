"""Schedule, altitude, travel, and rest multipliers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from nba_sim.config import SETTINGS
from nba_sim.features.shrink import shrink
from nba_sim.geo import load_arenas, miles
from nba_sim.sim.specs import GameContext, TeamModel
from nba_sim.teams import same_division


@dataclass
class TeamTrip:
    prev_date: date | None = None
    prev_lat: float | None = None
    prev_lon: float | None = None
    prev_tz: float | None = None
    games_in_last_4: int = 1
    measured_home_court: float | None = None
    home_court_games: int = 0
    form_mult: float = 1.0
    form_games: int = 0
    # Recent vs season: off efficiency, opponent scoring, and FGA tempo.
    off_form: float = 1.0
    def_form: float = 1.0
    pace_form: float = 1.0
    rating_games: int = 0


def _factor(name: str, team_ids: list[int], sample: float, value: float, detail: str) -> dict:
    return {
        "name": name,
        "team_ids": team_ids,
        "player_ids": [],
        "sample_size": sample,
        "value": value,
        "detail": detail,
    }


def build_context(
    home: TeamModel,
    away: TeamModel,
    game_date: date,
    home_trip: TeamTrip | None = None,
    away_trip: TeamTrip | None = None,
) -> GameContext:
    settings = SETTINGS
    home_trip = home_trip or TeamTrip()
    away_trip = away_trip or TeamTrip()
    arenas = load_arenas()
    arena = arenas.get(home.abbr)
    factors: list[dict] = []
    flags: list[str] = []
    home_points = 0.0
    away_points = 0.0
    pace_points = 0.0

    hca = shrink(
        home_trip.measured_home_court
        if home_trip.measured_home_court is not None
        else settings.home_court_points,
        home_trip.home_court_games,
        settings.home_court_points,
        settings.home_court_prior_games,
    )
    home_points += hca
    flags.append("home court")
    factors.append(
        _factor(
            "home_court",
            [home.team_id],
            float(home_trip.home_court_games),
            hca,
            f"{home.abbr.upper()} home court {hca:.2f} points, shrunk toward {settings.home_court_points:.1f}",
        )
    )

    def fatigue(trip: TeamTrip, team: TeamModel, side: str) -> float:
        points = 0.0
        if trip.prev_date is not None and (game_date - trip.prev_date).days <= 1:
            points += settings.b2b_points
            flags.append(f"{side} back-to-back")
            factors.append(
                _factor(
                    "rest",
                    [team.team_id],
                    1,
                    settings.b2b_points,
                    f"{team.abbr.upper()} played yesterday",
                )
            )
        if trip.games_in_last_4 >= 3:
            points += settings.three_in_four_points
            flags.append(f"{side} three-in-four")
            factors.append(
                _factor(
                    "schedule_density",
                    [team.team_id],
                    float(trip.games_in_last_4),
                    settings.three_in_four_points,
                    f"{team.abbr.upper()} has {trip.games_in_last_4} games in four nights",
                )
            )
        return points

    home_points += fatigue(home_trip, home, "home")
    away_points += fatigue(away_trip, away, "away")

    def travel(trip: TeamTrip, team: TeamModel, side: str) -> float:
        if (
            arena is None
            or trip.prev_lat is None
            or trip.prev_lon is None
        ):
            return 0.0
        distance = miles(trip.prev_lat, trip.prev_lon, arena.lat, arena.lon)
        points = settings.travel_points_per_1000_miles * (distance / 1000.0)
        tz_delta = 0.0
        if trip.prev_tz is not None:
            tz_delta = arena.offset(game_date) - trip.prev_tz
            points += settings.tz_points_per_hour * abs(tz_delta)
            if tz_delta > 0:
                points += 0.25 * settings.tz_points_per_hour * tz_delta
        if distance >= 400:
            flags.append(f"{side} travel {distance:.0f} mi")
            factors.append(
                _factor(
                    "travel",
                    [team.team_id],
                    distance,
                    points,
                    f"{team.abbr.upper()} traveled {distance:.0f} miles"
                    + (f", {tz_delta:+.0f}h" if tz_delta else ""),
                )
            )
        return points

    home_points += travel(home_trip, home, "home")
    away_points += travel(away_trip, away, "away")

    if arena is not None and arena.altitude_ft >= settings.altitude_floor_ft:
        thousands = (arena.altitude_ft - settings.altitude_floor_ft) / 1000.0
        pace_points += settings.altitude_total_points_per_1000_ft * thousands
        visitor = settings.altitude_visitor_points_per_1000_ft * (
            arena.altitude_ft / 1000.0
        )
        away_points += visitor
        flags.append(f"altitude {arena.altitude_ft:.0f} ft")
        factors.append(
            _factor(
                "altitude",
                [home.team_id, away.team_id],
                arena.altitude_ft,
                visitor,
                f"{arena.name} at {arena.altitude_ft:.0f} ft; visitor adjustment {visitor:.2f} points",
            )
        )

    for trip, team, bucket in (
        (home_trip, home, "home"),
        (away_trip, away, "away"),
    ):
        # Team trends join once on the point / pace edge — not a second PPP
        # product on the player blend, and not another make multiplier beside
        # the ORTG residual. Thin samples stay glued near the season card.
        form = shrink(trip.form_mult, trip.form_games, 1.0, 8.0)
        progress = min(1.0, float(trip.form_games) / 20.0)
        rating_n = float(trip.rating_games or trip.form_games or 0.0)
        rating_progress = min(1.0, rating_n / 16.0)
        off = shrink(trip.off_form, rating_n, 1.0, 8.0)
        defense = shrink(trip.def_form, rating_n, 1.0, 8.0)
        pace_form = shrink(trip.pace_form, rating_n, 1.0, 8.0)
        # Prefer efficiency/defense trends when present; keep a light pts form.
        form_points = (form - 1.0) * settings.points_per_ppp_unit * 0.18 * progress
        off_points = (off - 1.0) * settings.points_per_ppp_unit * 0.32 * rating_progress
        # Soft defense (def_form > 1) credits the opponent once.
        def_points = (defense - 1.0) * settings.points_per_ppp_unit * 0.28 * rating_progress
        pace_points += (pace_form - 1.0) * settings.points_per_ppp_unit * 0.22 * rating_progress
        if bucket == "home":
            home_points += form_points + off_points
            away_points += def_points
        else:
            away_points += form_points + off_points
            home_points += def_points
        if abs(form - 1.0) > 0.005 and abs(form_points) > 0.01:
            flags.append(f"{bucket} form {form:.3f}")
        if abs(off - 1.0) > 0.005 and abs(off_points) > 0.01:
            flags.append(f"{bucket} off {off:.3f}")
        if abs(defense - 1.0) > 0.005 and abs(def_points) > 0.01:
            flags.append(f"{bucket} def {defense:.3f}")
        factors.append(
            _factor(
                "form",
                [team.team_id],
                float(trip.form_games),
                form,
                f"{team.abbr.upper()} recent scoring {form:.3f} adds {form_points:+.2f} points once",
            )
        )
        if rating_n > 0:
            factors.append(
                _factor(
                    "rating_trend",
                    [team.team_id],
                    rating_n,
                    off,
                    (
                        f"{team.abbr.upper()} off {off:.3f} / def {defense:.3f} / pace {pace_form:.3f} "
                        f"adds {off_points:+.2f} own and {def_points:+.2f} to opponent"
                    ),
                )
            )

    if same_division(home.abbr, away.abbr):
        # Division nights widen pace a little without picking a winner.
        pace_points += 0.4
        flags.append("division")
        factors.append(
            _factor(
                "division",
                [home.team_id, away.team_id],
                0,
                0.4,
                f"{home.abbr.upper()}–{away.abbr.upper()} division game adds 0.4 pace points",
            )
        )

    factors.append(
        _factor(
            "referee",
            [home.team_id, away.team_id],
            0,
            1.0,
            "Neutral prior. No stable public crew feed is applied.",
        )
    )
    scale = settings.points_per_ppp_unit
    return GameContext(
        home_ppp_mult=1.0 + home_points / scale,
        away_ppp_mult=1.0 + away_points / scale,
        pace_mult=1.0 + pace_points / (2.0 * scale),
        flags=flags,
        factors=factors,
    )
