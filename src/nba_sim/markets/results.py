"""Settle a prop edge pick against the stored player log."""

from __future__ import annotations

from datetime import date

import pandas as pd

from nba_sim.geo import melbourne_tip_date
from nba_sim.teams import TEAM_ID_BY_ABBR, normalize_team

_RESULTS = {"W", "L", "Push", "DNP"}


def grade_pick(actual: float, line: float, side: str) -> str | None:
    """Win, loss, or push for the side the board took. A landing on the line pushes."""
    if side not in {"Over", "Under"}:
        return None
    if actual == line:
        return "Push"
    won = actual > line if side == "Over" else actual < line
    return "W" if won else "L"


def pick_record(results: list[str] | pd.Series) -> str | None:
    """The edge-pick record. Pushes and players who did not play stay out of the tally."""
    labels = [str(item) for item in list(results) if str(item) in _RESULTS]
    if not labels:
        return None
    wins = labels.count("W")
    losses = labels.count("L")
    pushes = labels.count("Push")
    missed = labels.count("DNP")
    text = f"Edge picks {wins}-{losses}"
    extra = []
    if pushes:
        extra.append("1 push" if pushes == 1 else f"{pushes} pushes")
    if missed:
        extra.append("1 did not play" if missed == 1 else f"{missed} did not play")
    if extra:
        text += ", " + ", ".join(extra)
    return text


def list_past_games(
    schedule: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    limit: int = 12,
) -> list[dict]:
    """Finished games, newest tip first. A stored player log counts even before the status says Final."""
    if schedule is None or schedule.empty:
        return []
    played = _played_team_dates(logs)
    rows = []
    for row in schedule.to_dict("records"):
        home = normalize_team(row.get("home"))
        away = normalize_team(row.get("away"))
        venue = _as_date(row.get("date"))
        if not home or not away or venue is None:
            continue
        status = str(row.get("status") or "")
        home_id = TEAM_ID_BY_ABBR.get(home)
        has_log = home_id is not None and (venue, home_id) in played
        if "final" not in status.lower() and not has_log:
            continue
        tip = _melbourne_day(row.get("start_utc"), venue)
        start = str(row.get("start_utc") or "")
        rows.append(
            {
                "game_id": str(row.get("game_id") or ""),
                "away": away,
                "home": home,
                "melbourne": tip.isoformat(),
                "start_utc": start,
                "label": f"{away.upper()} at {home.upper()} · {tip.strftime('%d %b')}",
            }
        )
    rows.sort(key=lambda item: item["start_utc"], reverse=True)
    return [row for row in rows if row["game_id"]][: max(limit, 0)]


def list_future_games(
    schedule: pd.DataFrame | None,
    logs: pd.DataFrame | None,
    today: date,
) -> list[dict]:
    """Games still to come, soonest tip first. A final or a stored player log is already played."""
    if schedule is None or schedule.empty:
        return []
    played = _played_team_dates(logs)
    rows = []
    for row in schedule.to_dict("records"):
        home = normalize_team(row.get("home"))
        away = normalize_team(row.get("away"))
        venue = _as_date(row.get("date"))
        if not home or not away or venue is None:
            continue
        tip = _melbourne_day(row.get("start_utc"), venue)
        if tip < today:
            continue
        status = str(row.get("status") or "")
        home_id = TEAM_ID_BY_ABBR.get(home)
        if "final" in status.lower():
            continue
        if home_id is not None and (venue, home_id) in played:
            continue
        rows.append(
            {
                "game_id": str(row.get("game_id") or ""),
                "away": away,
                "home": home,
                "melbourne": tip.isoformat(),
                "venue": venue.isoformat(),
                "start_utc": str(row.get("start_utc") or ""),
                "label": f"{away.upper()} at {home.upper()} · {tip.strftime('%d %b')}",
            }
        )
    rows.sort(key=lambda item: item["start_utc"])
    return [row for row in rows if row["game_id"]]


def box_actuals(
    logs: pd.DataFrame | None,
    on: date,
    home_id: int,
    away_id: int,
    players: list[tuple[int, str]],
) -> dict[str, dict[str, float]] | None:
    """Counting stats for one game, keyed by ESPN id.

    None means the player log has no game for this home team on this date.
    A player who is absent from the returned map did not play.
    """
    game = _game_rows(logs, on, home_id, away_id)
    if game is None:
        return None
    by_player: dict[int, dict[str, float]] = {}
    for row in game.to_dict("records"):
        player_id = _as_int(row.get("player_id"))
        if player_id is None:
            continue
        by_player[player_id] = _counting(row)
    actuals: dict[str, dict[str, float]] = {}
    for player_id, espn_id in players:
        stats = by_player.get(int(player_id))
        if stats is None or not espn_id:
            continue
        actuals[str(espn_id)] = stats
    return actuals


def _played_team_dates(logs: pd.DataFrame | None) -> set[tuple[date, int]]:
    frame = _player_logs(logs)
    if frame is None:
        return set()
    days = pd.to_datetime(frame["date"], errors="coerce").dt.date
    teams = frame["team_id"].map(_as_int)
    return {
        (day, team)
        for day, team in zip(days, teams)
        if day is not None and not pd.isna(day) and team is not None
    }


def _game_rows(
    logs: pd.DataFrame | None,
    on: date,
    home_id: int,
    away_id: int,
) -> pd.DataFrame | None:
    frame = _player_logs(logs)
    if frame is None:
        return None
    days = pd.to_datetime(frame["date"], errors="coerce").dt.date
    teams = frame["team_id"].map(_as_int)
    home_rows = frame[(days == on) & (teams == int(home_id))]
    if home_rows.empty:
        return None
    game_id = str(home_rows.iloc[0]["game_id"])
    same_game = frame["game_id"].astype(str) == game_id
    both = teams.isin([int(home_id), int(away_id)])
    return frame[same_game & both]


def _player_logs(logs: pd.DataFrame | None) -> pd.DataFrame | None:
    if logs is None or logs.empty or "entity_type" not in logs.columns:
        return None
    frame = logs[logs["entity_type"].astype(str).eq("P")]
    if frame.empty or "date" not in frame.columns or "team_id" not in frame.columns:
        return None
    return frame


def _counting(row: dict) -> dict[str, float]:
    pts = _number(row.get("pts"))
    reb = _number(row.get("reb"))
    ast = _number(row.get("ast"))
    tpm = _number(row.get("fg3m"))
    stats: dict[str, float] = {}
    if pts is not None:
        stats["pts"] = pts
    if reb is not None:
        stats["reb"] = reb
    if ast is not None:
        stats["ast"] = ast
    if tpm is not None:
        stats["tpm"] = tpm
    if pts is not None and reb is not None and ast is not None:
        stats["pra"] = pts + reb + ast
    return stats


def _melbourne_day(start_utc: object, venue: date) -> date:
    text = str(start_utc or "")
    if not text or text.lower() in {"nan", "none", "nat"}:
        return venue
    try:
        return melbourne_tip_date(text)
    except ValueError:
        return venue


def _as_date(value: object) -> date | None:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _as_int(value: object) -> int | None:
    number = _number(value)
    if number is None:
        return None
    return int(number)


def _number(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number
