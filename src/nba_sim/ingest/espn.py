"""ESPN schedule, rosters, and the injury report.

The scoreboard date is the venue-local date. Tip times are stored in that
arena's clock and in Melbourne time.
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta

import httpx
import pandas as pd

from nba_sim.config import SETTINGS
from nba_sim.features.availability import named_sit
from nba_sim.geo import local_clock
from nba_sim.names import norm_person
from nba_sim.teams import TEAM_ID_BY_ABBR, normalize_team
from nba_sim.warehouse import Warehouse

logger = logging.getLogger(__name__)

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
INJURY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
TEAM_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{slug}/{kind}"
ESPN_SLUG = {"nop": "no", "uta": "utah"}
# ESPN's league year for the 2026-27 season.
SEASON_YEAR = 2027


def espn_slug(abbr: str) -> str:
    canonical = normalize_team(abbr) or abbr
    return ESPN_SLUG.get(canonical, canonical)


def parse_injuries(payload: object) -> pd.DataFrame:
    rows = []
    blocks = payload.get("injuries", []) if isinstance(payload, dict) else []
    for block in blocks:
        team = normalize_team(block.get("displayName") or (block.get("team") or {}).get("displayName"))
        for item in block.get("injuries") or []:
            athlete = item.get("athlete") or {}
            details = item.get("details") if isinstance(item.get("details"), dict) else {}
            fantasy = details.get("fantasyStatus") or {}
            if not isinstance(fantasy, dict):
                fantasy = {}
            rows.append(
                {
                    "team_abbr": team,
                    "player_name": athlete.get("displayName") or item.get("displayName") or "",
                    "athlete_id": _athlete_id(athlete),
                    "status": str(item.get("status") or ""),
                    "fantasy": str(fantasy.get("abbreviation") or fantasy.get("description") or ""),
                    "body_part": str(details.get("type") or ""),
                    "injury_detail": str(details.get("detail") or ""),
                    "side": str(details.get("side") or ""),
                    "comment": str(item.get("shortComment") or ""),
                    "return_date": _date_text(details.get("returnDate")),
                    "report_date": _date_text(item.get("date")),
                    "detail": str(item.get("shortComment") or details.get("detail") or ""),
                }
            )
    return pd.DataFrame(rows)


def injury_decision(row: object, game_on: date) -> tuple[str, float]:
    """How this report row applies to one slate date.

    Returns a short label and a minutes multiplier. Zero means the player is out.
    ``details.type`` on the ESPN feed is the body part, so it is not read as a status.
    """
    status = str(_get(row, "status") or "").strip().lower()
    fantasy = str(_get(row, "fantasy") or "").strip().upper()
    return_on = _parse_date(_get(row, "return_date"))
    if fantasy in {"OFS", "OUT FOR SEASON"} or "SEASON" in fantasy:
        return "Out for season", 0.0
    if status in {"out", "suspended", "suspension"} or named_sit(status):
        return "Out", 0.0
    if return_on is not None and return_on > game_on and status not in {"probable"}:
        label = f"Out until {return_on.strftime('%b')} {return_on.day}"
        return label, 0.0
    if "doubt" in status:
        return "Doubtful", 0.25
    if "prob" in status:
        return "Probable", 0.9
    if status in {"questionable", "day-to-day", "day to day", "gtd"} or "day" in status:
        return "Day-to-day", 0.65
    if not status:
        return "Listed", 0.65
    return status.title(), 0.65


def parse_event(event: dict) -> dict | None:
    competitions = event.get("competitions") or []
    if not competitions:
        return None
    comp = competitions[0]
    home = away = None
    for competitor in comp.get("competitors") or []:
        abbr = normalize_team((competitor.get("team") or {}).get("abbreviation"))
        if competitor.get("homeAway") == "home":
            home = abbr
        elif competitor.get("homeAway") == "away":
            away = abbr
    if not home or not away:
        return None
    start = str(comp.get("date") or event.get("date") or "")
    slate_date, tip_local, tip_melbourne = local_clock(start, home)
    venue = comp.get("venue") or {}
    address = venue.get("address") or {}
    raw_status = comp.get("status") or event.get("status") or {}
    status = raw_status.get("type") if isinstance(raw_status, dict) else {}
    if not isinstance(status, dict):
        status = {}
    return {
        "game_id": str(event.get("id") or comp.get("id") or ""),
        "date": slate_date,
        "start_utc": start,
        "home": home,
        "away": away,
        "name": event.get("shortName") or event.get("name") or "",
        "venue": str(venue.get("fullName") or ""),
        "city": str(address.get("city") or ""),
        "state": str(address.get("state") or ""),
        "neutral": bool(comp.get("neutralSite")),
        "status": str(status.get("description") or status.get("name") or ""),
        "status_detail": str(status.get("shortDetail") or ""),
        "tip_local": tip_local,
        "tip_melbourne": tip_melbourne,
    }


def parse_roster(payload: object, abbr: str) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    team = normalize_team(abbr)
    rows = []
    for athlete in payload.get("athletes") or []:
        if not isinstance(athlete, dict):
            continue
        position = athlete.get("position") or {}
        status = athlete.get("status") or {}
        injury = _roster_injury(athlete.get("injuries"))
        rows.append(
            {
                "team_abbr": team,
                "team_id": TEAM_ID_BY_ABBR.get(team or "", 0),
                "espn_id": str(athlete.get("id") or ""),
                "player_name": athlete.get("displayName") or athlete.get("fullName") or "",
                "position": _position_code(position),
                "jersey": str(athlete.get("jersey") or ""),
                "height": str(athlete.get("displayHeight") or ""),
                "weight": str(athlete.get("displayWeight") or ""),
                "age": athlete.get("age") if athlete.get("age") is not None else "",
                "draft": str(athlete.get("displayDraft") or ""),
                "experience": _experience_years(athlete.get("experience")),
                "salary": _salary(athlete),
                "depth_rank": 99,
                "roster_status": str(status.get("abbreviation") or status.get("name") or ""),
                "roster_injury": injury,
                "name_key": norm_person(athlete.get("displayName") or ""),
            }
        )
    return rows


def ingest_espn_report(warehouse: Warehouse, season_year: int = SEASON_YEAR) -> list[str]:
    """Refresh the injury report. Fast enough to run when the board opens."""
    del season_year
    failures: list[str] = []
    headers = {"User-Agent": "nba-sim/0.1"}
    with httpx.Client(timeout=SETTINGS.request_timeout, follow_redirects=True, headers=headers) as client:
        injuries = _fetch_injuries(client, failures)
    rosters = warehouse.read("espn_rosters")
    if not injuries.empty:
        if rosters is not None and not getattr(rosters, "empty", True):
            injuries = _add_roster_only_outs(injuries, rosters)
        warehouse.write("injuries", injuries)
    logger.info("espn injury report %s", len(injuries))
    return failures


def ingest_espn(warehouse: Warehouse, season_year: int = SEASON_YEAR) -> list[str]:
    """Refresh the season schedule, every roster, and the injury report."""
    failures: list[str] = []
    headers = {"User-Agent": "nba-sim/0.1"}
    with httpx.Client(timeout=SETTINGS.request_timeout, follow_redirects=True, headers=headers) as client:
        injuries = _fetch_injuries(client, failures)
        schedule = _fetch_schedule(client, season_year, failures)
        rosters = _fetch_rosters(client, failures)
    if not injuries.empty:
        injuries = _add_roster_only_outs(injuries, rosters)
        warehouse.write("injuries", injuries)
    if not schedule.empty:
        warehouse.write("schedule", schedule)
        warehouse.write("scoreboard", schedule)
    if not rosters.empty:
        warehouse.write("espn_rosters", rosters)
    logger.info(
        "espn schedule %s games, rosters %s players, injuries %s",
        len(schedule),
        len(rosters),
        len(injuries),
    )
    return failures


def _fetch_injuries(client: httpx.Client, failures: list[str]) -> pd.DataFrame:
    try:
        response = client.get(INJURY_URL)
        response.raise_for_status()
        return parse_injuries(response.json())
    except Exception as exc:
        failures.append(f"injuries: {exc}")
        return pd.DataFrame()


def _fetch_schedule(client: httpx.Client, season_year: int, failures: list[str]) -> pd.DataFrame:
    rows: dict[str, dict] = {}
    for abbr in TEAM_ID_BY_ABBR:
        time.sleep(0.15)
        url = TEAM_URL.format(slug=espn_slug(abbr), kind="schedule")
        try:
            response = client.get(url, params={"season": season_year})
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            failures.append(f"schedule {abbr}: {exc}")
            continue
        for event in payload.get("events") or []:
            parsed = parse_event(event)
            if parsed and parsed["game_id"]:
                rows[parsed["game_id"]] = parsed
    _overlay_scoreboard(client, rows, failures)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows.values())
    return frame.sort_values(["date", "start_utc", "game_id"]).reset_index(drop=True)


def recent_final_dates(days: int = 4) -> list[date]:
    """Venue dates of games already final on the last few scoreboards."""
    today = date.today()
    found: list[date] = []
    headers = {"User-Agent": "nba-sim/0.1"}
    with httpx.Client(timeout=SETTINGS.request_timeout, follow_redirects=True, headers=headers) as client:
        for offset in range(max(days, 1)):
            day = today - timedelta(days=offset)
            try:
                response = client.get(SCOREBOARD_URL, params={"dates": day.strftime("%Y%m%d")})
                response.raise_for_status()
                payload = response.json()
            except Exception:
                logger.info("scoreboard %s unavailable", day.isoformat())
                continue
            for event in payload.get("events") or []:
                parsed = parse_event(event)
                if parsed is None:
                    continue
                status = f"{parsed.get('status') or ''} {parsed.get('status_detail') or ''}".lower()
                if "final" not in status:
                    continue
                raw = str(parsed.get("date") or "")[:10]
                try:
                    found.append(date.fromisoformat(raw))
                except ValueError:
                    continue
    return found


def _overlay_scoreboard(client: httpx.Client, rows: dict[str, dict], failures: list[str]) -> None:
    """The live scoreboard has the current venue, tip, and status for games on the board."""
    try:
        response = client.get(SCOREBOARD_URL)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        failures.append(f"scoreboard: {exc}")
        return
    for event in payload.get("events") or []:
        parsed = parse_event(event)
        if parsed and parsed["game_id"]:
            rows[parsed["game_id"]] = parsed


def parse_depth(payload: object) -> dict[str, int]:
    """Best depth-chart slot for each ESPN athlete. Zero is the first name at a position."""
    if not isinstance(payload, dict):
        return {}
    charts = payload.get("depthchart") or payload.get("depthcharts") or []
    if isinstance(charts, dict):
        charts = [charts]
    ranks: dict[str, int] = {}
    for chart in charts:
        if not isinstance(chart, dict):
            continue
        for slot in (chart.get("positions") or {}).values():
            athletes = slot.get("athletes") if isinstance(slot, dict) else slot
            for index, athlete in enumerate(athletes or []):
                if not isinstance(athlete, dict):
                    continue
                athlete_id = str(athlete.get("id") or "")
                if athlete_id:
                    ranks[athlete_id] = min(index, ranks.get(athlete_id, 99))
    return ranks


def _fetch_rosters(client: httpx.Client, failures: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for abbr in TEAM_ID_BY_ABBR:
        time.sleep(0.15)
        url = TEAM_URL.format(slug=espn_slug(abbr), kind="roster")
        try:
            response = client.get(url)
            response.raise_for_status()
            team_rows = parse_roster(response.json(), abbr)
        except Exception as exc:
            failures.append(f"roster {abbr}: {exc}")
            continue
        ranks = _fetch_depth(client, abbr, failures)
        for row in team_rows:
            row["depth_rank"] = ranks.get(str(row.get("espn_id") or ""), 99)
        rows.extend(team_rows)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).drop_duplicates(["team_abbr", "espn_id"])


def _fetch_depth(client: httpx.Client, abbr: str, failures: list[str]) -> dict[str, int]:
    try:
        response = client.get(TEAM_URL.format(slug=espn_slug(abbr), kind="depthcharts"))
        response.raise_for_status()
        return parse_depth(response.json())
    except Exception as exc:
        failures.append(f"depth {abbr}: {exc}")
        return {}


def _add_roster_only_outs(injuries: pd.DataFrame, rosters: pd.DataFrame) -> pd.DataFrame:
    if rosters.empty or "roster_injury" not in rosters.columns:
        return injuries
    known_ids = set(injuries["athlete_id"].astype(str)) if "athlete_id" in injuries.columns else set()
    known_names = {
        (str(row.team_abbr), norm_person(row.player_name))
        for row in injuries.itertuples(index=False)
        if getattr(row, "player_name", None)
    }
    extra = []
    for row in rosters.to_dict("records"):
        status = str(row.get("roster_injury") or "")
        athlete_id = str(row.get("espn_id") or "")
        name_key = (str(row.get("team_abbr") or ""), norm_person(row.get("player_name")))
        if status.lower() != "out" or not athlete_id or athlete_id in known_ids or name_key in known_names:
            continue
        extra.append(
            {
                "team_abbr": row.get("team_abbr"),
                "player_name": row.get("player_name"),
                "athlete_id": athlete_id,
                "status": "Out",
                "fantasy": "OUT",
                "body_part": "",
                "injury_detail": "",
                "side": "",
                "comment": "Listed out on the team roster.",
                "return_date": "",
                "report_date": "",
                "detail": "Listed out on the team roster.",
            }
        )
    if not extra:
        return injuries
    return pd.concat([injuries, pd.DataFrame(extra)], ignore_index=True)


def _athlete_id(athlete: dict) -> str:
    if athlete.get("id"):
        return str(athlete["id"])
    for link in athlete.get("links") or []:
        href = str((link or {}).get("href") or "")
        marker = "/id/"
        if marker not in href:
            continue
        token = href.split(marker, 1)[1].split("/", 1)[0].split("?", 1)[0]
        if token.isdigit():
            return token
    return ""


def _salary(athlete: dict) -> float:
    current = 0.0
    latest = 0.0
    for contract in athlete.get("contracts") or []:
        if not isinstance(contract, dict):
            continue
        try:
            salary = float(contract.get("salary") or 0)
        except (TypeError, ValueError):
            salary = 0.0
        year = (contract.get("season") or {}).get("year")
        latest = max(latest, salary)
        if year == SEASON_YEAR:
            current = salary
    return current or latest


def _roster_injury(value: object) -> str:
    if not isinstance(value, list) or not value:
        return ""
    first = value[0]
    if isinstance(first, dict):
        return str(first.get("status") or "")
    return ""


def _position_code(position: object) -> str:
    if not isinstance(position, dict):
        return "F"
    text = str(position.get("abbreviation") or position.get("name") or "F").upper()
    if text.startswith("G"):
        return "G"
    if text.startswith("C"):
        return "C"
    return "F"


def _experience_years(value: object) -> float:
    if isinstance(value, dict):
        try:
            return float(value.get("years") or 0)
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _date_text(value: object) -> str:
    parsed = _parse_date(value)
    return "" if parsed is None else parsed.isoformat()


def _parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "nat"}:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _get(row: object, name: str) -> object:
    if isinstance(row, dict):
        return row.get(name)
    return getattr(row, name, None)
