"""ESPN team and player profiles, plus DraftKings props for one game.

Props are a display comparison. They are not an input to the simulator.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import httpx
import pandas as pd

from nba_sim.config import SETTINGS
from nba_sim.geo import local_clock
from nba_sim.ingest.espn import SEASON_YEAR, espn_slug, parse_roster
from nba_sim.markets.devig import american_to_implied, devig_american
from nba_sim.names import norm_person
from nba_sim.teams import normalize_team, team_name
from nba_sim.warehouse import Warehouse

logger = logging.getLogger(__name__)

ATHLETE_URL = "https://site.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{espn_id}"
OVERVIEW_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{espn_id}/overview"
STATS_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{espn_id}/stats"
GAMELOG_URL = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba/athletes/{espn_id}/gamelog"
TEAM_STATS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{slug}/statistics"
ROSTER_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{slug}/roster"
PROP_URL = (
    "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events/{game_id}"
    "/competitions/{game_id}/odds/100/propBets"
)

# ESPN lists the over price first and the under price second for the same line.
PROP_STATS = {
    "Total Points": "pts",
    "Total Rebounds": "reb",
    "Total Assists": "ast",
    "Total 3-Point Field Goals": "tpm",
    "Total Points, Rebounds, and Assists": "pra",
}
STAT_LABELS = {
    "pts": "Points",
    "reb": "Rebounds",
    "ast": "Assists",
    "tpm": "Threes",
    "pra": "Points + rebounds + assists",
}


def parse_athlete(payload: object) -> dict:
    athlete = payload.get("athlete") if isinstance(payload, dict) and "athlete" in payload else payload
    if not isinstance(athlete, dict):
        return {}
    position = athlete.get("position") or {}
    team = athlete.get("team") or {}
    return {
        "espn_id": str(athlete.get("id") or ""),
        "name": str(athlete.get("displayName") or ""),
        "age": athlete.get("age") if athlete.get("age") is not None else "",
        "height": str(athlete.get("displayHeight") or ""),
        "weight": str(athlete.get("displayWeight") or ""),
        "position": str(position.get("displayName") or position.get("abbreviation") or ""),
        "draft": str(athlete.get("displayDraft") or ""),
        "experience": str(athlete.get("displayExperience") or ""),
        "birthplace": str(athlete.get("displayBirthPlace") or ""),
        "jersey": str(athlete.get("jersey") or athlete.get("displayJersey") or ""),
        "team": normalize_team(team.get("abbreviation")) or "",
        "team_name": str(team.get("displayName") or ""),
    }


def parse_season_rows(payload: object) -> list[dict]:
    """Regular-season and career averages from the overview feed."""
    if not isinstance(payload, dict):
        return []
    block = payload.get("statistics") or {}
    labels = [str(label) for label in (block.get("labels") or [])]
    rows = []
    for split in block.get("splits") or []:
        values = split.get("stats") or []
        for label, value in zip(labels, values):
            rows.append({"split": str(split.get("displayName") or ""), "stat": label, "value": value})
    return rows


def parse_advanced_rows(payload: object) -> list[dict]:
    """Efficiency and double-double rates. ESPN's counting line stays in the season table."""
    if not isinstance(payload, dict):
        return []
    rows = []
    for category in payload.get("categories") or []:
        name = str(category.get("name") or "")
        if name in {"averages", "totals"}:
            continue
        labels = [str(label) for label in (category.get("labels") or [])]
        blocks = category.get("statistics") or []
        if isinstance(blocks, dict):
            blocks = [blocks]
        for block in blocks:
            if not isinstance(block, dict):
                continue
            values = block.get("stats") or []
            split = str(block.get("displayName") or category.get("displayName") or name)
            for label, value in zip(labels, values):
                rows.append({"split": split, "stat": label, "value": value})
    return rows


def parse_gamelog(payload: object, team_abbr: str) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    labels = [str(label) for label in (payload.get("labels") or [])]
    events = payload.get("events") or {}
    rows = []
    for season in payload.get("seasonTypes") or []:
        season_name = str(season.get("displayName") or "")
        if "Regular Season" not in season_name and "Preseason" not in season_name:
            continue
        for category in season.get("categories") or []:
            for event in category.get("events") or []:
                event_id = str(event.get("eventId") or "")
                meta = events.get(event_id) or {}
                opponent = meta.get("opponent") or {}
                away = str(meta.get("atVs") or "") == "@"
                home_abbr = normalize_team(opponent.get("abbreviation")) if away else normalize_team(team_abbr)
                game_date = str(meta.get("gameDate") or "")
                slate_date, tip_local, tip_melbourne = _clock(game_date, home_abbr)
                stats = {
                    label: value
                    for label, value in zip(labels, event.get("stats") or [])
                }
                rows.append(
                    {
                        "season": season_name,
                        "date": slate_date,
                        "tip_local": tip_local,
                        "tip_melbourne": tip_melbourne,
                        "opponent": str(opponent.get("abbreviation") or ""),
                        "opponent_name": str(opponent.get("displayName") or ""),
                        "home_away": "Away" if away else "Home",
                        "result": str(meta.get("gameResult") or ""),
                        "score": str(meta.get("score") or ""),
                        "minutes": stats.get("MIN", ""),
                        "fg": stats.get("FG", ""),
                        "threes": stats.get("3PT", ""),
                        "ft": stats.get("FT", ""),
                        "rebounds": stats.get("REB", ""),
                        "assists": stats.get("AST", ""),
                        "steals": stats.get("STL", ""),
                        "blocks": stats.get("BLK", ""),
                        "turnovers": stats.get("TO", ""),
                        "points": stats.get("PTS", ""),
                    }
                )
    rows.sort(key=lambda row: row["date"], reverse=True)
    return rows


def parse_team_stats(payload: object) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    categories = (((payload.get("results") or {}).get("stats") or {}).get("categories") or [])
    rows = []
    for category in categories:
        title = str(category.get("displayName") or category.get("name") or "").title()
        for stat in category.get("stats") or []:
            if not isinstance(stat, dict):
                continue
            rows.append(
                {
                    "category": title,
                    "stat": _stat_label(stat),
                    "value": stat.get("displayValue") if stat.get("displayValue") not in (None, "") else stat.get("value"),
                }
            )
    return rows


def parse_props(items: list[dict], game_id: str) -> pd.DataFrame:
    """Pair each line's two prices. ESPN returns the over first, then the under."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    order: list[tuple] = []
    for item in items:
        market = str((item.get("type") or {}).get("name") or "")
        stat = PROP_STATS.get(market)
        if stat is None:
            continue
        athlete_id = _athlete_id(item.get("athlete"))
        line = _line(item.get("odds") or {})
        price = _american(item.get("odds") or {})
        if not athlete_id or line is None or price is None:
            continue
        key = (athlete_id, stat, line)
        if key not in grouped:
            order.append(key)
        grouped[key].append(price)
    rows = []
    for athlete_id, stat, line in order:
        prices = grouped[(athlete_id, stat, line)]
        over = prices[0]
        under = prices[1] if len(prices) > 1 else None
        fair_over = None
        if under is not None:
            fair_over = float(devig_american([over, under])[0])
        rows.append(
            {
                "game_id": str(game_id),
                "espn_id": athlete_id,
                "stat": stat,
                "market": STAT_LABELS[stat],
                "line": line,
                "over_american": over,
                "under_american": under,
                "fair_over": fair_over,
                "main": False,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["distance"] = [
        _distance_from_even(over, under)
        for over, under in zip(frame["over_american"], frame["under_american"], strict=False)
    ]
    mains = (
        frame.sort_values("distance")
        .groupby(["espn_id", "stat"], as_index=False)
        .head(1)
        .index
    )
    frame.loc[mains, "main"] = True
    return frame.drop(columns=["distance"])


def main_prop(props: pd.DataFrame | None, espn_id: str, stat: str) -> dict | None:
    if props is None or props.empty or "espn_id" not in props.columns:
        return None
    hit = props[
        (props["espn_id"].astype(str) == str(espn_id))
        & (props["stat"].astype(str) == stat)
        & (props["main"].astype(bool))
    ]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def sim_over_rate(rows: list[dict] | pd.DataFrame, line: float) -> float | None:
    """Share of sims over a prop line. A landing exactly on the line counts half."""
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    total = sum(int(row["sims"]) for row in rows)
    if total <= 0:
        return None
    over = sum(int(row["sims"]) for row in rows if float(row["value"]) > line)
    push = sum(int(row["sims"]) for row in rows if float(row["value"]) == line)
    return (over + 0.5 * push) / total


def load_team_profile(abbr: str, warehouse: Warehouse | None = None) -> dict:
    warehouse = warehouse or Warehouse()
    canonical = normalize_team(abbr) or abbr
    roster = _cached_frame(warehouse, f"roster_bio_{canonical}", lambda client: _team_roster(client, canonical))
    stats = _cached_frame(warehouse, f"team_stats_espn_{canonical}", lambda client: _team_stats(client, canonical))
    injuries = warehouse.read("injuries")
    if injuries is not None and not injuries.empty and "team_abbr" in injuries.columns:
        injuries = injuries[injuries["team_abbr"].astype(str) == canonical].copy()
    else:
        injuries = pd.DataFrame()
    return {
        "abbr": canonical,
        "name": team_name(canonical),
        "roster": roster if roster is not None else pd.DataFrame(),
        "stats": stats if stats is not None else pd.DataFrame(),
        "injuries": injuries,
    }


def load_player_profile(espn_id: str, team_abbr: str = "", warehouse: Warehouse | None = None) -> dict:
    warehouse = warehouse or Warehouse()
    espn_id = str(espn_id)
    bio = _cached_row(warehouse, "player_bios", "espn_id", espn_id, lambda client: _athlete(client, espn_id))
    season = _cached_frame(
        warehouse,
        f"player_season_{espn_id}",
        lambda client: pd.DataFrame(parse_season_rows(_get(client, OVERVIEW_URL.format(espn_id=espn_id)))),
    )
    advanced = _cached_frame(
        warehouse,
        f"player_advanced_{espn_id}",
        lambda client: pd.DataFrame(
            parse_advanced_rows(_get(client, STATS_URL.format(espn_id=espn_id), params={"season": SEASON_YEAR}))
        ),
    )
    team = str(bio.get("team") or team_abbr or "")
    logs = _cached_frame(
        warehouse,
        f"player_log_{espn_id}",
        lambda client: pd.DataFrame(parse_gamelog(_get(client, GAMELOG_URL.format(espn_id=espn_id)), team)),
    )
    if bio and not str(bio.get("wingspan") or ""):
        span = _wingspan(str(bio.get("name") or ""), str(bio.get("draft") or ""), warehouse)
        if span:
            bio["wingspan"] = span
            _upsert_row(warehouse, "player_bios", "espn_id", bio)
    return {"bio": bio, "season": season, "advanced": advanced, "logs": logs if logs is not None else pd.DataFrame()}


def load_props(game_id: str, warehouse: Warehouse | None = None) -> pd.DataFrame:
    warehouse = warehouse or Warehouse()
    game_id = str(game_id)
    stored = warehouse.read("player_props")
    if stored is not None and not stored.empty and "game_id" in stored.columns:
        mine = stored[stored["game_id"].astype(str) == game_id]
        if not mine.empty:
            return mine.reset_index(drop=True)
    frame = _fetch_props(game_id)
    if frame.empty:
        return frame
    if stored is not None and not stored.empty:
        others = stored[stored["game_id"].astype(str) != game_id]
        frame = pd.concat([others, frame], ignore_index=True)
    warehouse.write("player_props", frame)
    return frame[frame["game_id"].astype(str) == game_id].reset_index(drop=True)


def _fetch_props(game_id: str) -> pd.DataFrame:
    url = PROP_URL.format(game_id=game_id)
    headers = {"User-Agent": "nba-sim/0.1"}
    with httpx.Client(timeout=SETTINGS.request_timeout, follow_redirects=True, headers=headers) as client:
        items = _prop_pages(client, url)
    return parse_props(items, game_id)


def _prop_pages(client: httpx.Client, url: str) -> list[dict]:
    """Walk ESPN's 1-based `page` parameter. `pageIndex` is ignored by the API."""
    items: list[dict] = []
    seen: set[str] = set()
    page = 1
    page_count = 1
    while page <= page_count and page <= 12:
        payload = client.get(url, params={"limit": 200, "page": page}).json()
        batch = payload.get("items") or []
        stamp = _page_stamp(batch)
        if not batch or stamp in seen:
            break
        seen.add(stamp)
        items.extend(batch)
        page_count = int(payload.get("pageCount") or page)
        page += 1
    return items


def _page_stamp(batch: list[dict]) -> str:
    if not batch:
        return ""
    first = batch[0]
    odds = first.get("odds") or {}
    market = str((first.get("type") or {}).get("name") or "")
    return f"{_athlete_id(first.get('athlete'))}:{market}:{_line(odds)}:{_american(odds)}"


def _team_roster(client: httpx.Client, abbr: str) -> pd.DataFrame:
    payload = _get(client, ROSTER_URL.format(slug=espn_slug(abbr)))
    return pd.DataFrame(parse_roster(payload, abbr))


def _team_stats(client: httpx.Client, abbr: str) -> pd.DataFrame:
    payload = _get(client, TEAM_STATS_URL.format(slug=espn_slug(abbr)))
    return pd.DataFrame(parse_team_stats(payload))


def _athlete(client: httpx.Client, espn_id: str) -> dict:
    bio = parse_athlete(_get(client, ATHLETE_URL.format(espn_id=espn_id)))
    bio["espn_id"] = espn_id
    bio["wingspan"] = ""
    return bio


def _wingspan(name: str, draft: str, warehouse: Warehouse) -> str:
    year = _draft_year(draft)
    if year is None or not name:
        return ""
    key = norm_person(name)
    for season in (year, year - 1):
        if season < 2000:
            continue
        frame = _cached_frame(warehouse, f"combine_anthro_{season}", lambda _client, season=season: _combine_anthro(season))
        span = _span_from(frame, key)
        if span:
            return span
    return ""


def _combine_anthro(year: int) -> pd.DataFrame:
    try:
        from nba_api.stats.endpoints import DraftCombinePlayerAnthro
    except Exception:
        return pd.DataFrame()
    try:
        return DraftCombinePlayerAnthro(season_year=str(year), timeout=25).get_data_frames()[0]
    except Exception as exc:
        logger.info("combine anthro %s skipped: %s", year, exc)
        return pd.DataFrame()


def _span_from(frame: pd.DataFrame, name_key: str) -> str:
    if frame is None or frame.empty or "PLAYER_NAME" not in frame.columns:
        return ""
    names = frame["PLAYER_NAME"].map(norm_person)
    hit = frame[names == name_key]
    if hit.empty:
        return ""
    span = hit.iloc[0].get("WINGSPAN_FT_IN") or hit.iloc[0].get("WINGSPAN")
    text = "" if span is None else str(span).strip()
    if text in {"", "nan", "None"}:
        return ""
    return text


def _cached_frame(warehouse: Warehouse, name: str, build) -> pd.DataFrame:
    stored = warehouse.read(name)
    if stored is not None:
        return stored
    with httpx.Client(timeout=SETTINGS.request_timeout, follow_redirects=True, headers={"User-Agent": "nba-sim/0.1"}) as client:
        frame = build(client)
    if frame is None:
        frame = pd.DataFrame()
    if not frame.empty:
        warehouse.write(name, frame)
    return frame


def _cached_row(warehouse: Warehouse, table: str, key: str, value: str, build) -> dict:
    stored = warehouse.read(table)
    if stored is not None and not stored.empty and key in stored.columns:
        hit = stored[stored[key].astype(str) == str(value)]
        if not hit.empty:
            return hit.iloc[0].to_dict()
    with httpx.Client(timeout=SETTINGS.request_timeout, follow_redirects=True, headers={"User-Agent": "nba-sim/0.1"}) as client:
        row = build(client)
    if row:
        _upsert_row(warehouse, table, key, row)
    return row


def _upsert_row(warehouse: Warehouse, table: str, key: str, row: dict) -> None:
    frame = pd.DataFrame([row])
    stored = warehouse.read(table)
    if stored is not None and not stored.empty and key in stored.columns:
        stored = stored[stored[key].astype(str) != str(row.get(key))]
        frame = pd.concat([stored, frame], ignore_index=True)
    warehouse.write(table, frame)


def _get(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    response = client.get(url, params=params)
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def _clock(game_date: str, home_abbr: str | None) -> tuple[str, str, str]:
    if not game_date or not home_abbr:
        return game_date[:10], "", ""
    try:
        return local_clock(game_date, home_abbr)
    except Exception:
        return game_date[:10], "", ""


def _athlete_id(athlete: object) -> str:
    if not isinstance(athlete, dict):
        return ""
    if athlete.get("id"):
        return str(athlete["id"])
    ref = str(athlete.get("$ref") or "")
    match = re.search(r"/athletes/(\d+)", ref)
    return match.group(1) if match else ""


def _line(odds: dict) -> float | None:
    total = (odds.get("total") or {}).get("value")
    try:
        return float(total)
    except (TypeError, ValueError):
        return None


def _american(odds: dict) -> float | None:
    raw = (odds.get("american") or {}).get("value")
    try:
        return float(str(raw).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _distance_from_even(over: float, under: float | None) -> float:
    if under is None or (isinstance(under, float) and pd.isna(under)):
        return abs(american_to_implied(float(over)) - 0.5)
    fair = devig_american([float(over), float(under)])
    return abs(float(fair[0]) - 0.5)


def _draft_year(text: str) -> int | None:
    match = re.search(r"(19|20)\d\d", str(text or ""))
    return int(match.group(0)) if match else None


def _stat_label(stat: dict) -> str:
    for key in ("displayName", "shortDisplayName", "abbreviation"):
        if stat.get(key):
            return str(stat[key])
    name = str(stat.get("name") or "")
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return spaced[:1].upper() + spaced[1:] if spaced else name
