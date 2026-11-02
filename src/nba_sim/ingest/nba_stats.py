"""NBA stats dashboards at player and team grain.

Each request is a league-wide call. Responses are cached as parquet so a
rerun does not repeat them. A failed endpoint is recorded and skipped.
"""

from __future__ import annotations

import logging
from datetime import date, datetime

import pandas as pd
from nba_api.stats.endpoints import (
    BoxScoreMatchupsV3,
    CommonTeamRoster,
    LeagueDashLineups,
    LeagueDashPlayerBioStats,
    LeagueDashPlayerClutch,
    LeagueDashPlayerShotLocations,
    LeagueDashPlayerStats,
    LeagueDashPtDefend,
    LeagueDashPtStats,
    LeagueDashPtTeamDefend,
    LeagueDashTeamClutch,
    LeagueDashTeamShotLocations,
    LeagueDashTeamStats,
    LeagueGameLog,
    LeagueHustleStatsPlayer,
    LeagueHustleStatsTeam,
    SynergyPlayTypes,
    TeamPlayerOnOffSummary,
)

from nba_sim.config import SETTINGS
from nba_sim.constants import DEFENSE_PLAY_SKIP, PLAY_TYPES, TRACK_ZONE, canon_play
from nba_sim.features.prepare import season_for
from nba_sim.ingest.common import cached_frame, first_number
from nba_sim.ingest.shot_zones import parse_shot_zones
from nba_sim.teams import ABBR_BY_TEAM_ID, TEAM_ID_BY_ABBR, normalize_team
from nba_sim.warehouse import Warehouse

logger = logging.getLogger(__name__)

DEFENSE_CATEGORIES = [
    "Overall",
    "3 Pointers",
    "2 Pointers",
    "Less Than 6Ft",
    "Less Than 10Ft",
    "Greater Than 15Ft",
]
PT_MEASURES = [
    "Drives",
    "CatchShoot",
    "PullUpShot",
    "Passing",
    "Possessions",
    "Rebounding",
    "PostTouch",
    "PaintTouch",
    "Touches",
]


def _safe_call(factory):
    endpoint = factory()
    frames = endpoint.get_data_frames()
    if not frames:
        return pd.DataFrame()
    return frames[0]


def ingest_rate_history(
    warehouse: Warehouse | None = None,
    seasons: list[str] | None = None,
    log_seasons: list[str] | None = None,
    refresh: bool = False,
) -> list[str]:
    """Pull the feeds the ratings actually read, then rebuild the combined tables.

    Two prior seasons sit behind the latest full season. Game logs for the
    current season are included so rest, travel, and recent minutes can move.
    """
    warehouse = warehouse or Warehouse()
    seasons = list(seasons or ["2023-24", "2024-25", "2025-26"])
    log_seasons = list(log_seasons or ["2026-27"])
    failures: list[str] = []
    for season in seasons:
        logger.info("rate history %s", season)
        _ingest_rate_season(warehouse, season, refresh, failures)
    for season in log_seasons:
        if season in seasons:
            continue
        logger.info("game logs %s", season)
        _ingest_logs(warehouse, season, refresh, failures)
    _combine(warehouse, list(dict.fromkeys([*seasons, *log_seasons])), refresh, failures)
    return failures


def refresh_slate_season(
    warehouse: Warehouse | None = None,
    season: str | None = None,
    refresh: bool = True,
) -> list[str]:
    """Re-pull this year's dashboards, synergy, and game logs, then rebuild the tables.

    Older seasons already on disk stay in the combined tables. The Monte Carlo
    itself does not learn inside a sim. The next sim draws from these rates.
    """
    warehouse = warehouse or Warehouse()
    season = season or season_for(date.today()) or "2026-27"
    failures: list[str] = []
    logger.info("refreshing slate season %s", season)
    _ingest_rate_season(warehouse, season, refresh, failures, slate_only=True)
    _ingest_lineups(warehouse, season, refresh, failures)
    seasons = sorted(set(seasons_on_disk(warehouse)) | {season})
    _combine(warehouse, seasons, False, failures)
    _write_stamp(warehouse, season)
    return failures


def refresh_lineups(
    warehouse: Warehouse | None = None,
    season: str | None = None,
    refresh: bool = True,
) -> list[str]:
    """Pull this season's five-man groups and fold them into the lineup table."""
    warehouse = warehouse or Warehouse()
    season = season or season_for(date.today()) or "2026-27"
    failures: list[str] = []
    _ingest_lineups(warehouse, season, refresh, failures)
    seasons = sorted(set(seasons_on_disk(warehouse)) | {season})
    _write_combined(warehouse, "lineups", [_lineup_table(warehouse, item) for item in seasons])
    return failures


def slate_refresh_due(
    warehouse: Warehouse,
    season: str,
    finals: list[date] | None,
) -> bool:
    """True when this season's rates are missing, or a final is newer than the logs."""
    if not warehouse.exists(f"raw_player_Base_{season}"):
        return True
    if not warehouse.exists(f"raw_team_Advanced_{season}"):
        return True
    if not warehouse.exists(f"raw_play_offensive_T_Spotup_{season}"):
        return True
    if not finals:
        return False
    newest = max(finals)
    last = last_team_log_date(warehouse)
    if last is None or newest > last:
        return True
    finished = sum(1 for day in finals if day == newest)
    return finished > team_log_games_on(warehouse, newest)


def seasons_on_disk(warehouse: Warehouse) -> list[str]:
    """Season tokens already stored, so a one-year pull does not wipe the priors."""
    found: set[str] = set()
    for name in warehouse.tables():
        if len(name) < 7:
            continue
        token = name[-7:]
        if token[0].isdigit() and token[4] == "-" and token[5:].isdigit():
            found.add(token)
    return sorted(found)


def last_team_log_date(warehouse: Warehouse) -> date | None:
    if not warehouse.exists("game_logs"):
        return None
    frame = warehouse.read("game_logs")
    if frame is None or frame.empty or "date" not in frame.columns:
        return None
    if "entity_type" in frame.columns:
        frame = frame[frame["entity_type"].astype(str).eq("T")]
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date()


def team_log_games_on(warehouse: Warehouse, on: date) -> int:
    """Finished games already stored for one date. Each game is two team rows."""
    if not warehouse.exists("game_logs"):
        return 0
    frame = warehouse.read("game_logs")
    if frame is None or frame.empty or "date" not in frame.columns:
        return 0
    if "entity_type" in frame.columns:
        frame = frame[frame["entity_type"].astype(str).eq("T")]
    dates = pd.to_datetime(frame["date"], errors="coerce").dt.date
    return int((dates == on).sum()) // 2


def _write_stamp(warehouse: Warehouse, season: str) -> None:
    path = warehouse.root / "slate_stamp.txt"
    path.write_text(f"{season} {datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")


def _ingest_rate_season(
    warehouse: Warehouse,
    season: str,
    refresh: bool,
    failures: list[str],
    slate_only: bool = False,
) -> None:
    for measure in ("Base", "Advanced", "Scoring", "Usage"):
        cached_frame(
            warehouse,
            f"raw_player_{measure}_{season}",
            lambda measure=measure: _safe_call(
                lambda: LeagueDashPlayerStats(
                    season=season,
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense=measure,
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    for measure in ("Base", "Advanced", "Four Factors"):
        cached_frame(
            warehouse,
            f"raw_team_{measure}_{season}",
            lambda measure=measure: _safe_call(
                lambda: LeagueDashTeamStats(
                    season=season,
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense=measure,
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    if not slate_only:
        _ingest_shot_tracking(warehouse, season, refresh, failures)
    _ingest_logs(warehouse, season, refresh, failures)
    _ingest_plays(warehouse, season, refresh, failures)


def _ingest_lineups(warehouse: Warehouse, season: str, refresh: bool, failures: list[str]) -> None:
    """Five-man groups for one season. One league call, kept with the slate refresh."""
    cached_frame(
        warehouse,
        f"raw_lineups_{season}",
        lambda: _safe_call(
            lambda: LeagueDashLineups(
                season=season,
                group_quantity="5",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="Per100Possessions",
                timeout=SETTINGS.request_timeout,
            )
        ),
        refresh,
        failures,
    )


def _ingest_shot_tracking(
    warehouse: Warehouse,
    season: str,
    refresh: bool,
    failures: list[str],
) -> None:
    cached_frame(
        warehouse,
        f"raw_player_shots_{season}",
        lambda: _safe_call(
            lambda: LeagueDashPlayerShotLocations(
                season=season,
                distance_range="By Zone",
                per_mode_detailed="PerGame",
                timeout=SETTINGS.request_timeout,
            )
        ),
        refresh,
        failures,
    )
    cached_frame(
        warehouse,
        f"raw_team_shots_{season}",
        lambda: _safe_call(
            lambda: LeagueDashTeamShotLocations(
                season=season,
                distance_range="By Zone",
                per_mode_detailed="PerGame",
                timeout=SETTINGS.request_timeout,
            )
        ),
        refresh,
        failures,
    )
    for category in DEFENSE_CATEGORIES:
        cached_frame(
            warehouse,
            f"raw_pt_defend_{category}_{season}",
            lambda category=category: _safe_call(
                lambda: LeagueDashPtDefend(
                    season=season,
                    defense_category=category,
                    per_mode_simple="PerGame",
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
        cached_frame(
            warehouse,
            f"raw_pt_team_defend_{category}_{season}",
            lambda category=category: _safe_call(
                lambda: LeagueDashPtTeamDefend(
                    season=season,
                    defense_category=category,
                    per_mode_simple="PerGame",
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    _ingest_lineups(warehouse, season, refresh, failures)
    for abbr, team_id in TEAM_ID_BY_ABBR.items():
        cached_frame(
            warehouse,
            f"raw_onoff_{abbr}_{season}",
            lambda team_id=team_id: _on_off(team_id, season),
            refresh,
            failures,
        )


def _ingest_plays(warehouse: Warehouse, season: str, refresh: bool, failures: list[str]) -> None:
    for side in ("offensive", "defensive"):
        for play in PLAY_TYPES:
            if side == "defensive" and play in DEFENSE_PLAY_SKIP:
                continue
            for entity in ("P", "T"):
                cached_frame(
                    warehouse,
                    f"raw_play_{side}_{entity}_{play}_{season}",
                    lambda side=side, play=play, entity=entity: _play_type(season, side, play, entity),
                    refresh,
                    failures,
                )


def _ingest_logs(warehouse: Warehouse, season: str, refresh: bool, failures: list[str]) -> None:
    for side in ("P", "T"):
        cached_frame(
            warehouse,
            f"raw_gamelog_{side}_{season}",
            lambda side=side: _safe_call(
                lambda: LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation=side,
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )


def ingest_nba(
    warehouse: Warehouse,
    seasons: list[str] | None = None,
    refresh: bool = False,
    matchup_games: int = 30,
) -> list[str]:
    seasons = seasons or list(SETTINGS.seasons)
    failures: list[str] = []
    for season in seasons:
        _ingest_season(warehouse, season, refresh, failures, matchup_games)
    _combine(warehouse, seasons, refresh, failures)
    return failures


def _ingest_season(
    warehouse: Warehouse,
    season: str,
    refresh: bool,
    failures: list[str],
    matchup_games: int,
) -> None:
    logger.info("ingesting NBA stats %s", season)
    for measure in ("Base", "Advanced", "Misc", "Scoring", "Usage"):
        cached_frame(
            warehouse,
            f"raw_player_{measure}_{season}",
            lambda measure=measure: _safe_call(
                lambda: LeagueDashPlayerStats(
                    season=season,
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense=measure,
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    for measure in ("Base", "Advanced", "Misc", "Four Factors", "Scoring"):
        cached_frame(
            warehouse,
            f"raw_team_{measure}_{season}",
            lambda measure=measure: _safe_call(
                lambda: LeagueDashTeamStats(
                    season=season,
                    per_mode_detailed="PerGame",
                    measure_type_detailed_defense=measure,
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    cached_frame(
        warehouse,
        f"raw_player_shots_{season}",
        lambda: _safe_call(
            lambda: LeagueDashPlayerShotLocations(
                season=season,
                distance_range="By Zone",
                per_mode_detailed="PerGame",
                timeout=SETTINGS.request_timeout,
            )
        ),
        refresh,
        failures,
    )
    cached_frame(
        warehouse,
        f"raw_team_shots_{season}",
        lambda: _safe_call(
            lambda: LeagueDashTeamShotLocations(
                season=season,
                distance_range="By Zone",
                per_mode_detailed="PerGame",
                timeout=SETTINGS.request_timeout,
            )
        ),
        refresh,
        failures,
    )
    for category in DEFENSE_CATEGORIES:
        cached_frame(
            warehouse,
            f"raw_pt_defend_{category}_{season}",
            lambda category=category: _safe_call(
                lambda: LeagueDashPtDefend(
                    season=season,
                    defense_category=category,
                    per_mode_simple="PerGame",
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
        cached_frame(
            warehouse,
            f"raw_pt_team_defend_{category}_{season}",
            lambda category=category: _safe_call(
                lambda: LeagueDashPtTeamDefend(
                    season=season,
                    defense_category=category,
                    per_mode_simple="PerGame",
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    for measure in PT_MEASURES:
        cached_frame(
            warehouse,
            f"raw_pt_player_{measure}_{season}",
            lambda measure=measure: _safe_call(
                lambda: LeagueDashPtStats(
                    season=season,
                    pt_measure_type=measure,
                    player_or_team="Player",
                    per_mode_simple="PerGame",
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    for label, factory in (
        (
            "hustle_player",
            lambda: LeagueHustleStatsPlayer(
                season=season, per_mode_time="PerGame", timeout=SETTINGS.request_timeout
            ),
        ),
        (
            "hustle_team",
            lambda: LeagueHustleStatsTeam(
                season=season, per_mode_time="PerGame", timeout=SETTINGS.request_timeout
            ),
        ),
        (
            "lineups",
            lambda: LeagueDashLineups(
                season=season,
                group_quantity="5",
                measure_type_detailed_defense="Advanced",
                per_mode_detailed="Per100Possessions",
                timeout=SETTINGS.request_timeout,
            ),
        ),
        (
            "clutch_player",
            lambda: LeagueDashPlayerClutch(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                timeout=SETTINGS.request_timeout,
            ),
        ),
        (
            "clutch_team",
            lambda: LeagueDashTeamClutch(
                season=season,
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
                timeout=SETTINGS.request_timeout,
            ),
        ),
        (
            "bio",
            lambda: LeagueDashPlayerBioStats(
                season=season, per_mode_simple="PerGame", timeout=SETTINGS.request_timeout
            ),
        ),
    ):
        cached_frame(
            warehouse,
            f"raw_{label}_{season}",
            lambda factory=factory: _safe_call(factory),
            refresh,
            failures,
        )
    for side in ("P", "T"):
        cached_frame(
            warehouse,
            f"raw_gamelog_{side}_{season}",
            lambda side=side: _safe_call(
                lambda: LeagueGameLog(
                    season=season,
                    player_or_team_abbreviation=side,
                    timeout=SETTINGS.request_timeout,
                )
            ),
            refresh,
            failures,
        )
    for abbr, team_id in TEAM_ID_BY_ABBR.items():
        cached_frame(
            warehouse,
            f"raw_roster_{abbr}_{season}",
            lambda team_id=team_id: _safe_call(
                lambda: CommonTeamRoster(
                    team_id=team_id, season=season, timeout=SETTINGS.request_timeout
                )
            ),
            refresh,
            failures,
        )
        cached_frame(
            warehouse,
            f"raw_onoff_{abbr}_{season}",
            lambda team_id=team_id: _on_off(team_id, season),
            refresh,
            failures,
        )
    for side in ("offensive", "defensive"):
        for play in PLAY_TYPES:
            if side == "defensive" and play in DEFENSE_PLAY_SKIP:
                continue
            for entity in ("P", "T"):
                cached_frame(
                    warehouse,
                    f"raw_play_{side}_{entity}_{play}_{season}",
                    lambda side=side, play=play, entity=entity: _play_type(
                        season, side, play, entity
                    ),
                    refresh,
                    failures,
                )
    _matchups(warehouse, season, refresh, failures, matchup_games)


def _on_off(team_id: int, season: str) -> pd.DataFrame:
    endpoint = TeamPlayerOnOffSummary(
        team_id=team_id,
        season=season,
        per_mode_detailed="PerGame",
        timeout=SETTINGS.request_timeout,
    )
    on_court = endpoint.players_on_court_team_player_on_off_summary.get_data_frame()
    off_court = endpoint.players_off_court_team_player_on_off_summary.get_data_frame()
    if on_court is None or on_court.empty:
        return pd.DataFrame()
    on_court = on_court.rename(
        columns={
            "OFF_RATING": "OFF_RATING_ON",
            "DEF_RATING": "DEF_RATING_ON",
            "MIN": "MIN_ON",
            "PLUS_MINUS": "PLUS_MINUS_ON",
        }
    )
    keep_off = [
        col
        for col in ("VS_PLAYER_ID", "OFF_RATING", "DEF_RATING", "MIN")
        if col in off_court.columns
    ]
    if keep_off:
        off_court = off_court[keep_off].rename(
            columns={
                "OFF_RATING": "OFF_RATING_OFF",
                "DEF_RATING": "DEF_RATING_OFF",
                "MIN": "MIN_OFF",
            }
        )
        return on_court.merge(off_court, on="VS_PLAYER_ID", how="left")
    return on_court


def _play_type(season: str, side: str, play: str, entity: str) -> pd.DataFrame:
    last_error: Exception | None = None
    for grouping in (side, side.capitalize()):
        try:
            frame = _safe_call(
                lambda grouping=grouping: SynergyPlayTypes(
                    season=season,
                    player_or_team_abbreviation=entity,
                    season_type_all_star="Regular Season",
                    per_mode_simple="Totals",
                    type_grouping_nullable=grouping,
                    play_type_nullable=play,
                    timeout=SETTINGS.request_timeout,
                )
            )
        except Exception as exc:
            last_error = exc
            continue
        if frame is not None and not frame.empty:
            frame = frame.copy()
            frame["SIDE"] = side
            frame["ENTITY_TYPE"] = entity
            return frame
    if last_error:
        raise last_error
    return pd.DataFrame()


def _matchups(
    warehouse: Warehouse,
    season: str,
    refresh: bool,
    failures: list[str],
    limit_games: int,
) -> None:
    if limit_games <= 0:
        return
    log = warehouse.read(f"raw_gamelog_T_{season}")
    if log is None or log.empty or "GAME_ID" not in log.columns:
        return
    game_ids = [str(value).zfill(10) for value in log["GAME_ID"].dropna().unique()]
    # Recent games are the ones a current slate can actually match.
    game_ids = game_ids[-limit_games:]
    frames = []
    for game_id in game_ids:
        frame = cached_frame(
            warehouse,
            f"raw_matchup_{game_id}",
            lambda game_id=game_id: _safe_call(
                lambda: BoxScoreMatchupsV3(game_id=game_id, timeout=SETTINGS.request_timeout)
            ),
            refresh,
            failures,
        )
        if frame is not None and not frame.empty:
            frames.append(frame)
    if frames:
        warehouse.write(f"matchups_{season}", pd.concat(frames, ignore_index=True))


def _combine(warehouse: Warehouse, seasons: list[str], refresh: bool, failures: list[str]) -> None:
    players = []
    teams = []
    plays = []
    zones = []
    tracking = []
    hustle = []
    lineups = []
    clutch = []
    on_off = []
    logs = []
    rosters = []
    for season in seasons:
        player = _player_table(warehouse, season)
        if player is not None:
            players.append(player)
        team = _team_table(warehouse, season)
        if team is not None:
            teams.append(team)
        plays.append(_play_table(warehouse, season))
        zones.append(_zone_table(warehouse, season))
        tracking.append(_tracking_table(warehouse, season))
        hustle.append(_hustle_table(warehouse, season))
        lineups.append(_lineup_table(warehouse, season))
        clutch.append(_clutch_table(warehouse, season))
        on_off.append(_on_off_table(warehouse, season))
        logs.append(_log_table(warehouse, season))
        rosters.append(_roster_table(warehouse, season))
    _write_combined(warehouse, "player_stats", players)
    _write_combined(warehouse, "team_stats", teams)
    _write_combined(warehouse, "play_types", plays)
    _write_combined(warehouse, "shot_zones", zones)
    _write_combined(warehouse, "tracking_defense", tracking)
    _write_combined(warehouse, "hustle", hustle)
    _write_combined(warehouse, "lineups", lineups)
    _write_combined(warehouse, "clutch", clutch)
    _write_combined(warehouse, "on_off", on_off)
    _write_combined(warehouse, "game_logs", logs)
    _write_combined(warehouse, "rosters", rosters)
    if not players:
        failures.append("player_stats: no season produced a player table")


def _write_combined(warehouse: Warehouse, name: str, frames: list[pd.DataFrame]) -> None:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if frames:
        warehouse.write(name, pd.concat(frames, ignore_index=True))


def _num(frame: pd.DataFrame, *names: str) -> pd.Series:
    series = first_number(frame, *names)
    if series is None:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return series


def _text(frame: pd.DataFrame, *names: str) -> pd.Series:
    lookup = {str(col).upper(): col for col in frame.columns}
    for name in names:
        col = lookup.get(name.upper())
        if col is not None:
            return frame[col]
    return pd.Series([None] * len(frame), index=frame.index)


def _player_table(warehouse: Warehouse, season: str) -> pd.DataFrame | None:
    base = warehouse.read(f"raw_player_Base_{season}")
    if base is None or base.empty:
        return None
    frame = pd.DataFrame(
        {
            "season": season,
            "player_id": _num(base, "PLAYER_ID"),
            "player_name": _text(base, "PLAYER_NAME"),
            "team_id": _num(base, "TEAM_ID"),
            "team_abbr": _text(base, "TEAM_ABBREVIATION").map(lambda value: normalize_team(value) or ""),
            "gp": _num(base, "GP"),
            "min": _num(base, "MIN"),
            "pts": _num(base, "PTS"),
            "reb": _num(base, "REB"),
            "ast": _num(base, "AST"),
            "stl": _num(base, "STL"),
            "blk": _num(base, "BLK"),
            "tov": _num(base, "TOV"),
            "fg3m": _num(base, "FG3M"),
            "fga": _num(base, "FGA"),
            "fta": _num(base, "FTA"),
            "ft_pct": _num(base, "FT_PCT"),
            "pf": _num(base, "PF"),
        }
    )
    advanced = warehouse.read(f"raw_player_Advanced_{season}")
    if advanced is not None and not advanced.empty:
        extra = pd.DataFrame(
            {
                "player_id": _num(advanced, "PLAYER_ID"),
                "usg_pct": _num(advanced, "USG_PCT"),
                "ts_pct": _num(advanced, "TS_PCT"),
                "efg_pct": _num(advanced, "EFG_PCT"),
                "ast_pct": _num(advanced, "AST_PCT"),
                "tov_pct": _num(advanced, "TM_TOV_PCT"),
                "oreb_pct": _num(advanced, "OREB_PCT"),
                "dreb_pct": _num(advanced, "DREB_PCT"),
                "reb_pct": _num(advanced, "REB_PCT"),
                "off_rating": _num(advanced, "OFF_RATING"),
                "def_rating": _num(advanced, "DEF_RATING"),
                "pace": _num(advanced, "PACE"),
            }
        )
        frame = frame.merge(extra, on="player_id", how="left")
    bio = warehouse.read(f"raw_bio_{season}")
    if bio is not None and not bio.empty and "PLAYER_ID" in bio.columns:
        height = _text(bio, "PLAYER_HEIGHT")
        bio_frame = pd.DataFrame(
            {"player_id": _num(bio, "PLAYER_ID"), "height_in": height.map(_height_inches)}
        )
        frame = frame.merge(bio_frame, on="player_id", how="left")
    roster = _roster_table(warehouse, season)
    if roster is not None and not roster.empty:
        roster = roster.drop_duplicates("player_id")
        frame = frame.merge(
            roster[["player_id", "position"]],
            on="player_id",
            how="left",
        )
    if "position" not in frame.columns:
        frame["position"] = None
    frame["position"] = [
        pos if isinstance(pos, str) and pos else _guess_position(row)
        for pos, row in zip(frame["position"], frame.to_dict("records"))
    ]
    return frame.dropna(subset=["player_id"])


def _team_table(warehouse: Warehouse, season: str) -> pd.DataFrame | None:
    base = warehouse.read(f"raw_team_Advanced_{season}")
    if base is None or base.empty:
        base = warehouse.read(f"raw_team_Base_{season}")
    if base is None or base.empty:
        return None
    frame = pd.DataFrame(
        {
            "season": season,
            "team_id": _num(base, "TEAM_ID"),
            "team_abbr": _text(base, "TEAM_ABBREVIATION").map(lambda value: normalize_team(value) or ""),
            "gp": _num(base, "GP"),
            "pace": _num(base, "PACE"),
            "off_rating": _num(base, "OFF_RATING"),
            "def_rating": _num(base, "DEF_RATING"),
            "net_rating": _num(base, "NET_RATING"),
            "oreb_pct": _num(base, "OREB_PCT"),
            "dreb_pct": _num(base, "DREB_PCT"),
        }
    )
    factors = warehouse.read(f"raw_team_Four Factors_{season}")
    if factors is not None and not factors.empty:
        extra = pd.DataFrame(
            {
                "team_id": _num(factors, "TEAM_ID"),
                "efg_pct": _num(factors, "EFG_PCT"),
                "tov_pct": _num(factors, "TM_TOV_PCT"),
                "fta_rate": _num(factors, "FTA_RATE"),
                "opp_efg_pct": _num(factors, "OPP_EFG_PCT"),
                "opp_tov_pct": _num(factors, "OPP_TOV_PCT"),
                "opp_oreb_pct": _num(factors, "OPP_OREB_PCT"),
                "opp_fta_rate": _num(factors, "OPP_FTA_RATE"),
            }
        )
        frame = frame.merge(extra, on="team_id", how="left")
    misc = warehouse.read(f"raw_team_Misc_{season}")
    if misc is not None and not misc.empty:
        extra = pd.DataFrame(
            {
                "team_id": _num(misc, "TEAM_ID"),
                "pts_paint": _num(misc, "PTS_PAINT"),
                "pts_fb": _num(misc, "PTS_FB"),
                "pts_2nd_chance": _num(misc, "PTS_2ND_CHANCE"),
                "pts_off_tov": _num(misc, "PTS_OFF_TOV"),
                "opp_pts_paint": _num(misc, "OPP_PTS_PAINT"),
                "opp_pts_fb": _num(misc, "OPP_PTS_FB"),
                "opp_pts_2nd_chance": _num(misc, "OPP_PTS_2ND_CHANCE"),
                "opp_pts_off_tov": _num(misc, "OPP_PTS_OFF_TOV"),
            }
        )
        frame = frame.merge(extra, on="team_id", how="left")
    return frame.dropna(subset=["team_id"])


def _play_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    for name in warehouse.tables():
        if name.startswith(f"raw_play_") and name.endswith(f"_{season}"):
            frame = warehouse.read(name)
            if frame is None or frame.empty:
                continue
            entity = "P" if "PLAYER_ID" in frame.columns else "T"
            ident = _num(frame, "PLAYER_ID") if entity == "P" else _num(frame, "TEAM_ID")
            frames.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "entity_type": frame["ENTITY_TYPE"] if "ENTITY_TYPE" in frame.columns else entity,
                        "entity_id": ident,
                        "player_name": _text(frame, "PLAYER_NAME"),
                        "team_id": _num(frame, "TEAM_ID"),
                        "side": frame["SIDE"] if "SIDE" in frame.columns else None,
                        "play_type": _text(frame, "PLAY_TYPE").map(canon_play),
                        "poss": _num(frame, "POSS"),
                        "poss_pct": _num(frame, "POSS_PCT"),
                        "ppp": _num(frame, "PPP"),
                        "fg_pct": _num(frame, "FG_PCT"),
                        "efg_pct": _num(frame, "EFG_PCT"),
                        "ft_poss_pct": _num(frame, "FT_POSS_PCT"),
                        "tov_poss_pct": _num(frame, "TOV_POSS_PCT"),
                        "score_poss_pct": _num(frame, "SCORE_POSS_PCT"),
                        "percentile": _num(frame, "PERCENTILE"),
                        "pts": _num(frame, "PTS"),
                        "gp": _num(frame, "GP"),
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _zone_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    player = warehouse.read(f"raw_player_shots_{season}")
    team = warehouse.read(f"raw_team_shots_{season}")
    # Flattening already ran, so the parser also accepts pipe-joined names.
    if player is not None:
        frames.append(parse_shot_zones(player, season, "P"))
    if team is not None:
        frames.append(parse_shot_zones(team, season, "T"))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# Zone files name the make rate after the cut. Overall is the only one on D_FG_PCT.
_TRACK_COLUMNS = {
    "Overall": ("D_FG_PCT", "NORMAL_FG_PCT", "D_FGA"),
    "Less Than 6Ft": ("LT_06_PCT", "NS_LT_06_PCT", "FGA_LT_06"),
    "Less Than 10Ft": ("LT_10_PCT", "NS_LT_10_PCT", "FGA_LT_10"),
    "2 Pointers": ("FG2_PCT", "NS_FG2_PCT", "FG2A"),
    "Greater Than 15Ft": ("GT_15_PCT", "NS_GT_15_PCT", "FGA_GT_15"),
    "3 Pointers": ("FG3_PCT", "NS_FG3_PCT", "FG3A"),
}


def _tracking_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    for category, zone in TRACK_ZONE.items():
        allowed, normal, attempts = _TRACK_COLUMNS.get(category, ("D_FG_PCT", "NORMAL_FG_PCT", "D_FGA"))
        for entity, prefix, id_name in (
            ("P", "raw_pt_defend", "CLOSE_DEF_PERSON_ID"),
            ("T", "raw_pt_team_defend", "TEAM_ID"),
        ):
            frame = warehouse.read(f"{prefix}_{category}_{season}")
            if frame is None or frame.empty:
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "season": season,
                        "entity_type": entity,
                        "entity_id": _num(frame, id_name, "PLAYER_ID", "TEAM_ID"),
                        "category": category,
                        "zone": zone,
                        "d_fg_pct": _num(frame, allowed),
                        "normal_fg_pct": _num(frame, normal),
                        "freq": _num(frame, "FREQ"),
                        "d_fga": _num(frame, attempts),
                    }
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _hustle_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    player = warehouse.read(f"raw_hustle_player_{season}")
    team = warehouse.read(f"raw_hustle_team_{season}")
    if player is not None and not player.empty:
        frames.append(_hustle_rows(player, season, "P", "PLAYER_ID"))
    if team is not None and not team.empty:
        frames.append(_hustle_rows(team, season, "T", "TEAM_ID"))
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _hustle_rows(frame: pd.DataFrame, season: str, entity: str, id_name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": season,
            "entity_type": entity,
            "entity_id": _num(frame, id_name),
            "box_outs": _num(frame, "BOX_OUTS"),
            "off_boxouts": _num(frame, "OFF_BOXOUTS"),
            "def_boxouts": _num(frame, "DEF_BOXOUTS"),
            "screen_assists": _num(frame, "SCREEN_ASSISTS"),
            "deflections": _num(frame, "DEFLECTIONS"),
            "charges": _num(frame, "CHARGES_DRAWN"),
            "loose_balls": _num(frame, "LOOSE_BALLS_RECOVERED"),
            "contested_shots": _num(frame, "CONTESTED_SHOTS"),
        }
    )


def _lineup_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frame = warehouse.read(f"raw_lineups_{season}")
    if frame is None or frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "season": season,
            "team_id": _num(frame, "TEAM_ID"),
            "group_id": _text(frame, "GROUP_ID"),
            "group_name": _text(frame, "GROUP_NAME"),
            "min": _num(frame, "MIN"),
            "off_rating": _num(frame, "OFF_RATING"),
            "def_rating": _num(frame, "DEF_RATING"),
            "pace": _num(frame, "PACE"),
            "oreb_pct": _num(frame, "OREB_PCT"),
        }
    )


def _clutch_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    player = warehouse.read(f"raw_clutch_player_{season}")
    team = warehouse.read(f"raw_clutch_team_{season}")
    if player is not None and not player.empty:
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "entity_type": "P",
                    "entity_id": _num(player, "PLAYER_ID"),
                    "gp": _num(player, "GP"),
                    "min": _num(player, "MIN"),
                    "pts": _num(player, "PTS"),
                    "off_rating": _num(player, "OFF_RATING"),
                    "usg_pct": _num(player, "USG_PCT"),
                }
            )
        )
    if team is not None and not team.empty:
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "entity_type": "T",
                    "entity_id": _num(team, "TEAM_ID"),
                    "gp": _num(team, "GP"),
                    "min": _num(team, "MIN"),
                    "pts": _num(team, "PTS"),
                    "off_rating": _num(team, "OFF_RATING"),
                    "usg_pct": pd.Series(pd.NA, index=team.index),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _on_off_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    for abbr in TEAM_ID_BY_ABBR:
        frame = warehouse.read(f"raw_onoff_{abbr}_{season}")
        if frame is None or frame.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "team_id": _num(frame, "TEAM_ID"),
                    "player_id": _num(frame, "VS_PLAYER_ID"),
                    "min": _num(frame, "MIN_ON", "MIN"),
                    "off_rating_on": _num(frame, "OFF_RATING_ON", "OFF_RATING"),
                    "def_rating_on": _num(frame, "DEF_RATING_ON", "DEF_RATING"),
                    "off_rating_off": _num(frame, "OFF_RATING_OFF"),
                    "def_rating_off": _num(frame, "DEF_RATING_OFF"),
                    "plus_minus": _num(frame, "PLUS_MINUS_ON", "PLUS_MINUS"),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _log_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    for entity, name in (("P", f"raw_gamelog_P_{season}"), ("T", f"raw_gamelog_T_{season}")):
        frame = warehouse.read(name)
        if frame is None or frame.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "entity_type": entity,
                    "game_id": _text(frame, "GAME_ID").map(lambda value: str(value).zfill(10)),
                    "date": pd.to_datetime(_text(frame, "GAME_DATE"), errors="coerce"),
                    "player_id": _num(frame, "PLAYER_ID"),
                    "team_id": _num(frame, "TEAM_ID"),
                    "matchup": _text(frame, "MATCHUP"),
                    "wl": _text(frame, "WL"),
                    "min": _text(frame, "MIN").map(_minutes_value),
                    "pts": _num(frame, "PTS"),
                    "reb": _num(frame, "REB"),
                    "ast": _num(frame, "AST"),
                    "stl": _num(frame, "STL"),
                    "blk": _num(frame, "BLK"),
                    "tov": _num(frame, "TOV"),
                    "fg3m": _num(frame, "FG3M"),
                    "fga": _num(frame, "FGA"),
                    "plus_minus": _num(frame, "PLUS_MINUS"),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _roster_table(warehouse: Warehouse, season: str) -> pd.DataFrame:
    frames = []
    for abbr, team_id in TEAM_ID_BY_ABBR.items():
        frame = warehouse.read(f"raw_roster_{abbr}_{season}")
        if frame is None or frame.empty:
            continue
        frames.append(
            pd.DataFrame(
                {
                    "season": season,
                    "team_id": team_id,
                    "team_abbr": abbr,
                    "player_id": _num(frame, "PLAYER_ID"),
                    "player_name": _text(frame, "PLAYER"),
                    "position": _text(frame, "POSITION").map(_clean_position),
                }
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).dropna(subset=["player_id"])


def _height_inches(value: object) -> float | None:
    text = str(value)
    if "-" not in text:
        return None
    feet, inches = text.split("-", 1)
    try:
        return int(feet) * 12 + int(inches)
    except ValueError:
        return None


def _clean_position(value: object) -> str | None:
    text = str(value or "").strip().upper()
    if not text or text == "NONE":
        return None
    if text.startswith("G"):
        return "G"
    if text.startswith("C"):
        return "C"
    if text.startswith("F"):
        return "F"
    return text[:1]


def _guess_position(row: dict) -> str:
    height = row.get("height_in")
    try:
        height_n = float(height)
    except (TypeError, ValueError):
        height_n = 78
    oreb = row.get("oreb_pct") or 0
    try:
        oreb_n = float(oreb)
    except (TypeError, ValueError):
        oreb_n = 0
    if oreb_n > 0.08 or height_n >= 82:
        return "C"
    if height_n >= 79:
        return "F"
    return "G"


def _minutes_value(value: object) -> float:
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        try:
            return float(minutes) + float(seconds) / 60.0
        except ValueError:
            return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def team_abbr(team_id: int) -> str:
    return ABBR_BY_TEAM_ID.get(int(team_id), "")
