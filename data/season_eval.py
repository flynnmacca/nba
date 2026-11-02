"""Score the current engine on a spread of 2025-26 regular-season finals.

Playoffs and the play-in stay out. Team play rates are that season's full
sample, so this is a calibration, not a forecast. A player is on the team
from his first game for that team through his last. Injury notes from this
season are hidden.
The official NBA box score says who was inactive and why. That absence is
held. A coach's decision stays available, because that minute is part of the test.
"""

from __future__ import annotations

import json
import random
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx
import numpy as np
import pandas as pd

from nba_sim.features.availability import box_absences, fetch_nba_box, hold_known_outs, nba_box_absences
from nba_sim.features.prepare import build_specs_for_date
from nba_sim.names import norm_person
from nba_sim.sim.engine import simulate_game
from nba_sim.teams import TEAM_ID_BY_ABBR, normalize_team
from nba_sim.warehouse import Warehouse
from opening_eval import (
    _empty_plays,
    expected_mix,
    fetch_summary,
    join_players,
    parse_box,
    parse_plays,
    possession_formula,
    team_actual,
)

N_SIMS = 80
SEED = 7
PER_MONTH = 12
MONTHS = ("202510", "202511", "202512", "202601", "202602", "202603", "202604")
SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
EASTERN = ZoneInfo("America/New_York")
OUT = r"C:\Users\flynn\nba\data\season_eval.json"


class SeasonBoard:
    """Serve one month of finals as the schedule, and no current injury list."""

    def __init__(self, inner: Warehouse, schedule: pd.DataFrame) -> None:
        self.inner = inner
        self.schedule = schedule

    def read(self, name: str):
        if name in {"schedule", "scoreboard"}:
            return self.schedule
        if name == "injuries":
            return pd.DataFrame(columns=["report_date", "comment", "player", "team"])
        return self.inner.read(name)

    def tables(self) -> list[str]:
        return self.inner.tables()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def _season_type(event: dict) -> int:
    comp = (event.get("competitions") or [{}])[0]
    season = comp.get("season") or event.get("season") or {}
    if not isinstance(season, dict):
        return 0
    try:
        return int(season.get("type") or 0)
    except (TypeError, ValueError):
        return 0


def _slate_date(event: dict) -> date:
    raw = str(event.get("date") or "")
    stamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return stamp.astimezone(EASTERN).date()


def fetch_finals(client: httpx.Client) -> list[dict]:
    rows = []
    for month in MONTHS:
        response = client.get(SCOREBOARD, params={"seasontype": 2, "limit": 1000, "dates": month}, timeout=60)
        response.raise_for_status()
        for event in response.json().get("events") or []:
            if _season_type(event) != 2:
                continue
            comp = (event.get("competitions") or [{}])[0]
            status = ((comp.get("status") or {}).get("type") or {}).get("name")
            if status != "STATUS_FINAL":
                continue
            home = away = ""
            home_pts = away_pts = 0.0
            for competitor in comp.get("competitors") or []:
                abbr = normalize_team((competitor.get("team") or {}).get("abbreviation")) or ""
                score = float(competitor.get("score") or 0)
                if competitor.get("homeAway") == "home":
                    home, home_pts = abbr, score
                else:
                    away, away_pts = abbr, score
            if not home or not away:
                continue
            on = _slate_date(event)
            rows.append(
                {
                    "game_id": str(event.get("id") or ""),
                    "date": on.isoformat(),
                    "month": on.strftime("%Y-%m"),
                    "home": home,
                    "away": away,
                    "actual_home": home_pts,
                    "actual_away": away_pts,
                    "start_utc": str(event.get("date") or ""),
                    "venue": str((comp.get("venue") or {}).get("fullName") or ""),
                    "status": "Final",
                }
            )
    return rows


def sample_games(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    by_month: dict[str, list[dict]] = {}
    for row in rows:
        by_month.setdefault(row["month"], []).append(row)
    picked = []
    for month in sorted(by_month):
        pool = list(by_month[month])
        rng.shuffle(pool)
        picked.extend(pool[:PER_MONTH])
    picked.sort(key=lambda row: (row["date"], row["game_id"]))
    return picked


def _sim_row(player, box: dict) -> dict:
    return {
        "name": player.name,
        "minutes": float(box["minutes"].mean()),
        "pts": float(box["pts"].mean()),
        "min_p10": float(np.percentile(box["minutes"], 10)),
        "min_p90": float(np.percentile(box["minutes"], 90)),
        "pts_p10": float(np.percentile(box["pts"], 10)),
        "pts_p90": float(np.percentile(box["pts"], 90)),
        "reb": float(box["reb"].mean()),
        "ast": float(box["ast"].mean()),
        "ast_p10": float(np.percentile(box["ast"], 10)),
        "ast_p90": float(np.percentile(box["ast"], 90)),
        "fga": float(box["fga"].mean()),
        "fgm": float(box["fgm"].mean()),
        "fta": float(box["fta"].mean()),
        "tov": float(box["tov"].mean()),
        "tpm": float(box["tpm"].mean()),
        "tpa": float(box["tpa"].mean()),
        "ftm": float(box["ftm"].mean()),
        "oreb": float(box["oreb"].mean()),
        "dreb": float(box["dreb"].mean()),
        "stl": float(box["stl"].mean()),
        "blk": float(box["blk"].mean()),
        "pf": float(box["pf"].mean()),
    }


def _sum(result, players, key: str) -> float:
    total = 0.0
    for player in players:
        box = result.player_stats.get(player.player_id)
        if box is None or key not in box:
            continue
        total += float(box[key].mean())
    return total


def _nba_game_id(logs: pd.DataFrame | None, on: date, away: str, home: str) -> str | None:
    """The stats.nba.com game id shared by these two teams on this date."""
    if logs is None or logs.empty or "game_id" not in logs.columns:
        return None
    away_id = TEAM_ID_BY_ABBR.get(normalize_team(away) or "")
    home_id = TEAM_ID_BY_ABBR.get(normalize_team(home) or "")
    if away_id is None or home_id is None or "date" not in logs.columns or "team_id" not in logs.columns:
        return None
    frame = logs[logs["entity_type"].astype(str).eq("T")] if "entity_type" in logs.columns else logs
    if frame.empty:
        return None
    days = pd.to_datetime(frame["date"], errors="coerce").dt.date
    team_ids = pd.to_numeric(frame["team_id"], errors="coerce")
    subset = frame.loc[days.eq(on) & team_ids.isin([away_id, home_id]), ["game_id", "team_id"]]
    if subset.empty:
        return None
    tokens = subset["game_id"].map(lambda value: str(value).split(".")[0].zfill(10))
    for game_id, part in subset.groupby(tokens):
        found = {int(value) for value in pd.to_numeric(part["team_id"], errors="coerce").dropna()}
        if away_id in found and home_id in found:
            return str(game_id)
    return None


def _unlabeled_bridges(logs: pd.DataFrame) -> set[tuple[int, int, str]]:
    """Unlabeled zeros that sit in the same absence as a named out. Finished seasons only."""
    from nba_sim.config import warehouse_dir
    from nba_sim.features.availability import known_out_reason
    from nba_sim.features.court import _absence_runs, _game_token

    if logs is None or logs.empty or "season" not in logs.columns:
        return set()
    season_games = {
        _game_token(value)
        for value in logs.loc[logs["season"].astype(str).eq("2025-26"), "game_id"].dropna()
    }
    part = pd.read_parquet(warehouse_dir() / "participation.parquet")
    part["game_id"] = part["game_id"].map(_game_token)
    part = part[part["game_id"].isin(season_games)].copy()
    if part.empty:
        return set()
    dated = logs.loc[:, ["game_id", "date"]].copy()
    dated["game_id"] = dated["game_id"].map(_game_token)
    dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
    dated = dated.dropna(subset=["date"]).drop_duplicates("game_id")
    part = part.merge(dated, on="game_id", how="inner")
    part["player_id"] = pd.to_numeric(part["player_id"], errors="coerce")
    part["team_id"] = pd.to_numeric(part["team_id"], errors="coerce")
    part["min"] = pd.to_numeric(part["min"], errors="coerce").fillna(0.0)
    part["reason"] = part["reason"].fillna("").astype(str).replace({"nan": "", "None": ""})
    part = part.dropna(subset=["player_id", "team_id"])
    keys: set[tuple[int, int, str]] = set()
    for (pid, tid), group in part.groupby(["player_id", "team_id"], sort=False):
        group = group.sort_values("date").drop_duplicates("game_id", keep="last")
        flags = _absence_runs(group["min"].tolist(), group["reason"].tolist())
        for game_id, flag, reason in zip(group["game_id"], flags, group["reason"]):
            if flag and not known_out_reason(reason):
                keys.add((int(tid), int(pid), str(game_id)))
    return keys


def _dump(destination: str, finals_available: int, games_out: list, players_out: list, plays_out: list) -> None:
    payload = {
        "season": "2025-26",
        "n_sims": N_SIMS,
        "seed": SEED,
        "per_month": PER_MONTH,
        "finals_available": finals_available,
        "games": games_out,
        "players": players_out,
        "sides": plays_out,
    }
    path = Path(destination)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def main(out: str | None = None, hold_bridges: bool = False) -> None:
    destination = out or OUT
    warehouse = Warehouse()
    logs = warehouse.read("game_logs")
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "nba-sim/0.1"}) as client:
        games_out = []
        players_out = []
        plays_out = []
        done: set[str] = set()
        saved = Path(destination)
        if saved.exists():
            try:
                prior = json.loads(saved.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = {}
            if prior.get("n_sims") == N_SIMS and prior.get("seed") == SEED:
                games_out = list(prior.get("games") or [])
                players_out = list(prior.get("players") or [])
                plays_out = list(prior.get("sides") or [])
                done = {str(game.get("game_id")) for game in games_out}
                print(f"resume {len(done)}", flush=True)
        spec_cache: dict[date, list] = {}
        finals_available = 0
        if hold_bridges:
            locked = json.loads(open(OUT, encoding="utf-8").read())
            chosen = [
                {
                    "game_id": str(game["game_id"]),
                    "date": game["date"],
                    "month": str(game.get("month") or game["date"][:7]),
                    "home": str(game["home"]).lower(),
                    "away": str(game["away"]).lower(),
                    "actual_home": game.get("actual_home"),
                    "actual_away": game.get("actual_away"),
                    "start_utc": "",
                    "venue": "",
                    "status": "Final",
                }
                for game in locked["games"]
            ]
            print(f"locked sample {len(chosen)}", flush=True)
            finals_available = len(chosen)
        else:
            finals = fetch_finals(client)
            finals_available = len(finals)
            print(f"regular-season finals {len(finals)}", flush=True)
            chosen = sample_games(finals)
            print(f"sample {len(chosen)}", flush=True)
        frame = pd.DataFrame(chosen)
        board = SeasonBoard(warehouse, frame)
        bridges = _unlabeled_bridges(logs) if hold_bridges else set()
        if hold_bridges:
            print(f"bridge zeros {len(bridges)}", flush=True)
        for row in chosen:
            if str(row["game_id"]) in done:
                continue
            on = date.fromisoformat(row["date"])
            if on not in spec_cache:
                spec_cache[on] = build_specs_for_date(on, board)
                print(f"specs {on} {len(spec_cache[on])}", flush=True)
            specs = [spec for spec in spec_cache[on] if str(spec.game_id) == row["game_id"]]
            if not specs:
                print(f"skip {row['away']} at {row['home']}", flush=True)
                continue
            spec = specs[0]
            print(f"sim {row['date']} {row['away']} at {row['home']}", flush=True)
            nba_id = _nba_game_id(logs, on, spec.away.abbr, spec.home.abbr)
            nba_box = None
            for attempt in range(4):
                try:
                    payload = fetch_summary(client, row["game_id"])
                    nba_box = fetch_nba_box(client, nba_id, timeout=90) if nba_id else None
                    break
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    print(f"  retry {type(exc).__name__} {attempt + 1}", flush=True)
                    if attempt == 3:
                        print("  skipped after timeouts", flush=True)
                        payload = None
                        break
                    time.sleep(3 * (attempt + 1))
            if payload is None:
                continue
            actual_players, team_stats, ids = parse_box(payload)
            play_counts = parse_plays(payload, ids)
            absences = nba_box_absences(nba_box) if nba_box else box_absences(payload)
            if nba_id and bridges:
                absences = list(absences)
                for side, side_players in ((spec.home, spec.home_players), (spec.away, spec.away_players)):
                    for player in side_players:
                        if (int(player.team_id), int(player.player_id), nba_id) in bridges:
                            absences.append({"team": side.abbr, "name": player.name, "reason": "Out"})
            held = hold_known_outs(spec.home_players, spec.home.abbr, absences)
            held += hold_known_outs(spec.away_players, spec.away.abbr, absences)
            if held:
                print("  out", ", ".join(held).encode("ascii", "replace").decode(), flush=True)
            result = simulate_game(spec, n_sims=N_SIMS, seed=SEED)
            home_actual = team_actual(team_stats.get(spec.home.abbr, {}), actual_players, spec.home.abbr)
            away_actual = team_actual(team_stats.get(spec.away.abbr, {}), actual_players, spec.away.abbr)
            header = ((payload.get("header") or {}).get("competitions") or [{}])[0]
            for competitor in header.get("competitors") or []:
                abbr = normalize_team((competitor.get("team") or {}).get("abbreviation"))
                score = float(competitor.get("score") or 0)
                if abbr == spec.home.abbr:
                    home_actual["pts"] = score
                elif abbr == spec.away.abbr:
                    away_actual["pts"] = score
            home_sim = float(result.home_points.mean())
            away_sim = float(result.away_points.mean())
            game = {
                "date": row["date"],
                "month": row["month"],
                "game_id": row["game_id"],
                "away": spec.away.abbr.upper(),
                "home": spec.home.abbr.upper(),
                "sim_away": round(away_sim, 1),
                "sim_home": round(home_sim, 1),
                "actual_away": away_actual["pts"],
                "actual_home": home_actual["pts"],
                "home_win_p": round(float((result.home_points > result.away_points).mean()), 3),
                "margin_sd": round(float(np.std(result.home_points - result.away_points)), 2),
                "actual_home_win": int(home_actual["pts"] > away_actual["pts"]),
                "spread_error": round((home_sim - away_sim) - (home_actual["pts"] - away_actual["pts"]), 1),
                "total_error": round((home_sim + away_sim) - (home_actual["pts"] + away_actual["pts"]), 1),
                "away_off": round(float(spec.away.off_rating), 1),
                "home_off": round(float(spec.home.off_rating), 1),
                "away_def": round(float(spec.away.def_rating), 1),
                "home_def": round(float(spec.home.def_rating), 1),
            }
            for side, actual, players in (
                ("away", away_actual, spec.away_players),
                ("home", home_actual, spec.home_players),
            ):
                raw = {
                    key: _sum(result, players, key)
                    for key in ("fga", "fgm", "fta", "ftm", "tpa", "tov", "oreb", "dreb", "reb", "tpm", "ast", "stl", "blk", "pf")
                }
                for key, value in raw.items():
                    game[f"sim_{side}_{key}"] = round(value, 1)
                    game[f"actual_{side}_{key}"] = round(float(actual.get(key, 0.0)), 1)
                # The trip counter is the clock. The possession is the box formula on both sides.
                game[f"sim_{side}_trips"] = round(float(getattr(result, f"{side}_possessions").mean()), 1)
                game[f"sim_{side}_poss"] = round(
                    possession_formula(raw["fga"], raw["oreb"], raw["tov"], raw["fta"]), 1
                )
                game[f"actual_{side}_poss"] = round(float(actual["poss"]), 1)
                game[f"actual_{side}_paint"] = round(float(actual.get("paint", 0.0)), 1)
                game[f"actual_{side}_fast_break"] = round(float(actual.get("fast_break", 0.0)), 1)
                game[f"sim_{side}_second"] = round(float(getattr(result, f"second_chance_{side}").mean()), 2)
                mix = expected_mix(players)
                plays = play_counts.get(spec.away.abbr if side == "away" else spec.home.abbr) or _empty_plays()
                plays_out.append(
                    {
                        "date": row["date"],
                        "month": row["month"],
                        "game": f"{spec.away.abbr.upper()} at {spec.home.abbr.upper()}",
                        "team": (spec.away.abbr if side == "away" else spec.home.abbr).upper(),
                        "actual": {key: round(float(value), 1) if isinstance(value, float) else value for key, value in actual.items()},
                        "plays": plays,
                        "mix": {key: round(float(value), 4) for key, value in mix.items()},
                    }
                )
            game["sim_flagrant"] = round(float(np.mean(result.flagrant_fouls)), 3)
            game["sim_ejections"] = round(float(np.mean(result.ejections)), 3)
            game["sim_intentional"] = round(float(np.mean(result.intentional_fouls)), 2)
            games_out.append(game)
            home_rows = [
                _sim_row(player, result.player_stats[player.player_id])
                for player in spec.home_players
                if player.player_id in result.player_stats
            ]
            away_rows = [
                _sim_row(player, result.player_stats[player.player_id])
                for player in spec.away_players
                if player.player_id in result.player_stats
            ]
            # join_players expects the sim dict shape from opening_eval.sim_players
            for joined in join_players(actual_players, away_rows, spec.away.abbr.upper()) + join_players(actual_players, home_rows, spec.home.abbr.upper()):
                joined["date"] = row["date"]
                joined["month"] = row["month"]
                joined["game"] = f"{spec.away.abbr.upper()} at {spec.home.abbr.upper()}"
                players_out.append(joined)
            print(
                f"  actual {away_actual['pts']:.0f}-{home_actual['pts']:.0f} sim {away_sim:.1f}-{home_sim:.1f}",
                flush=True,
            )
            _dump(destination, finals_available, games_out, players_out, plays_out)
    print("wrote", destination, "games", len(games_out), "players", len(players_out), flush=True)


if __name__ == "__main__":
    main()
