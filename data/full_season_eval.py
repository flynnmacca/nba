"""Score every regular-season final in 2025-26 and every 2026-27 final to date.

A new file per season. An existing file with the same sim count and seed is
resumed. Older sample files are left alone.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from nba_sim.features.availability import box_absences, fetch_nba_box, hold_known_outs, nba_box_absences
from nba_sim.features.prepare import build_specs_for_date
from nba_sim.sim.engine import simulate_game
from nba_sim.teams import normalize_team
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
from season_eval import (
    SCOREBOARD,
    SeasonBoard,
    _nba_game_id,
    _season_type,
    _sim_row,
    _slate_date,
    _sum,
    _unlabeled_bridges,
)

N_SIMS = 120
SEED = 7
WORKERS = 8
LAST = "2025-26"
NOW = "2026-27"
MONTHS = {
    LAST: ("202510", "202511", "202512", "202601", "202602", "202603", "202604"),
    NOW: ("202610",),
}
PATHS = {
    LAST: Path(r"C:\Users\flynn\nba\data\season_eval_2025_26_adapt_fix.json"),
    NOW: Path(r"C:\Users\flynn\nba\data\season_eval_2026_27_adapt_fix.json"),
}


def fetch_finals(client: httpx.Client, months: tuple[str, ...]) -> list[dict]:
    rows = []
    for month in months:
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
                    "status": "Final",
                }
            )
    rows.sort(key=lambda row: (row["date"], row["game_id"]))
    return rows


def _load(path: Path) -> tuple[list, list, list, set[str]]:
    if not path.exists():
        return [], [], [], set()
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], [], [], set()
    if prior.get("n_sims") != N_SIMS or prior.get("seed") != SEED:
        return [], [], [], set()
    games = list(prior.get("games") or [])
    return games, list(prior.get("players") or []), list(prior.get("sides") or []), {str(game.get("game_id")) for game in games}


def _dump(path: Path, season: str, available: int, games: list, players: list, plays: list) -> None:
    payload = {
        "season": season,
        "n_sims": N_SIMS,
        "seed": SEED,
        "finals_available": available,
        "games": games,
        "players": players,
        "sides": plays,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(path)


def _score_game(client: httpx.Client, spec, row: dict, logs, bridges: set) -> tuple[dict, list, list] | None:
    on = date.fromisoformat(row["date"])
    nba_id = _nba_game_id(logs, on, spec.away.abbr, spec.home.abbr)
    payload = None
    nba_box = None
    for attempt in range(4):
        try:
            payload = fetch_summary(client, row["game_id"])
            nba_box = fetch_nba_box(client, nba_id, timeout=90) if nba_id else None
            break
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as exc:
            print(f"  retry {row['game_id']} {type(exc).__name__} {attempt + 1}", flush=True)
            if attempt == 3:
                return None
            time.sleep(3 * (attempt + 1))
    if payload is None:
        return None
    actual_players, team_stats, ids = parse_box(payload)
    play_counts = parse_plays(payload, ids)
    absences = nba_box_absences(nba_box) if nba_box else box_absences(payload)
    if nba_id and bridges:
        absences = list(absences)
        for side, side_players in ((spec.home, spec.home_players), (spec.away, spec.away_players)):
            for player in side_players:
                if (int(player.team_id), int(player.player_id), nba_id) in bridges:
                    absences.append({"team": side.abbr, "name": player.name, "reason": "Out"})
    hold_known_outs(spec.home_players, spec.home.abbr, absences)
    hold_known_outs(spec.away_players, spec.away.abbr, absences)
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
    plays_out = []
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
        game[f"sim_{side}_trips"] = round(float(getattr(result, f"{side}_possessions").mean()), 1)
        game[f"sim_{side}_poss"] = round(possession_formula(raw["fga"], raw["oreb"], raw["tov"], raw["fta"]), 1)
        game[f"actual_{side}_poss"] = round(float(actual["poss"]), 1)
        game[f"actual_{side}_paint"] = round(float(actual.get("paint", 0.0)), 1)
        game[f"actual_{side}_fast_break"] = round(float(actual.get("fast_break", 0.0)), 1)
        game[f"sim_{side}_second"] = round(float(getattr(result, f"second_chance_{side}").mean()), 2)
        mix = expected_mix(players)
        team = spec.away.abbr if side == "away" else spec.home.abbr
        plays = play_counts.get(team) or _empty_plays()
        plays_out.append(
            {
                "date": row["date"],
                "month": row["month"],
                "game": f"{spec.away.abbr.upper()} at {spec.home.abbr.upper()}",
                "team": team.upper(),
                "actual": {key: round(float(value), 1) if isinstance(value, float) else value for key, value in actual.items()},
                "plays": plays,
                "mix": {key: round(float(value), 4) for key, value in mix.items()},
            }
        )
    game["sim_flagrant"] = round(float(np.mean(result.flagrant_fouls)), 3)
    game["sim_ejections"] = round(float(np.mean(result.ejections)), 3)
    game["sim_intentional"] = round(float(np.mean(result.intentional_fouls)), 2)
    home_rows = [_sim_row(player, result.player_stats[player.player_id]) for player in spec.home_players if player.player_id in result.player_stats]
    away_rows = [_sim_row(player, result.player_stats[player.player_id]) for player in spec.away_players if player.player_id in result.player_stats]
    players_out = []
    for joined in join_players(actual_players, away_rows, spec.away.abbr.upper()) + join_players(actual_players, home_rows, spec.home.abbr.upper()):
        joined["date"] = row["date"]
        joined["month"] = row["month"]
        joined["game"] = f"{spec.away.abbr.upper()} at {spec.home.abbr.upper()}"
        players_out.append(joined)
    print(
        f"  {row['date']} {spec.away.abbr.upper()} at {spec.home.abbr.upper()} actual {away_actual['pts']:.0f}-{home_actual['pts']:.0f} sim {away_sim:.1f}-{home_sim:.1f}",
        flush=True,
    )
    return game, players_out, plays_out


def score_date(job: dict) -> dict:
    """One slate date. The warehouse is read in this process."""
    rows = job["rows"]
    on = date.fromisoformat(rows[0]["date"])
    warehouse = Warehouse()
    logs = warehouse.read("game_logs")
    bridges = _unlabeled_bridges(logs) if job["season"] == LAST else set()
    board = SeasonBoard(warehouse, pd.DataFrame(rows))
    specs = {str(spec.game_id): spec for spec in build_specs_for_date(on, board)}
    games: list = []
    players: list = []
    plays: list = []
    missed: list[str] = []
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "nba-sim/0.1"}) as client:
        for row in rows:
            spec = specs.get(str(row["game_id"]))
            if spec is None:
                print(f"skip {row['date']} {row['away']} at {row['home']}", flush=True)
                missed.append(str(row["game_id"]))
                continue
            scored = _score_game(client, spec, row, logs, bridges)
            if scored is None:
                missed.append(str(row["game_id"]))
                continue
            game, joined, side = scored
            games.append(game)
            players.extend(joined)
            plays.extend(side)
    return {"season": job["season"], "date": rows[0]["date"], "games": games, "players": players, "plays": plays, "missed": missed}


def _jobs(rows: list[dict], season: str, done: set[str], limit: int | None) -> list[dict]:
    pending = [row for row in rows if str(row["game_id"]) not in done]
    if limit is not None:
        pending = pending[:limit]
    by_date: dict[str, list[dict]] = {}
    for row in pending:
        by_date.setdefault(row["date"], []).append(row)
    return [{"season": season, "rows": by_date[key]} for key in sorted(by_date)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--season", choices=(LAST, NOW, "both"), default="both")
    args = parser.parse_args()
    seasons = (LAST, NOW) if args.season == "both" else (args.season,)
    slate: dict[str, list[dict]] = {}
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "nba-sim/0.1"}) as client:
        for season in seasons:
            slate[season] = fetch_finals(client, MONTHS[season])
            print(f"{season} finals {len(slate[season])}", flush=True)
    books = {}
    for season in seasons:
        games, players, plays, done = _load(PATHS[season])
        books[season] = {"games": games, "players": players, "plays": plays, "done": done, "available": len(slate[season])}
        print(f"{season} resume {len(done)}", flush=True)
    jobs = []
    for season in seasons:
        jobs.extend(_jobs(slate[season], season, books[season]["done"], 1 if args.smoke else None))
    if not jobs:
        print("nothing to sim", flush=True)
        return
    workers = 1 if args.smoke else WORKERS
    # One BLAS thread per worker. Eight workers each spinning every core
    # oversubscribed the last full run by about 8×.
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    print(f"dates {len(jobs)} workers {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(score_date, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            season = result["season"]
            book = books[season]
            book["games"].extend(result["games"])
            book["players"].extend(result["players"])
            book["plays"].extend(result["plays"])
            book["done"].update(str(game["game_id"]) for game in result["games"])
            if not args.smoke:
                _dump(PATHS[season], season, book["available"], book["games"], book["players"], book["plays"])
            print(f"saved {season} {len(book['games'])}/{book['available']} missed {len(result['missed'])}", flush=True)
    for season in seasons:
        book = books[season]
        print(f"wrote {PATHS[season]} games {len(book['games'])} players {len(book['players'])}", flush=True)


if __name__ == "__main__":
    main()
