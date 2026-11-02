"""Score the pregame sim on the first two final slates. No result is fed back in."""

from __future__ import annotations

import json
import re
import time
from datetime import date

import httpx
import numpy as np
import pandas as pd

from nba_sim.constants import PLAY_TYPES
from nba_sim.features.prepare import build_specs_for_date
from nba_sim.names import norm_person
from nba_sim.sim.clock import PLAY_THREE
from nba_sim.sim.engine import simulate_game
from nba_sim.teams import normalize_team
from nba_sim.warehouse import Warehouse

N_SIMS = 500
SEED = 7
DATES = (date(2026, 10, 20), date(2026, 10, 21))
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"
POSTGAME = re.compile(r"\d{2,3}\s*-\s*\d{2,3}|finished with|did not return|didn't return|loss to|win over", re.I)


class PregameWarehouse:
    """Injury rows dated on or after the slate are hidden. Results never enter."""

    def __init__(self, inner: Warehouse, on: date) -> None:
        self.inner = inner
        self.on = on

    def read(self, name: str):
        frame = self.inner.read(name)
        if name != "injuries" or frame is None or frame.empty or "report_date" not in frame.columns:
            return frame
        reported = pd.to_datetime(frame["report_date"], errors="coerce")
        keep = reported.notna() & (reported.dt.date < self.on)
        comment = frame.get("comment", pd.Series("", index=frame.index)).astype(str)
        keep = keep & ~comment.map(lambda text: bool(POSTGAME.search(text)))
        return frame.loc[keep].copy()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def _minutes(text: object, did_not_play: bool) -> float:
    if did_not_play:
        return 0.0
    raw = str(text or "").strip()
    if raw in {"", "--", "DNP", "NA"}:
        return 0.0
    if ":" in raw:
        mins, secs = raw.split(":", 1)
        return float(mins) + float(secs) / 60.0
    try:
        return float(raw)
    except ValueError:
        return 0.0


def _made_att(text: object) -> tuple[float, float]:
    raw = str(text or "0-0")
    if "-" not in raw:
        return 0.0, 0.0
    made, att = raw.split("-", 1)
    try:
        return float(made), float(att)
    except ValueError:
        return 0.0, 0.0


def _num(text: object) -> float:
    raw = str(text or "0").replace("+", "")
    try:
        return float(raw)
    except ValueError:
        return 0.0


def fetch_summary(client: httpx.Client, game_id: str) -> dict:
    last: Exception | None = None
    for attempt in range(4):
        try:
            response = client.get(SUMMARY, params={"event": game_id}, timeout=90)
            response.raise_for_status()
            return response.json()
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    if last:
        raise last
    raise RuntimeError(game_id)


def parse_box(payload: dict) -> tuple[list[dict], dict[str, dict], dict[str, str]]:
    players = []
    teams = {}
    ids = {}
    for side in payload["boxscore"]["players"]:
        abbr = normalize_team((side.get("team") or {}).get("abbreviation")) or ""
        ids[str((side.get("team") or {}).get("id"))] = abbr
        group = (side.get("statistics") or [{}])[0]
        labels = list(group.get("labels") or group.get("keys") or [])
        for athlete in group.get("athletes") or []:
            person = athlete.get("athlete") or {}
            stats = list(athlete.get("stats") or [])
            row = {label: stats[i] if i < len(stats) else "" for i, label in enumerate(labels)}
            fg_m, fg_a = _made_att(row.get("FG"))
            th_m, th_a = _made_att(row.get("3PT"))
            ft_m, ft_a = _made_att(row.get("FT"))
            players.append(
                {
                    "team": abbr,
                    "name": str(person.get("displayName") or ""),
                    "minutes": _minutes(row.get("MIN"), bool(athlete.get("didNotPlay"))),
                    "pts": _num(row.get("PTS")),
                    "reb": _num(row.get("REB")),
                    "oreb": _num(row.get("OREB")),
                    "dreb": max(0.0, _num(row.get("REB")) - _num(row.get("OREB"))),
                    "ast": _num(row.get("AST")),
                    "tov": _num(row.get("TO")),
                    "stl": _num(row.get("STL")),
                    "blk": _num(row.get("BLK")),
                    "pf": _num(row.get("PF")),
                    "fgm": fg_m,
                    "fga": fg_a,
                    "tpm": th_m,
                    "tpa": th_a,
                    "ftm": ft_m,
                    "fta": ft_a,
                }
            )
    for side in payload["boxscore"]["teams"]:
        abbr = normalize_team((side.get("team") or {}).get("abbreviation")) or ""
        ids[str((side.get("team") or {}).get("id"))] = abbr
        stats = {}
        for item in side.get("statistics") or []:
            stats[str(item.get("name") or item.get("label") or "")] = item.get("displayValue")
        teams[abbr] = stats
    return players, teams, ids


def _is_free_throw(label: str) -> bool:
    return label.startswith("free throw")


def _shot_family(play: dict) -> str:
    label = str((play.get("type") or {}).get("text") or "")
    text = label.lower()
    if play.get("pointsAttempted") == 3 or "three point" in str(play.get("text") or "").lower():
        return "three"
    if any(word in text for word in ("dunk", "layup", "tip shot", "finger roll", "putback")):
        return "rim"
    if "jump" in text or "hook" in text or "fade" in text:
        return "mid"
    return "other"


def parse_plays(payload: dict, ids: dict[str, str]) -> dict[str, dict]:
    found = {abbr: _empty_plays() for abbr in ids.values() if abbr}
    for play in payload.get("plays") or []:
        abbr = ids.get(str((play.get("team") or {}).get("id")))
        if not abbr or abbr not in found:
            continue
        bucket = found[abbr]
        label = str((play.get("type") or {}).get("text") or "").lower()
        if _is_free_throw(label):
            bucket["fta"] += 1
            if play.get("scoringPlay"):
                bucket["ftm"] += 1
            continue
        if play.get("shootingPlay"):
            bucket["fga"] += 1
            family = _shot_family(play)
            bucket[family] += 1
            if play.get("scoringPlay"):
                bucket["fgm"] += 1
            if "putback" in label or "tip shot" in label:
                bucket["putback"] += 1
        if "turnover" in label:
            bucket["tov"] += 1
        if label.startswith("offensive rebound"):
            bucket["oreb"] += 1
        if label.startswith("defensive rebound"):
            bucket["dreb"] += 1
        if "shooting foul" in label:
            bucket["shooting_foul"] += 1
        if "loose ball" in label:
            bucket["loose_ball"] += 1
        elif label.startswith("personal"):
            bucket["personal_foul"] += 1
        if "offensive foul" in label:
            bucket["offensive_foul"] += 1
    return found


def _empty_plays() -> dict:
    return {key: 0 for key in ("fga", "fgm", "three", "rim", "mid", "other", "putback", "fta", "ftm", "tov", "oreb", "dreb", "shooting_foul", "personal_foul", "loose_ball", "offensive_foul")}


def _pair(value: object) -> tuple[float, float]:
    text = str(value or "0-0")
    if "-" not in text:
        return _num(text), 0.0
    return _made_att(text)


def possession_formula(fga: float, oreb: float, tov: float, fta: float) -> float:
    """Box-score possessions: attempts, minus the glass, plus turnovers and free throws."""
    return float(fga) - float(oreb) + float(tov) + 0.44 * float(fta)


def team_actual(stats: dict, players: list[dict], abbr: str) -> dict:
    fg_m, fg_a = _pair(stats.get("fieldGoalsMade-fieldGoalsAttempted"))
    th_m, th_a = _pair(stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
    ft_m, ft_a = _pair(stats.get("freeThrowsMade-freeThrowsAttempted"))
    oreb = _num(stats.get("offensiveRebounds"))
    dreb = _num(stats.get("defensiveRebounds"))
    tov = _num(stats.get("totalTurnovers"))
    poss = possession_formula(fg_a, oreb, tov, ft_a)
    return {
        "pts": sum(player["pts"] for player in players if player["team"] == abbr),
        "fgm": fg_m,
        "fga": fg_a,
        "tpm": th_m,
        "tpa": th_a,
        "ftm": ft_m,
        "fta": ft_a,
        "oreb": oreb,
        "dreb": dreb,
        "reb": _num(stats.get("totalRebounds")),
        "ast": _num(stats.get("assists")),
        "tov": tov,
        "stl": _num(stats.get("steals")),
        "blk": _num(stats.get("blocks")),
        "pf": _num(stats.get("fouls")),
        "paint": _num(stats.get("pointsInPaint")),
        "fast_break": _num(stats.get("fastBreakPoints")),
        "poss": poss,
    }


def sim_players(spec_players, packed: dict) -> list[dict]:
    rows = []
    for player in spec_players:
        box = packed.get(player.player_id)
        if box is None:
            continue
        rows.append(
            {
                "name": player.name,
                "minutes": float(box["minutes"].mean()),
                "min_p10": float(np.percentile(box["minutes"], 10)),
                "min_p90": float(np.percentile(box["minutes"], 90)),
                "pts": float(box["pts"].mean()),
                "pts_p10": float(np.percentile(box["pts"], 10)),
                "pts_p90": float(np.percentile(box["pts"], 90)),
                "reb": float(box["reb"].mean()),
                "ast": float(box["ast"].mean()),
                "ast_p10": float(np.percentile(box["ast"], 10)),
                "ast_p90": float(np.percentile(box["ast"], 90)),
                "tov": float(box["tov"].mean()),
                "fga": float(box["fga"].mean()),
                "fgm": float(box["fgm"].mean()),
                "fta": float(box["fta"].mean()),
                "tpm": float(box["tpm"].mean()),
                "stl": float(box["stl"].mean()),
                "blk": float(box["blk"].mean()),
                "pf": float(box["pf"].mean()),
                "oreb": float(box["oreb"].mean()),
                "dreb": float(box["dreb"].mean()),
                "ftm": float(box["ftm"].mean()),
                "tpa": float(box["tpa"].mean()),
            }
        )
    return rows


def expected_mix(players) -> dict[str, float]:
    weight = {play: 0.0 for play in PLAY_TYPES}
    ft = 0.0
    tov = 0.0
    for player in players:
        minutes = float(player.minutes_mean)
        if minutes <= 0:
            continue
        for play in PLAY_TYPES:
            skill = player.offense.get(play)
            if skill is None or skill.poss <= 0:
                continue
            share = minutes * float(skill.poss)
            weight[play] += share
            ft += share * float(skill.ft_poss_pct)
            tov += share * float(skill.tov_poss_pct)
    total = sum(weight.values()) or 1.0
    mix = {play: weight[play] / total for play in PLAY_TYPES}
    three = sum(mix[play] * float(PLAY_THREE[index]) for index, play in enumerate(PLAY_TYPES))
    mix["three_rate"] = three
    mix["ft_rate"] = ft / total
    mix["tov_rate"] = tov / total
    return mix


def _foul_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    if values.ndim == 2:
        return float(values.sum(axis=1).mean())
    return float(values.mean())


def join_players(actual: list[dict], simulated: list[dict], team: str) -> list[dict]:
    team_key = team.lower()
    by_name = {norm_person(player["name"]): player for player in simulated}
    used = set()
    rows = []
    for player in actual:
        if player["team"] != team_key:
            continue
        key = norm_person(player["name"])
        sim = by_name.get(key)
        if sim is not None:
            used.add(key)
        rows.append(_joined(player["name"], team, player, sim))
    for sim in simulated:
        key = norm_person(sim["name"])
        if key in used:
            continue
        rows.append(_joined(sim["name"], team, None, sim))
    return rows


def _joined(name: str, team: str, actual: dict | None, sim: dict | None) -> dict:
    actual_min = 0.0 if actual is None else actual["minutes"]
    sim_min = 0.0 if sim is None else sim["minutes"]
    actual_pts = 0.0 if actual is None else actual["pts"]
    sim_pts = 0.0 if sim is None else sim["pts"]
    ppm = sim_pts / sim_min if sim_min > 1 else 0.0
    return {
        "name": name,
        "team": team,
        "actual_min": round(actual_min, 1),
        "sim_min": round(sim_min, 1),
        "min_p10": None if sim is None else round(sim["min_p10"], 1),
        "min_p90": None if sim is None else round(sim["min_p90"], 1),
        "actual_pts": round(actual_pts, 1),
        "sim_pts": round(sim_pts, 1),
        "pts_p10": None if sim is None else round(sim["pts_p10"], 1),
        "pts_p90": None if sim is None else round(sim["pts_p90"], 1),
        "actual_reb": 0.0 if actual is None else actual["reb"],
        "sim_reb": 0.0 if sim is None else round(sim["reb"], 1),
        "actual_ast": 0.0 if actual is None else actual["ast"],
        "sim_ast": 0.0 if sim is None else round(sim["ast"], 1),
        "ast_p10": None if sim is None or sim.get("ast_p10") is None else round(float(sim["ast_p10"]), 1),
        "ast_p90": None if sim is None or sim.get("ast_p90") is None else round(float(sim["ast_p90"]), 1),
        "actual_fgm": 0.0 if actual is None else actual.get("fgm", 0.0),
        "sim_fgm": 0.0 if sim is None else round(float(sim.get("fgm", 0.0)), 1),
        "actual_fga": 0.0 if actual is None else actual["fga"],
        "sim_fga": 0.0 if sim is None else round(sim["fga"], 1),
        "actual_fta": 0.0 if actual is None else actual["fta"],
        "sim_fta": 0.0 if sim is None else round(sim["fta"], 1),
        "actual_tov": 0.0 if actual is None else actual["tov"],
        "sim_tov": 0.0 if sim is None else round(sim["tov"], 1),
        "actual_tpm": 0.0 if actual is None else actual["tpm"],
        "sim_tpm": 0.0 if sim is None else round(sim["tpm"], 1),
        "actual_tpa": 0.0 if actual is None else actual.get("tpa", 0.0),
        "sim_tpa": 0.0 if sim is None else round(float(sim.get("tpa", 0.0)), 1),
        "actual_ftm": 0.0 if actual is None else actual.get("ftm", 0.0),
        "sim_ftm": 0.0 if sim is None else round(float(sim.get("ftm", 0.0)), 1),
        "actual_oreb": 0.0 if actual is None else actual.get("oreb", 0.0),
        "sim_oreb": 0.0 if sim is None else round(float(sim.get("oreb", 0.0)), 1),
        "actual_dreb": 0.0 if actual is None else actual.get("dreb", 0.0),
        "sim_dreb": 0.0 if sim is None else round(float(sim.get("dreb", 0.0)), 1),
        "actual_stl": 0.0 if actual is None else actual.get("stl", 0.0),
        "sim_stl": 0.0 if sim is None else round(float(sim.get("stl", 0.0)), 1),
        "actual_blk": 0.0 if actual is None else actual.get("blk", 0.0),
        "sim_blk": 0.0 if sim is None else round(float(sim.get("blk", 0.0)), 1),
        "actual_pf": 0.0 if actual is None else actual.get("pf", 0.0),
        "sim_pf": 0.0 if sim is None else round(float(sim.get("pf", 0.0)), 1),
        "minute_component": round(ppm * (sim_min - actual_min), 1),
        "rate_component": round(actual_pts - ppm * actual_min, 1) if sim_min > 1 else round(actual_pts - sim_pts, 1),
        "matched": sim is not None and actual is not None,
    }


def coverage(rows: list[dict], actual_key: str, low_key: str, high_key: str) -> float:
    usable = [row for row in rows if row.get(low_key) is not None]
    if not usable:
        return 0.0
    inside = sum(1 for row in usable if row[low_key] <= row[actual_key] <= row[high_key])
    return inside / len(usable)


def main(out: str | None = None) -> None:
    warehouse = Warehouse()
    schedule = warehouse.read("schedule")
    finals = schedule[schedule["status"].astype(str).eq("Final")]
    games_out = []
    players_out = []
    plays_out = []
    spec_cache: dict[date, list] = {}
    with httpx.Client(timeout=120, follow_redirects=True, headers={"User-Agent": "nba-sim/0.1"}) as client:
        for on in DATES:
            if on not in spec_cache:
                spec_cache[on] = build_specs_for_date(on, PregameWarehouse(warehouse, on))
                print(f"specs {on} {len(spec_cache[on])}", flush=True)
            specs = spec_cache[on]
            for spec in specs:
                game_id = spec.game_id
                row = finals[finals["game_id"].astype(str).eq(game_id)]
                if row.empty:
                    continue
                payload = fetch_summary(client, game_id)
                actual_players, team_stats, ids = parse_box(payload)
                play_counts = parse_plays(payload, ids)
                print(f"sim {spec.away.abbr} at {spec.home.abbr}", flush=True)
                result = simulate_game(spec, n_sims=N_SIMS, seed=SEED)
                home_actual = team_actual(team_stats.get(spec.home.abbr, {}), actual_players, spec.home.abbr)
                away_actual = team_actual(team_stats.get(spec.away.abbr, {}), actual_players, spec.away.abbr)
                # Box points can miss a team-credited bucket. Header score is the game.
                header = ((payload.get("header") or {}).get("competitions") or [{}])[0]
                for competitor in header.get("competitors") or []:
                    abbr = normalize_team((competitor.get("team") or {}).get("abbreviation"))
                    score = _num(competitor.get("score"))
                    if abbr == spec.home.abbr:
                        home_actual["pts"] = score
                    elif abbr == spec.away.abbr:
                        away_actual["pts"] = score
                home_sim = float(result.home_points.mean())
                away_sim = float(result.away_points.mean())
                home_players = sim_players(spec.home_players, result.player_stats)
                away_players = sim_players(spec.away_players, result.player_stats)
                game = {
                    "date": on.isoformat(),
                    "game_id": game_id,
                    "away": spec.away.abbr.upper(),
                    "home": spec.home.abbr.upper(),
                    "sim_away": round(away_sim, 1),
                    "sim_home": round(home_sim, 1),
                    "away_p10": round(float(np.percentile(result.away_points, 10)), 1),
                    "away_p90": round(float(np.percentile(result.away_points, 90)), 1),
                    "home_p10": round(float(np.percentile(result.home_points, 10)), 1),
                    "home_p90": round(float(np.percentile(result.home_points, 90)), 1),
                    "actual_away": away_actual["pts"],
                    "actual_home": home_actual["pts"],
                    "home_win_p": round(float((result.home_points > result.away_points).mean()), 3),
                    "margin_sd": round(float(np.std(result.home_points - result.away_points)), 2),
                    "actual_home_win": int(home_actual["pts"] > away_actual["pts"]),
                    "sim_away_trips": round(float(result.away_possessions.mean()), 1),
                    "sim_home_trips": round(float(result.home_possessions.mean()), 1),
                    "sim_away_pf": round(_foul_mean(result.away_team_fouls), 1),
                    "sim_home_pf": round(_foul_mean(result.home_team_fouls), 1),
                    "sim_away_fga": round(_safe_sum(result, spec.away_players, "fga"), 1),
                    "sim_home_fga": round(_safe_sum(result, spec.home_players, "fga"), 1),
                    "sim_away_fta": round(_safe_sum(result, spec.away_players, "fta"), 1),
                    "sim_home_fta": round(_safe_sum(result, spec.home_players, "fta"), 1),
                    "sim_away_tov": round(_safe_sum(result, spec.away_players, "tov"), 1),
                    "sim_home_tov": round(_safe_sum(result, spec.home_players, "tov"), 1),
                }
                for side_name, spec_players, actual in (
                    ("away", spec.away_players, away_actual),
                    ("home", spec.home_players, home_actual),
                ):
                    for key in ("fgm", "ftm", "tpa", "oreb", "dreb", "reb", "ast", "stl", "blk", "pf", "tpm"):
                        game[f"sim_{side_name}_{key}"] = round(_safe_sum(result, spec_players, key), 1)
                        if key in actual:
                            game[f"actual_{side_name}_{key}"] = round(float(actual[key]), 1)
                    game[f"actual_{side_name}_paint"] = round(float(actual["paint"]), 1)
                    game[f"actual_{side_name}_fast_break"] = round(float(actual["fast_break"]), 1)
                    game[f"actual_{side_name}_poss"] = round(float(actual["poss"]), 1)
                    game[f"sim_{side_name}_poss"] = round(
                        possession_formula(
                            game[f"sim_{side_name}_fga"],
                            game[f"sim_{side_name}_oreb"],
                            game[f"sim_{side_name}_tov"],
                            game[f"sim_{side_name}_fta"],
                        ),
                        1,
                    )
                    game[f"actual_{side_name}_fga"] = round(float(actual["fga"]), 1)
                    game[f"actual_{side_name}_fta"] = round(float(actual["fta"]), 1)
                    game[f"actual_{side_name}_tov"] = round(float(actual["tov"]), 1)
                game["sim_home_second"] = round(float(result.second_chance_home.mean()), 2)
                game["sim_away_second"] = round(float(result.second_chance_away.mean()), 2)
                game["sim_flagrant"] = round(float(np.mean(result.flagrant_fouls)), 3)
                game["sim_ejections"] = round(float(np.mean(result.ejections)), 3)
                game["sim_intentional"] = round(float(np.mean(result.intentional_fouls)), 2)
                for side, abbr, actual, spec_players in (
                    ("away", spec.away.abbr.upper(), away_actual, spec.away_players),
                    ("home", spec.home.abbr.upper(), home_actual, spec.home_players),
                ):
                    mix = expected_mix(spec_players)
                    plays = play_counts.get(abbr.lower()) or play_counts.get(abbr) or _empty_plays()
                    plays_out.append(
                        {
                            "date": on.isoformat(),
                            "game": f"{spec.away.abbr.upper()} at {spec.home.abbr.upper()}",
                            "team": abbr,
                            "actual": actual,
                            "plays": plays,
                            "mix": {key: round(float(value), 4) for key, value in mix.items()},
                        }
                    )
                joined = join_players(actual_players, away_players, spec.away.abbr.upper())
                joined += join_players(actual_players, home_players, spec.home.abbr.upper())
                for player in joined:
                    player["date"] = on.isoformat()
                    player["game"] = f"{spec.away.abbr.upper()} at {spec.home.abbr.upper()}"
                players_out.extend(joined)
                game["spread_error"] = round((home_sim - away_sim) - (home_actual["pts"] - away_actual["pts"]), 1)
                game["total_error"] = round((home_sim + away_sim) - (home_actual["pts"] + away_actual["pts"]), 1)
                games_out.append(game)
                print(
                    f"  actual {away_actual['pts']:.0f}-{home_actual['pts']:.0f} sim {away_sim:.1f}-{home_sim:.1f}",
                    flush=True,
                )
    payload = {"n_sims": N_SIMS, "seed": SEED, "games": games_out, "players": players_out, "sides": plays_out}
    path = out or r"C:\Users\flynn\nba\data\opening_eval.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print("wrote", path, "games", len(games_out), "players", len(players_out), flush=True)


def _safe_sum(result, players, key: str) -> float:
    total = 0.0
    for player in players:
        box = result.player_stats.get(player.player_id)
        if box is None:
            continue
        total += float(box[key].mean())
    return total


if __name__ == "__main__":
    main()
