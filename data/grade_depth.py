"""Every team metric and every player stat for the full-season eval files."""

from __future__ import annotations

import json
import math
from pathlib import Path

from grade_breakdown import (
    _games,
    _mean,
    _months,
    _plays,
    _r,
    _rates,
    _sides,
    _split_points,
    _stat_table,
)

ROOT = Path(r"C:\Users\flynn\nba\data")
FILES = {
    "2025-26": ROOT / "season_eval_2025_26_gaps_fix.json",
    "2026-27": ROOT / "season_eval_2026_27_gaps_fix.json",
}
PLAYER_STATS = (
    "min",
    "pts",
    "reb",
    "oreb",
    "dreb",
    "ast",
    "fga",
    "fgm",
    "fta",
    "ftm",
    "tpa",
    "tpm",
    "tov",
    "stl",
    "blk",
    "pf",
)
TEAM_STATS = (
    "pts",
    "poss",
    "trips",
    "fga",
    "fgm",
    "tpa",
    "tpm",
    "fta",
    "ftm",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "tov",
    "stl",
    "blk",
    "pf",
)


def _corr(left: list[float], right: list[float]) -> float:
    n = len(left)
    if n < 3:
        return 0.0
    ml, mr = _mean(left), _mean(right)
    num = sum((a - ml) * (b - mr) for a, b in zip(left, right))
    dl = math.sqrt(sum((a - ml) ** 2 for a in left))
    dr = math.sqrt(sum((b - mr) ** 2 for b in right))
    if dl <= 0 or dr <= 0:
        return 0.0
    return num / (dl * dr)


def _pack(sim: list[float], act: list[float]) -> dict:
    gap = [a - b for a, b in zip(sim, act)]
    return {
        "sim": _r(_mean(sim), 2),
        "actual": _r(_mean(act), 2),
        "bias": _r(_mean(gap), 2),
        "mae": _r(_mean([abs(value) for value in gap]), 2),
        "rmse": _r(math.sqrt(_mean([value * value for value in gap])), 2),
        "corr": _r(_corr(sim, act), 3),
        "n": len(gap),
    }


def _team_stats(games: list[dict]) -> list[dict]:
    rows = []
    for key in TEAM_STATS:
        sim, act = [], []
        for game in games:
            for side in ("home", "away"):
                if key == "pts":
                    sim.append(float(game[f"sim_{side}"]))
                    act.append(float(game[f"actual_{side}"]))
                    continue
                sk, ak = f"sim_{side}_{key}", f"actual_{side}_{key}"
                if sk in game and ak in game:
                    sim.append(float(game[sk]))
                    act.append(float(game[ak]))
        if sim and len(sim) == len(act):
            packed = _pack(sim, act)
            packed["stat"] = key
            rows.append(packed)
    return rows


def _rate_pair(games: list[dict], num: str, den: str) -> dict | None:
    sim, act = [], []
    for game in games:
        for side in ("home", "away"):
            if num == "pts":
                sn, an = float(game[f"sim_{side}"]), float(game[f"actual_{side}"])
            else:
                sn, an = float(game.get(f"sim_{side}_{num}", 0)), float(game.get(f"actual_{side}_{num}", 0))
            sd, ad = float(game.get(f"sim_{side}_{den}", 0)), float(game.get(f"actual_{side}_{den}", 0))
            if sd > 0 and ad > 0:
                sim.append(sn / sd)
                act.append(an / ad)
    if not sim:
        return None
    packed = _pack(sim, act)
    packed["stat"] = f"{num}/{den}"
    return packed


def _player_rows(players: list[dict]) -> list[dict]:
    matched = [row for row in players if row.get("matched")]
    return [row for row in matched if float(row.get("actual_min") or 0) >= 8 or float(row.get("sim_min") or 0) >= 8]


def _player_stats(rows: list[dict]) -> list[dict]:
    out = []
    for key in PLAYER_STATS:
        sim, act = [], []
        for row in rows:
            sk, ak = f"sim_{key}", f"actual_{key}"
            if sk not in row and ak not in row:
                continue
            sim.append(float(row.get(sk) or 0))
            act.append(float(row.get(ak) or 0))
        if sim:
            packed = _pack(sim, act)
            packed["stat"] = key
            out.append(packed)
    return out


def _player_rates(rows: list[dict]) -> list[dict]:
    pairs = (("fgm", "fga", "fg_pct"), ("tpm", "tpa", "three_pct"), ("ftm", "fta", "ft_pct"), ("tpa", "fga", "three_share"), ("fta", "fga", "fta_per_fga"), ("ast", "tov", "ast_per_tov"))
    out = []
    for num, den, name in pairs:
        sim, act = [], []
        for row in rows:
            sd, ad = float(row.get(f"sim_{den}") or 0), float(row.get(f"actual_{den}") or 0)
            if sd < 2 or ad < 2:
                continue
            sim.append(float(row.get(f"sim_{num}") or 0) / sd)
            act.append(float(row.get(f"actual_{num}") or 0) / ad)
        if sim:
            packed = _pack(sim, act)
            packed["stat"] = name
            out.append(packed)
    return out


def _roles(rows: list[dict]) -> list[dict]:
    edges = ((0, 12, "under 12"), (12, 24, "12–24"), (24, 32, "24–32"), (32, 80, "32 and up"))
    found = []
    for low, high, label in edges:
        group = [row for row in rows if low <= float(row.get("actual_min") or 0) < high]
        if not group:
            continue
        found.append(
            {
                "role": label,
                "players": len(group),
                "minutes": _one(group, "min"),
                "points": _one(group, "pts"),
                "rebounds": _one(group, "reb"),
                "assists": _one(group, "ast"),
                "fga": _one(group, "fga"),
                "turnovers": _one(group, "tov"),
            }
        )
    return found


def _one(rows: list[dict], key: str) -> dict:
    sim = [float(row.get(f"sim_{key}") or 0) for row in rows]
    act = [float(row.get(f"actual_{key}") or 0) for row in rows]
    return _pack(sim, act)


def _by_player(rows: list[dict], minimum: int) -> list[dict]:
    bag: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        bag.setdefault((str(row.get("team") or ""), str(row.get("name") or "")), []).append(row)
    ranked = []
    for (team, name), group in bag.items():
        if len(group) < minimum:
            continue
        points = _one(group, "pts")
        ranked.append(
            {
                "team": team.upper(),
                "name": name,
                "games": len(group),
                "minutes": _one(group, "min")["bias"],
                "pts": points["bias"],
                "pts_mae": points["mae"],
                "reb": _one(group, "reb")["bias"],
                "ast": _one(group, "ast")["bias"],
                "fga": _one(group, "fga")["bias"],
                "tov": _one(group, "tov")["bias"],
                "stl": _one(group, "stl")["bias"],
                "blk": _one(group, "blk")["bias"],
                "pf": _one(group, "pf")["bias"],
            }
        )
    ranked.sort(key=lambda row: abs(row["pts"]), reverse=True)
    return ranked


def _by_team(games: list[dict]) -> list[dict]:
    bag: dict[str, dict[str, list[float]]] = {}
    for game in games:
        for side in ("home", "away"):
            team = str(game[side]).upper()
            slot = bag.setdefault(team, {key: [] for key in ("pts", "poss", "trips", "fga", "fta", "tov", "reb", "ast", "stl", "blk", "pf", "fg_pct")})
            slot["pts"].append(float(game[f"sim_{side}"]) - float(game[f"actual_{side}"]))
            for key in ("poss", "trips", "fga", "fta", "tov", "reb", "ast", "stl", "blk", "pf"):
                sk, ak = f"sim_{side}_{key}", f"actual_{side}_{key}"
                if sk in game and ak in game:
                    slot[key].append(float(game[sk]) - float(game[ak]))
            sf, af = float(game.get(f"sim_{side}_fga", 0)), float(game.get(f"actual_{side}_fga", 0))
            if sf > 0 and af > 0:
                slot["fg_pct"].append(float(game.get(f"sim_{side}_fgm", 0)) / sf - float(game.get(f"actual_{side}_fgm", 0)) / af)
    rows = []
    for team, slot in bag.items():
        row = {"team": team, "games": len(slot["pts"])}
        for key, values in slot.items():
            row[key] = _r(_mean(values), 2) if values else 0.0
        rows.append(row)
    rows.sort(key=lambda row: row["pts"], reverse=True)
    return rows


def grade_file(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    games = payload["games"]
    sides = _sides(games)
    summary, _rows = _games(games)
    played = _player_rows(payload.get("players") or [])
    minimum = 8 if payload.get("season") == "2025-26" else 2
    return {
        "season": payload.get("season"),
        "n_sims": payload.get("n_sims"),
        "seed": payload.get("seed"),
        "finals_available": payload.get("finals_available"),
        "games": len(games),
        "player_rows": len(played),
        "summary": summary,
        "points": _split_points(sides),
        "stats": _team_stats(games),
        "rates": _rates(sides),
        "shot_rates": [row for row in (_rate_pair(games, "fgm", "fga"), _rate_pair(games, "tpm", "tpa"), _rate_pair(games, "ftm", "fta"), _rate_pair(games, "tpa", "fga"), _rate_pair(games, "fta", "fga"), _rate_pair(games, "pts", "poss"), _rate_pair(games, "pts", "trips"), _rate_pair(games, "fga", "trips")) if row],
        "months": _months(games),
        "teams": _by_team(games),
        "players": _player_stats(played),
        "player_rates": _player_rates(played),
        "roles": _roles(played),
        "player_board": _by_player(played, minimum),
        "plays": _plays(payload.get("sides") or [], games),
        "box_table": _stat_table(sides),
    }


def main() -> None:
    out = {}
    for season, path in FILES.items():
        if not path.exists():
            print("missing", path, flush=True)
            continue
        out[season] = grade_file(path)
        print(season, "games", out[season]["games"], "players", out[season]["player_rows"], "total_bias", out[season]["summary"]["total_bias"], flush=True)
    dest = ROOT / "grade_depth_gaps_fix.json"
    dest.write_text(json.dumps(out), encoding="utf-8")
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
