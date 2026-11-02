"""Turn a finished eval file into the tables the regrade canvas reads."""

from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(r"C:\Users\flynn\nba\data")

STAT_KEYS = (
    "pts",
    "poss",
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

PLAYER_KEYS = (
    "min",
    "pts",
    "reb",
    "oreb",
    "ast",
    "fga",
    "fta",
    "ftm",
    "tov",
    "tpm",
    "tpa",
    "stl",
    "blk",
    "pf",
)

PLAY_TYPES = (
    "Transition",
    "Isolation",
    "PRBallHandler",
    "PRRollman",
    "Postup",
    "Spotup",
    "Handoff",
    "Cut",
    "OffScreen",
    "OffRebound",
    "Misc",
)


def _r(value: float, places: int = 2) -> float:
    if value is None or isinstance(value, float) and math.isnan(value):
        return 0.0
    return round(float(value), places)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mae(values: list[float]) -> float:
    return _mean([abs(value) for value in values])


def _sides(games: list[dict]) -> list[dict]:
    rows = []
    for game in games:
        for side, opp in (("home", "away"), ("away", "home")):
            row = {"side": side, "opp": opp, "game": game}
            for key in STAT_KEYS:
                if key == "pts":
                    row["sim_pts"] = float(game[f"sim_{side}"])
                    row["actual_pts"] = float(game[f"actual_{side}"])
                    continue
                sim_key = f"sim_{side}_{key}"
                act_key = f"actual_{side}_{key}"
                if sim_key in game and act_key in game:
                    row[f"sim_{key}"] = float(game[sim_key])
                    row[f"actual_{key}"] = float(game[act_key])
            rows.append(row)
    return rows


def _stat_table(sides: list[dict]) -> list[dict]:
    table = []
    for key in STAT_KEYS:
        sim = [row[f"sim_{key}"] for row in sides if f"sim_{key}" in row]
        act = [row[f"actual_{key}"] for row in sides if f"actual_{key}" in row]
        if not sim or len(sim) != len(act):
            continue
        gap = [left - right for left, right in zip(sim, act)]
        table.append(
            {
                "stat": key,
                "sim": _r(_mean(sim), 2),
                "actual": _r(_mean(act), 2),
                "bias": _r(_mean(gap), 2),
                "mae": _r(_mae(gap), 2),
                "n": len(gap),
            }
        )
    return table


def _rate(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def _rates(sides: list[dict]) -> list[dict]:
    names = {
        "fg_pct": ("fgm", "fga"),
        "three_pct": ("tpm", "tpa"),
        "ft_pct": ("ftm", "fta"),
        "three_share": ("tpa", "fga"),
        "fta_per_fga": ("fta", "fga"),
        "ppp": ("pts", "poss"),
        "tov_per_poss": ("tov", "poss"),
        "oreb_per_miss": None,
        "ast_per_fgm": ("ast", "fgm"),
    }
    out = []
    for name, pair in names.items():
        sim_vals = []
        act_vals = []
        for row in sides:
            if name == "oreb_per_miss":
                game = row["game"]
                opp = row["opp"]
                sim_miss = row.get("sim_fga", 0) - row.get("sim_fgm", 0)
                act_miss = row.get("actual_fga", 0) - row.get("actual_fgm", 0)
                # A miss the other team rebounds, or this team does. Rate uses both glasses.
                sim_den = row.get("sim_oreb", 0) + float(game.get(f"sim_{opp}_dreb", 0) or 0)
                act_den = row.get("actual_oreb", 0) + float(game.get(f"actual_{opp}_dreb", 0) or 0)
                sim_rate = _rate(row.get("sim_oreb", 0), sim_den)
                act_rate = _rate(row.get("actual_oreb", 0), act_den)
            else:
                num, den = pair
                sim_rate = _rate(row.get(f"sim_{num}", 0), row.get(f"sim_{den}", 0))
                act_rate = _rate(row.get(f"actual_{num}", 0), row.get(f"actual_{den}", 0))
            if sim_rate is None or act_rate is None:
                continue
            sim_vals.append(sim_rate)
            act_vals.append(act_rate)
        if not sim_vals:
            continue
        gap = [left - right for left, right in zip(sim_vals, act_vals)]
        out.append(
            {
                "stat": name,
                "sim": _r(_mean(sim_vals), 3),
                "actual": _r(_mean(act_vals), 3),
                "bias": _r(_mean(gap), 3),
                "mae": _r(_mae(gap), 3),
                "n": len(gap),
            }
        )
    return out


def _split_points(sides: list[dict]) -> dict:
    pace = []
    eff = []
    volume = []
    accuracy = []
    three_extra = []
    free_throw = []
    box_gap = []
    for row in sides:
        poss_s, poss_a = row.get("sim_poss", 0), row.get("actual_poss", 0)
        pts_s, pts_a = row.get("sim_pts", 0), row.get("actual_pts", 0)
        if poss_a <= 0 or poss_s <= 0:
            continue
        actual_ppp = pts_a / poss_a
        pace.append((poss_s - poss_a) * actual_ppp)
        eff.append(pts_s - pts_a - pace[-1])
        fga_s, fga_a = row.get("sim_fga", 0), row.get("actual_fga", 0)
        fgm_s, fgm_a = row.get("sim_fgm", 0), row.get("actual_fgm", 0)
        if fga_a > 0 and fga_s > 0:
            fg_a = fgm_a / fga_a
            fg_s = fgm_s / fga_s
            volume.append((fga_s - fga_a) * fg_a * 2.0)
            accuracy.append(fga_a * (fg_s - fg_a) * 2.0)
        tpm_s, tpm_a = row.get("sim_tpm", 0), row.get("actual_tpm", 0)
        three_extra.append(tpm_s - tpm_a)
        ftm_s, ftm_a = row.get("sim_ftm", 0), row.get("actual_ftm", 0)
        free_throw.append(ftm_s - ftm_a)
        box = 2 * (fgm_s - fgm_a) + (tpm_s - tpm_a) + (ftm_s - ftm_a)
        box_gap.append((pts_s - pts_a) - box)
    return {
        "pace_points": _r(_mean(pace), 2),
        "efficiency_points": _r(_mean(eff), 2),
        "two_from_extra_fga": _r(_mean(volume), 2),
        "two_from_fg_pct": _r(_mean(accuracy), 2),
        "extra_made_threes": _r(_mean(three_extra), 2),
        "extra_made_free_throws": _r(_mean(free_throw), 2),
        "points_outside_the_box": _r(_mean(box_gap), 2),
        "sides": len(pace),
    }


def _games(games: list[dict]) -> tuple[dict, list[dict]]:
    spread = [float(game["spread_error"]) for game in games]
    total = [float(game["total_error"]) for game in games]
    home_p = [float(game["home_win_p"]) for game in games]
    home_y = [float(game["actual_home_win"]) for game in games]
    margin_sd = [float(game["margin_sd"]) for game in games]
    actual_margin = [float(game["actual_home"]) - float(game["actual_away"]) for game in games]
    sim_margin = [float(game["sim_home"]) - float(game["sim_away"]) for game in games]
    brier = _mean([(p - y) ** 2 for p, y in zip(home_p, home_y)])
    called = [1.0 if (p >= 0.5) == (y == 1.0) else 0.0 for p, y in zip(home_p, home_y)]
    bins = []
    edges = (0.0, 0.35, 0.45, 0.55, 0.65, 1.01)
    labels = ("under 35%", "35–45%", "45–55%", "55–65%", "65% and up")
    for index, label in enumerate(labels):
        chosen = [(p, y) for p, y in zip(home_p, home_y) if edges[index] <= p < edges[index + 1]]
        if not chosen:
            continue
        bins.append(
            {
                "bin": label,
                "games": len(chosen),
                "predicted": _r(_mean([p for p, _ in chosen]), 3),
                "actual": _r(_mean([y for _, y in chosen]), 3),
            }
        )
    close = [game for game in games if abs(float(game["actual_home"]) - float(game["actual_away"])) <= 5]
    blow = [game for game in games if abs(float(game["actual_home"]) - float(game["actual_away"])) >= 15]
    summary = {
        "games": len(games),
        "spread_bias": _r(_mean(spread), 2),
        "spread_mae": _r(_mae(spread), 2),
        "total_bias": _r(_mean(total), 2),
        "total_mae": _r(_mae(total), 2),
        "mean_home_win_p": _r(_mean(home_p), 3),
        "actual_home_win_rate": _r(_mean(home_y), 3),
        "brier": _r(brier, 3),
        "winner_hit": _r(_mean(called), 3),
        "mean_margin_sd": _r(_mean(margin_sd), 2),
        "actual_margin_sd": _r(math.sqrt(_mean([(m - _mean(actual_margin)) ** 2 for m in actual_margin])), 2),
        "sim_margin_sd": _r(math.sqrt(_mean([(m - _mean(sim_margin)) ** 2 for m in sim_margin])), 2),
        "close_games": len(close),
        "close_winner_hit": _r(_mean([1.0 if (float(g["home_win_p"]) >= 0.5) == (g["actual_home_win"] == 1) else 0.0 for g in close]), 3) if close else 0.0,
        "blowout_games": len(blow),
        "blowout_spread_bias": _r(_mean([float(g["spread_error"]) for g in blow]), 2) if blow else 0.0,
        "home_points_bias": _r(_mean([float(g["sim_home"]) - float(g["actual_home"]) for g in games]), 2),
        "away_points_bias": _r(_mean([float(g["sim_away"]) - float(g["actual_away"]) for g in games]), 2),
        "calibration": bins,
    }
    rows = []
    for game in sorted(games, key=lambda item: abs(float(item["total_error"])), reverse=True):
        rows.append(
            {
                "date": game["date"],
                "game": f"{game['away']} at {game['home']}",
                "sim": f"{game['sim_away']:.1f}–{game['sim_home']:.1f}",
                "actual": f"{game['actual_away']:.0f}–{game['actual_home']:.0f}",
                "spread_error": game["spread_error"],
                "total_error": game["total_error"],
                "home_win_p": game["home_win_p"],
                "home_won": game["actual_home_win"],
            }
        )
    return summary, rows


def _months(games: list[dict]) -> list[dict]:
    found: dict[str, list[dict]] = {}
    for game in games:
        found.setdefault(str(game.get("month") or game["date"][:7]), []).append(game)
    rows = []
    for month in sorted(found):
        group = found[month]
        rows.append(
            {
                "month": month,
                "games": len(group),
                "total_bias": _r(_mean([float(g["total_error"]) for g in group]), 1),
                "spread_mae": _r(_mae([float(g["spread_error"]) for g in group]), 1),
                "points_bias": _r(_mean([(float(g["sim_home"]) - float(g["actual_home"]) + float(g["sim_away"]) - float(g["actual_away"])) / 2 for g in group]), 1),
            }
        )
    return rows


def _teams(games: list[dict]) -> list[dict]:
    bag: dict[str, dict[str, list[float]]] = {}
    for game in games:
        for side in ("home", "away"):
            team = game[side]
            slot = bag.setdefault(team, {"pts": [], "oreb": [], "fta": [], "tov": [], "fga": [], "ast": []})
            slot["pts"].append(float(game[f"sim_{side}"]) - float(game[f"actual_{side}"]))
            for key in ("oreb", "fta", "tov", "fga", "ast"):
                sim_key, act_key = f"sim_{side}_{key}", f"actual_{side}_{key}"
                if sim_key in game and act_key in game:
                    slot[key].append(float(game[sim_key]) - float(game[act_key]))
    rows = []
    for team, slot in bag.items():
        rows.append(
            {
                "team": team.upper(),
                "games": len(slot["pts"]),
                "pts": _r(_mean(slot["pts"]), 1),
                "fga": _r(_mean(slot["fga"]), 1),
                "fta": _r(_mean(slot["fta"]), 1),
                "oreb": _r(_mean(slot["oreb"]), 1),
                "ast": _r(_mean(slot["ast"]), 1),
                "tov": _r(_mean(slot["tov"]), 1),
            }
        )
    rows.sort(key=lambda row: row["pts"], reverse=True)
    return rows


def _players(players: list[dict]) -> dict:
    matched = [row for row in players if row.get("matched")]
    played = [row for row in matched if float(row.get("actual_min") or 0) >= 8 or float(row.get("sim_min") or 0) >= 8]

    def pack(rows: list[dict], key: str) -> dict:
        gap = [float(row.get(f"sim_{key}") or 0) - float(row.get(f"actual_{key}") or 0) for row in rows]
        return {"bias": _r(_mean(gap), 2), "mae": _r(_mae(gap), 2), "n": len(gap)}

    roles = []
    edges = ((0, 12, "under 12 min"), (12, 24, "12–24 min"), (24, 32, "24–32 min"), (32, 60, "32 min and up"))
    for low, high, label in edges:
        group = [row for row in played if low <= float(row.get("sim_min") or 0) < high]
        if not group:
            continue
        roles.append({"role": label, "players": len(group), "minutes": pack(group, "min"), "points": pack(group, "pts"), "assists": pack(group, "ast"), "rebounds": pack(group, "reb"), "fga": pack(group, "fga")})

    def cover(rows: list[dict], actual: str, low: str, high: str) -> float:
        usable = [row for row in rows if row.get(low) is not None and row.get(high) is not None]
        if not usable:
            return 0.0
        inside = sum(1 for row in usable if float(row[low]) <= float(row[actual]) <= float(row[high]))
        return _r(inside / len(usable), 3)

    minute_part = [abs(float(row.get("minute_component") or 0)) for row in played if float(row.get("sim_min") or 0) > 1]
    rate_part = [abs(float(row.get("rate_component") or 0)) for row in played if float(row.get("sim_min") or 0) > 1]
    def _group(row: dict) -> str:
        actual = float(row.get("actual_min") or 0)
        sim = float(row.get("sim_min") or 0)
        if actual < 1 and sim >= 1:
            return "on the sheet, did not play"
        if sim < 1 and actual >= 8:
            return "played, off the sheet"
        if max(sim, actual) >= 24:
            return "24 min and up"
        if max(sim, actual) >= 12:
            return "12–24 min"
        return "under 12 min"

    grouped: dict[str, list[dict]] = {}
    for row in matched:
        grouped.setdefault(_group(row), []).append(row)
    team_games = len({(row.get("game"), str(row.get("team"))) for row in matched}) or 1
    where = []
    for name in ("24 min and up", "12–24 min", "under 12 min", "on the sheet, did not play", "played, off the sheet"):
        rows = grouped.get(name) or []
        if not rows:
            continue
        gap = sum(float(item.get("sim_pts") or 0) - float(item.get("actual_pts") or 0) for item in rows)
        where.append(
            {
                "group": name,
                "players": len(rows),
                "points_per_side": _r(gap / team_games, 2),
                "minute_bias": pack(rows, "min")["bias"],
                "point_mae": pack(rows, "pts")["mae"],
            }
        )
    worst = sorted(played, key=lambda row: abs(float(row.get("sim_pts") or 0) - float(row.get("actual_pts") or 0)), reverse=True)[:18]
    worst_rows = []
    for row in worst:
        worst_rows.append(
            {
                "date": row.get("date"),
                "game": row.get("game"),
                "team": str(row.get("team") or "").upper(),
                "name": row.get("name"),
                "sim_min": row.get("sim_min"),
                "actual_min": row.get("actual_min"),
                "sim_pts": row.get("sim_pts"),
                "actual_pts": row.get("actual_pts"),
                "sim_ast": row.get("sim_ast"),
                "actual_ast": row.get("actual_ast"),
                "sim_reb": row.get("sim_reb"),
                "actual_reb": row.get("actual_reb"),
                "sim_fga": row.get("sim_fga"),
                "actual_fga": row.get("actual_fga"),
                "minute_component": row.get("minute_component"),
                "rate_component": row.get("rate_component"),
            }
        )
    return {
        "rows": len(players),
        "matched": len(matched),
        "graded": len(played),
        "unmatched_actual_minutes": _r(sum(float(row.get("actual_min") or 0) for row in players if not row.get("matched")), 0),
        "coverage_minutes": cover(played, "actual_min", "min_p10", "min_p90"),
        "coverage_points": cover(played, "actual_pts", "pts_p10", "pts_p90"),
        "coverage_assists": cover(played, "actual_ast", "ast_p10", "ast_p90"),
        "minute_component_mae": _r(_mean(minute_part), 2),
        "rate_component_mae": _r(_mean(rate_part), 2),
        "by_stat": {key: pack(played, key) for key in PLAYER_KEYS},
        "roles": roles,
        "where_points": where,
        "worst": worst_rows,
    }


def _plays(sides_payload: list[dict], games: list[dict]) -> dict:
    if not sides_payload:
        return {}
    play_keys = ("fga", "fgm", "three", "rim", "mid", "other", "putback", "fta", "ftm", "tov", "oreb", "dreb", "shooting_foul", "personal_foul", "loose_ball", "offensive_foul")
    means = {key: _mean([float((row.get("plays") or {}).get(key) or 0) for row in sides_payload]) for key in play_keys}
    fga = means["fga"] or 1.0
    mix_keys = list(PLAY_TYPES) + ["three_rate", "ft_rate", "tov_rate"]
    mix = {key: _mean([float((row.get("mix") or {}).get(key) or 0) for row in sides_payload]) for key in mix_keys}
    # Box means from the same games, per side, so the play feed can be checked against the official line.
    box = {}
    for key in ("fga", "fta", "tov", "oreb", "dreb", "tpa", "pf"):
        vals = []
        for game in games:
            for side in ("home", "away"):
                name = f"actual_{side}_{key}"
                if name in game:
                    vals.append(float(game[name]))
        if vals:
            box[key] = _r(_mean(vals), 2)
    sim = {}
    for key in ("fga", "fta", "tov", "oreb", "dreb", "tpa", "pf", "tpm", "trips"):
        vals = []
        for game in games:
            for side in ("home", "away"):
                name = f"sim_{side}_{key}"
                if name in game:
                    vals.append(float(game[name]))
        if vals:
            sim[key] = _r(_mean(vals), 2)
    second = []
    flagrant = []
    intentional = []
    for game in games:
        if "sim_home_second" in game:
            second.append(float(game["sim_home_second"]))
            second.append(float(game["sim_away_second"]))
        if "sim_flagrant" in game:
            flagrant.append(float(game["sim_flagrant"]))
        if "sim_intentional" in game:
            intentional.append(float(game["sim_intentional"]))
    return {
        "sides": len(sides_payload),
        "play_means": {key: _r(value, 2) for key, value in means.items()},
        "shot_share": {
            "rim": _r(means["rim"] / fga, 3),
            "mid": _r(means["mid"] / fga, 3),
            "three": _r(means["three"] / fga, 3),
            "other": _r(means["other"] / fga, 3),
            "putback_per_oreb": _r(means["putback"] / means["oreb"], 3) if means["oreb"] else 0.0,
        },
        "expected_mix": {key: _r(value, 3) for key, value in mix.items()},
        "box": box,
        "sim": sim,
        "actual_paint": _r(_mean([float(game[f"actual_{side}_paint"]) for game in games for side in ("home", "away") if f"actual_{side}_paint" in game]), 2),
        "actual_fast_break": _r(_mean([float(game[f"actual_{side}_fast_break"]) for game in games for side in ("home", "away") if f"actual_{side}_fast_break" in game]), 2),
        "sim_second_chance_points": _r(_mean(second), 2) if second else 0.0,
        "sim_flagrant_per_game": _r(_mean(flagrant), 3) if flagrant else 0.0,
        "sim_intentional_per_game": _r(_mean(intentional), 2) if intentional else 0.0,
    }


def grade(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    games = payload["games"]
    sides = _sides(games)
    summary, game_rows = _games(games)
    return {
        "n_sims": payload.get("n_sims"),
        "seed": payload.get("seed"),
        "summary": summary,
        "points": _split_points(sides),
        "stats": _stat_table(sides),
        "rates": _rates(sides),
        "months": _months(games),
        "teams": _teams(games),
        "players": _players(payload.get("players") or []),
        "plays": _plays(payload.get("sides") or [], games),
        "games": game_rows,
    }


def main() -> None:
    out = {}
    for name in ("season", "opening"):
        path = ROOT / f"{name}_eval.json"
        if not path.exists():
            continue
        out[name] = grade(path)
        print(name, "games", out[name]["summary"]["games"], "total_bias", out[name]["summary"]["total_bias"], flush=True)
    dest = ROOT / "grade_breakdown.json"
    dest.write_text(json.dumps(out), encoding="utf-8")
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
