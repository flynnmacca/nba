"""Grade the star-fix regrade and print a compact scorecard."""

from __future__ import annotations

import json
from pathlib import Path

from grade_breakdown import grade

NEW = Path(r"C:\Users\flynn\nba\data\season_eval_star_fix.json")
RESIDUAL = Path(r"C:\Users\flynn\nba\data\season_eval_residual_fix.json")
STRUGGLE = Path(r"C:\Users\flynn\nba\data\season_eval_struggle_fix.json")


def _brief(label: str, report: dict) -> dict:
    games = report["summary"]
    points = report["points"]
    stats = {row["stat"]: row for row in report["stats"]}
    player_block = report.get("players") or {}
    by_stat = player_block.get("by_stat") or {}
    rates = {row["stat"]: row for row in report.get("rates", [])}
    return {
        "label": label,
        "n_games": games.get("games"),
        "total_bias": games.get("total_bias"),
        "spread_bias": games.get("spread_bias"),
        "spread_mae": games.get("spread_mae"),
        "winner_hit": games.get("winner_hit"),
        "brier": games.get("brier"),
        "sim_margin_sd": games.get("sim_margin_sd"),
        "actual_margin_sd": games.get("actual_margin_sd"),
        "pace_points": points.get("pace_points"),
        "efficiency_points": points.get("efficiency_points"),
        "pts": stats.get("pts"),
        "fga": stats.get("fga"),
        "fta": stats.get("fta"),
        "blk": stats.get("blk"),
        "poss": stats.get("poss"),
        "fg_pct": rates.get("fg_pct"),
        "ppp": rates.get("ppp"),
        "players_min": by_stat.get("min"),
        "players_pts": by_stat.get("pts"),
        "where_points": player_block.get("where_points"),
    }


def main() -> None:
    fresh = grade(NEW)
    Path(r"C:\Users\flynn\nba\data\grade_star_fix.json").write_text(
        json.dumps(fresh, indent=2), encoding="utf-8"
    )
    print("STAR")
    print(json.dumps(_brief("star_fix", fresh), indent=2, default=str))
    for path, label in ((RESIDUAL, "residual_fix"), (STRUGGLE, "struggle_fix")):
        if path.exists():
            print(label.upper())
            print(json.dumps(_brief(label, grade(path)), indent=2, default=str))


if __name__ == "__main__":
    main()
