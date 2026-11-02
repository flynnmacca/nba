"""Grade the residual-fix regrade and print a compact scorecard."""

from __future__ import annotations

import json
from pathlib import Path

from grade_breakdown import grade

NEW = Path(r"C:\Users\flynn\nba\data\season_eval_residual_fix.json")
STRUGGLE = Path(r"C:\Users\flynn\nba\data\season_eval_struggle_fix.json")
PACE = Path(r"C:\Users\flynn\nba\data\season_eval_pace_five.json")


def _brief(label: str, report: dict) -> dict:
    games = report["summary"]
    points = report["points"]
    stats = {row["stat"]: row for row in report["stats"]}
    player_block = report.get("players") or {}
    by_stat = player_block.get("by_stat") or {}
    rates = {row["stat"]: row for row in report.get("rates", [])}
    # Formula possessions vs trips live on the stats / rate tables; FGA/trip is derived.
    fga = stats.get("fga") or {}
    trips = stats.get("trips") or {}
    fga_per_trip = None
    if fga.get("sim") and trips.get("sim"):
        fga_per_trip = {
            "sim": round(float(fga["sim"]) / float(trips["sim"]), 3),
            "actual": round(float(fga["actual"]) / float(trips["actual"]), 3)
            if trips.get("actual")
            else None,
        }
    out = {
        "label": label,
        "n_games": games.get("games"),
        "spread_bias": games.get("spread_bias"),
        "spread_mae": games.get("spread_mae"),
        "total_bias": games.get("total_bias"),
        "winner_hit": games.get("winner_hit"),
        "brier": games.get("brier"),
        "sim_margin_sd": games.get("sim_margin_sd"),
        "actual_margin_sd": games.get("actual_margin_sd"),
        "mean_margin_sd": games.get("mean_margin_sd"),
        "pace_points": points.get("pace_points"),
        "efficiency_points": points.get("efficiency_points"),
        "pts": stats.get("pts"),
        "poss": stats.get("poss"),
        "trips": trips or None,
        "fga": fga or None,
        "fta": stats.get("fta"),
        "blk": stats.get("blk"),
        "pf": stats.get("pf"),
        "oreb": stats.get("oreb"),
        "tov": stats.get("tov"),
        "fga_per_trip": fga_per_trip,
        "fg_pct": rates.get("fg_pct"),
        "ppp": rates.get("ppp"),
        "players_min": by_stat.get("min"),
        "players_pts": by_stat.get("pts"),
        "roles": player_block.get("roles"),
        "where_points": player_block.get("where_points"),
    }
    return out


def main() -> None:
    fresh = grade(NEW)
    Path(r"C:\Users\flynn\nba\data\grade_residual_fix.json").write_text(
        json.dumps(fresh, indent=2), encoding="utf-8"
    )
    print("RESIDUAL")
    print(json.dumps(_brief("residual_fix", fresh), indent=2, default=str))
    if STRUGGLE.exists():
        prior = grade(STRUGGLE)
        print("STRUGGLE")
        print(json.dumps(_brief("struggle_fix", prior), indent=2, default=str))
    if PACE.exists():
        baseline = grade(PACE)
        print("PACE_FIVE")
        print(json.dumps(_brief("pace_five", baseline), indent=2, default=str))


if __name__ == "__main__":
    main()
