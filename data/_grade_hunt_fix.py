"""Grade hunt_fix gate files and print scorecard vs adapt_fix."""

from __future__ import annotations

import json
from pathlib import Path

from grade_depth import grade_file

ROOT = Path(r"C:\Users\flynn\nba\data")


def pack(g: dict) -> dict:
    st = {r["stat"]: r for r in g["stats"]}
    rates = {r["stat"]: r for r in g["rates"]}
    s = g["summary"]
    roles = [
        {
            "role": r["role"],
            "n": r["players"],
            "min_bias": r["minutes"]["bias"],
            "pts_bias": r["points"]["bias"],
        }
        for r in g["roles"]
    ]
    return {
        "games": g["games"],
        "pts": st["pts"]["bias"],
        "fga": st["fga"]["bias"],
        "dreb": st["dreb"]["bias"],
        "oreb": st["oreb"]["bias"],
        "ast": st["ast"]["bias"],
        "tov": st["tov"]["bias"],
        "stl": st["stl"]["bias"],
        "fg_pct": rates["fg_pct"]["bias"],
        "ast_rate": rates.get("ast_rate", {}).get("bias"),
        "three_pct": rates["three_pct"]["bias"],
        "ppp": rates["ppp"]["bias"],
        "winner": s["winner_hit"],
        "brier": s["brier"],
        "sim_margin_sd": s.get("sim_margin_sd"),
        "actual_margin_sd": s.get("actual_margin_sd"),
        "total_bias": s["total_bias"],
        "roles": roles,
        "points_split": g["points"],
    }


def _role(row: dict, name: str):
    return next((r["min_bias"] for r in row["roles"] if r["role"] == name), None)


def main() -> None:
    pairs = [
        ("hunt_84", ROOT / "season_eval_84_hunt_fix.json"),
        ("hunt_stod", ROOT / "season_eval_stod_hunt_fix.json"),
        ("adapt_84", ROOT / "season_eval_84_adapt_fix.json"),
        ("adapt_stod", ROOT / "season_eval_stod_adapt_fix.json"),
    ]
    out: dict = {}
    for key, path in pairs:
        if not path.exists():
            print("missing", path, flush=True)
            continue
        graded = grade_file(path)
        out[key] = pack(graded)
        row = out[key]
        print(
            key,
            "games",
            row["games"],
            "pts",
            round(row["pts"], 2),
            "fg%",
            round(row["fg_pct"], 3),
            "fga",
            round(row["fga"], 2),
            "dreb",
            round(row["dreb"], 2),
            "oreb",
            round(row["oreb"], 2),
            "ast",
            round(row["ast"], 2),
            "margin_sd",
            row["sim_margin_sd"],
            "32+",
            _role(row, "32 and up"),
            "under12",
            _role(row, "under 12"),
            "winner",
            round(row["winner"], 3),
            "brier",
            row["brier"],
            flush=True,
        )
    dest = ROOT / "grade_hunt_fix.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    depth = {}
    for season, path in (
        ("2025-26", ROOT / "season_eval_84_hunt_fix.json"),
        ("2026-27", ROOT / "season_eval_stod_hunt_fix.json"),
    ):
        if path.exists():
            depth[season] = grade_file(path)
    if depth:
        (ROOT / "grade_depth_hunt_fix.json").write_text(json.dumps(depth), encoding="utf-8")
        print("wrote grade_depth_hunt_fix.json", flush=True)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
