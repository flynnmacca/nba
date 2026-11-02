"""Grade form_fix gate / subsample files and print a short scorecard."""

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
        "tov": st["tov"]["bias"],
        "stl": st["stl"]["bias"],
        "fg_pct": rates["fg_pct"]["bias"],
        "three_pct": rates["three_pct"]["bias"],
        "ppp": rates["ppp"]["bias"],
        "winner": s["winner_hit"],
        "brier": s["brier"],
        "sim_margin_sd": s.get("sim_margin_sd"),
        "actual_margin_sd": s.get("actual_margin_sd"),
        "total_bias": s["total_bias"],
        "roles": roles,
        "points_split": g["points"],
        "months": g.get("months"),
    }


def main() -> None:
    pairs = [
        ("gate_84", ROOT / "season_eval_84_form_fix.json"),
        ("gate_stod", ROOT / "season_eval_stod_form_fix.json"),
        ("sub_84", ROOT / "season_eval_form_fix.json"),
        ("sub_stod", ROOT / "season_eval_2026_27_form_fix.json"),
    ]
    out: dict = {}
    for key, path in pairs:
        if not path.exists():
            continue
        graded = grade_file(path)
        out[key] = pack(graded)
        row = out[key]
        role32 = next((r["min_bias"] for r in row["roles"] if r["role"] == "32 and up"), None)
        print(
            key,
            "games",
            row["games"],
            "pts",
            round(row["pts"], 2),
            "fg%",
            round(row["fg_pct"], 3),
            "dreb",
            round(row["dreb"], 2),
            "fga",
            round(row["fga"], 2),
            "margin_sd",
            row["sim_margin_sd"],
            "32+",
            role32,
            "winner",
            round(row["winner"], 3),
            flush=True,
        )
    dest = ROOT / "grade_form_fix.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Full depth dump when both subsample files exist.
    depth = {}
    for season, path in (
        ("2025-26", ROOT / "season_eval_form_fix.json"),
        ("2026-27", ROOT / "season_eval_2026_27_form_fix.json"),
    ):
        if path.exists():
            depth[season] = grade_file(path)
    if depth:
        (ROOT / "grade_depth_form_fix.json").write_text(json.dumps(depth), encoding="utf-8")
        print("wrote grade_depth_form_fix.json", flush=True)
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
