"""Grade post_eval gate files and print scorecard vs logic_fix baselines."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from grade_depth import grade_file

ROOT = Path(r"C:\Users\flynn\nba\data")


def pack(g: dict, eval_path: Path | None = None) -> dict:
    st = {r["stat"]: r for r in g["stats"]}
    rates = {r["stat"]: r for r in g["rates"]}
    s = g["summary"]
    trips = None
    if eval_path and eval_path.exists():
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
        t = [float(x[f"sim_{side}_trips"]) for x in payload["games"] for side in ("home", "away")]
        fga = [float(x[f"sim_{side}_fga"]) for x in payload["games"] for side in ("home", "away")]
        trips = {
            "sim_trips": round(mean(t), 2),
            "fga_per_trip": round(mean(fga) / mean(t), 4),
        }
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
        "blk": st["blk"]["bias"],
        "fg_pct": rates["fg_pct"]["bias"],
        "ppp": rates["ppp"]["bias"],
        "winner": s["winner_hit"],
        "brier": s["brier"],
        "sim_margin_sd": s.get("sim_margin_sd"),
        "mean_margin_sd": s.get("mean_margin_sd"),
        "total_bias": s["total_bias"],
        "trips": trips,
        "roles": roles,
        "points_split": g["points"],
    }


def main() -> None:
    out = {
        "gate_84": pack(grade_file(ROOT / "season_eval_84_post_eval.json"), ROOT / "season_eval_84_post_eval.json"),
        "gate_stod": pack(grade_file(ROOT / "season_eval_stod_post_eval.json"), ROOT / "season_eval_stod_post_eval.json"),
        "base_84": pack(grade_file(ROOT / "season_eval_logic_fix.json"), ROOT / "season_eval_logic_fix.json"),
        "base_stod": pack(
            grade_file(ROOT / "season_eval_2026_27_logic_fix.json"),
            ROOT / "season_eval_2026_27_logic_fix.json",
        ),
    }
    dest = ROOT / "grade_phase_gate_post_eval.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    for key in ("gate_84", "base_84", "gate_stod", "base_stod"):
        row = out[key]
        print(
            key,
            "pts",
            row["pts"],
            "fg%",
            row["fg_pct"],
            "dreb",
            row["dreb"],
            "fga",
            row["fga"],
            "blk",
            row["blk"],
            "margin_sd",
            row["sim_margin_sd"],
            "winner",
            row["winner"],
            "fga/trip",
            (row["trips"] or {}).get("fga_per_trip"),
            "32+",
            next((r["min_bias"] for r in row["roles"] if "32" in r["role"]), None),
        )
    print("wrote", dest)


if __name__ == "__main__":
    main()
