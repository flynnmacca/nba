"""Grade the logic-fix regrade and print a compact scorecard."""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path

from grade_breakdown import grade

NEW = Path(r"C:\Users\flynn\nba\data\season_eval_logic_fix.json")
STAR = Path(r"C:\Users\flynn\nba\data\season_eval_star_fix.json")


def _trips(path: Path) -> dict:
    games = json.loads(path.read_text(encoding="utf-8"))["games"]
    rows = []
    for g in games:
        for side in ("home", "away"):
            fga = float(g[f"sim_{side}_fga"])
            afga = float(g[f"actual_{side}_fga"])
            trips = float(g.get(f"sim_{side}_trips") or 0)
            atrips = float(g.get(f"actual_{side}_poss") or 0)
            pts = float(g[f"sim_{side}"])
            apts = float(g[f"actual_{side}"])
            blk = float(g.get(f"sim_{side}_blk") or 0)
            ablk = float(g.get(f"actual_{side}_blk") or 0)
            rows.append((pts, apts, fga, afga, trips, atrips, blk, ablk))

    def mean(i: int) -> float:
        return sum(r[i] for r in rows) / len(rows)

    margins = [float(g["sim_home"]) - float(g["sim_away"]) for g in games]
    return {
        "pts_bias": round(mean(0) - mean(1), 2),
        "fga_bias": round(mean(2) - mean(3), 2),
        "fga_per_trip_sim": round(mean(2) / mean(4), 3) if mean(4) else None,
        "fga_per_trip_act": round(mean(3) / mean(5), 3) if mean(5) else None,
        "trips_bias": round(mean(4) - mean(5), 2),
        "blk_bias": round(mean(6) - mean(7), 2),
        "sim_margin_sd": round(st.pstdev(margins), 2),
    }


def main() -> None:
    fresh = grade(NEW)
    Path(r"C:\Users\flynn\nba\data\grade_logic_fix.json").write_text(
        json.dumps(fresh, indent=2), encoding="utf-8"
    )
    print("LOGIC", json.dumps({**fresh["summary"], **_trips(NEW)}, indent=2, default=str))
    if STAR.exists():
        prior = grade(STAR)
        print("STAR", json.dumps({**prior["summary"], **_trips(STAR)}, indent=2, default=str))


if __name__ == "__main__":
    main()
