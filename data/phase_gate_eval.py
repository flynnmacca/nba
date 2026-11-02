"""Gate eval: re-sim the locked 84 + current STOD finals into tagged paths."""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from full_season_eval import N_SIMS, SEED, WORKERS, _dump, score_date

ROOT = Path(r"C:\Users\flynn\nba\data")
LOCKED_84 = ROOT / "season_eval_logic_fix.json"
STOD_SRC = ROOT / "season_eval_2026_27_logic_fix.json"


def _jobs_from_payload(path: Path, season: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_date: dict[str, list[dict]] = {}
    for game in payload["games"]:
        row = {
            "season": season,
            "game_id": str(game["game_id"]),
            "date": str(game["date"]),
            "home": game["home"],
            "away": game["away"],
            "actual_home": float(game["actual_home"]),
            "actual_away": float(game["actual_away"]),
            "month": str(game.get("month") or str(game["date"])[:7]),
            "status": "Final",
        }
        by_date.setdefault(row["date"], []).append(row)
    return [{"season": season, "rows": by_date[key]} for key in sorted(by_date)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--season", choices=("84", "stod", "both"), default="both")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"

    targets = {}
    jobs = []
    if args.season in ("84", "both"):
        dest = ROOT / f"season_eval_84_{args.tag}.json"
        if dest.exists():
            raise SystemExit(f"refuse existing {dest}")
        targets["2025-26"] = dest
        jobs.extend(_jobs_from_payload(LOCKED_84, "2025-26"))
    if args.season in ("stod", "both"):
        dest = ROOT / f"season_eval_stod_{args.tag}.json"
        if dest.exists():
            raise SystemExit(f"refuse existing {dest}")
        targets["2026-27"] = dest
        jobs.extend(_jobs_from_payload(STOD_SRC, "2026-27"))

    books = {
        season: {"games": [], "players": [], "plays": [], "available": 0}
        for season in targets
    }
    # Count available from jobs
    for job in jobs:
        books[job["season"]]["available"] += len(job["rows"])

    print(f"tag {args.tag} n_sims {N_SIMS} seed {SEED} dates {len(jobs)} workers {WORKERS}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(score_date, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            season = result["season"]
            book = books[season]
            book["games"].extend(result["games"])
            book["players"].extend(result["players"])
            book["plays"].extend(result["plays"])
            _dump(targets[season], season, book["available"], book["games"], book["players"], book["plays"])
            print(
                f"saved {season} {len(book['games'])}/{book['available']} missed {len(result['missed'])}",
                flush=True,
            )
    for season, dest in targets.items():
        book = books[season]
        print(f"wrote {dest} games {len(book['games'])} players {len(book['players'])}", flush=True)


if __name__ == "__main__":
    main()
