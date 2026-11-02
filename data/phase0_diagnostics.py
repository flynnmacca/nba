"""Read-only post-eval diagnostics: move factors, miss identity, STOD minute sums."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from statistics import mean

import numpy as np

from nba_sim.features.prepare import build_specs_for_date
from nba_sim.sim.adjust import offensive_residual
from nba_sim.sim.clock import _side
from nba_sim.sim.engine import _draw_talent
from nba_sim.warehouse import Warehouse

ROOT = Path(r"C:\Users\flynn\nba\data")
OUT = ROOT / "phase0_diagnostics.json"


def _miss_identity(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    misses = orebs = drebs = []
    miss_v, oreb_v, dreb_v = [], [], []
    for g in payload["games"]:
        for side in ("home", "away"):
            fga = float(g[f"sim_{side}_fga"])
            fgm = float(g[f"sim_{side}_fgm"])
            oreb = float(g[f"sim_{side}_oreb"])
            dreb = float(g[f"sim_{side}_dreb"])
            miss = fga - fgm
            miss_v.append(miss)
            oreb_v.append(oreb)
            dreb_v.append(dreb)
    m, o, d = mean(miss_v), mean(oreb_v), mean(dreb_v)
    return {
        "file": str(path.name),
        "misses": round(m, 2),
        "oreb": round(o, 2),
        "dreb": round(d, 2),
        "miss_minus_oreb_minus_dreb": round(m - o - d, 2),
        "note": "Near 0 means every non-OREB miss is a player DREB (no team-reb sink).",
    }


def _stod_rosters(on: date) -> list[dict]:
    specs = build_specs_for_date(on, Warehouse())
    rows = []
    for spec in specs:
        for label, players in (("home", spec.home_players), ("away", spec.away_players)):
            avail = [p for p in players if p.available and float(p.minutes_mean or 0) > 0]
            phantoms = [
                p
                for p in avail
                if int(getattr(p, "minutes_sample", 0) or 0) <= 0
                and (
                    getattr(p, "court", None) is None
                    or int(getattr(p.court, "sample", 0) or 0) <= 0
                )
            ]
            rows.append(
                {
                    "date": on.isoformat(),
                    "game": f"{spec.away.abbr}@{spec.home.abbr}",
                    "side": label,
                    "team": spec.home.abbr if label == "home" else spec.away.abbr,
                    "n_available": len(avail),
                    "sum_minutes_mean": round(sum(float(p.minutes_mean) for p in avail), 1),
                    "n_sample0": len(phantoms),
                    "sample0": [
                        {
                            "name": p.name,
                            "minutes_mean": round(float(p.minutes_mean), 1),
                            "source": getattr(getattr(p, "court", None), "source", "") or getattr(p, "minutes_source", ""),
                        }
                        for p in sorted(phantoms, key=lambda x: -float(x.minutes_mean))[:8]
                    ],
                }
            )
    return rows


def _move_factors(on: date) -> dict:
    specs = build_specs_for_date(on, Warehouse())
    if not specs:
        return {}
    spec = specs[0]
    rng = np.random.default_rng(7)
    n = 64
    home_ppp = _draw_talent(rng, spec.home_players, n)
    away_ppp = _draw_talent(rng, spec.away_players, n)
    home = _side(spec.home_players, home_ppp)
    away = _side(spec.away_players, away_ppp)
    home.night = np.clip(rng.normal(1.0, 0.09, size=n), 0.82, 1.20)
    away.night = np.clip(rng.normal(1.0, 0.09, size=n), 0.82, 1.20)
    home.court_on = True
    away.court_on = True
    # Pull a draw through public path: simulate one short segment via engine is heavy;
    # report night means and residual as the live move inputs.
    return {
        "game": f"{spec.away.abbr}@{spec.home.abbr}",
        "home_night_mean": round(float(home.night.mean()), 4),
        "away_night_mean": round(float(away.night.mean()), 4),
        "home_residual": offensive_residual(spec.home_players, spec.home),
        "away_residual": offensive_residual(spec.away_players, spec.away),
        "home_context_mult": float(getattr(spec.context, "home_ppp_mult", 1.0) or 1.0),
        "away_context_mult": float(getattr(spec.context, "away_ppp_mult", 1.0) or 1.0),
        "home_team_games": home.team_games,
        "away_team_games": away.team_games,
        "note": "Baseline inputs before / after post-eval patches.",
    }


def main() -> None:
    out = {
        "miss_identity_2025_26": _miss_identity(ROOT / "season_eval_2025_26_logic_fix.json"),
        "miss_identity_2026_27": _miss_identity(ROOT / "season_eval_2026_27_logic_fix.json"),
        "stod_rosters": _stod_rosters(date(2026, 10, 22)),
        "move_inputs": _move_factors(date(2026, 10, 22)),
    }
    bloated = [r for r in out["stod_rosters"] if r["sum_minutes_mean"] > 250 or r["n_sample0"] > 0]
    out["stod_summary"] = {
        "nights": len(out["stod_rosters"]),
        "mean_sum_minutes": round(mean(r["sum_minutes_mean"] for r in out["stod_rosters"]), 1),
        "nights_over_250_or_sample0": len(bloated),
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out["miss_identity_2025_26"], indent=2))
    print(json.dumps(out["stod_summary"], indent=2))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
