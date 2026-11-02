"""Command line: ingest, simulate a date, walk-forward a short window."""

from __future__ import annotations

import argparse
import logging
from datetime import date

from nba_sim.backtest.walkforward import run_backtest
from nba_sim.config import SETTINGS
from nba_sim.features.prepare import build_specs_for_date, season_for
from nba_sim.ingest.nba_stats import refresh_slate_season
from nba_sim.ingest.run import ingest_all
from nba_sim.report import format_summary, summarize
from nba_sim.sim.demo import build_demo_slate
from nba_sim.sim.engine import simulate_game
from nba_sim.warehouse import Warehouse


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="nba-sim", description="NBA possession Monte Carlo")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Cache public NBA, injury, and market feeds")
    ingest.add_argument("--seasons", nargs="*", default=list(SETTINGS.seasons))
    ingest.add_argument("--refresh", action="store_true")
    ingest.add_argument("--skip-nba", action="store_true")
    ingest.add_argument("--matchup-games", type=int, default=30)

    update = sub.add_parser("update", help="Refresh this season's dashboards, synergy, and game logs")
    update.add_argument("--season", default=None)

    simulate = sub.add_parser("sim-date", help="Simulate one slate")
    simulate.add_argument("--date", default=date.today().isoformat())
    simulate.add_argument("--demo", action="store_true")
    simulate.add_argument("--sims", type=int, default=SETTINGS.n_sims)
    simulate.add_argument("--seed", type=int, default=7)

    backtest = sub.add_parser("backtest", help="Walk-forward team markets")
    backtest.add_argument("--last", type=int, default=8)
    backtest.add_argument("--sims", type=int, default=800)
    backtest.add_argument("--seed", type=int, default=1)

    args = parser.parse_args(argv)
    if args.command == "ingest":
        result = ingest_all(
            seasons=args.seasons,
            refresh=args.refresh,
            skip_nba=args.skip_nba,
            matchup_games=args.matchup_games,
        )
        print(f"tables: {', '.join(result['tables']) or 'none'}")
        if result["failures"]:
            print("skipped:")
            for failure in result["failures"]:
                print(f"  {failure}")
        return 0
    if args.command == "update":
        season = args.season or season_for(date.today())
        failures = refresh_slate_season(season=season)
        print(f"updated {season}")
        if failures:
            print("skipped:")
            for failure in failures:
                print(f"  {failure}")
        return 0
    if args.command == "sim-date":
        on = date.fromisoformat(args.date)
        specs = [] if args.demo else build_specs_for_date(on)
        source = "warehouse"
        if not specs and not args.demo and _schedule_loaded():
            print(f"No games on {on.isoformat()}.")
            return 0
        if not specs:
            specs = build_demo_slate(on)
            source = "demo"
            print("No warehouse slate for that date. Showing the demo slate.")
        for spec in specs:
            spec.source = source if spec.source == "demo" else spec.source
            result = simulate_game(spec, n_sims=args.sims, seed=args.seed)
            print(format_summary(summarize(spec, result)))
            print()
        return 0
    result = run_backtest(last_n=args.last, n_sims=args.sims, seed=args.seed)
    print(result["message"])
    props = result.get("player_props") or {}
    if props:
        print(props.get("message", ""))
    metrics = result.get("metrics") or {}
    for market, values in metrics.items():
        if market == "note":
            print(values)
            continue
        print(
            f"{market}: n={values['n']} log_loss={values['log_loss']} "
            f"brier={values['brier']} roi={values['roi']}"
        )
    return 0 if result["status"] in {"ok", "missing_lines"} else 1


def _schedule_loaded() -> bool:
    schedule = Warehouse().read("schedule")
    return schedule is not None and not schedule.empty and "date" in schedule.columns


if __name__ == "__main__":
    raise SystemExit(main())
