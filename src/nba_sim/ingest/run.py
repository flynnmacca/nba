"""Pull every public feed the slate and the backtest can see."""

from __future__ import annotations

import logging

from nba_sim.config import SETTINGS
from nba_sim.ingest.espn import ingest_espn
from nba_sim.ingest.markets import (
    combine_markets,
    ingest_kalshi,
    ingest_odds_api,
    ingest_polymarket,
    load_kaggle,
    load_manual_odds,
    try_download_kaggle,
)
from nba_sim.ingest.nba_stats import ingest_nba
from nba_sim.warehouse import Warehouse

logger = logging.getLogger(__name__)


def ingest_all(
    seasons: list[str] | None = None,
    refresh: bool = False,
    skip_nba: bool = False,
    matchup_games: int = 30,
) -> dict:
    warehouse = Warehouse()
    failures: list[str] = []
    if not skip_nba:
        failures.extend(
            ingest_nba(
                warehouse,
                seasons=seasons or list(SETTINGS.seasons),
                refresh=refresh,
                matchup_games=matchup_games,
            )
        )
    failures.extend(ingest_espn(warehouse))
    for label, func in (
        ("polymarket", lambda: ingest_polymarket(warehouse)),
        ("kalshi", lambda: ingest_kalshi(warehouse)),
        ("manual", lambda: load_manual_odds(warehouse)),
    ):
        try:
            func()
        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)
            failures.append(f"{label}: {exc}")
    try:
        odds = ingest_odds_api(warehouse)
        if odds is None:
            failures.append("odds_api: ODDS_API_KEY is not set")
    except Exception as exc:
        failures.append(f"odds_api: {exc}")
    try:
        path = try_download_kaggle()
        if path is None:
            failures.append(
                "kaggle: place nba_2008-2026.csv in data/raw or set KAGGLE_USERNAME and KAGGLE_KEY"
            )
        else:
            load_kaggle(path, warehouse)
    except Exception as exc:
        failures.append(f"kaggle: {exc}")
    combine_markets(warehouse)
    return {"tables": warehouse.tables(), "failures": failures}
