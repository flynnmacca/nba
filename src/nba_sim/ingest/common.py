"""Shared helpers for cached public feeds."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx
import pandas as pd

from nba_sim.config import SETTINGS
from nba_sim.warehouse import Warehouse

logger = logging.getLogger(__name__)


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "|".join(str(part) for part in col if str(part) not in ("", "nan", "None"))
            for col in out.columns
        ]
    out.columns = [str(col) for col in out.columns]
    return out


def cached_frame(
    warehouse: Warehouse,
    name: str,
    builder: Callable[[], pd.DataFrame | None],
    refresh: bool,
    failures: list[str],
    pause: bool = True,
) -> pd.DataFrame | None:
    if not refresh and warehouse.exists(name):
        return warehouse.read(name)
    if pause:
        time.sleep(SETTINGS.request_pause_seconds)
    try:
        frame = builder()
    except Exception as exc:  # network and stats.nba.com both fail in the open
        logger.warning("%s failed: %s", name, exc)
        failures.append(f"{name}: {exc}")
        return None
    if frame is None:
        failures.append(f"{name}: empty")
        return None
    frame = flatten_columns(frame)
    warehouse.write(name, frame)
    return frame


def get_json(url: str, params: dict | None = None) -> object:
    with httpx.Client(timeout=SETTINGS.request_timeout, follow_redirects=True) as client:
        response = client.get(url, params=params, headers={"User-Agent": "nba-sim/0.1"})
        response.raise_for_status()
        return response.json()


def first_number(frame: pd.DataFrame, *names: str) -> pd.Series | None:
    lookup = {str(col).upper(): col for col in frame.columns}
    for name in names:
        col = lookup.get(name.upper())
        if col is not None:
            return pd.to_numeric(frame[col], errors="coerce")
    return None


def as_rate(series: pd.Series | None) -> pd.Series | None:
    if series is None:
        return None
    values = pd.to_numeric(series, errors="coerce")
    if values.dropna().median() > 1.5:
        return values / 100.0
    return values
