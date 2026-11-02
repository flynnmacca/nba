"""Parquet warehouse for cached public feeds."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nba_sim.config import warehouse_dir


class Warehouse:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or warehouse_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, name: str) -> Path:
        return self.root / f"{name}.parquet"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def write(self, name: str, frame: pd.DataFrame) -> Path:
        path = self.path(name)
        frame.to_parquet(path, index=False)
        return path

    def read(self, name: str) -> pd.DataFrame | None:
        path = self.path(name)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def tables(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob("*.parquet"))
