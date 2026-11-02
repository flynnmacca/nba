"""Runtime configuration."""

from pathlib import Path

from pydantic import BaseModel, Field


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    n_sims: int = 20_000
    kelly_fraction: float = 0.25
    kelly_cap: float = 0.02
    ci_z: float = 1.64
    seasons: list[str] = Field(
        default_factory=lambda: ["2024-25", "2025-26", "2026-27"]
    )
    request_pause_seconds: float = 0.65
    request_timeout: float = 60.0
    home_court_points: float = 2.2
    home_court_prior_games: float = 15.0
    b2b_points: float = -1.6
    three_in_four_points: float = -0.8
    travel_points_per_1000_miles: float = -0.4
    tz_points_per_hour: float = -0.3
    altitude_total_points_per_1000_ft: float = 0.4
    altitude_visitor_points_per_1000_ft: float = -0.35
    altitude_floor_ft: float = 2000.0
    points_per_ppp_unit: float = 112.0


def data_dir() -> Path:
    path = repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def warehouse_dir() -> Path:
    path = data_dir() / "warehouse"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir() -> Path:
    path = data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


SETTINGS = Settings()
