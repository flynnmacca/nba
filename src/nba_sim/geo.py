"""Arena geography, travel distance, and tip times."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from nba_sim.teams import normalize_team


@dataclass(frozen=True)
class Arena:
    team_abbr: str
    name: str
    lat: float
    lon: float
    altitude_ft: float
    tz_standard: float
    tz_dst: float

    def offset(self, on: date) -> float:
        return self.tz_dst if us_dst(on) else self.tz_standard


def us_dst(on: date) -> bool:
    """US daylight saving time, second Sunday in March through first Sunday in November."""
    start = _nth_sunday(on.year, 3, 2)
    end = _nth_sunday(on.year, 11, 1)
    return start <= on < end


def _nth_sunday(year: int, month: int, n: int) -> date:
    first = date(year, month, 1)
    # Monday is 0. Sunday is 6.
    offset = (6 - first.weekday()) % 7
    day = 1 + offset + 7 * (n - 1)
    return date(year, month, day)


def miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


_US_ZONE_NAMES = {
    -5: ("EST", "EDT"),
    -6: ("CST", "CDT"),
    -7: ("MST", "MDT"),
    -8: ("PST", "PDT"),
}


def melbourne_today(now: datetime | None = None) -> date:
    """Calendar date in Australia/Melbourne."""
    moment = datetime.now(timezone.utc) if now is None else _as_utc(now)
    provisional = moment + timedelta(hours=melbourne_offset(moment.date()))
    hours = melbourne_offset(provisional.date())
    return (moment + timedelta(hours=hours)).date()


def melbourne_tip_date(start_utc: str) -> date:
    """The Melbourne calendar date of a UTC tip."""
    return melbourne_today(_parse_utc(start_utc))


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def melbourne_offset(on: date) -> int:
    """Australia/Melbourne. AEDT is UTC+11 from the first Sunday in October to the first Sunday in April."""
    dst_ends = _nth_sunday(on.year, 4, 1)
    dst_starts = _nth_sunday(on.year, 10, 1)
    if dst_ends <= on < dst_starts:
        return 10
    return 11


def local_clock(start_utc: str, home_abbr: str) -> tuple[str, str, str]:
    """Venue-local date (YYYY-MM-DD), local tip label, and Melbourne tip label."""
    start = _parse_utc(start_utc)
    arenas = load_arenas()
    abbr = normalize_team(home_abbr) or str(home_abbr).lower()
    arena = arenas.get(abbr)
    offset = -5.0 if arena is None else arena.offset(start.date())
    local = start + timedelta(hours=offset)
    offset = -5.0 if arena is None else arena.offset(local.date())
    local = start + timedelta(hours=offset)
    melbourne = start + timedelta(hours=melbourne_offset(start.date()))
    melbourne_offset_hours = melbourne_offset(melbourne.date())
    melbourne = start + timedelta(hours=melbourne_offset_hours)
    zone = _us_zone_name(abbr, arena, offset)
    melb_zone = "AEDT" if melbourne_offset_hours == 11 else "AEST"
    return local.date().isoformat(), _format_clock(local, zone), _format_clock(melbourne, melb_zone)


def _parse_utc(value: str) -> datetime:
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _us_zone_name(abbr: str, arena: Arena | None, offset: float) -> str:
    if abbr == "phx":
        return "MST"
    if arena is None:
        return "ET"
    names = _US_ZONE_NAMES.get(int(arena.tz_standard))
    if names is None:
        return "ET"
    standard, daylight = names
    if arena.tz_dst != arena.tz_standard and offset == arena.tz_dst:
        return daylight
    return standard


_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _format_clock(moment: datetime, zone: str) -> str:
    hour = moment.hour % 12 or 12
    period = "AM" if moment.hour < 12 else "PM"
    weekday = _WEEKDAYS[moment.weekday()]
    month = _MONTHS[moment.month - 1]
    return f"{weekday} {moment.day} {month}, {hour}:{moment.minute:02d} {period} {zone}"


def load_arenas(path: Path | None = None) -> dict[str, Arena]:
    csv_path = path or Path(__file__).resolve().parent / "data" / "arenas.csv"
    frame = pd.read_csv(csv_path)
    arenas: dict[str, Arena] = {}
    for row in frame.itertuples(index=False):
        abbr = normalize_team(row.team_abbr)
        if abbr is None:
            continue
        arenas[abbr] = Arena(
            team_abbr=abbr,
            name=str(row.arena),
            lat=float(row.lat),
            lon=float(row.lon),
            altitude_ft=float(row.altitude_ft),
            tz_standard=float(row.tz_standard),
            tz_dst=float(row.tz_dst),
        )
    return arenas
