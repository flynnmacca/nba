"""Team identifiers, abbreviations, and market-name matching."""

from __future__ import annotations

import math
import re

# Canonical abbreviation -> stats.nba.com team id.
TEAM_ID_BY_ABBR = {
    "atl": 1610612737,
    "bos": 1610612738,
    "bkn": 1610612751,
    "cha": 1610612766,
    "chi": 1610612741,
    "cle": 1610612739,
    "dal": 1610612742,
    "den": 1610612743,
    "det": 1610612765,
    "gsw": 1610612744,
    "hou": 1610612745,
    "ind": 1610612754,
    "lac": 1610612746,
    "lal": 1610612747,
    "mem": 1610612763,
    "mia": 1610612748,
    "mil": 1610612749,
    "min": 1610612750,
    "nop": 1610612740,
    "nyk": 1610612752,
    "okc": 1610612760,
    "orl": 1610612753,
    "phi": 1610612755,
    "phx": 1610612756,
    "sac": 1610612758,
    "sas": 1610612759,
    "tor": 1610612761,
    "uta": 1610612762,
    "was": 1610612764,
}

ABBR_BY_TEAM_ID = {v: k for k, v in TEAM_ID_BY_ABBR.items()}

TEAM_NAMES = {
    "atl": "Atlanta Hawks",
    "bos": "Boston Celtics",
    "bkn": "Brooklyn Nets",
    "cha": "Charlotte Hornets",
    "chi": "Chicago Bulls",
    "cle": "Cleveland Cavaliers",
    "dal": "Dallas Mavericks",
    "den": "Denver Nuggets",
    "det": "Detroit Pistons",
    "gsw": "Golden State Warriors",
    "hou": "Houston Rockets",
    "ind": "Indiana Pacers",
    "lac": "LA Clippers",
    "lal": "Los Angeles Lakers",
    "mem": "Memphis Grizzlies",
    "mia": "Miami Heat",
    "mil": "Milwaukee Bucks",
    "min": "Minnesota Timberwolves",
    "nop": "New Orleans Pelicans",
    "nyk": "New York Knicks",
    "okc": "Oklahoma City Thunder",
    "orl": "Orlando Magic",
    "phi": "Philadelphia 76ers",
    "phx": "Phoenix Suns",
    "sac": "Sacramento Kings",
    "sas": "San Antonio Spurs",
    "tor": "Toronto Raptors",
    "uta": "Utah Jazz",
    "was": "Washington Wizards",
}

# Six-team NBA divisions keyed by canonical abbreviation.
DIVISION_BY_ABBR = {
    "atl": "southeast",
    "bos": "atlantic",
    "bkn": "atlantic",
    "cha": "southeast",
    "chi": "central",
    "cle": "central",
    "dal": "southwest",
    "den": "northwest",
    "det": "central",
    "gsw": "pacific",
    "hou": "southwest",
    "ind": "central",
    "lac": "pacific",
    "lal": "pacific",
    "mem": "southwest",
    "mia": "southeast",
    "mil": "central",
    "min": "northwest",
    "nop": "southwest",
    "nyk": "atlantic",
    "okc": "northwest",
    "orl": "southeast",
    "phi": "atlantic",
    "phx": "pacific",
    "sac": "pacific",
    "sas": "southwest",
    "tor": "atlantic",
    "uta": "northwest",
    "was": "southeast",
}


def same_division(home_abbr: str, away_abbr: str) -> bool:
    """True when both sides share a division."""
    left = normalize_team(home_abbr) or str(home_abbr or "").lower()
    right = normalize_team(away_abbr) or str(away_abbr or "").lower()
    home_div = DIVISION_BY_ABBR.get(left)
    away_div = DIVISION_BY_ABBR.get(right)
    return bool(home_div and away_div and home_div == away_div)


# Short codes used by public odds files and other sites.
_ALIASES = {
    "atl": "atl",
    "atlanta": "atl",
    "hawks": "atl",
    "bos": "bos",
    "boston": "bos",
    "celtics": "bos",
    "bkn": "bkn",
    "brk": "bkn",
    "brooklyn": "bkn",
    "nets": "bkn",
    "cha": "cha",
    "cho": "cha",
    "charlotte": "cha",
    "hornets": "cha",
    "chi": "chi",
    "chicago": "chi",
    "bulls": "chi",
    "cle": "cle",
    "cleveland": "cle",
    "cavaliers": "cle",
    "cavs": "cle",
    "dal": "dal",
    "dallas": "dal",
    "mavericks": "dal",
    "mavs": "dal",
    "den": "den",
    "denver": "den",
    "nuggets": "den",
    "det": "det",
    "detroit": "det",
    "pistons": "det",
    "gsw": "gsw",
    "gs": "gsw",
    "goldenstate": "gsw",
    "warriors": "gsw",
    "hou": "hou",
    "houston": "hou",
    "rockets": "hou",
    "ind": "ind",
    "indiana": "ind",
    "pacers": "ind",
    "lac": "lac",
    "clippers": "lac",
    "lal": "lal",
    "lakers": "lal",
    "mem": "mem",
    "memphis": "mem",
    "grizzlies": "mem",
    "mia": "mia",
    "miami": "mia",
    "heat": "mia",
    "mil": "mil",
    "milwaukee": "mil",
    "bucks": "mil",
    "min": "min",
    "minnesota": "min",
    "timberwolves": "min",
    "wolves": "min",
    "nop": "nop",
    "no": "nop",
    "nol": "nop",
    "neworleans": "nop",
    "pelicans": "nop",
    "nyk": "nyk",
    "ny": "nyk",
    "newyork": "nyk",
    "knicks": "nyk",
    "okc": "okc",
    "oklahomacity": "okc",
    "thunder": "okc",
    "orl": "orl",
    "orlando": "orl",
    "magic": "orl",
    "phi": "phi",
    "phila": "phi",
    "philadelphia": "phi",
    "sixers": "phi",
    "76ers": "phi",
    "phx": "phx",
    "pho": "phx",
    "phoenix": "phx",
    "suns": "phx",
    "sac": "sac",
    "sacramento": "sac",
    "kings": "sac",
    "sas": "sas",
    "sa": "sas",
    "sanantonio": "sas",
    "spurs": "sas",
    "tor": "tor",
    "toronto": "tor",
    "raptors": "tor",
    "uta": "uta",
    "utah": "uta",
    "jazz": "uta",
    "was": "was",
    "wsh": "was",
    "washington": "was",
    "wizards": "was",
}

# Longer phrases win so "los angeles lakers" does not become the Clippers.
_PHRASES = sorted(
    [
        ("los angeles lakers", "lal"),
        ("la lakers", "lal"),
        ("los angeles clippers", "lac"),
        ("la clippers", "lac"),
        ("golden state warriors", "gsw"),
        ("golden state", "gsw"),
        ("oklahoma city thunder", "okc"),
        ("oklahoma city", "okc"),
        ("new orleans pelicans", "nop"),
        ("new york knicks", "nyk"),
        ("san antonio spurs", "sas"),
        ("portland trail blazers", "por"),
        ("trail blazers", "por"),
        ("minnesota timberwolves", "min"),
        ("philadelphia 76ers", "phi"),
        ("atlanta hawks", "atl"),
        ("boston celtics", "bos"),
        ("brooklyn nets", "bkn"),
        ("charlotte hornets", "cha"),
        ("chicago bulls", "chi"),
        ("cleveland cavaliers", "cle"),
        ("dallas mavericks", "dal"),
        ("denver nuggets", "den"),
        ("detroit pistons", "det"),
        ("houston rockets", "hou"),
        ("indiana pacers", "ind"),
        ("memphis grizzlies", "mem"),
        ("miami heat", "mia"),
        ("milwaukee bucks", "mil"),
        ("orlando magic", "orl"),
        ("phoenix suns", "phx"),
        ("sacramento kings", "sac"),
        ("toronto raptors", "tor"),
        ("utah jazz", "uta"),
        ("washington wizards", "was"),
    ],
    key=lambda item: len(item[0]),
    reverse=True,
)

# Portland is in phrases; add the abbr alias here too.
_ALIASES["por"] = "por"
_ALIASES["portland"] = "por"
_ALIASES["blazers"] = "por"
TEAM_ID_BY_ABBR.setdefault("por", 1610612757)
ABBR_BY_TEAM_ID[1610612757] = "por"
TEAM_NAMES["por"] = "Portland Trail Blazers"


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_team(value: object) -> str | None:
    """Map a code, nickname, or sentence onto a canonical abbreviation."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    for phrase, abbr in _PHRASES:
        if phrase in lowered:
            return abbr
    key = _key(text)
    if key in _ALIASES:
        return _ALIASES[key]
    return None


def team_name(abbr: str) -> str:
    canonical = normalize_team(abbr) or abbr
    return TEAM_NAMES.get(canonical, canonical.upper())
