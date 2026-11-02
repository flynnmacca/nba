"""Shared basketball constants."""

# Points per 100 possessions for a league-average team. Team ratings are
# already written in this unit. The default on a team model is the same number.
LEAGUE_RATING = 114.0

PLAY_TYPES = [
    "Isolation",
    "Transition",
    "PRBallHandler",
    "PRRollman",  # Synergy spells the roll man "PRRollMan"; stored here as PRRollman.
    "Postup",
    "Spotup",
    "Handoff",
    "Cut",
    "OffScreen",
    "OffRebound",
    "Misc",
]

# Defensive Synergy queries for these types usually come back empty.
DEFENSE_PLAY_SKIP = {"Transition", "Cut", "OffRebound", "Misc"}

_PLAY_ALIASES = {"PRRollMan": "PRRollman"}


def canon_play(value: object) -> str:
    text = str(value or "")
    return _PLAY_ALIASES.get(text, text)

ZONES = ["ra", "paint", "mid", "corner3", "atb3"]
THREE_ZONES = {"corner3", "atb3"}

DEFAULT_ZONE_FREQ = {
    "ra": 0.28,
    "paint": 0.16,
    "mid": 0.12,
    "corner3": 0.10,
    "atb3": 0.34,
}
DEFAULT_ZONE_FG = {
    "ra": 0.66,
    "paint": 0.42,
    "mid": 0.41,
    "corner3": 0.38,
    "atb3": 0.35,
}

# Share of made field goals of this type that are assisted.
# These are the relative rates. A player's own assist percentage is the
# share of teammate makes he created, so the sim scales that share by
# this rate divided by the made-field-goal weighted mean below.
ASSIST_RATE = {
    "Isolation": 0.12,
    "Transition": 0.48,
    "PRBallHandler": 0.32,
    "PRRollman": 0.78,
    "Postup": 0.22,
    "Spotup": 0.86,
    "Handoff": 0.72,
    "Cut": 0.90,
    "OffScreen": 0.82,
    "OffRebound": 0.08,
    "Misc": 0.40,
}

# 2025-26 team synergy share of made field goals.
_ASSIST_FGM_SHARE = {
    "Isolation": 0.065,
    "Transition": 0.211,
    "PRBallHandler": 0.132,
    "PRRollman": 0.063,
    "Postup": 0.036,
    "Spotup": 0.236,
    "Handoff": 0.041,
    "Cut": 0.102,
    "OffScreen": 0.035,
    "OffRebound": 0.068,
    "Misc": 0.012,
}
ASSIST_RATE_BASE = sum(ASSIST_RATE[name] * share for name, share in _ASSIST_FGM_SHARE.items())

# Minute-weighted 2025-26 assist percentage (0.162) at 2.93 assists per game.
# A player with no assist percentage on the dashboard keeps this conversion
# of the per-game assists already stored on him.
AST_PCT_FROM_PER_GAME = 0.162 / 2.93

STINT_SHARES = (0.28, 0.18, 0.22, 0.17, 0.15)
BLOWOUT_MARGIN = 18.0
CLUTCH_MARGIN = 10.0

SHOT_ZONE_LABELS = {
    "Restricted Area": "ra",
    "In The Paint (Non-RA)": "paint",
    "Mid-Range": "mid",
    "Left Corner 3": "corner3",
    "Right Corner 3": "corner3",
    "Above the Break 3": "atb3",
}

TRACK_ZONE = {
    "Less Than 6Ft": "ra",
    "Less Than 10Ft": "paint",
    "2 Pointers": "mid",
    "Greater Than 15Ft": "atb3",
    "3 Pointers": "corner3",
    "Overall": "overall",
}
