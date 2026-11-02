"""How deep each coach plays, and the minute curve that follows from it.

A season of game logs replaces this. Until those minutes exist, the depth
chart is shaped like the coach's regular-season habit. Tom Thibodeau's teams
were the short end: about eight players, starters in the high 30s. Steve Kerr
is the long end: about eleven, starters closer to 31. The 2026-27 staff sit
on that scale. A first-year coach starts at a 10-man league shape.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoachRotation:
    name: str
    players: int
    starter_minutes: float
    # How long he leaves a player in foul trouble. The league sit is two fouls
    # in the first quarter, three before half, and four in the third (the Q+1
    # line from play-by-play). A deeper bench has a better replacement, so
    # those staffs sit sooner. A short staff plays a starter through it.
    foul_patience: float = 0.45


# Not a 2026-27 head coach. Kept so the short end of the scale has his name on it.
THIBODEAU = CoachRotation("Tom Thibodeau", 8, 37.5, 0.90)

LEAGUE = CoachRotation("League rotation", 10, 32.5)

# players: how many men regularly play. starter_minutes: average of the top five.
_STAFF = {
    "atl": CoachRotation("Quin Snyder", 10, 33.0),
    "bos": CoachRotation("Joe Mazzulla", 9, 34.0),
    "bkn": CoachRotation("Jordi Fernandez", 11, 30.0, 0.26),
    "cha": CoachRotation("Charles Lee", 11, 31.0, 0.32),
    "chi": CoachRotation("Tiago Splitter", 10, 32.5),
    "cle": CoachRotation("Kenny Atkinson", 10, 33.0),
    "dal": CoachRotation("Dusty May", 10, 32.5),
    "den": CoachRotation("David Adelman", 9, 34.5, 0.55),
    "det": CoachRotation("J.B. Bickerstaff", 9, 34.0, 0.58),
    "gsw": CoachRotation("Steve Kerr", 11, 31.0, 0.22),
    "hou": CoachRotation("Ime Udoka", 9, 35.0, 0.74),
    "ind": CoachRotation("Rick Carlisle", 9, 33.5, 0.36),
    "lac": CoachRotation("Tyronn Lue", 9, 33.0),
    "lal": CoachRotation("JJ Redick", 10, 32.5),
    "mem": CoachRotation("Tuomas Iisalo", 10, 32.0),
    "mia": CoachRotation("Erik Spoelstra", 10, 33.0, 0.34),
    "mil": CoachRotation("Taylor Jenkins", 10, 33.5),
    "min": CoachRotation("Chris Finch", 9, 35.0, 0.60),
    "nop": CoachRotation("Jamahl Mosley", 10, 33.0),
    "nyk": CoachRotation("Mike Brown", 9, 36.0, 0.72),
    "okc": CoachRotation("Mark Daigneault", 10, 33.0, 0.40),
    "orl": CoachRotation("Sean Sweeney", 10, 32.5),
    "phi": CoachRotation("Nick Nurse", 9, 34.5, 0.40),
    "phx": CoachRotation("Jordan Ott", 10, 32.0),
    "por": CoachRotation("Micah Nori", 10, 32.0),
    "sac": CoachRotation("Doug Christie", 10, 32.0),
    "sas": CoachRotation("Mitch Johnson", 11, 31.0, 0.30),
    "tor": CoachRotation("Darko Rajakovic", 11, 31.0, 0.30),
    "uta": CoachRotation("Will Hardy", 12, 29.0, 0.20),
    "was": CoachRotation("Brian Keefe", 11, 30.0, 0.24),
}

# Spot minutes for the men just past the rotation. A short staff leaves almost
# none. A 10-man staff leaves a few. A staff that already plays 12 has used the game.
_SPOT_BUDGET = {8: 3.0, 9: 5.0, 10: 8.0, 11: 5.0, 12: 3.0}


def coach_for(abbr: str) -> CoachRotation:
    return _STAFF.get((abbr or "").lower(), LEAGUE)


def depth_minutes(coach: CoachRotation, roster_size: int) -> list[float]:
    """Minutes for each depth-chart rank.

    A 15-man roster sums to 240. A shorter list is the top of that same curve,
    so the first name is still the starter minute and the second name is still
    the second minute. The first ``coach.players`` ranks are the rotation.
    The next ranks are spot minutes. The end of the roster is zero.
    """
    asked = max(int(roster_size), 1)
    count = max(asked, 15)
    regulars = min(max(int(coach.players), 5), 12, count)
    spot_budget = 0.0 if count <= regulars else _SPOT_BUDGET.get(regulars, 6.0)
    core = 240.0 - spot_budget
    bench_count = regulars - 5
    starter_cap = core - 8.0 * bench_count
    starter_sum = min(float(coach.starter_minutes) * 5.0, starter_cap)
    starter_sum = max(starter_sum, min(core * 0.55, starter_cap))
    bench_sum = core - starter_sum
    starter_weights = [1.08, 1.04, 1.00, 0.96, 0.92]
    minutes = [_portion(starter_weights, starter_sum, index) for index in range(5)]
    bench_weights = [0.82 ** index for index in range(bench_count)]
    minutes.extend(_portion(bench_weights, bench_sum, index) for index in range(bench_count))
    spot_count = count - regulars
    spot_weights = [0.5 ** index for index in range(spot_count)]
    spots = [_portion(spot_weights, spot_budget, index) for index in range(spot_count)]
    minutes.extend(value if value >= 0.4 else 0.0 for value in spots)
    minutes[0] += 240.0 - sum(minutes)
    return minutes[:asked]


def finish_window(coach: CoachRotation) -> tuple[float, float]:
    """How late the reserves enter, and how large the lead has to be.

    A short staff keeps his starters through most of the fourth and only
    empties the bench when the game is decided. A long staff hooks them
    earlier. The two numbers come from this coach's rotation, not from a
    league clock.
    """
    players = min(max(int(coach.players), 8), 12)
    starter = float(coach.starter_minutes)
    seconds = 240.0 + (players - 8) * 60.0
    margin = 14.0 + (starter - 29.0) * 0.9
    return seconds, float(min(max(margin, 12.0), 26.0))


def _portion(weights: list[float], total: float, index: int) -> float:
    if not weights or total <= 0:
        return 0.0
    return total * weights[index] / sum(weights)
