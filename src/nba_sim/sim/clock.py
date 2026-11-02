"""Quarter clock, shot clock, and a running foul count.

Pace is the average length of a possession inside 48 minutes. The game then
walks four 12-minute quarters, so a faster lineup simply uses less clock per
trip. Team fouls reset each quarter. The fifth team foul of a quarter (fourth
in overtime) is the bonus. A free throw is awarded only from a foul charged
to a defender, and that defender is more likely to be the one matched up with
the action.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nba_sim.constants import (
    ASSIST_RATE,
    ASSIST_RATE_BASE,
    AST_PCT_FROM_PER_GAME,
    CLUTCH_MARGIN,
    LEAGUE_RATING,
    PLAY_TYPES,
)
from nba_sim.features.coaches import coach_for, finish_window
from nba_sim.features.court import relative_usage, select_lineup
from nba_sim.features.play_priors import position_key
from nba_sim.features.shrink import FILE_PRIOR, LEAGUE_PRIOR, MATCHUP_PRIOR, shrink
from nba_sim.sim.adjust import (
    _POS_MATCH,
    _zone_ratio,
    compose_defense,
    defense_multiplier,
    lineup_oreb,
    make_points,
    matchup_multiplier,
    team_fg_level,
    team_ppp_level,
    zone_make_rates,
    oreb_probability,
    zone_matchup_mult,
)
from nba_sim.sim.specs import GameSpec, PlayerModel, TeamModel

# 2025-26 team synergy, possession-weighted. Defensive turnover rates sit
# below the offensive column, so a defense is judged against this base
# before it scales the ballhandler.
_LEAGUE_DEF_TOV = 0.093
QUARTER_SECONDS = 720.0
OT_SECONDS = 300.0
# Inside two scores the game is still being decided. A closer can finish the
# stint he is already in. Forty minutes is a heavy night, and the cap stops there.
_CLOSE_MARGIN = 8.0
_CLOSE_CEILING = 40.0
# The men a coach holds in a tight game and rests when the lead is safe.
_CLOSER_FLOOR = 22.0
# The trailing closers stay one more score past the bar that sits the leader.
_CHASE_MARGIN = 8.0
# Each minute still to play, beyond the finish window, asks for another
# three-quarters of a point before a closer can sit.
_LEAD_PER_MINUTE = 0.75
BONUS_FOULS = 5
OT_BONUS_FOULS = 4
SHOT_CLOCK = 24.0
FOUL_OUT = 6
# One short rest per quarter. Play-by-play puts the average cost of foul
# trouble near a minute and a half, so he is not parked for the rest of the half.
FOUL_REST = 120.0

SHOT = 0
COMMON = 1
TAKE = 2
OFFENSIVE = 3
INTENTIONAL = 4
FLAGRANT = 5

# Relative possession length. A transition trip is short. A post-up uses the clock.
_TEMPO_RAW = np.array(
    [1.20, 0.50, 1.08, 1.02, 1.25, 0.86, 0.92, 0.75, 1.00, 0.65, 1.00],
    dtype=float,
)
_MIX = np.array([0.08, 0.14, 0.18, 0.08, 0.06, 0.20, 0.06, 0.07, 0.05, 0.04, 0.04])
TEMPO = _TEMPO_RAW / float(np.sum(_TEMPO_RAW * _MIX))

# A league foul rate is 3.2 personal fouls per 36 minutes. At pace 100 he
# faces 75 defensive possessions in those minutes, so this is fouls per trip.
_FOULS_PER_TRIP = 3.2 / 75.0
# Defender position (G, F, C) weight for being charged with the foul.
MATCHUP = np.array(
    [
        [1.35, 1.05, 0.70],
        [0.90, 1.10, 1.35],
        [1.40, 1.05, 0.65],
        [0.55, 1.15, 1.70],
        [0.40, 1.10, 1.85],
        [1.25, 1.05, 0.70],
        [1.20, 1.10, 0.75],
        [0.70, 1.15, 1.45],
        [1.20, 1.10, 0.75],
        [0.50, 1.05, 1.70],
        [1.00, 1.00, 1.00],
    ],
    dtype=float,
)
PAINT = np.array([False, True, False, True, True, False, False, True, False, True, False])
PLAY_THREE = np.array([0.25, 0.32, 0.22, 0.05, 0.02, 0.78, 0.55, 0.08, 0.62, 0.02, 0.20])
# A flagrant is rare. Transition and the post are where the hard contact is.
_FLAG_PLAY = np.array([1.35, 1.80, 1.25, 1.10, 1.30, 0.70, 0.75, 1.15, 0.70, 0.55, 0.80])
# About one flagrant foul per eight games for an average defender's whistle share.
_FLAG_RATE = 0.006
_FLAG_EJECT = 0.12
POSITION_FOUL = {"G": 0.82, "F": 1.00, "C": 1.22}
TRANSITION = PLAY_TYPES.index("Transition")
HANDLER = PLAY_TYPES.index("PRBallHandler")
ROLL = PLAY_TYPES.index("PRRollman")
OFFREB = PLAY_TYPES.index("OffRebound")
# A live rebound is the rebounder's putback or a teammate's catch.
KICK = (PLAY_TYPES.index("Spotup"), PLAY_TYPES.index("Cut"), PLAY_TYPES.index("OffScreen"))


def bonus_on(period: int) -> int:
    """Team foul that starts the bonus. Overtime begins at period 5."""
    return OT_BONUS_FOULS if period >= 5 else BONUS_FOULS


def trailing_mask(
    period: int,
    seconds: np.ndarray,
    deficit: np.ndarray,
    fouls_before: np.ndarray,
    bonus_line: int,
) -> np.ndarray:
    """The defense fouls on purpose when it is trailing late and a stoppage helps.

    ``deficit`` is how many points the offense is ahead. The foul that reaches
    the bonus is included. Inside 12 seconds the trailer fouls even before the
    bonus, because the clock would otherwise run out.
    """
    window = (period >= 4) & (seconds <= 40.0) & (seconds > 0.3) & (deficit >= 1.0)
    cap = np.where(seconds <= 12.0, 8.0, np.where(seconds <= 24.0, 5.0, 3.0))
    enters_bonus = (fouls_before + 1) >= bonus_line
    must_stop = seconds <= 12.0
    return window & (deficit <= cap) & (enters_bonus | must_stop)


def free_throw_award(
    kind: np.ndarray,
    is_three: np.ndarray,
    made: np.ndarray,
    fouls_before: np.ndarray,
    bonus_line: int,
    seconds_left: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shots, whether it is a team foul, whether the offense keeps the ball, and whether the last shot is live.

    A shooting foul is two, or three on a three-point attempt. An and-1 is one.
    A common foul before the bonus is a side-out. From the bonus foul onward it
    is two shots. A take foul is one shot and the ball, or two and the ball once
    the bonus is on. A flagrant is two shots and the ball. An offensive foul
    draws no shots and is not a team foul.
    """
    kind = np.asarray(kind, dtype=int)
    is_three = np.asarray(is_three, dtype=bool)
    made = np.asarray(made, dtype=bool)
    fouls_before = np.asarray(fouls_before, dtype=float)
    seconds_left = np.asarray(seconds_left, dtype=float)
    in_bonus = (fouls_before + 1) >= bonus_line
    time_left = seconds_left > 0.05
    fta = np.zeros(kind.shape, dtype=int)
    keep = np.zeros(kind.shape, dtype=bool)
    live = np.zeros(kind.shape, dtype=bool)

    shot = kind == SHOT
    fta = np.where(shot & made, 1, fta)
    fta = np.where(shot & ~made & is_three, 3, fta)
    fta = np.where(shot & ~made & ~is_three, 2, fta)
    live = np.where(shot, time_left, live)

    common = kind == COMMON
    fta = np.where(common & in_bonus, 2, fta)
    keep = np.where(common & ~in_bonus, True, keep)
    live = np.where(common & in_bonus, time_left, live)

    # A take foul is one shot and the ball in every quarter, bonus included.
    # Two shots plus the ball would price a transition reach-in as a flagrant.
    take = kind == TAKE
    fta = np.where(take, 1, fta)
    keep = np.where(take, True, keep)

    intentional = kind == INTENTIONAL
    fta = np.where(intentional & in_bonus, 2, fta)
    fta = np.where(intentional & ~in_bonus, 1, fta)
    keep = np.where(intentional & ~in_bonus, True, keep)
    live = np.where(intentional & in_bonus, time_left, live)

    flagrant = kind == FLAGRANT
    fta = np.where(flagrant, 2, fta)
    keep = np.where(flagrant, True, keep)
    live = np.where(flagrant, False, live)

    offensive = kind == OFFENSIVE
    fta = np.where(offensive, 0, fta)
    keep = np.where(offensive, False, keep)
    live = np.where(offensive, False, live)
    team_foul = ~offensive
    return fta, team_foul, keep, live


def _sample_rows(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    totals = np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    cdf = np.cumsum(weights / totals, axis=1)
    draw = rng.random(weights.shape[0])
    return np.clip((draw[:, None] >= cdf).sum(axis=1), 0, weights.shape[1] - 1).astype(np.int64)


def _foul_rate(player: PlayerModel) -> float:
    if player.foul_rate > 0:
        return float(player.foul_rate)
    return POSITION_FOUL.get(position_key(player.position), 1.0)


def _player_ast_pct(player: PlayerModel) -> float:
    """Share of teammate makes this player assists.

    The advanced line is that share. A player without it keeps the same
    conversion from the per-game assists already on his model.
    """
    if player.ast_pct > 0:
        return float(np.clip(player.ast_pct, 0.01, 0.65))
    return float(np.clip(max(player.ast_share, 0.0) * AST_PCT_FROM_PER_GAME, 0.01, 0.65))


def _pass_chance(
    rng: np.random.Generator,
    ast_pct: np.ndarray,
    lineup: np.ndarray,
    actor_slot: np.ndarray,
    actor: np.ndarray,
    type_idx: np.ndarray,
    rates: np.ndarray,
    follow: np.ndarray,
    glass: np.ndarray,
    pos: np.ndarray | None = None,
    ast_allowed: float | np.ndarray = 1.0,
    ast_share: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Who passed, and whether this trip was a pass.

    A teammate's assist percentage is the share of makes he creates. A
    spot-up is more often that pass than an isolation, so the play type
    scales the share around the league mix of makes. The shooter is out
    of the draw. A shot off the glass was passed by the rebounder.
    Defenses that concede assists raise the chance; their G/F/C share
    tilts who among the four remaining gets the pass credit.
    """
    n = lineup.shape[0]
    rows = np.arange(n)
    weights = np.array(ast_pct, dtype=float, copy=True)
    weights[rows, actor_slot] = 0.0
    # Volume stays on personal ast_pct × defense ast_allowed. Position shares
    # only tilt who among the four remaining gets the credit.
    total = weights.sum(axis=1)
    if ast_share is not None:
        who = weights * np.asarray(ast_share, dtype=float)
        who[rows, actor_slot] = 0.0
    elif pos is not None:
        codes = np.clip(np.asarray(pos, dtype=int), 0, 2)
        prior = np.array(
            [_AST_SHARE_PRIOR["G"], _AST_SHARE_PRIOR["F"], _AST_SHARE_PRIOR["C"]],
            dtype=float,
        )
        who = weights * prior[codes]
        who[rows, actor_slot] = 0.0
    else:
        who = weights
    factor = rates[type_idx] / ASSIST_RATE_BASE
    volume = np.clip(np.asarray(ast_allowed, dtype=float), 0.88, 1.12)
    chance = np.clip(factor * total * volume, 0.0, 1.0)
    slot = _sample_rows(rng, np.maximum(who, 1e-6))
    passer = lineup[rows, slot]
    kick = follow & (glass >= 0) & (glass != actor)
    chance = np.where(kick, np.clip(rates[type_idx] * volume, 0.0, 0.95), chance)
    passer = np.where(kick, glass, passer)
    passed = rng.random(n) < chance
    return passed, passer


def _pos_share_rows(pos: np.ndarray, share: dict | None, prior: dict) -> np.ndarray:
    """Map (n, 5) position codes through a G/F/C share dict."""
    table = share or prior
    lookup = np.array(
        [float(table.get("G", prior["G"])), float(table.get("F", prior["F"])), float(table.get("C", prior["C"]))],
        dtype=float,
    )
    codes = np.clip(np.asarray(pos, dtype=int), 0, 2)
    return lookup[codes]


def _dreb_weights(
    base: np.ndarray,
    pos: np.ndarray,
    type_idx: np.ndarray,
    reb_share_allowed: dict | None,
) -> np.ndarray:
    """Personal DREB × defense position prior × mild shot-type tilt."""
    weights = np.array(base, dtype=float, copy=True)
    weights *= _pos_share_rows(pos, reb_share_allowed, _REB_SHARE_PRIOR)
    codes = np.clip(np.asarray(pos, dtype=int), 0, 2)
    tilt = _DREB_SHOT_TILT[np.asarray(type_idx, dtype=int)[:, None], codes]
    weights *= tilt
    return np.maximum(weights, 1e-6)


def _pos_code(player: PlayerModel) -> int:
    key = position_key(player.position)
    return {"G": 0, "F": 1, "C": 2}.get(key, 1)


def _three_attempts(player: PlayerModel) -> float:
    freq = player.zone_freq or {}
    return float(freq.get("corner3", 0.0) + freq.get("atb3", 0.0))


@dataclass
class Side:
    players: list[PlayerModel]
    poss: np.ndarray
    tov: np.ndarray
    ft: np.ndarray
    ft_pct: np.ndarray
    foul_rate: np.ndarray
    pos: np.ndarray
    blk: np.ndarray
    stl: np.ndarray
    oreb: np.ndarray
    dreb: np.ndarray
    box: np.ndarray
    ast_pct: np.ndarray
    screen: np.ndarray
    three_att: np.ndarray
    three_make: np.ndarray
    two_fg: np.ndarray
    three_fg: np.ndarray
    pts_make: np.ndarray
    usage: np.ndarray
    fg: np.ndarray
    fg_sample: np.ndarray
    putback: np.ndarray
    clutch: np.ndarray
    ppp: np.ndarray
    order: list[int]
    available: np.ndarray
    star: np.ndarray
    forced_tov: np.ndarray
    forced_n: np.ndarray
    def_ppp: np.ndarray
    def_n: np.ndarray
    off_impact: np.ndarray
    def_impact: np.ndarray
    ft_draw: np.ndarray
    # Opponent free-throw rate against the league. 1.0 leaves the whistle alone.
    fta_allowed: float = 1.0
    # Team offensive rebound rate, team defensive rebound rate, and the
    # offensive rebound rate this defense gives up.
    team_oreb: float = 0.27
    team_dreb: float = 0.73
    team_oreb_allowed: float = 0.27
    # Opponent assist volume multiplier and G/F/C concession shares.
    ast_allowed: float = 1.0
    reb_share_allowed: dict = field(default_factory=lambda: dict(_REB_SHARE_PRIOR))
    ast_share_allowed: dict = field(default_factory=lambda: dict(_AST_SHARE_PRIOR))
    # Mean five-man OREB% from the lineup book (mild blend into glass chance).
    lineup_oreb_mean: float = 0.0
    squads: dict = field(default_factory=dict)
    team_off: float = 114.0
    team_def: float = 114.0
    team_pace: float = 99.5
    court_on: bool = False
    # Per-sim offense night multiplier. Applied on move, not on the PPP table,
    # so five-man squad residuals cannot cancel the same shock.
    night: np.ndarray = field(default_factory=lambda: np.ones(1))
    # Team games already played this season (schedule stamp; cold uses appearances).
    team_games: float = 20.0
    # Per-player make form (recent pts/FGA vs card) and appearance counts.
    make_form: np.ndarray = field(default_factory=lambda: np.ones(1))
    make_form_games: np.ndarray = field(default_factory=lambda: np.zeros(1))
    appearances: np.ndarray = field(default_factory=lambda: np.zeros(1))


def _side(players: list[PlayerModel], ppp: np.ndarray) -> Side:
    n_types = len(PLAY_TYPES)
    poss = np.ones((len(players), n_types))
    tov = np.full((len(players), n_types), 0.12)
    ft = np.full((len(players), n_types), 0.08)
    forced_tov = np.full((len(players), n_types), np.nan)
    forced_n = np.zeros((len(players), n_types))
    def_ppp = np.full((len(players), n_types), np.nan)
    def_n = np.zeros((len(players), n_types))
    three_make = np.zeros(len(players))
    two_fg = np.full(len(players), 0.52)
    three_fg = np.full(len(players), 0.36)
    pts_make = np.full(len(players), 2.2)
    putback = np.full(len(players), 0.52)
    fg = np.full((len(players), n_types), 0.46)
    fg_sample = np.zeros((len(players), n_types))
    for index, player in enumerate(players):
        pts_make[index], three_make[index] = make_points(player.zone_freq, player.zone_fg)
        two_fg[index], three_fg[index] = zone_make_rates(player.zone_freq, player.zone_fg)
        skill = player.offense.get("OffRebound")
        if skill is not None and skill.fg_pct > 0:
            putback[index] = skill.fg_pct
        for type_index, name in enumerate(PLAY_TYPES):
            skill = player.offense.get(name)
            if skill is None:
                continue
            poss[index, type_index] = max(0.0, skill.poss)
            tov[index, type_index] = skill.tov_poss_pct
            ft[index, type_index] = skill.ft_poss_pct
            if skill.fg_pct > 0:
                fg[index, type_index] = float(skill.fg_pct)
            if skill.fg_sample > 0:
                fg_sample[index, type_index] = float(skill.fg_sample)
            defended = player.defense.get(name)
            if defended is not None and defended.poss > 0:
                def_ppp[index, type_index] = defended.ppp
                def_n[index, type_index] = defended.poss
                if defended.tov_poss_pct > 0:
                    forced_tov[index, type_index] = defended.tov_poss_pct
                    forced_n[index, type_index] = defended.poss
    order = sorted(range(len(players)), key=lambda i: players[i].minutes_mean, reverse=True)
    available = np.array([player.available and player.minutes_mean > 0 for player in players])
    return Side(
        players=players,
        poss=poss,
        tov=tov,
        ft=ft,
        ft_pct=np.clip([player.ft_pct for player in players], 0.40, 0.95),
        foul_rate=np.array([_foul_rate(player) for player in players]),
        pos=np.array([_pos_code(player) for player in players], dtype=int),
        blk=np.array([max(0.05, player.blk_weight) for player in players]),
        stl=np.array([max(0.05, player.stl_weight) for player in players]),
        oreb=np.array([max(0.005, player.oreb_pct) for player in players]),
        dreb=np.array([max(0.02, player.dreb_pct) for player in players]),
        box=np.array([float(player.box_outs) for player in players]),
        ast_pct=np.array([_player_ast_pct(player) for player in players]),
        screen=np.array([player.screen_assists for player in players]),
        three_att=np.array([_three_attempts(player) for player in players]),
        three_make=three_make,
        two_fg=two_fg,
        three_fg=three_fg,
        pts_make=pts_make,
        usage=np.array([float(player.usg) for player in players], dtype=float),
        fg=fg,
        fg_sample=fg_sample,
        putback=putback,
        clutch=np.array([player.clutch_mult for player in players]),
        ppp=ppp,
        order=order,
        available=available,
        star=np.array([player.minutes_mean >= 32.0 for player in players]),
        forced_tov=forced_tov,
        forced_n=forced_n,
        def_ppp=def_ppp,
        def_n=def_n,
        off_impact=np.array([float(player.off_impact) for player in players], dtype=float),
        def_impact=np.array([float(player.def_impact) for player in players], dtype=float),
        ft_draw=np.clip([float(player.ft_draw) for player in players], 0.45, 1.85),
        team_games=max((float(getattr(player, "minutes_team_games", 0.0) or 0.0) for player in players), default=20.0),
        make_form=np.array([float(getattr(player, "make_form", 1.0) or 1.0) for player in players], dtype=float),
        make_form_games=np.array(
            [float(getattr(player, "make_form_games", 0.0) or 0.0) for player in players], dtype=float
        ),
        appearances=np.array(
            [float(getattr(player, "minutes_appearances", 0.0) or 0.0) for player in players], dtype=float
        ),
    )


@dataclass
class Tables:
    pts: np.ndarray
    reb: np.ndarray
    oreb: np.ndarray
    dreb: np.ndarray
    ast: np.ndarray
    pot_ast: np.ndarray
    stl: np.ndarray
    blk: np.ndarray
    tpm: np.ndarray
    tpa: np.ndarray
    fgm: np.ndarray
    fga: np.ndarray
    pf: np.ndarray
    ftm: np.ndarray
    fta: np.ndarray
    tov: np.ndarray


def _tables(n_sims: int, n_players: int) -> Tables:
    zeros = lambda: np.zeros((n_sims, n_players))
    return Tables(
        pts=zeros(),
        reb=zeros(),
        oreb=zeros(),
        dreb=zeros(),
        ast=zeros(),
        pot_ast=zeros(),
        stl=zeros(),
        blk=zeros(),
        tpm=zeros(),
        tpa=zeros(),
        fgm=zeros(),
        fga=zeros(),
        pf=zeros(),
        ftm=zeros(),
        fta=zeros(),
        tov=zeros(),
    )


def _add(box: np.ndarray, index: np.ndarray, amount: np.ndarray, mask: np.ndarray) -> None:
    if not np.any(mask):
        return
    use = np.where(mask, index, 0)
    value = np.where(mask, amount, 0.0)
    np.add.at(box, (np.arange(box.shape[0]), use), value)


def should_sit(
    period: int,
    seconds_left: np.ndarray,
    fouls: np.ndarray,
    margin: np.ndarray,
    patience: float,
    foul_rate: np.ndarray,
    is_star: np.ndarray,
) -> np.ndarray:
    """True when this coach takes the player out for foul trouble.

    The league line, measured off play-by-play, is two fouls in the first
    quarter, three before half, and four in the third. A loose staff only
    does that when it happens early, and a starter on that staff gets one
    extra foul. A player who already fouls a lot sits deeper into the quarter.
    Five fouls in the fourth only clear the floor when the game is not close
    and more than six minutes remain. Inside that window he plays.
    """
    fouls = np.asarray(fouls, dtype=float)
    seconds = np.asarray(seconds_left, dtype=float)
    margin = np.asarray(margin, dtype=float)
    rates = np.asarray(foul_rate, dtype=float)
    stars = np.asarray(is_star, dtype=bool)
    if fouls.ndim == 2 and seconds.ndim == 1:
        seconds = seconds[:, None]
        margin = margin[:, None]
    limit = np.full(fouls.shape, period + 1, dtype=float)
    # A starter on a normal staff plays through one extra foul. Only a strict
    # staff sits him on the raw league line.
    star_slack = stars & (patience >= 0.35) & (period < 4)
    limit = np.where(star_slack, limit + 1.0, limit)
    window = 120.0 + float(patience) * 420.0
    window = np.where((rates >= 1.25) & ~stars, window * 0.65, window)
    window = np.where(star_slack, window * 1.25, window)
    early = (period < 4) & (fouls >= limit) & (seconds >= window)
    late = (
        (period >= 4)
        & (fouls >= 5.0)
        & (float(patience) < 0.6)
        & (seconds > 360.0)
        & (np.abs(margin) >= 10.0)
    )
    return (fouls >= FOUL_OUT) | early | late


def _in_trouble(side: Side, candidate: int, pf_row: np.ndarray, period: int, qclock: float, margin: float, patience: float) -> bool:
    return bool(
        should_sit(
            period,
            np.array([qclock]),
            np.array([pf_row[candidate]]),
            np.array([margin]),
            patience,
            np.array([side.foul_rate[candidate]]),
            np.array([side.star[candidate]]),
        )[0]
    )


def _replacement(
    side: Side,
    present: set[int],
    pf_row: np.ndarray,
    current: int,
    period: int,
    qclock: float,
    margin: float,
    patience: float,
    must: bool,
) -> int | None:
    """Next man for a player who has to leave.

    He comes from further down the rotation, at the same position when that
    body is there, and the one who has played with the other four when more
    than one is. A resting star is the last resort, and only when somebody has
    fouled out and nobody deeper is available. Early foul trouble with no clean
    body behind him leaves him on the floor.
    """
    current_min = float(side.players[current].minutes_mean)
    current_pos = _pos_code(side.players[current])
    fallback = None
    best = None
    best_key = None
    for candidate in side.order:
        if candidate in present or not bool(side.available[candidate]):
            continue
        if pf_row[candidate] >= FOUL_OUT:
            continue
        trouble = _in_trouble(side, candidate, pf_row, period, qclock, margin, patience)
        if trouble and not must:
            continue
        minutes = float(side.players[candidate].minutes_mean)
        if minutes < current_min - 0.05:
            key = _sub_key(side, candidate, present, current, current_pos, minutes)
            if best_key is None or key < best_key:
                best = candidate
                best_key = key
        elif fallback is None:
            fallback = candidate
    if best is not None:
        return best
    return fallback if must else None


def _floor_share(player: PlayerModel, mates: list[PlayerModel]) -> float:
    book = player.shared_minutes
    if not book:
        return 0.0
    return sum(float(book.get(int(mate.player_id), 0.0)) for mate in mates)


def _sub_key(side: Side, candidate: int, present: set[int], current: int, current_pos: int, minutes: float):
    """Same position first, then the man who has played with the other four, then minutes.

    A spot minute waits behind every regular. He is the body when the rotation is gone.
    """
    player = side.players[candidate]
    mates = [side.players[index] for index in present if index != current]
    spot = 1 if minutes < 6.0 else 0
    distance = abs(_pos_code(player) - current_pos)
    return (spot, distance, -_floor_share(player, mates), -minutes)


def _substitute(lineup: np.ndarray, pf: np.ndarray, side: Side, period: int, qclock: np.ndarray, margin: np.ndarray, patience: float) -> None:
    on_floor = pf[np.arange(pf.shape[0])[:, None], lineup] >= FOUL_OUT
    if not on_floor.any():
        return
    for sim in np.flatnonzero(on_floor.any(axis=1)):
        present = {int(slot) for slot in lineup[sim]}
        for slot in range(5):
            current = int(lineup[sim, slot])
            if pf[sim, current] < FOUL_OUT:
                continue
            candidate = _replacement(
                side, present, pf[sim], current, period, float(qclock[sim]), float(margin[sim]), patience, True,
            )
            if candidate is None:
                continue
            present.discard(current)
            present.add(candidate)
            lineup[sim, slot] = candidate


def _restore_foul_rests(
    lineup: np.ndarray,
    pf: np.ndarray,
    sat_who: np.ndarray,
    sat_back: np.ndarray,
    qclock: np.ndarray,
) -> None:
    due = (sat_who >= 0) & (qclock[:, None] <= sat_back)
    if not np.any(due):
        return
    for sim, slot in zip(*np.nonzero(due)):
        original = int(sat_who[sim, slot])
        sat_who[sim, slot] = -1
        if pf[sim, original] >= FOUL_OUT:
            continue
        if original in {int(player) for player in lineup[sim]}:
            continue
        lineup[sim, slot] = original


def _sit_foul_trouble(
    lineup: np.ndarray,
    pf: np.ndarray,
    side: Side,
    period: int,
    qclock: np.ndarray,
    margin: np.ndarray,
    patience: float,
    rested: np.ndarray,
    sat_who: np.ndarray,
    sat_back: np.ndarray,
) -> None:
    rows = np.arange(lineup.shape[0])
    fouls = pf[rows[:, None], lineup]
    sit = should_sit(
        period,
        qclock,
        fouls,
        margin,
        patience,
        side.foul_rate[lineup],
        side.star[lineup],
    )
    if not np.any(sit):
        return
    for sim in np.flatnonzero(sit.any(axis=1)):
        present = {int(slot) for slot in lineup[sim]}
        for slot in range(5):
            if not bool(sit[sim, slot]):
                continue
            current = int(lineup[sim, slot])
            if int(rested[sim, current]) == period:
                continue
            candidate = _replacement(
                side, present, pf[sim], current, period, float(qclock[sim]), float(margin[sim]), patience, False,
            )
            if candidate is None:
                continue
            present.discard(current)
            present.add(candidate)
            lineup[sim, slot] = candidate
            rested[sim, current] = period
            sat_who[sim, slot] = current
            sat_back[sim, slot] = float(qclock[sim]) - FOUL_REST


def _add_floor_time(state: GameState, time_used: np.ndarray) -> None:
    home_floor = getattr(state, "home_floor", None)
    away_floor = getattr(state, "away_floor", None)
    if home_floor is None or away_floor is None:
        return
    played = np.where(time_used > 0, time_used, 0.0)
    rows = np.arange(state.n)
    for slot in range(state.home_lineup.shape[1]):
        np.add.at(home_floor, (rows, state.home_lineup[:, slot]), played)
        np.add.at(away_floor, (rows, state.away_lineup[:, slot]), played)


def _team_pace(team: TeamModel) -> float:
    pace = float(team.pace or 0.0)
    return pace if pace > 80.0 else 99.5


def _fit_pace_impacts(team: TeamModel) -> dict[int, float]:
    """How many possessions per 48 each player adds, from the fives he played.

    A five's pace is the team pace plus the five player effects. The same
    96-minute prior used for a five is split across its players, so a short
    look stays near the team and a player keeps his share in a new group.
    """
    team_pace = _team_pace(team)
    rows: list[tuple[tuple[int, ...], float, float]] = []
    index: dict[int, int] = {}
    for group, pace in team.lineup_pace.items():
        if pace is None or float(pace) <= 80.0 or len(group) != 5:
            continue
        minutes = float(team.lineup_minutes.get(group, 0.0) or 0.0)
        if minutes <= 0.0:
            continue
        rows.append((group, float(pace), minutes))
        for player_id in group:
            if int(player_id) not in index:
                index[int(player_id)] = len(index)
    if not rows:
        return {}
    width = len(index)
    gram = np.zeros((width, width))
    side = np.zeros(width)
    for group, pace, minutes in rows:
        residual = pace - team_pace
        cols = [index[int(player_id)] for player_id in group]
        for left in cols:
            side[left] += minutes * residual
            for right in cols:
                gram[left, right] += minutes
    # Five players share one observation, so the prior is five times the
    # minutes that would trust the five itself.
    gram.flat[:: width + 1] += 5.0 * _FIVE_PRIOR_MINUTES
    try:
        beta = np.linalg.solve(gram, side)
    except np.linalg.LinAlgError:
        return {}
    return {player_id: float(beta[slot]) for player_id, slot in index.items()}


def _pace_profile(team: TeamModel) -> dict[int, float]:
    stored = team.pace_impact
    if stored is not None:
        return stored
    fitted = _fit_pace_impacts(team)
    team.pace_impact = fitted
    return fitted


def _unit_pace(team: TeamModel, ids: list[int]) -> float:
    """Pace this five brings. The players carry it. The exact group does not have to have played."""
    team_pace = _team_pace(team)
    if len(ids) < 5:
        return team_pace
    impacts = _pace_profile(team)
    return team_pace + sum(float(impacts.get(int(player_id), 0.0)) for player_id in ids[:5])


def _pace(home: TeamModel, away: TeamModel, home_ids: list[int], away_ids: list[int], pace_mult: float) -> float:
    """Possessions per 48 from the two fives on the floor.

    A five's pace is the team pace plus each player's share of the fives he
    has already played. The game is the product of those two rates against a
    100-pace night, so a grinding group lengthens the trip and a running
    group shortens it. Altitude is the only extra scale.
    """
    base = _unit_pace(home, home_ids) * _unit_pace(away, away_ids) / 100.0
    return max(80.0, base * pace_mult)


def _unit_ids(players: list[PlayerModel], unit: list[int] | np.ndarray) -> list[int]:
    return [players[int(index)].player_id for index in list(unit)[:5]]


def _broadcast(unit: list[int], n_sims: int) -> np.ndarray:
    return np.tile(np.asarray(unit[:5], dtype=np.int64), (n_sims, 1))


@dataclass
class ClockGame:
    home: Tables
    away: Tables
    second_home: np.ndarray
    second_away: np.ndarray
    margin_before_q4: np.ndarray
    home_team_fouls: np.ndarray
    away_team_fouls: np.ndarray
    home_max_quarter_fouls: np.ndarray
    away_max_quarter_fouls: np.ndarray
    home_ft_trips: np.ndarray
    away_ft_trips: np.ndarray
    home_possessions: np.ndarray
    away_possessions: np.ndarray
    intentional_fouls: np.ndarray
    flagrant_fouls: np.ndarray
    ejections: np.ndarray
    blowout: np.ndarray
    home_floor: np.ndarray
    away_floor: np.ndarray
    home_plan: np.ndarray
    away_plan: np.ndarray
    oreb_p: dict[tuple[str, str], float] = field(default_factory=dict)


@dataclass
class GameState:
    n: int
    qclock: np.ndarray
    segment_left: np.ndarray
    offense_home: np.ndarray
    arrow_home: np.ndarray
    live_next: np.ndarray
    retain: np.ndarray
    home_tf: np.ndarray
    away_tf: np.ndarray
    home_game_fouls: np.ndarray
    away_game_fouls: np.ndarray
    home_max_q: np.ndarray
    away_max_q: np.ndarray
    home_to: np.ndarray
    away_to: np.ndarray
    home_lineup: np.ndarray
    away_lineup: np.ndarray
    home_ft_trips: np.ndarray
    away_ft_trips: np.ndarray
    home_poss: np.ndarray
    away_poss: np.ndarray
    intentional: np.ndarray
    flagrant: np.ndarray
    ejected: np.ndarray
    second_home: np.ndarray
    second_away: np.ndarray


def _live_board(chance: np.ndarray, follow: np.ndarray) -> np.ndarray:
    """A second-chance trip does not start another offensive rebound."""
    return np.where(follow, 0.0, chance)


def _score(box: Tables) -> np.ndarray:
    return box.pts.sum(axis=1)


def _close_quarter(state: GameState) -> None:
    state.home_max_q = np.maximum(state.home_max_q, state.home_tf)
    state.away_max_q = np.maximum(state.away_max_q, state.away_tf)
    state.home_tf[:] = 0
    state.away_tf[:] = 0


def _charge(state: GameState, offense_home: np.ndarray, team_foul: np.ndarray, mask: np.ndarray) -> None:
    home_charge = mask & team_foul & ~offense_home
    away_charge = mask & team_foul & offense_home
    state.home_tf += home_charge
    state.away_tf += away_charge
    state.home_game_fouls += home_charge
    state.away_game_fouls += away_charge


def _matchup_vectors(offense: list[PlayerModel], defense: list[PlayerModel], defense_team: TeamModel, league) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def_mult = np.array(
        [defense_multiplier(name, defense, defense_team, league) for name in PLAY_TYPES],
        dtype=float,
    )
    zone = np.array([zone_matchup_mult(player, defense_team, league, defense) for player in offense])
    pair = np.array([matchup_multiplier(player, defense) for player in offense])
    return def_mult, zone, pair


def _tov_vector(team: TeamModel) -> np.ndarray:
    return np.array([float(team.def_tov.get(name, np.nan)) for name in PLAY_TYPES], dtype=float)


def _touch_weights(poss: np.ndarray, usage: np.ndarray, flat: np.ndarray | None = None) -> np.ndarray:
    """Play mix times this player's usage.

    A side still stored at the flat 20% default keeps its possession weights.
    Once any player has his own usage, the five on the floor split the trip
    by that usage and each man's mix.
    """
    check = usage if flat is None else flat
    if check.size and float(np.max(np.abs(np.asarray(check, dtype=float) - 0.20))) <= 0.0101:
        return poss
    totals = poss.sum(axis=-1, keepdims=True)
    mix = np.divide(poss, np.maximum(totals, 1e-9))
    rate = np.clip(np.asarray(usage, dtype=float), 0.04, 0.45)
    return mix * rate[..., None]


def _second_chance_poss(
    side: Side,
    lineup: np.ndarray,
    glass: np.ndarray,
    on_poss: np.ndarray,
    follow: np.ndarray,
) -> np.ndarray:
    """The rebounder puts it back. Everyone else is a catch off the glass.

    Weights are those players' own possessions, not a league putback rate.
    A rebounder who is no longer on the floor leaves the normal mix alone.
    """
    del side
    if glass is None or not np.any(follow):
        return on_poss
    source = on_poss
    out = source.copy()
    same = lineup == glass[:, None]
    live = follow & same.any(axis=1)
    if not np.any(live):
        return source
    out[live] = 0.0
    for slot in range(lineup.shape[1]):
        mask = live & same[:, slot]
        if np.any(mask):
            out[mask, slot, OFFREB] = source[mask, slot, OFFREB]
        for play in KICK:
            other = live & ~same[:, slot]
            if np.any(other):
                out[other, slot, play] = source[other, slot, play]
    empty = live & (out.sum(axis=(1, 2)) <= 1e-9)
    out[empty] = source[empty]
    return out


def _forced_rate(
    defense: Side,
    def_lineup: np.ndarray,
    type_idx: np.ndarray,
    team_tov: np.ndarray,
) -> np.ndarray:
    """Turnovers this lineup forces on the drawn action.

    A defender's own sample shrinks toward his team's rate for that play.
    Who is likely to be guarding the action decides the weight. No team
    sample and no defender sample leaves the rate unknown.
    """
    n = def_lineup.shape[0]
    team = team_tov[type_idx]
    team_ok = np.isfinite(team)
    pos = defense.pos[def_lineup]
    match = MATCHUP[type_idx[:, None], pos]
    rates = []
    weights = []
    for slot in range(def_lineup.shape[1]):
        players = def_lineup[:, slot]
        sample = defense.forced_n[players, type_idx]
        raw = defense.forced_tov[players, type_idx]
        prior = np.where(team_ok, team, raw)
        shrunk = np.where(sample > 0, (sample * raw + FILE_PRIOR * prior) / (sample + FILE_PRIOR), prior)
        ok = np.isfinite(shrunk) & (shrunk > 0)
        weight = np.where(ok, match[:, slot] * (sample + np.where(team_ok, 40.0, 0.0)), 0.0)
        rates.append(np.where(ok, shrunk, 0.0))
        weights.append(weight)
    rate = np.stack(rates, axis=1)
    weight = np.stack(weights, axis=1)
    total = weight.sum(axis=1)
    out = np.full(n, np.nan)
    good = total > 0
    out[good] = (rate[good] * weight[good]).sum(axis=1) / total[good]
    return out


def _matchup_tilt(defense: Side, def_lineup: np.ndarray, type_idx: np.ndarray) -> np.ndarray:
    """Primary defender's points allowed on this action, divided by the five's rate.

    Position decides who is likely guarding it. A player with no defensive
    profile stays out of the weight, and the five-man rate is left as it is.
    """
    n = def_lineup.shape[0]
    pos = defense.pos[def_lineup]
    role = MATCHUP[type_idx[:, None], pos]
    sample = defense.def_n[def_lineup, type_idx[:, None]]
    ppp = defense.def_ppp[def_lineup, type_idx[:, None]]
    known = np.isfinite(ppp) & (sample > 0)
    poss_w = np.where(known, sample, 0.0)
    match_w = np.where(known, role * np.sqrt(np.maximum(sample, 0.0)), 0.0)
    poss_sum = poss_w.sum(axis=1)
    match_sum = match_w.sum(axis=1)
    tilt = np.ones(n)
    good = (poss_sum > 0) & (match_sum > 0)
    if not np.any(good):
        return tilt
    team = (np.where(known, ppp, 0.0) * poss_w).sum(axis=1) / np.maximum(poss_sum, 1e-9)
    matched = (np.where(known, ppp, 0.0) * match_w).sum(axis=1) / np.maximum(match_sum, 1e-9)
    raw = matched[good] / np.maximum(team[good], 1e-6)
    tilt[good] = np.where(np.isfinite(raw), raw, 1.0)
    return tilt


def _conceded_rate(defense: Side, def_lineup: np.ndarray, type_idx: np.ndarray, def_pf: np.ndarray) -> np.ndarray:
    """Foul rate of the defender this action is charged to.

    Position says who is guarding it. His personal-foul rate says how often
    that matchup sends the offense to the line. A player who has fouled out
    is already off the whistle.
    """
    rows = np.arange(def_lineup.shape[0])
    pos = defense.pos[def_lineup]
    rates = defense.foul_rate[def_lineup]
    match = MATCHUP[type_idx[:, None], pos]
    raw = rates * match
    raw = np.where(def_pf[rows[:, None], def_lineup] >= FOUL_OUT, 0.0, raw)
    weight = np.maximum(raw, 1e-6)
    conceded = (weight * rates).sum(axis=1) / weight.sum(axis=1)
    return np.clip(conceded, 0.45, 2.4)


# Possessions of field-goal evidence that tie the shot chart. A longer sample
# keeps his Synergy percentage. The count is the season, not possessions per game.
# Possessions of zone-chart evidence beside Synergy. Lower lets this season's
# measured make rate outrun the prior-year chart sooner.
_FG_PRIOR = 24.0
# Share of non-OREB misses that are team/dead-ball boards (no player DREB).
# Sized once from the 2025-26 logic-fix DREB gap (~+6.6 on ~39 player DREB),
# then trimmed after the live-clock gate also moved period-end boards here.
_TEAM_REB_SHARE = 0.155
# Per-actor early cold from his own appearances. Threes start colder than twos
# so October three-point heat cools first; both rise to the season make level.
_EARLY_COLD_FLOOR_TWO = 0.93
_EARLY_COLD_FLOOR_THREE = 0.91
_EARLY_COLD_GAMES = 16.0
# Season-long make cooler sized from post-eval FG% +0.010 (0.482/0.472). Not a
# PPP stack and not a foul retune — a direct make-rate level.
_MAKE_LEVEL = 0.980
# Shot-end mass ceiling. Lower leaves more room for non-shot ends (fewer FGA).
_SHOT_SCALE_CEILING = 0.895
# Soft-defense mix hunt: volume tilt only (makes already use def_mult).
_MIX_HUNT_ALPHA = 0.35
_MIX_HUNT_CLIP = (0.85, 1.18)
# League priors for position×defense concession shares.
_REB_SHARE_PRIOR = {"G": 0.22, "F": 0.38, "C": 0.40}
_AST_SHARE_PRIOR = {"G": 0.55, "F": 0.35, "C": 0.10}
# Shot-aware DREB tilt by PLAY_TYPES index × pos code (G, F, C).
# Paint/roll → bigs; threes/spot → wings.
_DREB_SHOT_TILT = np.array(
    [
        [0.85, 1.00, 1.20],  # Isolation
        [1.05, 1.00, 0.95],  # Transition
        [1.10, 1.05, 0.85],  # PRBallHandler
        [0.80, 1.00, 1.25],  # PRRollman
        [0.75, 1.00, 1.30],  # Postup
        [1.20, 1.05, 0.80],  # Spotup
        [1.15, 1.05, 0.85],  # Handoff
        [0.90, 1.05, 1.10],  # Cut
        [1.15, 1.05, 0.85],  # OffScreen
        [0.70, 1.00, 1.35],  # OffRebound
        [1.00, 1.00, 1.00],  # Misc
    ],
    dtype=float,
)
# Mild TOV volume lift; steal share of live turnovers (post-eval STL/TOV short).
_TOV_LEVEL = 1.05
_STEAL_SHARE = 0.62
_SPOT_PLAYS = ("Isolation", "Spotup", "Handoff", "Cut", "OffScreen")
_PICK_PLAYS = ("PRBallHandler", "PRRollman")
_COARSE_PLAY = {"transition": "Transition", "post": "Postup"}


def _around(signal: np.ndarray, level: np.ndarray) -> np.ndarray:
    """The part of one shot read the defensive level has not already said.

    A read on 1 has nothing to add. A read that matches the level, or is a
    smaller move in the same direction, is that same edge. A deeper move in
    that direction contributes only the extra, so the product is the deeper
    read and not the product of both. A read that points the other way keeps
    its own distance from 1 and does not take the level's place.
    """
    signal, level = np.broadcast_arrays(np.asarray(signal, dtype=float), np.asarray(level, dtype=float))
    extra = np.ones(signal.shape, dtype=float)
    safe_signal = np.clip(signal, 1e-3, 1e3)
    safe_level = np.clip(level, 1e-3, 1e3)
    sig = np.log(safe_signal)
    lev = np.log(safe_level)
    informative = np.abs(signal - 1.0) > 0.005
    level_neutral = np.abs(level - 1.0) <= 0.005
    extra = np.where(informative & level_neutral, signal, extra)
    same = informative & ~level_neutral & (sig * lev > 0.0)
    deeper = same & (np.abs(sig) > np.abs(lev) + 1e-6)
    extra = np.where(deeper, signal / np.maximum(level, 1e-6), extra)
    opposite = informative & ~level_neutral & (sig * lev < 0.0)
    extra = np.where(opposite, signal, extra)
    return np.where(np.isfinite(extra) & (extra > 0.0), extra, 1.0)


def _shot_residual(signals: list[np.ndarray], level: np.ndarray) -> np.ndarray:
    """Zone and the head-to-head share one shot residual. The stronger read wins.

    They are two measurements of the same shooting edge. Multiplying both
    disagreements would divide the defensive level out twice.
    """
    level = np.asarray(level, dtype=float)
    best = np.ones(level.shape, dtype=float)
    strength = np.zeros(level.shape, dtype=float)
    for signal in signals:
        residual = _around(signal, level)
        moved = np.abs(np.log(np.clip(residual, 1e-3, 1e3)))
        take = moved > strength
        best = np.where(take, residual, best)
        strength = np.where(take, moved, strength)
    return best


def _crosswalk_mix(coarse: dict, synergy: np.ndarray) -> np.ndarray:
    """Spread a play-by-play bucket across the Synergy actions inside it.

    Spot covers isolation, spot-up, handoff, cut, and off-screen, in the
    proportion this five already runs them. Pick covers the handler and the
    roll man the same way. Transition and post map onto themselves. Actions
    the file does not record keep their Synergy share.
    """
    base = np.asarray(synergy, dtype=float)
    total = float(base.sum())
    base = base / total if total > 0 else np.ones(len(PLAY_TYPES)) / len(PLAY_TYPES)
    if not coarse:
        return base
    lowered = {
        str(key).lower().replace(" ", "").replace("-", ""): float(value)
        for key, value in coarse.items()
    }
    out = np.zeros(len(PLAY_TYPES), dtype=float)
    touched = np.zeros(len(PLAY_TYPES), dtype=bool)

    def pour(names: tuple[str, ...], mass: float) -> None:
        indexes = [PLAY_TYPES.index(name) for name in names]
        weights = np.array([max(float(base[index]), 0.0) for index in indexes])
        if float(weights.sum()) <= 0:
            weights[:] = 1.0
        weights = weights / float(weights.sum())
        for index, weight in zip(indexes, weights):
            out[index] = mass * float(weight)
            touched[index] = True

    vocab = ("spot", "pick", "transition", "post")
    if any(key in lowered for key in vocab):
        # A missing recorded bucket is zero. The file saw the trip and it was not that action.
        pour(_SPOT_PLAYS, float(lowered.get("spot", 0.0)))
        pour(_PICK_PLAYS, float(lowered.get("pick", 0.0)))
        pour(("Transition",), float(lowered.get("transition", 0.0)))
        pour(("Postup",), float(lowered.get("post", 0.0)))
    named = {name.lower(): name for name in PLAY_TYPES}
    for key, mass in lowered.items():
        if key in vocab:
            continue
        play = named.get(key)
        if play is None:
            continue
        pour((play,), mass)
    if not touched.any():
        return base
    # Offensive rebounds and misc were not in the play-by-play file. They keep
    # the rate card. The recorded buckets fill whatever share is left.
    untouched = ~touched
    out[untouched] = base[untouched]
    kept = float(out[untouched].sum())
    poured = float(out[touched].sum())
    room = max(0.0, 1.0 - kept)
    if poured <= 0:
        return base
    out[touched] *= room / poured
    return out


def _composed_ppp(side: Side, indexes: list[int], rows: np.ndarray) -> np.ndarray:
    """Possession-weighted points per possession of these players, per path."""
    count = int(np.count_nonzero(rows))
    chosen = np.array(indexes[:5], dtype=int)
    if chosen.size == 0:
        return np.ones(count)
    poss = np.maximum(side.poss[chosen], 0.0)
    if float(poss.sum()) <= 0:
        return np.ones(count)
    drawn = side.ppp[rows][:, chosen, :]
    weight = poss[None, :, :]
    level = (drawn * weight).sum(axis=(1, 2)) / float(poss.sum())
    return np.maximum(level, 0.4)


def _draw_side(
    rng: np.random.Generator,
    side: Side,
    lineup: np.ndarray,
    defense: Side,
    def_lineup: np.ndarray,
    def_pf: np.ndarray,
    def_mult: np.ndarray,
    zone_mult: np.ndarray,
    pair_mult: np.ndarray,
    live_next: np.ndarray,
    retain: np.ndarray,
    context_mult: float,
    residual: float,
    period: int,
    home_score: np.ndarray,
    away_score: np.ndarray,
    clutch_on: np.ndarray,
    follow: np.ndarray | None = None,
    glass: np.ndarray | None = None,
    team_tov: np.ndarray | None = None,
    oreb_prob: float = 0.0,
    team_fouls: np.ndarray | None = None,
    bonus_line: int = 5,
    continuation: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    n = lineup.shape[0]
    rows = np.arange(n)
    if follow is None:
        follow = np.zeros(n, dtype=bool)
    # A putback, or the inbound after a whistle that kept the ball, is still
    # this possession. The five-man reach-in was already rolled.
    kept = np.zeros(n, dtype=bool) if continuation is None else np.asarray(continuation, dtype=bool)
    same_possession = follow | kept
    if getattr(side, "court_on", False) and getattr(side, "squads", None):
        rel = _relative_rows(side, lineup)
        base = _touch_weights(side.poss[lineup], rel, None)
        base = _apply_squad_mix(side, lineup, base)
    else:
        base = _touch_weights(side.poss[lineup], side.usage[lineup], side.usage)
    on_poss = _second_chance_poss(side, lineup, glass, base, follow)
    weights = on_poss.sum(axis=1)
    screens = side.screen[lineup].sum(axis=1)
    boost = np.clip(1.0 + 0.04 * np.clip((screens - 12.0) / 12.0, -1.0, 1.5), 0.9, 1.15)
    weights[:, HANDLER] *= boost
    weights[:, ROLL] *= boost
    # Soft defenses concede volume on their weak play types. Damped so the
    # relative mix shift stays near ±15% even when def_mult is wide.
    hunt = np.clip(np.asarray(def_mult, dtype=float), *_MIX_HUNT_CLIP) ** _MIX_HUNT_ALPHA
    weights *= hunt
    mix = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    # The five's own mix is the average trip. A transition is short against
    # that average, and a post-up is long. The team's pace is applied later,
    # so a team that already runs is not shortened a second time.
    ordinary = mix
    if np.any(live_next):
        trans = np.zeros(len(PLAY_TYPES))
        trans[TRANSITION] = 1.0
        mix = np.where(live_next[:, None], 0.42 * trans + 0.58 * mix, mix)
        mix = mix / mix.sum(axis=1, keepdims=True)
    type_idx = _sample_rows(rng, mix)
    actor_w = on_poss[rows[:, None], np.arange(5), type_idx[:, None]]
    actor_slot = _sample_rows(rng, np.maximum(actor_w, 1e-6))
    actor = lineup[rows, actor_slot]

    def_pos = defense.pos[def_lineup]
    paint = PAINT[type_idx]
    conceded = _conceded_rate(defense, def_lineup, type_idx, def_pf)

    base_ft = side.ft[actor, type_idx]
    # The play says how often the foul is a shooting foul. His free-throw
    # residual and the matched defender each move that once. The five's foul
    # rates are the whole whistle, so the team rate is not applied again.
    play_sfoil = np.clip(base_ft * side.ft_draw[actor] * conceded, 0.0, 0.34)
    p_off = np.clip(0.018 * np.sqrt(np.clip(side.foul_rate[actor], 0.40, 2.6)), 0.008, 0.05)
    player_tov = side.tov[actor, type_idx]
    if team_tov is None:
        lineup_tov = np.full(n, np.nan)
    else:
        lineup_tov = _forced_rate(defense, def_lineup, type_idx, team_tov)
    known = np.isfinite(lineup_tov) & (lineup_tov > 0) & (player_tov > 0)
    # The handler's rate stays his. This defense moves it by how often that
    # defense forces a turnover compared with the league. The two synergy
    # columns are on different bases, so the defense is rescaled first.
    ratio = np.clip(lineup_tov / _LEAGUE_DEF_TOV, 0.50, 1.85)
    mixed = player_tov * ratio
    p_tov = np.clip(np.where(known, mixed, player_tov) - p_off, 0.02, 0.26)
    p_tov = np.clip(p_tov * _TOV_LEVEL, 0.02, 0.28)
    # Personal fouls per trip from the five on the floor. The charge is drawn
    # on its own, and those whistles are already inside each player's foul
    # rate, so they come out of this budget. Shooting fouls take their share.
    # What remains is a reach-in. The action was already chosen.
    # A continuation does not open that budget again. The shot keeps its own
    # foul rate, and the reach-in that used to sit on the second draw is not
    # turned into a field-goal attempt: the continuation is the play's shot,
    # turnover, and shooting foul.
    budget = (defense.foul_rate[def_lineup] * _FOULS_PER_TRIP).sum(axis=1) - p_off
    budget = np.clip(budget, 0.02, 0.45)
    # When the play's shooting-foul rate would eat the whole five-man budget,
    # keep a reach-in slice. Those reach-ins end as FT-only in the bonus (no
    # extra FGA) instead of another field-goal attempt. Below the budget the
    # leftover already is the reach-in, so the free-throw residual stays linear.
    eats_budget = play_sfoil >= (budget - 1e-9)
    reserved_nsf = np.where(eats_budget, np.minimum(budget * 0.28, 0.07), 0.0)
    capped_sfoil = np.minimum(play_sfoil, np.maximum(budget - reserved_nsf, 0.0))
    # Two kept whistles end the side-out on a fresh trip. The next reach-in is played.
    opening_nsf = np.clip(budget - capped_sfoil, 0.0, 0.30)
    fresh_nsf = np.where(retain >= 2, 0.0, opening_nsf)
    opening_mass = p_off + p_tov + capped_sfoil + fresh_nsf
    opening_scale = np.minimum(1.0, _SHOT_SCALE_CEILING / np.maximum(opening_mass, 1e-6))
    off_open = p_off * opening_scale
    tov_open = p_tov * opening_scale
    sfoil_open = capped_sfoil * opening_scale
    nsf_open = fresh_nsf * opening_scale
    shot_open = 1.0 - off_open - tov_open - sfoil_open - nsf_open
    # Bonus state belongs in the continuation split: a kept whistle that has
    # already loaded the team foul line can end as two free throws instead of
    # another field-goal attempt. Putbacks still do not reopen the reach-in.
    if team_fouls is None:
        team_fouls = np.zeros(n)
    in_bonus = np.asarray(team_fouls, dtype=float) + 1.0 >= float(bonus_line)
    # Continuations keep the budget-capped shooting foul, not the uncapped play
    # rate. A side-out in the bonus can still roll the leftover reach-in as a
    # common foul so the trip can end without another FGA. Putbacks drop the
    # reach-in entirely — they do not reopen it and do not pour it into TOV.
    nsf_continued = np.where(follow, 0.0, opening_nsf)
    continued_mass = p_off + p_tov + nsf_continued + capped_sfoil
    continued_scale = np.minimum(1.0, _SHOT_SCALE_CEILING / np.maximum(continued_mass, 1e-6))
    nsf_kept = np.where(kept & in_bonus, opening_nsf * continued_scale, 0.0)
    rest_nsf = np.where(follow | (kept & in_bonus), 0.0, opening_nsf)
    rest = (p_off + p_tov + rest_nsf) * continued_scale
    live = p_off + p_tov
    off_kept = np.where(live > 1e-9, rest * p_off / np.maximum(live, 1e-9), 0.0)
    tov_kept = rest - off_kept
    sfoil_kept = capped_sfoil * continued_scale
    shot_kept = 1.0 - off_kept - tov_kept - sfoil_kept - nsf_kept
    p_off = np.where(same_possession, off_kept, off_open)
    p_tov = np.where(same_possession, tov_kept, tov_open)
    p_sfoil = np.where(same_possession, sfoil_kept, sfoil_open)
    p_nsf = np.where(follow, 0.0, np.where(kept, nsf_kept, nsf_open))
    p_shot = np.where(same_possession, shot_kept, shot_open)

    # One three-point rate for the make, the miss, and the shooting foul.
    three_p = np.clip(0.35 * PLAY_THREE[type_idx] + 0.65 * side.three_att[actor], 0.02, 0.92)
    ft_pct = side.ft_pct[actor]
    # A common foul is two shots once the bonus is on, and a side-out before it.
    # A late take foul is priced separately when the whistle fires; ordinary
    # reach-ins, including transition, use the common formula here.
    e_nsf = np.where(in_bonus, 2.0 * ft_pct, 0.0)
    base = side.ppp[rows, actor, type_idx]
    if np.ndim(def_mult) == 1:
        def_term = def_mult[type_idx]
        zone_term = zone_mult[actor]
        pair_term = pair_mult[actor]
    else:
        def_term = def_mult[rows, type_idx]
        zone_term = zone_mult[rows, actor]
        pair_term = pair_mult[rows, actor]
    # Play-type points allowed are the defensive level. Zone makes, the
    # head-to-head sample, and a five the file has seen are residuals around
    # that level. On/off is the prior only when the five has not been seen.
    player_off = side.off_impact[lineup].mean(axis=1)
    player_def = defense.def_impact[def_lineup].mean(axis=1)
    if getattr(side, "court_on", False) and getattr(side, "squads", None):
        squad_off, squad_def, off_trust, def_trust, off_known, def_known = _level_rows(
            side, defense, lineup, def_lineup
        )
        lineup_off = np.where(off_known, (1.0 - off_trust) * player_off + off_trust * squad_off, player_off)
        lineup_signal = np.where(def_known, (1.0 - def_trust) * player_def + def_trust * squad_def, player_def)
    else:
        lineup_off = player_off
        lineup_signal = player_def
    # Make move: defense, zone/pair, five-man D signal, matchup tilt, night,
    # clutch, and the team ORTG residual once. Context PPP and lineup_off are
    # trip-value layers — they must not also scale Bernoulli FG%.
    move = def_term * _shot_residual([zone_term, pair_term], def_term) * lineup_signal
    move = move * _matchup_tilt(defense, def_lineup, type_idx)
    night = getattr(side, "night", None)
    if night is not None:
        night = np.asarray(night, dtype=float).reshape(-1)
        if night.size == 1:
            move = move * float(night[0])
        elif night.size == n:
            move = move * night
    if period >= 4:
        move = move * np.where(clutch_on, side.clutch[actor], 1.0)
    residual_term = float(np.clip(residual if residual is not None else 1.0, 0.80, 1.20))
    move = move * residual_term
    # Personal make form (recent pts/FGA vs card) on the make move only — not a
    # second PPP product. Thin samples stay glued near 1.0.
    form_raw = np.asarray(side.make_form[actor], dtype=float)
    form_n = np.asarray(side.make_form_games[actor], dtype=float)
    appear = np.asarray(side.appearances[actor], dtype=float)
    form_prior = np.where(appear < 8.0, 16.0, 8.0)
    form_weight = np.maximum(form_n, 0.0)
    form_shrunk = np.where(
        form_weight > 0.0,
        (form_weight * form_raw + form_prior * 1.0) / (form_weight + form_prior),
        1.0,
    )
    move = move * np.clip(form_shrunk, 0.88, 1.12)
    # Context PPP and lineup_off are trip-value layers, not make multipliers.
    # Keep them off Bernoulli FG% and off the TOV budget (cutting TOV here made
    # the post-eval STL/TOV shortfall worse).
    value = np.clip(float(context_mult) * lineup_off, 0.88, 1.14)
    # The chart says where the shot is. His Synergy percentage says how often
    # it falls, shrunk toward that chart. His points per possession are a
    # prior on that rate. Defense, the night, and the five then move it once.
    two_chart = side.two_fg[actor]
    three_chart = side.three_fg[actor]
    chart = (1.0 - three_p) * two_chart + three_p * three_chart
    # Per-game frequency is the mix. The season count is the evidence. A test
    # that stored the count in ``poss`` still works when the season field is empty.
    evidence = side.fg_sample[actor, type_idx]
    frequency = side.poss[actor, type_idx]
    sample = np.maximum(np.where(evidence > 0, evidence, frequency), 0.0)
    synergy = side.fg[actor, type_idx]
    level = (sample * synergy + _FG_PRIOR * chart) / (sample + _FG_PRIOR)
    gap = level / np.maximum(chart, 1e-6)
    two = two_chart * gap
    three = three_chart * gap
    # The chart and his Synergy percentage are the make. Points per possession
    # already shrunk his rate card. They do not bend the shot again, and the
    # foul's expected points do not take makes away to make room.
    del sample
    # Per-actor cold from his own appearances (not the team's game count).
    early = np.clip(appear / _EARLY_COLD_GAMES, 0.0, 1.0)
    cold_two = _EARLY_COLD_FLOOR_TWO + (_MAKE_LEVEL - _EARLY_COLD_FLOOR_TWO) * early
    cold_three = _EARLY_COLD_FLOOR_THREE + (_MAKE_LEVEL - _EARLY_COLD_FLOOR_THREE) * early
    two_make = np.clip(two * move * cold_two, 0.05, 0.95)
    three_make = np.clip(three * move * cold_three, 0.05, 0.80)
    # The and-1 is this shot. A make is the field goal plus one free throw.
    # A miss is the free throws only.
    e2 = two_make * (2.0 + ft_pct) + (1.0 - two_make) * (2.0 * ft_pct)
    e3 = three_make * (3.0 + ft_pct) + (1.0 - three_make) * (3.0 * ft_pct)
    foul_pts = p_sfoil * ((1.0 - three_p) * e2 + three_p * e3) + p_nsf * e_nsf
    p_volume = p_shot
    make_p = (1.0 - three_p) * two_make + three_p * three_make
    target = base * move * value

    foul_w = defense.foul_rate[def_lineup] * MATCHUP[type_idx[:, None], def_pos]
    foul_w = foul_w * np.where(paint[:, None], 1.0 + 0.12 * np.maximum(defense.blk[def_lineup] - 0.4, 0.0), 1.0)
    foul_w = np.where(def_pf[rows[:, None], def_lineup] >= FOUL_OUT, 0.0, foul_w)
    foul_w = np.where(foul_w.sum(axis=1, keepdims=True) <= 0, 1.0, foul_w)
    def_slot = _sample_rows(rng, foul_w)

    # 14.4 seconds is this lineup at a 100 pace. The action moves that length.
    # A steal is still a transition, because it was not in the ordinary mix.
    expected = np.maximum((ordinary * TEMPO).sum(axis=1), 0.55)
    shape = TEMPO[type_idx] / expected
    noise = rng.lognormal(-0.5 * 0.18 * 0.18, 0.18, n)
    raw = (2880.0 / 200.0) * shape * noise
    # The drawn length is the whole possession, and that already includes a
    # board. The follow-up only decides who shoots it. A fresh shot clock here
    # would play the game long. The gather is the time between the rebound and
    # the shot — longer than a tip-in, shorter than a new trip.
    raw = np.where(follow, np.where(type_idx == OFFREB, 4.0, 6.0), raw)
    ordinary = np.clip(raw, 3.2, 30.0)
    gather = np.clip(raw, 0.5, 30.0)
    return {
        "type_idx": type_idx,
        "actor_slot": actor_slot,
        "actor": actor,
        "def_slot": def_slot,
        "p_nsf": p_nsf,
        "p_off": p_off,
        "p_tov": p_tov,
        "p_sfoil": p_sfoil,
        "p_shot": p_shot,
        "p_volume": p_volume,
        "three_p": three_p,
        "ft_pct": ft_pct,
        "make_p": make_p,
        "two_fg": two_make,
        "three_fg": three_make,
        "foul_pts": foul_pts,
        "target": target,
        "three_make": side.three_make[actor],
        "raw": np.where(follow, gather, ordinary),
        "conceded": conceded,
        "def_foul": defense.foul_rate[def_lineup][rows, def_slot],
    }


def _pick(mask: np.ndarray, home: np.ndarray, away: np.ndarray) -> np.ndarray:
    home = np.asarray(home)
    away = np.asarray(away)
    use = np.asarray(mask)
    if home.ndim > 1 and use.ndim == 1:
        use = use[:, None]
    return np.where(use, home, away)


def _eject(pf: np.ndarray, who: np.ndarray, mask: np.ndarray) -> None:
    """A flagrant-two foul ends his night. The next trip brings the replacement."""
    rows = np.flatnonzero(np.asarray(mask, dtype=bool))
    if rows.size == 0:
        return
    current = pf[rows, who[rows]]
    pf[rows, who[rows]] = np.maximum(current, FOUL_OUT)


def _shoot_fts(
    rng: np.random.Generator,
    fta: np.ndarray,
    ft_pct: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    draws = rng.random((fta.shape[0], 3))
    made = draws < ft_pct[:, None]
    used = np.arange(3)[None, :] < fta[:, None]
    made = made & used
    ftm = made.sum(axis=1).astype(float)
    last = np.clip(fta, 1, 3) - 1
    missed_last = (fta > 0) & ~made[np.arange(fta.shape[0]), last]
    return ftm, missed_last


def _run_segment(
    rng: np.random.Generator,
    state: GameState,
    home: Side,
    away: Side,
    home_box: Tables,
    away_box: Tables,
    home_def_mult: np.ndarray,
    away_def_mult: np.ndarray,
    home_zone: np.ndarray,
    away_zone: np.ndarray,
    home_pair: np.ndarray,
    away_pair: np.ndarray,
    home_context: float,
    away_context: float,
    home_residual: float,
    away_residual: float,
    mean_len_scale: np.ndarray,
    oreb_home: float,
    oreb_away: float,
    period: int,
    bonus_line: int,
) -> None:
    """Play until every simulation has used this segment's clock or the quarter."""
    assist = np.array([ASSIST_RATE[name] for name in PLAY_TYPES])
    for _ in range(900):
        active = (state.qclock > 0.15) & (state.segment_left > 0.15)
        if not active.any():
            return
        _restore_foul_rests(state.home_lineup, home_box.pf, state.home_sat_who, state.home_sat_back, state.qclock)
        _restore_foul_rests(state.away_lineup, away_box.pf, state.away_sat_who, state.away_sat_back, state.qclock)
        home_score = _score(home_box)
        away_score = _score(away_box)
        margin = home_score - away_score
        _substitute(
            state.home_lineup, home_box.pf, home, period, state.qclock, margin,
            float(getattr(state, "home_patience", 0.45)),
        )
        _substitute(
            state.away_lineup, away_box.pf, away, period, state.qclock, -margin,
            float(getattr(state, "away_patience", 0.45)),
        )
        _sit_foul_trouble(
            state.home_lineup, home_box.pf, home, period, state.qclock, margin,
            float(getattr(state, "home_patience", 0.45)),
            state.home_rested, state.home_sat_who, state.home_sat_back,
        )
        _sit_foul_trouble(
            state.away_lineup, away_box.pf, away, period, state.qclock, -margin,
            float(getattr(state, "away_patience", 0.45)),
            state.away_rested, state.away_sat_who, state.away_sat_back,
        )
        if getattr(state, "court", False):
            home_def_mult, away_def_mult, home_zone, away_zone, home_pair, away_pair = _court_vectors(
                state, home, away
            )
            mean_len_scale = _court_pace(state)
        clutch_on = np.abs(home_score - away_score) < CLUTCH_MARGIN
        off_home = state.offense_home
        state.was_follow = state.follow.copy()
        home_follow = state.follow & off_home
        away_follow = state.follow & ~off_home
        home_view = _draw_side(
            rng, home, state.home_lineup, away, state.away_lineup, away_box.pf,
            away_def_mult, home_zone, home_pair, state.live_next, state.retain,
            home_context, home_residual, period, home_score, away_score, clutch_on,
            home_follow, state.glass_home, state.away_def_tov, oreb_home,
            state.away_tf, bonus_line, continuation=(state.spent > 0) & off_home,
        )
        away_view = _draw_side(
            rng, away, state.away_lineup, home, state.home_lineup, home_box.pf,
            home_def_mult, away_zone, away_pair, state.live_next, state.retain,
            away_context, away_residual, period, home_score, away_score, clutch_on,
            away_follow, state.glass_away, state.home_def_tov, oreb_away,
            state.home_tf, bonus_line, continuation=(state.spent > 0) & ~off_home,
        )
        view = {key: _pick(off_home, home_view[key], away_view[key]) for key in home_view}
        rows = np.arange(state.n)
        actor = _pick(off_home, state.home_lineup[rows, home_view["actor_slot"]], state.away_lineup[rows, away_view["actor_slot"]])
        defender = _pick(off_home, state.away_lineup[rows, home_view["def_slot"]], state.home_lineup[rows, away_view["def_slot"]])

        deficit = _pick(off_home, home_score - away_score, away_score - home_score)
        fouls_before = _pick(off_home, state.away_tf, state.home_tf).astype(float)
        hack = active & trailing_mask(period, state.qclock, deficit, fouls_before, bonus_line)
        heave = active & ~hack & (state.qclock < 1.8)
        normal = active & ~hack & ~heave

        off_timeouts = _pick(off_home, state.home_to, state.away_to)
        offense_trails = _pick(off_home, away_score - home_score, home_score - away_score)
        want_timeout = (
            normal
            & (period >= 4)
            & (state.qclock <= 42.0)
            & (state.qclock >= 8.0)
            & (offense_trails >= 1.0)
            & (offense_trails <= 16.0)
            & (off_timeouts > 0)
            & ~state.live_next
        )
        push = (
            normal
            & (period >= 4)
            & (state.qclock > 35.0)
            & (state.qclock < 52.0)
            & (np.abs(home_score - away_score) < 15.0)
        )
        raw = view["raw"] * mean_len_scale
        # A side-out already used part of this possession. The shot gets what is left.
        raw = np.maximum(raw - state.spent, 3.0)
        raw = np.where(want_timeout, np.maximum(3.5, raw - 3.0), raw)
        raw = np.where(push, np.minimum(raw, np.maximum(6.0, state.qclock - 28.0)), raw)
        state.home_to -= want_timeout & off_home
        state.away_to -= want_timeout & ~off_home

        # --- heave ---
        heave_make = heave & (rng.random(state.n) < np.clip(0.03 + 0.035 * state.qclock, 0.0, 0.14))
        _add(home_box.pts, actor, np.where(heave_make, 3.0, 0.0), heave & off_home)
        _add(away_box.pts, actor, np.where(heave_make, 3.0, 0.0), heave & ~off_home)
        _add(home_box.tpm, actor, np.ones(state.n), heave_make & off_home)
        _add(away_box.tpm, actor, np.ones(state.n), heave_make & ~off_home)
        _add(home_box.tpa, actor, np.ones(state.n), heave & off_home)
        _add(away_box.tpa, actor, np.ones(state.n), heave & ~off_home)
        _add(home_box.fgm, actor, np.ones(state.n), heave_make & off_home)
        _add(away_box.fgm, actor, np.ones(state.n), heave_make & ~off_home)
        _add(home_box.fga, actor, np.ones(state.n), heave & off_home)
        _add(away_box.fga, actor, np.ones(state.n), heave & ~off_home)
        state.second_home += np.where(state.was_follow & heave_make & off_home, 3.0, 0.0)
        state.second_away += np.where(state.was_follow & heave_make & ~off_home, 3.0, 0.0)
        state.home_poss += heave & off_home
        state.away_poss += heave & ~off_home
        state.follow = np.where(heave, False, state.follow)
        state.spent = np.where(heave | hack, 0.0, state.spent)

        # --- intentional foul ---
        _resolve_intentional(
            rng, state, home, away, home_box, away_box, hack, actor, off_home,
            home_score, away_score, bonus_line, oreb_home, oreb_away,
        )

        # --- ordinary trip ---
        _resolve_trip(
            rng, state, home, away, home_box, away_box, view, normal, actor, defender,
            off_home, raw, assist, bonus_line, oreb_home, oreb_away,
            period=period, deficit=deficit,
        )

        time_used = np.zeros(state.n)
        time_used = np.where(heave, state.qclock, time_used)
        time_used = np.where(hack, np.minimum(1.4, state.qclock), time_used)
        # normal time is written onto state by _resolve_trip via the attribute
        time_used = np.where(normal, state.pending_time, time_used)
        time_used = np.where(active, np.maximum(time_used, 0.4), 0.0)
        state.qclock = np.maximum(0.0, state.qclock - time_used)
        state.segment_left = np.maximum(0.0, state.segment_left - time_used)
        state.live_next = np.where(active, state.next_live, state.live_next)
        state.next_live = np.zeros(state.n, dtype=bool)
        _add_floor_time(state, time_used)


def _resolve_intentional(
    rng: np.random.Generator,
    state: GameState,
    home: Side,
    away: Side,
    home_box: Tables,
    away_box: Tables,
    hack: np.ndarray,
    actor: np.ndarray,
    off_home: np.ndarray,
    home_score: np.ndarray,
    away_score: np.ndarray,
    bonus_line: int,
    oreb_home: float,
    oreb_away: float,
) -> None:
    n = state.n
    rows = np.arange(n)
    if not hack.any():
        state.intentional += 0
        return
    # Fouler: the defender most willing to foul who is not already out.
    home_rates = home.foul_rate[state.home_lineup]
    away_rates = away.foul_rate[state.away_lineup]
    home_pf = home_box.pf[rows[:, None], state.home_lineup]
    away_pf = away_box.pf[rows[:, None], state.away_lineup]
    def_rates = _pick(off_home, away_rates, home_rates)
    def_pf = _pick(off_home, away_pf, home_pf)
    under = def_pf < 5
    weights = np.where(under, def_rates, 0.0)
    empty = weights.sum(axis=1) <= 0
    weights = np.where(empty[:, None], np.where(def_pf < FOUL_OUT, def_rates, 0.05), weights)
    fouler_slot = _sample_rows(rng, weights)
    home_ft = home.ft_pct[state.home_lineup]
    away_ft = away.ft_pct[state.away_lineup]
    off_ft = _pick(off_home, home_ft, away_ft)
    victim_slot = np.argmin(off_ft + rng.random((n, 5)) * 0.015, axis=1)
    victim = _pick(
        off_home,
        state.home_lineup[rows, victim_slot],
        state.away_lineup[rows, victim_slot],
    )
    fouler = _pick(
        off_home,
        state.away_lineup[rows, fouler_slot],
        state.home_lineup[rows, fouler_slot],
    )
    seconds_left = np.maximum(0.0, state.qclock - 1.4)
    kind = np.full(n, INTENTIONAL)
    fta, team_foul, keep, live = free_throw_award(
        kind, np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), _pick(off_home, state.away_tf, state.home_tf), bonus_line, seconds_left
    )
    fta = np.where(hack, fta, 0)
    home_victim = state.home_lineup[rows, victim_slot]
    away_victim = state.away_lineup[rows, victim_slot]
    ft_pct = np.where(off_home, home.ft_pct[home_victim], away.ft_pct[away_victim])
    _apply_foul_result(
        rng, state, home, away, home_box, away_box, hack, off_home, victim, fouler, actor,
        ft_pct, fta, team_foul, keep, live, False, oreb_home, oreb_away, ended_if_flip=True,
    )
    state.intentional += hack


def _resolve_trip(
    rng: np.random.Generator,
    state: GameState,
    home: Side,
    away: Side,
    home_box: Tables,
    away_box: Tables,
    view: dict[str, np.ndarray],
    normal: np.ndarray,
    actor: np.ndarray,
    defender: np.ndarray,
    off_home: np.ndarray,
    raw: np.ndarray,
    assist: np.ndarray,
    bonus_line: int,
    oreb_home: float,
    oreb_away: float,
    period: int = 1,
    deficit: np.ndarray | None = None,
) -> None:
    n = state.n
    u_end = rng.random(n)
    # One draw. Offensive foul, turnover, shooting foul, non-shooting foul, shot.
    c_off = view["p_off"]
    c_tov = c_off + view["p_tov"]
    c_sfoil = c_tov + view["p_sfoil"]
    c_nsf = c_sfoil + view["p_nsf"]
    is_off = normal & (u_end < c_off)
    is_tov = normal & (u_end >= c_off) & (u_end < c_tov)
    is_sfoil = normal & (u_end >= c_tov) & (u_end < c_sfoil)
    is_nsf = normal & (u_end >= c_sfoil) & (u_end < c_nsf)
    is_shot = normal & (u_end >= c_nsf)
    # A charge is a turnover and cannot be a steal. Live balls take this share
    # of non-charge turnovers (post-eval steal share was a touch low at 0.58).
    live_share = np.clip(_STEAL_SHARE * (view["p_tov"] + view["p_off"]) / np.maximum(view["p_tov"], 1e-6), 0.0, 0.92)
    is_steal = is_tov & (rng.random(n) < live_share)
    is_dead = is_tov & ~is_steal
    # A take foul is a late stop-the-clock reach-in in transition. Every other
    # transition reach-in is a common foul: two free throws and the ball once
    # the bonus is on, so the trip can end without another field-goal attempt.
    fouls_for_take = _pick(off_home, state.away_tf, state.home_tf).astype(float)
    lead = np.zeros(n) if deficit is None else np.asarray(deficit, dtype=float)
    late_take = trailing_mask(period, state.qclock, lead, fouls_for_take, bonus_line)
    is_take = is_nsf & (view["type_idx"] == TRANSITION) & late_take

    buzzer = normal & (raw + 0.05 >= state.qclock)
    in_time = ~buzzer | is_shot | (rng.random(n) < 0.65)
    is_nsf = is_nsf & in_time
    is_off = is_off & in_time
    is_tov = is_tov & in_time
    is_steal = is_steal & in_time
    is_dead = is_dead & in_time
    is_sfoil = is_sfoil & in_time
    is_take = is_take & in_time
    violation = is_shot & ~buzzer & (raw > SHOT_CLOCK) & (rng.random(n) < np.clip((raw - SHOT_CLOCK) / 6.0, 0.0, 0.8))
    is_shot = is_shot & ~violation
    is_dead = is_dead | violation
    # The shot bucket is a field goal. Attempts are not thrown away to force
    # the possession back onto a stacked price.

    # The same draw says whether the attempt, made or missed, is a three.
    is_three_try = (is_shot | is_sfoil) & (rng.random(n) < view["three_p"])
    make_rate = np.where(is_three_try, view["three_fg"], view["two_fg"])
    make_rate = np.where(buzzer & (state.qclock < 4.0), make_rate * np.clip(state.qclock / 4.0, 0.15, 1.0), make_rate)
    is_make = is_shot & (rng.random(n) < make_rate)
    is_miss = is_shot & ~is_make
    shot_pts = np.where(is_make, np.where(is_three_try, 3.0, 2.0), 0.0)

    # A flagrant replaces the ordinary whistle: two shots and the ball, and
    # the field goal does not also count. The reckless ones send him out.
    personal = is_sfoil | (is_nsf & ~is_take)
    p_flag = np.clip(_FLAG_RATE * view["def_foul"] * _FLAG_PLAY[view["type_idx"]], 0.001, 0.028)
    is_flag = personal & (rng.random(n) < p_flag)
    is_eject = is_flag & (rng.random(n) < _FLAG_EJECT)
    is_sfoil = is_sfoil & ~is_flag
    is_nsf = is_nsf & ~is_flag
    is_take = is_take & ~is_flag

    is_three_foul = is_sfoil & is_three_try
    # The foul and the make are the same shot. His make rate is the and-1.
    and1 = is_sfoil & (rng.random(n) < make_rate)
    and1_pts = np.where(and1, np.where(is_three_foul, 3.0, 2.0), 0.0)

    time_used = np.minimum(np.maximum(raw, 3.0), SHOT_CLOCK)
    time_used = np.where(buzzer, state.qclock, time_used)
    # A trip that ends spends the length that was drawn. Giving the leftover
    # clock back is how a foul manufactured another possession. A foul that
    # keeps the ball spends half of this draw, and the same trip finishes
    # with what is left. Free throws are dead time and do not run the clock.
    fouls_before = _pick(off_home, state.away_tf, state.home_tf)
    keeps = is_nsf & (is_take | (fouls_before + 1.0 < float(bonus_line)))
    time_used = np.where(keeps, np.minimum(np.maximum(raw * 0.5, 1.0), state.qclock), time_used)
    time_used = np.where(is_sfoil | is_off | is_flag, np.minimum(np.maximum(raw, 2.0), state.qclock), time_used)
    state.pending_time = np.where(normal, time_used, 0.0)
    seconds_left = np.maximum(0.0, state.qclock - time_used)

    fouls_before = _pick(off_home, state.away_tf, state.home_tf)
    kind = np.full(n, -1)
    kind = np.where(is_sfoil, SHOT, kind)
    kind = np.where(is_take, TAKE, kind)
    kind = np.where(is_nsf & ~is_take, COMMON, kind)
    kind = np.where(is_off, OFFENSIVE, kind)
    kind = np.where(is_flag, FLAGRANT, kind)
    fta, team_foul, keep, live = free_throw_award(
        np.where(kind < 0, OFFENSIVE, kind),
        is_three_foul,
        and1,
        fouls_before,
        bonus_line,
        seconds_left,
    )
    whistle = is_sfoil | is_nsf | is_off | is_flag
    fta = np.where(whistle, fta, 0)

    _add(home_box.pts, actor, shot_pts + and1_pts, off_home & (is_make | and1))
    _add(away_box.pts, actor, shot_pts + and1_pts, ~off_home & (is_make | and1))
    state.second_home += np.where(state.was_follow & off_home, shot_pts + and1_pts, 0.0)
    state.second_away += np.where(state.was_follow & ~off_home, shot_pts + and1_pts, 0.0)
    made_three = (is_make & is_three_try) | (and1 & is_three_foul)
    miss_three = is_miss & is_three_try
    _add(home_box.tpm, actor, np.ones(n), off_home & made_three)
    _add(away_box.tpm, actor, np.ones(n), ~off_home & made_three)
    _add(home_box.tpa, actor, np.ones(n), off_home & (made_three | miss_three))
    _add(away_box.tpa, actor, np.ones(n), ~off_home & (made_three | miss_three))
    made_fg = is_make | and1
    attempt = made_fg | is_miss
    _add(home_box.fgm, actor, np.ones(n), off_home & made_fg)
    _add(away_box.fgm, actor, np.ones(n), ~off_home & made_fg)
    _add(home_box.fga, actor, np.ones(n), off_home & attempt)
    _add(away_box.fga, actor, np.ones(n), ~off_home & attempt)
    # A shooting-foul miss is free throws only in the box. And-1 makes already
    # sit in attempt via made_fg; do not credit a second FGA/TPA here.
    _add(home_box.tov, actor, np.ones(n), off_home & (is_tov | is_off | violation))
    _add(away_box.tov, actor, np.ones(n), ~off_home & (is_tov | is_off | violation))

    _apply_foul_result(
        rng, state, home, away, home_box, away_box, whistle, off_home, actor, defender, actor,
        view["ft_pct"], fta, team_foul, keep, live, is_off, oreb_home, oreb_away, ended_if_flip=False,
    )
    _eject(away_box.pf, defender, is_eject & off_home)
    _eject(home_box.pf, defender, is_eject & ~off_home)
    state.flagrant += is_flag.astype(float)
    state.ejected += is_eject.astype(float)

    # Field-goal miss, block, rebound, putback. Free-throw rebounds are inside _apply_foul_result.
    blk_w_home = home.blk[state.home_lineup]
    blk_w_away = away.blk[state.away_lineup]
    blk_w = _pick(off_home, blk_w_away, blk_w_home)
    # A block is a two-point miss. After the fouled-miss FGA fix, 0.16 sat
    # ~0.6 blocks high on the locked sample; 0.14 tracks the box near 4.7.
    blk_rate = blk_w.mean(axis=1) / 0.5
    is_blk = is_miss & ~is_three_try & (rng.random(n) < np.clip(0.14 * blk_rate, 0.0, 1.0))
    blk_slot = _sample_rows(rng, np.maximum(blk_w, 1e-6))
    blk_player = _pick(off_home, state.away_lineup[np.arange(n), blk_slot], state.home_lineup[np.arange(n), blk_slot])
    _add(away_box.blk, blk_player, np.ones(n), is_blk & off_home)
    _add(home_box.blk, blk_player, np.ones(n), is_blk & ~off_home)

    oreb_w = _pick(off_home, home.oreb[state.home_lineup], away.oreb[state.away_lineup])
    reb_slot = _sample_rows(rng, oreb_w)
    home_reb = state.home_lineup[np.arange(n), reb_slot]
    away_reb = state.away_lineup[np.arange(n), reb_slot]
    reb_player = np.where(off_home, home_reb, away_reb)
    # The five on the floor sets the chance. The scripted unit only opened the segment.
    home_chance = _on_floor_oreb(home, away, state.home_lineup, state.away_lineup)
    away_chance = _on_floor_oreb(away, home, state.away_lineup, state.home_lineup)
    p_oreb = np.where(
        is_blk,
        _pick(off_home, home_chance, away_chance) * 0.55,
        _pick(off_home, home_chance, away_chance),
    )
    p_oreb = _live_board(p_oreb, state.was_follow)
    # Same live-clock gate for both boards. A horn/heave miss with no time
    # left is a team rebound, not a player DREB.
    live_glass = seconds_left > 0.2
    is_oreb = is_miss & live_glass & (rng.random(n) < p_oreb)
    _add(home_box.oreb, reb_player, np.ones(n), is_oreb & off_home)
    _add(away_box.oreb, reb_player, np.ones(n), is_oreb & ~off_home)
    _add(home_box.reb, reb_player, np.ones(n), is_oreb & off_home)
    _add(away_box.reb, reb_player, np.ones(n), is_oreb & ~off_home)

    home_dreb = _dreb_weights(
        home.dreb[state.home_lineup],
        home.pos[state.home_lineup],
        view["type_idx"],
        home.reb_share_allowed,
    )
    away_dreb = _dreb_weights(
        away.dreb[state.away_lineup],
        away.pos[state.away_lineup],
        view["type_idx"],
        away.reb_share_allowed,
    )
    dreb_w = _pick(off_home, away_dreb, home_dreb)
    dreb_slot = _sample_rows(rng, dreb_w)
    dreb_player = _pick(off_home, state.away_lineup[np.arange(n), dreb_slot], state.home_lineup[np.arange(n), dreb_slot])
    # Three-way: OREB / player DREB / team-reb (no player box). Team share is
    # the dead-ball and period-end boards the NBA box leaves uncredited.
    loose = is_miss & ~is_oreb
    is_team_reb = loose & ((~live_glass) | (rng.random(n) < _TEAM_REB_SHARE))
    is_dreb = loose & ~is_team_reb
    _add(away_box.reb, dreb_player, np.ones(n), is_dreb & off_home)
    _add(home_box.reb, dreb_player, np.ones(n), is_dreb & ~off_home)
    _add(away_box.dreb, dreb_player, np.ones(n), is_dreb & off_home)
    _add(home_box.dreb, dreb_player, np.ones(n), is_dreb & ~off_home)

    ast_pct = _pick(off_home, home.ast_pct[state.home_lineup], away.ast_pct[state.away_lineup])
    off_pos = _pick(off_home, home.pos[state.home_lineup], away.pos[state.away_lineup])
    # Defense that faced the shot sets pass volume and G/F/C passer tilt.
    ast_volume = np.where(off_home, away.ast_allowed, home.ast_allowed)
    home_ast_share = _pos_share_rows(home.pos[state.home_lineup], away.ast_share_allowed, _AST_SHARE_PRIOR)
    away_ast_share = _pos_share_rows(away.pos[state.away_lineup], home.ast_share_allowed, _AST_SHARE_PRIOR)
    pass_share = _pick(off_home, home_ast_share, away_ast_share)
    actor_slot = view["actor_slot"]
    glass = _pick(off_home, state.glass_home, state.glass_away)
    # An and-1 is a made field goal. The same pass that created a miss is a potential assist.
    passed, ast_player = _pass_chance(
        rng, ast_pct, _pick(off_home, state.home_lineup, state.away_lineup),
        actor_slot, actor, view["type_idx"], assist, state.was_follow, glass,
        pos=off_pos,
        ast_allowed=ast_volume,
        ast_share=pass_share,
    )
    is_ast = made_fg & passed & (ast_player != actor)
    is_pot = is_ast | (is_miss & passed & (ast_player != actor))
    _add(home_box.ast, ast_player, np.ones(n), is_ast & off_home)
    _add(away_box.ast, ast_player, np.ones(n), is_ast & ~off_home)
    _add(home_box.pot_ast, ast_player, np.ones(n), is_pot & off_home)
    _add(away_box.pot_ast, ast_player, np.ones(n), is_pot & ~off_home)

    stl_w = _pick(off_home, away.stl[state.away_lineup], home.stl[state.home_lineup])
    stl_slot = _sample_rows(rng, stl_w)
    stl_player = _pick(off_home, state.away_lineup[np.arange(n), stl_slot], state.home_lineup[np.arange(n), stl_slot])
    _add(away_box.stl, stl_player, np.ones(n), is_steal & off_home)
    _add(home_box.stl, stl_player, np.ones(n), is_steal & ~off_home)

    # A missed shot or a live free throw that the offense rebounds keeps the
    # possession. The next trip is the putback or the kick-out. An and-1 whose
    # free throw is rebounded ends the scoring possession and starts another.
    ft_oreb = state.pending_ft_oreb
    ft_reb = state.pending_ft_reb
    shot_board = is_oreb
    ft_board = ft_oreb & ~and1
    and1_board = and1 & ft_oreb
    continues = (shot_board | ft_board | and1_board) & normal
    # The horn beat the action, so the trip ends with no shot and no whistle.
    is_empty = normal & ~(is_off | is_tov | is_sfoil | is_nsf | is_shot | is_flag | violation)
    ball_changes = (is_shot & ~is_oreb) | is_empty | is_tov | is_off | violation | (whistle & ~keep & ~ft_oreb)
    counts = ball_changes | and1_board
    reb = np.where(shot_board, reb_player, ft_reb)
    state.home_poss += counts & off_home
    state.away_poss += counts & ~off_home
    state.offense_home = np.where(ball_changes, ~off_home, state.offense_home)
    state.retain = np.where(normal & whistle & keep, state.retain + 1, state.retain)
    state.retain = np.where(normal & ball_changes, 0, state.retain)
    # The whistle that keeps the ball spent this much of the possession.
    # A rebound resets the shot clock, so the follow-up starts clean.
    carry = normal & whistle & keep & ~continues
    state.spent = np.where(carry, state.spent + time_used, np.where(normal, 0.0, state.spent))
    state.follow = np.where(normal, continues, state.follow)
    state.glass_home = np.where(continues & off_home, reb, state.glass_home)
    state.glass_away = np.where(continues & ~off_home, reb, state.glass_away)
    state.next_live = np.where(is_steal, True, state.next_live)
    state.next_live = np.where(continues, False, state.next_live)
    flip_arrow = is_dead & (rng.random(n) < 0.10)
    state.arrow_home = np.where(flip_arrow, ~state.arrow_home, state.arrow_home)


def _apply_foul_result(
    rng: np.random.Generator,
    state: GameState,
    home: Side,
    away: Side,
    home_box: Tables,
    away_box: Tables,
    mask: np.ndarray,
    off_home: np.ndarray,
    shooter: np.ndarray,
    fouler: np.ndarray,
    offender: np.ndarray,
    ft_pct: np.ndarray,
    fta: np.ndarray,
    team_foul: np.ndarray,
    keep: np.ndarray,
    live: np.ndarray,
    offensive_on: np.ndarray,
    oreb_home: float,
    oreb_away: float,
    ended_if_flip: bool,
) -> None:
    n = state.n
    # The fouler is on the defense. The offensive foul is charged to the player with the ball.
    _add(away_box.pf, fouler, np.ones(n), mask & team_foul & off_home)
    _add(home_box.pf, fouler, np.ones(n), mask & team_foul & ~off_home)
    _add(home_box.pf, offender, np.ones(n), mask & offensive_on & off_home)
    _add(away_box.pf, offender, np.ones(n), mask & offensive_on & ~off_home)
    _charge(state, off_home, team_foul, mask)

    ftm, missed_last = _shoot_fts(rng, np.where(mask, fta, 0), ft_pct)
    _add(home_box.pts, shooter, ftm, mask & off_home)
    _add(away_box.pts, shooter, ftm, mask & ~off_home)
    was = getattr(state, "was_follow", None)
    if was is not None:
        state.second_home += np.where(was & mask & off_home, ftm, 0.0)
        state.second_away += np.where(was & mask & ~off_home, ftm, 0.0)
    _add(home_box.ftm, shooter, ftm, mask & off_home)
    _add(away_box.ftm, shooter, ftm, mask & ~off_home)
    _add(home_box.fta, shooter, fta.astype(float), mask & off_home)
    _add(away_box.fta, shooter, fta.astype(float), mask & ~off_home)
    trips = mask & (fta > 0)
    state.home_ft_trips += trips & off_home
    state.away_ft_trips += trips & ~off_home

    oreb_w = _pick(off_home, home.oreb[state.home_lineup], away.oreb[state.away_lineup])
    reb_slot = _sample_rows(rng, np.maximum(oreb_w, 1e-6))
    home_reb = state.home_lineup[np.arange(n), reb_slot]
    away_reb = state.away_lineup[np.arange(n), reb_slot]
    reb_player = np.where(off_home, home_reb, away_reb)
    home_chance = _on_floor_oreb(home, away, state.home_lineup, state.away_lineup)
    away_chance = _on_floor_oreb(away, home, state.away_lineup, state.home_lineup)
    p_oreb = _pick(off_home, home_chance, away_chance)
    follow = was if isinstance(was, np.ndarray) else np.zeros(n, dtype=bool)
    p_oreb = _live_board(p_oreb, follow)
    is_oreb = trips & live & missed_last & (rng.random(n) < p_oreb)
    state.pending_ft_oreb = is_oreb
    state.pending_ft_reb = reb_player
    _add(home_box.oreb, reb_player, np.ones(n), is_oreb & off_home)
    _add(away_box.oreb, reb_player, np.ones(n), is_oreb & ~off_home)
    _add(home_box.reb, reb_player, np.ones(n), is_oreb & off_home)
    _add(away_box.reb, reb_player, np.ones(n), is_oreb & ~off_home)
    # Free-throw misses have no play-type signal; use Misc tilt row.
    ft_type = np.full(n, len(PLAY_TYPES) - 1, dtype=int)
    home_ft_dreb = _dreb_weights(
        home.dreb[state.home_lineup], home.pos[state.home_lineup], ft_type, home.reb_share_allowed,
    )
    away_ft_dreb = _dreb_weights(
        away.dreb[state.away_lineup], away.pos[state.away_lineup], ft_type, away.reb_share_allowed,
    )
    dreb_w = _pick(off_home, away_ft_dreb, home_ft_dreb)
    dreb_slot = _sample_rows(rng, np.maximum(dreb_w, 1e-6))
    dreb_player = _pick(off_home, state.away_lineup[np.arange(n), dreb_slot], state.home_lineup[np.arange(n), dreb_slot])
    loose = trips & live & missed_last & ~is_oreb
    is_dreb = loose & (rng.random(n) >= _TEAM_REB_SHARE)
    _add(away_box.reb, dreb_player, np.ones(n), is_dreb & off_home)
    _add(home_box.reb, dreb_player, np.ones(n), is_dreb & ~off_home)
    _add(away_box.dreb, dreb_player, np.ones(n), is_dreb & off_home)
    _add(home_box.dreb, dreb_player, np.ones(n), is_dreb & ~off_home)

    if ended_if_flip:
        continues = mask & is_oreb
        ends = mask & ~keep & ~is_oreb
        state.home_poss += ends & off_home
        state.away_poss += ends & ~off_home
        state.offense_home = np.where(ends, ~off_home, state.offense_home)
        state.retain = np.where(mask & keep, state.retain + 1, np.where(ends, 0, state.retain))
        state.next_live = np.where(mask, False, state.next_live)
        state.follow = np.where(mask, continues, state.follow)
        state.glass_home = np.where(continues & off_home, reb_player, state.glass_home)
        state.glass_away = np.where(continues & ~off_home, reb_player, state.glass_away)


def _segment_scale(spec: GameSpec, home_unit: list[int] | np.ndarray, away_unit: list[int] | np.ndarray) -> float:
    pace = _pace(
        spec.home,
        spec.away,
        _unit_ids(spec.home_players, home_unit),
        _unit_ids(spec.away_players, away_unit),
        spec.context.pace_mult,
    )
    # TEMPO is applied inside the draw as a fraction of a 100-pace game.
    # Scale that draw so this lineup's mean length matches its own pace.
    return 100.0 / pace


def _mean_lineup_oreb(team: TeamModel) -> float:
    book = getattr(team, "lineup_oreb", None) or {}
    values = [float(v) for v in book.values() if isinstance(v, (int, float)) and 0.05 < float(v) < 0.55]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _on_floor_oreb(offense: Side, defense: Side, off_lineup: np.ndarray, def_lineup: np.ndarray) -> np.ndarray:
    """Chance the offense keeps the miss, from the five actually on the floor."""
    team_oreb = float(offense.team_oreb)
    # Mild trust in the five-man OREB book beside player sums + team rates.
    lineup_mean = float(getattr(offense, "lineup_oreb_mean", 0.0) or 0.0)
    if 0.12 < lineup_mean < 0.45:
        team_oreb = 0.85 * team_oreb + 0.15 * lineup_mean
    return lineup_oreb(
        offense.oreb[off_lineup].sum(axis=1),
        defense.dreb[def_lineup].sum(axis=1),
        offense.box[off_lineup].sum(axis=1),
        defense.box[def_lineup].sum(axis=1),
        team_oreb,
        defense.team_dreb,
        defense.team_oreb_allowed,
    )


def _oreb(offense: list[PlayerModel], defense: list[PlayerModel], off_team: TeamModel, def_team: TeamModel) -> float:
    return oreb_probability(offense, defense, off_team.oreb_pct, def_team.dreb_pct, def_team.oreb_allowed)


def _nominal(players: list[PlayerModel], unit: list[int]) -> list[PlayerModel]:
    return [players[index] for index in unit]


def play_game(
    rng: np.random.Generator,
    spec: GameSpec,
    home_units: dict[str, list[int]] | None,
    away_units: dict[str, list[int]] | None,
    home_draw: np.ndarray,
    away_draw: np.ndarray,
    home_residual: float,
    away_residual: float,
    home_targets: np.ndarray | None = None,
    away_targets: np.ndarray | None = None,
    home_night: np.ndarray | None = None,
    away_night: np.ndarray | None = None,
) -> ClockGame:
    n = home_draw.shape[0]
    home = _side(spec.home_players, home_draw)
    away = _side(spec.away_players, away_draw)
    home.night = np.ones(n, dtype=float) if home_night is None else np.asarray(home_night, dtype=float).reshape(n)
    away.night = np.ones(n, dtype=float) if away_night is None else np.asarray(away_night, dtype=float).reshape(n)
    home_box = _tables(n, len(spec.home_players))
    away_box = _tables(n, len(spec.away_players))
    tip_home = rng.random(n) < 0.53
    if _court_game(spec):
        # Profiles set the five at the first whistle. The scripted units are
        # not a second rotation waiting underneath.
        home_lineup = np.zeros((n, 5), dtype=int)
        away_lineup = np.zeros((n, 5), dtype=int)
    else:
        home_lineup = _broadcast(home_units["starters"], n)
        away_lineup = _broadcast(away_units["starters"], n)
    state = GameState(
        n=n,
        qclock=np.zeros(n),
        segment_left=np.zeros(n),
        offense_home=tip_home.copy(),
        arrow_home=~tip_home,
        live_next=np.zeros(n, dtype=bool),
        retain=np.zeros(n),
        home_tf=np.zeros(n),
        away_tf=np.zeros(n),
        home_game_fouls=np.zeros(n),
        away_game_fouls=np.zeros(n),
        home_max_q=np.zeros(n),
        away_max_q=np.zeros(n),
        home_to=np.full(n, 7.0),
        away_to=np.full(n, 7.0),
        home_lineup=home_lineup,
        away_lineup=away_lineup,
        home_ft_trips=np.zeros(n),
        away_ft_trips=np.zeros(n),
        home_poss=np.zeros(n),
        away_poss=np.zeros(n),
        intentional=np.zeros(n),
        flagrant=np.zeros(n),
        ejected=np.zeros(n),
        second_home=np.zeros(n),
        second_away=np.zeros(n),
    )
    home.fta_allowed = float(spec.home.fta_allowed)
    away.fta_allowed = float(spec.away.fta_allowed)
    home.team_oreb = float(spec.home.oreb_pct)
    home.team_dreb = float(spec.home.dreb_pct)
    home.team_oreb_allowed = float(spec.home.oreb_allowed)
    away.team_oreb = float(spec.away.oreb_pct)
    away.team_dreb = float(spec.away.dreb_pct)
    away.team_oreb_allowed = float(spec.away.oreb_allowed)
    home.ast_allowed = float(getattr(spec.home, "ast_allowed", 1.0) or 1.0)
    away.ast_allowed = float(getattr(spec.away, "ast_allowed", 1.0) or 1.0)
    home.reb_share_allowed = dict(getattr(spec.home, "reb_share_allowed", None) or _REB_SHARE_PRIOR)
    away.reb_share_allowed = dict(getattr(spec.away, "reb_share_allowed", None) or _REB_SHARE_PRIOR)
    home.ast_share_allowed = dict(getattr(spec.home, "ast_share_allowed", None) or _AST_SHARE_PRIOR)
    away.ast_share_allowed = dict(getattr(spec.away, "ast_share_allowed", None) or _AST_SHARE_PRIOR)
    home.lineup_oreb_mean = _mean_lineup_oreb(spec.home)
    away.lineup_oreb_mean = _mean_lineup_oreb(spec.away)
    state.pending_time = np.zeros(n)
    state.next_live = np.zeros(n, dtype=bool)
    state.home_floor = np.zeros((n, home_box.pts.shape[1]))
    state.away_floor = np.zeros((n, away_box.pts.shape[1]))
    state.home_plan = np.zeros((n, home_box.pts.shape[1]))
    state.away_plan = np.zeros((n, away_box.pts.shape[1]))
    state.home_target = _target_sheet(spec.home_players, state.home_plan, home_targets)
    state.away_target = _target_sheet(spec.away_players, state.away_plan, away_targets)
    state.home_rested = np.full((n, home_box.pts.shape[1]), -1)
    state.away_rested = np.full((n, away_box.pts.shape[1]), -1)
    state.home_sat_who = np.full((n, 5), -1)
    state.away_sat_who = np.full((n, 5), -1)
    state.home_sat_back = np.zeros((n, 5))
    state.away_sat_back = np.zeros((n, 5))
    state.home_patience = float(getattr(spec.home, "foul_patience", 0.45) or 0.45)
    state.away_patience = float(getattr(spec.away, "foul_patience", 0.45) or 0.45)
    state.follow = np.zeros(n, dtype=bool)
    state.was_follow = np.zeros(n, dtype=bool)
    state.spent = np.zeros(n)
    state.glass_home = np.full(n, -1)
    state.glass_away = np.full(n, -1)
    state.pending_ft_oreb = np.zeros(n, dtype=bool)
    state.pending_ft_reb = np.full(n, -1)
    state.home_def_tov = _tov_vector(spec.home)
    state.away_def_tov = _tov_vector(spec.away)

    if _court_game(spec):
        state.court = True
        state.spec = spec
        _arm_court(home, spec.home)
        _arm_court(away, spec.away)
        margin, blowout = _play_profiles(
            rng, spec, state, home, away, home_box, away_box, home_residual, away_residual,
        )
    else:
        margin, blowout = _play_script(
            rng, spec, state, home, away, home_box, away_box,
            home_units, away_units, home_residual, away_residual,
        )

    return ClockGame(
        home=home_box,
        away=away_box,
        second_home=state.second_home,
        second_away=state.second_away,
        margin_before_q4=margin,
        home_team_fouls=state.home_game_fouls,
        away_team_fouls=state.away_game_fouls,
        home_max_quarter_fouls=state.home_max_q,
        away_max_quarter_fouls=state.away_max_q,
        home_ft_trips=state.home_ft_trips,
        away_ft_trips=state.away_ft_trips,
        home_possessions=state.home_poss,
        away_possessions=state.away_poss,
        intentional_fouls=state.intentional,
        flagrant_fouls=state.flagrant,
        ejections=state.ejected,
        blowout=blowout,
        home_floor=state.home_floor,
        away_floor=state.away_floor,
        home_plan=state.home_plan,
        away_plan=state.away_plan,
    )


def _charge_plan(plan: np.ndarray, unit: list[int], length: float, rows: np.ndarray | None = None) -> None:
    for index in dict.fromkeys(unit):
        if rows is None:
            plan[:, index] += length
        else:
            plan[rows, index] += length


def _start_period(state: GameState, seconds: float, use_arrow: bool) -> None:
    state.qclock[:] = seconds
    state.home_tf[:] = 0
    state.away_tf[:] = 0
    state.live_next[:] = False
    state.retain[:] = 0
    state.next_live[:] = False
    state.follow[:] = False
    state.spent[:] = 0
    if use_arrow:
        state.offense_home = state.arrow_home.copy()
        state.arrow_home = ~state.arrow_home


def _target_sheet(players: list[PlayerModel], plan: np.ndarray, target: np.ndarray | None) -> np.ndarray:
    """Minutes owed this night. Missing draws fall back to the season number."""
    if target is None:
        base = np.array([float(player.minutes_mean) for player in players], dtype=float)
        return np.repeat(base[None, :], plan.shape[0], axis=0)
    return np.asarray(target, dtype=float)


def _regulation_left(period: int, qclock) -> np.ndarray:
    """Minutes left in regulation, from the quarter clock."""
    remaining = np.asarray(qclock, dtype=float) / 60.0
    extra = {1: 36.0, 2: 24.0, 3: 12.0}.get(int(period), 0.0)
    return remaining + extra


def _rest_margin(hook: float, span_minutes: float, minutes_left) -> np.ndarray:
    """The lead that can sit a closer.

    Inside the finish window this is the coach's own hook. Each minute still
    to play before that window adds three-quarters of a point.
    """
    left = np.asarray(minutes_left, dtype=float)
    extra = np.maximum(0.0, left - float(span_minutes))
    return float(hook) + extra * _LEAD_PER_MINUTE


def _script_cap(
    sheet: np.ndarray,
    means: np.ndarray,
    played: np.ndarray,
    lead: np.ndarray,
    period: int,
    hook: float,
    span_minutes: float,
    minutes_left,
    segment_minutes: float,
) -> np.ndarray:
    """Bend the minute cap with the score.

    The first half stays on the draw. After that, a lead that has cleared the
    time-adjusted hook makes his closers eligible to sit. The team that is
    chasing, and both teams while the game is inside two scores, can keep a
    closer through the stint he is in, up to a 40-minute night. Past that bar
    by another score, the trailing closers sit too.
    """
    sheet = np.asarray(sheet, dtype=float)
    if int(period) < 3:
        return sheet
    means = np.asarray(means, dtype=float)
    played = np.asarray(played, dtype=float)
    lead = np.asarray(lead, dtype=float)
    closers = means >= _CLOSER_FLOOR
    if not bool(np.any(closers)):
        return sheet
    out = np.array(sheet, dtype=float, copy=True)
    room = np.minimum(_CLOSE_CEILING, np.maximum(sheet, played + max(float(segment_minutes), 0.0)))
    rest_at = _rest_margin(hook, span_minutes, minutes_left)
    chasing = (lead <= -rest_at) & (np.abs(lead) < rest_at + _CHASE_MARGIN)
    extend = (np.abs(lead) <= _CLOSE_MARGIN) | chasing
    out[:, closers] = np.where(extend[:, None], np.maximum(out[:, closers], room[:, closers]), out[:, closers])
    resting = (lead >= rest_at) | (np.abs(lead) >= rest_at + _CHASE_MARGIN)
    out[:, closers] = np.where(resting[:, None], np.minimum(out[:, closers], played[:, closers]), out[:, closers])
    return out


def _night_sheet(players, plan, target, lead, period, abbr, length, minutes_left) -> np.ndarray:
    """The draw, bent by the score once the second half is underway."""
    sheet = _target_sheet(players, plan, target)
    if int(period) < 3:
        return sheet
    means = np.array([float(player.minutes_mean) for player in players], dtype=float)
    played = np.asarray(plan, dtype=float) / 60.0
    span, hook = finish_window(coach_for(abbr or ""))
    return _script_cap(
        sheet,
        means,
        played,
        np.asarray(lead, dtype=float),
        int(period),
        float(hook),
        float(span) / 60.0,
        minutes_left,
        float(length) / 60.0,
    )


def _scripted_lineup(unit, players, plan, target, lead, period, abbr, length, stick: bool, minutes_left) -> np.ndarray:
    """Who plays this stint. A blowout finish keeps the unit it was given."""
    sheet = target if stick else _night_sheet(players, plan, target, lead, period, abbr, length, minutes_left)
    return _assign(list(unit), players, plan, sheet, stick=stick)


def _assign(
    unit: list[int],
    players: list[PlayerModel],
    plan: np.ndarray,
    target: np.ndarray | None,
    stick: bool,
) -> np.ndarray:
    """Who is on the floor in each simulation.

    The script names the group. A regular leaves that group in the sims where
    he has already played the minutes drawn for that night. The man who comes
    in is the same position when one is available, and among those the one who
    has played with the other four. Minutes break the tie. A spot minute waits
    behind every regular. A blowout finish keeps the reserves it was given.
    """
    base = np.asarray([int(index) for index in list(unit)[:5]], dtype=np.int64)
    chosen = np.tile(base, (plan.shape[0], 1))
    if stick or len(players) < 5:
        return chosen
    sheet = _target_sheet(players, plan, target)
    means = np.array([float(player.minutes_mean) for player in players], dtype=float)
    available = np.array(
        [bool(player.available) and float(player.minutes_mean) > 0.0 for player in players]
    )
    pos = np.array([_pos_code(player) for player in players], dtype=float)
    ids = {int(player.player_id): index for index, player in enumerate(players)}
    share = np.zeros((len(players), len(players)))
    for index, player in enumerate(players):
        for mate, played_with in player.shared_minutes.items():
            other = ids.get(int(mate))
            if other is not None:
                share[index, other] = float(played_with)
    order = list(np.argsort(-means))
    played = np.asarray(plan, dtype=float) / 60.0
    occupied = np.zeros((plan.shape[0], len(players)), dtype=bool)
    rows = np.arange(plan.shape[0])
    for slot in range(chosen.shape[1]):
        occupied[rows, chosen[:, slot]] = True
    for slot in range(chosen.shape[1]):
        current = chosen[:, slot]
        leave = (means[current] >= 12.0) & (played[rows, current] + 0.05 >= sheet[rows, current])
        if not np.any(leave):
            continue
        best = np.full(plan.shape[0], -1, dtype=np.int64)
        best_rank = np.full(plan.shape[0], np.inf)
        for candidate in order:
            if not bool(available[candidate]):
                continue
            ok = leave & ~occupied[:, candidate]
            if means[candidate] >= 12.0:
                ok &= played[:, candidate] + 0.05 < sheet[:, candidate]
            mates = np.zeros(plan.shape[0])
            for slot2 in range(chosen.shape[1]):
                if slot2 == slot:
                    continue
                mates += share[candidate, chosen[:, slot2]]
            distance = np.abs(pos[candidate] - pos[current])
            spot = np.where(means[candidate] < 6.0, 10.0, 0.0)
            rank = (distance + spot) * 1e6 - mates * 0.05 - means[candidate]
            better = ok & (rank < best_rank)
            best = np.where(better, candidate, best)
            best_rank = np.where(better, rank, best_rank)
        take = leave & (best >= 0)
        if not np.any(take):
            continue
        occupied[rows[take], current[take]] = False
        occupied[rows[take], best[take]] = True
        chosen[take, slot] = best[take]
    return chosen


def _within_minutes(
    unit: list[int],
    players: list[PlayerModel],
    plan: np.ndarray,
    target: np.ndarray | None = None,
) -> list[int]:
    """The next man takes a spot once this player has his minutes."""
    lineup = _assign(unit, players, plan, target, stick=False)
    return [int(index) for index in lineup[0]]


def _charge_lineup(plan: np.ndarray, lineup: np.ndarray, length: float, rows: np.ndarray | None = None) -> None:
    if rows is None:
        rows = np.arange(plan.shape[0])
    if rows.size == 0:
        return
    for slot in range(lineup.shape[1]):
        np.add.at(plan, (rows, lineup[rows, slot]), length)


def _play_named(
    rng, spec, state, home, away, home_box, away_box,
    home_unit, away_unit, home_residual, away_residual, length, period,
) -> None:
    margin = _score(home_box) - _score(away_box)
    left = _regulation_left(period, state.qclock)
    home_line = _scripted_lineup(
        home_unit, home.players, state.home_plan, state.home_target,
        margin, period, getattr(spec.home, "abbr", ""), length, False, left,
    )
    away_line = _scripted_lineup(
        away_unit, away.players, state.away_plan, state.away_target,
        -margin, period, getattr(spec.away, "abbr", ""), length, False, left,
    )
    state.home_lineup = home_line
    state.away_lineup = away_line
    state.home_sat_who[:] = -1
    state.away_sat_who[:] = -1
    state.segment_left[:] = np.minimum(length, state.qclock)
    if period <= 4:
        _charge_lineup(state.home_plan, home_line, length)
        _charge_lineup(state.away_plan, away_line, length)
    scale = np.full(state.n, _segment_scale(spec, list(home_unit), list(away_unit)))
    _play_scale(
        rng, spec, state, home, away, home_box, away_box,
        home_unit, away_unit, home_residual, away_residual, scale, period,
    )


def _fourth_marks(home_span: float, away_span: float) -> list[float]:
    """Elapsed seconds where a coach's finish can change who is on the floor."""
    marks = {0.0, QUARTER_SECONDS}
    for span in (float(home_span), float(away_span)):
        marks.add(min(QUARTER_SECONDS, max(0.0, QUARTER_SECONDS - span)))
    return sorted(marks)


def _reserves_enter(elapsed: float, span: float, margin: np.ndarray, hook: float) -> np.ndarray:
    """Sims where this coach's reserves take the rest of the fourth.

    The window is his own. A longer rotation opens it sooner and with a smaller lead.
    The lead is the absolute margin, so the trailing closers leave a decided game too.
    """
    margin = np.asarray(margin, dtype=float)
    if float(elapsed) + 1e-6 < QUARTER_SECONDS - float(span):
        return np.zeros(margin.shape, dtype=bool)
    return np.abs(margin) >= float(hook)


def _play_fourth(
    rng, spec, state, home, away, home_box, away_box,
    home_units, away_units, home_residual, away_residual,
) -> np.ndarray:
    """Closers start the fourth. Each coach's reserves enter only at his hook."""
    home_span, home_hook = finish_window(coach_for(getattr(spec.home, "abbr", "") or ""))
    away_span, away_hook = finish_window(coach_for(getattr(spec.away, "abbr", "") or ""))
    home_starters = home_units["starters"]
    away_starters = away_units["starters"]
    home_garbage = home_units.get("garbage") or home_units.get("bench_deep") or home_units["bench"]
    away_garbage = away_units.get("garbage") or away_units.get("bench_deep") or away_units["bench"]
    went = np.zeros(state.n, dtype=bool)
    elapsed = 0.0
    for end in _fourth_marks(home_span, away_span):
        if end <= elapsed + 1e-9:
            continue
        length = float(end - elapsed)
        margin = _score(home_box) - _score(away_box)
        home_sit = _reserves_enter(elapsed, home_span, margin, home_hook)
        away_sit = _reserves_enter(elapsed, away_span, margin, away_hook)
        went |= home_sit | away_sit
        # Closers are on the script for this stretch. Reserves take it only when the lead is safe.
        groups = (
            (~home_sit & ~away_sit, home_starters, away_starters, False, False),
            (home_sit & ~away_sit, home_garbage, away_starters, True, False),
            (~home_sit & away_sit, home_starters, away_garbage, False, True),
            (home_sit & away_sit, home_garbage, away_garbage, True, True),
        )
        for mask, home_unit, away_unit, home_stick, away_stick in groups:
            rows = np.flatnonzero(mask)
            if rows.size == 0:
                continue
            left = _regulation_left(4, state.qclock)
            home_line = _scripted_lineup(
                home_unit, home.players, state.home_plan, state.home_target,
                margin, 4, getattr(spec.home, "abbr", ""), length, home_stick, left,
            )
            away_line = _scripted_lineup(
                away_unit, away.players, state.away_plan, state.away_target,
                -margin, 4, getattr(spec.away, "abbr", ""), length, away_stick, left,
            )
            _charge_lineup(state.home_plan, home_line, length, rows)
            _charge_lineup(state.away_plan, away_line, length, rows)
            _play_masked(
                rng, spec, state, home, away, home_box, away_box,
                home_line, away_line, home_residual, away_residual, length, rows,
            )
        elapsed = end
    return went


def _play_masked(
    rng, spec, state, home, away, home_box, away_box,
    home_line, away_line, home_residual, away_residual, length, rows,
) -> None:
    """One lineup pair, on the sims in ``rows``, for ``length`` seconds.

    Matchup tables follow that pair. The other sims stay on the clock they already have.
    The reserves are not charged a script, so a blowout finish shows up as the minutes they played.
    """
    state.segment_left[:] = 0.0
    state.segment_left[rows] = np.minimum(length, state.qclock[rows])
    state.home_lineup[rows] = home_line[rows]
    state.away_lineup[rows] = away_line[rows]
    state.home_sat_who[rows] = -1
    state.away_sat_who[rows] = -1
    home_ids = [int(index) for index in home_line[int(rows[0])]]
    away_ids = [int(index) for index in away_line[int(rows[0])]]
    scale = np.full(state.n, _segment_scale(spec, home_ids, away_ids))
    _play_scale(
        rng, spec, state, home, away, home_box, away_box,
        home_ids, away_ids, home_residual, away_residual, scale, 4,
    )


def _play_scale(
    rng, spec, state, home, away, home_box, away_box,
    home_unit, away_unit, home_residual, away_residual, scale, period,
) -> None:
    home_names = _nominal(spec.home_players, home_unit)
    away_names = _nominal(spec.away_players, away_unit)
    away_def_mult, home_zone, home_pair = _matchup_vectors(spec.home_players, away_names, spec.away, spec.league)
    home_def_mult, away_zone, away_pair = _matchup_vectors(spec.away_players, home_names, spec.home, spec.league)
    _run_segment(
        rng,
        state,
        home,
        away,
        home_box,
        away_box,
        home_def_mult,
        away_def_mult,
        home_zone,
        away_zone,
        home_pair,
        away_pair,
        spec.context.home_ppp_mult,
        spec.context.away_ppp_mult,
        home_residual,
        away_residual,
        scale,
        _oreb(home_names, away_names, spec.home, spec.away),
        _oreb(away_names, home_names, spec.away, spec.home),
        period,
        bonus_on(period),
    )


def _play_script(
    rng, spec, state, home, away, home_box, away_box,
    home_units, away_units, home_residual, away_residual,
):
    """The named starter and bench stretches, for a game with no court profiles."""
    n = state.n
    deep_bench = (
        list(home_units["bench"]) != list(home_units.get("bench_deep", home_units["bench"]))
        or list(away_units["bench"]) != list(away_units.get("bench_deep", away_units["bench"]))
    )
    second_quarter = (
        [(300.0, "bench", "bench"), (120.0, "bench_deep", "bench_deep")]
        if deep_bench
        else [(420.0, "bench", "bench")]
    )
    quarters = [
        (1, [(480.0, "starters", "starters"), (240.0, "mixed_early", "mixed_early")], False),
        (2, second_quarter + [(300.0, "starters", "starters")], True),
        (3, [(420.0, "starters", "starters"), (300.0, "mixed_late", "mixed_late")], True),
    ]
    for period, segments, use_arrow in quarters:
        _start_period(state, QUARTER_SECONDS, use_arrow)
        for length, home_kind, away_kind in segments:
            _play_named(
                rng, spec, state, home, away, home_box, away_box,
                home_units[home_kind], away_units[away_kind],
                home_residual, away_residual, length, period,
            )
        _close_quarter(state)

    margin = _score(home_box) - _score(away_box)
    _start_period(state, QUARTER_SECONDS, True)
    blowout = _play_fourth(
        rng, spec, state, home, away, home_box, away_box,
        home_units, away_units, home_residual, away_residual,
    )
    _close_quarter(state)

    ot = 0
    while ot < 4 and np.any(np.abs(_score(home_box) - _score(away_box)) < 0.1):
        ot += 1
        tied = np.abs(_score(home_box) - _score(away_box)) < 0.1
        state.home_tf[:] = 0
        state.away_tf[:] = 0
        state.qclock[:] = np.where(tied, OT_SECONDS, 0.0)
        state.segment_left[:] = state.qclock.copy()
        state.live_next[:] = False
        state.retain[:] = 0
        state.follow[:] = False
        state.spent[:] = 0
        state.home_to = np.where(tied, state.home_to + 2.0, state.home_to)
        state.away_to = np.where(tied, state.away_to + 2.0, state.away_to)
        jump = rng.random(n) < 0.53
        state.offense_home = np.where(tied, jump, state.offense_home)
        state.home_lineup = _broadcast(home_units["starters"], n)
        state.away_lineup = _broadcast(away_units["starters"], n)
        _play_named(
            rng, spec, state, home, away, home_box, away_box,
            home_units["starters"], away_units["starters"],
            home_residual, away_residual, OT_SECONDS, 4 + ot,
        )
        _close_quarter(state)
    return margin, blowout


def _court_game(spec: GameSpec) -> bool:
    return any(getattr(player, "court", None) is not None for player in list(spec.home_players) + list(spec.away_players))


def _arm_court(side: Side, team: TeamModel) -> None:
    side.court_on = True
    side.squads = getattr(team, "squads", None) or {}
    side.team_off = float(getattr(team, "off_rating", 114.0) or 114.0)
    side.team_def = float(getattr(team, "def_rating", 114.0) or 114.0)
    side.team_pace = float(getattr(team, "pace", 99.5) or 99.5)


def _group_ids(lineup: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(lineup, dtype=np.int64)
    void = packed.view(np.dtype((np.void, packed.shape[1] * packed.dtype.itemsize))).reshape(-1)
    _unique, inverse = np.unique(void, return_inverse=True)
    return inverse


def _whistle_marks(players: list[PlayerModel]) -> list[float]:
    marks = {0.0, 180.0, 360.0, 540.0, 720.0}
    profiled = False
    for player in players:
        profile = getattr(player, "court", None)
        if profile is None:
            continue
        profiled = True
        for value in (profile.sit_remaining, profile.return_remaining, profile.blowout_remaining):
            number = float(value or 0.0)
            if 0.0 < number < 720.0:
                marks.add(number)
    if profiled:
        marks.update(float(second) for second in range(0, 721, 60))
    return sorted(marks, reverse=True)


def _lineups_for(
    players: list[PlayerModel],
    lead: np.ndarray,
    active,
    period: int,
    seconds_left: float,
    played: np.ndarray | None = None,
) -> np.ndarray:
    n = int(lead.shape[0])
    out = np.zeros((n, 5), dtype=np.int64)
    if not players:
        return out
    states = np.where(lead >= 15.0, 1, np.where(lead <= -18.0, -1, 0)).astype(np.int64)
    if active is None:
        codes = np.zeros(n, dtype=np.int64)
    else:
        width = min(int(active.shape[1]), 62)
        weights = (1 << np.arange(width)).astype(np.int64)
        codes = (np.asarray(active[:, :width], dtype=np.int64) * weights).sum(axis=1)
    columns = [states.reshape(-1, 1), codes.reshape(-1, 1)]
    floor = None
    if played is not None and int(played.shape[1]) == len(players):
        floor = np.asarray(played, dtype=float)
        columns.append(np.rint(floor / 60.0).astype(np.int64))
    packed = np.ascontiguousarray(np.concatenate(columns, axis=1))
    void = packed.view(np.dtype((np.void, packed.shape[1] * packed.dtype.itemsize))).reshape(-1)
    _unique, inverse = np.unique(void, return_inverse=True)
    for group in range(int(inverse.max()) + 1):
        row = int(np.flatnonzero(inverse == group)[0])
        lead_value = 20.0 if states[row] == 1 else (-20.0 if states[row] == -1 else 0.0)
        active_ids = None
        if active is not None:
            active_ids = {int(players[index].player_id) for index in range(len(players)) if active[row, index]}
        played_seconds = None
        if floor is not None:
            played_seconds = {int(players[index].player_id): float(floor[row, index]) for index in range(len(players))}
        chosen = select_lineup(players, lead_value, period, float(seconds_left), active_ids, played_seconds)
        if not chosen:
            chosen = [0]
        while len(chosen) < 5:
            chosen.append(chosen[-1])
        out[inverse == group] = chosen[:5]
    return out


def _play_court_stretch(rng, spec, state, home, away, home_box, away_box, home_line, away_line, home_residual, away_residual, length, period) -> None:
    state.segment_left[:] = np.minimum(length, state.qclock)
    state.home_lineup = home_line
    state.away_lineup = away_line
    state.home_sat_who[:] = -1
    state.away_sat_who[:] = -1
    home_ids = [int(index) for index in home_line[0]]
    away_ids = [int(index) for index in away_line[0]]
    scale = np.full(state.n, _segment_scale(spec, home_ids, away_ids))
    _play_scale(
        rng, spec, state, home, away, home_box, away_box,
        home_ids, away_ids, home_residual, away_residual, scale, period,
    )


def _play_profiles(rng, spec, state, home, away, home_box, away_box, home_residual, away_residual):
    """Whistles from the profiles. The next trip is the court those whistles left."""
    n = state.n
    home_active = getattr(spec, "home_active", None)
    away_active = getattr(spec, "away_active", None)
    marks = _whistle_marks(list(spec.home_players) + list(spec.away_players))
    margin = np.zeros(n)
    for period in (1, 2, 3, 4):
        _start_period(state, QUARTER_SECONDS, period > 1)
        for start, end in zip(marks[:-1], marks[1:]):
            length = float(start - end)
            if length <= 1e-6:
                continue
            margin = _score(home_box) - _score(away_box)
            home_line = _lineups_for(spec.home_players, margin, home_active, period, start, state.home_floor)
            away_line = _lineups_for(spec.away_players, -margin, away_active, period, start, state.away_floor)
            _charge_lineup(state.home_plan, home_line, length)
            _charge_lineup(state.away_plan, away_line, length)
            _play_court_stretch(
                rng, spec, state, home, away, home_box, away_box,
                home_line, away_line, home_residual, away_residual, length, period,
            )
        _close_quarter(state)
    margin = _score(home_box) - _score(away_box)
    blowout = np.abs(margin) >= 15.0
    ot = 0
    while ot < 4 and np.any(np.abs(_score(home_box) - _score(away_box)) < 0.1):
        ot += 1
        tied = np.abs(_score(home_box) - _score(away_box)) < 0.1
        state.home_tf[:] = 0
        state.away_tf[:] = 0
        state.qclock[:] = np.where(tied, OT_SECONDS, 0.0)
        state.segment_left[:] = state.qclock.copy()
        state.live_next[:] = False
        state.retain[:] = 0
        state.follow[:] = False
        state.spent[:] = 0
        state.home_to = np.where(tied, state.home_to + 2.0, state.home_to)
        state.away_to = np.where(tied, state.away_to + 2.0, state.away_to)
        jump = rng.random(n) < 0.53
        state.offense_home = np.where(tied, jump, state.offense_home)
        lead = _score(home_box) - _score(away_box)
        home_line = _lineups_for(spec.home_players, lead, home_active, 4, OT_SECONDS, state.home_floor)
        away_line = _lineups_for(spec.away_players, -lead, away_active, 4, OT_SECONDS, state.away_floor)
        state.home_lineup = np.where(tied[:, None], home_line, state.home_lineup)
        state.away_lineup = np.where(tied[:, None], away_line, state.away_lineup)
        rows = np.flatnonzero(tied)
        if rows.size:
            _play_court_stretch(
                rng, spec, state, home, away, home_box, away_box,
                state.home_lineup, state.away_lineup, home_residual, away_residual, OT_SECONDS, 4 + ot,
            )
        _close_quarter(state)
    return margin, blowout


def _court_vectors(state, home: Side, away: Side):
    spec = state.spec
    n = state.n
    n_types = len(PLAY_TYPES)
    home_def = np.ones((n, n_types))
    away_def = np.ones((n, n_types))
    home_zone = np.ones((n, len(home.players)))
    away_zone = np.ones((n, max(len(away.players), 1)))
    home_pair = np.ones((n, len(home.players)))
    away_pair = np.ones((n, max(len(away.players), 1)))
    cache = getattr(state, "_vector_cache", None)
    if cache is None:
        cache = {}
        state._vector_cache = cache
    _fill_vectors(away_def, home_zone, home_pair, state.away_lineup, spec.home_players, spec.away_players, spec.away, spec.league, cache, "away")
    _fill_vectors(home_def, away_zone, away_pair, state.home_lineup, spec.away_players, spec.home_players, spec.home, spec.league, cache, "home")
    return home_def, away_def, home_zone, away_zone, home_pair, away_pair


def _price_table(offense: list[PlayerModel], defense: list[PlayerModel], defense_team: TeamModel, league) -> dict:
    """Player-versus-player prices. A lineup is a sum, not another pass through the zones."""
    n_off = len(offense)
    n_def = len(defense)
    n_play = len(PLAY_TYPES)
    team_zone = np.ones(n_off)
    zone_ratio = np.ones((n_off, n_def))
    zone_w = np.zeros((n_off, n_def))
    pair_num = np.zeros((n_off, n_def))
    pair_den = np.zeros((n_off, n_def))
    pair_on = np.zeros(n_off, dtype=bool)
    for i, player in enumerate(offense):
        freq = player.zone_freq or league.zone_freq
        team_zone[i] = _zone_ratio(freq, defense_team.zone_d_fg, defense_team.zone_normal_fg, league)
        off_pos = position_key(player.position)
        pair_on[i] = bool(player.matchup_fg)
        base = 0.0
        base_w = 0.0
        if player.matchup_fg:
            for zone, share in (player.zone_freq or {}).items():
                base += share * player.zone_fg.get(zone, 0.46)
                base_w += share
        base_fg = base / base_w if base_w else 0.46
        for j, defender in enumerate(defense):
            if defender.zone_d_fg:
                zone_ratio[i, j] = _zone_ratio(
                    freq,
                    defender.zone_d_fg,
                    defender.zone_normal_fg,
                    league,
                    fallback_allowed=defense_team.zone_d_fg,
                    fallback_normal=defense_team.zone_normal_fg,
                )
                weight = sum(defender.zone_d_fga.values()) or defender.sample_games
                zone_w[i, j] = weight * _POS_MATCH[off_pos][position_key(defender.position)]
            sample = player.matchup_fg.get(defender.player_id) if player.matchup_fg and base_fg > 0 else None
            if sample is None:
                continue
            fg_pct, attempts = sample
            if attempts <= 0:
                continue
            ratio = shrink(fg_pct / base_fg, attempts, 1.0, MATCHUP_PRIOR)
            pair_num[i, j] = attempts * ratio
            pair_den[i, j] = attempts
    poss = np.zeros((n_def, n_play))
    mass = np.zeros((n_def, n_play))
    fg_poss = np.zeros((n_def, n_play))
    fg_mass = np.zeros((n_def, n_play))
    for j, defender in enumerate(defense):
        for p, name in enumerate(PLAY_TYPES):
            skill = defender.defense.get(name)
            if skill is None or skill.poss <= 0:
                continue
            poss[j, p] = float(skill.poss)
            mass[j, p] = float(skill.poss) * float(skill.ppp)
            if skill.fg_sample > 0 and skill.fg_pct > 0:
                fg_poss[j, p] = float(skill.fg_sample)
                fg_mass[j, p] = float(skill.fg_sample) * float(skill.fg_pct)
    league_ppp = np.ones(n_play)
    team_ppp = np.ones(n_play)
    league_fg = np.zeros(n_play)
    team_fg = np.zeros(n_play)
    league_fg_map = getattr(league, "fg_by_type", {}) or {}
    team_fg_map = getattr(defense_team, "def_fg", {}) or {}
    team_fg_n = getattr(defense_team, "def_fg_poss", {}) or {}
    for p, name in enumerate(PLAY_TYPES):
        league_rate = float(league.ppp_by_type.get(name, 1.0) or 1.0)
        if league_rate <= 0:
            league_rate = 1.0
        league_ppp[p] = league_rate
        raw = float(defense_team.def_ppp.get(name, league_rate) or league_rate)
        team_n = float(defense_team.def_poss.get(name, 0.0) or 0.0)
        team_ppp[p] = shrink(raw, team_n, league_rate, LEAGUE_PRIOR)
        baseline_fg = float(league_fg_map.get(name, 0.0) or 0.0)
        raw_fg = float(team_fg_map.get(name, 0.0) or 0.0)
        sample_fg = float(team_fg_n.get(name, 0.0) or 0.0)
        league_fg[p] = baseline_fg
        if baseline_fg > 0 and raw_fg > 0:
            team_fg[p] = shrink(raw_fg, sample_fg, baseline_fg, LEAGUE_PRIOR)
    ppp_level, ppp_measured = team_ppp_level(defense_team, league)
    fg_level, fg_measured = team_fg_level(defense_team, league)
    rating = float(getattr(defense_team, "def_rating", LEAGUE_RATING) or LEAGUE_RATING) / LEAGUE_RATING
    if not np.isfinite(rating) or rating <= 0:
        rating = 1.0
    return {
        "team_zone": team_zone,
        "zone_ratio": zone_ratio,
        "zone_w": zone_w,
        "pair_num": pair_num,
        "pair_den": pair_den,
        "pair_on": pair_on,
        "poss": poss,
        "mass": mass,
        "fg_poss": fg_poss,
        "fg_mass": fg_mass,
        "league_ppp": league_ppp,
        "team_ppp": team_ppp,
        "league_fg": league_fg,
        "team_fg": team_fg,
        "rating": rating,
        "ppp_level": ppp_level,
        "ppp_measured": ppp_measured,
        "fg_level": fg_level,
        "fg_measured": fg_measured,
    }


def _blend_lineup(table: dict, indexes: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    valid = [index for index in indexes if 0 <= index < table["poss"].shape[0]]
    poss = table["poss"][valid].sum(axis=0) if valid else np.zeros(len(PLAY_TYPES))
    mass = table["mass"][valid].sum(axis=0) if valid else np.zeros(len(PLAY_TYPES))
    team_ppp = table["team_ppp"]
    player = np.array(
        [
            shrink(mass[p] / poss[p], poss[p], team_ppp[p], FILE_PRIOR) if poss[p] > 0 else team_ppp[p]
            for p in range(len(PLAY_TYPES))
        ]
    )
    # Field-goal percentage allowed moves the make. Points allowed are the
    # fallback when that column was not recorded. The rating is the level of
    # the file that was used, so a ball-screen that is worse than the rest of
    # the team stays worse.
    league_ppp = table["league_ppp"]
    ppp_ratio = np.divide(player, league_ppp, out=np.ones_like(player), where=league_ppp > 1e-6)
    if valid:
        shot_n = table["fg_poss"][valid].sum(axis=0)
        shot_mass = table["fg_mass"][valid].sum(axis=0)
    else:
        shot_n = np.zeros(len(PLAY_TYPES))
        shot_mass = np.zeros(len(PLAY_TYPES))
    league_fg = table["league_fg"]
    team_fg = table["team_fg"]
    player_fg = np.ones(len(PLAY_TYPES))
    for p in range(len(PLAY_TYPES)):
        anchor = team_fg[p] if team_fg[p] > 0 else league_fg[p]
        if shot_n[p] > 0 and anchor > 0:
            player_fg[p] = shrink(shot_mass[p] / shot_n[p], shot_n[p], anchor, FILE_PRIOR)
        elif team_fg[p] > 0:
            player_fg[p] = team_fg[p]
    fg_ratio = np.divide(player_fg, league_fg, out=np.ones_like(player_fg), where=league_fg > 1e-6)
    use_fg = (league_fg > 1e-6) & (player_fg > 0) & ((shot_n > 0) | (team_fg > 0))
    rating = float(table.get("rating", 1.0) or 1.0)
    def_mult = np.array(
        [
            compose_defense(
                float(fg_ratio[p] if use_fg[p] else ppp_ratio[p]),
                rating,
                float(table["fg_level"] if use_fg[p] else table["ppp_level"]),
                bool(table["fg_measured"] if use_fg[p] else table["ppp_measured"]),
            )
            for p in range(len(PLAY_TYPES))
        ]
    )
    if valid:
        num = (table["zone_w"][:, valid] * table["zone_ratio"][:, valid]).sum(axis=1)
        den = table["zone_w"][:, valid].sum(axis=1)
        pair_num = table["pair_num"][:, valid].sum(axis=1)
        pair_den = table["pair_den"][:, valid].sum(axis=1)
    else:
        num = np.zeros(len(table["team_zone"]))
        den = num
        pair_num = num
        pair_den = num
    zone = np.empty(len(table["team_zone"]))
    pair = np.ones(len(table["team_zone"]))
    for i, team_zone in enumerate(table["team_zone"]):
        blended = team_zone if den[i] <= 0 else shrink(num[i] / den[i], den[i], team_zone, FILE_PRIOR)
        zone[i] = float(blended) if np.isfinite(blended) else 1.0
        if table["pair_on"][i] and pair_den[i] > 0:
            pair[i] = float(pair_num[i] / pair_den[i])
    return def_mult, zone, pair


def _fill_vectors(def_out, zone_out, pair_out, def_lineup, offense, defense, defense_team, league, cache, tag: str) -> None:
    if def_lineup.size == 0 or not defense:
        return
    table_key = ("prices", tag)
    table = cache.get(table_key)
    if table is None:
        table = _price_table(offense, defense, defense_team, league)
        cache[table_key] = table
    inverse = _group_ids(def_lineup)
    for group in range(int(inverse.max()) + 1):
        row = int(np.flatnonzero(inverse == group)[0])
        indexes = tuple(int(index) for index in def_lineup[row])
        key = (tag, indexes)
        if key not in cache:
            cache[key] = _blend_lineup(table, indexes)
        def_mult, zone, pair = cache[key]
        rows = inverse == group
        def_out[rows] = def_mult
        width = min(zone_out.shape[1], len(zone))
        zone_out[rows, :width] = zone[:width]
        pair_out[rows, :width] = pair[:width]


def _court_pace(state) -> np.ndarray:
    spec = state.spec
    both = np.concatenate([state.home_lineup, state.away_lineup], axis=1)
    inverse = _group_ids(both)
    out = np.empty(state.n)
    for group in range(int(inverse.max()) + 1):
        row = int(np.flatnonzero(inverse == group)[0])
        home_idx = [int(index) for index in state.home_lineup[row]]
        away_idx = [int(index) for index in state.away_lineup[row]]
        pace = _pace(
            spec.home,
            spec.away,
            _unit_ids(spec.home_players, home_idx),
            _unit_ids(spec.away_players, away_idx),
            spec.context.pace_mult,
        )
        out[inverse == group] = 100.0 / pace
    return out


def _relative_rows(side: Side, lineup: np.ndarray) -> np.ndarray:
    inverse = _group_ids(lineup)
    out = np.full(lineup.shape, 0.2)
    priors = {int(player.player_id): float(player.usg or 0.2) for player in side.players}
    for group in range(int(inverse.max()) + 1):
        row = int(np.flatnonzero(inverse == group)[0])
        indexes = [int(index) for index in lineup[row]]
        ids = [int(side.players[index].player_id) for index in indexes if 0 <= index < len(side.players)]
        if len(ids) < 5:
            continue
        usage = relative_usage(ids[:5], side.squads, priors)
        shares = [float(usage.get(pid, 0.2)) for pid in ids[:5]]
        out[inverse == group] = shares
    return out


def _apply_squad_mix(side: Side, lineup: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Reweight the five's Synergy mix by the buckets play-by-play can see.

    A thin five keeps the rate card. Spot and pick are spread across the
    actions already on that card, then shrunk back toward it.
    """
    inverse = _group_ids(lineup)
    individual = side.poss[lineup].sum(axis=1)
    individual = individual / np.maximum(individual.sum(axis=1, keepdims=True), 1e-9)
    out = weights
    changed = False
    for group in range(int(inverse.max()) + 1):
        row = int(np.flatnonzero(inverse == group)[0])
        indexes = [int(index) for index in lineup[row]]
        ids = [int(side.players[index].player_id) for index in indexes if 0 <= index < len(side.players)]
        if len(ids) < 5:
            continue
        face = side.squads.get(frozenset(ids[:5]))
        if face is None or not getattr(face, "mix", None) or float(getattr(face, "possessions", 0.0) or 0.0) < 15.0:
            continue
        synergy = individual[row]
        expanded = _crosswalk_mix(face.mix, synergy)
        sample = float(face.possessions)
        trust = sample / (sample + FILE_PRIOR)
        blended = (1.0 - trust) * synergy + trust * expanded
        blended = blended / max(float(blended.sum()), 1e-9)
        scale = np.ones(len(PLAY_TYPES), dtype=float)
        live = synergy > 1e-8
        scale[live] = np.clip(blended[live] / synergy[live], 0.25, 4.0)
        rows = inverse == group
        out = out.copy() if not changed else out
        changed = True
        out[rows] = out[rows] * scale[None, None, :]
    return out


# Minutes of this exact five that tie its rating to the players already priced.
# A short look stays near their on/off. A five that has played together stands on its own.
_FIVE_PRIOR_MINUTES = 96.0


def _exact_face(squads: dict, ids: list[int]):
    if not squads or len(ids) < 5:
        return None
    face = squads.get(frozenset(int(player_id) for player_id in ids[:5]))
    if face is None or float(getattr(face, "minutes", 0.0) or 0.0) < 8.0:
        return None
    return face


def _level_rows(side: Side, defense: Side, lineup: np.ndarray, def_lineup: np.ndarray):
    inverse = _group_ids(np.concatenate([lineup, def_lineup], axis=1))
    off_scale = np.ones(lineup.shape[0])
    def_scale = np.ones(lineup.shape[0])
    off_trust = np.zeros(lineup.shape[0])
    def_trust = np.zeros(lineup.shape[0])
    off_known = np.zeros(lineup.shape[0], dtype=bool)
    def_known = np.zeros(lineup.shape[0], dtype=bool)
    team_def = float(defense.team_def or 114.0)
    for group in range(int(inverse.max()) + 1):
        row = int(np.flatnonzero(inverse == group)[0])
        off_ids = [int(side.players[int(index)].player_id) for index in lineup[row] if 0 <= int(index) < len(side.players)]
        def_ids = [int(defense.players[int(index)].player_id) for index in def_lineup[row] if 0 <= int(index) < len(defense.players)]
        rows = inverse == group
        face = _exact_face(getattr(side, "squads", None) or {}, off_ids)
        if face is not None and float(face.off_rating or 0.0) > 70.0:
            off_indexes = [
                int(index)
                for index in lineup[row]
                if 0 <= int(index) < len(side.players)
            ]
            if len(off_indexes) >= 5:
                # This five against the players already priced. A short sample
                # stays on their on/off. Minutes, not a cap, decide the weight.
                composed = _composed_ppp(side, off_indexes, rows)
                off_scale[rows] = (float(face.off_rating) / 100.0) / composed
                off_trust[rows] = float(face.minutes) / (float(face.minutes) + _FIVE_PRIOR_MINUTES)
                off_known[rows] = True
        other = _exact_face(getattr(defense, "squads", None) or {}, def_ids)
        if other is not None and float(other.def_rating or 0.0) > 70.0 and team_def > 0:
            def_scale[rows] = float(other.def_rating) / team_def
            def_trust[rows] = float(other.minutes) / (float(other.minutes) + _FIVE_PRIOR_MINUTES)
            def_known[rows] = True
    return off_scale, def_scale, off_trust, def_trust, off_known, def_known
