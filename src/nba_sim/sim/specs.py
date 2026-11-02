"""Game, player, and team inputs for the simulator."""

from __future__ import annotations

from dataclasses import dataclass, field

from nba_sim.constants import DEFAULT_ZONE_FG, DEFAULT_ZONE_FREQ, PLAY_TYPES


@dataclass
class PlaySkill:
    poss: float
    ppp: float
    poss_pct: float = 0.0
    fg_pct: float = 0.46
    # Season possessions behind fg_pct. Zero means ``poss`` is already that count.
    # A per-game frequency must not be used as the sample.
    fg_sample: float = 0.0
    ft_poss_pct: float = 0.08
    tov_poss_pct: float = 0.12
    score_poss_pct: float = 0.48
    efg_pct: float = 0.52
    percentile: float = 50.0


@dataclass
class PlayerModel:
    player_id: int
    name: str
    team_id: int
    position: str
    minutes_mean: float
    minutes_std: float = 2.5
    available: bool = True
    offense: dict[str, PlaySkill] = field(default_factory=dict)
    defense: dict[str, PlaySkill] = field(default_factory=dict)
    zone_freq: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ZONE_FREQ))
    zone_fg: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ZONE_FG))
    zone_d_fg: dict[str, float] = field(default_factory=dict)
    zone_normal_fg: dict[str, float] = field(default_factory=dict)
    zone_d_fga: dict[str, float] = field(default_factory=dict)
    oreb_pct: float = 0.05
    dreb_pct: float = 0.15
    ast_share: float = 1.0
    # Share of teammate makes he assisted while on the floor. Zero until the
    # advanced line is joined, and the clock then converts ast_share.
    ast_pct: float = 0.0
    stl_weight: float = 1.0
    blk_weight: float = 0.4
    ft_pct: float = 0.78
    usg: float = 0.20
    sample_games: float = 40.0
    clutch_mult: float = 1.0
    screen_assists: float = 0.0
    box_outs: float = 0.0
    espn_id: str = ""
    # defender id -> (field-goal percentage allowed to this offensive player, attempts)
    matchup_fg: dict[int, tuple[float, float]] = field(default_factory=dict)
    minutes_source: str = ""
    minutes_base: float = 0.0
    minutes_sample: int = 0
    minutes_slot: int = 0
    # ESPN depth-chart opening five for tonight, after outs are removed.
    projected_starter: bool = False
    # Minutes and games behind the scoring line. Zero until that line is joined.
    minutes_season: float = 0.0
    minutes_season_games: float = 0.0
    # Depth-chart minutes, before the share of the team's schedule moves them.
    minutes_depth: float = 0.0
    # Team games, his games for that team, and the average of those games.
    # Counted from logs dated before the slate.
    minutes_team_games: float = 0.0
    minutes_appearances: float = 0.0
    minutes_appearance_avg: float = 0.0
    # Recent scoring per FGA vs season card, folded into the make move only.
    make_form: float = 1.0
    make_form_games: float = 0.0
    injury_label: str = ""
    injury_factor: float = 1.0
    # Personal fouls per 36 minutes, divided by a league average of 3.2.
    # Zero uses a position prior: guards foul less, centers foul more.
    # On defense this is the rate he concedes a foul.
    foul_rate: float = 0.0
    # His free throws per shot, divided by the rate his own play mix already
    # implies. 1.0 leaves that mix where it is. A short sample stays near 1.
    ft_draw: float = 1.0
    # Team offense and points allowed while he is on the floor, divided by
    # that team's own rating. 1.0 leaves the lineup where the rating already is.
    off_impact: float = 1.0
    def_impact: float = 1.0
    # Possessions per 48 he adds to the team pace, from the fives he has
    # played in. Zero leaves a new group on the team dashboard.
    pace_impact: float = 0.0
    # Teammate id -> minutes those two shared a five. Empty until the lineup file is joined.
    shared_minutes: dict[int, float] = field(default_factory=dict)
    # His court profile under tonight's coach. None keeps the older rotation script.
    court: object = None


@dataclass
class TeamModel:
    team_id: int
    abbr: str
    name: str
    pace: float = 99.5
    off_rating: float = 114.0
    def_rating: float = 114.0
    oreb_pct: float = 0.27
    dreb_pct: float = 0.73
    games: float = 40.0
    off_ppp: dict[str, float] = field(default_factory=dict)
    off_poss: dict[str, float] = field(default_factory=dict)
    def_ppp: dict[str, float] = field(default_factory=dict)
    def_poss: dict[str, float] = field(default_factory=dict)
    # Turnover rate this defense forces on each play, from its own sample.
    def_tov: dict[str, float] = field(default_factory=dict)
    # Field-goal percentage allowed on each action, and the possessions behind it.
    def_fg: dict[str, float] = field(default_factory=dict)
    def_fg_poss: dict[str, float] = field(default_factory=dict)
    zone_d_fg: dict[str, float] = field(default_factory=dict)
    zone_normal_fg: dict[str, float] = field(default_factory=dict)
    lineup_pace: dict[tuple[int, ...], float] = field(default_factory=dict)
    # Player id -> possessions per 48 above the team pace. None until the
    # lineup book is fit. An empty dict means the book had nothing to say.
    pace_impact: dict[int, float] | None = None
    # Five-man groups that actually played, and how many minutes they shared.
    lineup_minutes: dict[tuple[int, ...], float] = field(default_factory=dict)
    # Sorted pair of player ids -> minutes those two shared a five.
    shared_minutes: dict[tuple[int, int], float] = field(default_factory=dict)
    coach: str = ""
    # How many players this coach regularly plays. Game logs replace the curve.
    rotation_size: int = 10
    # 0 sits at the first sign of foul trouble. 1 plays a starter through it.
    foul_patience: float = 0.45
    # Opponent free-throw rate divided by a league rate of 0.26.
    fta_allowed: float = 1.0
    # Offensive rebound rate this defense gives up. Four Factors OPP_OREB_PCT.
    oreb_allowed: float = 0.27
    # Opponent assist rate vs league (~0.60 AST/FGM), as a multiplier on 1.0.
    ast_allowed: float = 1.0
    # Share of opponent REB / AST by position (G/F/C) this defense concedes.
    reb_share_allowed: dict[str, float] = field(default_factory=lambda: {"G": 0.22, "F": 0.38, "C": 0.40})
    ast_share_allowed: dict[str, float] = field(default_factory=lambda: {"G": 0.55, "F": 0.35, "C": 0.10})
    # Five-man offensive rating, defensive rating, and offensive rebound rate.
    lineup_off: dict[tuple[int, ...], float] = field(default_factory=dict)
    lineup_def: dict[tuple[int, ...], float] = field(default_factory=dict)
    lineup_oreb: dict[tuple[int, ...], float] = field(default_factory=dict)
    # Squad faces and the position matrix, filled when the court profiles are fit.
    squads: dict = field(default_factory=dict)
    sub_matrix: dict = field(default_factory=dict)
    court_whistle: object = None


@dataclass
class MarketQuote:
    source: str
    market: str
    side: str
    american: float
    line: float | None = None
    player_id: int | None = None
    player_name: str | None = None
    book: str = ""


@dataclass
class GameContext:
    home_ppp_mult: float = 1.0
    away_ppp_mult: float = 1.0
    pace_mult: float = 1.0
    flags: list[str] = field(default_factory=list)
    factors: list[dict] = field(default_factory=list)


@dataclass
class LeaguePriors:
    ppp_by_type: dict[str, float]
    # Field-goal percentage on each action. Empty until the play file is joined.
    fg_by_type: dict[str, float] = field(default_factory=dict)
    zone_fg: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ZONE_FG))
    zone_freq: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_ZONE_FREQ))
    oreb: float = 0.27
    dreb: float = 0.73
    ft_pct: float = 0.78

    @staticmethod
    def flat(ppp: float = 1.02) -> "LeaguePriors":
        return LeaguePriors(ppp_by_type={play: ppp for play in PLAY_TYPES})


@dataclass
class GameSpec:
    game_id: str
    date: str
    home: TeamModel
    away: TeamModel
    home_players: list[PlayerModel]
    away_players: list[PlayerModel]
    context: GameContext
    league: LeaguePriors
    quotes: list[MarketQuote] = field(default_factory=list)
    source: str = "model"
    venue: str = ""
    city: str = ""
    start_utc: str = ""
    tip_local: str = ""
    tip_melbourne: str = ""
    game_status: str = ""
    injuries: list[dict] = field(default_factory=list)
    # Per-path availability, one column per player. None draws everyone who is on the sheet.
    home_active: object = None
    away_active: object = None
