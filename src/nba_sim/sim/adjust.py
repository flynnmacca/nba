"""Matchup multipliers shared by the simulator and the slate explanation."""

from __future__ import annotations

import numpy as np

from nba_sim.constants import LEAGUE_RATING, PLAY_TYPES
from nba_sim.features.play_priors import position_key
from nba_sim.features.shrink import FILE_PRIOR, LEAGUE_PRIOR, MATCHUP_PRIOR, shrink
from nba_sim.sim.specs import LeaguePriors, PlayerModel, TeamModel

# Same shot mix the possession clock uses. A paint action lives at the rim.
# A spot-up lives on the three. The defensive profile follows that shot.
_SHOT_THREE = (0.25, 0.32, 0.22, 0.05, 0.02, 0.78, 0.55, 0.08, 0.62, 0.02, 0.20)
_SHOT_PAINT = (False, True, False, True, True, False, False, True, False, True, False)
# Who is likely to be the primary defender, by the offensive player's position.
_POS_MATCH = {
    "G": {"G": 1.75, "F": 1.00, "C": 0.55},
    "F": {"G": 1.00, "F": 1.55, "C": 1.00},
    "C": {"G": 0.55, "F": 1.05, "C": 1.75},
}


def offense_weights(players: list[PlayerModel], screen_boost: float = 1.0) -> np.ndarray:
    weights = np.zeros(len(PLAY_TYPES), dtype=float)
    for player in players:
        for index, name in enumerate(PLAY_TYPES):
            skill = player.offense.get(name)
            if skill is not None and skill.poss > 0:
                weights[index] += skill.poss
    for index, name in enumerate(PLAY_TYPES):
        if name in ("PRBallHandler", "PRRollman"):
            weights[index] *= screen_boost
    if float(weights.sum()) <= 0:
        weights[:] = 1.0
    return weights


def screen_boost(players: list[PlayerModel]) -> float:
    screens = sum(p.screen_assists for p in players)
    # League rotation units are around the mid-teens in screen assists per game.
    bump = 0.04 * float(np.clip((screens - 12.0) / 12.0, -1.0, 1.5))
    return float(np.clip(1.0 + bump, 0.9, 1.15))


def _rating_index(defense: TeamModel) -> float:
    rating = float(defense.def_rating or LEAGUE_RATING)
    if not np.isfinite(rating) or rating <= 0:
        return 1.0
    return rating / LEAGUE_RATING


def _file_level(values: list[tuple[float, float, float]]) -> tuple[float, bool]:
    """Possession-weighted mean of value / baseline, and whether any sample existed."""
    num = 0.0
    den = 0.0
    for value, baseline, poss in values:
        if baseline <= 0 or poss <= 0 or not np.isfinite(value) or value <= 0:
            continue
        num += poss * (value / baseline)
        den += poss
    if den <= 0:
        return 1.0, False
    return num / den, True


def team_ppp_level(defense: TeamModel, league: LeaguePriors) -> tuple[float, bool]:
    """Where this team's play-type points sit against the league, across actions."""
    rows: list[tuple[float, float, float]] = []
    for play in PLAY_TYPES:
        baseline = float(league.ppp_by_type.get(play, 1.0) or 1.0)
        if baseline <= 0:
            baseline = 1.0
        raw = float(defense.def_ppp.get(play, baseline) or baseline)
        poss = float(defense.def_poss.get(play, 0.0) or 0.0)
        rows.append((shrink(raw, poss, baseline, LEAGUE_PRIOR), baseline, poss))
    return _file_level(rows)


def team_fg_level(defense: TeamModel, league: LeaguePriors) -> tuple[float, bool]:
    """Where the field-goal percentage this team allows sits against the league."""
    rows: list[tuple[float, float, float]] = []
    league_fg = getattr(league, "fg_by_type", {}) or {}
    team_fg = getattr(defense, "def_fg", {}) or {}
    team_n = getattr(defense, "def_fg_poss", {}) or {}
    for play in PLAY_TYPES:
        baseline = float(league_fg.get(play, 0.0) or 0.0)
        raw = float(team_fg.get(play, 0.0) or 0.0)
        poss = float(team_n.get(play, 0.0) or 0.0)
        if baseline <= 0 or raw <= 0 or poss <= 0:
            continue
        rows.append((shrink(raw, poss, baseline, LEAGUE_PRIOR), baseline, poss))
    return _file_level(rows)


def compose_defense(ratio: float, rating: float, level: float, measured: bool) -> float:
    """Rating is the level. The file keeps the shape across actions.

    A measured file is divided back out, so a team whose play points already
    match its rating is left alone, and a ball-screen that is worse than the
    rest of that team stays worse. An empty file uses the rating itself.
    """
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0
    if not np.isfinite(rating) or rating <= 0:
        rating = 1.0
    if measured and level > 1e-6 and np.isfinite(level):
        return float(ratio * rating / level)
    if abs(ratio - 1.0) > 0.02:
        return float(ratio)
    return float(rating)


def defense_multiplier(
    play_type: str,
    defenders: list[PlayerModel],
    defense: TeamModel,
    league: LeaguePriors,
) -> float:
    """How this action is defended relative to a league-average team.

    A measured field-goal percentage moves the make. Points allowed are the
    fallback when that column was not recorded. Either way the team's
    defensive rating is the level, and the action only says how it sits
    against the rest of that team.
    """
    rating = _rating_index(defense)
    league_fg = float((getattr(league, "fg_by_type", {}) or {}).get(play_type, 0.0) or 0.0)
    team_fg_raw = float((getattr(defense, "def_fg", {}) or {}).get(play_type, 0.0) or 0.0)
    team_fg_n = float((getattr(defense, "def_fg_poss", {}) or {}).get(play_type, 0.0) or 0.0)
    fg_weight = 0.0
    fg_sum = 0.0
    for defender in defenders:
        skill = defender.defense.get(play_type)
        if skill is None or skill.fg_sample <= 0 or skill.fg_pct <= 0:
            continue
        fg_sum += skill.fg_sample * skill.fg_pct
        fg_weight += skill.fg_sample
    if league_fg > 0 and (team_fg_n > 0 or fg_weight > 0):
        anchor = team_fg_raw if team_fg_raw > 0 else league_fg
        team_fg = shrink(anchor, team_fg_n, league_fg, LEAGUE_PRIOR)
        player_fg = shrink(fg_sum / fg_weight, fg_weight, team_fg, FILE_PRIOR) if fg_weight > 0 else team_fg
        level, measured = team_fg_level(defense, league)
        return compose_defense(player_fg / league_fg, rating, level, measured)

    league_ppp = float(league.ppp_by_type.get(play_type, 1.0) or 1.0)
    if league_ppp <= 0:
        league_ppp = 1.0
    team_ppp = float(defense.def_ppp.get(play_type, league_ppp) or league_ppp)
    team_n = float(defense.def_poss.get(play_type, 0.0) or 0.0)
    team_ppp = shrink(team_ppp, team_n, league_ppp, LEAGUE_PRIOR)
    weighted = 0.0
    weight = 0.0
    for defender in defenders:
        skill = defender.defense.get(play_type)
        if skill is None or skill.poss <= 0:
            continue
        weighted += skill.poss * skill.ppp
        weight += skill.poss
    player_ppp = shrink(weighted / weight, weight, team_ppp, FILE_PRIOR) if weight > 0 else team_ppp
    level, measured = team_ppp_level(defense, league)
    if team_ppp <= 0 or not np.isfinite(player_ppp) or player_ppp <= 0:
        return compose_defense(1.0, rating, level, measured)
    return compose_defense(player_ppp / league_ppp, rating, level, measured)


def _zone_ratio(
    freq: dict[str, float],
    allowed: dict[str, float],
    normal_map: dict[str, float],
    league: LeaguePriors,
    fallback_allowed: dict[str, float] | None = None,
    fallback_normal: dict[str, float] | None = None,
) -> float:
    acc = 0.0
    total = 0.0
    for zone, share in freq.items():
        if share <= 0:
            continue
        normal = normal_map.get(zone)
        if normal is None and fallback_normal is not None:
            normal = fallback_normal.get(zone)
        if normal is None:
            normal = league.zone_fg.get(zone, 0.45)
        d_fg = allowed.get(zone)
        if d_fg is None and fallback_allowed is not None:
            d_fg = fallback_allowed.get(zone, normal)
        if d_fg is None:
            d_fg = normal
        acc += float(share) * (float(d_fg) / max(float(normal), 0.05))
        total += float(share)
    if total <= 0:
        return 1.0
    return acc / total


def play_shot_freq(play: str) -> dict[str, float]:
    """Where shots from this action come from, so a defender's zones can price it."""
    index = PLAY_TYPES.index(play)
    three = float(_SHOT_THREE[index])
    two = max(0.0, 1.0 - three)
    if _SHOT_PAINT[index]:
        freq = {
            "ra": 0.62 * two,
            "paint": 0.38 * two,
            "mid": 0.0,
            "atb3": 0.72 * three,
            "corner3": 0.28 * three,
        }
    else:
        freq = {
            "ra": 0.12 * two,
            "paint": 0.28 * two,
            "mid": 0.60 * two,
            "atb3": 0.78 * three,
            "corner3": 0.22 * three,
        }
    total = sum(freq.values()) or 1.0
    return {zone: value / total for zone, value in freq.items()}


def zone_matchup_mult(
    player: PlayerModel,
    defense: TeamModel,
    league: LeaguePriors,
    defenders: list[PlayerModel] | None = None,
) -> float:
    freq = player.zone_freq or league.zone_freq
    team_ratio = _zone_ratio(freq, defense.zone_d_fg, defense.zone_normal_fg, league)
    if not defenders:
        return float(team_ratio) if np.isfinite(team_ratio) else 1.0
    num = 0.0
    den = 0.0
    off_pos = position_key(player.position)
    for defender in defenders:
        if not defender.zone_d_fg:
            continue
        ratio = _zone_ratio(
            freq,
            defender.zone_d_fg,
            defender.zone_normal_fg,
            league,
            fallback_allowed=defense.zone_d_fg,
            fallback_normal=defense.zone_normal_fg,
        )
        weight = sum(defender.zone_d_fga.values()) or defender.sample_games
        weight *= _POS_MATCH[off_pos][position_key(defender.position)]
        num += weight * ratio
        den += weight
    if den <= 0:
        blended = team_ratio
    else:
        blended = shrink(num / den, den, team_ratio, FILE_PRIOR)
    return float(blended) if np.isfinite(blended) else 1.0


def matchup_multiplier(player: PlayerModel, defenders: list[PlayerModel]) -> float:
    if not player.matchup_fg:
        return 1.0
    base = 0.0
    base_w = 0.0
    for zone, share in player.zone_freq.items():
        base += share * player.zone_fg.get(zone, 0.46)
        base_w += share
    base_fg = base / base_w if base_w else 0.46
    num = 0.0
    den = 0.0
    for defender in defenders:
        sample = player.matchup_fg.get(defender.player_id)
        if sample is None:
            continue
        fg_pct, attempts = sample
        if attempts <= 0 or base_fg <= 0:
            continue
        ratio = shrink(fg_pct / base_fg, attempts, 1.0, MATCHUP_PRIOR)
        num += attempts * ratio
        den += attempts
    if den <= 0:
        return 1.0
    blended = float(num / den)
    return blended if np.isfinite(blended) else 1.0


def offensive_residual(players: list[PlayerModel], team: TeamModel) -> float:
    """Pull a lineup's composed PPP toward that team's own offensive rating."""
    num = 0.0
    den = 0.0
    for player in players:
        for skill in player.offense.values():
            if skill.poss <= 0:
                continue
            num += skill.poss * skill.ppp
            den += skill.poss
    if den <= 0 or team.off_rating <= 0:
        return 1.0
    composed = num / den
    raw = (team.off_rating / 100.0) / composed
    return float(np.clip(shrink(raw, team.games, 1.0, 25.0), 0.9, 1.1))


# The five on the floor is the glass. The team rate pulls that sum, it does not replace it.
_PLAYER_GLASS = 0.70
_TEAM_GLASS = 0.30


def lineup_oreb(
    off_sum: np.ndarray,
    def_sum: np.ndarray,
    off_box: np.ndarray,
    def_box: np.ndarray,
    off_team: float | None = None,
    def_team: float | None = None,
    def_allowed: float | None = None,
) -> np.ndarray:
    """Offensive rebound chance for each simulation.

    ``off_sum`` is the five offensive players' OREB_PCT. ``def_sum`` is the five
    defenders' DREB_PCT, which is the glass they personally secure and therefore
    what they concede. ``off_team`` is the offense's team rate. ``def_team`` is
    the defense's team defensive rebound rate and ``def_allowed`` is the opponent
    offensive rebound rate that defense gives up. Team numbers count for 30 percent.
    """
    nudge = 1.0 + 0.02 * np.clip((np.asarray(off_box, dtype=float) - np.asarray(def_box, dtype=float)) / 8.0, -1.0, 1.0)
    offense = np.maximum(0.01, np.asarray(off_sum, dtype=float) * nudge)
    glass = np.asarray(def_sum, dtype=float)
    if off_team is not None:
        offense = _PLAYER_GLASS * offense + _TEAM_GLASS * float(off_team)
    if def_team is not None or def_allowed is not None:
        if def_team is None:
            kept = 1.0 - float(def_allowed)
        elif def_allowed is None:
            kept = float(def_team)
        else:
            kept = 0.5 * float(def_team) + 0.5 * (1.0 - float(def_allowed))
        glass = _PLAYER_GLASS * glass + _TEAM_GLASS * kept
    # Five defenders' rebound shares should land near a team rate. A hole in
    # the roster is not a team that gives up every miss.
    glass = np.maximum(0.55, glass)
    return np.clip(offense / (offense + glass), 0.02, 0.55)


def oreb_probability(
    offense: list[PlayerModel],
    defense: list[PlayerModel],
    off_team: float | None = None,
    def_team: float | None = None,
    def_allowed: float | None = None,
) -> float:
    """On-court offensive rebound share, pulled toward the two team rates."""
    chance = lineup_oreb(
        sum(p.oreb_pct for p in offense),
        sum(p.dreb_pct for p in defense),
        sum(p.box_outs for p in offense),
        sum(p.box_outs for p in defense),
        off_team,
        def_team,
        def_allowed,
    )
    return float(chance)


def zone_make_rates(zone_freq: dict[str, float], zone_fg: dict[str, float]) -> tuple[float, float]:
    """Field-goal percentage on twos, and on threes, from the shot chart."""
    two_weight = 0.0
    two_makes = 0.0
    three_weight = 0.0
    three_makes = 0.0
    for zone, share in zone_freq.items():
        weight = float(share)
        made = weight * float(zone_fg.get(zone, 0.40))
        if zone in ("corner3", "atb3"):
            three_weight += weight
            three_makes += made
        else:
            two_weight += weight
            two_makes += made
    two_fg = two_makes / two_weight if two_weight > 1e-6 else 0.52
    three_fg = three_makes / three_weight if three_weight > 1e-6 else 0.36
    return float(two_fg), float(three_fg)


def make_points(zone_freq: dict[str, float], zone_fg: dict[str, float]) -> tuple[float, float]:
    """Expected points given a make, and the share of makes that are threes."""
    two_fg, three_fg = zone_make_rates(zone_freq, zone_fg)
    two_weight = 0.0
    three_weight = 0.0
    for zone, share in zone_freq.items():
        weight = float(share) * float(zone_fg.get(zone, 0.40))
        if zone in ("corner3", "atb3"):
            three_weight += weight
        else:
            two_weight += weight
    total = two_weight + three_weight
    if total <= 0:
        return 2.2, 0.35
    three_share = three_weight / total
    return 2.0 + three_share, three_share
