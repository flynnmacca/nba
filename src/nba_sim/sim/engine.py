"""Seeded Monte Carlo on a quarter clock.

Each possession draws a play type from the on-court players, an actor from the
players who actually run that action, and a result against this opponent's
defense. Free throws come from a foul charged to a defender. Player points are
the team score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from nba_sim.constants import PLAY_TYPES
from nba_sim.features.minutes import MinutePlayer, draw_night_minutes, rotation_player_ids
from nba_sim.features.play_priors import position_key
from nba_sim.sim.adjust import offensive_residual
from nba_sim.sim.clock import play_game
from nba_sim.sim.specs import GameSpec, PlayerModel


@dataclass
class SimResult:
    home_points: np.ndarray
    away_points: np.ndarray
    player_stats: dict[int, dict[str, np.ndarray]]
    home_ids: list[int]
    away_ids: list[int]
    second_chance_home: np.ndarray
    second_chance_away: np.ndarray
    n_sims: int
    seed: int
    blowout: np.ndarray = field(default_factory=lambda: np.array([]))
    home_team_fouls: np.ndarray = field(default_factory=lambda: np.zeros(0))
    away_team_fouls: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home_max_quarter_fouls: np.ndarray = field(default_factory=lambda: np.zeros(0))
    away_max_quarter_fouls: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home_fta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    away_fta: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home_ftm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    away_ftm: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home_ft_trips: np.ndarray = field(default_factory=lambda: np.zeros(0))
    away_ft_trips: np.ndarray = field(default_factory=lambda: np.zeros(0))
    home_possessions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    away_possessions: np.ndarray = field(default_factory=lambda: np.zeros(0))
    intentional_fouls: np.ndarray = field(default_factory=lambda: np.zeros(0))
    flagrant_fouls: np.ndarray = field(default_factory=lambda: np.zeros(0))
    ejections: np.ndarray = field(default_factory=lambda: np.zeros(0))


def _rotation_indexes(players: list[PlayerModel]) -> set[int]:
    chosen = rotation_player_ids(players)
    return {index for index, player in enumerate(players) if player.player_id in chosen}


# A regular plays at least this many minutes. Below it, the prior is a spot minute
# and that player comes in when someone fouls out.
_REGULAR_MINUTES = 6.0


def _unit(
    players: list[PlayerModel],
    minutes: np.ndarray,
    kind: str,
    groups: dict[tuple[int, ...], float] | None = None,
) -> list[int]:
    """Who shares this stretch.

    A five that has actually played together is used when those five are
    available. Starters prefer the ESPN depth-chart opening five when that
    board is marked, else the group with the most minutes that still keeps
    the two highest-minute players, so a short unit does not take the opening
    when a star's old group is missing. The stagger keeps three or two of
    that five. The second unit shares at most two of them. With no lineup
    file, the five keeps two guards, two forwards, and a center when the
    rotation has them, and the rest of the script follows minutes.
    """
    available = [
        i
        for i, player in enumerate(players)
        if player.available and player.minutes_mean >= _REGULAR_MINUTES
    ]
    if len(available) < 5:
        available = [i for i, player in enumerate(players) if player.available and player.minutes_mean > 0]
    if len(available) < 5:
        available = [i for i, player in enumerate(players) if player.available]
    if len(available) < 5:
        available = list(range(min(5, len(players))))
    order = sorted(available, key=lambda i: float(minutes[i]), reverse=True)
    if len(order) < 5:
        order = (order + list(range(len(players))))[:5]
    if kind == "garbage":
        return _garbage_unit(players, minutes)
    projected = [
        index
        for index in order
        if bool(getattr(players[index], "projected_starter", False))
    ]
    starters = _observed_unit(players, available, "starters", groups, order, None)
    if starters is None and len(projected) >= 3:
        # Depth-chart openers win when the lineup file has no usable five.
        starters = _fill(projected, order)
    if starters is None:
        starters = _balanced_five(players, order)
    if kind == "starters":
        return starters
    starter_ids = {int(players[index].player_id) for index in starters}
    found = _observed_unit(players, available, kind, groups, order, starter_ids)
    if found is not None:
        return found
    rest = [index for index in order if index not in set(starters)]
    filler = list(reversed(starters))
    if kind == "bench":
        return _fill(rest[:5], filler)
    if kind == "bench_deep":
        # The five deepest regulars. On a rotation of ten or fewer this is the same bench.
        second = rest[:5] if len(order) <= 10 else rest[-5:]
        return _fill(second, filler)
    if kind == "mixed_early":
        return _fill(starters[:3], rest[:2], filler) if len(order) >= 7 else list(starters)
    extra = rest[:3]
    if len(order) > 10 and rest:
        extra = rest[:2] + [rest[-1]]
    return _fill(starters[:2], extra, filler) if len(order) >= 8 else list(starters)


def _fill(*parts: list[int]) -> list[int]:
    chosen: list[int] = []
    for part in parts:
        for index in part:
            if index in chosen:
                continue
            chosen.append(index)
            if len(chosen) == 5:
                return chosen
    return chosen


def _balanced_five(players: list[PlayerModel], order: list[int]) -> list[int]:
    """Two guards, two forwards, and a center, filled by minutes when a spot is empty."""
    want = ("G", "G", "F", "F", "C")
    chosen: list[int] = []
    for pos in want:
        for index in order:
            if index in chosen:
                continue
            if position_key(players[index].position) == pos:
                chosen.append(index)
                break
        if len(chosen) == 5:
            return chosen
    return _fill(chosen, order)


def indexes_of(by_id: dict[int, int], players: list[PlayerModel], ids: tuple[int, ...]) -> list[int]:
    chosen = [by_id[int(pid)] for pid in ids]
    chosen.sort(key=lambda index: float(players[index].minutes_mean), reverse=True)
    return chosen


def _observed_unit(
    players: list[PlayerModel],
    eligible: list[int],
    kind: str,
    groups: dict[tuple[int, ...], float] | None,
    order: list[int],
    starter_ids: set[int] | None,
) -> list[int] | None:
    """The five-man group from the lineup file that matches this stretch."""
    if not groups:
        return None
    by_id = {int(players[index].player_id): index for index in eligible}
    usable: list[tuple[float, tuple[int, ...]]] = []
    for ids, played in groups.items():
        if float(played) < 48.0 or len(ids) != 5:
            continue
        if not all(int(pid) in by_id for pid in ids):
            continue
        usable.append((float(played), tuple(int(pid) for pid in ids)))
    if not usable:
        return None
    usable.sort(reverse=True)
    if starter_ids is None:
        anchors = []
        for index in order:
            pid = int(players[index].player_id)
            if pid in by_id:
                anchors.append(pid)
            if len(anchors) == 2:
                break
        needed = set(anchors)
        starter_ids = None
        for _played, ids in usable:
            if needed.issubset(set(ids)):
                starter_ids = set(ids)
                break
        if starter_ids is None:
            return None
    if kind == "starters":
        return indexes_of(by_id, players, tuple(sorted(starter_ids)))

    def indexes(ids: tuple[int, ...]) -> list[int]:
        return indexes_of(by_id, players, ids)

    def best(match) -> list[int] | None:
        for _played, ids in usable:
            if set(ids) == starter_ids:
                continue
            if match(set(ids)):
                return indexes(ids)
        return None

    def apart(limit: int, skip: set[int]) -> list[int] | None:
        ranked: list[tuple[int, float, tuple[int, ...]]] = []
        for played, ids in usable:
            chosen = set(ids)
            if chosen == starter_ids or chosen == skip:
                continue
            overlap = len(chosen & starter_ids)
            if overlap <= limit:
                ranked.append((overlap, -played, ids))
        if not ranked:
            return None
        ranked.sort()
        return indexes(ranked[0][2])

    if kind == "mixed_early":
        return best(lambda ids: len(ids & starter_ids) == 3)
    if kind == "mixed_late":
        return best(lambda ids: len(ids & starter_ids) == 2)
    if kind == "bench":
        return apart(2, set())
    if kind == "bench_deep":
        bench = apart(2, set())
        bench_ids = {int(players[index].player_id) for index in bench} if bench else set()
        return apart(1, bench_ids)
    return None


def _garbage_unit(players: list[PlayerModel], minutes: np.ndarray) -> list[int]:
    """The end of a blowout. Lowest minute totals, with the closers off the floor."""
    pool = [
        i
        for i, player in enumerate(players)
        if player.available and player.minutes_mean > 0 and float(minutes[i]) > 0
    ]
    if len(pool) < 5:
        pool = [i for i, player in enumerate(players) if player.available] or list(range(min(5, len(players))))
    ranked = sorted(pool, key=lambda i: float(minutes[i]), reverse=True)
    # A full rotation can sit every closer. A short one still keeps its top of the group off.
    if len(ranked) >= 10:
        protect = 5
    elif len(ranked) >= 8:
        protect = 3
    elif len(ranked) >= 6:
        protect = 1
    else:
        protect = 0
    blocked = set(ranked[:protect])
    chosen: list[int] = []
    for index in reversed(ranked):
        if index in chosen or index in blocked:
            continue
        chosen.append(index)
        if len(chosen) == 5:
            break
    if len(chosen) < 5:
        for index in reversed(ranked):
            if index not in chosen:
                chosen.append(index)
            if len(chosen) == 5:
                break
    return chosen[:5]


def _minute_spread(player: PlayerModel) -> float:
    """How wide this player's minutes are.

    A game log of his own pulls the width toward that sample. With no log, a
    closer sits in a narrow band and a reserve's night is the wide one.
    """
    mean = float(player.minutes_mean)
    # A game log replaces this. Closers move less than reserves, and the
    # width is a night, so a 32-minute player is not pinned to 32.
    if mean >= 32.0:
        role = 4.2
    elif mean >= 22.0:
        role = 5.2
    elif mean >= 10.0:
        role = 6.4
    else:
        role = 5.0
    sample = int(player.minutes_sample)
    own = float(player.minutes_std)
    if sample < 1 or own <= 0:
        return role
    weight = min(sample, 8) / 8.0
    return role * (1.0 - weight) + own * weight


def _draw_talent(
    rng: np.random.Generator,
    players: list[PlayerModel],
    n_sims: int,
) -> np.ndarray:
    means = np.ones((len(players), len(PLAY_TYPES)), dtype=float)
    poss = np.ones((len(players), len(PLAY_TYPES)), dtype=float)
    for i, player in enumerate(players):
        # Possessions per game are the mix and the sample. Usage is applied
        # when the trip is drawn. Uncertainty is in the season rate.
        total = sum(max(skill.poss, 0.0) for skill in player.offense.values())
        per_game = total < 80.0
        games = max(float(player.sample_games), 1.0)
        for t, name in enumerate(PLAY_TYPES):
            skill = player.offense.get(name)
            if skill is None:
                continue
            means[i, t] = skill.ppp
            sample = max(skill.poss, 1.0)
            if per_game:
                sample *= games
            poss[i, t] = sample
    sd = 0.26 / np.sqrt(poss / 25.0)
    sd = np.clip(sd, 0.012, 0.19)
    draw = rng.normal(means, sd, size=(n_sims, len(players), len(PLAY_TYPES)))
    return np.clip(draw, 0.50, 1.75)


BOX_KEYS = (
    "pts",
    "reb",
    "oreb",
    "dreb",
    "ast",
    "pot_ast",
    "stl",
    "blk",
    "tpm",
    "tpa",
    "fgm",
    "fga",
    "ftm",
    "fta",
    "tov",
    "pf",
)



def _floor_minutes(seconds: np.ndarray, plan_seconds: np.ndarray, drawn: np.ndarray, profiled: bool = False) -> np.ndarray:
    """Minutes he spent on the floor.

    A profiled game publishes the tracked floor time. Five men for 48 minutes
    is 240, and overtime adds the extra stint. A game without profiles still
    scales a tracked floor back to 240, and falls back to the draw when the
    clock was not tracked.
    """
    floor = np.asarray(seconds, dtype=float) / 60.0
    if profiled:
        return floor
    del plan_seconds
    base = np.asarray(drawn, dtype=float)
    totals = floor.sum(axis=1, keepdims=True)
    good = totals.ravel() > 50.0
    scaled = floor * (240.0 / np.maximum(totals, 1e-6))
    out = np.array(base, dtype=float, copy=True)
    out[good] = scaled[good]
    return out


def _box_from(table) -> dict[str, np.ndarray]:
    return {key: getattr(table, key) for key in BOX_KEYS}


def simulate_game(spec: GameSpec, n_sims: int = 20_000, seed: int = 1) -> SimResult:
    rng = np.random.default_rng(seed)
    home_players = spec.home_players
    away_players = spec.away_players
    home_rotation = _rotation_indexes(home_players)
    away_rotation = _rotation_indexes(away_players)
    home_min_players = [
        MinutePlayer(
            p.player_id,
            p.minutes_mean if i in home_rotation else 0.0,
            _minute_spread(p) if i in home_rotation else 0.0,
            i in home_rotation,
        )
        for i, p in enumerate(home_players)
    ]
    away_min_players = [
        MinutePlayer(
            p.player_id,
            p.minutes_mean if i in away_rotation else 0.0,
            _minute_spread(p) if i in away_rotation else 0.0,
            i in away_rotation,
        )
        for i, p in enumerate(away_players)
    ]
    home_minutes = draw_night_minutes(home_min_players, n_sims, rng)
    away_minutes = draw_night_minutes(away_min_players, n_sims, rng)
    profiled = any(getattr(player, "court", None) is not None for player in list(home_players) + list(away_players))
    if profiled:
        from nba_sim.features.court import availability_mask

        flags = [str(flag).lower() for flag in (spec.context.flags or [])]
        spec.home_active = availability_mask(
            home_players,
            n_sims,
            rng,
            b2b=any("home" in flag and "back-to-back" in flag for flag in flags),
            three=any("home" in flag and "three-in-four" in flag for flag in flags),
        )
        spec.away_active = availability_mask(
            away_players,
            n_sims,
            rng,
            b2b=any("away" in flag and "back-to-back" in flag for flag in flags),
            three=any("away" in flag and "three-in-four" in flag for flag in flags),
        )
    if profiled:
        # The whistle comes from the profiles. Building the scripted fives
        # would leave a second rotation the clock never scores.
        home_units = None
        away_units = None
        home_targets = None
        away_targets = None
    else:
        home_mean = np.array([p.minutes_mean if i in home_rotation else 0.0 for i, p in enumerate(home_players)])
        away_mean = np.array([p.minutes_mean if i in away_rotation else 0.0 for i, p in enumerate(away_players)])
        home_groups = spec.home.lineup_minutes
        away_groups = spec.away.lineup_minutes
        home_units = {
            "starters": _unit(home_players, home_mean, "starters", home_groups),
            "mixed_early": _unit(home_players, home_mean, "mixed_early", home_groups),
            "mixed_late": _unit(home_players, home_mean, "mixed_late", home_groups),
            "bench": _unit(home_players, home_mean, "bench", home_groups),
            "bench_deep": _unit(home_players, home_mean, "bench_deep", home_groups),
            "garbage": _unit(home_players, home_mean, "garbage", home_groups),
        }
        away_units = {
            "starters": _unit(away_players, away_mean, "starters", away_groups),
            "mixed_early": _unit(away_players, away_mean, "mixed_early", away_groups),
            "mixed_late": _unit(away_players, away_mean, "mixed_late", away_groups),
            "bench": _unit(away_players, away_mean, "bench", away_groups),
            "bench_deep": _unit(away_players, away_mean, "bench_deep", away_groups),
            "garbage": _unit(away_players, away_mean, "garbage", away_groups),
        }
        home_targets = home_minutes
        away_targets = away_minutes
    home_draw = _draw_talent(rng, home_players, n_sims)
    away_draw = _draw_talent(rng, away_players, n_sims)
    # One correlated night for the whole offense, applied on the make move
    # inside the clock — not on the PPP table, where five-man squad residuals
    # would cancel the same shock.
    home_night = np.clip(rng.normal(1.0, 0.09, size=n_sims), 0.82, 1.20)
    away_night = np.clip(rng.normal(1.0, 0.09, size=n_sims), 0.82, 1.20)
    home_residual = offensive_residual(home_players, spec.home)
    away_residual = offensive_residual(away_players, spec.away)
    # Stretch relative ORTG gaps, then center the geometric mean at 1 so both
    # sides do not heat league FG% together. Gain widens between-game mean
    # margins without fattening within-sim night noise.
    residual_gain = 3.2
    home_residual = 1.0 + residual_gain * (home_residual - 1.0)
    away_residual = 1.0 + residual_gain * (away_residual - 1.0)
    geo = float(np.sqrt(max(home_residual * away_residual, 1e-9)))
    if geo > 0:
        home_residual = float(np.clip(home_residual / geo, 0.80, 1.25))
        away_residual = float(np.clip(away_residual / geo, 0.80, 1.25))
    # Deterministic matchup tilt from rating gap — shifts game *means*, which
    # is what sim_margin_sd measures across the slate.
    gap = (float(spec.home.off_rating) - float(spec.away.off_rating)) - (
        float(spec.home.def_rating) - float(spec.away.def_rating)
    )
    tilt = float(np.clip(gap / 400.0, -0.065, 0.065))
    home_residual = float(np.clip(home_residual * (1.0 + tilt), 0.80, 1.25))
    away_residual = float(np.clip(away_residual * (1.0 - tilt), 0.80, 1.25))
    played = play_game(
        rng,
        spec,
        home_units,
        away_units,
        home_draw,
        away_draw,
        home_residual,
        away_residual,
        home_targets,
        away_targets,
        home_night=home_night,
        away_night=away_night,
    )
    home_box = _box_from(played.home)
    away_box = _box_from(played.away)
    home_box["minutes"] = _floor_minutes(played.home_floor, played.home_plan, home_minutes, profiled)
    away_box["minutes"] = _floor_minutes(played.away_floor, played.away_plan, away_minutes, profiled)
    second_home = played.second_home
    second_away = played.second_away
    use_bench = played.blowout

    def pack(players: list[PlayerModel], box: dict[str, np.ndarray]) -> dict[int, dict[str, np.ndarray]]:
        packed: dict[int, dict[str, np.ndarray]] = {}
        for index, player in enumerate(players):
            pts = box["pts"][:, index]
            reb = box["reb"][:, index]
            ast = box["ast"][:, index]
            packed[player.player_id] = {
                "pts": pts,
                "reb": reb,
                "oreb": box["oreb"][:, index],
                "dreb": box["dreb"][:, index],
                "ast": ast,
                "pot_ast": box["pot_ast"][:, index],
                "stl": box["stl"][:, index],
                "blk": box["blk"][:, index],
                "tpm": box["tpm"][:, index],
                "tpa": box["tpa"][:, index],
                "fgm": box["fgm"][:, index],
                "fga": box["fga"][:, index],
                "ftm": box["ftm"][:, index],
                "fta": box["fta"][:, index],
                "tov": box["tov"][:, index],
                "pf": box["pf"][:, index],
                "minutes": box["minutes"][:, index],
                "pra": pts + reb + ast,
            }
        return packed

    home_points = home_box["pts"].sum(axis=1)
    away_points = away_box["pts"].sum(axis=1)
    stats = pack(home_players, home_box)
    stats.update(pack(away_players, away_box))
    return SimResult(
        home_points=home_points,
        away_points=away_points,
        player_stats=stats,
        home_ids=[p.player_id for p in home_players],
        away_ids=[p.player_id for p in away_players],
        second_chance_home=second_home,
        second_chance_away=second_away,
        n_sims=n_sims,
        seed=seed,
        blowout=use_bench,
        home_team_fouls=played.home_team_fouls,
        away_team_fouls=played.away_team_fouls,
        home_max_quarter_fouls=played.home_max_quarter_fouls,
        away_max_quarter_fouls=played.away_max_quarter_fouls,
        home_fta=played.home.fta.sum(axis=1),
        away_fta=played.away.fta.sum(axis=1),
        home_ftm=played.home.ftm.sum(axis=1),
        away_ftm=played.away.ftm.sum(axis=1),
        home_ft_trips=played.home_ft_trips,
        away_ft_trips=played.away_ft_trips,
        home_possessions=played.home_possessions,
        away_possessions=played.away_possessions,
        intentional_fouls=played.intentional_fouls,
        flagrant_fouls=played.flagrant_fouls,
        ejections=played.ejections,
    )
