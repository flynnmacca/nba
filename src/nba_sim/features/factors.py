"""Play-type gaps, rebound matchups, and the factor list for one game."""

from __future__ import annotations

import numpy as np

from nba_sim.constants import PLAY_TYPES
from nba_sim.features.minutes import rotation_player_ids
from nba_sim.sim.adjust import oreb_probability
from nba_sim.sim.engine import _unit
from nba_sim.sim.specs import GameSpec, PlayerModel


def _lineup(players: list[PlayerModel]) -> list[PlayerModel]:
    available = [p for p in players if p.available]
    return sorted(available, key=lambda p: p.minutes_mean, reverse=True)[:5]


def play_type_share(player: PlayerModel) -> str:
    total = sum(skill.poss for skill in player.offense.values())
    if total <= 0:
        return ""
    ranked = sorted(player.offense.items(), key=lambda item: item[1].poss, reverse=True)
    parts = []
    for name, skill in ranked[:3]:
        if skill.poss <= 0:
            continue
        parts.append(f"{name} {100.0 * skill.poss / total:.0f}%")
    return ", ".join(parts)


def _rotation(players: list[PlayerModel]) -> list[PlayerModel]:
    chosen = rotation_player_ids(players)
    group = [player for player in players if player.player_id in chosen and player.available]
    if len(group) >= 5:
        return group
    return sorted((player for player in players if player.available), key=lambda player: player.minutes_mean, reverse=True)[:5]


def _play_profile(players: list[PlayerModel], side: str) -> dict[str, tuple[float, float]]:
    """Minutes-weighted PPP and possession share for each play."""
    weight = {play: 0.0 for play in PLAY_TYPES}
    scored = {play: 0.0 for play in PLAY_TYPES}
    for player in players:
        minutes = max(float(player.minutes_mean), 0.0)
        if minutes <= 0:
            continue
        skills = player.offense if side == "offense" else player.defense
        claim = max(float(player.usg), 0.04)
        total = sum(max(skill.poss, 0.0) for skill in skills.values())
        for play in PLAY_TYPES:
            skill = skills.get(play)
            if skill is None or skill.poss <= 0:
                continue
            if side == "offense" and total > 0:
                share = minutes * claim * (float(skill.poss) / total)
            else:
                share = minutes * float(skill.poss)
            weight[play] += share
            scored[play] += share * float(skill.ppp)
    total = sum(weight.values())
    profile = {}
    for play in PLAY_TYPES:
        if weight[play] <= 0 or total <= 0:
            continue
        profile[play] = (scored[play] / weight[play], weight[play] / total)
    return profile


def play_type_gaps(spec: GameSpec) -> list[dict]:
    """Lineup play rates. Offense is who runs it. Defense is who is guarding it."""
    profiles = {
        "home": {
            "offense": _play_profile(_rotation(spec.home_players), "offense"),
            "defense": _play_profile(_rotation(spec.home_players), "defense"),
        },
        "away": {
            "offense": _play_profile(_rotation(spec.away_players), "offense"),
            "defense": _play_profile(_rotation(spec.away_players), "defense"),
        },
    }
    measured = sum(spec.home.off_poss.values()) + sum(spec.away.off_poss.values()) > 0
    rows = []
    for side, offense_team, defense_team in (
        ("home", spec.home, spec.away),
        ("away", spec.away, spec.home),
    ):
        offense = profiles[side]["offense"]
        defense = profiles["away" if side == "home" else "home"]["defense"]
        for play in PLAY_TYPES:
            off = offense.get(play)
            if off is None:
                team_ppp = offense_team.off_ppp.get(play)
                if team_ppp is None:
                    continue
                off = (float(team_ppp), 1.0 / len(PLAY_TYPES))
            defended = defense.get(play)
            if defended is None:
                team_def = defense_team.def_ppp.get(play)
                if team_def is None:
                    continue
                defended = (float(team_def), 0.0)
            league = spec.league.ppp_by_type.get(play, 1.0) or 1.0
            rows.append(
                {
                    "side": side,
                    "team": offense_team.abbr.upper(),
                    "opponent": defense_team.abbr.upper(),
                    "play_type": play,
                    "off_ppp": off[0],
                    "def_ppp": defended[0],
                    "off_share": off[1],
                    "expected": off[0] * (defended[0] / league),
                    "off_poss": offense_team.off_poss.get(play, 0.0),
                    "def_poss": defense_team.def_poss.get(play, 0.0),
                    "source": "player and team" if measured else "position prior",
                }
            )
    return rows


def _opening(players: list[PlayerModel], groups: dict) -> list[PlayerModel]:
    """The five that takes the floor, the same selection the simulator opens with."""
    if len(players) < 5:
        return _lineup(players)
    minutes = np.array([float(player.minutes_mean) for player in players], dtype=float)
    indexes = _unit(players, minutes, "starters", groups or None)
    return [players[index] for index in indexes]


def rebound_snapshot(spec: GameSpec) -> dict:
    home = _opening(spec.home_players, spec.home.lineup_minutes)
    away = _opening(spec.away_players, spec.away.lineup_minutes)
    return {
        "home_oreb": sum(p.oreb_pct for p in home),
        "home_dreb": sum(p.dreb_pct for p in home),
        "away_oreb": sum(p.oreb_pct for p in away),
        "away_dreb": sum(p.dreb_pct for p in away),
        "home_box_outs": sum(p.box_outs for p in home),
        "away_box_outs": sum(p.box_outs for p in away),
        "home_chance": oreb_probability(
            home, away, spec.home.oreb_pct, spec.away.dreb_pct, spec.away.oreb_allowed
        ),
        "away_chance": oreb_probability(
            away, home, spec.away.oreb_pct, spec.home.dreb_pct, spec.home.oreb_allowed
        ),
    }


def _floor_impact(players: list[PlayerModel]) -> tuple[float, float]:
    if not players:
        return 1.0, 1.0
    return (
        sum(player.off_impact for player in players) / len(players),
        sum(player.def_impact for player in players) / len(players),
    )


def matchup_factor_rows(spec: GameSpec) -> list[dict]:
    attempts = 0.0
    pairs = 0
    for player in spec.home_players + spec.away_players:
        for _defender, sample in player.matchup_fg.items():
            attempts += sample[1]
            pairs += 1
    home_floor = _floor_impact(_lineup(spec.home_players))
    away_floor = _floor_impact(_lineup(spec.away_players))
    floor = (
        f" On the floor, {spec.home.abbr.upper()} offense {home_floor[0]:.3f} and defense {home_floor[1]:.3f}"
        f" of its rating; {spec.away.abbr.upper()} offense {away_floor[0]:.3f} and defense {away_floor[1]:.3f}."
    )
    if pairs == 0:
        detail = (
            "Each defender's play-type profile and the zones he contests set the five-man defense. "
            "The player matched to the action, by position, moves that trip."
            + floor
        )
        sample = 0.0
    else:
        detail = f"{pairs} defender samples, {attempts:.0f} field-goal attempts, shrunk toward play type." + floor
        sample = attempts
    rows = [
        {
            "name": "individual_matchup",
            "team_ids": [spec.home.team_id, spec.away.team_id],
            "player_ids": [],
            "sample_size": sample,
            "value": 1.0,
            "detail": detail,
        }
    ]
    glass = rebound_snapshot(spec)
    rows.append(
        {
            "name": "offensive_rebounding",
            "team_ids": [spec.home.team_id, spec.away.team_id],
            "player_ids": [
                p.player_id
                for p in _opening(spec.home_players, spec.home.lineup_minutes)
                + _opening(spec.away_players, spec.away.lineup_minutes)
            ],
            "sample_size": glass["home_oreb"] + glass["away_oreb"],
            "value": glass["home_oreb"],
            "detail": (
                f"Opening five OREB share {spec.home.abbr.upper()} {glass['home_oreb']:.3f} "
                f"vs {spec.away.abbr.upper()} {glass['away_oreb']:.3f}; "
                f"DREB share {glass['home_dreb']:.3f} vs {glass['away_dreb']:.3f}. "
                f"Team rates {spec.home.oreb_pct:.3f} and {spec.away.oreb_pct:.3f}, "
                f"and the rates they give up {spec.home.oreb_allowed:.3f} and {spec.away.oreb_allowed:.3f}, "
                f"pull the chance to {glass['home_chance']:.0%} and {glass['away_chance']:.0%}"
            ),
        }
    )
    return rows


def explain(spec: GameSpec) -> list[dict]:
    return list(spec.context.factors) + matchup_factor_rows(spec)
