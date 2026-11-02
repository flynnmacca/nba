"""Text and dashboard summaries of one simulated game."""

from __future__ import annotations

import numpy as np

from nba_sim.features.factors import explain, play_type_gaps, play_type_share, rebound_snapshot
from nba_sim.features.minutes import fill_night
from nba_sim.markets.edge import price_quotes
from nba_sim.sim.engine import SimResult
from nba_sim.sim.specs import GameSpec


def _integer_counts(values: np.ndarray, start_at_zero: bool = False) -> list[dict]:
    """One row per scoring number, including numbers in the middle that no sim hit."""
    rounded = np.rint(np.asarray(values, dtype=float)).astype(int)
    if rounded.size == 0:
        return []
    low = 0 if start_at_zero else int(rounded.min())
    high = int(rounded.max())
    if high < low:
        high = low
    counts = np.bincount(rounded - low, minlength=high - low + 1)
    rows = [{"value": low + offset, "sims": int(count)} for offset, count in enumerate(counts)]
    while rows and rows[0]["sims"] == 0:
        rows.pop(0)
    while rows and rows[-1]["sims"] == 0:
        rows.pop()
    return rows


def _histogram(values: np.ndarray, bins: int = 18) -> dict:
    counts, edges = np.histogram(values, bins=bins)
    return {
        "edges": [float(value) for value in edges],
        "counts": [int(value) for value in counts],
        "centers": [float(value) for value in (edges[:-1] + edges[1:]) / 2],
    }


def _score_bins(home: np.ndarray, away: np.ndarray, home_name: str, away_name: str, bins: int = 16) -> list[dict]:
    low = float(min(home.min(), away.min()))
    high = float(max(home.max(), away.max()))
    if high <= low:
        high = low + 1.0
    edges = np.linspace(low, high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    rows = []
    for team, values in ((home_name, home), (away_name, away)):
        counts, _ = np.histogram(values, bins=edges)
        for center, count in zip(centers, counts):
            rows.append({"team": team, "points": round(float(center), 1), "sims": int(count)})
    return rows


def summarize(spec: GameSpec, result: SimResult) -> dict:
    total = result.home_points + result.away_points
    margin = result.home_points - result.away_points
    total_hist = _histogram(total)
    margin_hist = _histogram(margin)
    gaps = play_type_gaps(spec)
    home_abbr = spec.home.abbr.upper()
    away_abbr = spec.away.abbr.upper()
    return {
        "game_id": spec.game_id,
        "date": spec.date,
        "source": spec.source,
        "home": spec.home.abbr.upper(),
        "away": spec.away.abbr.upper(),
        "home_name": spec.home.name,
        "away_name": spec.away.name,
        "flags": list(spec.context.flags),
        "factors": explain(spec),
        "rebounds": rebound_snapshot(spec),
        "matchups": gaps,
        "team": {
            "home_mean": float(result.home_points.mean()),
            "away_mean": float(result.away_points.mean()),
            "total_mean": float(total.mean()),
            "spread_mean": float(margin.mean()),
            "home_p10": float(np.quantile(result.home_points, 0.10)),
            "home_p90": float(np.quantile(result.home_points, 0.90)),
            "away_p10": float(np.quantile(result.away_points, 0.10)),
            "away_p90": float(np.quantile(result.away_points, 0.90)),
            "total_p10": float(np.quantile(total, 0.10)),
            "total_p90": float(np.quantile(total, 0.90)),
            "home_win_prob": float(np.mean(result.home_points > result.away_points)),
            "away_win_prob": float(np.mean(result.away_points > result.home_points)),
            "hist_total": total_hist,
            "score_bins": _score_bins(result.home_points, result.away_points, home_abbr, away_abbr),
            "score_counts": (
                [{"team": away_abbr, **row} for row in _integer_counts(result.away_points)]
                + [{"team": home_abbr, **row} for row in _integer_counts(result.home_points)]
            ),
            "total_counts": _integer_counts(total),
            "spread_counts": _integer_counts(margin),
            "total_bins": [
                {"points": center, "sims": count}
                for center, count in zip(total_hist["centers"], total_hist["counts"])
            ],
            "margin_bins": [
                {"margin": center, "sims": count}
                for center, count in zip(margin_hist["centers"], margin_hist["counts"])
            ],
        },
        "players": _players(spec, result),
        "fouls": _whistle_summary(result),
        "box_score": _box_score(spec, result),
        "player_bins": _player_bins(spec, result),
        "minutes": _minutes_table(spec, result),
        "prices": price_quotes(spec, result),
        "n_sims": result.n_sims,
        "seed": result.seed,
        "venue": spec.venue,
        "city": spec.city,
        "tip_local": spec.tip_local,
        "tip_melbourne": spec.tip_melbourne,
        "game_status": spec.game_status,
        "injuries": list(spec.injuries),
    }


def _players(spec: GameSpec, result: SimResult) -> list[dict]:
    rows = []
    for team_abbr, ids in ((spec.home.abbr.upper(), result.home_ids), (spec.away.abbr.upper(), result.away_ids)):
        roster = spec.home_players if team_abbr == spec.home.abbr.upper() else spec.away_players
        by_id = {player.player_id: player for player in roster}
        for player_id in ids:
            stats = result.player_stats[player_id]
            player = by_id[player_id]
            rows.append(
                {
                    "player_id": player_id,
                    "player": player.name,
                    "team": team_abbr,
                    "position": player.position,
                    "minutes": float(stats["minutes"].mean()),
                    "pts": float(stats["pts"].mean()),
                    "pts_p10": float(np.quantile(stats["pts"], 0.10)),
                    "pts_p90": float(np.quantile(stats["pts"], 0.90)),
                    "reb": float(stats["reb"].mean()),
                    "reb_p10": float(np.quantile(stats["reb"], 0.10)),
                    "reb_p90": float(np.quantile(stats["reb"], 0.90)),
                    "ast": float(stats["ast"].mean()),
                    "ast_p10": float(np.quantile(stats["ast"], 0.10)),
                    "ast_p90": float(np.quantile(stats["ast"], 0.90)),
                    "tpm": float(stats["tpm"].mean()),
                    "pra": float(stats["pra"].mean()),
                    "pra_p10": float(np.quantile(stats["pra"], 0.10)),
                    "pra_p90": float(np.quantile(stats["pra"], 0.90)),
                    "play_types": play_type_share(player),
                    "usg": float(player.usg),
                    "available": player.available,
                }
            )
    return rows


def _whistle_summary(result: SimResult) -> dict | None:
    if result.home_team_fouls.size != result.n_sims or result.n_sims == 0:
        return None

    def line(makes: np.ndarray, attempts: np.ndarray) -> str:
        return f"{float(makes.mean()):.1f}/{float(attempts.mean()):.1f}"

    return {
        "home_fouls": float(result.home_team_fouls.mean()),
        "away_fouls": float(result.away_team_fouls.mean()),
        "home_ft": line(result.home_ftm, result.home_fta),
        "away_ft": line(result.away_ftm, result.away_fta),
        "home_poss": float(result.home_possessions.mean()),
        "away_poss": float(result.away_possessions.mean()),
        "intentional": float(result.intentional_fouls.mean()),
        "flagrant": float(result.flagrant_fouls.mean()) if result.flagrant_fouls.size else 0.0,
        "ejections": float(result.ejections.mean()) if result.ejections.size else 0.0,
    }


def _box_mean(stats: dict, key: str) -> float:
    values = stats.get(key)
    if values is None:
        return 0.0
    return float(np.mean(values))


def _box_row(player: str, stats: dict, total: bool = False) -> dict:
    return {
        "player": player,
        "total": total,
        "min": round(_box_mean(stats, "minutes"), 1),
        "pts": round(_box_mean(stats, "pts"), 1),
        "fgm": round(_box_mean(stats, "fgm"), 1),
        "fga": round(_box_mean(stats, "fga"), 1),
        "tpm": round(_box_mean(stats, "tpm"), 1),
        "tpa": round(_box_mean(stats, "tpa"), 1),
        "ftm": round(_box_mean(stats, "ftm"), 1),
        "fta": round(_box_mean(stats, "fta"), 1),
        "reb": round(_box_mean(stats, "reb"), 1),
        "oreb": round(_box_mean(stats, "oreb"), 1),
        "dreb": round(_box_mean(stats, "dreb"), 1),
        "ast": round(_box_mean(stats, "ast"), 1),
        "pot_ast": round(_box_mean(stats, "pot_ast"), 1),
        "stl": round(_box_mean(stats, "stl"), 1),
        "blk": round(_box_mean(stats, "blk"), 1),
        "tov": round(_box_mean(stats, "tov"), 1),
        "pf": round(_box_mean(stats, "pf"), 1),
    }


def _box_score(spec: GameSpec, result: SimResult) -> list[dict]:
    """Mean box score for each team, rotation first, then the team total."""
    rows = []
    sides = (
        (spec.away.abbr.upper(), result.away_ids, spec.away_players),
        (spec.home.abbr.upper(), result.home_ids, spec.home_players),
    )
    made_keys = ("fgm", "fga", "tpm", "tpa", "ftm", "fta", "reb", "oreb", "dreb", "ast", "pot_ast", "stl", "blk", "tov", "pf", "pts", "minutes")
    for team_abbr, ids, roster in sides:
        by_id = {player.player_id: player for player in roster}
        team_rows = []
        totals = {key: 0.0 for key in made_keys}
        for player_id in ids:
            stats = result.player_stats.get(player_id)
            player = by_id.get(player_id)
            if stats is None or player is None:
                continue
            for key in made_keys:
                totals[key] += _box_mean(stats, key)
            row = _box_row(player.name, stats)
            if row["min"] >= 0.5 or row["pts"] >= 0.2:
                team_rows.append({"team": team_abbr, **row})
        team_rows.sort(key=lambda row: row["min"], reverse=True)
        total_stats = {key: np.array([value]) for key, value in totals.items()}
        rows.extend(team_rows)
        rows.append({"team": team_abbr, **_box_row("Total", total_stats, total=True)})
    return rows


def _player_bins(spec: GameSpec, result: SimResult, minute_floor: float = 4.0, limit: int = 15) -> list[dict]:
    """Integer histograms for the rotation, one series per player and stat."""
    rows = []
    for team_abbr, ids, roster in (
        (spec.away.abbr.upper(), result.away_ids, spec.away_players),
        (spec.home.abbr.upper(), result.home_ids, spec.home_players),
    ):
        by_id = {player.player_id: player for player in roster}
        ranked = []
        for player_id in ids:
            stats = result.player_stats[player_id]
            minutes = float(stats["minutes"].mean())
            if minutes < minute_floor:
                continue
            ranked.append((minutes, player_id, stats, by_id[player_id]))
        ranked.sort(key=lambda item: item[0], reverse=True)
        for minutes, _player_id, stats, player in ranked[:limit]:
            defense_play, defense_ppp = _best_defense(player)
            for stat in ("pts", "reb", "ast", "tpm", "pra"):
                values = stats[stat]
                mean = float(values.mean())
                low = float(np.quantile(values, 0.10))
                high = float(np.quantile(values, 0.90))
                for bucket in _integer_counts(values, start_at_zero=True):
                    rows.append(
                        {
                            "player": player.name,
                            "espn_id": player.espn_id,
                            "team": team_abbr,
                            "position": player.position,
                            "stat": stat,
                            "minutes": round(minutes, 1),
                            "usg": round(float(player.usg), 3),
                            "floor_off": round(float(player.off_impact), 3),
                            "floor_def": round(float(player.def_impact), 3),
                            "def_play": defense_play,
                            "def_ppp": round(defense_ppp, 3),
                            "ft_draw": round(float(player.ft_draw), 3),
                            "foul_rate": round(float(player.foul_rate), 3),
                            "three": round(
                                float(player.zone_freq.get("corner3", 0.0) + player.zone_freq.get("atb3", 0.0)),
                                3,
                            ),
                            "mean": round(mean, 1),
                            "p10": round(low, 1),
                            "p90": round(high, 1),
                            "value": bucket["value"],
                            "sims": bucket["sims"],
                        }
                    )
    return rows


def _minutes_table(spec: GameSpec, result: SimResult) -> list[dict]:
    """Every roster player, from the minutes prior through the simulated mean."""
    rows = []
    sides = (
        (spec.away.abbr.upper(), spec.away_players, spec.away),
        (spec.home.abbr.upper(), spec.home_players, spec.home),
    )
    for team_abbr, roster, team in sides:
        means = np.array(
            [
                float(player.minutes_mean) if player.available and player.minutes_mean > 0 else 0.0
                for player in roster
            ]
        )
        profiled = any(getattr(player, "court", None) is not None for player in roster)
        filled = means if profiled else fill_night(means)
        for index, player in enumerate(roster):
            stats = result.player_stats.get(player.player_id)
            if stats is None:
                sim = low = high = 0.0
            else:
                played = stats["minutes"]
                sim = float(played.mean())
                low = float(np.quantile(played, 0.10))
                high = float(np.quantile(played, 0.90))
            base = player.minutes_base if player.minutes_base > 0 else player.minutes_mean
            prior = float(player.minutes_mean)
            scaled = float(filled[index])
            how = _minutes_how(player)
            rows.append(
                {
                    "team": team_abbr,
                    "player": player.name,
                    "position": player.position,
                    "how": how,
                    "base": round(base, 1),
                    "injury": _minutes_injury(player),
                    "prior": round(prior, 1),
                    "scaled": round(scaled, 1),
                    "sim": round(sim, 1),
                    "low": round(low, 1),
                    "high": round(high, 1),
                    "projected": sim >= 4.0,
                    "usg": float(player.usg),
                    "coach": team.coach,
                    "rotation_size": int(team.rotation_size),
                    "fives": _five_text(player, team),
                }
            )
    return rows


def _as_float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def floor_bits(info) -> list[str]:
    """One sentence of on-court offense and defense, when the rating actually moves."""
    bits = []
    off = _as_float(info.get("floor_off")) if hasattr(info, "get") else None
    deff = _as_float(info.get("floor_def")) if hasattr(info, "get") else None
    if off is not None and deff is not None and (abs(off - 1.0) > 0.005 or abs(deff - 1.0) > 0.005):
        bits.append(
            f"With him on the floor the offense is {off:.0%} of its usual rate and the defense allows {deff:.0%}."
        )
    play = str(info.get("def_play") or "") if hasattr(info, "get") else ""
    ppp = _as_float(info.get("def_ppp")) if hasattr(info, "get") else None
    if play and ppp is not None and ppp > 0:
        bits.append(f"His best defensive action is {play} at {ppp:.2f} points allowed per possession.")
    draw = _as_float(info.get("ft_draw")) if hasattr(info, "get") else None
    if draw is not None and abs(draw - 1.0) >= 0.08:
        bits.append(f"He draws free throws at {draw:.0%} of the rate his actions already carry.")
    foul = _as_float(info.get("foul_rate")) if hasattr(info, "get") else None
    if foul is not None and foul > 0 and abs(foul - 1.0) >= 0.12:
        bits.append(f"He fouls at {foul:.0%} of a league rate, and that is what he concedes on the matchup.")
    return bits


def _best_defense(player) -> tuple[str, float]:
    best_play = ""
    best_ppp = 0.0
    for play, skill in player.defense.items():
        if skill.poss < 20:
            continue
        if not best_play or skill.ppp < best_ppp:
            best_play = play
            best_ppp = float(skill.ppp)
    return best_play, best_ppp


def _minutes_how(player) -> str:
    source = player.minutes_source
    if source in {"His games", "Age and coach", "No games yet"}:
        return source
    if source == "Last 12 games":
        games = player.minutes_sample or 12
        return f"Last {games} games"
    if source == "Season average":
        return "Season average"
    if source == "Depth chart":
        if player.minutes_slot:
            return f"Depth chart, slot {player.minutes_slot}"
        return "Depth chart"
    if getattr(player, "court", None) is not None and source:
        return source
    return "Assigned"


def _five_text(player, team) -> str:
    """Offensive and defensive rating of the fives he has actually played in."""
    squads = getattr(team, "squads", None) or {}
    off_num = def_num = weight = 0.0
    pid = int(player.player_id)
    for key, face in squads.items():
        if pid not in key:
            continue
        minutes = float(getattr(face, "minutes", 0) or 0)
        offense = float(getattr(face, "off_rating", 0) or 0)
        defense = float(getattr(face, "def_rating", 0) or 0)
        if minutes <= 0 or offense <= 70:
            continue
        off_num += offense * minutes
        def_num += defense * minutes
        weight += minutes
    if weight <= 0:
        return ""
    return f"{off_num / weight:.0f} / {def_num / weight:.0f}"


def _minutes_injury(player) -> str:
    label = player.injury_label
    if not label:
        return ""
    factor = float(player.injury_factor)
    if factor <= 0:
        return label
    if abs(factor - 1.0) < 1e-9:
        return label
    return f"{label} × {factor:g}"


def _posted_home_spread(prices: list[dict]) -> float | None:
    for price in prices:
        if price.get("market") != "spread" or price.get("side") != "home":
            continue
        line = price.get("line")
        if line is None or (isinstance(line, float) and np.isnan(line)):
            continue
        return float(line)
    return None


def _spread_text(value: float) -> str:
    if abs(value) < 0.05:
        return "PK"
    rounded = round(float(value), 1)
    if float(rounded).is_integer():
        return f"{int(rounded):+d}"
    return f"{rounded:+.1f}"


def _line_label(home: str, away: str, home_number: float) -> str:
    if abs(home_number) < 0.05:
        return "Pick"
    if home_number < 0:
        return f"{home} {_spread_text(home_number)}"
    return f"{away} {_spread_text(-home_number)}"


def _sim_margin_text(home: str, away: str, margin: float) -> str:
    if abs(margin) < 0.05:
        return "Pick"
    favorite = home if margin >= 0 else away
    return f"{favorite} by {abs(margin):.1f}"


def spread_view(summary: dict) -> dict:
    """Numbers for the game card and the spread chart.

    A posted home spread stays in betting form (favorite negative). With no
    price, the card uses the simulated home margin flipped the same way.
    ``threshold`` is that posted number on a home-margin axis.
    """
    team = summary["team"]
    home = summary["home"]
    away = summary["away"]
    posted = _posted_home_spread(summary.get("prices") or [])
    sim_mean = float(team["spread_mean"])
    home_number = -sim_mean if posted is None else posted
    return {
        "source": "sim" if posted is None else "line",
        "home_text": _spread_text(home_number),
        "away_text": _spread_text(-home_number),
        "sim_text": _sim_margin_text(home, away, sim_mean),
        "line_text": "" if posted is None else _line_label(home, away, posted),
        "threshold": None if posted is None else -posted,
    }


def format_summary(summary: dict) -> str:
    team = summary["team"]
    lines = [
        f"{summary['away']} @ {summary['home']}  ({summary['date']}, {summary['source']})",
        "Flags: " + (", ".join(summary["flags"]) or "none"),
        (
            f"Score {summary['away']} {team['away_mean']:.1f}  {summary['home']} {team['home_mean']:.1f}"
            f"   total {team['total_mean']:.1f} ({team['total_p10']:.0f}-{team['total_p90']:.0f})"
            f"   {summary['home']} win {team['home_win_prob']:.1%}"
        ),
        "",
        f"{'Player':<22} {'Min':>6} {'PTS':>8} {'REB':>6} {'AST':>6} {'3PM':>6} {'PRA':>8}  Plays",
    ]
    for row in sorted(summary["players"], key=lambda item: item["minutes"], reverse=True):
        if row["minutes"] < 4:
            continue
        lines.append(
            f"{row['player'][:22]:<22} {row['minutes']:6.1f} "
            f"{row['pts']:5.1f} ({row['pts_p10']:.0f}-{row['pts_p90']:.0f}) "
            f"{row['reb']:6.1f} {row['ast']:6.1f} {row['tpm']:6.1f} {row['pra']:8.1f}  {row['play_types']}"
        )
    if summary["prices"]:
        lines.append("")
        lines.append(f"{'Market':<16} {'Side':<8} {'Model':>7} {'Price':>7} {'Edge':>7} {'Stake':>7} Source")
        for price in summary["prices"]:
            lines.append(
                f"{price['market']:<16} {price['side']:<8} {price['model_prob']:7.1%} "
                f"{price['market_prob']:7.1%} {price['edge']:+7.1%} {price['stake']:7.1%} {price['source']}"
            )
    return "\n".join(lines)
