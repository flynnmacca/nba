"""Who is on the court, from each player's own games.

Every player id carries the same profile. Minutes, the clock he sits, who
leaves with him, and who comes in for him are estimated from his games before
the slate date. A trip is then priced from the ten players standing there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd

from nba_sim.constants import PLAY_TYPES
from nba_sim.features.shrink import FILE_PRIOR


_ACTIONS = {
    "transition": "Transition",
    "pick": "PRBallHandler",
    "post": "Postup",
    "spot": "Spotup",
}


@dataclass
class PlayerCourt:
    """One player's court profile under one coach. The same fields for every id."""

    player_id: int
    minutes: float = 0.0
    minutes_std: float = 4.0
    minutes_close: float = 0.0
    minutes_blowout: float = 0.0
    starts: bool = False
    # Seconds remaining in the quarter when his first stint ends. 0 means he stays.
    sit_remaining: float = 0.0
    # Seconds remaining when he checks back in. He is off between the two.
    return_remaining: float = 0.0
    blowout_remaining: float = 720.0
    leaves_with: tuple[int, ...] = ()
    enters_with: tuple[int, ...] = ()
    comes_in_for: tuple[int, ...] = ()
    # Share of his check-ins that replaced each teammate. A guide, not a door.
    enter_share: dict[int, float] = field(default_factory=dict)
    replaced_by: tuple[int, ...] = ()
    rest_b2b: float = 0.0
    rest_3in4: float = 0.0
    rest_heavy: float = 0.0
    # Share of this team's games he has actually played. 1 until he has a game.
    play_rate: float = 1.0
    margin_slope: float = 0.0
    usage: float = 0.2
    age: float = 0.0
    sample: int = 0
    position: str = "W"
    source: str = "His games"
    clock_from_subs: bool = False


@dataclass
class CoachWhistle:
    """How this coach changes the court. The same fields for every coach."""

    coach: str
    size_share: dict[int, float] = field(default_factory=dict)
    # (outgoing ids, incoming ids, weight, decided game)
    bundles: list[tuple[tuple[int, ...], tuple[int, ...], float, bool]] = field(default_factory=list)
    # Offensive-rating change when this position is absent. Fit on the same games as the minute matrix.
    absence_off: dict[str, float] = field(default_factory=dict)


@dataclass
class SquadFace:
    """What this five, or the nearest five, actually did."""

    minutes: float = 0.0
    off_rating: float = 0.0
    def_rating: float = 0.0
    pace: float = 0.0
    oreb: float = 0.0
    usage: dict[int, float] = field(default_factory=dict)
    mix: dict[str, float] = field(default_factory=dict)
    possessions: float = 0.0


def position_group(value: object) -> str:
    text = str(value or "").upper().replace(" ", "")
    if text in {"C", "CENTER"} or text.startswith("C"):
        return "C"
    if text in {"PF", "F", "F-C", "C-F", "FC"}:
        return "F"
    if text in {"SF", "W"}:
        return "W"
    if "G" in text or text in {"PG", "SG"}:
        return "G"
    return "W"


def _trust(groups: list[np.ndarray]) -> float:
    """Games of prior per game of evidence, from the panel's own variance."""
    means = []
    within = []
    for values in groups:
        if len(values) < 2:
            continue
        means.append(float(np.mean(values)))
        within.append(float(np.var(values)))
    if len(means) < 2 or not within:
        return 8.0
    between = float(np.var(means))
    if between <= 1e-6:
        return 8.0
    return float(np.clip(float(np.mean(within)) / between, 1.0, 40.0))


def _pool(count: float, mean: float, prior: float, trust: float) -> float:
    if count <= 0:
        return float(prior)
    return float((count * mean + trust * prior) / (count + trust))


def _game_token(value: object) -> str:
    return str(value or "").split(".")[0].zfill(10)


_EVENTS_BEFORE: dict[tuple[int, int], pd.DataFrame] = {}


def _events_before(frame: pd.DataFrame | None, logs: pd.DataFrame | None) -> pd.DataFrame:
    """Event rows whose game is already in the pre-slate logs. A later game stays out."""
    if frame is None or getattr(frame, "empty", True) or "game_id" not in frame.columns:
        return pd.DataFrame()
    if logs is None or getattr(logs, "empty", True) or "game_id" not in logs.columns:
        return pd.DataFrame()
    key = (id(frame), id(logs))
    cached = _EVENTS_BEFORE.get(key)
    if cached is not None:
        return cached
    allowed = {_game_token(value) for value in logs["game_id"].dropna().unique() if str(value).strip()}
    if frame.attrs.get("game_tokens"):
        work = frame
        tokens = frame["game_id"]
    else:
        work = frame
        tokens = work["game_id"].map(_game_token)
    kept = work.loc[tokens.isin(allowed)].copy()
    if not frame.attrs.get("game_tokens"):
        kept["game_id"] = tokens.loc[kept.index]
    if kept.empty or "date" in kept.columns or "date" not in logs.columns:
        _EVENTS_BEFORE[key] = kept
        return kept
    dates = logs.loc[:, ["game_id", "date"]].copy()
    dates["game_id"] = dates["game_id"].map(_game_token)
    dates = dates.drop_duplicates("game_id")
    out = kept.merge(dates, on="game_id", how="left")
    if len(_EVENTS_BEFORE) > 24:
        _EVENTS_BEFORE.clear()
    _EVENTS_BEFORE[key] = out
    return out


def _before(frame: pd.DataFrame | None, on) -> pd.DataFrame:
    if frame is None or getattr(frame, "empty", True):
        return pd.DataFrame()
    out = frame.copy()
    if on is None or "date" not in out.columns:
        return out
    dates = pd.to_datetime(out["date"], errors="coerce")
    return out.loc[dates.notna() & (dates < pd.Timestamp(on))].copy()


def _split_team(rows: pd.DataFrame, team_id: int | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Games with this coach, then the player's earlier teams."""
    if rows is None or rows.empty or team_id is None or "team_id" not in rows.columns:
        empty = rows.iloc[0:0] if rows is not None else pd.DataFrame()
        return rows, empty
    ids = pd.to_numeric(rows["team_id"], errors="coerce")
    here = rows.loc[ids == int(team_id)]
    earlier = rows.loc[ids != int(team_id)]
    return here, earlier


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


_CAREER_MINUTES: dict[int, dict[int, float]] = {}


def _career_minutes(logs: pd.DataFrame | None) -> dict[int, float]:
    """League-wide played-minute means. Age peers need more than tonight's roster."""
    if logs is None or getattr(logs, "empty", True) or "player_id" not in logs.columns:
        return {}
    key = id(logs)
    cached = _CAREER_MINUTES.get(key)
    if cached is not None:
        return cached
    frame = logs
    if "entity_type" in frame.columns:
        frame = frame.loc[frame["entity_type"].astype(str).eq("P")]
    if frame.empty:
        _CAREER_MINUTES[key] = {}
        return {}
    pid = pd.to_numeric(frame["player_id"], errors="coerce")
    minutes = _num(frame, "min")
    work = pd.DataFrame({"player_id": pid, "min": minutes}).dropna(subset=["player_id"])
    work = work.loc[work["min"] > 0]
    career = {int(player): float(values.mean()) for player, values in work.groupby("player_id", sort=False)["min"]}
    if len(_CAREER_MINUTES) > 4:
        _CAREER_MINUTES.clear()
    _CAREER_MINUTES[key] = career
    return career


def _court_rows(logs: pd.DataFrame | None, ids: list[int], team_id: int | None) -> pd.DataFrame:
    """Tonight's roster history plus this team's boxes. Drop the rest of the league."""
    if logs is None or getattr(logs, "empty", True):
        return pd.DataFrame()
    frame = logs
    pid = pd.to_numeric(frame["player_id"], errors="coerce")
    wanted = set(int(value) for value in ids if value)
    mask = pid.isin(wanted) if wanted else pd.Series(False, index=frame.index)
    if team_id is not None and "team_id" in frame.columns:
        tid = pd.to_numeric(frame["team_id"], errors="coerce")
        mask = mask | (tid == int(team_id))
    if "entity_type" in frame.columns:
        mask = mask & frame["entity_type"].astype(str).eq("P")
    return frame.loc[mask]


def fit_court(
    logs: pd.DataFrame | None,
    roster: list,
    on=None,
    participation: pd.DataFrame | None = None,
    subs: pd.DataFrame | None = None,
    possessions: pd.DataFrame | None = None,
    lineups: pd.DataFrame | None = None,
    ages: pd.DataFrame | None = None,
    coach: str = "",
    team_id: int | None = None,
    panel_trust: float | None = None,
    current_slate: bool = False,
) -> tuple[dict[int, PlayerCourt], CoachWhistle, dict[frozenset[int], SquadFace], dict[tuple[str, str], float]]:
    """Profiles, whistles, squad faces, and the position-substitution matrix.

    ``roster`` is the players on the team tonight. Rows dated on or after ``on``
    are ignored. A later game cannot move tonight.
    """
    played = _before(logs, on)
    career = _career_minutes(played)
    ids = [int(getattr(player, "player_id", player)) for player in roster]
    played = _court_rows(played, ids, team_id)
    parts = _before(participation, on)
    whistles = _before(subs, on)
    trips = _before(possessions, on)
    ages = ages if ages is not None else pd.DataFrame()
    positions = {}
    for player in roster:
        pid = int(getattr(player, "player_id", 0) or 0)
        if pid:
            positions[pid] = position_group(getattr(player, "position", "W"))
    profiles = _player_profiles(
        played,
        parts,
        whistles,
        ids,
        positions,
        ages,
        coach,
        team_id,
        panel_trust,
        career=career,
        current_slate=current_slate,
    )
    whistle = _coach_whistle(whistles, coach)
    squads = _squads(trips, lineups, profiles)
    matrix_rows = parts if not parts.empty else played
    matrix, absence = _position_matrix(matrix_rows, profiles, positions)
    whistle.absence_off = absence
    _apply_bundles(profiles, whistle)
    return profiles, whistle, squads, matrix


def _player_profiles(
    logs,
    participation,
    subs,
    ids,
    positions,
    ages,
    coach: str,
    team_id: int | None = None,
    panel_trust: float | None = None,
    career: dict[int, float] | None = None,
    current_slate: bool = False,
) -> dict[int, PlayerCourt]:
    profiles: dict[int, PlayerCourt] = {}
    if logs is None or logs.empty or "player_id" not in logs.columns:
        for pid in ids:
            profiles[pid] = PlayerCourt(
                player_id=pid,
                position=positions.get(pid, "W"),
                source="No games yet",
            )
        return profiles
    work = pd.DataFrame(
        {
            "player_id": pd.to_numeric(logs["player_id"], errors="coerce"),
            "min": _num(logs, "min"),
            "margin": _num(logs, "margin") if "margin" in logs.columns else 0.0,
            "team_id": pd.to_numeric(logs["team_id"], errors="coerce") if "team_id" in logs.columns else np.nan,
        },
        index=logs.index,
    )
    if "date" in logs.columns:
        work["date"] = pd.to_datetime(logs["date"], errors="coerce")
    work = work.dropna(subset=["player_id"])
    work["player_id"] = work["player_id"].astype(int)
    groups = []
    by_player = {int(pid): group for pid, group in work.groupby("player_id", sort=False)}
    for rows in by_player.values():
        played = rows.loc[rows["min"] > 0, "min"].to_numpy(dtype=float)
        if len(played):
            groups.append(played)
    trust = float(panel_trust) if panel_trust is not None else _trust(groups)
    age_map = _ages(ages)
    if career is None:
        career = {
            pid: float(rows.loc[rows["min"] > 0, "min"].mean()) if (rows["min"] > 0).any() else 0.0
            for pid, rows in by_player.items()
        }
    age_prior = _age_curve(career, age_map)
    for pid in ids:
        rows = by_player.get(pid, work.iloc[0:0])
        here, earlier = _split_team(rows, team_id)
        played_rows = here[here["min"] > 0] if not here.empty else here
        played = played_rows["min"].to_numpy(dtype=float) if not played_rows.empty else np.array([])
        count = int(len(played))
        own = float(played.mean()) if count else 0.0
        earlier_played = earlier[earlier["min"] > 0]["min"] if not earlier.empty else pd.Series(dtype=float)
        earlier_mean = float(earlier_played.mean()) if len(earlier_played) else None
        curve = age_prior.get(pid)
        if earlier_mean is not None:
            prior = _pool(len(earlier_played), earlier_mean, curve if curve else earlier_mean, trust)
        elif curve:
            prior = curve
        else:
            prior = own
        # A heavy minute role unlocks his own sample sooner. Pooling a 36-minute
        # star toward a peer mean for 40 games is what pulled Jokic-scale priors
        # into the high twenties. On the live slate, 32+ stars unlock after two
        # same-team games so STOD minutes win immediately.
        role_trust = float(trust)
        if own >= 32.0 or (prior is not None and float(prior or 0.0) >= 32.0):
            role_trust = min(role_trust, 8.0)
        live_star = bool(current_slate) and own >= 32.0
        unlock = max(role_trust, 2.0) if live_star else (max(role_trust, 4.0) if own >= 32.0 else max(trust, 8.0))
        # A five-minute gap vs another franchise is a new role at any band.
        role_shift = earlier_mean is not None and abs(own - float(earlier_mean)) >= 5.0
        tiny_role = own < 6.0
        if count == 0:
            # No games here yet. Another team or his age curve is all we have.
            minutes = float(prior or 0.0)
        elif live_star and count >= 1:
            # This season's star minutes win immediately on the live slate —
            # even a single same-team appearance beats the age-peer pool.
            minutes = own
        elif count >= unlock or (curve is None and earlier_mean is None) or role_shift or tiny_role:
            # This team's minutes win on a clear cross-team role change, a spot
            # role that would otherwise shrink toward the age curve, or a full sample.
            minutes = own
        else:
            # Games on this team set the role. Age can shrink a wild number;
            # another franchise's minutes do not.
            peer = float(curve) if curve else own
            minutes = _pool(count, own, peer, role_trust)
        close = played_rows[played_rows["margin"].abs() <= 8]["min"] if count else pd.Series(dtype=float)
        blow = played_rows[played_rows["margin"].abs() >= 15]["min"] if count else pd.Series(dtype=float)
        # Close and blowout clocks shrink toward the played mean, not a second
        # peer prior that can replace a star's owe on a thin sample.
        minutes_close = _pool(len(close), float(close.mean()) if len(close) else minutes, minutes, min(role_trust, 8.0))
        minutes_blow = _pool(len(blow), float(blow.mean()) if len(blow) else minutes, minutes, min(role_trust, 8.0))
        slope = 0.0
        if count >= 6 and played_rows["margin"].nunique() > 2:
            x = played_rows["margin"].to_numpy(dtype=float)
            y = played_rows["min"].to_numpy(dtype=float)
            x = x - x.mean()
            denom = float(np.dot(x, x))
            if denom > 1:
                slope = float(np.dot(x, y - y.mean()) / denom)
        starts = False
        sit, back, blow_at = _clocks_from_minutes(minutes, minutes_blow)
        if count <= 0 and earlier_mean is None and not curve:
            origin = "No games yet"
        elif count > 0 and (role_shift or tiny_role):
            origin = "His games"
        elif (earlier_mean is not None or curve) and count < max(trust, 8):
            origin = "Age and coach"
        elif count:
            origin = "His games"
        else:
            origin = "Age and coach"
        profile = PlayerCourt(
            player_id=pid,
            minutes=minutes,
            minutes_std=float(played.std()) if count > 1 else 4.0,
            minutes_close=minutes_close,
            minutes_blowout=minutes_blow,
            starts=starts,
            sit_remaining=sit,
            return_remaining=back,
            blowout_remaining=blow_at,
            margin_slope=slope,
            usage=0.2,
            age=float(age_map.get(pid, 0.0) or 0.0),
            sample=count,
            position=positions.get(pid, "W"),
            source=origin,
        )
        profiles[pid] = profile
    _rest_rates(profiles, participation if participation is not None and not participation.empty else work, trust)
    cut = _starter_cut([profile.minutes for profile in profiles.values() if profile.sample])
    for profile in profiles.values():
        if profile.sample and profile.minutes >= cut:
            profile.starts = True
    _clocks_from_subs(profiles, subs)
    return profiles


def _starter_cut(means: list[float]) -> float:
    ordered = sorted((value for value in means if value > 0), reverse=True)
    if len(ordered) >= 5:
        return float(ordered[4]) - 0.1
    return 20.0


def _clocks_from_minutes(minutes: float, blowout: float) -> tuple[float, float, float]:
    """A clock that yields his own minutes when the whistles have not been read.

    He starts each quarter and sits once the quarter has given him his share.
    A shorter decided-game history sits him earlier.
    """
    seconds_on = float(np.clip(minutes / 4.0 * 60.0, 0.0, 720.0))
    if seconds_on >= 700.0:
        sit, back = 0.0, 0.0
    else:
        sit = 720.0 - seconds_on
        back = 0.0
    if blowout + 1.0 < minutes:
        blow_on = float(np.clip(blowout / 4.0 * 60.0, 0.0, 720.0))
        blow_at = 720.0 - blow_on
    else:
        blow_at = 720.0
    return sit, back, blow_at


def _ages(frame: pd.DataFrame) -> dict[int, float]:
    if frame is None or frame.empty or "player_id" not in frame.columns:
        return {}
    out = {}
    column = "age" if "age" in frame.columns else None
    if column is None:
        return {}
    for row in frame.to_dict("records"):
        pid = row.get("player_id")
        age = row.get(column)
        if pid is None or age is None or not np.isfinite(float(age)):
            continue
        out[int(pid)] = float(age)
    return out


def _age_curve(career: dict[int, float], ages: dict[int, float]) -> dict[int, float]:
    """Minutes of players the same age whose career load is close to his."""
    if not ages:
        return {}
    rows = [(pid, ages[pid], career.get(pid, 0.0)) for pid in ages if career.get(pid, 0.0) > 0]
    prior = {}
    for pid, age, load in rows:
        mates = [other for other, oage, oload in rows if other != pid and abs(oage - age) <= 2 and abs(oload - load) <= 6]
        if mates:
            prior[pid] = float(np.mean([career[other] for other in mates]))
    return prior


def _absence_runs(minutes: list[float], reasons: list[str]) -> list[bool]:
    """An unlabeled zero touching an out is that same absence.

    The label spreads through a run of blank zeros and stops at a game he played.
    """
    from nba_sim.features.availability import known_out_reason

    kinds: list[str] = []
    for minute, reason in zip(minutes, reasons):
        if float(minute) > 0:
            kinds.append("play")
        elif known_out_reason(reason):
            kinds.append("out")
        else:
            kinds.append("open")
    changed = True
    while changed:
        changed = False
        for index, kind in enumerate(kinds):
            if kind != "open":
                continue
            left = kinds[index - 1] if index else ""
            right = kinds[index + 1] if index + 1 < len(kinds) else ""
            if left == "out" or right == "out":
                kinds[index] = "out"
                changed = True
    return [kind == "out" for kind in kinds]


def _rest_rates(profiles: dict[int, PlayerCourt], frame: pd.DataFrame, trust: float) -> None:
    if frame is None or frame.empty or "date" not in frame.columns:
        return
    work = frame.copy()
    work["player_id"] = pd.to_numeric(work["player_id"], errors="coerce")
    work = work.dropna(subset=["player_id"])
    work["player_id"] = work["player_id"].astype(int)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["min"] = _num(work, "min")
    if "reason" not in work.columns:
        work["reason"] = ""
    work["reason"] = work["reason"].fillna("").astype(str).replace({"nan": "", "None": ""})
    for pid, profile in profiles.items():
        rows = work[work["player_id"] == pid].sort_values("date")
        if rows.empty:
            continue
        dates = list(rows["date"])
        minutes = list(rows["min"])
        reasons = list(rows["reason"])
        carried = _absence_runs(minutes, reasons)
        b2b_sit = b2b_n = heavy_sit = heavy_n = three_sit = three_n = 0
        trials = sits = 0
        for index, day in enumerate(dates):
            if pd.isna(day):
                continue
            gap = (day - dates[index - 1]).days if index else 9
            if carried[index]:
                sat = None
            else:
                sat = float(minutes[index]) <= 0
            if sat is None:
                continue
            trials += 1
            sits += int(sat)
            if gap == 1:
                b2b_n += 1
                b2b_sit += int(sat)
            if index and minutes[index - 1] >= 36 and not carried[index - 1] and minutes[index - 1] > 0:
                heavy_n += 1
                heavy_sit += int(sat)
            if index >= 2 and (day - dates[index - 2]).days <= 4:
                earlier = None if carried[index - 2] else float(minutes[index - 2]) <= 0
                if earlier is not None:
                    three_n += 1
                    three_sit += int(sat)
        if trials:
            profile.play_rate = 1.0 if sits == 0 else 1.0 - sits / trials
        profile.rest_b2b = _pool(b2b_n, b2b_sit / b2b_n if b2b_n else 0.0, 0.08, trust)
        profile.rest_heavy = _pool(heavy_n, heavy_sit / heavy_n if heavy_n else 0.0, 0.05, trust)
        profile.rest_3in4 = _pool(three_n, three_sit / three_n if three_n else 0.0, 0.06, trust)


def _clocks_from_subs(profiles: dict[int, PlayerCourt], subs: pd.DataFrame | None) -> None:
    if subs is None or subs.empty:
        return
    frame = subs.copy()
    for column in ("player_out", "player_in", "period"):
        if column not in frame.columns:
            return
    frame["player_out"] = pd.to_numeric(frame["player_out"], errors="coerce")
    frame["player_in"] = pd.to_numeric(frame["player_in"], errors="coerce")
    frame["period"] = pd.to_numeric(frame["period"], errors="coerce")
    frame["remaining"] = frame["clock"].map(_clock_remaining) if "clock" in frame.columns else 720.0
    frame["decided"] = _num(frame, "margin").abs() >= 15 if "margin" in frame.columns else False
    if "game_id" not in frame.columns:
        frame["game_id"] = 0
    sits: dict[int, list[float]] = {}
    returns: dict[int, list[float]] = {}
    blowouts: dict[int, list[float]] = {}
    enter_counts: dict[int, dict[int, int]] = {}
    out_values = frame["player_out"].to_numpy(dtype=float)
    in_values = frame["player_in"].to_numpy(dtype=float)
    remaining = frame["remaining"].to_numpy(dtype=float)
    decided_row = frame["decided"].fillna(False).to_numpy(dtype=bool)
    for _key, index in frame.groupby(["game_id", "period", "remaining"], dropna=False, sort=False).indices.items():
        outs = _finite_ids(out_values[index])
        ins = _finite_ids(in_values[index])
        if not outs:
            continue
        out_set = set(outs)
        in_set = set(ins)
        for pid in outs:
            profile = profiles.get(pid)
            if profile is None:
                continue
            profile.leaves_with = tuple(sorted(set(profile.leaves_with) | (out_set - {pid})))
            profile.replaced_by = tuple(sorted(set(profile.replaced_by) | in_set))
        for pid in ins:
            profile = profiles.get(pid)
            if profile is None:
                continue
            profile.comes_in_for = tuple(sorted(set(profile.comes_in_for) | out_set))
            profile.enters_with = tuple(sorted(set(profile.enters_with) | (in_set - {pid})))
            counts = enter_counts.setdefault(pid, {})
            for other in outs:
                counts[other] = counts.get(other, 0) + 1
    for key, index in frame.groupby(["game_id", "period"], dropna=False, sort=False).indices.items():
        period = key[1]
        period_no = int(period) if pd.notna(period) else 1
        any_decided = bool(decided_row[index].any())
        decided = any_decided and period_no >= 4
        close = period_no in (1, 3) and not any_decided
        if not decided and not close:
            continue
        sub_out = out_values[index]
        sub_in = in_values[index]
        sub_remaining = remaining[index]
        seen: set[int] = set()
        for value in sub_out:
            if value != value:
                continue
            pid = int(value)
            if pid in seen or pid not in profiles:
                continue
            seen.add(pid)
            first = float(sub_remaining[sub_out == pid].max())
            if decided:
                blowouts.setdefault(pid, []).append(first)
            if not close:
                continue
            sits.setdefault(pid, []).append(first)
            came = sub_remaining[sub_in == pid]
            back = came[came < first - 1]
            if len(back):
                returns.setdefault(pid, []).append(float(back.max()))
    for pid, samples in sits.items():
        profile = profiles[pid]
        profile.sit_remaining = float(np.median(samples))
        profile.clock_from_subs = True
        back = returns.get(pid) or []
        if back:
            returned = float(np.median(back))
            if returned < profile.sit_remaining:
                profile.return_remaining = returned
    for pid, samples in blowouts.items():
        profiles[pid].blowout_remaining = float(np.median(samples))
    for pid, counts in enter_counts.items():
        profile = profiles.get(pid)
        total = sum(counts.values())
        if profile is None or total <= 0:
            continue
        profile.enter_share = {int(other): count / total for other, count in counts.items()}
    for profile in profiles.values():
        if profile.sample and not profile.starts and profile.comes_in_for:
            continue
        if profile.minutes >= _starter_cut([item.minutes for item in profiles.values() if item.sample]):
            profile.starts = True


def _finite_ids(values: np.ndarray) -> tuple[int, ...]:
    found: set[int] = set()
    for value in values:
        if value != value:
            continue
        found.add(int(value))
    return tuple(sorted(found))


def _clock_remaining(value: object) -> float:
    text = str(value or "")
    if text.startswith("PT"):
        minutes = 0.0
        seconds = 0.0
        body = text[2:]
        if "M" in body:
            mins, body = body.split("M", 1)
            minutes = float(mins or 0)
        if body.endswith("S"):
            seconds = float(body[:-1] or 0)
        return minutes * 60.0 + seconds
    if ":" in text:
        left, right = text.split(":", 1)
        return float(left) * 60.0 + float(right)
    try:
        return float(text)
    except ValueError:
        return 720.0


def _coach_whistle(subs: pd.DataFrame | None, coach: str) -> CoachWhistle:
    whistle = CoachWhistle(coach=coach or "")
    if subs is None or subs.empty or "player_out" not in subs.columns:
        whistle.size_share = {1: 1.0}
        return whistle
    frame = subs.copy()
    frame["remaining"] = frame["clock"].map(_clock_remaining) if "clock" in frame.columns else 0.0
    frame["player_out"] = pd.to_numeric(frame["player_out"], errors="coerce")
    frame["player_in"] = pd.to_numeric(frame["player_in"], errors="coerce")
    keys = [column for column in ("game_id", "period", "remaining") if column in frame.columns or column == "remaining"]
    sizes: dict[int, int] = {}
    bundles: dict[tuple, list] = {}
    out_values = frame["player_out"].to_numpy(dtype=float)
    in_values = frame["player_in"].to_numpy(dtype=float)
    margins = _num(frame, "margin").to_numpy(dtype=float) if "margin" in frame.columns else None
    for _key, index in frame.groupby(keys, dropna=False, sort=False).indices.items():
        outs = _finite_ids(out_values[index])
        ins = _finite_ids(in_values[index])
        if not outs:
            continue
        size = min(5, len(outs))
        sizes[size] = sizes.get(size, 0) + 1
        if margins is None:
            decided = False
        else:
            finite = margins[index]
            finite = finite[np.isfinite(finite)]
            decided = bool(len(finite) and float(np.max(np.abs(finite))) >= 15)
        slot = bundles.setdefault((outs, ins, decided), [0])
        slot[0] += 1
    total = sum(sizes.values()) or 1
    whistle.size_share = {size: count / total for size, count in sizes.items()}
    whistle.bundles = [
        (outs, ins, count[0] / total, decided) for (outs, ins, decided), count in bundles.items()
    ]
    whistle.bundles.sort(key=lambda item: -item[2])
    return whistle


def _apply_bundles(profiles: dict[int, PlayerCourt], whistle: CoachWhistle) -> None:
    """A player who always leaves inside a size-3 whistle keeps those partners."""
    for outs, ins, _weight, _decided in whistle.bundles[:12]:
        for pid in outs:
            profile = profiles.get(pid)
            if profile is None:
                continue
            profile.leaves_with = tuple(sorted(set(profile.leaves_with) | {other for other in outs if other != pid}))
            profile.replaced_by = tuple(sorted(set(profile.replaced_by) | set(ins)))
        for pid in ins:
            profile = profiles.get(pid)
            if profile is None:
                continue
            profile.comes_in_for = tuple(sorted(set(profile.comes_in_for) | set(outs)))


_LINEUP_KEYS: dict[str, frozenset[int] | None] = {}


def _squads(trips: pd.DataFrame | None, lineups: pd.DataFrame | None, profiles: dict[int, PlayerCourt]) -> dict[frozenset[int], SquadFace]:
    found: dict[frozenset[int], SquadFace] = {}
    if lineups is not None and not getattr(lineups, "empty", True):
        for row in lineups.to_dict("records"):
            key = _lineup_key(row.get("group_id"))
            if key is None:
                continue
            face = found.setdefault(key, SquadFace())
            face.minutes = float(row.get("min") or face.minutes or 0)
            face.off_rating = float(row.get("off_rating") or 0)
            face.def_rating = float(row.get("def_rating") or 0)
            face.pace = float(row.get("pace") or 0)
            face.oreb = float(row.get("oreb_pct") or row.get("oreb") or 0)
    if trips is None or trips.empty or "player_id" not in trips.columns or "lineup" not in trips.columns:
        for profile in profiles.values():
            profile.usage = profile.usage or 0.2
        return found
    player_ids = pd.to_numeric(trips["player_id"], errors="coerce").to_numpy()
    raw_lineups = trips["lineup"].tolist()
    raw_actions = trips["action"].astype(str).str.lower().tolist() if "action" in trips.columns else None
    kept_keys: list[frozenset[int]] = []
    kept_players: list[int] = []
    kept_actions: list[str] | None = [] if raw_actions is not None else None
    for index, pid in enumerate(player_ids):
        if not np.isfinite(pid):
            continue
        key = _lineup_key(raw_lineups[index])
        if key is None:
            continue
        kept_keys.append(key)
        kept_players.append(int(pid))
        if kept_actions is not None:
            kept_actions.append(raw_actions[index])
    if not kept_keys:
        return found
    by_key: dict[frozenset[int], list[int]] = {}
    for offset, key in enumerate(kept_keys):
        by_key.setdefault(key, []).append(offset)
    roster = set(profiles)
    for key, offsets in by_key.items():
        face = found.setdefault(key, SquadFace())
        face.possessions = float(len(offsets))
        if face.minutes <= 0:
            face.minutes = face.possessions
        counts: dict[int, int] = {}
        for offset in offsets:
            pid = kept_players[offset]
            counts[pid] = counts.get(pid, 0) + 1
        total = float(sum(counts.values())) or 1.0
        face.usage = {pid: count / total for pid, count in counts.items()}
        if kept_actions is not None:
            mix_counts: dict[str, int] = {}
            for offset in offsets:
                name = kept_actions[offset]
                mix_counts[name] = mix_counts.get(name, 0) + 1
            mix_total = float(sum(mix_counts.values())) or 1.0
            face.mix = {name: count / mix_total for name, count in mix_counts.items()}
        if face.possessions >= 20:
            for pid, share in face.usage.items():
                if pid in roster:
                    profiles[pid].usage = float(share)
    return found


def _lineup_key(value: object) -> frozenset[int] | None:
    if isinstance(value, frozenset):
        return value
    text = str(value or "")
    if text in _LINEUP_KEYS:
        return _LINEUP_KEYS[text]
    parts = text.replace(",", "-").split("-")
    try:
        ids = [int(float(part)) for part in parts if part]
    except ValueError:
        _LINEUP_KEYS[text] = None
        return None
    key = frozenset(ids) if len(ids) == 5 else None
    if len(_LINEUP_KEYS) > 200_000:
        _LINEUP_KEYS.clear()
    _LINEUP_KEYS[text] = key
    return key


def _position_matrix(logs: pd.DataFrame, profiles: dict[int, PlayerCourt], positions: dict[int, str]) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    """Extra minutes at each position when a position is absent. Columns sum to 1.

    The offensive-rating shift of that absence is estimated on the same games.
    """
    groups = ("G", "W", "F", "C")
    raw = {pair: 0.0 for left in groups for pair in ((left, right) for right in groups)}
    if logs is None or logs.empty:
        for left in groups:
            for right in groups:
                raw[(left, right)] = 1.0 if left == right else 0.15
        return _normalize_columns(raw), {}
    frame = logs.copy()
    frame["player_id"] = pd.to_numeric(frame["player_id"], errors="coerce")
    frame["min"] = _num(frame, "min")
    if "game_id" not in frame.columns:
        return _normalize_columns(raw), {}
    usual = {}
    for pid, rows in frame.groupby("player_id"):
        if not np.isfinite(pid):
            continue
        played = rows.loc[rows["min"] > 0, "min"]
        if len(played):
            usual[int(pid)] = float(played.mean())
    off_when = {pos: [] for pos in groups}
    off_without = {pos: [] for pos in groups}
    for _game, group in frame.groupby("game_id"):
        absent = []
        present = []
        seen = set()
        for row in group.to_dict("records"):
            pid = int(row["player_id"]) if np.isfinite(row["player_id"]) else 0
            if not pid:
                continue
            pos = positions.get(pid) or (profiles.get(pid).position if pid in profiles else "W")
            seen.add(pos)
            if float(row["min"] or 0) <= 0:
                absent.append(pos)
            else:
                present.append((pos, float(row["min"]) - usual.get(pid, 0.0)))
        missing = [pos for pos in absent if pos not in seen or absent.count(pos) >= 1]
        # A position is absent when every player at that position has zero minutes.
        by_pos: dict[str, list[float]] = {}
        for row in group.to_dict("records"):
            pid = int(row["player_id"]) if np.isfinite(row["player_id"]) else 0
            if not pid:
                continue
            pos = positions.get(pid) or "W"
            by_pos.setdefault(pos, []).append(float(row["min"] or 0))
        gone = [pos for pos, mins in by_pos.items() if mins and max(mins) <= 0]
        for missing_pos in gone:
            for pos, delta in present:
                if delta > 0:
                    raw[(pos, missing_pos)] += delta
        off = group["off_rating"].map(lambda value: float(value)) if "off_rating" in group.columns else None
        if off is not None and off.notna().any():
            level = float(off.dropna().iloc[0])
            for pos in groups:
                if pos in gone:
                    off_without[pos].append(level)
                elif pos in by_pos and max(by_pos[pos]) > 0:
                    off_when[pos].append(level)
    if sum(raw.values()) <= 0:
        for left in groups:
            for right in groups:
                raw[(left, right)] = 1.0 if left == right else 0.15
    absence = {}
    for pos in groups:
        if off_when[pos] and off_without[pos]:
            absence[pos] = float(np.mean(off_without[pos]) - np.mean(off_when[pos]))
    return _normalize_columns(raw), absence


def _normalize_columns(raw: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    groups = ("G", "W", "F", "C")
    out = {}
    for missing in groups:
        total = sum(max(0.0, raw[(pos, missing)]) for pos in groups)
        for pos in groups:
            out[(pos, missing)] = (max(0.0, raw[(pos, missing)]) / total) if total > 0 else (1.0 if pos == missing else 0.0)
    return out


def resting(profile: PlayerCourt, period: int, seconds_left: float, lead: float) -> bool:
    """True when his own clock says he is off the floor."""
    decided = lead >= 15.0 or lead <= -18.0
    if decided and profile.minutes >= 22.0 and seconds_left < profile.blowout_remaining < 700:
        return True
    if profile.sit_remaining <= 0:
        return False
    # A learned stint has a return, so the same gap repeats each quarter.
    # A sit with no return is the first-half pattern and stays in the odd quarters.
    repeats = profile.clock_from_subs and 0 < profile.return_remaining < profile.sit_remaining
    if profile.clock_from_subs and not repeats and period not in (1, 3):
        return False
    if period >= 5:
        return False
    if seconds_left > profile.sit_remaining:
        return False
    if profile.return_remaining > 0 and seconds_left <= profile.return_remaining:
        return False
    return True


def _night_target(player, profile: PlayerCourt | None, lead: float) -> float:
    """Minutes this game owes him. A decided lead uses his blowout nights."""
    decided = lead >= 15.0 or lead <= -18.0
    if profile is None:
        return float(getattr(player, "minutes_mean", 0) or 0.0)
    if decided and profile.minutes_blowout > 0:
        return float(profile.minutes_blowout)
    if not decided and profile.minutes_close > 0:
        return float(profile.minutes_close)
    return float(profile.minutes or getattr(player, "minutes_mean", 0) or 0.0)


def _lineup_guide(profile: PlayerCourt | None, owed: float, owed_by_id: dict[int, float], preferred_off: set[int]) -> float:
    """At most a couple of minutes. The clock and the partner never open the door alone."""
    if profile is None:
        return 1.0
    guide = 0.0 if profile.player_id in preferred_off else 1.0
    share = 0.0
    for other, weight in profile.enter_share.items():
        if int(other) in preferred_off:
            share += float(weight)
    guide += min(1.0, max(0.0, share))
    partners = [owed_by_id[int(other)] for other in profile.leaves_with if int(other) in owed_by_id]
    if partners and max(abs(owed - other) for other in partners) <= 2.0:
        guide += 0.5
    return min(2.0, guide)


def select_lineup(
    players: list,
    lead: float,
    period: int,
    seconds_left: float,
    active_ids: set[int] | None = None,
    played_seconds: dict[int, float] | None = None,
) -> list[int]:
    """The five still shortest of the minutes this game owes them.

    The usual exit, the return, and who he has replaced add at most a couple of
    minutes to that rank. They reorder two men who are similarly short. They do
    not hand the stint to one historical link.
    """

    def is_active(player) -> bool:
        if active_ids is not None:
            return int(player.player_id) in active_ids
        return bool(getattr(player, "available", True))

    available = [(index, player) for index, player in enumerate(players) if is_active(player)]
    if len(available) <= 5:
        return [index for index, _player in available][:5]
    played_seconds = played_seconds or {}
    owed_by_id: dict[int, float] = {}
    preferred_off: set[int] = set()
    for _index, player in available:
        profile = getattr(player, "court", None)
        pid = int(player.player_id)
        owed_by_id[pid] = _night_target(player, profile, lead) - float(played_seconds.get(pid, 0.0)) / 60.0
        if profile is not None and resting(profile, period, seconds_left, lead):
            preferred_off.add(pid)
    ranked = []
    for index, player in available:
        profile = getattr(player, "court", None)
        pid = int(player.player_id)
        owed = owed_by_id[pid]
        guide = _lineup_guide(profile, owed, owed_by_id, preferred_off)
        mean = float(getattr(player, "minutes_mean", 0) or 0.0)
        ranked.append((owed + guide, owed, mean, -index, index))
    ranked.sort(reverse=True)
    return [item[-1] for item in ranked[:5]]


def relative_usage(player_ids: list[int], squads: dict[frozenset[int], SquadFace], priors: dict[int, float]) -> dict[int, float]:
    """Shares of this court. They sum to one. A thin five borrows the nearest four."""
    ids = [int(pid) for pid in player_ids]
    key = frozenset(ids)
    face = squads.get(key)
    usage = {}
    possessions = 0.0
    if face is not None and face.usage and face.possessions >= 20:
        usage = {pid: float(face.usage.get(pid, 0.0)) for pid in ids}
        possessions = face.possessions
    else:
        neighbor, overlap = _nearest(key, squads, need=4)
        if neighbor is not None and neighbor.possessions >= 20:
            for pid in ids:
                if pid in neighbor.usage:
                    usage[pid] = float(neighbor.usage[pid])
                else:
                    usage[pid] = float(priors.get(pid, 0.2))
            possessions = neighbor.possessions * (overlap / 5.0)
        else:
            usage = {pid: float(priors.get(pid, 0.2)) for pid in ids}
    total = sum(max(0.0, value) for value in usage.values())
    if total <= 0:
        share = 1.0 / max(len(ids), 1)
        return {pid: share for pid in ids}
    trust = possessions / (possessions + FILE_PRIOR) if possessions else 0.0
    prior = {pid: float(priors.get(pid, 0.2)) for pid in ids}
    prior_total = sum(max(0.0, value) for value in prior.values()) or 1.0
    mixed = {}
    for pid in ids:
        base = prior[pid] / prior_total
        seen = usage.get(pid, 0.0) / total
        mixed[pid] = trust * seen + (1.0 - trust) * base
    mix_total = sum(mixed.values()) or 1.0
    return {pid: value / mix_total for pid, value in mixed.items()}


def squad_mix(player_ids: list[int], squads: dict[frozenset[int], SquadFace], individual: dict[str, float]) -> dict[str, float]:
    """The actions this five runs, shrunk toward the sum of the players."""
    key = frozenset(int(pid) for pid in player_ids)
    face = squads.get(key)
    sample = 0.0
    seen: dict[str, float] = {}
    if face is not None and face.mix and face.possessions >= 15:
        seen = dict(face.mix)
        sample = face.possessions
    else:
        neighbor, _overlap = _nearest(key, squads, need=4)
        if neighbor is not None and neighbor.mix:
            seen = dict(neighbor.mix)
            sample = neighbor.possessions * 0.6
    if not seen:
        return dict(individual)
    trust = sample / (sample + FILE_PRIOR)
    names = set(individual) | set(seen)
    mixed = {name: trust * float(seen.get(name, 0.0)) + (1.0 - trust) * float(individual.get(name, 0.0)) for name in names}
    total = sum(mixed.values()) or 1.0
    return {name: value / total for name, value in mixed.items()}


def squad_level(player_ids: list[int], squads: dict[frozenset[int], SquadFace]) -> tuple[float, float, float, float]:
    """Offensive rating, defensive rating, pace, and glass for this five."""
    key = frozenset(int(pid) for pid in player_ids)
    face = squads.get(key)
    if face is None or face.minutes < 8:
        face, _overlap = _nearest(key, squads, need=4)
    if face is None:
        return 0.0, 0.0, 0.0, 0.0
    return face.off_rating, face.def_rating, face.pace, face.oreb


# id(squads) -> (fingerprint, 4-man index, answers). A trip asks for the same five
# hundreds of times, and a scan of every stored five made the board sit blank.
_NEIGHBOR_TABLES: dict[int, tuple] = {}


def _neighbor_tables(
    squads: dict[frozenset[int], SquadFace],
) -> tuple[dict[frozenset[int], list[tuple[int, frozenset[int], SquadFace]]], dict]:
    sample = next(iter(squads.values()), None)
    fingerprint = (len(squads), id(sample))
    token = id(squads)
    found = _NEIGHBOR_TABLES.get(token)
    if found is not None and found[0] == fingerprint:
        return found[1], found[2]
    index: dict[frozenset[int], list[tuple[int, frozenset[int], SquadFace]]] = {}
    for order, (other, face) in enumerate(squads.items()):
        if len(other) < 4:
            continue
        for subset in combinations(other, 4):
            index.setdefault(frozenset(subset), []).append((order, other, face))
    if len(_NEIGHBOR_TABLES) > 24:
        _NEIGHBOR_TABLES.clear()
    answers: dict[tuple[frozenset[int], int], tuple[SquadFace | None, int]] = {}
    _NEIGHBOR_TABLES[token] = (fingerprint, index, answers)
    return index, answers


def _nearest(key: frozenset[int], squads: dict[frozenset[int], SquadFace], need: int) -> tuple[SquadFace | None, int]:
    """The five with the most shared players, then the one with more minutes.

    An equal minute total keeps the five stored first. Need is four: the index is
    every group of four, which is exactly the overlap this asks for.
    """
    if not squads or need <= 0:
        return None, 0
    query = key if isinstance(key, frozenset) else frozenset(int(pid) for pid in key)
    if len(query) < need:
        return None, 0
    if need != 4:
        best = None
        overlap = 0
        best_minutes = -1.0
        for other, face in squads.items():
            shared = len(query & other)
            if shared < need:
                continue
            if shared > overlap or (shared == overlap and face.minutes > best_minutes):
                best = face
                overlap = shared
                best_minutes = face.minutes
        return best, overlap
    index, answers = _neighbor_tables(squads)
    cached = answers.get((query, need))
    if cached is not None:
        return cached
    best = None
    overlap = 0
    best_minutes = -1.0
    best_order = 1 << 30
    seen: set[int] = set()
    for subset in combinations(query, 4):
        for order, other, face in index.get(frozenset(subset), ()):
            if order in seen:
                continue
            seen.add(order)
            shared = len(query & other)
            if shared < need:
                continue
            if shared > overlap or (
                shared == overlap and (face.minutes > best_minutes or (face.minutes == best_minutes and order < best_order))
            ):
                best = face
                overlap = shared
                best_minutes = face.minutes
                best_order = order
    answers[(query, need)] = (best, overlap)
    return best, overlap


def regulation_minutes(players: list, lead: float = 2.0) -> dict[int, float]:
    """Floor time across a regulation game. Five men on the court is 240."""
    totals = {int(player.player_id): 0.0 for player in players}
    played = {int(player.player_id): 0.0 for player in players}
    for period in (1, 2, 3, 4):
        left = 720.0
        while left > 1e-6:
            five = select_lineup(players, lead, period, left, played_seconds=played)
            step = min(12.0, left)
            for index in five:
                pid = int(players[index].player_id)
                totals[pid] += step / 60.0
                played[pid] += step
            left -= step
    return totals


def absorb(profiles: dict[int, PlayerCourt], absent: list[int], matrix: dict[tuple[str, str], float]) -> None:
    """Move an absent player's own minutes onto the positions the matrix names.

    The matrix is position→position and each missing-position column sums to 1.
    That share is split across the men still available at the receiving position,
    weighted by their current minutes, so a full wing room does not each take
    the whole column.
    """
    absent_set = {int(pid) for pid in absent}
    for pid in absent:
        missing = profiles.get(pid)
        if missing is None:
            continue
        donated = float(missing.minutes)
        close = float(missing.minutes_close) if missing.minutes_close > 0 else donated
        blow = float(missing.minutes_blowout) if missing.minutes_blowout > 0 else donated
        group = missing.position or "W"
        missing.minutes = 0.0
        missing.minutes_close = 0.0
        missing.minutes_blowout = 0.0
        missing.starts = False
        if donated <= 0:
            continue
        by_pos: dict[str, list[PlayerCourt]] = {}
        for other in profiles.values():
            if int(other.player_id) in absent_set or int(other.player_id) == int(pid):
                continue
            by_pos.setdefault(other.position or "W", []).append(other)
        for pos, others in by_pos.items():
            share = float(matrix.get((pos, group), 0.0))
            if share <= 0 or not others:
                continue
            weights = np.array([max(float(other.minutes), 0.1) for other in others], dtype=float)
            weights = weights / float(weights.sum())
            for other, weight in zip(others, weights):
                other.minutes += share * float(weight) * donated
                other.minutes_close += share * float(weight) * close
                other.minutes_blowout += share * float(weight) * blow
                if share > 0 and other.position == group and pid not in other.comes_in_for:
                    other.comes_in_for = tuple(sorted(set(other.comes_in_for) | {pid}))


def status_play_rates(joined: pd.DataFrame | None) -> dict[str, float]:
    """How often each pre-tip tag actually played. A thin tag shrinks toward the pooled rate."""
    if joined is None or getattr(joined, "empty", True) or "status" not in joined.columns:
        return {}
    work = joined.copy()
    work["status"] = work["status"].astype(str).str.strip().str.lower()
    if "played" in work.columns:
        played = pd.to_numeric(work["played"], errors="coerce").fillna(0) > 0
    elif "min" in work.columns:
        played = _num(work, "min") > 0
    else:
        return {}
    work = work.assign(played=played)
    work = work[work["status"] != ""]
    if work.empty:
        return {}
    pooled = float(work["played"].mean())
    rates = {}
    for status, group in work.groupby("status"):
        count = float(len(group))
        rates[str(status)] = _pool(count, float(group["played"].mean()), pooled, 8.0)
    return rates


def play_chance(label: str, rates: dict[str, float] | None, profile: PlayerCourt | None = None, b2b: bool = False, three: bool = False, heavy: bool = False) -> float:
    """Chance he is available. Out is zero. A tag uses the rate that tag played."""
    text = str(label or "").strip().lower()
    if text.startswith("out") or "suspend" in text:
        return 0.0
    if text and rates and text in rates:
        return float(rates[text])
    if profile is None:
        return 1.0 if not text else 0.5
    chance = float(profile.play_rate)
    if b2b:
        chance = min(chance, 1.0 - float(profile.rest_b2b))
    elif three:
        chance = min(chance, 1.0 - float(profile.rest_3in4))
    elif heavy:
        chance = min(chance, 1.0 - float(profile.rest_heavy))
    if text and not rates:
        # No joined sample yet. The tag is a chance of sitting, read from his own rest history.
        chance *= 1.0 - max(float(profile.rest_b2b), float(profile.rest_heavy), 0.0)
    return float(np.clip(chance, 0.0, 1.0))


def availability_mask(players: list, n: int, rng: np.random.Generator, *, b2b: bool = False, three: bool = False, heavy: bool = False, rates: dict[str, float] | None = None) -> np.ndarray:
    """One draw per path. A confirmed out is off in every path. A rest draw removes him from some paths only."""
    from nba_sim.features.availability import named_sit

    mask = np.ones((n, len(players)), dtype=bool)
    for index, player in enumerate(players):
        if not getattr(player, "available", True):
            mask[:, index] = False
            continue
        profile = getattr(player, "court", None)
        label = str(getattr(player, "injury_label", "") or "")
        factor = float(getattr(player, "injury_factor", 1.0) or 1.0)
        text = label.strip().lower()
        if text.startswith("out") or "suspend" in text or factor <= 0 or named_sit(text):
            mask[:, index] = False
            continue
        rates_key = text if rates and text in rates else ""
        if rates_key:
            chance = float(rates[rates_key])
        elif text and factor < 0.999:
            chance = factor
        else:
            chance = play_chance("", None, profile, b2b=b2b, three=three, heavy=heavy)
        if chance >= 0.999:
            continue
        if chance <= 0.0:
            mask[:, index] = False
            continue
        mask[:, index] = rng.random(n) < chance
    return mask


def trip_price(player_ids: list[int], squads: dict[frozenset[int], SquadFace], priors: dict[int, float] | None = None, individual: dict[str, float] | None = None) -> dict:
    """Usage, mix, and the stored level of the five standing there."""
    priors = priors or {int(pid): 0.2 for pid in player_ids}
    individual = individual or {}
    usage = relative_usage(player_ids, squads, priors)
    mix = squad_mix(player_ids, squads, individual) if individual or squads else {}
    off, deff, pace, oreb = squad_level(player_ids, squads)
    return {"usage": usage, "mix": mix, "off": off, "def": deff, "pace": pace, "oreb": oreb}


def repaired_level(players: list, player_ids: list[int], squads: dict[frozenset[int], SquadFace], team_off: float = 114.0) -> float:
    """A five the file has not seen. The level is the five's own on/off synergy."""
    key = frozenset(int(pid) for pid in player_ids)
    face = squads.get(key)
    if face is not None and face.off_rating > 0 and face.minutes >= 8:
        return float(face.off_rating)
    by_id = {int(getattr(player, "player_id", 0)): player for player in players}
    impacts = [float(getattr(by_id[pid], "off_impact", 1.0) or 1.0) for pid in player_ids if pid in by_id]
    if not impacts:
        return float(team_off)
    return float(team_off * float(np.mean(impacts)))
