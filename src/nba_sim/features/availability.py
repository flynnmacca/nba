"""Known absences from a finished box score.

A finished box names every sit, including a dressed coach's decision. Rest,
inactive, injury, illness, and assignment are the same kind of sit once the
box has printed them. Without a finished box, a coach's decision stays on the
player's own scratch rate for a live prediction.
"""

from __future__ import annotations

import re

from nba_sim.names import norm_person
from nba_sim.sim.specs import PlayerModel
from nba_sim.teams import normalize_team

NBA_BOX_URL = "https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json"
_COACH = re.compile(r"\bcoach(?:es|'s)?\b")
_REST = re.compile(r"\brest\b")


def _reason_text(reason: object) -> str:
    return " ".join(str(reason or "").lower().replace("_", " ").split())


def named_sit(reason: object) -> bool:
    """Rest or inactive. The team has already taken him off the floor."""
    text = _reason_text(reason)
    if not text:
        return False
    return text.startswith("inactive") or _REST.search(text) is not None


def known_out_reason(reason: object) -> bool:
    """A named absence. A dressed coach's decision stays his own scratch rate."""
    text = _reason_text(reason)
    if not text:
        return False
    if named_sit(text):
        return True
    if _COACH.search(text):
        return False
    return True


def fetch_nba_box(client, game_id: str, timeout: float = 60) -> dict | None:
    """The official NBA.com box score. A missing file returns nothing."""
    token = str(game_id or "").split(".")[0].zfill(10)
    response = client.get(
        NBA_BOX_URL.format(game_id=token),
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"},
    )
    if response.status_code == 404:
        return None
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else None


def nba_box_absences(payload: dict) -> list[dict]:
    """Players the official box lists with zero minutes, and the reason it prints."""
    game = payload.get("game") or payload
    rows = []
    for side_key in ("awayTeam", "homeTeam"):
        side = game.get(side_key) or {}
        team = normalize_team(side.get("teamTricode")) or ""
        for person in side.get("players") or []:
            if not _did_not_play(person):
                continue
            name = str(person.get("name") or "").strip()
            if not name:
                name = f"{person.get('firstName') or ''} {person.get('familyName') or ''}".strip()
            rows.append({"team": team, "name": name, "reason": _nba_reason(person)})
    return rows


def _did_not_play(person: dict) -> bool:
    if str(person.get("status") or "").upper() == "INACTIVE":
        return True
    return str(person.get("played")).strip().lower() in {"0", "false"}


def _nba_reason(person: dict) -> str:
    detail = str(person.get("notPlayingDescription") or "").strip()
    if detail:
        return detail
    code = str(person.get("notPlayingReason") or "").strip()
    if code:
        return code
    if str(person.get("status") or "").upper() == "ACTIVE":
        return "Coach's Decision"
    return "Inactive"


def box_absences(payload: dict) -> list[dict]:
    """Players the box lists as not playing, with the reason printed beside the name."""
    rows = []
    for side in (payload.get("boxscore") or {}).get("players") or []:
        team = normalize_team((side.get("team") or {}).get("abbreviation")) or ""
        group = (side.get("statistics") or [{}])[0]
        for athlete in group.get("athletes") or []:
            if not athlete.get("didNotPlay"):
                continue
            person = athlete.get("athlete") or {}
            rows.append(
                {
                    "team": team,
                    "name": str(person.get("displayName") or ""),
                    "reason": str(athlete.get("reason") or ""),
                }
            )
    return rows


def hold_known_outs(players: list[PlayerModel], team: str, absences: list[dict]) -> list[str]:
    """Sit the men this finished box lists as out, including coach's decisions.

    A live injury report still filters through ``known_out_reason``, which leaves
    a dressed coach scratch on his own play rate. Eval and profiled games pass
    the box here, so every named zero-minute player is held and his minutes
    move onto the men who absorbed him.
    """
    side = (normalize_team(team) or str(team or "").lower())
    outs = {
        norm_person(row.get("name")): str(row.get("reason") or "")
        for row in absences
        if (normalize_team(row.get("team")) or str(row.get("team") or "").lower()) == side
        and _reason_text(row.get("reason"))
    }
    held: list[str] = []
    for player in players:
        reason = outs.get(norm_person(player.name))
        if reason is None:
            continue
        if player.minutes_base <= 0:
            player.minutes_base = player.minutes_mean
        player.available = False
        player.minutes_mean = 0.0
        player.injury_factor = 0.0
        # Roster inactives / G-League sits should not dump historical minutes onto actives.
        text = _reason_text(reason)
        player.injury_label = "Inactive" if text.startswith("inactive") else "Out"
        held.append(player.name)
    if held:
        _release_minutes(players)
    return held


def _release_minutes(players: list[PlayerModel]) -> None:
    """His night moves onto the positions that have absorbed him, then his court is clear."""
    from nba_sim.features.court import absorb

    profiles = {int(player.player_id): player.court for player in players if getattr(player, "court", None) is not None}
    matrix = getattr(players[0], "_sub_matrix", None) if players else None
    if not profiles or not matrix:
        for player in players:
            if player.available:
                continue
            profile = getattr(player, "court", None)
            if profile is None:
                continue
            profile.minutes = 0.0
            profile.minutes_close = 0.0
            profile.minutes_blowout = 0.0
            profile.starts = False
        return
    donate: list[int] = []
    for player in players:
        if player.available or int(player.player_id) not in profiles:
            continue
        profile = profiles[int(player.player_id)]
        if str(getattr(player, "injury_label", "") or "").strip().lower() == "inactive":
            profile.minutes = 0.0
            profile.minutes_close = 0.0
            profile.minutes_blowout = 0.0
            profile.starts = False
            player.minutes_mean = 0.0
            continue
        donate.append(int(player.player_id))
    if donate:
        absorb(profiles, donate, matrix)
    for player in players:
        profile = profiles.get(int(player.player_id))
        if profile is None:
            continue
        if not player.available:
            profile.minutes = 0.0
            profile.minutes_close = 0.0
            profile.minutes_blowout = 0.0
            profile.starts = False
            player.minutes_mean = 0.0
            continue
        if int(getattr(profile, "sample", 0) or 0) > 0 or float(profile.minutes) > 0:
            player.minutes_mean = float(profile.minutes)
