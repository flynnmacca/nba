"""Box participation and the whistles inside games already in the logs.

A box row is who played, for how many minutes, and why he sat. A substitution
row is who left, who entered, the period, and the clock. Rows on the same
clock are one whistle. A possession row is the five on the floor, who used it,
and the action. The pull is resumable and does not touch the schedule.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx
import pandas as pd

from nba_sim.config import SETTINGS, warehouse_dir
from nba_sim.features.availability import NBA_BOX_URL, fetch_nba_box
from nba_sim.teams import normalize_team

PBP_URL = "https://cdn.nba.com/static/json/liveData/playbyplay/playbyplay_{game_id}.json"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.nba.com/"}

_ACTION = {
    "transition": "transition",
    "prballhandler": "pick",
    "prrollman": "pick",
    "postup": "post",
    "spotup": "spot",
    "isolation": "spot",
    "handoff": "spot",
    "cut": "spot",
    "offscreen": "spot",
    "pullup": "spot",
    "stepback": "spot",
    "driving": "spot",
    "dunk": "spot",
    "layup": "spot",
    "hook": "post",
    "fadeaway": "post",
}


def participation_path():
    return warehouse_dir() / "participation.parquet"


def subs_path():
    return warehouse_dir() / "substitutions.parquet"


def possessions_path():
    return warehouse_dir() / "possessions.parquet"


def _token(game_id: object) -> str:
    return str(game_id or "").split(".")[0].zfill(10)


def _pt_minutes(value: object) -> float:
    text = str(value or "").strip()
    if not text or text in {"0", "0:00", "PT00M00.00S"}:
        return 0.0
    if text.startswith("PT"):
        body = text[2:]
        minutes = 0.0
        seconds = 0.0
        if "M" in body:
            mins, body = body.split("M", 1)
            minutes = float(mins or 0)
        if body.endswith("S"):
            seconds = float(body[:-1] or 0)
        return minutes + seconds / 60.0
    if ":" in text:
        left, right = text.split(":", 1)
        return float(left or 0) + float(right or 0) / 60.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _did_not_play(person: dict) -> bool:
    if str(person.get("status") or "").upper() == "INACTIVE":
        return True
    return str(person.get("played")).strip().lower() in {"0", "false"}


def _reason(person: dict) -> str:
    detail = str(person.get("notPlayingDescription") or "").strip()
    if detail:
        return detail
    code = str(person.get("notPlayingReason") or "").strip()
    if code:
        return code
    if str(person.get("status") or "").upper() == "ACTIVE":
        return "Coach's Decision"
    return ""


def parse_participation(payload: dict | None, game_id: str) -> list[dict]:
    """One row per player on the official box."""
    if not payload:
        return []
    game = payload.get("game") or payload
    rows = []
    token = _token(game_id)
    for side_key in ("awayTeam", "homeTeam"):
        side = game.get(side_key) or {}
        team_id = side.get("teamId")
        team = normalize_team(side.get("teamTricode")) or ""
        for person in side.get("players") or []:
            stats = person.get("statistics") or {}
            sat = _did_not_play(person)
            minutes = 0.0 if sat else _pt_minutes(stats.get("minutes") or person.get("minutes"))
            if sat:
                minutes = 0.0
            rows.append(
                {
                    "game_id": token,
                    "player_id": int(person.get("personId") or 0),
                    "team_id": int(team_id or 0),
                    "team": team,
                    "min": float(minutes),
                    "status": str(person.get("status") or ""),
                    "reason": _reason(person) if sat else "",
                    "played": 0 if sat or minutes <= 0 else 1,
                }
            )
    return [row for row in rows if row["player_id"]]


def fetch_pbp(client: httpx.Client, game_id: str) -> dict | None:
    token = _token(game_id)
    response = client.get(PBP_URL.format(game_id=token), timeout=15, headers=_HEADERS)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else None


def _clock(action: dict) -> str:
    return str(action.get("clock") or action.get("actionClock") or "")


def _lineup_text(action: dict, side: str) -> str:
    players = action.get(side) or action.get(f"{side}Players") or []
    ids = []
    for person in players:
        if isinstance(person, dict):
            pid = person.get("personId") or person.get("playerId")
        else:
            pid = person
        if pid:
            ids.append(str(int(pid)))
    if len(ids) != 5:
        return ""
    return "-".join(ids)


def _action_name(action: dict) -> str:
    pieces = " ".join(
        str(action.get(key) or "")
        for key in ("actionType", "subType", "descriptor", "shotType")
    ).lower()
    for key, label in _ACTION.items():
        if key in pieces.replace(" ", ""):
            return label
    if "3pt" in pieces or "three" in pieces:
        return "three"
    if "turnover" in pieces:
        return "turnover"
    if "foul" in pieces:
        return "foul"
    return "spot"


def parse_events(payload: dict | None, game_id: str, starters: dict[int, list[int]] | None = None) -> tuple[list[dict], list[dict]]:
    """Substitution rows and possession rows.

    The five is rebuilt from the box starters and the substitutions, because
    the live feed does not print the lineup on the shot.
    """
    if not payload:
        return [], []
    game = payload.get("game") or payload
    actions = game.get("actions") or []
    token = _token(game_id)
    actions = sorted(actions, key=lambda row: int(row.get("orderNumber") or row.get("actionNumber") or 0))
    on_court: dict[int, list[int]] = {team: list(ids) for team, ids in (starters or {}).items()}
    subs = []
    trips = []
    waiting: dict[tuple, list[int]] = {}
    for action in actions:
        kind = str(action.get("actionType") or "").lower()
        period = int(action.get("period") or 0)
        clock = _clock(action)
        team_id = int(action.get("teamId") or 0)
        if kind == "substitution":
            sub = str(action.get("subType") or "").lower()
            pid = int(action.get("personId") or 0)
            if not pid or not team_id:
                continue
            court = on_court.setdefault(team_id, [])
            if sub == "out":
                if pid in court:
                    court.remove(pid)
                waiting.setdefault((team_id, period, clock), []).append(pid)
            elif sub == "in":
                if pid not in court:
                    court.append(pid)
                outs = waiting.get((team_id, period, clock)) or []
                left = outs.pop(0) if outs else 0
                if left:
                    subs.append(
                        {
                            "game_id": token,
                            "period": period,
                            "clock": clock,
                            "player_out": left,
                            "player_in": pid,
                            "team_id": team_id,
                            "margin": 0,
                        }
                    )
            continue
        if kind not in {"2pt", "3pt", "turnover"}:
            continue
        shooter = int(action.get("personId") or 0)
        court = on_court.get(team_id) or []
        if not shooter or len(court) != 5 or shooter not in court:
            continue
        trips.append(
            {
                "game_id": token,
                "period": period,
                "clock": clock,
                "player_id": shooter,
                "lineup": "-".join(str(int(pid)) for pid in court),
                "action": "three" if kind == "3pt" else ("turnover" if kind == "turnover" else _action_name(action)),
                "team_id": team_id,
            }
        )
    return subs, trips


def _starters(payload: dict | None) -> dict[int, list[int]]:
    if not payload:
        return {}
    game = payload.get("game") or payload
    found: dict[int, list[int]] = {}
    for side_key in ("awayTeam", "homeTeam"):
        side = game.get(side_key) or {}
        team_id = int(side.get("teamId") or 0)
        if not team_id:
            continue
        ids = []
        for person in side.get("players") or []:
            if str(person.get("starter") or "") in {"1", "true", "True"}:
                pid = int(person.get("personId") or 0)
                if pid:
                    ids.append(pid)
        if len(ids) == 5:
            found[team_id] = ids
    return found


def _read(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write(path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def game_ids_in_logs(logs: pd.DataFrame | None = None) -> list[str]:
    if logs is None:
        path = warehouse_dir() / "game_logs.parquet"
        if not path.exists():
            return []
        logs = pd.read_parquet(path, columns=["game_id"])
    if logs is None or logs.empty or "game_id" not in logs.columns:
        return []
    return sorted({_token(value) for value in logs["game_id"].dropna().unique() if str(value).strip()})


def _pull_one(token: str) -> tuple[list[dict], list[dict], list[dict], bool]:
    """Box rows, substitution rows, possession rows, and whether the box was empty."""
    try:
        with httpx.Client(timeout=12, follow_redirects=True, headers=_HEADERS) as client:
            try:
                payload = fetch_nba_box(client, token, timeout=12)
            except httpx.HTTPError:
                payload = None
            rows = parse_participation(payload, token)
            try:
                feed = fetch_pbp(client, token)
                subs, trips = parse_events(feed, token, _starters(payload))
            except httpx.HTTPError:
                subs, trips = [], []
    except httpx.HTTPError:
        return [], [], [], True
    return rows, subs, trips, not rows


def cache_participation(game_ids: list[str] | None = None, limit: int | None = None, pause: float | None = None) -> dict:
    """Pull the boxes and the whistles that are not already on disk."""
    ids = game_ids if game_ids is not None else game_ids_in_logs()
    have = set()
    existing = _read(participation_path())
    if not existing.empty and "game_id" in existing.columns:
        have = set(existing["game_id"].astype(str))
    pending = [token for token in ids if token not in have]
    if limit is not None:
        pending = pending[: int(limit)]
    pause = SETTINGS.request_pause_seconds if pause is None else pause
    boxes: list[dict] = []
    subs: list[dict] = []
    trips: list[dict] = []
    missed = 0
    workers = 8 if pause else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_pull_one, token) for token in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            rows, left, used, empty = future.result()
            if rows:
                boxes.extend(rows)
            if empty:
                missed += 1
            subs.extend(left)
            trips.extend(used)
            if index % 25 == 0 or index == len(pending):
                _flush(existing, boxes, subs, trips)
                existing = _read(participation_path())
                boxes, subs, trips = [], [], []
                print(index, "of", len(pending), "missed", missed, flush=True)
            if pause and workers == 1:
                time.sleep(pause)
    stored = _read(participation_path())
    return {"games": int(stored["game_id"].nunique()) if not stored.empty else 0, "missed": missed, "pulled": len(pending)}


def _flush(existing: pd.DataFrame, boxes: list[dict], subs: list[dict], trips: list[dict]) -> None:
    if boxes:
        frame = pd.DataFrame(boxes)
        merged = pd.concat([existing, frame], ignore_index=True) if not existing.empty else frame
        merged = merged.drop_duplicates(["game_id", "player_id"], keep="last")
        _write(participation_path(), merged)
    if subs:
        old = _read(subs_path())
        frame = pd.DataFrame(subs)
        merged = pd.concat([old, frame], ignore_index=True) if not old.empty else frame
        merged = merged.drop_duplicates(["game_id", "period", "clock", "player_out", "player_in"], keep="last")
        _write(subs_path(), merged)
    if trips:
        old = _read(possessions_path())
        frame = pd.DataFrame(trips)
        merged = pd.concat([old, frame], ignore_index=True) if not old.empty else frame
        _write(possessions_path(), merged)


def main() -> None:
    print(cache_participation())


if __name__ == "__main__":
    main()
