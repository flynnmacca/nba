"""Parse league shot-location frames into one row per entity and zone."""

from __future__ import annotations

import pandas as pd

from nba_sim.constants import SHOT_ZONE_LABELS


def _parts(column: object) -> list[str]:
    if isinstance(column, tuple):
        return [str(part) for part in column if str(part) not in ("", "nan", "None")]
    text = str(column)
    if "|" in text:
        return text.split("|")
    return [text]


def _find(columns: list[object], token: str) -> object | None:
    for column in columns:
        if any(part.upper() == token for part in _parts(column)):
            return column
    return None


def parse_shot_zones(frame: pd.DataFrame, season: str, entity_type: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(
            columns=["season", "entity_type", "entity_id", "zone", "fgm", "fga", "fg_pct", "freq"]
        )
    columns = list(frame.columns)
    id_token = "PLAYER_ID" if entity_type == "P" else "TEAM_ID"
    id_col = _find(columns, id_token) or _find(columns, "TEAM_ID")
    if id_col is None:
        return pd.DataFrame()
    made: dict[str, pd.Series] = {}
    attempted: dict[str, pd.Series] = {}
    for column in columns:
        label = None
        metric = ""
        for part in _parts(column):
            if part in SHOT_ZONE_LABELS:
                label = SHOT_ZONE_LABELS[part]
            metric = part.upper()
        if label is None:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        if metric == "FGM":
            made[label] = made.get(label, 0.0) + values
        elif metric == "FGA":
            attempted[label] = attempted.get(label, 0.0) + values
    if not attempted:
        return pd.DataFrame()
    entity_ids = frame[id_col]
    rows = []
    total_fga = None
    for zone, fga in attempted.items():
        total_fga = fga if total_fga is None else total_fga + fga
    for zone, fga in attempted.items():
        fgm = made.get(zone, pd.Series(0.0, index=frame.index))
        for entity_id, made_n, att_n, tot in zip(entity_ids, fgm, fga, total_fga):
            if pd.isna(entity_id) or att_n <= 0:
                continue
            rows.append(
                {
                    "season": season,
                    "entity_type": entity_type,
                    "entity_id": int(entity_id),
                    "zone": zone,
                    "fgm": float(made_n),
                    "fga": float(att_n),
                    "fg_pct": float(made_n) / float(att_n),
                    "freq": float(att_n) / float(tot) if tot else 0.0,
                }
            )
    return pd.DataFrame(rows)
