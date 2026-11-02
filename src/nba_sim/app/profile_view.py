"""Team and player pages opened from the slate."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from nba_sim.profiles import load_player_profile, load_team_profile, main_prop, sim_over_rate
from nba_sim.report import floor_bits
from nba_sim.teams import team_name


def open_team(abbr: str) -> None:
    st.session_state.screen = "team"
    st.session_state.profile_team = abbr.lower()
    st.rerun()


def open_player(espn_id: str, team: str, name: str) -> None:
    st.session_state.screen = "player"
    st.session_state.profile_player = str(espn_id)
    st.session_state.profile_team = team.lower()
    st.session_state.profile_name = name
    st.rerun()


def close_profile() -> None:
    st.session_state.screen = "slate"
    st.rerun()


def render_profile() -> bool:
    screen = st.session_state.get("screen", "slate")
    if screen == "team":
        _team_page(str(st.session_state.get("profile_team") or ""))
        return True
    if screen == "player":
        _player_page(
            str(st.session_state.get("profile_player") or ""),
            str(st.session_state.get("profile_team") or ""),
            str(st.session_state.get("profile_name") or ""),
        )
        return True
    return False


def _team_page(abbr: str) -> None:
    if st.button("Back to slate"):
        close_profile()
    with st.spinner(f"Loading {team_name(abbr)}…"):
        try:
            profile = load_team_profile(abbr)
        except Exception as exc:
            st.error(f"The team profile did not load. {exc}")
            return
    st.title(profile["name"])
    st.caption("Season stats from ESPN, with the current roster and injury report.")
    _stat_groups(profile["stats"])
    _injury_table(profile["injuries"])
    st.subheader("Roster")
    roster = profile["roster"]
    if roster.empty:
        st.info("ESPN did not return a roster for this team.")
        return
    _roster_buttons(roster)
    st.dataframe(_roster_table(roster, profile["injuries"]), hide_index=True, width="stretch")


def _player_page(espn_id: str, team: str, name: str) -> None:
    back, team_button = st.columns(2)
    with back:
        if st.button("Back to slate"):
            close_profile()
    with team_button:
        if team and st.button(f"{team_name(team)} roster"):
            open_team(team)
    if not espn_id:
        st.info("This sim player is not linked to an ESPN id.")
        return
    with st.spinner(f"Loading {name or 'player'}…"):
        try:
            profile = load_player_profile(espn_id, team)
        except Exception as exc:
            st.error(f"The player profile did not load. {exc}")
            return
    bio = profile["bio"] or {}
    st.title(str(bio.get("name") or name or "Player"))
    _bio_line(bio)
    _sim_panel(espn_id)
    _wide_stats("Season stats", profile["season"])
    _wide_stats("Advanced", profile["advanced"])
    _gamelog(profile["logs"])


def _stat_groups(frame: pd.DataFrame) -> None:
    if frame is None or frame.empty:
        st.info("ESPN has no season stat line for this team yet.")
        return
    categories = list(dict.fromkeys(frame["category"].astype(str)))
    tabs = st.tabs(categories)
    for tab, category in zip(tabs, categories, strict=False):
        with tab:
            shown = frame[frame["category"] == category][["stat", "value"]].rename(
                columns={"stat": "Stat", "value": "Value"}
            )
            st.dataframe(shown, hide_index=True, width="stretch")


def _injury_table(frame: pd.DataFrame) -> None:
    st.subheader("Injuries")
    if frame is None or frame.empty:
        st.caption("No current injury rows for this team.")
        return
    columns = [
        column
        for column in ("player_name", "status", "fantasy", "body_part", "return_date", "comment")
        if column in frame.columns
    ]
    shown = frame[columns].rename(
        columns={
            "player_name": "Player",
            "status": "Status",
            "fantasy": "Tag",
            "body_part": "Body part",
            "return_date": "Target return",
            "comment": "Report",
        }
    )
    st.dataframe(shown, hide_index=True, width="stretch")


def _roster_buttons(roster: pd.DataFrame) -> None:
    columns = st.columns(3)
    for index, row in enumerate(roster.to_dict("records")):
        name = str(row.get("player_name") or "")
        espn_id = str(row.get("espn_id") or "")
        team = str(row.get("team_abbr") or "")
        with columns[index % 3]:
            if st.button(name, key=f"roster-{team}-{espn_id}"):
                open_player(espn_id, team, name)


def _roster_table(roster: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    status = {}
    if injuries is not None and not injuries.empty and "player_name" in injuries.columns:
        for row in injuries.to_dict("records"):
            status.setdefault(str(row.get("player_name") or ""), str(row.get("status") or ""))
    rows = []
    for row in roster.to_dict("records"):
        name = str(row.get("player_name") or "")
        rows.append(
            {
                "Player": name,
                "Pos": row.get("position") or "",
                "No": row.get("jersey") or "",
                "Height": row.get("height") or "",
                "Weight": row.get("weight") or "",
                "Age": row.get("age") or "",
                "Injury": status.get(name, ""),
            }
        )
    return pd.DataFrame(rows)


def _bio_line(bio: dict) -> None:
    bits = []
    for label, key in (
        ("Age", "age"),
        ("Height", "height"),
        ("Weight", "weight"),
        ("Wingspan", "wingspan"),
        ("Position", "position"),
        ("Draft", "draft"),
        ("Experience", "experience"),
        ("Birthplace", "birthplace"),
    ):
        value = bio.get(key)
        if value is None or str(value).strip() in {"", "nan", "None"}:
            if key == "wingspan":
                bits.append("Wingspan not in the draft combine")
            continue
        bits.append(f"{label} {value}")
    st.caption(" · ".join(bits) if bits else "ESPN did not return a bio for this player.")


def _wide_stats(title: str, frame: pd.DataFrame | None) -> None:
    st.subheader(title)
    if frame is None or frame.empty or "stat" not in frame.columns:
        st.caption("No rows from ESPN for this table.")
        return
    wide = frame.pivot_table(index="split", columns="stat", values="value", aggfunc="first")
    ordered = [stat for stat in frame["stat"].tolist() if stat in wide.columns]
    wide = wide.reindex(columns=list(dict.fromkeys(ordered)))
    st.dataframe(wide.reset_index().rename(columns={"split": "Split"}), hide_index=True, width="stretch")


def _gamelog(frame: pd.DataFrame) -> None:
    st.subheader("Game log")
    if frame is None or frame.empty:
        st.caption("No games logged for this season.")
        return
    shown = frame.rename(
        columns={
            "date": "Date",
            "tip_local": "Local tip",
            "tip_melbourne": "Melbourne",
            "opponent": "Opp",
            "home_away": "Site",
            "result": "Result",
            "score": "Score",
            "minutes": "MIN",
            "fg": "FG",
            "threes": "3PT",
            "ft": "FT",
            "rebounds": "REB",
            "assists": "AST",
            "steals": "STL",
            "blocks": "BLK",
            "turnovers": "TO",
            "points": "PTS",
            "season": "Season",
        }
    )
    columns = [
        "Date",
        "Local tip",
        "Melbourne",
        "Opp",
        "Site",
        "Result",
        "Score",
        "MIN",
        "FG",
        "3PT",
        "FT",
        "REB",
        "AST",
        "STL",
        "BLK",
        "TO",
        "PTS",
        "Season",
    ]
    st.dataframe(shown[[column for column in columns if column in shown.columns]], hide_index=True, width="stretch")


def _sim_panel(espn_id: str) -> None:
    summary = st.session_state.get("summary") or {}
    bins = pd.DataFrame(summary.get("player_bins") or [])
    if bins.empty or "espn_id" not in bins.columns:
        return
    mine = bins[bins["espn_id"].astype(str) == str(espn_id)]
    if mine.empty:
        return
    st.subheader("This slate's simulations")
    points = mine[mine["stat"] == "pts"]
    if points.empty:
        return
    info = points.iloc[0]
    prop = main_prop(st.session_state.get("props"), espn_id, "pts")
    rate = None if prop is None else sim_over_rate(points, float(prop["line"]))
    fair = None if prop is None else prop.get("fair_over")
    sentence = (
        f"{info['player']} averages {info['mean']:.1f} points in this sim, "
        f"with a typical game between {info['p10']:.0f} and {info['p90']:.0f}."
    )
    usg = info.get("usg")
    if usg is not None and not pd.isna(usg):
        sentence += f" While he is on the floor he uses {float(usg):.0%} of the possessions."
    sentence += "".join(f" {bit}" for bit in floor_bits(info))
    if prop is not None and rate is not None:
        sentence += f" The points prop is {float(prop['line']):g}. The sim is over that number in {rate:.0%} of runs."
        if fair is not None and not pd.isna(fair):
            sentence += f" The de-vigged over price is {float(fair):.0%}."
    st.caption(sentence)
    bars = (
        alt.Chart(points)
        .mark_bar(color="#1d4ed8", cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("value:Q", title="Points", axis=alt.Axis(format="d")),
            y=alt.Y("sims:Q", title="Simulations"),
            tooltip=[alt.Tooltip("value:Q", title="Points", format="d"), "sims"],
        )
        .properties(height=220)
    )
    layers = [bars]
    if prop is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"line": [float(prop["line"])]}))
            .mark_rule(color="#b45309", strokeWidth=2)
            .encode(x="line:Q")
        )
    st.altair_chart(alt.layer(*layers), width="stretch")
