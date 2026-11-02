"""One-slate board: scoreboard, distributions, matchups, and prices."""

from __future__ import annotations

import time
from contextlib import contextmanager
from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

from nba_sim.app.profile_view import open_player, open_team, render_profile
from nba_sim.backtest.walkforward import run_backtest
from nba_sim.config import raw_dir
from nba_sim.features.prepare import build_specs_for_date, season_for
from nba_sim.geo import melbourne_tip_date, melbourne_today
from nba_sim.markets.results import box_actuals, grade_pick, list_future_games, pick_record
from nba_sim.profiles import load_props, main_prop, sim_over_rate
from nba_sim.report import floor_bits, spread_view, summarize
from nba_sim.sim.demo import build_demo_slate
from nba_sim.sim.engine import simulate_game
from nba_sim.sim.specs import GameSpec
from nba_sim.warehouse import Warehouse

HOME_COLOR = "#1b4dff"
AWAY_COLOR = "#c1121f"
PLAY_LABELS = {
    "Isolation": "Isolation",
    "Transition": "Transition",
    "PRBallHandler": "PnR handler",
    "PRRollman": "PnR roll",
    "Postup": "Post-up",
    "Spotup": "Spot-up",
    "Handoff": "Handoff",
    "Cut": "Cut",
    "OffScreen": "Off screen",
    "OffRebound": "Putback",
    "Misc": "Misc",
}
def _schedule() -> pd.DataFrame:
    frame = Warehouse().read("schedule")
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame


def _tip_date(value: object) -> date | None:
    text = str(value or "")
    if not text or text in {"nan", "None", "NaT"}:
        return None
    try:
        return melbourne_tip_date(text)
    except ValueError:
        return None


def _on_melbourne_date(schedule: pd.DataFrame, on: date) -> pd.DataFrame:
    """Games whose tip falls on this Melbourne calendar date."""
    if schedule.empty or "start_utc" not in schedule.columns:
        return schedule.iloc[0:0]
    tips = schedule["start_utc"].map(_tip_date)
    return schedule.loc[tips == on].sort_values("start_utc")


def _spec_cache() -> dict:
    """Survive Streamlit reruns. Module globals in this file are wiped each click."""
    store = st.session_state.setdefault("_spec_cache", {})
    if not isinstance(store, dict):
        store = {}
        st.session_state["_spec_cache"] = store
    return store


def _specs(source: str, on: date, game_id: str | None = None) -> tuple[list[GameSpec], str]:
    stamp = _warehouse_stamp() if source == "Warehouse" else "demo"
    cache_key = (source, on.isoformat(), str(game_id or ""), stamp)
    cache = _spec_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if source == "Warehouse":
        schedule = _schedule()
        if schedule.empty:
            return build_demo_slate(on), "demo-fallback"
        day = _on_melbourne_date(schedule, on)
        if day.empty:
            return [], "empty"
        wanted = {str(game_id) for game_id in day["game_id"]}
        if game_id is not None:
            wanted = {str(game_id)} if str(game_id) in wanted else set()
            if not wanted:
                return [], "empty"
        order = [str(item) for item in day["game_id"] if str(item) in wanted]
        if game_id is not None:
            match = day[day["game_id"].astype(str) == str(game_id)]
            venue_days = [date.fromisoformat(str(match.iloc[0]["date"])[:10])]
        else:
            venue_days = sorted({date.fromisoformat(str(value)[:10]) for value in day["date"]})
        specs = []
        seen: set[str] = set()
        for venue_day in venue_days:
            for spec in build_specs_for_date(venue_day, Warehouse(), game_id=game_id):
                if spec.game_id in wanted and spec.game_id not in seen:
                    specs.append(spec)
                    seen.add(spec.game_id)
        specs.sort(key=lambda spec: order.index(spec.game_id) if spec.game_id in order else len(order))
        if specs:
            result = (specs, "warehouse")
            cache[cache_key] = result
            while len(cache) > 48:
                oldest = next(iter(cache))
                if oldest == cache_key:
                    break
                cache.pop(oldest, None)
            return result
        return [], "empty"
    return build_demo_slate(on), "demo"


# Specs built after a pull, so a fresh sim does not read the warehouse once per game.
_PRIMED: dict[str, tuple[GameSpec, str]] = {}


def _prime_specs(specs: list[GameSpec], resolved: str) -> None:
    _PRIMED.clear()
    for spec in specs:
        _PRIMED[str(spec.game_id)] = (spec, resolved)


def _parse_prompt(token: str) -> tuple[dict[int, str], dict[int, float]]:
    """``12:on,18:off,7:min=34`` from the game prompt. Anything else stays on the model."""
    modes: dict[int, str] = {}
    minutes: dict[int, float] = {}
    for part in str(token or "").split(","):
        if ":" not in part:
            continue
        pid_text, rest = part.split(":", 1)
        if not pid_text.strip().lstrip("-").isdigit():
            continue
        pid = int(pid_text)
        rest = rest.strip().lower()
        if rest in {"on", "off"}:
            modes[pid] = rest
            continue
        if not rest.startswith("min="):
            continue
        try:
            value = float(rest[4:])
        except ValueError:
            continue
        if 0.0 <= value <= 48.0:
            minutes[pid] = value
    return modes, minutes


def _parse_availability(token: str) -> dict[int, str]:
    """``12:on,18:off`` from the game prompt. Anything else stays on the model."""
    return _parse_prompt(token)[0]


def _snapshot_player(player) -> tuple:
    profile = getattr(player, "court", None)
    return (
        player,
        profile,
        player.available,
        player.injury_label,
        player.injury_factor,
        player.minutes_mean,
        None
        if profile is None
        else (
            profile.minutes,
            profile.minutes_close,
            float(getattr(profile, "minutes_blowout", 0.0) or 0.0),
            profile.starts,
            profile.play_rate,
            profile.rest_b2b,
            profile.rest_3in4,
            profile.rest_heavy,
        ),
    )


def _restore_player(saved: tuple) -> None:
    player, profile, available, label, factor, minutes, court = saved
    player.available = available
    player.injury_label = label
    player.injury_factor = factor
    player.minutes_mean = minutes
    if profile is not None and court is not None:
        (
            profile.minutes,
            profile.minutes_close,
            profile.minutes_blowout,
            profile.starts,
            profile.play_rate,
            profile.rest_b2b,
            profile.rest_3in4,
            profile.rest_heavy,
        ) = court


def _force_minutes(player, profile, value: float) -> None:
    """Lock this player's floor time for the run."""
    minutes = max(0.0, min(48.0, float(value)))
    if minutes <= 0.0:
        player.available = False
        player.injury_label = "Out"
        player.injury_factor = 0.0
        player.minutes_mean = 0.0
        if profile is not None:
            profile.minutes = 0.0
            profile.minutes_close = 0.0
            profile.minutes_blowout = 0.0
        return
    player.available = True
    if not player.injury_label or str(player.injury_label).strip().lower() in {"out", "inactive"}:
        player.injury_label = ""
    player.injury_factor = 1.0
    player.minutes_mean = minutes
    if profile is None:
        return
    profile.minutes = minutes
    profile.minutes_close = minutes
    profile.minutes_blowout = minutes
    profile.play_rate = 1.0
    profile.rest_b2b = 0.0
    profile.rest_3in4 = 0.0
    profile.rest_heavy = 0.0


@contextmanager
def _forced_availability(spec: GameSpec, token: str):
    """On plays every path. Off sits every path. min= locks minutes. Spec is restored."""
    wanted, minute_map = _parse_prompt(token)
    saved: list[tuple] = []
    if not wanted and not minute_map:
        yield
        return
    players = list(spec.home_players) + list(spec.away_players)
    seen: set[int] = set()
    try:
        for player in players:
            pid = int(player.player_id)
            mode = wanted.get(pid)
            has_minutes = pid in minute_map
            if mode not in {"on", "off"} and not has_minutes:
                continue
            if pid in seen:
                continue
            seen.add(pid)
            profile = getattr(player, "court", None)
            saved.append(_snapshot_player(player))
            if mode == "off" or (has_minutes and minute_map[pid] <= 0.0):
                player.available = False
                player.injury_label = "Out"
                player.injury_factor = 0.0
                player.minutes_mean = 0.0
                if profile is not None:
                    profile.minutes = 0.0
                    profile.minutes_close = 0.0
                    profile.minutes_blowout = 0.0
                continue
            if mode == "on":
                player.available = True
                player.injury_label = ""
                player.injury_factor = 1.0
                if player.minutes_mean <= 0:
                    restored = float(player.minutes_base or 0.0)
                    if profile is not None:
                        restored = max(restored, float(profile.minutes_close or 0.0), float(profile.minutes or 0.0))
                    player.minutes_mean = restored
                if profile is not None:
                    if profile.minutes <= 0:
                        profile.minutes = float(player.minutes_mean or 0.0)
                    if profile.minutes_close <= 0:
                        profile.minutes_close = float(profile.minutes or 0.0)
                    profile.play_rate = 1.0
                    profile.rest_b2b = 0.0
                    profile.rest_3in4 = 0.0
                    profile.rest_heavy = 0.0
            if has_minutes:
                _force_minutes(player, profile, minute_map[pid])
        yield
    finally:
        for item in saved:
            _restore_player(item)


@st.cache_data(show_spinner="Simulating possessions…")
def _cached_summary(
    game_id: str,
    source: str,
    on_iso: str,
    n_sims: int,
    seed: int,
    stamp: str,
    availability: str = "",
) -> dict:
    primed = _PRIMED.get(str(game_id))
    if primed is None:
        specs, resolved = _specs(source, date.fromisoformat(on_iso))
        spec = next(item for item in specs if str(item.game_id) == str(game_id))
    else:
        spec, resolved = primed
    spec.source = resolved
    with _forced_availability(spec, availability):
        summary = summarize(spec, simulate_game(spec, n_sims=n_sims, seed=seed))
    summary["board"] = "lineup-season"
    summary["stamp"] = stamp
    summary["availability"] = availability
    return summary


def _rerun_future(n_sims: int, seed: int, stamp: str, today: date) -> None:
    """Run every game that has not been played, from the rates just written."""
    warehouse = Warehouse()
    schedule = warehouse.read("schedule")
    logs = warehouse.read("game_logs") if warehouse.exists("game_logs") else None
    games = list_future_games(schedule, logs, today)
    if not games:
        st.info("No games still to come.")
        return
    progress = st.progress(0.0, text=f"Running {len(games)} games still to come")
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for game in games:
        venue = str(game["venue"])
        if venue not in grouped:
            order.append(venue)
            grouped[venue] = []
        grouped[venue].append(game)
    done = 0
    total = len(games)
    for venue in order:
        specs = build_specs_for_date(date.fromisoformat(venue), warehouse)
        by_id = {str(spec.game_id): spec for spec in specs}
        for game in grouped[venue]:
            done += 1
            spec = by_id.get(str(game["game_id"]))
            if spec is None:
                progress.progress(done / total, text=f"Skipped {game['label']} ({done} of {total})")
                continue
            progress.progress(done / total, text=f"{game['label']} ({done} of {total})")
            _prime_specs([spec], "warehouse")
            _cached_summary(
                str(spec.game_id),
                "Warehouse",
                str(game["melbourne"]),
                n_sims,
                seed,
                stamp,
                "",
            )
    progress.empty()


def _warehouse_stamp(warehouse: Warehouse | None = None) -> str:
    """Changes when this season's tables are rewritten, so the board draws again."""
    warehouse = warehouse or Warehouse()
    latest = 0
    token = ""
    for name in ("game_logs", "player_stats", "team_stats", "play_types", "injuries", "schedule"):
        path = warehouse.path(name)
        if path.exists():
            latest = max(latest, path.stat().st_mtime_ns)
    stamp = warehouse.root / "slate_stamp.txt"
    if stamp.exists():
        latest = max(latest, stamp.stat().st_mtime_ns)
        token = stamp.read_text(encoding="utf-8").strip()
    return f"{latest}|{token}"


def _injuries_age_seconds(warehouse: Warehouse) -> float:
    path = warehouse.path("injuries")
    if not path.exists():
        return 1e18
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return 1e18


def _soft_refresh_espn(warehouse: Warehouse, force: bool = False) -> list[str]:
    """Keep the injury report current without a full Update / future resim."""
    checked = float(st.session_state.get("espn_checked") or 0.0)
    # Prefer file age so a board restart does not re-hit ESPN when the report is fresh.
    if not force and _injuries_age_seconds(warehouse) < 3600:
        return []
    if not force and time.time() - checked < 1800:
        return []
    st.session_state["espn_checked"] = time.time()
    from nba_sim.ingest.espn import ingest_espn, ingest_espn_report

    before = _warehouse_stamp(warehouse)
    with st.spinner("Updating injuries from ESPN…"):
        failures = ingest_espn(warehouse) if force else ingest_espn_report(warehouse)
    if _warehouse_stamp(warehouse) != before:
        st.session_state.pop("_spec_cache", None)
        _cached_summary.clear()
    return failures


def _maybe_refresh(force: bool = False) -> str:
    """Pull this season when a new final is in, or when the user asks.

    The button refreshes ESPN (schedule, rosters, injuries), NBA.com rates, then
    every game still to come. Opening the board also refreshes the ESPN injury
    report on its own so Who plays is not stuck on a stale file.
    """
    warehouse = Warehouse()
    season = season_for(melbourne_today())
    st.session_state["slate_rerun"] = False
    failures: list[str] = []
    try:
        failures.extend(_soft_refresh_espn(warehouse, force=force))
    except Exception as exc:
        st.caption(f"Injury report left as last saved ({exc.__class__.__name__}).")
    checked = float(st.session_state.get("rates_checked") or 0.0)
    feeds_present = warehouse.exists(f"raw_player_Base_{season}") and warehouse.exists(
        f"raw_play_offensive_T_Spotup_{season}"
    )
    if not force and time.time() - checked < 600 and feeds_present:
        if failures:
            st.caption("Some feeds were skipped: " + "; ".join(failures[:4]))
        return _warehouse_stamp(warehouse)
    st.session_state["rates_checked"] = time.time()
    try:
        from nba_sim.ingest.espn import recent_final_dates
        from nba_sim.ingest.nba_stats import refresh_slate_season, slate_refresh_due

        due = True if force else slate_refresh_due(warehouse, season, recent_final_dates())
        if due:
            with st.spinner("Updating this season from NBA.com…"):
                failures.extend(refresh_slate_season(warehouse, season))
            _cached_summary.clear()
            st.session_state.pop("_spec_cache", None)
            st.session_state["slate_rerun"] = force
    except Exception as exc:
        st.session_state["slate_rerun"] = False
        st.caption(f"The last saved tables are still in use ({exc.__class__.__name__}).")
    if failures:
        st.caption("Some feeds were skipped: " + "; ".join(failures[:4]))
    return _warehouse_stamp(warehouse)


def _rates_status() -> None:
    path = Warehouse().root / "slate_stamp.txt"
    if not path.exists():
        st.sidebar.caption("No update stored yet.")
        return
    st.sidebar.caption("Last update · " + path.read_text(encoding="utf-8").strip())


def _theme() -> None:
    st.markdown(
        """
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style>
        :root {
          --ink: #0e1412;
          --muted: #5a6460;
          --line: #c5d0c9;
          --paper: #eef2ef;
          --panel: #f7faf8;
          --away: #c1121f;
          --home: #1b4dff;
          --court: #0d7a4f;
        }
        html, body, [data-testid="stAppViewContainer"] {
          font-family: "IBM Plex Sans", sans-serif;
          color: var(--ink);
        }
        [data-testid="stAppViewContainer"] {
          background:
            radial-gradient(1000px 480px at 6% -12%, rgba(13,122,79,0.28), transparent 58%),
            radial-gradient(860px 420px at 94% -4%, rgba(27,77,255,0.18), transparent 52%),
            radial-gradient(700px 360px at 50% 8%, rgba(193,18,31,0.08), transparent 55%),
            linear-gradient(180deg, #dce8e1 0%, var(--paper) 38%, #e3ebe6 100%);
        }
        header[data-testid="stHeader"] { display: none; }
        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #10201a 0%, #0b1511 100%);
        }
        section[data-testid="stSidebar"] * { color: #e8f2ec !important; }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.4rem; }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] [data-testid="stElementContainer"]:has(button) {
          width: 100%;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] .stButton,
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] .stTooltipHoverTarget {
          width: 100%;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] button[kind] {
          width: 100%;
          justify-content: flex-start;
          text-align: left;
          border-radius: 10px;
          padding: 0.45rem 0.8rem;
          font-weight: 650;
          border: 1px solid rgba(255,255,255,0.12);
          background: rgba(255,255,255,0.04);
        }
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-primary"] {
          background: var(--court);
          border-color: #2a8f63;
          color: white !important;
        }
        section[data-testid="stSidebar"] button[data-testid="stBaseButton-secondary"] {
          background: transparent;
          border: 1px solid rgba(255,255,255,0.16);
          color: #e8f2ec !important;
        }
        section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { opacity: 0.72; }
        section.stMain .block-container {
          max-width: 1240px;
          padding: 1rem 1.25rem 2.4rem;
        }
        section.stMain [data-testid="stVerticalBlock"] { gap: 0.45rem; }
        section.stMain h1 {
          font-family: "Barlow Condensed", sans-serif;
          font-size: 2.45rem;
          font-weight: 800;
          letter-spacing: 0.02em;
          line-height: 1.05;
          margin: 0.15rem 0 0.2rem;
          color: var(--ink);
        }
        section.stMain h2 {
          font-family: "Barlow Condensed", sans-serif;
          font-size: 1.45rem;
          font-weight: 700;
          letter-spacing: 0.03em;
          margin: 1rem 0 0.2rem;
        }
        section.stMain h3 { font-size: 1rem; margin: 0; }
        section.stMain [data-testid="stCaptionContainer"] { margin-bottom: 0.15rem; color: var(--muted); }
        section.stMain button[kind="primary"],
        section.stMain button[kind="primaryFormSubmit"],
        section.stMain button[data-testid="stBaseButton-primary"],
        section.stMain button[data-testid="stBaseButton-primaryFormSubmit"],
        section.stMain button[data-testid="baseButton-primary"] {
          background: var(--court) !important;
          border: 1px solid #0a5f3d !important;
          color: #fff !important;
          font-weight: 700 !important;
          border-radius: 10px !important;
          min-height: 2.6rem;
          padding: 0.55rem 1.15rem !important;
        }
        .brand-bar {
          display: flex; align-items: baseline; gap: 0.75rem; margin-bottom: 0.35rem;
          animation: rise 420ms ease-out both;
        }
        .brand-bar .mark {
          font-family: "Barlow Condensed", sans-serif;
          font-weight: 800;
          font-size: 1.1rem;
          letter-spacing: 0.16em;
          text-transform: uppercase;
          color: var(--court);
        }
        .brand-bar .sub {
          color: var(--muted);
          font-size: 0.88rem;
        }
        .match-strip {
          display: flex; align-items: center; gap: 0.65rem;
          margin: 0.15rem 0 0.55rem;
          animation: rise 460ms ease-out both;
        }
        .match-strip .side {
          font-family: "Barlow Condensed", sans-serif;
          font-weight: 800;
          font-size: 1.05rem;
          letter-spacing: 0.08em;
          padding: 0.2rem 0.55rem;
          border-radius: 4px;
          color: #fff;
        }
        .match-strip .away { background: var(--away); }
        .match-strip .home { background: var(--home); }
        .match-strip .at { color: var(--muted); font-weight: 700; font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; }
        .board { max-width: 1240px; margin: 0 auto; animation: rise 520ms ease-out both; }
        .scoreboard { display: grid; grid-template-columns: minmax(0, 1fr) 56px minmax(0, 1fr); gap: 10px; align-items: stretch; }
        .team-card {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 4px;
          padding: 12px 14px 10px;
          min-width: 0;
        }
        .team-card.away { border-left: 5px solid var(--away); }
        .team-card.home { border-right: 5px solid var(--home); text-align: right; }
        .role { font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted); font-weight: 700; }
        .abbr {
          font-family: "Barlow Condensed", sans-serif;
          font-size: 1.85rem;
          font-weight: 800;
          color: var(--ink);
          margin-top: 2px;
          letter-spacing: 0.04em;
        }
        .city { color: var(--muted); font-size: 0.92rem; }
        .spread { font-size: 1.15rem; font-weight: 700; margin-top: 8px; letter-spacing: 0.01em; }
        .team-card.away .spread { color: var(--away); }
        .team-card.home .spread { color: var(--home); }
        .spread span { color: #7a8680; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; margin-left: 6px; }
        .points {
          font-family: "Barlow Condensed", sans-serif;
          font-size: 3.25rem;
          font-weight: 800;
          line-height: 1;
          color: var(--ink);
          margin-top: 4px;
          animation: tick 560ms cubic-bezier(.2,.8,.2,1) both;
        }
        .band { color: var(--muted); margin-top: 6px; font-size: 0.9rem; }
        .fact .sub { color: var(--muted); font-size: 0.78rem; font-weight: 650; margin-top: 2px; }
        .versus {
          display: flex; align-items: center; justify-content: center;
          font-family: "Barlow Condensed", sans-serif;
          color: var(--muted);
          font-weight: 800;
          letter-spacing: 0.12em;
          font-size: 1.1rem;
        }
        .winbar {
          display: flex; height: 34px; border-radius: 4px; overflow: hidden;
          margin-top: 12px; background: #d5ddd8;
        }
        .winbar span { display: flex; align-items: center; color: white; font-weight: 700; font-size: 0.84rem; white-space: nowrap; overflow: hidden; }
        .winbar .away { background: var(--away); justify-content: flex-start; padding-left: 12px; }
        .winbar .home { background: var(--home); justify-content: flex-end; padding-right: 12px; }
        .facts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-top: 10px; }
        .fact {
          background: transparent;
          border-top: 2px solid var(--line);
          border-radius: 0;
          padding: 8px 2px 4px;
          min-width: 0;
        }
        .fact .k { color: var(--muted); font-size: 0.7rem; letter-spacing: 0.05em; text-transform: uppercase; font-weight: 700; }
        .fact .v { color: var(--ink); font-size: 1.05rem; font-weight: 750; margin-top: 2px; overflow-wrap: anywhere; }
        .chips { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 2px; }
        .chip {
          background: rgba(13,122,79,0.1);
          border: 1px solid rgba(13,122,79,0.28);
          color: #0a5638;
          border-radius: 4px;
          padding: 4px 10px;
          font-size: 0.82rem;
          font-weight: 600;
        }
        .when { color: #2c3531; margin: 2px 0 6px; font-size: 0.9rem; line-height: 1.35; }
        .injury-note { color: var(--muted); font-size: 0.85rem; }
        .callout {
          background: rgba(13,122,79,0.08);
          border-left: 4px solid var(--court);
          border-radius: 0 8px 8px 0;
          padding: 12px 14px;
          margin: 8px 0 14px;
        }
        .callout strong { color: #0a5638; }
        .defs { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 2px 0 10px; }
        .defs div { border-top: 2px solid var(--line); padding: 8px 2px 4px; min-width: 0; }
        .defs strong { display: block; color: var(--ink); font-size: 0.72rem; letter-spacing: 0.04em; text-transform: uppercase; }
        .defs span { display: block; color: #5a6460; font-size: 0.8rem; line-height: 1.35; margin-top: 3px; }
        .roster-head {
          font-family: "Barlow Condensed", sans-serif;
          font-size: 1.3rem;
          font-weight: 700;
          letter-spacing: 0.04em;
          margin: 0.35rem 0 0.25rem;
          color: var(--ink);
        }
        .empty-run {
          margin-top: 0.4rem;
          padding: 0.85rem 1rem;
          border: 1px dashed var(--line);
          border-radius: 8px;
          background: rgba(247,250,248,0.7);
          color: var(--muted);
          animation: rise 360ms ease-out both;
        }
        @keyframes rise {
          from { opacity: 0; transform: translateY(8px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes tick {
          from { opacity: 0; transform: translateY(10px) scale(0.96); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
        div.stElementContainer:has(.dist-anchor) + div.stElementContainer [data-testid="stCaptionContainer"] { min-height: 2.7rem; }
        @media (max-width: 1100px) {
          .facts, .defs { grid-template-columns: 1fr 1fr; }
        }
        @media (max-width: 800px) {
          .scoreboard, .facts { grid-template-columns: 1fr; }
          .defs { grid-template-columns: 1fr 1fr; }
          .team-card.home { text-align: left; border-right: none; border-left: 5px solid var(--home); }
          section.stMain h1 { font-size: 1.85rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _colors(summary: dict) -> alt.Scale:
    return alt.Scale(domain=[summary["away"], summary["home"]], range=[AWAY_COLOR, HOME_COLOR])


def _scoreboard(summary: dict) -> None:
    team = summary["team"]
    home_win = team["home_win_prob"]
    away_win = max(0.0, 1.0 - home_win)
    home_label = f"{summary['home']} {home_win:.0%}" if home_win >= 0.12 else ""
    away_label = f"{summary['away']} {away_win:.0%}" if away_win >= 0.12 else ""
    view = spread_view(summary)
    if view["source"] == "line":
        spread_value = view["line_text"]
        spread_sub = f'<div class="sub">sim {view["sim_text"]}</div>'
        spread_note = ""
    else:
        spread_value = view["sim_text"]
        spread_sub = ""
        spread_note = "<span>sim</span>"
    glass = summary["rebounds"]
    home_orc = glass.get("home_chance", _share(glass["home_oreb"], glass["away_dreb"]))
    away_orc = glass.get("away_chance", _share(glass["away_oreb"], glass["home_dreb"]))
    chips = "".join(f'<span class="chip">{flag}</span>' for flag in summary["flags"])
    st.markdown(
        f"""
        <div class="board">
        <div class="scoreboard">
          <div class="team-card away">
            <div class="role">Away</div>
            <div class="abbr">{summary['away']}</div>
            <div class="city">{summary['away_name']}</div>
            <div class="spread">{view['away_text']}{spread_note}</div>
            <div class="points">{team['away_mean']:.1f}</div>
            <div class="band">typical game {team['away_p10']:.0f}–{team['away_p90']:.0f}</div>
          </div>
          <div class="versus">AT</div>
          <div class="team-card home">
            <div class="role">Home</div>
            <div class="abbr">{summary['home']}</div>
            <div class="city">{summary['home_name']}</div>
            <div class="spread">{view['home_text']}{spread_note}</div>
            <div class="points">{team['home_mean']:.1f}</div>
            <div class="band">typical game {team['home_p10']:.0f}–{team['home_p90']:.0f}</div>
          </div>
        </div>
        <div class="winbar" title="{summary['home']} wins {home_win:.1%} of simulations">
          <span class="away" style="width:{away_win:.1%}">{away_label}</span>
          <span class="home" style="width:{home_win:.1%}">{home_label}</span>
        </div>
        <div class="facts">
          <div class="fact"><div class="k">Spread</div><div class="v">{spread_value}</div>{spread_sub}</div>
          <div class="fact"><div class="k">Total</div><div class="v">{team['total_mean']:.1f}</div></div>
          <div class="fact"><div class="k">Total range</div><div class="v">{team['total_p10']:.0f}–{team['total_p90']:.0f}</div></div>
          <div class="fact"><div class="k">Second-chance glass</div><div class="v">{summary['home']} {home_orc:.0%} · {summary['away']} {away_orc:.0%}</div></div>
        </div>
        <div class="chips">{chips}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The offensive rebound chance is the five on the floor, pulled toward that team's own rate and toward the rate the defense gives up. "
        "This season counts a little more for usage, form, and team strength. A short year still leans on last season."
    )
    when = _when(summary)
    if when:
        st.markdown(f'<div class="when">{when}</div>', unsafe_allow_html=True)
    _injuries(summary)


def _when(summary: dict) -> str:
    parts = []
    venue = summary.get("venue") or ""
    city = summary.get("city") or ""
    if venue and city:
        parts.append(f"{venue}, {city}")
    elif venue:
        parts.append(venue)
    if summary.get("tip_local"):
        parts.append(f"Local {summary['tip_local']}")
    if summary.get("tip_melbourne"):
        parts.append(f"Melbourne {summary['tip_melbourne']}")
    if summary.get("game_status"):
        parts.append(str(summary["game_status"]))
    return " · ".join(parts)


def _injuries(summary: dict) -> None:
    rows = summary.get("injuries") or []
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame["on roster"] = frame["matched"].map(lambda value: "Yes" if bool(value) else "No")
    shown = frame.rename(
        columns={
            "team": "Team",
            "player": "Player",
            "label": "Status",
            "body_part": "Body part",
            "comment": "Report",
            "return_date": "Target return",
            "on roster": "On roster",
        }
    )
    columns = ["Team", "Player", "Status", "Body part", "Target return", "Report", "On roster"]
    st.subheader("Injuries")
    st.dataframe(
        shown[columns],
        hide_index=True,
        width="stretch",
        height="content",
        column_config={
            "Team": st.column_config.TextColumn(width="small"),
            "Status": st.column_config.TextColumn(width="small"),
            "Body part": st.column_config.TextColumn(width="small"),
            "Target return": st.column_config.TextColumn(width="small"),
            "Report": st.column_config.TextColumn(width="large"),
            "On roster": st.column_config.TextColumn(width="small"),
        },
    )


def _share(offense: float, defense: float) -> float:
    total = offense + defense
    if total <= 0:
        return 0.0
    return offense / total


_DIST_HEIGHT = 230


def _distributions(summary: dict) -> None:
    team = summary["team"]
    colors = _colors(summary)
    scores = pd.DataFrame(team.get("score_counts") or [])
    spreads = pd.DataFrame(team.get("spread_counts") or [])
    totals = pd.DataFrame(team.get("total_counts") or [])
    view = spread_view(summary)
    posted_total = _total_line(summary["prices"])
    st.caption(f"Orange is {summary['away']}. Blue is {summary['home']}. Height is how many simulations landed there.")
    st.markdown('<div class="dist-anchor"></div>', unsafe_allow_html=True)
    score_col, spread_col, total_col = st.columns(3, gap="medium")
    with score_col:
        st.subheader("Score")
        st.caption("Each team’s simulated score.")
        if not scores.empty:
            _score_chart(summary, scores, colors)
    with spread_col:
        st.subheader("Game spread")
        spread_caption = f"{summary['home']} margin."
        if view["line_text"]:
            spread_caption += f" Amber is the posted {view['line_text']}."
        st.caption(spread_caption)
        if not spreads.empty:
            _spread_chart(spreads, team, summary, view)
    with total_col:
        st.subheader("Game total")
        total_caption = "Combined points."
        if posted_total is not None:
            total_caption += f" Amber is the posted total {posted_total:g}."
        st.caption(total_caption)
        if not totals.empty:
            _total_chart(totals, team, summary["prices"])


def _score_chart(summary: dict, scores: pd.DataFrame, colors: alt.Scale) -> None:
    team = summary["team"]
    chart = (
        alt.Chart(scores)
        .mark_area(opacity=0.35, interpolate="monotone", line={"strokeWidth": 2})
        .encode(
            x=alt.X("value:Q", title="Points in one simulation", axis=alt.Axis(format="d", tickMinStep=20)),
            y=alt.Y("sims:Q", title="Sims", stack=None),
            color=alt.Color("team:N", scale=colors, legend=None),
            tooltip=["team", alt.Tooltip("value:Q", title="Points", format="d"), "sims"],
        )
        .properties(height=_DIST_HEIGHT)
    )
    means = pd.DataFrame(
        [
            {"team": summary["away"], "value": team["away_mean"]},
            {"team": summary["home"], "value": team["home_mean"]},
        ]
    )
    rules = (
        alt.Chart(means)
        .mark_rule(strokeDash=[4, 3], strokeWidth=2)
        .encode(x="value:Q", color=alt.Color("team:N", scale=colors, legend=None))
    )
    st.altair_chart(chart + rules, width="stretch")


def _spread_chart(spreads: pd.DataFrame, team: dict, summary: dict, view: dict) -> None:
    home = summary["home"]
    away = summary["away"]
    base = spreads.copy()
    if not (base["value"] == 0).any():
        base = pd.concat([base, pd.DataFrame([{"value": 0, "sims": 0}])], ignore_index=True)
    wings = []
    away_side = base[base["value"] <= 0].copy()
    home_side = base[base["value"] >= 0].copy()
    if not away_side.empty:
        away_side["side"] = away
        wings.append(away_side)
    if not home_side.empty:
        home_side["side"] = home
        wings.append(home_side)
    frame = pd.concat(wings, ignore_index=True)
    colors = alt.Scale(domain=[away, home], range=[AWAY_COLOR, HOME_COLOR])
    chart = (
        alt.Chart(frame)
        .mark_area(opacity=0.35, interpolate="monotone", line={"strokeWidth": 2})
        .encode(
            x=alt.X("value:Q", title=f"{home} margin", axis=alt.Axis(format="d", tickMinStep=20)),
            y=alt.Y("sims:Q", title="Sims", stack=None),
            color=alt.Color("side:N", scale=colors, legend=None),
            tooltip=[
                "side",
                alt.Tooltip("value:Q", title="Margin", format="d"),
                "sims",
            ],
        )
        .properties(height=_DIST_HEIGHT)
    )
    layers = [chart, _mean_rule(team["spread_mean"], "#0f172a")]
    threshold = view.get("threshold")
    if threshold is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"line": [threshold]}))
            .mark_rule(color="#b45309", strokeWidth=2)
            .encode(x="line:Q")
        )
    st.altair_chart(alt.layer(*layers), width="stretch")


def _total_chart(totals: pd.DataFrame, team: dict, prices: list[dict]) -> None:
    total_chart = (
        alt.Chart(totals)
        .mark_area(color="#0f766e", opacity=0.35, interpolate="monotone", line={"strokeWidth": 2})
        .encode(
            x=alt.X("value:Q", title="Combined points", axis=alt.Axis(format="d", tickMinStep=20)),
            y=alt.Y("sims:Q", title="Sims"),
            tooltip=[alt.Tooltip("value:Q", title="Points", format="d"), "sims"],
        )
        .properties(height=_DIST_HEIGHT)
    )
    layers = [total_chart, _mean_rule(team["total_mean"], "#0f766e")]
    line = _total_line(prices)
    if line is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"line": [line]}))
            .mark_rule(color="#b45309", strokeWidth=2)
            .encode(x="line:Q")
        )
    st.altair_chart(alt.layer(*layers), width="stretch")


def _total_line(prices: list[dict]) -> float | None:
    for price in prices:
        if price["market"] == "total" and price.get("line") is not None and not pd.isna(price["line"]):
            return float(price["line"])
    return None


def _matchups(summary: dict) -> None:
    rows = []
    for row in summary["matchups"]:
        rows.append(
            {
                "team": row["team"],
                "play": PLAY_LABELS.get(row["play_type"], row["play_type"]),
                "expected": round(float(row["expected"]), 3),
                "offense": round(float(row["off_ppp"]), 3),
                "defense": round(float(row["def_ppp"]), 3),
                "share": float(row.get("off_share") or 0.0),
                "source": row.get("source") or "",
            }
        )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    home = frame[frame["team"] == summary["home"]].set_index("play")
    away = frame[frame["team"] == summary["away"]].set_index("play")
    gaps = []
    for play in home.index.intersection(away.index):
        gap = float(home.loc[play, "expected"] - away.loc[play, "expected"])
        gaps.append(
            {
                "play": play,
                "gap": round(gap, 3),
                "leader": summary["home"] if gap >= 0 else summary["away"],
                "home_expected": float(home.loc[play, "expected"]),
                "away_expected": float(away.loc[play, "expected"]),
                "home_share": float(home.loc[play, "share"]),
                "away_share": float(away.loc[play, "share"]),
            }
        )
    if not gaps:
        return
    compared = pd.DataFrame(gaps)
    st.subheader("Play types")
    source = str(frame["source"].iloc[0]) if "source" in frame.columns and not frame.empty else ""
    if source == "position prior":
        st.caption(
            f"Each dot is points per possession for the players on the floor, and its size is how often they run it. "
            f"Usage is that player's claim on the trip. His shot profile sets the look. "
            f"Blue is {summary['home']}. Orange is {summary['away']}."
        )
    else:
        st.caption(
            f"Size is how often this rotation runs it, weighted by each player's usage and minutes. "
            f"Usage is his claim on the trip while he is on the floor, and the action is his own mix. "
            f"A sample keeps that mix and his points per possession. "
            f"Defense is the profile of whoever is on the floor, and the defender matched to the action moves the trip. "
            f"Without a sample, his spot in this coach's offense supplies the role, and his shot profile sets how much of it is a spot-up, a roll, or a post. "
            f"Blue is {summary['home']}. Orange is {summary['away']}."
        )
    values = compared[["home_expected", "away_expected"]].to_numpy(dtype=float)
    low = float(values.min())
    high = float(values.max())
    pad = max(0.05, (high - low) * 0.6)
    scale = alt.Scale(domain=[low - pad, high + pad], zero=False, nice=False)
    base = alt.Chart(compared).encode(
        y=alt.Y("play:N", sort=alt.EncodingSortField(field="home_expected", order="descending"), title=None)
    )
    link = base.mark_rule(color="#cbd5e1", strokeWidth=2).encode(
        x=alt.X("away_expected:Q", title="Expected points per possession", scale=scale),
        x2="home_expected:Q",
    )
    away = base.mark_point(filled=True, color=AWAY_COLOR).encode(
        x=alt.X("away_expected:Q", scale=scale),
        size=alt.Size("away_share:Q", scale=alt.Scale(domain=[0, 0.35], range=[40, 220]), legend=None),
        tooltip=[
            "play",
            alt.Tooltip("away_expected:Q", title=summary["away"], format=".3f"),
            alt.Tooltip("away_share:Q", title=f"{summary['away']} share", format=".0%"),
        ],
    )
    home = base.mark_point(filled=True, color=HOME_COLOR).encode(
        x=alt.X("home_expected:Q", scale=scale),
        size=alt.Size("home_share:Q", scale=alt.Scale(domain=[0, 0.35], range=[40, 220]), legend=None),
        tooltip=[
            "play",
            alt.Tooltip("home_expected:Q", title=summary["home"], format=".3f"),
            alt.Tooltip("home_share:Q", title=f"{summary['home']} share", format=".0%"),
        ],
    )
    st.altair_chart((link + away + home).properties(height=380), width="stretch")
    table = compared.rename(
        columns={
            "play": "Play",
            "home_share": summary["home"],
            "away_share": summary["away"],
            "home_expected": f"{summary['home']} PPP",
            "away_expected": f"{summary['away']} PPP",
        }
    )
    shown = table[["Play", summary["away"], f"{summary['away']} PPP", summary["home"], f"{summary['home']} PPP"]].copy()
    for column in (summary["away"], summary["home"]):
        shown[column] = (shown[column] * 100).round(0).astype(int).astype(str) + "%"
    for column in (f"{summary['away']} PPP", f"{summary['home']} PPP"):
        shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    st.dataframe(shown, hide_index=True, width="stretch", height="content")


_STAT_LINES = {
    "Points": ("pts", (10, 20, 30)),
    "Rebounds": ("reb", (4, 8, 12)),
    "Assists": ("ast", (4, 8, 12)),
    "Threes": ("tpm", (1, 2, 3)),
    "PRA": ("pra", (15, 25, 35)),
}


def _mean_rule(value: float, color: str) -> alt.Chart:
    return (
        alt.Chart(pd.DataFrame({"value": [value]}))
        .mark_rule(color=color, strokeDash=[4, 3], strokeWidth=2)
        .encode(x="value:Q")
    )


_MINUTE_HELP = {
    "Pos": "Listed position.",
    "Usage": "Share of the possessions he uses while he is on the floor. That share, times his own play mix, is his claim on the trip.",
    "Role": "The five largest minute totals on this roster. The men after them are the bench.",
    "Source": "His minutes, from his own games with this coach. A short run shrinks toward his earlier seasons, then toward players his age with a similar career load.",
    "Base": "Minutes before a confirmed absence.",
    "Injury": "The pre-tip tag. Out is off in every path. Doubtful, questionable, and probable are the chance he plays.",
    "After injury": "His minutes when he is available. A tag changes how often he plays, not the size of the night.",
    "Rotation": "His minutes from the games before this slate. The paths then record the time he actually spends on the floor.",
    "Sim": "Floor time across the paths. The court changes on his coach's whistle, a foul sit, or a decided-game hook. Five men for 48 minutes is 240. Overtime adds the extra stint.",
    "Range": "The middle 80% of simulations, from the 10th percentile to the 90th.",
    "Fives": "Offensive and defensive rating of the fives he has played in, weighted by the minutes those fives shared.",
    "Why": "Out, inactive, or past the number of players this coach uses.",
}
_MINUTE_GRID = ("Source", "Base", "Injury", "After injury", "Role", "Usage", "Rotation", "Sim", "Fives", "Range")


def _minute_defs() -> None:
    cards = "".join(
        f"<div><strong>{name}</strong><span>{_MINUTE_HELP[name]}</span></div>"
        for name in _MINUTE_GRID
    )
    st.markdown(f"<div class='defs'>{cards}</div>", unsafe_allow_html=True)


def _minute_config(columns: list[str]) -> dict:
    config = {}
    for column in columns:
        help_text = _MINUTE_HELP.get(column)
        if column in {"Base", "After injury", "Rotation", "Sim"}:
            config[column] = st.column_config.NumberColumn(help=help_text, format="%.1f", width="small")
        elif column == "Usage":
            config[column] = st.column_config.TextColumn(help=help_text, width="small")
        elif column == "Player":
            config[column] = st.column_config.TextColumn(width="medium")
        elif column == "Injury":
            config[column] = st.column_config.TextColumn(help=help_text, width="medium")
        elif column == "Why":
            config[column] = st.column_config.TextColumn(help=help_text, width="large")
        elif column == "Source":
            config[column] = st.column_config.TextColumn(help=help_text, width="small")
        else:
            config[column] = st.column_config.TextColumn(help=help_text, width="small")
    return config


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _source_cell(value: object) -> str:
    text = str(value or "").split(" · ")[0]
    if text.startswith("Last ") and text.endswith(" games"):
        return text.replace(" games", "")
    if text == "Season average":
        return "Season"
    if text.startswith("Depth chart, slot "):
        return "Depth " + text.removeprefix("Depth chart, slot ")
    if text == "Depth chart":
        return "Depth"
    return text


def _injury_cell(value: object) -> str:
    text = _cell(value)
    if " × " not in text:
        return text
    label, factor = text.split(" × ", 1)
    try:
        share = float(factor)
    except ValueError:
        return text
    return f"{label} · {share:.0%}"


def _minute_table(frame: pd.DataFrame, role_of) -> pd.DataFrame:
    records = []
    for index, row in frame.iterrows():
        records.append(
            {
                "Player": row["player"],
                "Pos": _cell(row["position"]),
                "Usage": "" if "usg" not in row or pd.isna(row["usg"]) else f"{float(row['usg']):.0%}",
                "Role": role_of(index, row),
                "Source": _source_cell(row["how"]),
                "Base": float(row["base"]),
                "Injury": _injury_cell(row["injury"]),
                "After injury": float(row["prior"]),
                "Rotation": float(row["scaled"]),
                "Sim": float(row["sim"]),
                "Range": "" if float(row["sim"]) <= 0 else f"{float(row['low']):.0f}–{float(row['high']):.0f}",
                "Fives": _cell(row.get("fives")),
            }
        )
    return pd.DataFrame(records)


def _rotation_roles(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values(["scaled", "player"], ascending=[False, True])
    roles = {}
    for place, index in enumerate(ordered.index):
        minutes = float(ordered.loc[index, "scaled"])
        if place < 5:
            roles[index] = "Starter"
        elif minutes >= 6:
            roles[index] = "Bench"
        else:
            roles[index] = "Reserve"
    return roles


def _sitting_reason(_index, row) -> str:
    injury = str(row.get("injury") or "").strip()
    if injury:
        return injury.split(" × ")[0]
    size = int(row.get("rotation_size") or 0)
    if size:
        return f"Past the {size}-man rotation"
    return "Past the rotation"


def _staff_label(side: pd.DataFrame) -> str:
    coach = str(side["coach"].iloc[0] or "").strip() if "coach" in side.columns else ""
    size = int(side["rotation_size"].iloc[0] or 0) if "rotation_size" in side.columns else 0
    from_chart = "how" in side.columns and all("Depth chart" in str(value) for value in side["how"])
    if coach and from_chart and size:
        return f"{coach}, {size}-man rotation"
    if coach:
        return coach
    return ""


def _minutes_caption(frame: pd.DataFrame) -> str:
    bits = []
    for team in frame["team"].drop_duplicates():
        side = frame[frame["team"] == team]
        staff = _staff_label(side)
        if staff:
            bits.append(f"{team} is {staff}")
    lead = (". ".join(bits) + ". ") if bits else ""
    return (
        lead
        + "Every trip is priced from the ten players on the floor at that second. "
        "His minutes are his own games with this coach. A short run shrinks toward his earlier seasons, then toward players his age with a similar career load. "
        "The court opens with the five this coach has started when these players are available. "
        "It changes when one of his whistles fires, when a foul sit brings in the man he has sent in at that position, or when a decided game uses the exit he has in those games. "
        "Usage and the play mix are the shares of this five, or of the nearest five. "
        "A stored five keeps its own rating. A five the file has not seen moves by the on/off of the men now standing there. "
        "Five men for 48 minutes is 240. Overtime adds the extra stint. "
        "A report filed before tip, for this team, is the chance he plays. Out is off in every path. "
        "Two fouls in the first quarter, three before half, or four in the third sit a player while enough time is left for the sub. "
        "A starter plays through one extra foul. A strict staff sits him on the normal line, and sits him sooner. "
        "A player who fouls more reaches that line sooner. With no season foul rate, that is his position: centers foul more than guards. "
        "Five fouls in the fourth only sit him when the game is not close and more than six minutes are left. "
        "A sixth foul brings in the teammate this coach has sent in for him."
    )


def _minutes(summary: dict) -> None:
    rows = summary.get("minutes") or []
    if not rows:
        return
    frame = pd.DataFrame(rows)
    st.subheader("Minutes")
    st.caption(_minutes_caption(frame))
    _minute_defs()
    names = {
        summary["away"]: summary.get("away_name") or "",
        summary["home"]: summary.get("home_name") or "",
    }
    for team in (summary["away"], summary["home"]):
        side = frame[frame["team"] == team]
        if side.empty:
            continue
        playing = side[(side["scaled"] > 0) | (side["sim"] > 0)].sort_values(
            ["scaled", "player"], ascending=[False, True]
        )
        sitting = side.drop(index=playing.index).sort_values(["prior", "player"], ascending=[False, True])
        staff = _staff_label(side)
        name = names.get(team) or ""
        label = f"**{team}** · {name}" if name else f"**{team}**"
        if staff:
            label = f"{label} · {staff}"
        st.markdown(label)
        if not playing.empty:
            roles = _rotation_roles(playing)
            shown = _minute_table(playing, lambda index, _row, roles=roles: roles[index])
            st.dataframe(
                shown,
                hide_index=True,
                width="stretch",
                height="content",
                column_config=_minute_config(list(shown.columns)),
            )
            st.caption(
                f"His minutes {float(playing['prior'].sum()):.0f}. "
                f"Floor time {float(playing['sim'].sum()):.0f}."
            )
        if not sitting.empty:
            bench = _minute_table(sitting, _sitting_reason).rename(columns={"Role": "Why"})
            bench = bench[["Player", "Pos", "Why", "Source", "Base", "Injury", "After injury"]]
            with st.expander(f"End of the roster · {len(sitting)}"):
                st.dataframe(
                    bench,
                    hide_index=True,
                    width="stretch",
                    height="content",
                    column_config=_minute_config(list(bench.columns)),
                )


def _players(summary: dict) -> None:
    bins = pd.DataFrame(summary.get("player_bins") or [])
    if bins.empty:
        return
    st.subheader("Players")
    st.caption(
        "Each panel is one player’s own simulations. Bar height is how many sims landed on that number. "
        "The whole rotation is included. The minutes table above shows why each player is at that number."
    )
    choice = st.radio("Player stat", list(_STAT_LINES), horizontal=True, label_visibility="collapsed")
    stat, lines = _STAT_LINES[choice]
    frame = bins[bins["stat"] == stat].copy()
    if frame.empty:
        return
    order = (
        frame.drop_duplicates("player")
        .sort_values(["mean", "player"], ascending=[False, True])
    )
    labels = [f"{row.team} · {row.player}" for row in order.itertuples(index=False)]
    picked = st.selectbox("Player", labels)
    team_code, name = picked.split(" · ", 1)
    one = frame[(frame["player"] == name) & (frame["team"] == team_code)]
    if one.empty:
        return
    info = one.iloc[0]
    color = HOME_COLOR if team_code == summary["home"] else AWAY_COLOR
    prop = main_prop(st.session_state.get("props"), str(info.get("espn_id") or ""), stat)
    espn_id = str(info.get("espn_id") or "")
    if espn_id and st.button("Open profile", key="open-sim-player"):
        open_player(espn_id, team_code, str(info["player"]))
    _player_focus(one, info, choice, lines, color, summary["n_sims"], prop)
    away_col, home_col = st.columns(2)
    with away_col:
        st.markdown(f"**{summary['away']}**")
        _player_facets(frame[frame["team"] == summary["away"]], AWAY_COLOR, choice)
    with home_col:
        st.markdown(f"**{summary['home']}**")
        _player_facets(frame[frame["team"] == summary["home"]], HOME_COLOR, choice)


def _player_focus(
    one: pd.DataFrame,
    info: pd.Series,
    stat_name: str,
    lines: tuple[int, ...],
    color: str,
    n_sims: int,
    prop: dict | None,
) -> None:
    total = int(one["sims"].sum()) or 1
    bits = [f"{info['player']} averages {info['mean']:.1f} {stat_name.lower()} in {info['minutes']:.0f} minutes."]
    usg = info.get("usg")
    if usg is not None and not pd.isna(usg):
        bits.append(f"While he is on the floor he uses {float(usg):.0%} of the possessions.")
    bits.extend(floor_bits(info))
    three = info.get("three")
    if three is not None and not pd.isna(three) and stat_name == "Points":
        bits.append(f"{float(three):.0%} of his shots are threes.")
    bits.append(f"A typical sim falls between {info['p10']:.0f} and {info['p90']:.0f}.")
    for line in lines:
        share = float(one.loc[one["value"] >= line, "sims"].sum()) / total
        bits.append(f"{share:.0%} of sims reach {line}.")
    if prop is not None:
        rate = sim_over_rate(one, float(prop["line"]))
        if rate is not None:
            bits.append(f"Prop {float(prop['line']):g}: sim over {rate:.0%}.")
        fair = prop.get("fair_over")
        if fair is not None and not pd.isna(fair):
            bits.append(f"De-vigged over price {float(fair):.0%}.")
    st.caption(" ".join(bits) + f" {n_sims:,} simulations.")
    bars = (
        alt.Chart(one)
        .mark_bar(color=color, cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X("value:Q", title=stat_name, axis=alt.Axis(format="d")),
            y=alt.Y("sims:Q", title="Simulations"),
            tooltip=[
                alt.Tooltip("value:Q", title=stat_name, format="d"),
                "sims",
            ],
        )
        .properties(height=240)
    )
    layers = [bars, _mean_rule(float(info["mean"]), "#0f172a")]
    if prop is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"line": [float(prop["line"])], "label": [f"prop {float(prop['line']):g}"]}))
            .mark_rule(color="#b45309", strokeWidth=2)
            .encode(x="line:Q")
        )
    st.altair_chart(alt.layer(*layers), width="stretch")


def _player_facets(frame: pd.DataFrame, color: str, stat_name: str) -> None:
    if frame.empty:
        st.caption("No rotation players in this sim.")
        return
    ranked = frame.drop_duplicates("player").sort_values(["mean", "player"], ascending=[False, True])
    labels = {row.player: f"{row.player} · {row['mean']:.1f}" for _, row in ranked.iterrows()}
    shown = frame.copy()
    shown["label"] = shown["player"].map(labels)
    sort = [labels[name] for name in ranked["player"]]
    chart = (
        alt.Chart(shown)
        .mark_bar(color=color, size=7)
        .encode(
            x=alt.X("value:Q", title=stat_name, axis=alt.Axis(format="d", tickCount=5)),
            y=alt.Y("sims:Q", title=None, axis=alt.Axis(labels=False, ticks=False)),
            tooltip=["player", alt.Tooltip("value:Q", title=stat_name, format="d"), "sims", "minutes"],
            facet=alt.Facet(
                "label:N",
                title=None,
                sort=sort,
                header=alt.Header(labelFontSize=13, labelFontWeight=600, labelColor="#0f172a"),
            ),
        )
        .properties(height=72, width=280)
        .resolve_scale(y="independent")
    )
    st.altair_chart(chart, width="stretch")


def _prices(summary: dict) -> None:
    prices = summary["prices"]
    if not prices:
        return
    rows = []
    for price in prices:
        edge = float(price["edge"])
        rows.append(
            {
                "bet": _price_label(price, summary["home"], summary["away"]),
                "edge": edge,
                "model": float(price["model_prob"]),
                "market": float(price["market_prob"]),
                "stake": float(price["stake"]),
                "source": price["source"],
                "tone": "edge" if edge > 0 and price["stake"] > 0 else "no edge",
            }
        )
    frame = pd.DataFrame(rows).sort_values("edge", ascending=False)
    best = frame.iloc[0]
    if best["tone"] == "edge":
        st.markdown(
            (
                f'<div class="callout"><strong>{best["bet"]}</strong> · model {best["model"]:.0%}, '
                f'price {best["market"]:.0%}, edge {best["edge"]:+.0%}. Stake {best["stake"]:.0%} of bankroll. '
                f'Source: {best["source"]}.</div>'
            ),
            unsafe_allow_html=True,
        )
    st.subheader("Model versus price")
    st.caption("Green means the model’s chance clears the de-vigged price by enough to survive its own uncertainty.")
    chart = (
        alt.Chart(frame)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            y=alt.Y("bet:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X("edge:Q", title="Model probability minus price", axis=alt.Axis(format="+%")),
            color=alt.Color(
                "tone:N",
                title=None,
                scale=alt.Scale(domain=["edge", "no edge"], range=["#15803d", "#cbd5e1"]),
            ),
            tooltip=[
                "bet",
                alt.Tooltip("model:Q", format=".1%"),
                alt.Tooltip("market:Q", format=".1%"),
                alt.Tooltip("edge:Q", format="+.1%"),
                alt.Tooltip("stake:Q", format=".1%"),
                "source",
            ],
        )
        .properties(height=max(220, 36 * len(frame)))
    )
    st.altair_chart(chart, width="stretch")


def _price_label(price: dict, home: str, away: str) -> str:
    market = {
        "moneyline": "Moneyline",
        "spread": "Spread",
        "total": "Total",
        "player_points": "Points",
    }.get(price["market"], str(price["market"]).replace("_", " ").title())
    side = {"home": home, "away": away}.get(price["side"], str(price["side"]).title())
    who = price.get("player") or side
    line = price.get("line")
    line_text = "" if line is None or pd.isna(line) else f" {float(line):g}"
    odds = int(price["american"])
    odds_text = f"{odds:+d}" if odds > 0 else str(odds)
    return f"{market} · {who}{line_text} ({odds_text})"


def _factors(summary: dict) -> None:
    rows = []
    for factor in summary["factors"]:
        rows.append(
            {
                "factor": str(factor.get("name", "")).replace("_", " "),
                "sample": round(float(factor.get("sample_size", 0)), 1),
                "value": round(float(factor.get("value", 0)), 3),
                "why": factor.get("detail", ""),
            }
        )
    with st.expander("Why the rates moved"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _log_stamp() -> int:
    path = Warehouse().path("game_logs")
    return path.stat().st_mtime_ns if path.exists() else 0


@st.cache_data(show_spinner=False)
def _cached_actuals(
    logs_mtime: int,
    venue_iso: str,
    home_id: int,
    away_id: int,
    players: tuple[tuple[int, str], ...],
) -> dict | None:
    if not logs_mtime:
        return None
    return box_actuals(
        Warehouse().read("game_logs"),
        date.fromisoformat(venue_iso),
        home_id,
        away_id,
        list(players),
    )


def _played_actuals(spec: GameSpec) -> dict[str, dict[str, float]] | None:
    players = tuple(
        (int(player.player_id), str(player.espn_id))
        for player in [*spec.home_players, *spec.away_players]
        if player.player_id and player.espn_id
    )
    try:
        return _cached_actuals(_log_stamp(), spec.date, int(spec.home.team_id), int(spec.away.team_id), players)
    except (TypeError, ValueError):
        return None


def _default_date(schedule: pd.DataFrame) -> date:
    today = melbourne_today()
    if schedule.empty or "start_utc" not in schedule.columns:
        return today
    dates = sorted({day for day in (_tip_date(value) for value in schedule["start_utc"]) if day is not None})
    if today in dates:
        return today
    upcoming = [value for value in dates if value >= today]
    return upcoming[0] if upcoming else (dates[-1] if dates else today)


def _day_games(schedule: pd.DataFrame, on: date) -> list[dict]:
    day = _on_melbourne_date(schedule, on)
    rows = []
    for row in day.to_dict("records"):
        away = str(row.get("away") or "").upper()
        home = str(row.get("home") or "").upper()
        game_id = str(row.get("game_id") or "")
        if not away or not home or not game_id or away == "NAN" or home == "NAN":
            continue
        label = f"{away} at {home}"
        if "final" in str(row.get("status") or "").lower():
            label += " · Final"
        rows.append({"game_id": game_id, "label": label})
    return rows


def _choose_game(games: list[dict]) -> str:
    ids = [game["game_id"] for game in games]
    current = str(st.session_state.get("slate_game") or "")
    if current not in ids:
        current = ids[0]
        st.session_state["slate_game"] = current
    st.sidebar.caption("Games")
    for game in games:
        if st.sidebar.button(
            game["label"],
            key=f"nav-{game['game_id']}",
            type="primary" if game["game_id"] == current else "secondary",
        ):
            # Streamlit already reruns on the click. Keep this run on the new
            # game so the model is not built twice.
            st.session_state["slate_game"] = game["game_id"]
            current = game["game_id"]
    return current


def _sheet_minutes(player) -> float:
    """Expected minutes on this board. Depth-chart base is only a prior."""
    if not player.available or float(player.injury_factor or 0.0) <= 0.0:
        return 0.0
    profile = getattr(player, "court", None)
    if profile is not None and float(profile.minutes or 0.0) > 0.0:
        rate = float(getattr(profile, "play_rate", 1.0) or 1.0)
        return float(profile.minutes) * max(0.0, min(1.0, rate))
    return float(player.minutes_mean or 0.0)


def _player_note(player) -> str:
    label = str(player.injury_label or "").strip()
    if not player.available or float(player.injury_factor or 0.0) <= 0.0:
        text = label or "Out"
    else:
        text = label
    if len(text) > 42:
        return text[:41] + "…"
    return text


def _roster_line(player) -> str:
    model_minutes = _sheet_minutes(player)
    note = _player_note(player)
    bits = [player.name]
    if player.position:
        bits.append(str(player.position))
    if bool(getattr(player, "projected_starter", False)):
        bits.append("Starter")
    if model_minutes > 0:
        bits.append(f"{model_minutes:.0f} min")
    if note:
        bits.append(note)
    return " · ".join(bits)


def _roster_editor(
    players: list,
    current: dict[int, str],
    current_minutes: dict[int, float],
    game_id: str,
    side: str,
) -> tuple[dict[int, str], dict[int, float]]:
    ordered = sorted(
        players,
        key=lambda item: (
            0 if bool(getattr(item, "projected_starter", False)) else 1,
            -_sheet_minutes(item),
            item.name,
        ),
    )
    rows = []
    model_by_id: dict[int, float] = {}
    for player in ordered:
        pid = int(player.player_id)
        model_minutes = _sheet_minutes(player)
        model_by_id[pid] = model_minutes
        rows.append(
            {
                "id": pid,
                "Player": _roster_line(player),
                "Role": {"on": "On", "off": "Off"}.get(current.get(pid, ""), "Model"),
                "Min": int(round(current_minutes.get(pid, model_minutes))),
            }
        )
    frame = pd.DataFrame(rows)
    edited = st.data_editor(
        frame,
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        key=f"who-edit-{game_id}-{side}",
        column_order=("Player", "Role", "Min"),
        column_config={
            "id": None,
            "Player": st.column_config.TextColumn(disabled=True, width="large"),
            "Role": st.column_config.SelectboxColumn(options=["Model", "On", "Off"], required=True, width="small"),
            "Min": st.column_config.NumberColumn(min_value=0, max_value=48, step=1, width="small"),
        },
    )
    drafts: dict[int, str] = {}
    minute_drafts: dict[int, float] = {}
    for row in edited.to_dict("records"):
        pid = int(row["id"])
        choice = str(row.get("Role") or "Model")
        minutes_value = float(row.get("Min") or 0)
        if choice in {"On", "Off"}:
            drafts[pid] = choice.lower()
        if choice == "Off":
            continue
        if abs(minutes_value - float(round(model_by_id.get(pid, 0.0)))) >= 0.5:
            minute_drafts[pid] = minutes_value
    return drafts, minute_drafts


def _who_plays(spec: GameSpec) -> tuple[str, bool]:
    """Model, on/off, and minutes for this game. True when Run this game was pressed."""
    game_id = str(spec.game_id)
    store = st.session_state.setdefault("availability", {})
    committed = str(store.get(game_id) or "")
    current, current_minutes = _parse_prompt(committed)
    edit = st.toggle("Adjust roster", key=f"edit-who-{game_id}", value=False)
    with st.form(f"who-plays-{game_id}"):
        drafts: dict[int, str] = {}
        minute_drafts: dict[int, float] = {}
        if edit:
            st.caption("Model keeps the report. On / Off force availability. Edit Min to lock floor time.")
            st.markdown(f'<div class="roster-head">{spec.away.name}</div>', unsafe_allow_html=True)
            modes, mins = _roster_editor(spec.away_players, current, current_minutes, game_id, "away")
            drafts.update(modes)
            minute_drafts.update(mins)
            st.markdown(f'<div class="roster-head">{spec.home.name}</div>', unsafe_allow_html=True)
            modes, mins = _roster_editor(spec.home_players, current, current_minutes, game_id, "home")
            drafts.update(modes)
            minute_drafts.update(mins)
        elif committed:
            st.caption("Using your saved roster locks for this game.")
        else:
            st.caption("Model roster. Turn on Adjust roster only if you need overrides.")
        submitted = st.form_submit_button("Run this game", type="primary")
    if submitted:
        if edit:
            parts = [f"{pid}:{mode}" for pid, mode in sorted(drafts.items())]
            parts.extend(f"{pid}:min={value:g}" for pid, value in sorted(minute_drafts.items()))
            token = ",".join(parts)
            store[game_id] = token
        else:
            token = committed
        return token, True
    return committed, False


def _availability_line(spec: GameSpec, token: str) -> str:
    wanted, minute_map = _parse_prompt(token)
    if not wanted and not minute_map:
        return ""
    names = {int(player.player_id): player.name for player in list(spec.home_players) + list(spec.away_players)}
    on = [names.get(pid, str(pid)) for pid, mode in sorted(wanted.items()) if mode == "on"]
    off = [names.get(pid, str(pid)) for pid, mode in sorted(wanted.items()) if mode == "off"]
    locked = [
        f"{names.get(pid, str(pid))} {value:g}"
        for pid, value in sorted(minute_map.items())
        if wanted.get(pid) != "off" and value > 0
    ]
    parts = []
    if on:
        parts.append("On: " + ", ".join(on))
    if off:
        parts.append("Off: " + ", ".join(off))
    if locked:
        parts.append("Minutes: " + ", ".join(locked))
    return " · ".join(parts)


def main() -> None:
    st.set_page_config(page_title="NBA Monte Carlo", layout="wide")
    _theme()
    st.markdown(
        '<div class="brand-bar"><div class="mark">Monte Carlo</div>'
        '<div class="sub">possession paths · Melbourne slate</div></div>',
        unsafe_allow_html=True,
    )
    if render_profile():
        return
    schedule = _schedule()
    with st.sidebar.expander("Source", expanded=False):
        source = st.radio(
            "Data",
            ["Warehouse", "Demo"],
            index=0 if not schedule.empty else 1,
            label_visibility="collapsed",
        )
        show_backtest = st.checkbox("Backtest tools", value=False)
    if show_backtest:
        st.session_state.screen = "slate"
        _backtest_page()
        return
    force = False
    if source == "Warehouse":
        force = st.sidebar.button(
            "Update",
            help="NBA.com rates for this season, the ESPN schedule, and the injury report. Then every game still to come is run again.",
        )
        st.sidebar.caption("Pulls ESPN injuries/schedule/rosters and NBA.com rates.")
    stamp = _maybe_refresh(force) if source == "Warehouse" else "demo"
    rerun = bool(st.session_state.pop("slate_rerun", False)) and source == "Warehouse"
    if source == "Warehouse":
        _rates_status()
    today = melbourne_today().isoformat()
    if st.session_state.get("slate_melbourne") != today:
        st.session_state["slate_melbourne"] = today
        st.session_state["slate_date"] = _default_date(schedule)
    on = st.sidebar.date_input("Melbourne date", key="slate_date")
    if not isinstance(on, date):
        on = melbourne_today()
    game_id = ""
    games: list[dict] = []
    if source == "Warehouse" and not schedule.empty:
        games = _day_games(schedule, on)
        if games:
            game_id = _choose_game(games)
    n_sims = st.sidebar.select_slider("Simulations", options=[500, 1500, 4000, 10000, 20000], value=1500)
    seed = st.sidebar.number_input("Seed", value=7, step=1)
    if rerun:
        _rerun_future(int(n_sims), int(seed), stamp, melbourne_today())
    if st.session_state.get("slate_board_stamp") != stamp:
        st.session_state["slate_board_stamp"] = stamp
        st.session_state["slate_boards"] = {}
    running = next((game["label"] for game in games if game["game_id"] == game_id), "")
    wait = st.empty()
    if running:
        wait.info(f"Loading {running.split(' · ')[0]}…")
    specs, resolved = _specs(source, on, game_id or None)
    wait.empty()
    if resolved == "empty":
        st.title("No games on this date")
        st.caption("Choose another day on the left.")
        return
    if game_id and specs:
        choice_id = game_id
    else:
        labels = {f"{spec.away.name} at {spec.home.name}": spec.game_id for spec in specs}
        names = list(labels)
        if st.session_state.get("game_choice") not in names:
            st.session_state["game_choice"] = names[0]
        choice = st.selectbox("Game", names, key="game_choice")
        choice_id = str(labels[choice])
        st.session_state["slate_game"] = choice_id
    if resolved == "demo-fallback":
        st.warning("The warehouse has no slate for that date, so this board is the demo slate.")
    elif resolved == "demo":
        st.caption("Demo slate. Prices are labeled demo and stay out of the basketball model.")
    chosen = next(item for item in specs if str(item.game_id) == choice_id)
    _prime_specs([chosen], resolved)
    st.title(f"{chosen.away.name} at {chosen.home.name}")
    st.markdown(
        f'<div class="match-strip"><span class="side away">{str(chosen.away.abbr).upper()}</span>'
        f'<span class="at">at</span><span class="side home">{str(chosen.home.abbr).upper()}</span></div>',
        unsafe_allow_html=True,
    )
    token, run_now = _who_plays(chosen)
    boards = st.session_state.setdefault("slate_boards", {})
    board = boards.get(str(choice_id))
    if run_now:
        with st.spinner(f"Running {chosen.away.name} at {chosen.home.name}…"):
            summary = _cached_summary(choice_id, source, on.isoformat(), int(n_sims), int(seed), stamp, token)
        boards[str(choice_id)] = {
            "summary": summary,
            "token": token,
            "n_sims": int(n_sims),
            "seed": int(seed),
        }
    elif board is not None:
        summary = board["summary"]
        token = str(board.get("token") or token)
    else:
        st.markdown(
            '<div class="empty-run">Press <strong>Run this game</strong> when you are ready. '
            "Nothing is simulated until then.</div>",
            unsafe_allow_html=True,
        )
        return
    st.session_state.summary = summary
    props = _game_props(summary, resolved)
    st.session_state.props = props
    actuals = _played_actuals(chosen) if resolved == "warehouse" else None
    played = _availability_line(chosen, token)
    caption = f"Melbourne {on.isoformat()} · {summary['n_sims']:,} simulations"
    st.caption(f"{caption} · {played}" if played else caption)
    _scoreboard(summary)
    away_col, home_col, _rest = st.columns([1.1, 1.1, 2.4])
    with away_col:
        if st.button(f"{summary['away_name']} profile", key="team-away"):
            open_team(summary["away"])
    with home_col:
        if st.button(f"{summary['home_name']} profile", key="team-home"):
            open_team(summary["home"])
    _prices(summary)
    _distributions(summary)
    _whistles(summary)
    _box_score(summary)
    _prop_board(summary, props, actuals)
    _matchups(summary)
    _minutes(summary)
    _players(summary)
    _factors(summary)


def _game_props(summary: dict, resolved: str) -> pd.DataFrame:
    if resolved != "warehouse" or not summary.get("game_id"):
        return pd.DataFrame()
    try:
        return load_props(str(summary["game_id"]))
    except Exception as exc:
        st.caption(f"Player props did not load. {exc}")
        return pd.DataFrame()


def _percent_points(value: float, signed: bool = False) -> float:
    """Whole-number percent, so a column can display 5% and still sort as 5."""
    text = f"{float(value):+.0%}" if signed else f"{float(value):.0%}"
    return float(text[:-1])


def _prop_table(
    summary: dict,
    props: pd.DataFrame,
    actuals: dict[str, dict[str, float]] | None = None,
) -> pd.DataFrame:
    if props is None or props.empty or "main" not in props.columns:
        return pd.DataFrame()
    mains = props[props["main"].astype(bool)].copy()
    if mains.empty:
        return pd.DataFrame()
    bins = pd.DataFrame(summary.get("player_bins") or [])
    names = {}
    teams = {}
    if not bins.empty and "espn_id" in bins.columns:
        for row in bins.drop_duplicates("espn_id").itertuples(index=False):
            names[str(row.espn_id)] = row.player
            teams[str(row.espn_id)] = row.team
    rows = []
    for prop in mains.to_dict("records"):
        espn_id = str(prop.get("espn_id") or "")
        if espn_id not in names:
            continue
        series = bins[(bins["espn_id"].astype(str) == espn_id) & (bins["stat"] == prop["stat"])]
        rate = None if series.empty else sim_over_rate(series, float(prop["line"]))
        fair = prop.get("fair_over")
        fair_missing = fair is None or pd.isna(fair)
        edge = None if rate is None or fair_missing else rate - float(fair)
        shown_edge = None if edge is None else _percent_points(edge, signed=True)
        over_price = prop.get("over_american")
        under_price = prop.get("under_american")
        take_under = shown_edge is not None and shown_edge < 0 and under_price is not None and not pd.isna(under_price)
        if take_under:
            side = "Under"
            sim_value = _percent_points(1.0 - rate)
            fair_value = _percent_points(1.0 - float(fair))
            edge_value = _percent_points((1.0 - rate) - (1.0 - float(fair)), signed=True)
            price_value = float(under_price)
        else:
            side = "Over"
            sim_value = None if rate is None else _percent_points(rate)
            fair_value = None if fair_missing else _percent_points(float(fair))
            edge_value = shown_edge
            price_value = None if over_price is None or pd.isna(over_price) else float(over_price)
        line = prop.get("line")
        line_value = None if line is None or pd.isna(line) else float(line)
        row = {
            "Team": teams.get(espn_id, ""),
            "Player": names.get(espn_id, ""),
            "Market": prop.get("market") or "",
            "Line": line_value,
            "Side": side,
            "Sim": sim_value,
            "Fair": fair_value,
            "Edge": edge_value,
            "Price": price_value,
        }
        if actuals is not None:
            counted = actuals.get(espn_id)
            actual_value = None if counted is None else counted.get(str(prop.get("stat") or ""))
            if edge_value is None:
                result = None
            elif counted is None:
                result = "DNP"
            elif actual_value is None or line_value is None:
                result = None
            else:
                result = grade_pick(float(actual_value), line_value, side)
            row["Actual"] = actual_value
            row["Result"] = result
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Team", "Player", "Market"])


def _made_line(made: float, attempted: float) -> str:
    return f"{made:.1f}-{attempted:.1f}"


def _whistles(summary: dict) -> None:
    fouls = summary.get("fouls")
    if not fouls:
        return
    st.subheader("Fouls and clock")
    st.caption(
        "Four 12-minute quarters, plus overtime when the game is tied. "
        "Team fouls reset each quarter. The fifth is the bonus, and the fourth is the bonus in overtime. "
        "A player's free-throw rate is how often he gets to the line, and the defender matched to the action is how often that foul is conceded. "
        "A shooting foul is two free throws, three when the shot is a three, and one when the shot goes in. "
        "A flagrant is two shots and the ball. The reckless ones end his night."
    )
    st.markdown(
        f"""
        <div class="facts">
          <div class="fact"><div class="k">Team fouls</div><div class="v">{summary['away']} {fouls['away_fouls']:.1f} · {summary['home']} {fouls['home_fouls']:.1f}</div></div>
          <div class="fact"><div class="k">Free throws</div><div class="v">{summary['away']} {fouls['away_ft']} · {summary['home']} {fouls['home_ft']}</div></div>
          <div class="fact"><div class="k">Possessions</div><div class="v">{summary['away']} {fouls['away_poss']:.0f} · {summary['home']} {fouls['home_poss']:.0f}</div></div>
          <div class="fact"><div class="k">Late fouls</div><div class="v">{fouls['intentional']:.1f} per game</div></div>
          <div class="fact"><div class="k">Flagrants</div><div class="v">{fouls.get('flagrant', 0):.2f} per game</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _box_score(summary: dict) -> None:
    rows = summary.get("box_score") or []
    if not rows:
        return
    st.subheader("Box score")
    st.caption(
        "Mean line from the same simulations. FG, 3PT, and FT are makes and attempts. "
        "An assist is a teammate's share of the makes, higher on a spot-up than an isolation, and the rebounder is credited for a shot off the glass. "
        "PF is the running personal-foul total, including the foul that sent a player to the line."
    )
    frame = pd.DataFrame(rows)
    for team in (summary["away"], summary["home"]):
        side = frame[frame["team"] == team]
        if side.empty:
            continue
        shown = pd.DataFrame(
            {
                "Player": side["player"],
                "MIN": side["min"],
                "PTS": side["pts"],
                "FG": [_made_line(made, attempted) for made, attempted in zip(side["fgm"], side["fga"])],
                "3PT": [_made_line(made, attempted) for made, attempted in zip(side["tpm"], side["tpa"])],
                "FT": [_made_line(made, attempted) for made, attempted in zip(side["ftm"], side["fta"])],
                "REB": side["reb"],
                "OREB": side["oreb"],
                "DREB": side["dreb"],
                "AST": side["ast"],
                "POT AST": side["pot_ast"],
                "STL": side["stl"],
                "BLK": side["blk"],
                "TO": side["tov"],
                "PF": side["pf"],
            }
        )
        number = st.column_config.NumberColumn(format="%.1f")
        st.markdown(f"**{team}**")
        st.dataframe(
            shown,
            hide_index=True,
            width="stretch",
            height="content",
            column_config={column: number for column in shown.columns if column not in {"Player", "FG", "3PT", "FT"}},
        )


def _prop_board(summary: dict, props: pd.DataFrame, actuals: dict | None = None) -> None:
    shown = _prop_table(summary, props, actuals)
    if shown.empty:
        return
    st.subheader("Player props")
    caption = (
        "Each row is the DraftKings line closest to a coin flip. "
        "The side is over when the sim beats that price, and under when it does not. "
        "Sim, edge, and price are for that side. The price stays out of the simulation."
    )
    if actuals is not None and "Result" in shown.columns:
        record = pick_record(shown["Result"].dropna())
        if record:
            caption += " " + record + ". A player missing from the box score did not play."
    st.caption(caption)
    config = {
        "Line": st.column_config.NumberColumn(format="%g"),
        "Sim": st.column_config.NumberColumn(format="%.0f%%"),
        "Fair": st.column_config.NumberColumn(format="%.0f%%"),
        "Edge": st.column_config.NumberColumn(format="%+.0f%%"),
        "Price": st.column_config.NumberColumn(format="%+.0f"),
    }
    if "Actual" in shown.columns:
        config["Actual"] = st.column_config.NumberColumn(format="%g")
    st.dataframe(shown, hide_index=True, width="stretch", height="content", column_config=config)


def _backtest_page() -> None:
    st.title("Backtest")
    st.caption("Earlier games set the ratings. The simulator then grades the closing spread, total, and moneyline.")
    path = raw_dir() / "nba_2008-2026.csv"
    last_n = st.slider("Games at the end of the file", 2, 12, 4)
    if st.button("Run backtest", type="primary"):
        result = run_backtest(last_n=int(last_n), n_sims=400, seed=1)
        if result["status"] == "missing_lines":
            st.warning(result["message"])
        else:
            st.success(result["message"])
            metrics = {
                market: values
                for market, values in result["metrics"].items()
                if isinstance(values, dict)
            }
            if metrics:
                st.dataframe(_metric_frame(metrics), hide_index=True, width="stretch")
            note = result["metrics"].get("note")
            if note:
                st.caption(note)
            if result.get("bets"):
                st.dataframe(pd.DataFrame(result["bets"]), hide_index=True, width="stretch")
        props = result.get("player_props") or {}
        st.info(props.get("message", ""))
    else:
        st.info(f"Place the historical line file at {path}, then run the backtest.")


def _metric_frame(metrics: dict) -> pd.DataFrame:
    frame = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "market"})
    for column in frame.columns:
        if column == "market":
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: None if pd.isna(value) else round(float(value), 3))
    return frame


if __name__ == "__main__":
    main()
