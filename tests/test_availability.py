import numpy as np
import pytest

from nba_sim.features.availability import box_absences, hold_known_outs, known_out_reason, nba_box_absences
from nba_sim.features.court import PlayerCourt, availability_mask
from nba_sim.sim.specs import PlayerModel


def _player(name: str, minutes: float) -> PlayerModel:
    return PlayerModel(player_id=1, name=name, team_id=1, position="F", minutes_mean=minutes, available=True)


def test_an_injury_is_known_and_a_box_holds_a_coach_decision():
    assert known_out_reason("RIGHT ANKLE SURGERY")
    assert known_out_reason("ILLNESS")
    assert known_out_reason("G LEAGUE - TWO-WAY")
    assert known_out_reason("Left Knee; Injury Management")
    assert known_out_reason("INACTIVE_GLEAGUE_TWOWAY")
    # Live injury reports still leave a dressed scratch on his play rate.
    assert not known_out_reason("COACH'S DECISION")
    assert not known_out_reason("DNP_COACH")
    assert known_out_reason("REST")
    assert known_out_reason("DND_REST")
    assert known_out_reason("INACTIVE_REST")
    assert known_out_reason("INACTIVE_COACH")
    assert not known_out_reason("")
    tate = _player("Jae'Sean Tate", 22.0)
    green = _player("Jeff Green", 12.0)
    holiday = _player("Aaron Holiday", 18.0)
    held = hold_known_outs(
        [tate, green, holiday],
        "hou",
        [
            {"team": "hou", "name": "Jae'Sean Tate", "reason": "RIGHT ANKLE SURGERY"},
            {"team": "hou", "name": "Jeff Green", "reason": "COACH'S DECISION"},
            {"team": "hou", "name": "Aaron Holiday", "reason": "DND_REST"},
            {"team": "okc", "name": "Jae'Sean Tate", "reason": "ILLNESS"},
        ],
    )
    assert held == ["Jae'Sean Tate", "Jeff Green", "Aaron Holiday"]
    assert tate.available is False
    assert tate.minutes_mean == 0.0
    assert tate.minutes_base == 22.0
    assert green.available is False
    assert green.minutes_mean == 0.0
    assert holiday.available is False
    assert holiday.minutes_mean == 0.0


def test_box_absences_keep_only_the_players_who_did_not_play():
    payload = {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "HOU"},
                    "statistics": [
                        {
                            "athletes": [
                                {
                                    "didNotPlay": True,
                                    "reason": "ILLNESS",
                                    "athlete": {"displayName": "Kristaps Porzingis"},
                                },
                                {
                                    "didNotPlay": False,
                                    "reason": "",
                                    "athlete": {"displayName": "Jalen Green"},
                                },
                            ]
                        }
                    ],
                }
            ]
        }
    }
    rows = box_absences(payload)
    assert rows == [{"team": "hou", "name": "Kristaps Porzingis", "reason": "ILLNESS"}]


def test_the_official_box_holds_an_inactive_injury_and_a_dressed_scratch():
    payload = {
        "game": {
            "awayTeam": {
                "teamTricode": "DEN",
                "players": [
                    {
                        "name": "Nikola Jokić",
                        "status": "INACTIVE",
                        "played": "0",
                        "notPlayingReason": "INACTIVE_INJURY",
                        "notPlayingDescription": "Left Knee; Injury Management",
                    },
                    {
                        "name": "Hunter Tyson",
                        "status": "ACTIVE",
                        "played": "0",
                        "notPlayingReason": None,
                        "notPlayingDescription": None,
                    },
                    {
                        "name": "Jamal Murray",
                        "status": "ACTIVE",
                        "played": "1",
                        "notPlayingReason": None,
                        "notPlayingDescription": None,
                    },
                ],
            },
            "homeTeam": {
                "teamTricode": "CLE",
                "players": [
                    {
                        "name": "Sam Merrill",
                        "status": "ACTIVE",
                        "played": "0",
                        "notPlayingReason": "DND_INJURY",
                        "notPlayingDescription": "Left Ankle; Sprain",
                    }
                ],
            },
        }
    }
    rows = nba_box_absences(payload)
    assert rows == [
        {"team": "den", "name": "Nikola Jokić", "reason": "Left Knee; Injury Management"},
        {"team": "den", "name": "Hunter Tyson", "reason": "Coach's Decision"},
        {"team": "cle", "name": "Sam Merrill", "reason": "Left Ankle; Sprain"},
    ]
    jokic = _player("Nikola Jokić", 35.0)
    tyson = _player("Hunter Tyson", 12.0)
    murray = _player("Jamal Murray", 34.0)
    held = hold_known_outs([jokic, tyson, murray], "den", rows)
    assert held == ["Nikola Jokić", "Hunter Tyson"]
    assert jokic.available is False
    assert jokic.minutes_mean == 0.0
    assert tyson.available is False
    assert tyson.minutes_mean == 0.0
    assert murray.available is True
    assert murray.minutes_mean == 34.0


def test_a_named_rest_moves_his_night_onto_the_man_who_stays():
    sitter = PlayerModel(player_id=1, name="Pascal Siakam", team_id=1, position="F", minutes_mean=34.0, available=True)
    mate = PlayerModel(player_id=2, name="Aaron Nesmith", team_id=1, position="F", minutes_mean=24.0, available=True)
    sitter.court = PlayerCourt(player_id=1, minutes=34.0, minutes_close=33.0, minutes_blowout=26.0, position="F")
    mate.court = PlayerCourt(player_id=2, minutes=24.0, minutes_close=24.0, minutes_blowout=18.0, position="F")
    matrix = {("F", "F"): 1.0}
    sitter._sub_matrix = matrix
    mate._sub_matrix = matrix
    held = hold_known_outs(
        [sitter, mate],
        "ind",
        [{"team": "ind", "name": "Pascal Siakam", "reason": "DND_REST"}],
    )
    assert held == ["Pascal Siakam"]
    assert sitter.available is False
    assert sitter.court.minutes == 0.0
    assert sitter.court.minutes_close == 0.0
    assert mate.court.minutes == 58.0
    assert mate.court.minutes_close == 57.0
    assert mate.court.minutes_blowout == 44.0
    assert mate.minutes_mean == 58.0


def test_absent_minutes_split_across_the_position_not_copied_to_each_man():
    # The matrix column is a position share. Three wings still available must
    # split it, or holding a few coach DNPs inflates everyone into the 300s.
    sitter = PlayerModel(player_id=1, name="Out", team_id=1, position="F", minutes_mean=20.0, available=True)
    a = PlayerModel(player_id=2, name="A", team_id=1, position="F", minutes_mean=30.0, available=True)
    b = PlayerModel(player_id=3, name="B", team_id=1, position="F", minutes_mean=20.0, available=True)
    c = PlayerModel(player_id=4, name="C", team_id=1, position="F", minutes_mean=10.0, available=True)
    sitter.court = PlayerCourt(player_id=1, minutes=20.0, minutes_close=20.0, minutes_blowout=12.0, position="F")
    a.court = PlayerCourt(player_id=2, minutes=30.0, minutes_close=30.0, minutes_blowout=22.0, position="F")
    b.court = PlayerCourt(player_id=3, minutes=20.0, minutes_close=20.0, minutes_blowout=14.0, position="F")
    c.court = PlayerCourt(player_id=4, minutes=10.0, minutes_close=10.0, minutes_blowout=8.0, position="F")
    matrix = {("F", "F"): 1.0}
    for player in (sitter, a, b, c):
        player._sub_matrix = matrix
    held = hold_known_outs([sitter, a, b, c], "ind", [{"team": "ind", "name": "Out", "reason": "Coach's Decision"}])
    assert held == ["Out"]
    gained = (a.court.minutes - 30.0) + (b.court.minutes - 20.0) + (c.court.minutes - 10.0)
    assert gained == pytest.approx(20.0, abs=0.05)
    assert a.court.minutes < 45.0
    assert b.court.minutes < 35.0
    assert c.court.minutes < 25.0
    assert a.court.minutes > b.court.minutes > c.court.minutes
    rested = PlayerModel(player_id=3, name="Lauri Markkanen", team_id=1, position="F", minutes_mean=36.0, available=True)
    rested.injury_label = "Rest"
    rested.injury_factor = 1.0
    rested.court = PlayerCourt(player_id=3, minutes=36.0, play_rate=1.0)
    mask = availability_mask([rested], 12, np.random.default_rng(1))
    assert not mask.any()
