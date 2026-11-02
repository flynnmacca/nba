import pandas as pd

from nba_sim.ingest.shot_zones import parse_shot_zones


def test_zone_parser_combines_corners_and_keeps_player_ids():
    columns = pd.MultiIndex.from_tuples(
        [
            ("", "PLAYER_ID"),
            ("Restricted Area", "FGM"),
            ("Restricted Area", "FGA"),
            ("Restricted Area", "FG_PCT"),
            ("Left Corner 3", "FGM"),
            ("Left Corner 3", "FGA"),
            ("Right Corner 3", "FGM"),
            ("Right Corner 3", "FGA"),
            ("Above the Break 3", "FGM"),
            ("Above the Break 3", "FGA"),
        ]
    )
    frame = pd.DataFrame(
        [[7, 4, 6, 0.67, 1, 2, 1, 2, 2, 6]],
        columns=columns,
    )
    parsed = parse_shot_zones(frame, "2025-26", "P")
    zones = dict(zip(parsed["zone"], parsed["fga"]))
    assert zones["ra"] == 6
    assert zones["corner3"] == 4
    assert zones["atb3"] == 6
    assert parsed["entity_id"].iloc[0] == 7
    assert abs(parsed.loc[parsed["zone"].eq("corner3"), "freq"].iloc[0] - 4 / 16) < 1e-9
