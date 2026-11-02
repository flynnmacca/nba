import pandas as pd

from nba_sim.ingest.markets import parse_kaggle


def test_kaggle_home_favorite_is_a_negative_home_spread():
    raw = pd.DataFrame(
        [
            {
                "date": "2026-01-15",
                "season": 2026,
                "home": "utah",
                "away": "ny",
                "score_home": 110,
                "score_away": 102,
                "whos_favored": "home",
                "spread": 6.5,
                "total": 220.5,
                "moneyline_home": -240,
                "moneyline_away": 200,
            }
        ]
    )
    parsed = parse_kaggle(raw)
    assert parsed.iloc[0]["home"] == "uta"
    assert parsed.iloc[0]["away"] == "nyk"
    assert parsed.iloc[0]["home_spread"] == -6.5
    assert parsed.iloc[0]["home_score"] == 110
