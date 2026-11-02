from nba_sim.teams import normalize_team


def test_public_odds_codes_map_to_one_abbreviation():
    cases = {
        "utah": "uta",
        "GSW": "gsw",
        "gs": "gsw",
        "ny": "nyk",
        "no": "nop",
        "sa": "sas",
        "pho": "phx",
        "bkn": "bkn",
        "brk": "bkn",
        "wsh": "was",
        "cho": "cha",
        "okc": "okc",
    }
    for raw, expected in cases.items():
        assert normalize_team(raw) == expected


def test_full_names_and_city_pairs():
    assert normalize_team("Los Angeles Lakers") == "lal"
    assert normalize_team("Los Angeles Clippers") == "lac"
    assert normalize_team("Golden State Warriors") == "gsw"
    assert normalize_team("Oklahoma City Thunder") == "okc"
    assert normalize_team("Portland Trail Blazers") == "por"
