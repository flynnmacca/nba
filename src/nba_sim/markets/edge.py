"""Model probability at a posted line, versus a de-vigged price."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from nba_sim.markets.devig import devig_american
from nba_sim.markets.kelly import interval_contains, kelly_stake
from nba_sim.sim.engine import SimResult
from nba_sim.sim.specs import GameSpec, MarketQuote


def prob_over(samples: np.ndarray, line: float) -> float:
    samples = np.asarray(samples, dtype=float)
    return float(np.mean(samples > line) + 0.5 * np.mean(samples == line))


def prob_cover(home: np.ndarray, away: np.ndarray, spread_for_side: float) -> float:
    """``spread_for_side`` is added to that side's margin. Home -4.5 uses -4.5."""
    margin = np.asarray(home, dtype=float) - np.asarray(away, dtype=float)
    # Caller passes the side's own margin against the line. For the away side
    # the margin is away - home, and the spread is the away number.
    adjusted = margin + spread_for_side
    return float(np.mean(adjusted > 0) + 0.5 * np.mean(adjusted == 0))


def _side_probability(quote: MarketQuote, result: SimResult, spec: GameSpec) -> float:
    if quote.market == "moneyline":
        if quote.side == "home":
            return prob_cover(result.home_points, result.away_points, 0.0)
        return prob_cover(result.away_points, result.home_points, 0.0)
    if quote.market == "spread":
        if quote.side == "home":
            return prob_cover(result.home_points, result.away_points, float(quote.line or 0.0))
        return prob_cover(result.away_points, result.home_points, float(quote.line or 0.0))
    if quote.market == "total":
        total = result.home_points + result.away_points
        over = prob_over(total, float(quote.line))
        return over if quote.side == "over" else 1.0 - over
    stat_name = {
        "player_points": "pts",
        "player_rebounds": "reb",
        "player_assists": "ast",
        "player_threes": "tpm",
        "player_pra": "pra",
    }.get(quote.market)
    if stat_name is None or quote.player_id is None:
        return float("nan")
    samples = result.player_stats[quote.player_id][stat_name]
    over = prob_over(samples, float(quote.line))
    return over if quote.side == "over" else 1.0 - over


def price_quotes(spec: GameSpec, result: SimResult) -> list[dict]:
    groups: dict[tuple, list[MarketQuote]] = defaultdict(list)
    for quote in spec.quotes:
        groups[(quote.market, quote.player_id, quote.line)].append(quote)
    rows: list[dict] = []
    for key, quotes in groups.items():
        model = {quote.side: _side_probability(quote, result, spec) for quote in quotes}
        if any(not np.isfinite(value) for value in model.values()):
            continue
        fair = None
        if len(quotes) >= 2:
            order = sorted(quotes, key=lambda quote: quote.side)
            fair_values = devig_american([quote.american for quote in order])
            fair = {quote.side: float(value) for quote, value in zip(order, fair_values)}
        for quote in quotes:
            model_p = float(model[quote.side])
            if fair is not None and quote.side in fair:
                market_p = fair[quote.side]
                devigged = True
            else:
                from nba_sim.markets.devig import american_to_implied

                market_p = american_to_implied(quote.american)
                devigged = False
            edge = model_p - market_p
            swallowed = interval_contains(model_p, market_p, result.n_sims)
            stake = 0.0 if swallowed or edge <= 0 else kelly_stake(model_p, quote.american)
            rows.append(
                {
                    "source": quote.source,
                    "book": quote.book,
                    "market": quote.market,
                    "side": quote.side,
                    "line": quote.line,
                    "player_id": quote.player_id,
                    "player": quote.player_name or "",
                    "american": quote.american,
                    "model_prob": model_p,
                    "market_prob": market_p,
                    "edge": edge,
                    "stake": stake,
                    "devigged": devigged,
                    "swallowed": swallowed,
                }
            )
    return rows
