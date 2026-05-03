"""
Game moneyline win probability model.

Uses net rating differential + home court + rest as primary predictors.
Sportsbook consensus line is treated as the primary anchor — our model
is used when the sportsbook line isn't available or to detect divergence.
"""
import numpy as np
from scipy.stats import norm
from typing import Optional

import nba_data


# Historical std dev of point differential in NBA games
GAME_STD_DEV = 11.5

# Home court advantage in equivalent points
HOME_COURT_POINTS = 3.5

# Back-to-back penalty in equivalent points
B2B_PENALTY = 2.5


def net_rating_to_win_prob(home_net_rtg: float, away_net_rtg: float,
                            home_rest_days: int = 2, away_rest_days: int = 2) -> float:
    """
    Estimate home team win probability from net ratings.

    Net rating = ORTG - DRTG per 100 possessions.
    ~4 net rating points ≈ 1 win probability point.
    """
    rating_diff = home_net_rtg - away_net_rtg

    # Home court is worth about 3.5 points
    adjusted_diff = rating_diff + HOME_COURT_POINTS

    # Rest adjustment
    if home_rest_days == 1:  # back-to-back
        adjusted_diff -= B2B_PENALTY
    if away_rest_days == 1:
        adjusted_diff += B2B_PENALTY

    # Convert point differential to win probability using Normal CDF
    win_prob = norm.cdf(adjusted_diff / GAME_STD_DEV)
    return round(float(win_prob), 4)


def spread_to_win_prob(spread: float) -> float:
    """
    Convert a point spread to win probability.
    Spread is from home team's perspective (negative = home favored).
    """
    # spread of -5.5 means home team expected to win by 5.5
    win_prob = norm.cdf(-spread / GAME_STD_DEV)
    return round(float(win_prob), 4)


def get_team_net_rating(team_name: str) -> Optional[float]:
    """Fetch a team's net rating from NBA API."""
    try:
        stats = nba_data.get_team_stats()
        name_lower = team_name.lower()

        # Try to match team name
        for _, row in stats.iterrows():
            if name_lower in str(row.get("TEAM_NAME", "")).lower():
                return float(row.get("NET_RATING", 0))

        return None
    except Exception:
        return None


def estimate_game_probability(
    home_team: str,
    away_team: str,
    sportsbook_home_prob: Optional[float] = None,
    home_rest_days: int = 2,
    away_rest_days: int = 2,
) -> dict:
    """
    Master function: returns best win probability estimate for home team.
    If sportsbook line exists, use it as primary anchor.
    Our model is used when no line is available or to flag divergences.
    """
    home_net = get_team_net_rating(home_team)
    away_net = get_team_net_rating(away_team)

    if home_net is not None and away_net is not None:
        model_prob = net_rating_to_win_prob(home_net, away_net, home_rest_days, away_rest_days)
    else:
        model_prob = None

    # If sportsbook line available, it's more efficient — weight 70/30 vs model
    if sportsbook_home_prob is not None and model_prob is not None:
        # Flag large divergences (model vs sportsbook > 7pp) — those are interesting
        divergence = abs(sportsbook_home_prob - model_prob)
        blended = 0.3 * model_prob + 0.7 * sportsbook_home_prob
    elif sportsbook_home_prob is not None:
        blended = sportsbook_home_prob
        divergence = None
    elif model_prob is not None:
        blended = model_prob
        divergence = None
    else:
        blended = None
        divergence = None

    return {
        "home_team": home_team,
        "away_team": away_team,
        "model_prob_home": model_prob,
        "sportsbook_prob_home": sportsbook_home_prob,
        "estimated_true_prob_home": round(blended, 4) if blended is not None else None,
        "model_sb_divergence": round(divergence, 4) if divergence is not None else None,
        "flag_divergence": divergence is not None and divergence > 0.07,
    }
