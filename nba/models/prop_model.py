"""
Player prop probability model.

Estimates the probability a player goes over/under a stat threshold.
Uses exponential-weighted rolling average, matchup adjustment, rest, and pace.
Converts to probability via Poisson (counting stats) or Normal distribution.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson, norm
from typing import Optional

import nba_data


STAT_COLUMNS = {
    "points": "PTS",
    "rebounds": "REB",
    "assists": "AST",
    "threes": "FG3M",
    "blocks": "BLK",
    "steals": "STL",
}

# How much variance each stat has relative to mean (overdispersion factor)
# Poisson assumes var=mean; these adjust for higher real-world variance
DISPERSION = {
    "points": 1.5,
    "rebounds": 1.3,
    "assists": 1.4,
    "threes": 1.8,
    "blocks": 2.0,
    "steals": 1.9,
}


def project_player_stat(
    player_name: str,
    stat: str,
    opponent_team_id: Optional[int] = None,
    is_home: bool = True,
    back_to_back: bool = False,
) -> Optional[float]:
    """
    Returns projected stat value for the player in tonight's game.
    Returns None if player data can't be fetched.
    """
    player_id = nba_data.find_player_id(player_name)
    if player_id is None:
        return None

    col = STAT_COLUMNS.get(stat)
    if col is None:
        return None

    logs = nba_data.get_player_game_logs(player_id, last_n=20)
    if logs.empty or col not in logs.columns:
        return None

    stats = pd.to_numeric(logs[col], errors="coerce").dropna().values

    if len(stats) < 3:
        return None

    # Exponential-weighted average: recent games matter more
    weights = np.exp(np.linspace(-1, 0, len(stats)))[::-1]
    weighted_avg = np.average(stats, weights=weights)

    # Season average (for regression to mean)
    season_avg = stats.mean()

    # Blend: 60% recent trend, 40% season average
    projected = 0.6 * weighted_avg + 0.4 * season_avg

    # Matchup adjustment (if we have opponent data)
    if opponent_team_id is not None:
        matchup_factor = get_matchup_factor(stat, opponent_team_id)
        projected *= matchup_factor

    # Home/away: slight bump at home for scorers, neutral for others
    if stat == "points" and is_home:
        projected *= 1.03
    elif stat == "points" and not is_home:
        projected *= 0.97

    # Back-to-back penalty: fatigue reduces output
    if back_to_back:
        projected *= 0.93

    return round(projected, 2)


def get_matchup_factor(stat: str, opponent_team_id: int) -> float:
    """
    Returns a multiplier based on how the opponent allows that stat.
    1.0 = league average. 1.1 = opponent allows 10% more than average.
    """
    try:
        def_stats = nba_data.get_team_defensive_stats()
        opp_row = def_stats[def_stats["TEAM_ID"] == opponent_team_id]
        if opp_row.empty:
            return 1.0

        col_map = {
            "points": "OPP_PTS",
            "rebounds": "OPP_REB",
            "assists": "OPP_AST",
            "threes": "OPP_FG3M",
        }
        col = col_map.get(stat)
        if not col or col not in opp_row.columns:
            return 1.0

        opp_val = float(opp_row[col].values[0])
        league_avg = float(def_stats[col].mean())
        if league_avg == 0:
            return 1.0

        return opp_val / league_avg
    except Exception:
        return 1.0


def prob_over(projected: float, threshold: float, stat: str) -> float:
    """
    Probability player exceeds threshold given projected mean.
    Uses negative binomial approximation (Poisson with overdispersion).
    """
    if projected <= 0:
        return 0.0

    disp = DISPERSION.get(stat, 1.4)
    variance = projected * disp
    std_dev = variance ** 0.5

    # For discrete stats use Normal approximation with continuity correction
    # (Poisson is good for low means; Normal works better for points ~20+)
    if projected < 10:
        # Poisson works well here
        p_over = 1 - poisson.cdf(int(threshold) - 1, mu=projected)
    else:
        # Normal with continuity correction
        p_over = 1 - norm.cdf(threshold - 0.5, loc=projected, scale=std_dev)

    return round(float(p_over), 4)


def prob_over_from_sportsbook(sportsbook_prob: float) -> float:
    """
    Sportsbook over probability already has vig baked in.
    We trust it as an anchor but discount by ~3% for vig.
    """
    return min(0.99, sportsbook_prob / 1.03)


def estimate_prop_probability(
    player_name: str,
    stat: str,
    threshold: float,
    sportsbook_prob: Optional[float] = None,
    opponent_team_id: Optional[int] = None,
    is_home: bool = True,
    back_to_back: bool = False,
) -> dict:
    """
    Master function: returns best probability estimate for a player prop.
    Blends our model with sportsbook line if available.
    """
    model_proj = project_player_stat(player_name, stat, opponent_team_id, is_home, back_to_back)

    if model_proj is not None:
        model_prob = prob_over(model_proj, threshold, stat)
    else:
        model_prob = None

    # If sportsbook line available, blend 40% model + 60% sportsbook
    # Sportsbook is more efficient — give it more weight
    if sportsbook_prob is not None and model_prob is not None:
        sb_prob = prob_over_from_sportsbook(sportsbook_prob)
        blended = 0.4 * model_prob + 0.6 * sb_prob
    elif sportsbook_prob is not None:
        blended = prob_over_from_sportsbook(sportsbook_prob)
    elif model_prob is not None:
        blended = model_prob
    else:
        blended = None

    return {
        "player": player_name,
        "stat": stat,
        "threshold": threshold,
        "model_projection": model_proj,
        "model_prob": model_prob,
        "sportsbook_prob": sportsbook_prob,
        "estimated_true_prob": round(blended, 4) if blended is not None else None,
    }
