"""
NBA data fetching via nba_api.
Caches results to disk to avoid rate limiting.
"""
import json
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from nba_api.stats.endpoints import (
    playergamelog,
    leaguedashteamstats,
    leaguedashplayerstats,
    scoreboardv2,
    leaguegamefinder,
)
from nba_api.stats.static import players, teams

from config import config

os.makedirs(config.cache_dir, exist_ok=True)


def _cache_path(key: str) -> str:
    return os.path.join(config.cache_dir, f"{key}.json")


def _cache_valid(key: str) -> bool:
    path = _cache_path(key)
    if not os.path.exists(path):
        return False
    mtime = datetime.fromtimestamp(os.path.getmtime(path))
    return datetime.now() - mtime < timedelta(hours=config.cache_ttl_hours)


def _cache_read(key: str):
    with open(_cache_path(key)) as f:
        return json.load(f)


def _cache_write(key: str, data):
    with open(_cache_path(key), "w") as f:
        json.dump(data, f)


def _nba_api_call(endpoint_cls, cache_key: str, **kwargs) -> pd.DataFrame:
    """Generic cached NBA API call — returns first dataframe."""
    if _cache_valid(cache_key):
        return pd.DataFrame(_cache_read(cache_key))
    time.sleep(0.6)  # respect rate limit
    ep = endpoint_cls(**kwargs)
    df = ep.get_data_frames()[0]
    _cache_write(cache_key, df.to_dict("records"))
    return df


def get_todays_games() -> list[dict]:
    """Return today's NBA games from the scoreboard."""
    today = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"scoreboard_{today}"

    if _cache_valid(cache_key):
        return _cache_read(cache_key)

    time.sleep(0.6)
    sb = scoreboardv2.ScoreboardV2(game_date=today)
    games_df = sb.get_data_frames()[0]

    result = []
    for _, row in games_df.iterrows():
        result.append({
            "game_id": row.get("GAME_ID", ""),
            "home_team_id": row.get("HOME_TEAM_ID", ""),
            "away_team_id": row.get("VISITOR_TEAM_ID", ""),
            "home_team": team_id_to_name(row.get("HOME_TEAM_ID")),
            "away_team": team_id_to_name(row.get("VISITOR_TEAM_ID")),
            "game_time": row.get("GAME_STATUS_TEXT", ""),
        })

    _cache_write(cache_key, result)
    return result


def team_id_to_name(team_id) -> str:
    if not team_id:
        return ""
    all_teams = teams.get_teams()
    match = next((t for t in all_teams if t["id"] == int(team_id)), None)
    return match["full_name"] if match else str(team_id)


def find_player_id(name: str) -> Optional[int]:
    """Fuzzy match player name to NBA API player ID."""
    name_lower = name.lower()
    all_players = players.get_players()

    # Exact match first
    for p in all_players:
        if p["full_name"].lower() == name_lower:
            return p["id"]

    # Partial match
    parts = name_lower.split()
    for p in all_players:
        fn = p["full_name"].lower()
        if all(part in fn for part in parts):
            return p["id"]

    return None


def get_player_game_logs(player_id: int, last_n: int = 15) -> pd.DataFrame:
    """Get recent game logs for a player."""
    cache_key = f"gamelog_{player_id}_{config.nba_season_year}"
    df = _nba_api_call(
        playergamelog.PlayerGameLog,
        cache_key,
        player_id=str(player_id),
        season=config.nba_season,
    )
    return df.head(last_n)


def get_team_stats() -> pd.DataFrame:
    """Season team stats including pace and defensive ratings."""
    cache_key = f"team_stats_{config.nba_season_year}"
    return _nba_api_call(
        leaguedashteamstats.LeagueDashTeamStats,
        cache_key,
        season=config.nba_season,
        measure_type_simple="Advanced",
        per_mode_simple="PerGame",
    )


def get_player_season_stats() -> pd.DataFrame:
    """All players' season averages."""
    cache_key = f"player_stats_{config.nba_season_year}"
    return _nba_api_call(
        leaguedashplayerstats.LeagueDashPlayerStats,
        cache_key,
        season=config.nba_season,
        per_mode_simple="PerGame",
    )


def get_team_defensive_stats() -> pd.DataFrame:
    """Opponent stats allowed — for matchup adjustments."""
    cache_key = f"team_def_{config.nba_season_year}"
    return _nba_api_call(
        leaguedashteamstats.LeagueDashTeamStats,
        cache_key + "_opp",
        season=config.nba_season,
        measure_type_simple="Opponent",
        per_mode_simple="PerGame",
    )
