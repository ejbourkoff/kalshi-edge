"""
ESPN NBA player stats — free, fast, no auth needed.
Replaces slow nba_api (stats.nba.com) calls for player prop probability estimation.
"""
import math
import time
import requests
from typing import Optional

_ESPN_LEADERS = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/leaders"
_ESPN_ATHLETES = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/athletes"

_STAT_MAP = {
    "points": "pts",
    "rebounds": "reb",
    "assists": "ast",
    "steals": "stl",
    "blocks": "blk",
    "threes": "thr",
}

_DISPERSION = {
    "points":   (0.36, 2.0, 4.5),  # (slope, intercept, min_sigma)
    "rebounds": (0.40, 0.8, 1.8),
    "assists":  (0.42, 0.8, 1.5),
    "steals":   (0.58, 0.3, 0.5),
    "blocks":   (0.58, 0.3, 0.5),
    "threes":   (0.55, 0.5, 0.8),
}

_cache: dict = {}
_cache_ts: float = 0.0
_CACHE_TTL = 3600  # 1 hour


def get_player_averages() -> dict[str, dict]:
    """
    Fetch NBA player season averages from ESPN leaders + search fallback.
    Returns {player_name_lower: {"pts": x, "reb": y, "ast": z, ...}}
    Cached for 1 hour.
    """
    global _cache, _cache_ts
    if _cache and (time.time() - _cache_ts) < _CACHE_TTL:
        return _cache

    result = _fetch_leaders()
    _cache = result
    _cache_ts = time.time()
    return result


def _fetch_leaders() -> dict[str, dict]:
    """
    Fetch from ESPN leaders endpoint — covers top ~100 players per stat.
    Tries regular season (seasontype=2) first, falls back to playoffs (3).
    Always requests season=2025 explicitly so we get current-year stats.
    """
    cat_map = {
        "ptsLeader": "pts",
        "rebLeader": "reb",
        "astLeader": "ast",
        "stlLeader": "stl",
        "blkLeader": "blk",
        "3pmLeader": "thr",
        # alternate names ESPN uses in some seasons
        "points":    "pts",
        "rebounds":  "reb",
        "assists":   "ast",
        "steals":    "stl",
        "blocks":    "blk",
        "threePointFieldGoalsMade": "thr",
    }
    result: dict[str, dict] = {}

    for seasontype in ("2", "3"):  # regular season, then playoffs
        try:
            data = requests.get(
                _ESPN_LEADERS,
                params={"season": "2026", "seasontype": seasontype, "limit": "150"},
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=8,
            ).json()
        except Exception as e:
            print(f"[ESPN NBA] leaders error (seasontype={seasontype}): {e}")
            continue

        for cat in data.get("categories", []):
            key = cat_map.get(cat.get("name", ""))
            if not key:
                continue
            for leader in cat.get("leaders", []):
                athlete = leader.get("athlete") or {}
                name = athlete.get("displayName", "").lower()
                if not name:
                    continue
                # ESPN returns stats as a string like "32.7" or nested object
                raw_val = leader.get("statistics") or leader.get("displayValue") or "0"
                try:
                    val = float(str(raw_val).split()[0].replace(",", ""))
                except (TypeError, ValueError):
                    continue
                result.setdefault(name, {})[key] = val

        if result:
            print(f"[ESPN NBA] loaded {len(result)} players (seasontype={seasontype})")
            break

    if not result:
        print("[ESPN NBA] leaders returned 0 players — prop model disabled")
    return result


def estimate_over_prob(player: str, stat: str, threshold: float,
                       averages: dict | None = None) -> Optional[float]:
    """
    P(player achieves >= threshold for stat in a single game).
    Uses normal distribution with empirical std dev from season average.
    Returns None if player not in averages.
    """
    if averages is None:
        averages = get_player_averages()

    key = _STAT_MAP.get(stat)
    if not key:
        return None

    name = player.lower()
    pstats = averages.get(name)
    if not pstats:
        last = name.split()[-1]
        for k, v in averages.items():
            if k.split()[-1] == last and len(last) > 3:
                pstats = v
                break

    if not pstats:
        return None

    avg = pstats.get(key, 0.0)
    if avg <= 0:
        return None

    slope, intercept, min_sigma = _DISPERSION.get(stat, (0.40, 1.0, 1.5))
    sigma = max(avg * slope + intercept, min_sigma)

    # Normal CDF with continuity correction for discrete stat
    z = (threshold - 0.5 - avg) / sigma
    prob = 1 - 0.5 * (1 + math.erf(z / math.sqrt(2)))

    return round(max(0.03, min(0.97, prob)), 4)
