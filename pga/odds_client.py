"""
The Odds API client for PGA Tour golf markets.
Returns outright winner probabilities with vig removed via overround normalization.

For multi-outcome markets (80+ players), vig is removed by dividing each player's
implied probability by the total overround of that book.
"""
import os
import json
import time
import requests
from difflib import SequenceMatcher
from typing import Optional

from config import config

# Known sport keys on The Odds API
GOLF_SPORT_KEYS = [
    "golf_pga_championship_winner",
    "golf_masters_tournament_winner",
    "golf_us_open_winner",
    "golf_the_open_championship_winner",
    "golf_the_players_championship_winner",
    "golf_wells_fargo_championship_winner",
    "golf_memorial_tournament_winner",
    "golf_travelers_championship_winner",
    "golf_canadian_open_winner",
    "golf_scottish_open_winner",
]


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (with vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def name_similarity(a: str, b: str) -> float:
    """Fuzzy name match ratio 0–1."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def best_name_match(target: str, candidates: list[str], threshold: float = 0.80) -> Optional[str]:
    """Return the best matching name from candidates, or None if below threshold."""
    best, best_score = None, 0.0
    for c in candidates:
        score = name_similarity(target, c)
        if score > best_score:
            best, best_score = c, score
    # Also try last-name-only match
    target_last = target.split()[-1].lower()
    for c in candidates:
        c_last = c.split()[-1].lower()
        if target_last == c_last and len(target_last) > 3:
            return c
    return best if best_score >= threshold else None


class OddsPGAClient:
    def __init__(self):
        self.base_url = config.odds_base_url
        self.key = config.odds_api_key
        self._cache: dict = {}
        self._cache_time: dict = {}
        self._ttl = config.cache_ttl_hours * 3600

    def _get(self, path: str, params: dict = None) -> dict | list:
        p = {"apiKey": self.key, **(params or {})}
        resp = requests.get(f"{self.base_url}{path}", params=p, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def get_available_sports(self) -> list[str]:
        """Return list of active golf sport keys."""
        try:
            data = self._get("/sports", params={"all": "false"})
            return [s["key"] for s in data if "golf" in s.get("key", "").lower() and s.get("has_outrights")]
        except Exception:
            return []

    def get_tournament_outrights(self, sport_key: str) -> list[dict]:
        """
        Fetch outright winner odds for a tournament from 4-6 books.
        Returns list of {player, true_prob, implied_prob, num_books, books_detail}
        with vig removed using overround normalization.
        """
        cache_key = sport_key
        now = time.time()
        if cache_key in self._cache and (now - self._cache_time.get(cache_key, 0)) < self._ttl:
            return self._cache[cache_key]

        try:
            data = self._get(
                f"/sports/{sport_key}/odds",
                params={
                    "regions": "us",
                    "markets": "outrights",
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel,betmgm,caesars,betrivers,lowvig",
                },
            )
        except Exception as e:
            print(f"[OddsAPI] {sport_key}: {e}")
            return []

        if not data:
            return []

        # data is a list of events; for outrights there's usually one event
        event = data[0] if isinstance(data, list) else data

        # Collect per-book: {player_name: implied_prob}
        book_data: dict[str, dict[str, float]] = {}  # book_key → {player: prob}
        for book in event.get("bookmakers", []):
            book_key = book["key"]
            player_probs = {}
            for market in book.get("markets", []):
                if market["key"] != "outrights":
                    continue
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "")
                    price = outcome.get("price", 0)
                    if name and price:
                        player_probs[name] = american_to_prob(price)
            if player_probs:
                book_data[book_key] = player_probs

        if not book_data:
            return []

        # Collect all known player names
        all_players = set()
        for bp in book_data.values():
            all_players.update(bp.keys())

        # For each book, remove vig by normalizing to sum = 1.0
        # book_true: {book_key → {player: true_prob}}
        book_true: dict[str, dict[str, float]] = {}
        for book_key, bp in book_data.items():
            overround = sum(bp.values())
            if overround == 0:
                continue
            book_true[book_key] = {p: v / overround for p, v in bp.items()}

        # Average true probabilities across books for each player
        results = []
        for player in all_players:
            probs = [bt[player] for bt in book_true.values() if player in bt]
            if not probs:
                continue
            avg_true = sum(probs) / len(probs)
            avg_implied = sum(book_data[bk].get(player, 0) for bk in book_data) / len(book_data)
            results.append({
                "player": player,
                "true_prob": round(avg_true, 5),
                "implied_prob": round(avg_implied, 5),
                "num_books": len(probs),
                "books": list(book_true.keys()),
            })

        results.sort(key=lambda x: -x["true_prob"])
        self._cache[cache_key] = results
        self._cache_time[cache_key] = now
        return results

    def get_all_tournaments(self) -> dict[str, list[dict]]:
        """Fetch outrights for all active golf tournaments. Returns {sport_key: [players]}."""
        sports = self.get_available_sports()
        out = {}
        for sk in sports:
            data = self.get_tournament_outrights(sk)
            if data:
                out[sk] = data
            time.sleep(0.3)
        return out
