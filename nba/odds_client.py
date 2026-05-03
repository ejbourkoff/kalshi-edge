"""
The Odds API client — fetches sportsbook consensus lines.
These are treated as the "true probability" anchor, since sportsbooks
are more efficient and faster to reprice than Kalshi.
"""
import requests
from typing import Optional
from config import config


class OddsClient:
    def __init__(self):
        self.base_url = config.odds_base_url
        self.key = config.odds_api_key

    def _get(self, path: str, params: dict = None) -> dict | list:
        p = {"apiKey": self.key, **(params or {})}
        resp = requests.get(f"{self.base_url}{path}", params=p, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def get_nba_moneylines(self) -> list[dict]:
        """
        Returns list of today's NBA games with consensus win probabilities.
        Uses h2h (moneyline) market, averages across bookmakers.
        """
        try:
            data = self._get(
                "/sports/basketball_nba/odds",
                params={
                    "regions": "us",
                    "markets": "h2h",
                    "oddsFormat": "american",
                    "bookmakers": "draftkings,fanduel,betmgm,caesars,pointsbet",
                },
            )
        except Exception as e:
            print(f"[OddsAPI] Could not fetch lines: {e}")
            return []

        games = []
        for event in data:
            home = event.get("home_team", "")
            away = event.get("away_team", "")
            commence = event.get("commence_time", "")

            home_probs = []
            away_probs = []

            for book in event.get("bookmakers", []):
                for market in book.get("markets", []):
                    if market["key"] != "h2h":
                        continue
                    outcomes = {o["name"]: o["price"] for o in market["outcomes"]}
                    if home in outcomes and away in outcomes:
                        hp = american_to_prob(outcomes[home])
                        ap = american_to_prob(outcomes[away])
                        # Remove vig
                        total = hp + ap
                        home_probs.append(hp / total)
                        away_probs.append(ap / total)

            if home_probs:
                games.append({
                    "home_team": home,
                    "away_team": away,
                    "commence_time": commence,
                    "home_win_prob": round(sum(home_probs) / len(home_probs), 4),
                    "away_win_prob": round(sum(away_probs) / len(away_probs), 4),
                    "num_books": len(home_probs),
                })

        return games

    def get_nba_player_props(self) -> list[dict]:
        """
        Fetches player props (points, rebounds, assists) from sportsbooks.
        Returns implied over probabilities we can compare to Kalshi.
        """
        prop_markets = ["player_points", "player_rebounds", "player_assists", "player_threes"]
        results = []

        for market_key in prop_markets:
            try:
                data = self._get(
                    "/sports/basketball_nba/odds",
                    params={
                        "regions": "us",
                        "markets": market_key,
                        "oddsFormat": "american",
                    },
                )
            except Exception:
                continue

            stat_type = market_key.replace("player_", "")

            for event in data:
                for book in event.get("bookmakers", []):
                    for market in book.get("markets", []):
                        if market["key"] != market_key:
                            continue
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                results.append({
                                    "player": outcome.get("description", ""),
                                    "stat": stat_type,
                                    "line": outcome.get("point", 0),
                                    "over_odds": outcome.get("price", 0),
                                    "over_prob": american_to_prob(outcome.get("price", -110)),
                                    "game": f"{event['away_team']} @ {event['home_team']}",
                                    "bookmaker": book["key"],
                                })

        # Average over_prob per player/stat/line across books
        from collections import defaultdict
        grouped = defaultdict(list)
        for r in results:
            key = (r["player"], r["stat"], r["line"])
            grouped[key].append(r["over_prob"])

        averaged = []
        for (player, stat, line), probs in grouped.items():
            # Adjust for vig: sportsbook over price already has ~5% vig baked in
            # Raw average of over probs from multiple books ≈ true prob + small vig
            # We apply a ~3% vig discount to get closer to true probability
            raw_avg = sum(probs) / len(probs)
            true_prob_est = raw_avg / 1.05  # rough vig removal
            averaged.append({
                "player": player,
                "stat": stat,
                "line": line,
                "sportsbook_over_prob": round(true_prob_est, 4),
                "num_books": len(probs),
            })

        return averaged


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (with vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)
