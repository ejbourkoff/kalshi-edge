"""
Kalshi API client for PGA Tour prediction markets.
Base URL: https://api.elections.kalshi.com/trade-api/v2

All PGA Tour markets live under series KXPGATOUR.
Market type: outright winner binary contracts — "Will [Player] win [Tournament]?"
Prices in USD dollars (0.07 = 7¢), no auth needed for reads.
"""
import base64
import re
import time
from typing import Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding

from config import config

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Known KXPGATOUR event suffixes → tournament name
TOURNAMENT_MAP = {
    "PGC26":   "2026 PGA Championship",
    "CAC26":   "2026 Cadillac Championship",
    "MAST26":  "2026 Masters Tournament",
    "USOP26":  "2026 US Open",
    "BRIT26":  "2026 The Open Championship",
    "PLAY26":  "2026 The Players Championship",
    "HERO26":  "2026 Genesis Scottish Open",
    "ZUCON26": "2026 Zurich Classic",
    "RBCH26":  "2026 RBC Heritage",
    "RBH26":   "2026 RBC Heritage",
    "WELLS26": "2026 Wells Fargo Championship",
    "BYRON26": "2026 AT&T Byron Nelson",
    "MEMO26":  "2026 Memorial Tournament",
    "TRAX26":  "2026 Travelers Championship",
    "ROCKET26":"2026 Rocket Classic",
    "JOHN26":  "2026 John Deere Classic",
    "GENSC26": "2026 Scottish Open",
    "CANADI26":"2026 Canadian Open",
    "BARR26":  "2026 Barracuda Championship",
}

# The Odds API sport key for each Kalshi event suffix
TOURNAMENT_ODDS_KEY = {
    "PGC26":  "golf_pga_championship_winner",
    "MAST26": "golf_masters_tournament_winner",
    "USOP26": "golf_us_open_winner",
    "BRIT26": "golf_the_open_championship_winner",
}


class KalshiPGAClient:
    def __init__(self):
        self.base_url = BASE_URL
        self.api_key = config.kalshi_api_key
        self._session = requests.Session()
        self._session.headers.update({"accept": "application/json"})

    def _get(self, endpoint: str, params: dict = None, retries: int = 3) -> dict:
        for attempt in range(retries):
            resp = self._session.get(
                f"{self.base_url}{endpoint}",
                params=params or {},
                timeout=10,
            )
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    def get_pga_events(self) -> list[dict]:
        """Return all open KXPGATOUR events."""
        data = self._get("/events", params={
            "series_ticker": "KXPGATOUR",
            "status": "open",
            "limit": 20,
        })
        return data.get("events", [])

    def get_markets_for_event(self, event_ticker: str) -> list[dict]:
        data = self._get("/markets", params={
            "event_ticker": event_ticker,
            "limit": 200,
        })
        return data.get("markets", [])

    def get_all_pga_markets(self) -> list[dict]:
        """Fetch every open PGA winner market, grouped by event."""
        events = self.get_pga_events()
        all_markets = []
        for ev in events:
            time.sleep(0.3)
            mkts = self.get_markets_for_event(ev["event_ticker"])
            suffix = self._parse_event_suffix(ev["event_ticker"])
            tournament = TOURNAMENT_MAP.get(suffix, ev.get("title", ev["event_ticker"]))
            odds_key = TOURNAMENT_ODDS_KEY.get(suffix)
            for m in mkts:
                m["_event_title"] = ev.get("title", "")
                m["_tournament"] = tournament
                m["_tournament_suffix"] = suffix
                m["_odds_key"] = odds_key
            all_markets.extend(mkts)
        return all_markets

    def parse_market(self, raw: dict) -> Optional[dict]:
        """
        Normalize a raw Kalshi PGA market to a consistent shape.
        Returns None if no live price.
        """
        yes_ask_usd = raw.get("yes_ask_dollars")
        no_ask_usd  = raw.get("no_ask_dollars")
        yes_bid_usd = raw.get("yes_bid_dollars")

        try:
            yes_ask = round(float(yes_ask_usd) * 100)
            no_ask  = round(float(no_ask_usd or 0) * 100)
        except (TypeError, ValueError):
            return None

        if yes_ask == 0:
            return None

        # Illiquid filter: sum of spreads way above $1 = no real orderbook
        if no_ask > 0 and (yes_ask + no_ask) > 115:
            return None

        ticker = raw.get("ticker", "")
        title  = raw.get("title", "")
        player = self._parse_player_name(title)

        return {
            "ticker": ticker,
            "event_ticker": raw.get("event_ticker", ""),
            "title": title,
            "tournament": raw.get("_tournament", ""),
            "tournament_suffix": raw.get("_tournament_suffix", ""),
            "odds_key": raw.get("_odds_key"),
            "player": player,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "yes_bid": round(float(yes_bid_usd or 0) * 100),
            "kalshi_prob": yes_ask / 100,
            "volume": float(raw.get("volume_fp", 0) or 0),
            "close_time": raw.get("close_time", ""),
        }

    def get_all_parsed_markets(self) -> list[dict]:
        raw = self.get_all_pga_markets()
        parsed = [self.parse_market(m) for m in raw]
        return [p for p in parsed if p is not None]

    def _parse_event_suffix(self, event_ticker: str) -> str:
        """'KXPGATOUR-PGC26' → 'PGC26'"""
        parts = event_ticker.split("-")
        return parts[-1] if len(parts) > 1 else ""

    def get_finishing_markets(self, event_suffix: str) -> dict[int, list[dict]]:
        """
        Fetch Top 5/10/20 finishing position markets for a specific event.
        Returns {5: [markets], 10: [markets], 20: [markets]}
        """
        from finishing_scanner import FINISHING_SERIES
        result: dict[int, list[dict]] = {}

        for threshold, series in FINISHING_SERIES.items():
            event_ticker = f"{series}-{event_suffix}"
            data = self._get("/markets", params={
                "event_ticker": event_ticker,
                "limit": 200,
                "status": "open",
            })
            raw_markets = data.get("markets", [])
            parsed = []
            for raw in raw_markets:
                p = self._parse_finishing_market(raw, threshold)
                if p:
                    parsed.append(p)
            result[threshold] = parsed
            time.sleep(0.2)

        return result

    def get_active_finishing_events(self) -> list[str]:
        """Return event suffixes that have active finishing-position markets (liquid + priced)."""
        from finishing_scanner import FINISHING_SERIES
        active = set()
        for series in FINISHING_SERIES.values():
            data = self._get("/markets", params={
                "series_ticker": series,
                "status": "open",
                "limit": 200,
            })
            for m in data.get("markets", []):
                ya = m.get("yes_ask_dollars")
                vol = float(m.get("volume_fp", 0) or 0)
                if ya and float(ya) > 0 and vol > 500:
                    # Extract event suffix: KXPGATOP20-CAC26-KRR → CAC26
                    parts = m.get("event_ticker", "").split("-")
                    if len(parts) >= 2:
                        active.add(parts[-1])
        return list(active)

    def _parse_finishing_market(self, raw: dict, threshold: int) -> Optional[dict]:
        """Parse a finishing-position market from raw Kalshi API response."""
        ya = raw.get("yes_ask_dollars")
        na = raw.get("no_ask_dollars")
        if not ya:
            return None
        try:
            yes_ask = round(float(ya) * 100)
            no_ask  = round(float(na) * 100) if na else 100 - yes_ask
        except (TypeError, ValueError):
            return None

        if yes_ask <= 0:
            return None
        if (yes_ask + no_ask) > 130:  # illiquid — finishing markets have wider spreads
            return None

        volume = float(raw.get("volume_fp", 0) or 0)
        title  = raw.get("title", "")
        player = self._parse_finishing_player_name(title, threshold)

        return {
            "ticker": raw.get("ticker", ""),
            "event_ticker": raw.get("event_ticker", ""),
            "title": title,
            "player": player,
            "threshold": threshold,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "kalshi_prob": yes_ask / 100,
            "volume": volume,
            "last_price": round(float(raw.get("last_price_dollars", 0) or 0) * 100),
        }

    def _parse_finishing_player_name(self, title: str, threshold: int) -> Optional[str]:
        """
        'Cadillac Championship: Will Kristoffer Reitan finish top 20?' → 'Kristoffer Reitan'
        """
        m = re.match(r".*Will\s+(.+?)\s+finish\s+top\s+\d+", title, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return None

    def _parse_player_name(self, title: str) -> Optional[str]:
        """
        Extract player name from market title.
        "Will Scottie Scheffler win the 2026 PGA Championship?" → "Scottie Scheffler"
        "Scottie Scheffler to win 2026 PGA Championship" → "Scottie Scheffler"
        """
        # Pattern: "Will [Name] win..." or "[Name] to win..."
        m = re.match(r"^Will\s+(.+?)\s+win\b", title, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.match(r"^(.+?)\s+(?:to win|wins?)\b", title, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # Fallback: first two words
        words = title.split()
        if len(words) >= 2:
            return " ".join(words[:2])
        return title.strip() or None
