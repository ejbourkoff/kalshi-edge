"""
Kalshi API client for NBA prediction markets.
Base URL: https://api.elections.kalshi.com/trade-api/v2
Prices are in USD dollars (0.44 = 44¢), not cents.
Read endpoints are public; auth is only needed for trading.
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

# All NBA prop/game event series tickers
NBA_SERIES = {
    # Single-stat props
    "KXNBAPTS":         "points",
    "KXNBA3PT":         "threes",
    "KXNBAAST":         "assists",
    "KXNBAREB":         "rebounds",
    "KXNBABLK":         "blocks",
    "KXNBASTL":         "steals",
    # Combo stat props
    "KXNBAPRA":         "pra",       # points+rebounds+assists
    "KXNBAPA":          "pa",        # points+assists
    "KXNBAPR":          "pr",        # points+rebounds
    "KXNBARA":          "ra",        # rebounds+assists
    "KXNBA2D":          "double_double",
    "KXNBA3D":          "triple_double",
    # Game-level
    "KXNBASERIES":      "series",
    "KXNBAGAME":        "game",
    # Futures
    "KXNBA":            "champ",     # NBA Championship
    "KXNBAEAST":        "conf",      # Eastern Conference
    "KXNBAWEST":        "conf",      # Western Conference
    "KXTEAMSINNBAF":    "finals",    # Teams in Finals
    # Pre-packaged combos
    "KXNBAPREPACK2ML":  "combo",
    "KXNBAPREPACK3ML":  "combo",
    "KXMVENBAMULTIGAMEEXTENDED": "mve",
    "KXMVENBASINGLEGAME":        "mve",
    # Playoff-specific props
    "KXNBAPLAYOFFPTS":  "points",
}

# Kalshi ticker abbreviation → full team name
TEAM_TICKER_MAP = {
    "LAL": "Los Angeles Lakers",
    "LAC": "Los Angeles Clippers",
    "BOS": "Boston Celtics",
    "GSW": "Golden State Warriors",
    "MIA": "Miami Heat",
    "NYK": "New York Knicks",
    "DEN": "Denver Nuggets",
    "MIN": "Minnesota Timberwolves",
    "OKC": "Oklahoma City Thunder",
    "CLE": "Cleveland Cavaliers",
    "PHI": "Philadelphia 76ers",
    "ORL": "Orlando Magic",
    "DET": "Detroit Pistons",
    "TOR": "Toronto Raptors",
    "SAS": "San Antonio Spurs",
    "MIL": "Milwaukee Bucks",
    "IND": "Indiana Pacers",
    "ATL": "Atlanta Hawks",
    "CHI": "Chicago Bulls",
    "CHA": "Charlotte Hornets",
    "SAC": "Sacramento Kings",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "UTA": "Utah Jazz",
    "DAL": "Dallas Mavericks",
    "HOU": "Houston Rockets",
    "MEM": "Memphis Grizzlies",
    "NOP": "New Orleans Pelicans",
    "BKN": "Brooklyn Nets",
    "WAS": "Washington Wizards",
}


class KalshiClient:
    def __init__(self):
        self.base_url = BASE_URL
        self.api_key = config.kalshi_api_key
        self._private_key = self._load_key() if config.kalshi_api_key else None
        self._session = requests.Session()
        self._session.headers.update({"accept": "application/json"})

    def _load_key(self):
        try:
            with open(config.kalshi_private_key_path, "rb") as f:
                return serialization.load_pem_private_key(f.read(), password=None)
        except Exception:
            return None

    def _auth_headers(self, method: str, path: str) -> dict:
        if not self._private_key:
            return {}
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode()
        sig = self._private_key.sign(
            msg,
            asym_padding.PSS(
                mgf=asym_padding.MGF1(hashes.SHA256()),
                salt_length=asym_padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
        }

    def _get(self, endpoint: str, params: dict = None, auth: bool = False, retries: int = 3) -> dict:
        path = f"/trade-api/v2{endpoint}"
        headers = self._auth_headers("GET", path) if auth else {}
        for attempt in range(retries):
            resp = self._session.get(
                f"{self.base_url}{endpoint}",
                headers=headers,
                params=params or {},
                timeout=10,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s, 4s
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        resp.raise_for_status()
        return {}

    # ── Market data (public, no auth) ────────────────────────────────────────

    def get_nba_events(self) -> list[dict]:
        """Return all open NBA events across all prop/game series."""
        events = []
        for series_ticker in NBA_SERIES:
            data = self._get("/events", params={
                "series_ticker": series_ticker,
                "status": "open",
                "limit": 50,
            })
            events.extend(data.get("events", []))
        return events

    def get_markets_for_event(self, event_ticker: str) -> list[dict]:
        data = self._get("/markets", params={
            "event_ticker": event_ticker,
            "limit": 100,
        })
        return data.get("markets", [])

    def get_all_nba_markets(self) -> list[dict]:
        """Fetch every open NBA market across all series."""
        events = self.get_nba_events()
        all_markets = []
        for ev in events:
            time.sleep(0.35)  # stay under rate limit
            mkts = self.get_markets_for_event(ev["event_ticker"])
            for m in mkts:
                m["_event_title"] = ev.get("title", "")
            all_markets.extend(mkts)
        return all_markets

    def parse_market(self, raw: dict) -> Optional[dict]:
        """
        Normalize a raw Kalshi market to a consistent shape.
        Returns None if the market has no live price.
        """
        yes_ask_usd = raw.get("yes_ask_dollars")
        no_ask_usd = raw.get("no_ask_dollars")
        yes_bid_usd = raw.get("yes_bid_dollars")

        # Skip markets with no liquidity
        try:
            yes_ask = round(float(yes_ask_usd) * 100)  # convert to cents
            no_ask = round(float(no_ask_usd or 0) * 100)
        except (TypeError, ValueError):
            return None

        if yes_ask == 0:
            return None

        # Broken/illiquid market: ask prices sum way above $1 (no real orderbook)
        if no_ask > 0 and (yes_ask + no_ask) > 115:
            return None

        ticker = raw.get("ticker", "")
        event_ticker = raw.get("event_ticker", "")
        title = raw.get("title", "")
        stat_type = self._infer_stat(ticker)
        player, threshold = self._parse_title(title, stat_type)
        yes_team = self._extract_yes_team(ticker, stat_type)
        game_date = self._parse_game_date(event_ticker)

        return {
            "ticker": ticker,
            "event_ticker": event_ticker,
            "title": title,
            "yes_ask": yes_ask,
            "no_ask": no_ask,
            "yes_bid": round(float(yes_bid_usd or 0) * 100),
            "kalshi_prob": yes_ask / 100,
            "stat_type": stat_type,
            "player": player,
            "threshold": threshold,
            "yes_team": yes_team,
            "game_date": game_date,       # "May 4" or None
            "volume": float(raw.get("volume_fp", 0) or 0),
            "close_time": raw.get("close_time", ""),
        }

    def get_all_parsed_markets(self) -> list[dict]:
        raw = self.get_all_nba_markets()
        parsed = [self.parse_market(m) for m in raw]
        return [p for p in parsed if p is not None]

    # ── Authenticated endpoints ───────────────────────────────────────────────

    def get_balance(self) -> float:
        data = self._get("/portfolio/balance", auth=True)
        return data.get("balance", 0) / 100

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _infer_stat(self, ticker: str) -> str:
        for prefix, stat in NBA_SERIES.items():
            if ticker.startswith(prefix):
                return stat
        return "unknown"

    def _parse_game_date(self, event_ticker: str) -> Optional[str]:
        """
        Extract game date from event ticker.
        'KXNBAPTS-26MAY04PHINYK' → 'May 4'
        'KXNBASERIES-26LALOKCR2' → None (series, no single date)
        """
        m = re.search(r'26([A-Z]{3})(\d{2})', event_ticker)
        if not m:
            return None
        month_abbr = m.group(1).capitalize()
        day = int(m.group(2))
        return f"{month_abbr} {day}"

    def _extract_yes_team(self, ticker: str, stat_type: str) -> Optional[str]:
        """
        For game/series markets, extract which team's win the YES side represents.
        Ticker format: KXNBASERIES-26LALOKCR2-OKC → YES = OKC wins
        """
        if stat_type not in ("series", "game"):
            return None
        parts = ticker.split("-")
        if len(parts) >= 3:
            abbrev = parts[-1].upper()
            return TEAM_TICKER_MAP.get(abbrev)
        return None

    def _parse_title(self, title: str, stat_type: str) -> tuple[Optional[str], Optional[float]]:
        """
        Extract player name and numeric threshold from market title.
        'Cade Cunningham: 35+ points' → ('Cade Cunningham', 35.0)
        'Oklahoma City wins Game 3' → (None, None)
        """
        if stat_type in ("series", "game", "unknown"):
            return None, None

        # Pattern: "Player Name: N+ stat"
        m = re.match(r"^(.+?):\s*(\d+\.?\d*)\+?", title)
        if m:
            return m.group(1).strip(), float(m.group(2))

        return None, None
