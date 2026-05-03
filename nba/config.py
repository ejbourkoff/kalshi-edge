import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_private_key_path: str = field(default_factory=lambda: os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem"))
    kalshi_base_url: str = "https://api.elections.kalshi.com/trade-api/v2"

    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    odds_base_url: str = "https://api.the-odds-api.com/v4"

    bankroll: float = field(default_factory=lambda: float(os.getenv("BANKROLL", "1000")))

    # Edge thresholds
    min_edge_to_show: float = 0.02     # 2%+ shown in output
    min_edge_to_bet: float = 0.04      # 4%+ gets a BET recommendation

    # Kelly settings
    kelly_fraction: float = 0.25       # quarter-Kelly (safer)
    max_bet_pct: float = 0.03          # hard cap: 3% of bankroll per trade

    # Kalshi fee
    fee_rate: float = 0.02

    # NBA
    nba_season: str = "2025-26"
    nba_season_year: int = 2026

    # Cache (avoid hammering NBA API)
    cache_dir: str = ".cache"
    cache_ttl_hours: int = 6

config = Config()
