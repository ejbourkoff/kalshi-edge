import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    kalshi_api_key: str = field(default_factory=lambda: os.getenv("KALSHI_API_KEY", ""))
    kalshi_private_key_path: str = field(default_factory=lambda: os.getenv("KALSHI_PRIVATE_KEY_PATH", "kalshi_private_key.pem"))

    odds_api_key: str = field(default_factory=lambda: os.getenv("ODDS_API_KEY", ""))
    odds_base_url: str = "https://api.the-odds-api.com/v4"

    bankroll: float = field(default_factory=lambda: float(os.getenv("BANKROLL", "1000")))

    # Edge thresholds
    min_edge_to_show: float = 0.02     # 2%+ shown in output
    min_edge_to_bet: float = 0.03      # 3%+ actionable (golf outrights — lower than props)

    # Kelly settings
    kelly_fraction: float = 0.20       # 20% Kelly — golf is volatile, be conservative
    max_bet_pct: float = 0.025         # 2.5% of bankroll per player max

    # Kalshi fee
    fee_rate: float = 0.02

    # Cache
    cache_dir: str = ".cache"
    cache_ttl_hours: int = 1           # Golf odds move fast, cache shorter

    # Min volume to show/bet
    min_volume_show: float = 50.0
    min_volume_bet: float = 200.0

    # Min books confirming an edge (3+ = much stronger consensus)
    min_books: int = 3

config = Config()
