"""
PGA Tour edge scanner.

Core algorithm:
  1. Fetch all open Kalshi PGA outright-winner markets
  2. Fetch bookmaker consensus (6 books) with vig removed via overround normalization
  3. Match players by name (fuzzy match for spelling differences)
  4. Calculate EV and Kelly for every matched player
  5. Confidence: HIGH if 2+ books confirm AND gap >= 2pp
  6. Internal consistency: sum of all Kalshi YES prices should ≈ 1.0 — find
     players priced above/below their fair share of the book

Why this works:
  Kalshi's golf markets are set by retail bettors who follow media narratives.
  Sharp sportsbooks reprice faster on late news (withdrawals, weather, course
  setup changes). The gap between Kalshi and books is the edge.
"""
from itertools import combinations
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from kalshi_client import KalshiPGAClient

from kalshi_client import KalshiPGAClient
from odds_client import OddsPGAClient
from ev_engine import edge_summary, calculate_ev_yes, calculate_ev_no, kelly_bet
from config import config


def name_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def match_player(kalshi_name: str, sb_players: list[dict], threshold: float = 0.78) -> Optional[dict]:
    """
    Fuzzy-match a Kalshi player name to a sportsbook player entry.
    Prefers exact last-name match, then full-name similarity.
    """
    k_last = kalshi_name.split()[-1].lower()

    # 1. Exact last-name match
    for p in sb_players:
        if p["player"].split()[-1].lower() == k_last and len(k_last) > 3:
            return p

    # 2. Fuzzy full-name match
    best, best_score = None, 0.0
    for p in sb_players:
        score = name_similarity(kalshi_name, p["player"])
        if score > best_score:
            best, best_score = p, score

    return best if best_score >= threshold else None


def _confidence(sportsbook_anchored: bool, gap_pp: float, num_books: int) -> str:
    if sportsbook_anchored and gap_pp >= 2.0 and num_books >= 2:
        return "HIGH"
    if sportsbook_anchored and gap_pp >= 1.0:
        return "MEDIUM"
    return "LOW"


def scan_tournament(
    kalshi_markets: list[dict],
    sb_players: list[dict],
    tournament_name: str,
) -> list[dict]:
    """
    Scan one tournament: compare each Kalshi player market against bookmaker consensus.
    Returns edge list sorted by best_ev descending.
    """
    results = []

    for market in kalshi_markets:
        player = market.get("player")
        if not player:
            continue
        if market["volume"] < config.min_volume_show:
            continue

        sb_match = match_player(player, sb_players)
        if not sb_match:
            continue

        sb_prob   = sb_match["true_prob"]
        num_books = sb_match["num_books"]
        sb_name   = sb_match["player"]

        yes_ask = market["yes_ask"]
        no_ask  = market["no_ask"] or (100 - yes_ask)
        kalshi_implied = yes_ask / 100
        gap_pp = abs(sb_prob - kalshi_implied) * 100

        # Need at least 1pp gap to show
        if gap_pp < 1.0:
            continue

        conf = _confidence(True, gap_pp, num_books)

        summary = edge_summary(
            yes_ask=yes_ask,
            no_ask=no_ask,
            true_prob=sb_prob,
            player=player,
            tournament=tournament_name,
            sb_prob=sb_prob,
            num_books=num_books,
            confidence=conf,
            gap_pp=gap_pp,
            volume=market["volume"],
        )

        if summary["best_ev"] < config.min_edge_to_show:
            continue

        results.append({
            **summary,
            "ticker": market["ticker"],
            "volume": market["volume"],
            "sb_player_name": sb_name,
            "kalshi_name": player,
            "market_type": "outright",
        })

    return sorted(results, key=lambda x: -x["best_ev"])


def scan_internal_consistency(kalshi_markets: list[dict], tournament_name: str) -> list[dict]:
    """
    In a tournament, exactly one player wins. The true probabilities must sum to 1.0.
    Kalshi's YES prices have overround — but individual players can still be mispriced
    relative to each other.

    Key check: if Kalshi's total overround is X%, then each player's YES price represents
    their Kalshi implied probability. If one player's implied probability significantly
    exceeds what the field structure implies, their NO side has value.

    We flag players where:
      - Kalshi implied prob is much higher than their rank-implied fair share
      - (i.e., Kalshi bettors are piling into a hot name beyond what the market supports)

    We also find structural inconsistencies: if player A is priced at 15¢ and B at 10¢
    but books agree they're equal-likelihood, the cheaper one is a buy.
    """
    results = []

    # Only look at markets with some volume
    liquid = [m for m in kalshi_markets if m["volume"] >= config.min_volume_show]
    if len(liquid) < 5:
        return []

    # Total Kalshi overround
    total_yes = sum(m["yes_ask"] for m in liquid)

    for market in liquid:
        yes_ask = market["yes_ask"]
        no_ask  = market["no_ask"] or (100 - yes_ask)
        player  = market.get("player", "")

        # Each player's "fair share" of a 100-point probability budget
        fair_share = yes_ask / total_yes  # their fraction of the total book
        kalshi_implied = yes_ask / 100

        # If their Kalshi implied is much higher than their fair share suggests,
        # the field is underpriced relative to them → consider NO on this player
        # This is only useful to flag for context — not a tradable edge without more data

    return results  # Structural consistency returns empty for now — rely on sb matching


def scan_parlays(actionable_edges: list[dict]) -> list[dict]:
    """
    Find optimal 2-player parlay combinations from HIGH confidence edges in the SAME tournament.

    Golf parlays (on Kalshi as independent bets):
      - Bet YES on player A (true prob 12%) AND YES on player B (true prob 9%)
      - These are independent events if they're different players — but NOT truly
        independent since only ONE player can win. However, both bets are positive EV.
      - For "both win" you'd need them to be in different tournaments.

    So we build two types:
      1. SAME tournament, different players: show as portfolio bets (both YES, both +EV)
      2. DIFFERENT tournaments: true independence — show as parlay with combined probability
    """
    combos = []

    # Group by tournament
    from collections import defaultdict
    by_tournament = defaultdict(list)
    for e in actionable_edges:
        by_tournament[e["tournament"]].append(e)

    # Cross-tournament parlays (truly independent)
    tournaments = list(by_tournament.keys())
    if len(tournaments) >= 2:
        for t1, t2 in combinations(tournaments, 2):
            legs1 = by_tournament[t1][:3]  # top 3 from each
            legs2 = by_tournament[t2][:3]
            for l1 in legs1:
                for l2 in legs2:
                    p1 = l1["true_prob"] if l1["best_side"] == "YES" else 1 - l1["true_prob"]
                    p2 = l2["true_prob"] if l2["best_side"] == "YES" else 1 - l2["true_prob"]
                    combined_prob = p1 * p2
                    pr1 = l1["yes_ask"] if l1["best_side"] == "YES" else l1["no_ask"]
                    pr2 = l2["yes_ask"] if l2["best_side"] == "YES" else l2["no_ask"]
                    combined_cost = (pr1 / 100) * (pr2 / 100)
                    parlay_ev = combined_prob - combined_cost
                    if parlay_ev > 0.001:
                        combos.append({
                            "type": "2-tournament parlay",
                            "legs": 2,
                            "combined_true_prob": round(combined_prob, 5),
                            "combined_kalshi_price": round(combined_cost, 5),
                            "parlay_ev": round(parlay_ev, 4),
                            "recommended_bets": [
                                f"{l1['player']} ({l1['tournament']}) → {l1['best_side']} @ {pr1}¢",
                                f"{l2['player']} ({l2['tournament']}) → {l2['best_side']} @ {pr2}¢",
                            ],
                            "total_kelly_exposure": round(l1["bet_size_$"] + l2["bet_size_$"], 2),
                            "recommendation": (
                                f"Different tournaments → truly independent. "
                                f"True prob {combined_prob*100:.2f}% vs Kalshi cost {combined_cost*100:.2f}%."
                            ),
                        })

    # Same-tournament portfolio (both positive EV, place independently)
    for tournament, edges in by_tournament.items():
        yes_edges = [e for e in edges if e["best_side"] == "YES"]
        if len(yes_edges) >= 2:
            # Show top 2-3 as a portfolio
            for n in (2, 3):
                combo = yes_edges[:n]
                if len(combo) < n:
                    continue
                total_bet = sum(e["bet_size_$"] for e in combo)
                sum_ev = sum(e["best_ev"] for e in combo)
                combos.append({
                    "type": f"{n}-player portfolio ({tournament.split()[-1]})",
                    "legs": n,
                    "combined_true_prob": round(sum(e["true_prob"] for e in combo), 4),
                    "combined_kalshi_price": round(sum(e["yes_ask"] / 100 for e in combo), 4),
                    "parlay_ev": round(sum_ev / n, 4),  # avg EV
                    "recommended_bets": [
                        f"{e['player']} → YES @ {e['yes_ask']}¢ (books {e['sb_prob']*100:.1f}%)"
                        for e in combo
                    ],
                    "total_kelly_exposure": round(total_bet, 2),
                    "recommendation": (
                        f"Same tournament — players are mutually exclusive but individually +EV. "
                        f"Place each bet independently. Total Kelly exposure ${total_bet:.0f}."
                    ),
                })

    return sorted(combos, key=lambda x: -x["parlay_ev"])[:10]


def run_scan() -> tuple[list[dict], list[dict]]:
    """
    Full PGA scan. Returns (all_edges, parlays).

    For each open tournament on Kalshi:
      1. Fetch all player outright markets
      2. Match to sportsbook odds
      3. Find edges
    Also scans finishing-position markets (top 5/10/20) using live ESPN leaderboard.
    """
    print("Fetching Kalshi PGA markets...")
    kalshi = KalshiPGAClient()
    all_markets = kalshi.get_all_parsed_markets()
    print(f"  Found {len(all_markets)} open PGA player markets")

    # Group by tournament
    from collections import defaultdict
    by_tournament: dict[str, list] = defaultdict(list)
    by_odds_key: dict[str, str] = {}  # tournament → odds_key
    for m in all_markets:
        t = m.get("tournament", "Unknown")
        by_tournament[t].append(m)
        ok = m.get("odds_key")
        if ok:
            by_odds_key[t] = ok

    print("Fetching sportsbook consensus odds...")
    odds_client = OddsPGAClient()

    all_edges = []
    for tournament, markets in by_tournament.items():
        odds_key = by_odds_key.get(tournament)
        if not odds_key:
            print(f"  {tournament}: no Odds API key mapped — skipping sportsbook anchoring")
            continue

        sb_players = odds_client.get_tournament_outrights(odds_key)
        if not sb_players:
            print(f"  {tournament}: no sportsbook data available")
            continue

        print(f"  {tournament}: {len(markets)} Kalshi markets, {len(sb_players)} book players")

        edges = scan_tournament(markets, sb_players, tournament)
        all_edges.extend(edges)

    # ── Finishing position scan (live in-tournament markets) ────────────────
    finishing_edges = _scan_finishing_positions(kalshi)
    all_edges.extend(finishing_edges)

    # Sort: HIGH first, then by EV
    def sort_key(e):
        tier = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(e.get("confidence", "LOW"), 2)
        return (tier, -e["best_ev"])

    all_edges = sorted(all_edges, key=sort_key)

    high = [e for e in all_edges if e.get("confidence") == "HIGH"]
    med  = [e for e in all_edges if e.get("confidence") == "MEDIUM"]
    actionable = [e for e in all_edges if e.get("actionable")]

    print(f"\nResults:")
    print(f"  HIGH confidence: {len(high)} edges ({len(actionable)} actionable)")
    print(f"  MEDIUM confidence: {len(med)} edges")

    parlays = scan_parlays(actionable)
    if parlays:
        print(f"  Parlay/portfolio combos: {len(parlays)}")
    print()

    return all_edges, parlays


_ESPN_TO_KALSHI_SUFFIX = {
    "cadillac championship": "CAC26",
    "pga championship": "PGC26",
    "masters tournament": "MAST26",
    "u.s. open": "USOP26",
    "us open": "USOP26",
    "the open championship": "BRIT26",
    "the players championship": "PLAY26",
    "wells fargo championship": "WELLS26",
    "memorial tournament": "MEMO26",
    "travelers championship": "TRAX26",
    "canadian open": "CANADI26",
    "scottish open": "GENSC26",
}


def _map_espn_to_kalshi_suffix(event_name: str) -> Optional[str]:
    """Map ESPN event name (e.g. 'Cadillac Championship') to Kalshi suffix (e.g. 'CAC26')."""
    key = event_name.lower().strip()
    # Exact match
    if key in _ESPN_TO_KALSHI_SUFFIX:
        return _ESPN_TO_KALSHI_SUFFIX[key]
    # Partial match
    for k, v in _ESPN_TO_KALSHI_SUFFIX.items():
        if k in key or key in k:
            return v
    return None


def _scan_finishing_positions(kalshi: "KalshiPGAClient") -> list[dict]:
    """
    Scan live finishing-position markets (top 5/10/20) using ESPN leaderboard.
    Only runs when a tournament is active/suspended in a live round.
    Matches the live ESPN event to the correct Kalshi event suffix.
    """
    try:
        from espn_client import get_live_leaderboard
        from finishing_scanner import scan_finishing_positions
    except ImportError as e:
        print(f"  [Finishing scan] Import error: {e}")
        return []

    espn_data = get_live_leaderboard()
    if not espn_data:
        return []

    status = espn_data.get("status", "pre")
    event_name = espn_data.get("event_name", "")
    if status == "pre":
        print(f"  [Finishing scan] {event_name} not yet started — skipping")
        return []
    if status == "post":
        print(f"  [Finishing scan] {event_name} complete — skipping")
        return []

    # Map the live ESPN event to the correct Kalshi event suffix
    suffix = _map_espn_to_kalshi_suffix(event_name)
    if not suffix:
        print(f"  [Finishing scan] No Kalshi suffix mapped for '{event_name}'")
        return []

    print(f"Fetching finishing position markets for {event_name} ({suffix})...")
    finishing_markets = kalshi.get_finishing_markets(suffix)
    market_count = sum(len(v) for v in finishing_markets.values())
    print(f"  {suffix}: {market_count} finishing position markets")

    edges = scan_finishing_positions(finishing_markets, espn_data)
    print(f"  {suffix}: {len(edges)} finishing edges found")
    return edges
