"""
Daily scanner — orchestrates all data fetching, probability estimation,
and EV calculation. Returns ranked list of actionable trades.

Profit-maximizing principles:
  1. Sportsbook consensus is used as-is (no model dilution) — books are more efficient than our model.
  2. Internal consistency check catches mathematical mispricings across prop thresholds.
  3. Only HIGH confidence edges are actionable (sportsbook-anchored or internal mispricing).
  4. Volume filter skips thin markets that can't be filled cleanly.
  5. Kelly fraction is scaled by confidence to reduce variance.

Confidence tiers:
  HIGH   — sportsbook-anchored with gap > 3pp, OR mathematical internal mispricing.
  MEDIUM — sportsbook-anchored but small gap (1-3pp), OR model-only with huge divergence (>20pp).
  LOW    — model-only, moderate divergence. Watch only — do not bet.
"""
from collections import defaultdict
from datetime import date
from typing import Optional

from kalshi_client import KalshiClient
from odds_client import OddsClient
from models.ev_engine import edge_summary, calculate_ev, calculate_ev_no
from models.prop_model import estimate_prop_probability
from models.game_model import estimate_game_probability
from config import config

TEAM_ALIASES = {
    "Los Angeles Lakers": ["Lakers", "LAL", "Los Angeles L"],
    "Los Angeles Clippers": ["Clippers", "LAC"],
    "Golden State Warriors": ["Warriors", "GSW"],
    "Boston Celtics": ["Celtics", "BOS"],
    "Miami Heat": ["Heat", "MIA"],
    "New York Knicks": ["Knicks", "NYK"],
    "Denver Nuggets": ["Nuggets", "DEN"],
    "Minnesota Timberwolves": ["Timberwolves", "MIN"],
    "Oklahoma City Thunder": ["Thunder", "OKC"],
    "Cleveland Cavaliers": ["Cavaliers", "CLE"],
    "Philadelphia 76ers": ["76ers", "PHI", "Philadelphia"],
    "Orlando Magic": ["Magic", "ORL", "Orlando"],
    "Detroit Pistons": ["Pistons", "DET", "Detroit"],
    "Toronto Raptors": ["Raptors", "TOR", "Toronto"],
    "San Antonio Spurs": ["Spurs", "SAS", "San Antonio"],
    "Milwaukee Bucks": ["Bucks", "MIL"],
    "Indiana Pacers": ["Pacers", "IND"],
    "Atlanta Hawks": ["Hawks", "ATL"],
    "Chicago Bulls": ["Bulls", "CHI"],
    "Charlotte Hornets": ["Hornets", "CHA"],
}

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Minimum volume ($) to show a market. Prevents unfillable thin markets.
MIN_VOLUME_BET = 200.0    # must have at least $200 volume to be actionable
MIN_VOLUME_SHOW = 50.0    # must have at least $50 volume to appear at all

# Kelly fraction by confidence tier
KELLY_FRACTION = {
    "HIGH": 0.25,   # quarter Kelly — confirmed edge
    "MEDIUM": 0.10, # 10% Kelly — uncertain, watch
    "LOW": 0.0,     # don't bet low confidence
}


def _is_future_or_today(game_date: Optional[str]) -> bool:
    """Returns True if game_date is today or future (keep it). None = futures/series (keep)."""
    if game_date is None:
        return True
    try:
        parts = game_date.split()
        month = MONTH_MAP.get(parts[0])
        day = int(parts[1])
        if month is None:
            return True
        game_dt = date(date.today().year, month, day)
        return game_dt >= date.today()
    except Exception:
        return True


def _confidence_tier(sportsbook_anchored: bool, gap_pp: float, num_books: int = 0) -> str:
    """
    HIGH  = sportsbook says so AND the gap is real (≥3pp).
    MEDIUM = anchored with small gap, or model-only huge divergence.
    LOW   = model-only moderate divergence. Watch only.
    """
    if sportsbook_anchored and gap_pp >= 3.0:
        return "HIGH"
    if sportsbook_anchored and gap_pp >= 1.0:
        return "MEDIUM"
    if not sportsbook_anchored and gap_pp >= 20.0:
        return "MEDIUM"
    return "LOW"


def _kelly_bet(true_prob: float, price_cents: float, confidence: str) -> float:
    """Kelly bet sized by confidence tier."""
    from config import config as cfg
    fraction = KELLY_FRACTION.get(confidence, 0.0)
    if fraction == 0:
        return 0.0
    price = min(max(price_cents, 1), 99) / 100
    b = (1 - price) / price
    if b == 0:
        return 0.0
    kelly = max(0, (b * true_prob - (1 - true_prob)) / b)
    capped = min(kelly * fraction, cfg.max_bet_pct)
    return round(capped * cfg.bankroll, 2)


def _team_in_title(team_full: str, title: str) -> bool:
    t = title.upper()
    for alias in TEAM_ALIASES.get(team_full, [team_full.split()[-1]]):
        if alias.upper() in t:
            return True
    return False


# ── Internal consistency check ────────────────────────────────────────────────

def scan_internal_consistency(kalshi_markets: list[dict]) -> list[dict]:
    """
    Mathematical mispricing: for the same player+stat, a higher threshold
    MUST have a lower YES probability. If YES@25pts > YES@20pts, something
    is wrong and we have a near-riskless edge on YES@20pts.

    These are HIGH confidence by construction — no model needed.
    """
    groups = defaultdict(list)
    for m in kalshi_markets:
        if m["stat_type"] in ("series", "game", "unknown"):
            continue
        if not _is_future_or_today(m.get("game_date")):
            continue
        player = m.get("player")
        stat = m.get("stat_type")
        threshold = m.get("threshold")
        if not player or not stat or threshold is None:
            continue
        if m["volume"] < MIN_VOLUME_SHOW:
            continue
        groups[(player, stat)].append(m)

    results = []
    for (player, stat), markets in groups.items():
        if len(markets) < 2:
            continue
        markets_sorted = sorted(markets, key=lambda m: m["threshold"])

        for i in range(len(markets_sorted) - 1):
            lo = markets_sorted[i]    # lower threshold
            hi = markets_sorted[i + 1]  # higher threshold

            # YES at lower threshold must be >= YES at higher threshold (monotonicity)
            # Violation: lo["yes_ask"] < hi["yes_ask"] — lower threshold is priced cheaper
            if lo["yes_ask"] >= hi["yes_ask"]:
                continue

            gap_pp = (hi["yes_ask"] - lo["yes_ask"]) / 100 * 100  # in pp

            # The fair value for lo's YES is at least hi's YES price
            # → lo is underpriced, bet YES on lo
            true_prob = hi["yes_ask"] / 100
            kalshi_implied = lo["yes_ask"] / 100

            ev_yes = calculate_ev(lo["yes_ask"], true_prob)
            if ev_yes < config.min_edge_to_show:
                continue

            bet = _kelly_bet(true_prob, lo["yes_ask"], "HIGH")
            actionable = (
                ev_yes >= config.min_edge_to_bet
                and lo["volume"] >= MIN_VOLUME_BET
            )

            reasoning = (
                f"Inconsistent pricing: YES@{hi['threshold']:.0f}{stat[:3]} = {hi['yes_ask']}¢ "
                f"but YES@{lo['threshold']:.0f}{stat[:3]} = {lo['yes_ask']}¢ — "
                f"higher threshold can't be more likely → BET YES@{lo['threshold']:.0f} @ {lo['yes_ask']}¢"
            )

            results.append({
                "label": lo["title"],
                "kalshi_price": lo["yes_ask"],
                "implied_prob": round(kalshi_implied, 4),
                "true_prob": round(true_prob, 4),
                "raw_edge": round(true_prob - kalshi_implied, 4),
                "ev_yes": round(ev_yes, 4),
                "ev_no": 0.0,
                "best_side": "YES",
                "best_ev": round(ev_yes, 4),
                "bet_size_$": bet,
                "actionable": actionable,
                "market_type": "prop",
                "confidence": "HIGH",
                "sportsbook_anchored": False,
                "consistency_mispricing": True,
                "sb_prob": None,
                "kalshi_implied": round(kalshi_implied, 4),
                "gap_pp": round(gap_pp, 1),
                "num_books": 0,
                "reasoning": reasoning,
                "player": player,
                "stat": stat,
                "threshold": lo["threshold"],
                "model_proj": None,
                "model_prob": None,
                "volume": lo["volume"],
                "ticker": lo["ticker"],
                "game_date": lo.get("game_date"),
            })

    return results


# ── Prop scanner ──────────────────────────────────────────────────────────────

def scan_props(kalshi_markets: list[dict], sportsbook_props: list[dict]) -> list[dict]:
    """
    Scan prop markets for EV. When sportsbook line is available, use it
    as the true probability directly (no model dilution — books are efficient).
    Model is only used when no sportsbook anchor exists.
    """
    results = []

    # Index sportsbook props: (player_lower, stat, line) → entry
    sb_index = {}
    for sp in sportsbook_props:
        key = (sp["player"].lower(), sp["stat"], sp["line"])
        sb_index[key] = sp

    for market in kalshi_markets:
        if market["stat_type"] in ("series", "game", "unknown"):
            continue
        if not _is_future_or_today(market.get("game_date")):
            continue
        if market["volume"] < MIN_VOLUME_SHOW:
            continue

        player = market.get("player")
        stat = market.get("stat_type")
        threshold = market.get("threshold")
        game_date = market.get("game_date")

        if not player or not stat or threshold is None:
            continue

        # Find matching sportsbook line (allow ±0.5/1.0 line difference)
        sb_entry = None
        for offset in [0, 0.5, -0.5, 1.0, -1.0]:
            key = (player.lower(), stat, threshold + offset)
            if key in sb_index:
                sb_entry = sb_index[key]
                break

        sb_prob = sb_entry["sportsbook_over_prob"] if sb_entry else None
        num_books = sb_entry.get("num_books", 1) if sb_entry else 0

        if sb_prob is not None:
            # USE SPORTSBOOK DIRECTLY — it's more efficient than our model.
            # The sportsbook already prices in recent news, injuries, lineups.
            true_prob = sb_prob
            model_proj = None
            model_prob = None
        else:
            # Fallback: use our model only (high uncertainty)
            prob_result = estimate_prop_probability(player, stat, threshold)
            true_prob = prob_result["estimated_true_prob"]
            model_proj = prob_result["model_projection"]
            model_prob = prob_result["model_prob"]
            if true_prob is None:
                continue

        kalshi_implied = market["yes_ask"] / 100
        sportsbook_anchored = sb_prob is not None
        gap_pp = abs(true_prob - kalshi_implied) * 100

        # Strict filtering: sportsbook needs >1pp gap, model-only needs >20pp
        if sportsbook_anchored and gap_pp < 1.0:
            continue
        if not sportsbook_anchored and gap_pp < 20.0:
            continue

        confidence = _confidence_tier(sportsbook_anchored, gap_pp, num_books)
        summary = edge_summary(
            market["yes_ask"], true_prob,
            label=market["title"],
            kalshi_no_price=market.get("no_ask"),
        )
        if summary["best_ev"] < config.min_edge_to_show:
            continue

        direction = summary["best_side"]
        no_ask = market.get("no_ask") or (100 - market["yes_ask"])
        bet_price = market["yes_ask"] if direction == "YES" else no_ask
        bet = _kelly_bet(
            true_prob if direction == "YES" else 1 - true_prob,
            bet_price,
            confidence,
        )

        # Override bet_size with confidence-adjusted Kelly
        summary["bet_size_$"] = bet

        actionable = (
            confidence == "HIGH"
            and summary["best_ev"] >= config.min_edge_to_bet
            and market["volume"] >= MIN_VOLUME_BET
        )

        if sportsbook_anchored:
            reasoning = (
                f"{num_books} book{'s' if num_books != 1 else ''} no-vig: {sb_prob*100:.0f}% | "
                f"Kalshi: {market['yes_ask']}¢ YES → {gap_pp:.0f}pp gap → BET {direction} @ {bet_price}¢"
            )
        else:
            proj_str = f"{model_proj:.1f}" if model_proj else "?"
            reasoning = (
                f"Model only — proj {proj_str} → {true_prob*100:.0f}% | "
                f"Kalshi: {market['yes_ask']}¢ → {gap_pp:.0f}pp divergence (unconfirmed)"
            )

        results.append({
            **summary,
            "actionable": actionable,
            "market_type": "prop",
            "confidence": confidence,
            "sportsbook_anchored": sportsbook_anchored,
            "sb_prob": round(sb_prob, 4) if sb_prob else None,
            "kalshi_implied": round(kalshi_implied, 4),
            "gap_pp": round(gap_pp, 1),
            "num_books": num_books,
            "reasoning": reasoning,
            "player": player,
            "stat": stat,
            "threshold": threshold,
            "model_proj": model_proj,
            "model_prob": round(model_prob, 4) if model_prob else None,
            "volume": market["volume"],
            "ticker": market["ticker"],
            "game_date": game_date,
        })

    return results


# ── Game/series scanner ───────────────────────────────────────────────────────

def _parse_sb_date(commence_time: str) -> Optional[date]:
    """Parse sportsbook commence_time '2026-05-03T19:40:00Z' → date(2026, 5, 3)."""
    try:
        return date.fromisoformat(commence_time[:10])
    except Exception:
        return None


def _parse_kalshi_date(game_date: Optional[str]) -> Optional[date]:
    """Parse Kalshi game_date 'May 3' → date(2026, 5, 3)."""
    if not game_date:
        return None
    try:
        parts = game_date.split()
        month = MONTH_MAP.get(parts[0])
        day = int(parts[1])
        if month is None:
            return None
        return date(date.today().year, month, day)
    except Exception:
        return None


def _dates_match(kalshi_game_date: Optional[str], sb_commence: str, tolerance_days: int = 1) -> bool:
    """
    Return True only if the sportsbook game date is within tolerance_days of the Kalshi game date.
    If kalshi_game_date is None (series markets), always return True.
    """
    if kalshi_game_date is None:
        return True
    kd = _parse_kalshi_date(kalshi_game_date)
    sd = _parse_sb_date(sb_commence)
    if kd is None or sd is None:
        return True
    return abs((kd - sd).days) <= tolerance_days


def scan_games(kalshi_markets: list[dict], sportsbook_games: list[dict]) -> list[dict]:
    """
    Scan game and series markets for EV.
    Uses sportsbook moneyline consensus directly as true probability.

    IMPORTANT: Only individual game markets (stat_type == 'game') get sportsbook-anchored.
    Series markets are skipped — comparing game-level win odds to series winner markets
    is an apples-to-oranges mismatch that creates fake edges.

    For game markets, we require the sportsbook game date to be within 1 day of the
    Kalshi game date (prevents matching Game 4 May 10 to the May 5 line).
    """
    results = []

    for market in kalshi_markets:
        stat_type = market["stat_type"]
        if stat_type not in ("series", "game"):
            continue
        # Series winner markets can't be properly priced using game-level moneylines.
        # Game odds for Game 2 ≠ series win probability. Skip series markets here.
        if stat_type == "series":
            continue
        if not _is_future_or_today(market.get("game_date")):
            continue
        if market["volume"] < MIN_VOLUME_SHOW:
            continue

        game_date = market.get("game_date")
        yes_team = market.get("yes_team")
        if not yes_team:
            continue

        # Require date proximity — prevents matching "Game 4 May 10" to the "May 5" line
        matched_game = None
        for game in sportsbook_games:
            team_match = (yes_team == game["home_team"] or yes_team == game["away_team"])
            if not team_match:
                yes_last = yes_team.split()[-1].upper()
                team_match = (yes_last in game["home_team"].upper() or yes_last in game["away_team"].upper())
            if not team_match:
                continue
            if not _dates_match(game_date, game.get("commence_time", ""), tolerance_days=1):
                continue
            matched_game = game
            break

        if not matched_game:
            continue

        home_team = matched_game["home_team"]
        away_team = matched_game["away_team"]
        num_books = matched_game.get("num_books", 1)
        yes_is_home = (yes_team == home_team or yes_team.split()[-1] in home_team)
        sb_home_prob = matched_game["home_win_prob"]

        # USE SPORTSBOOK DIRECTLY — consensus of 4-5 books is the best estimate.
        # Our net rating model adds noise; sportsbooks already have this info.
        true_prob_yes = sb_home_prob if yes_is_home else (1 - sb_home_prob)

        kalshi_implied = market["yes_ask"] / 100
        gap_pp = abs(true_prob_yes - kalshi_implied) * 100
        confidence = _confidence_tier(True, gap_pp, num_books)

        summary = edge_summary(
            market["yes_ask"], true_prob_yes,
            label=market["title"],
            kalshi_no_price=market.get("no_ask"),
        )
        if summary["best_ev"] < config.min_edge_to_show:
            continue

        direction = summary["best_side"]
        no_ask = market.get("no_ask") or (100 - market["yes_ask"])
        bet_price = market["yes_ask"] if direction == "YES" else no_ask

        bet = _kelly_bet(
            true_prob_yes if direction == "YES" else 1 - true_prob_yes,
            bet_price,
            confidence,
        )
        summary["bet_size_$"] = bet

        # For sportsbook-confirmed game edges, accept 2.5% EV — the edge is more reliable
        # than model-only props which need 4%. Sportsbook consensus of 3+ books is efficient.
        min_ev_for_game = 0.025
        actionable = (
            confidence == "HIGH"
            and summary["best_ev"] >= min_ev_for_game
            and market["volume"] >= MIN_VOLUME_BET
        )

        sb_pct = round(true_prob_yes * 100, 0)
        reasoning = (
            f"{num_books} book{'s' if num_books != 1 else ''} consensus: "
            f"{yes_team.split()[-1]} {sb_pct:.0f}% | "
            f"Kalshi: {market['yes_ask']}¢ YES → {gap_pp:.0f}pp gap → BET {direction} @ {bet_price}¢"
        )

        results.append({
            **summary,
            "actionable": actionable,
            "market_type": "game",
            "confidence": confidence,
            "sportsbook_anchored": True,
            "sb_prob": round(true_prob_yes, 4),
            "kalshi_implied": round(kalshi_implied, 4),
            "gap_pp": round(gap_pp, 1),
            "num_books": num_books,
            "reasoning": reasoning,
            "yes_team": yes_team,
            "home_team": home_team,
            "away_team": away_team,
            "volume": market["volume"],
            "ticker": market["ticker"],
            "game_date": game_date,
        })

    return results


# ── Parlay / combo analysis ───────────────────────────────────────────────────

def scan_parlays(actionable_edges: list[dict], max_legs: int = 3) -> list[dict]:
    """
    Find optimal parlay combinations from HIGH confidence actionable edges.

    On Kalshi, "parlays" aren't a native feature — each market is independent.
    A parlay here means: which COMBINATION of bets gives you the most uncorrelated
    expected value with the smallest total bankroll exposure?

    Parlay EV for N independent legs:
      combined_prob = prod(true_prob_i)
      combined_kalshi_price = price1/100 * price2/100 * ... (cost of all legs)
      parlay_payout = (1 - combined_kalshi_price) / combined_kalshi_price
      Parlay advantage: if combined_true_prob > combined_kalshi_price

    We filter out correlated legs: same game matchup = correlated (e.g. SAS YES + MIN NO
    from the same game carry the same exposure).
    """
    from itertools import combinations

    combos = []

    # Only combine edges that are sportsbook-anchored game edges (most reliable)
    game_edges = [e for e in actionable_edges if e.get("market_type") == "game" and e.get("sportsbook_anchored")]

    for n in range(2, min(max_legs + 1, len(game_edges) + 1)):
        for combo in combinations(range(len(game_edges)), n):
            legs = [game_edges[i] for i in combo]

            # Avoid correlated bets: skip if any two legs have the same teams (same game)
            matchups = set()
            correlated = False
            for leg in legs:
                key = frozenset([leg.get("home_team", ""), leg.get("away_team", "")])
                if key in matchups:
                    correlated = True
                    break
                matchups.add(key)
            if correlated:
                continue

            # Calculate combined stats
            combined_true_prob = 1.0
            combined_kalshi_price = 1.0
            total_ev = 0.0
            total_bet = 0.0

            for leg in legs:
                side = leg["best_side"]
                no_ask = leg.get("no_ask") or (100 - leg["kalshi_price"])
                bet_price = leg["kalshi_price"] if side == "YES" else no_ask
                true_prob = leg["true_prob"] if side == "YES" else (1 - leg["true_prob"])
                combined_true_prob *= true_prob
                combined_kalshi_price *= bet_price / 100
                total_ev += leg["best_ev"]
                total_bet += leg["bet_size_$"]

            # True parlay EV: bet $1 on all legs simultaneously
            parlay_ev = combined_true_prob - combined_kalshi_price

            legs_desc = []
            for leg in legs:
                side = leg["best_side"]
                no_ask = leg.get("no_ask") or (100 - leg["kalshi_price"])
                bet_price = leg["kalshi_price"] if side == "YES" else no_ask
                legs_desc.append(f"{leg['label']} → {side} @ {bet_price}¢")

            combos.append({
                "type": f"{n}-leg parlay",
                "legs": n,
                "combined_true_prob": round(combined_true_prob, 4),
                "combined_kalshi_price": round(combined_kalshi_price, 4),
                "parlay_ev": round(parlay_ev, 4),
                "sum_individual_ev": round(total_ev, 4),
                "recommended_bets": legs_desc,
                "total_kelly_exposure": round(total_bet, 2),
                "recommendation": (
                    f"Bet independently: {', '.join(leg['label'] for leg in legs)}. "
                    f"Combined true prob {combined_true_prob*100:.1f}% vs Kalshi implied {combined_kalshi_price*100:.1f}%. "
                    f"Each bet is uncorrelated — this is your highest-EV portfolio."
                    if parlay_ev > 0 else
                    f"Skip — parlay price ({combined_kalshi_price*100:.1f}%) exceeds true probability ({combined_true_prob*100:.1f}%)."
                ),
            })

    return sorted(combos, key=lambda x: -x["parlay_ev"])


# ── Main entry ────────────────────────────────────────────────────────────────

def run_scan() -> list[dict]:
    """
    Full daily scan. Returns all edges sorted by confidence tier then EV.

    Edge sources:
      1. Game moneylines vs sportsbook consensus
      2. Player props vs sportsbook lines (when available on free tier)
      3. Internal consistency mispricings (mathematical, no sportsbook needed)
    """
    print("Fetching Kalshi NBA markets...")
    kalshi = KalshiClient()
    markets = kalshi.get_all_parsed_markets()
    print(f"  Found {len(markets)} open NBA markets")

    print("Fetching sportsbook consensus lines...")
    odds = OddsClient()
    sb_games = odds.get_nba_moneylines()
    sb_props = odds.get_nba_player_props()
    print(f"  Found {len(sb_games)} games, {len(sb_props)} prop lines")

    print("Running EV scan...")
    prop_edges = scan_props(markets, sb_props)
    game_edges = scan_games(markets, sb_games)
    consistency_edges = scan_internal_consistency(markets)

    # Remove consistency edges already captured by prop scan (same ticker)
    prop_tickers = {e["ticker"] for e in prop_edges}
    consistency_edges = [e for e in consistency_edges if e["ticker"] not in prop_tickers]

    # Merge all, sort: HIGH first then by EV descending
    def sort_key(e):
        tier = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}.get(e.get("confidence", "LOW"), 2)
        return (tier, -e["best_ev"])

    all_edges = sorted(prop_edges + game_edges + consistency_edges, key=sort_key)

    actionable = [e for e in all_edges if e.get("actionable")]
    high = [e for e in all_edges if e.get("confidence") == "HIGH"]
    med  = [e for e in all_edges if e.get("confidence") == "MEDIUM"]
    low  = [e for e in all_edges if e.get("confidence") == "LOW"]
    consistency_count = len([e for e in all_edges if e.get("consistency_mispricing")])

    print(f"  HIGH confidence: {len(high)} edges ({len(actionable)} actionable)")
    print(f"  MEDIUM confidence: {len(med)} edges")
    print(f"  LOW confidence: {len(low)} edges")
    if consistency_count:
        print(f"  Mathematical mispricings: {consistency_count}")

    # Parlay analysis on actionable edges
    parlays = scan_parlays(actionable)
    if parlays:
        print(f"  Parlay combinations found: {len(parlays)}")
    print()

    return all_edges, parlays
