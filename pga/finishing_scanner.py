"""
Live finishing-position edge scanner for PGA Tour events.

Algorithm:
  1. Fetch live leaderboard from ESPN (positions, scores, holes remaining)
  2. Fetch Kalshi top-5/10/20 binary markets (KXPGATOP5/10/20 series)
  3. Skill-adjusted Monte Carlo simulation:
       - Each player's expected remaining score is adjusted by their OWGR rank
       - Top-10 world players get a ~+0.5 stroke advantage per round vs average field
       - This fixes the key failure mode: treating Scheffler the same as a #100 player
  4. Compare simulated finish probabilities to Kalshi YES prices
  5. Only flag edges for "CLEAR" positions (player well inside/outside threshold)
     — skips "COMPETITIVE" spots where model uncertainty is too high

Why skill adjustment matters:
  Without it, Scheffler and Si Woo Kim get identical top-5 probability (both at -9).
  But sportsbooks price Scheffler at ~82% and Kim at ~64% because Scheffler is world #1
  with a much higher expected final-round score. The skill term corrects this.

Model parameters:
  - SIGMA_18 = 1.8  (PGA Tour round std dev relative to field)
  - SKILL_ADVANTAGE = up to +0.8 strokes per round for world #1 vs average field
  - Minimum gap to flag: 20pp for competitive positions, 12pp for clear positions
"""
import numpy as np
from difflib import SequenceMatcher
from typing import Optional

from config import config

SIGMA_18 = 1.8          # PGA Tour score std dev per 18-hole round (relative to field mean)
N_SIMS = 15_000
TARGETS = [5, 10, 20]

# Finishing market series tickers
FINISHING_SERIES = {
    5:  "KXPGATOP5",
    10: "KXPGATOP10",
    20: "KXPGATOP20",
}

# OWGR-derived skill advantage in strokes per 18 holes relative to field average.
# Based on historical strokes-gained vs field data.
# World #1 ≈ +0.8 strokes/round, #10 ≈ +0.4, #50 ≈ +0.1, #100+ ≈ 0.
_SKILL_ADVANTAGE = {
    "Scottie Scheffler":   0.85,
    "Rory McIlroy":        0.65,
    "Jon Rahm":            0.55,
    "Bryson DeChambeau":   0.50,
    "Xander Schauffele":   0.55,
    "Ludvig Aberg":        0.45,
    "Viktor Hovland":      0.40,
    "Collin Morikawa":     0.40,
    "Patrick Cantlay":     0.40,
    "Wyndham Clark":       0.35,
    "Max Homa":            0.30,
    "Tommy Fleetwood":     0.30,
    "Hideki Matsuyama":    0.30,
    "Justin Thomas":       0.30,
    "Jordan Spieth":       0.30,
    "Rickie Fowler":       0.25,
    "Shane Lowry":         0.25,
    "Adam Scott":          0.20,
    "Min Woo Lee":         0.25,
    "Si Woo Kim":          0.20,
    "Cameron Young":       0.35,
    "Akshay Bhatia":       0.30,
    "Kristoffer Reitan":   0.15,
    "Ben Griffin":         0.20,
    "Nick Taylor":         0.20,
    "Matthew McCarty":     0.15,
    "Matt McCarty":        0.15,
    "Alex Noren":          0.15,
    "Alex Smalley":        0.15,
    "Sepp Straka":         0.20,
    "Andrew Putnam":       0.10,
    "Aldrich Potgieter":   0.15,
    "Chris Gotterup":      0.15,
    "Russell Henley":      0.20,
    "Daniel Berger":       0.20,
    "Kurt Kitayama":       0.20,
    "Michael Kim":         0.10,
    "Harry Hall":          0.10,
}


def _skill_adj(name: str) -> float:
    """Strokes-per-round advantage vs average PGA Tour field player."""
    # Exact match
    if name in _SKILL_ADVANTAGE:
        return _SKILL_ADVANTAGE[name]
    # Partial last-name match
    last = name.split()[-1]
    for k, v in _SKILL_ADVANTAGE.items():
        if k.split()[-1] == last:
            return v
    return 0.0  # unknown player = average field


def _name_sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def match_espn_to_kalshi(espn_name: str, kalshi_markets: list[dict]) -> Optional[dict]:
    """Fuzzy-match an ESPN player name to a Kalshi finishing-position market."""
    e_last = espn_name.split()[-1].lower()

    # 1. Exact last name
    for m in kalshi_markets:
        k = m.get("player", "") or ""
        if k.split()[-1].lower() == e_last and len(e_last) > 3:
            return m

    # 2. Fuzzy full name — require higher threshold to avoid wrong matches
    best, best_score = None, 0.0
    for m in kalshi_markets:
        s = _name_sim(espn_name, m.get("player", "") or "")
        if s > best_score:
            best, best_score = m, s

    return best if best_score >= 0.82 else None


def simulate_probs(players: list[dict], holes_remaining: int) -> dict[str, dict[int, float]]:
    """
    Skill-adjusted Monte Carlo simulation for finishing probabilities.

    players: [{name, score_int, position}]
    holes_remaining: int (e.g. 18 for full round 4)

    Returns {player_name: {5: prob, 10: prob, 20: prob}}
    """
    if not players:
        return {}

    sigma = SIGMA_18 * (holes_remaining / 18) ** 0.5
    n = len(players)

    # Base scores
    scores = np.array([p["score_int"] for p in players], dtype=float)

    # Skill adjustment: subtract advantage (lower score = better in golf)
    skill = np.array([_skill_adj(p["name"]) for p in players], dtype=float)
    # Scale skill to holes remaining
    skill_scaled = skill * (holes_remaining / 18)

    # (n_sims, n_players): raw variance draw + skill advantage
    noise = np.random.normal(0, sigma, size=(N_SIMS, n))
    final = scores + noise - skill_scaled  # subtract advantage = lower score

    result = {}
    for i, p in enumerate(players):
        rank = 1 + np.sum(final < final[:, [i]], axis=1)
        probs = {t: float(np.mean(rank <= t)) for t in TARGETS}
        result[p["name"]] = probs

    return result


def _is_clear_position(position: int, threshold: int, score_gap: float, n_players: int) -> bool:
    """
    Returns True if the player's position is 'clear' — well inside or outside
    the threshold. Clear positions are more reliable for model-based edges.

    position: current rank
    threshold: 5, 10, or 20
    score_gap: strokes separating player from the threshold bubble
    """
    margin = threshold - position  # positive = inside, negative = outside
    if margin >= 5 and score_gap >= 2:
        return True   # well inside — should almost certainly finish top-N
    if margin <= -10:
        return True   # well outside — very unlikely to recover
    return False


def _ev(true_prob: float, ask_cents: float) -> float:
    price = ask_cents / 100
    return true_prob * (1 - price) * (1 - config.fee_rate) - (1 - true_prob) * price


def _kelly(true_prob: float, ask_cents: float) -> float:
    price = ask_cents / 100
    if price <= 0 or price >= 1:
        return 0.0
    b = (1 - price) / price
    if b == 0:
        return 0.0
    k = max(0, (b * true_prob - (1 - true_prob)) / b)
    capped = min(k * 0.10, 0.015)
    return round(capped * config.bankroll, 2)


def _confidence(gap_pp: float, ev: float, volume: float, is_clear: bool) -> str:
    # Require much larger gaps for competitive positions (model is less reliable)
    min_gap_high = 14 if is_clear else 22
    min_gap_med  = 9  if is_clear else 16
    if gap_pp >= min_gap_high and ev >= 0.06 and volume >= 5000:
        return "HIGH"
    if gap_pp >= min_gap_med and ev >= 0.03 and volume >= 2000:
        return "MEDIUM"
    return "LOW"


def scan_finishing_positions(
    kalshi_markets_by_threshold: dict[int, list[dict]],
    espn_data: dict,
) -> list[dict]:
    """
    Main scanner: compare skill-adjusted simulation to Kalshi prices.
    """
    players = espn_data.get("players", [])
    holes_remaining = espn_data.get("holes_remaining", 18)
    tournament_name = espn_data.get("event_name", "Live PGA Event")
    status = espn_data.get("status", "in")

    if not players:
        return []

    n_players = len(players)
    print(f"  [{tournament_name}] {n_players} players, {holes_remaining}h remaining ({status})")

    # Build score-at-bubble lookup for each threshold
    bubble_scores: dict[int, float] = {}
    for threshold in TARGETS:
        if threshold < n_players:
            bubble_scores[threshold] = players[threshold - 1]["score_int"]

    sim_probs = simulate_probs(players, max(holes_remaining, 1))

    edges = []

    for threshold, kalshi_markets in kalshi_markets_by_threshold.items():
        if not kalshi_markets:
            continue

        bubble_score = bubble_scores.get(threshold, 0)

        for espn_player in players:
            espn_name = espn_player["name"]
            true_prob = sim_probs.get(espn_name, {}).get(threshold)
            if true_prob is None:
                continue

            kalshi_market = match_espn_to_kalshi(espn_name, kalshi_markets)
            if not kalshi_market:
                continue

            yes_ask = kalshi_market.get("yes_ask")
            no_ask  = kalshi_market.get("no_ask") or (100 - yes_ask)
            volume  = kalshi_market.get("volume", 0)

            if not yes_ask or yes_ask <= 0:
                continue

            kalshi_implied = yes_ask / 100
            gap_pp_yes = (true_prob - kalshi_implied) * 100

            ev_yes = _ev(true_prob, yes_ask)
            ev_no  = _ev(1 - true_prob, no_ask)

            if ev_yes >= ev_no and ev_yes > 0:
                side, best_ev, bet_price = "YES", ev_yes, yes_ask
                gap_pp = gap_pp_yes
            elif ev_no > ev_yes and ev_no > 0:
                side, best_ev, bet_price = "NO", ev_no, no_ask
                gap_pp = -gap_pp_yes
            else:
                continue

            abs_gap = abs(gap_pp)
            if abs_gap < 5.0 or best_ev < 0.01:
                continue

            position = espn_player.get("position", 999)
            score_int = espn_player.get("score_int", 0)
            score_gap = abs(score_int - bubble_score)
            is_clear = _is_clear_position(position, threshold, score_gap, n_players)

            conf = _confidence(abs_gap, best_ev, volume, is_clear)

            # Only surface competitive positions if gap is extremely large (25pp+)
            if not is_clear and abs_gap < 25:
                conf = "LOW"

            bet = _kelly(true_prob if side == "YES" else 1 - true_prob, bet_price)

            actionable = (
                conf in ("HIGH", "MEDIUM")
                and best_ev >= config.min_edge_to_bet
                and volume >= 2000
            )

            score_str = f"{score_int:+d}" if score_int != 0 else "E"
            position_label = "CLEAR" if is_clear else "COMPETITIVE"

            reasoning = (
                f"R{espn_data.get('round',4)} #{position} ({score_str}) [{position_label}] | "
                f"Skill-adj model top-{threshold}: {true_prob*100:.0f}% vs Kalshi {yes_ask}¢ → "
                f"BET {side} @ {bet_price}¢  ({abs_gap:.1f}pp gap)"
            )

            edges.append({
                "player": espn_name,
                "kalshi_player": kalshi_market.get("player", espn_name),
                "tournament": tournament_name,
                "market_type": f"top{threshold}",
                "threshold": threshold,
                "position": position,
                "score_str": score_str,
                "position_label": position_label,
                "ticker": kalshi_market.get("ticker", ""),
                "yes_ask": yes_ask,
                "no_ask": no_ask,
                "kalshi_implied": round(kalshi_implied, 4),
                "true_prob": round(true_prob, 4),
                "sb_prob": round(true_prob, 4),
                "gap_pp": round(abs_gap, 2),
                "ev_yes": round(ev_yes, 4),
                "ev_no": round(ev_no, 4),
                "best_side": side,
                "best_ev": round(best_ev, 4),
                "bet_size_$": bet,
                "confidence": conf,
                "actionable": actionable,
                "num_books": 0,
                "volume": volume,
                "reasoning": reasoning,
                "label": f"{espn_name} top {threshold} — {tournament_name}",
                "holes_remaining": holes_remaining,
                "tournament_status": status,
            })

    edges.sort(key=lambda x: ({"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["confidence"]], -x["best_ev"]))
    return edges
