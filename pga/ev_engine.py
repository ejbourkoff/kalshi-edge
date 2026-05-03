"""
EV calculation and Kelly criterion for PGA outright markets.

Golf outrights are low-probability multi-outcome markets. The key differences
from NBA game markets:
  - True probabilities are small (0.5%–20%)
  - The NO side is usually cheap (86-99¢) but has low payout
  - YES side has high payout but is binary (one winner per tournament)
  - Kelly sizing is conservative — high variance events
"""
from config import config


def calculate_ev_yes(yes_price_cents: float, true_prob: float, fee_rate: float = None) -> float:
    """EV per dollar on YES. Positive = profitable."""
    if fee_rate is None:
        fee_rate = config.fee_rate
    price = yes_price_cents / 100
    net_win = (1 - price) * (1 - fee_rate)
    return true_prob * net_win - (1 - true_prob) * price


def calculate_ev_no(no_price_cents: float, true_prob_yes: float, fee_rate: float = None) -> float:
    """EV per dollar on NO (betting against the player winning)."""
    if fee_rate is None:
        fee_rate = config.fee_rate
    true_prob_no = 1 - true_prob_yes
    return calculate_ev_yes(no_price_cents, true_prob_no, fee_rate)


def kelly_bet(true_prob: float, price_cents: float, confidence: str = "HIGH") -> float:
    """
    Fractional Kelly bet in dollars.
    Golf is high-variance — use a smaller fraction than NBA.
    """
    fractions = {"HIGH": config.kelly_fraction, "MEDIUM": 0.08, "LOW": 0.0}
    fraction = fractions.get(confidence, 0.0)
    if fraction == 0:
        return 0.0

    price = min(max(price_cents, 0.1), 99.9) / 100
    b = (1 - price) / price
    if b == 0:
        return 0.0

    kelly = max(0, (b * true_prob - (1 - true_prob)) / b)
    capped = min(kelly * fraction, config.max_bet_pct)
    return round(capped * config.bankroll, 2)


def edge_summary(
    yes_ask: float,
    no_ask: float,
    true_prob: float,
    player: str,
    tournament: str,
    sb_prob: float,
    num_books: int,
    confidence: str,
    gap_pp: float,
    volume: float,
) -> dict:
    """Full edge breakdown for one player's outright market."""
    kalshi_implied = yes_ask / 100

    ev_yes = calculate_ev_yes(yes_ask, true_prob)
    ev_no  = calculate_ev_no(no_ask, true_prob)

    if ev_yes >= ev_no and ev_yes > 0:
        side, best_ev = "YES", ev_yes
        bet_price = yes_ask
    elif ev_no > ev_yes and ev_no > 0:
        side, best_ev = "NO", ev_no
        bet_price = no_ask
    else:
        side, best_ev = "SKIP", max(ev_yes, ev_no)
        bet_price = yes_ask

    bet = kelly_bet(
        true_prob if side == "YES" else 1 - true_prob,
        bet_price,
        confidence,
    )

    # actionable: HIGH conf + min EV + min volume + enough books
    actionable = (
        confidence == "HIGH"
        and best_ev >= config.min_edge_to_bet
        and volume >= config.min_volume_bet
        and num_books >= config.min_books
    )

    direction_str = "YES" if side == "YES" else "NO"
    reasoning = (
        f"{num_books} books no-vig: {sb_prob*100:.1f}% | "
        f"Kalshi: {yes_ask}¢ YES → {gap_pp:.1f}pp gap → "
        f"BET {direction_str} @ {bet_price}¢"
    )

    return {
        "player": player,
        "tournament": tournament,
        "label": f"{player} — {tournament}",
        "yes_ask": yes_ask,
        "no_ask": no_ask,
        "kalshi_implied": round(kalshi_implied, 5),
        "sb_prob": round(sb_prob, 5),
        "true_prob": round(true_prob, 5),
        "gap_pp": round(gap_pp, 2),
        "ev_yes": round(ev_yes, 4),
        "ev_no": round(ev_no, 4),
        "best_side": side,
        "best_ev": round(best_ev, 4),
        "bet_size_$": bet,
        "actionable": actionable,
        "confidence": confidence,
        "num_books": num_books,
        "reasoning": reasoning,
    }
