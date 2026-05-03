"""
EV calculation and Kelly criterion bet sizing.
"""
from config import config


def calculate_ev(kalshi_yes_price: float, true_prob: float, fee_rate: float = None) -> float:
    """
    EV per dollar wagered on YES at kalshi_yes_price cents.
    Positive = profitable bet.

    kalshi_yes_price: e.g. 58 means 58¢
    true_prob: your estimated probability of YES (0–1)
    """
    if fee_rate is None:
        fee_rate = config.fee_rate

    price = kalshi_yes_price / 100
    net_win = (1 - price) * (1 - fee_rate)
    net_loss = price

    return (true_prob * net_win) - ((1 - true_prob) * net_loss)


def calculate_ev_no(kalshi_no_price: float, true_prob_yes: float, fee_rate: float = None) -> float:
    """EV for buying NO at kalshi_no_price cents."""
    if fee_rate is None:
        fee_rate = config.fee_rate

    true_prob_no = 1 - true_prob_yes
    return calculate_ev(kalshi_no_price, true_prob_no, fee_rate)


def best_side(yes_ask: float, no_ask: float, true_prob: float) -> tuple[str, float]:
    """
    Given YES ask and NO ask prices, return the better side and its EV.
    Returns ('YES'|'NO'|'SKIP', ev)
    """
    ev_yes = calculate_ev(yes_ask, true_prob)
    ev_no = calculate_ev_no(no_ask, true_prob)

    if ev_yes >= ev_no and ev_yes > 0:
        return "YES", ev_yes
    elif ev_no > ev_yes and ev_no > 0:
        return "NO", ev_no
    return "SKIP", max(ev_yes, ev_no)


def kelly_bet(true_prob: float, price_cents: float, bankroll: float = None) -> float:
    """
    Returns dollar amount to bet using fractional Kelly.
    Caps at max_bet_pct of bankroll.
    """
    if bankroll is None:
        bankroll = config.bankroll

    price = min(max(price_cents, 1), 99) / 100  # clamp to [1¢, 99¢]
    b = (1 - price) / price  # net odds: win b dollars for every 1 risked
    if b == 0:
        return 0.0
    q = 1 - true_prob
    p = true_prob

    kelly = (b * p - q) / b
    kelly = max(0, kelly)

    fractional = kelly * config.kelly_fraction
    capped = min(fractional, config.max_bet_pct)

    return round(capped * bankroll, 2)


def edge_summary(kalshi_yes_price: float, true_prob: float, label: str = "",
                 kalshi_no_price: float = None) -> dict:
    """Full edge breakdown for a single market."""
    implied = kalshi_yes_price / 100
    raw_edge = true_prob - implied
    ev = calculate_ev(kalshi_yes_price, true_prob)

    # Use actual NO ask if provided and non-zero; otherwise approximate as 100 - yes_ask
    no_ask = kalshi_no_price if (kalshi_no_price and kalshi_no_price > 0) else (100 - kalshi_yes_price)
    ev_no = calculate_ev_no(no_ask, true_prob)
    side, best_ev = best_side(kalshi_yes_price, no_ask, true_prob)

    bet_size = kelly_bet(true_prob if side == "YES" else 1 - true_prob,
                         kalshi_yes_price if side == "YES" else no_ask)

    return {
        "label": label,
        "kalshi_price": kalshi_yes_price,
        "implied_prob": round(implied, 4),
        "true_prob": round(true_prob, 4),
        "raw_edge": round(raw_edge, 4),
        "ev_yes": round(ev, 4),
        "ev_no": round(ev_no, 4),
        "best_side": side,
        "best_ev": round(best_ev, 4),
        "bet_size_$": bet_size,
        "actionable": best_ev >= config.min_edge_to_bet,
    }
