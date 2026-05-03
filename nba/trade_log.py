"""
Trade logging — records every scan result and actual trades to CSV.
"""
import csv
import os
from datetime import datetime


LOG_FILE = "trade_log.csv"

HEADERS = [
    "date", "time", "ticker", "label", "market_type",
    "kalshi_price", "true_prob", "best_side", "best_ev",
    "bet_size_$", "actionable",
    # filled in after trade resolves:
    "traded", "trade_price", "outcome", "pnl", "notes"
]


def ensure_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()


def log_scan_results(edges: list[dict]):
    """Append today's scan results to the log (actionable only)."""
    ensure_log()
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")

    actionable = [e for e in edges if e["actionable"]]

    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        for edge in actionable:
            row = {
                "date": date_str,
                "time": time_str,
                "ticker": edge.get("ticker", ""),
                "label": edge.get("label", ""),
                "market_type": edge.get("market_type", ""),
                "kalshi_price": edge.get("kalshi_price", ""),
                "true_prob": edge.get("true_prob", ""),
                "best_side": edge.get("best_side", ""),
                "best_ev": edge.get("best_ev", ""),
                "bet_size_$": edge.get("bet_size_$", ""),
                "actionable": edge.get("actionable", ""),
                "traded": "",
                "trade_price": "",
                "outcome": "",
                "pnl": "",
                "notes": "",
            }
            writer.writerow(row)

    print(f"Logged {len(actionable)} actionable trades to {LOG_FILE}")


def show_performance_summary():
    """Print a quick P&L summary from the log."""
    ensure_log()
    rows = []
    with open(LOG_FILE, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    traded = [r for r in rows if r["traded"].lower() == "yes"]
    resolved = [r for r in traded if r["outcome"] in ("WIN", "LOSS")]

    if not resolved:
        print("No resolved trades yet.")
        return

    wins = [r for r in resolved if r["outcome"] == "WIN"]
    total_pnl = sum(float(r["pnl"]) for r in resolved if r["pnl"])

    print(f"\nPerformance Summary ({len(resolved)} resolved trades)")
    print(f"  Win rate: {len(wins)}/{len(resolved)} = {len(wins)/len(resolved):.1%}")
    print(f"  Total P&L: ${total_pnl:.2f}")

    # Break down by EV bucket
    buckets = {"2-4%": [], "4-7%": [], "7%+": []}
    for r in resolved:
        try:
            ev = float(r["best_ev"])
            pnl = float(r["pnl"]) if r["pnl"] else 0
            if ev < 0.04:
                buckets["2-4%"].append(pnl)
            elif ev < 0.07:
                buckets["4-7%"].append(pnl)
            else:
                buckets["7%+"].append(pnl)
        except ValueError:
            pass

    print("\nP&L by edge bucket:")
    for bucket, pnls in buckets.items():
        if pnls:
            print(f"  {bucket}: {len(pnls)} trades, ${sum(pnls):.2f} total")
