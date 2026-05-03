"""
SQLite database for storing scan history and trade log.
"""
import json
import sqlite3
import os
from datetime import datetime

DB_PATH = os.getenv("NBA_DB_PATH", os.getenv("DB_PATH", "nba.db"))


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scanned_at TEXT NOT NULL,
                total_markets INTEGER,
                edges_json TEXT,
                summary_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                scan_id INTEGER,
                ticker TEXT,
                label TEXT,
                market_type TEXT,
                kalshi_price INTEGER,
                true_prob REAL,
                best_side TEXT,
                best_ev REAL,
                bet_size REAL,
                traded INTEGER DEFAULT 0,
                trade_price INTEGER,
                outcome TEXT,
                pnl REAL,
                notes TEXT
            )
        """)
        conn.commit()


def save_scan(edges: list[dict], total_markets: int) -> int:
    actionable = [e for e in edges if e.get("actionable")]
    summary = {
        "total_edges": len(edges),
        "actionable": len(actionable),
        "game_edges": len([e for e in actionable if e.get("market_type") == "game"]),
        "prop_edges": len([e for e in actionable if e.get("market_type") == "prop"]),
        "combo_edges": len([e for e in actionable if e.get("market_type") == "combo"]),
        "futures_edges": len([e for e in actionable if e.get("market_type") in ("champ", "conf", "finals")]),
        "top_ev": round(actionable[0]["best_ev"] * 100, 1) if actionable else 0,
    }
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO scans (scanned_at, total_markets, edges_json, summary_json) VALUES (?,?,?,?)",
            (datetime.now().isoformat(), total_markets, json.dumps(edges), json.dumps(summary))
        )
        scan_id = cur.lastrowid

        for e in actionable:
            conn.execute("""
                INSERT INTO trades (logged_at, scan_id, ticker, label, market_type,
                    kalshi_price, true_prob, best_side, best_ev, bet_size)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().isoformat(), scan_id,
                e.get("ticker", ""), e.get("label", ""), e.get("market_type", ""),
                e.get("kalshi_price", 0), e.get("true_prob", 0),
                e.get("best_side", ""), e.get("best_ev", 0), e.get("bet_size_$", 0),
            ))
        conn.commit()
    return scan_id


def get_latest_scan() -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM scans ORDER BY scanned_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "scanned_at": row["scanned_at"],
            "total_markets": row["total_markets"],
            "edges": json.loads(row["edges_json"]),
            "summary": json.loads(row["summary_json"]),
        }


def get_scan_history(limit: int = 30) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, scanned_at, total_markets, summary_json FROM scans ORDER BY scanned_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        return [{"id": r["id"], "scanned_at": r["scanned_at"],
                 "total_markets": r["total_markets"],
                 "summary": json.loads(r["summary_json"])} for r in rows]


def get_trades(limit: int = 200) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY logged_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def update_trade_outcome(trade_id: int, outcome: str, pnl: float, notes: str = ""):
    with get_conn() as conn:
        conn.execute(
            "UPDATE trades SET outcome=?, pnl=?, traded=1 WHERE id=?",
            (outcome, pnl, trade_id)
        )
        conn.commit()


def get_performance_stats() -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT best_ev, outcome, pnl, market_type FROM trades WHERE outcome IS NOT NULL AND outcome != ''"
        ).fetchall()
        if not rows:
            return {"resolved": 0}

        resolved = [dict(r) for r in rows]
        wins = [r for r in resolved if r["outcome"] == "WIN"]
        total_pnl = sum(r["pnl"] or 0 for r in resolved)

        buckets = {"2-4%": [], "4-7%": [], "7%+": []}
        for r in resolved:
            ev = r["best_ev"] or 0
            pnl = r["pnl"] or 0
            if ev < 0.04:
                buckets["2-4%"].append(pnl)
            elif ev < 0.07:
                buckets["4-7%"].append(pnl)
            else:
                buckets["7%+"].append(pnl)

        return {
            "resolved": len(resolved),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(resolved), 3) if resolved else 0,
            "total_pnl": round(total_pnl, 2),
            "by_type": {
                t: len([r for r in resolved if r["market_type"] == t])
                for t in ("game", "prop", "combo", "champ", "conf")
            },
            "ev_buckets": {k: {"count": len(v), "pnl": round(sum(v), 2)} for k, v in buckets.items()},
        }
