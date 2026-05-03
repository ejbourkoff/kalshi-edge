"""SQLite storage for PGA scan history and trade log."""
import json
import sqlite3
import os
from datetime import datetime

DB_PATH = os.getenv("PGA_DB_PATH", os.getenv("DB_PATH", "pga.db"))


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
                parlays_json TEXT,
                summary_json TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                scan_id INTEGER,
                ticker TEXT,
                player TEXT,
                tournament TEXT,
                kalshi_price INTEGER,
                sb_prob REAL,
                true_prob REAL,
                gap_pp REAL,
                best_side TEXT,
                best_ev REAL,
                bet_size REAL,
                confidence TEXT,
                num_books INTEGER,
                traded INTEGER DEFAULT 0,
                trade_price INTEGER,
                outcome TEXT,
                pnl REAL,
                notes TEXT
            )
        """)
        conn.commit()


def save_scan(edges: list[dict], parlays: list[dict], total_markets: int) -> int:
    actionable = [e for e in edges if e.get("actionable")]
    high = [e for e in edges if e.get("confidence") == "HIGH"]
    summary = {
        "total_edges": len(edges),
        "actionable": len(actionable),
        "high_conf": len(high),
        "yes_edges": len([e for e in actionable if e.get("best_side") == "YES"]),
        "no_edges":  len([e for e in actionable if e.get("best_side") == "NO"]),
        "top_ev": round(actionable[0]["best_ev"] * 100, 2) if actionable else 0,
        "tournaments": list(set(e.get("tournament", "") for e in edges)),
    }
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO scans (scanned_at, total_markets, edges_json, parlays_json, summary_json) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(), total_markets, json.dumps(edges), json.dumps(parlays), json.dumps(summary))
        )
        scan_id = cur.lastrowid
        for e in actionable:
            conn.execute("""
                INSERT INTO trades (logged_at, scan_id, ticker, player, tournament,
                    kalshi_price, sb_prob, true_prob, gap_pp, best_side, best_ev, bet_size, confidence, num_books)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                datetime.now().isoformat(), scan_id,
                e.get("ticker", ""), e.get("player", ""), e.get("tournament", ""),
                e.get("yes_ask", 0), e.get("sb_prob", 0), e.get("true_prob", 0),
                e.get("gap_pp", 0), e.get("best_side", ""), e.get("best_ev", 0),
                e.get("bet_size_$", 0), e.get("confidence", ""), e.get("num_books", 0),
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
            "parlays": json.loads(row["parlays_json"] or "[]"),
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


def get_performance_stats() -> dict:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT best_ev, outcome, pnl, tournament, best_side FROM trades WHERE outcome IS NOT NULL AND outcome != ''"
        ).fetchall()
        if not rows:
            return {"resolved": 0}
        resolved = [dict(r) for r in rows]
        wins = [r for r in resolved if r["outcome"] == "WIN"]
        total_pnl = sum(r["pnl"] or 0 for r in resolved)
        return {
            "resolved": len(resolved),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(resolved), 3),
            "total_pnl": round(total_pnl, 2),
            "yes_bets": len([r for r in resolved if r["best_side"] == "YES"]),
            "no_bets":  len([r for r in resolved if r["best_side"] == "NO"]),
        }
