"""
ESPN unofficial API client for live PGA Tour leaderboard.
Used to get real-time scores for finishing-position probability simulation.
"""
import requests
from typing import Optional

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/golf/pga"


def _parse_score(val) -> int:
    """'-15' → -15, 'E' → 0, '+2' → 2, 0.0 → 0"""
    if isinstance(val, (int, float)):
        return int(val)
    s = str(val).strip()
    if s in ("E", "-", "", "None"):
        return 0
    try:
        return int(s)
    except ValueError:
        return 0


def get_live_leaderboard() -> Optional[dict]:
    """
    Returns dict with keys:
      - event_name: str
      - round: int (1-4)
      - status: 'pre' | 'in' | 'post' | 'suspended'
      - holes_per_round: int (usually 18)
      - total_holes: int (usually 72)
      - holes_remaining_in_round: int
      - total_holes_remaining: int
      - players: list of {name, score_int, position, holes_done_today, active}
    """
    try:
        r = requests.get(f"{ESPN_BASE}/scoreboard", timeout=8)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"[ESPN] Leaderboard fetch failed: {e}")
        return None

    events = data.get("events", [])
    if not events:
        return None

    event = events[0]
    comp = event.get("competitions", [{}])[0]
    status_obj = comp.get("status", {})
    status_type = status_obj.get("type", {})

    current_round = status_obj.get("period", 4)
    status_name = status_type.get("name", "")
    completed = status_type.get("completed", False)

    if completed or status_name in ("STATUS_FINAL", "STATUS_PLAY_COMPLETE"):
        status = "post"
    elif "SUSPEND" in status_name.upper():
        status = "suspended"
    elif status_name in ("STATUS_IN_PROGRESS", "STATUS_PLAY_RESUMED"):
        status = "in"
    elif status_name in ("STATUS_SCHEDULED", "STATUS_UPCOMING"):
        status = "pre"
    else:
        # fallback: use state field
        raw_state = status_type.get("state", "pre")
        status = "in" if raw_state == "in" else ("post" if completed else "suspended")

    holes_per_round = 18
    players = []

    for c in comp.get("competitors", []):
        name = c.get("athlete", {}).get("displayName", "")
        if not name:
            continue

        total_score = _parse_score(c.get("score", 0))
        order = c.get("order", 999)

        # Count holes played in current round from linescores
        linescores = c.get("linescores", [])
        r4_entry = next((ls for ls in linescores if ls.get("period") == current_round), None)
        holes_done_today = 0
        if r4_entry:
            hole_data = r4_entry.get("linescores", [])
            holes_done_today = len([h for h in hole_data if h.get("value") is not None])

        players.append({
            "name": name,
            "score_int": total_score,
            "position": order,
            "holes_done_today": holes_done_today,
            "active": True,
        })

    players.sort(key=lambda p: (p["score_int"], p["name"]))

    # Fix positions (handle ties)
    for i, p in enumerate(players):
        p["position"] = i + 1

    holes_remaining_in_round = holes_per_round - (players[0]["holes_done_today"] if players else 0)
    rounds_complete = current_round - 1
    total_holes_remaining = rounds_complete * 0 + holes_remaining_in_round  # just this round

    return {
        "event_name": event.get("name", "PGA Event"),
        "round": current_round,
        "status": status,
        "holes_per_round": holes_per_round,
        "holes_remaining": holes_remaining_in_round,
        "players": players,
    }
