"""
Unified Kalshi Edge Scanner — NBA + PGA on one site.

Loads each sport's modules from its own subdirectory using sys.path isolation
so conflicting module names (config, database, scanner, etc.) don't clash.
"""
import asyncio
import importlib.util
import os
import sys
import threading
import types
from datetime import datetime, timedelta

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── Write private key from env var if present (Railway deployment) ───────────
_pkey_b64 = os.environ.get("KALSHI_PRIVATE_KEY_B64", "")
if _pkey_b64:
    import base64
    _pkey_path = os.path.join(BASE, "kalshi_private_key.pem")
    with open(_pkey_path, "wb") as _f:
        _f.write(base64.b64decode(_pkey_b64))
    os.environ.setdefault("KALSHI_PRIVATE_KEY_PATH", _pkey_path)
NBA_DIR = os.path.join(BASE, "nba")
PGA_DIR = os.path.join(BASE, "pga")


# ─── Module isolation loader ──────────────────────────────────────────────────

def _load_sport_modules(sport_dir: str, prefix: str, module_names: list[str]) -> dict:
    """
    Load modules from sport_dir, isolating them under a unique prefix in sys.modules.
    Returns a dict of {name: module}.

    Approach:
      1. Add sport_dir to front of sys.path
      2. For each module: load via importlib with a prefixed name in sys.modules
         so cross-imports within the package resolve correctly
      3. Remove sport_dir from sys.path when done
    """
    # Temporarily make sport_dir the first path entry
    original_path = sys.path.copy()
    sys.path = [sport_dir] + [p for p in sys.path if p != sport_dir]

    # Clear any previously cached short names that would shadow our modules
    cached = {name: sys.modules.pop(name, None) for name in module_names}

    loaded = {}
    for name in module_names:
        path = os.path.join(sport_dir, f"{name}.py")
        if not os.path.exists(path):
            continue
        qualified = f"{prefix}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, path,
                submodule_search_locations=[])
        mod = importlib.util.module_from_spec(spec)
        # Register under BOTH the short name (for cross-imports) and the qualified name
        sys.modules[name] = mod
        sys.modules[qualified] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[loader] Warning loading {prefix}.{name}: {e}")
        loaded[name] = mod

    # Restore sys.path and purge short-name aliases (keep qualified names)
    sys.path = original_path
    for name in module_names:
        sys.modules.pop(name, None)
        # Restore any previously cached modules that weren't from this sport
        if cached.get(name) is not None:
            sys.modules[name] = cached[name]

    return loaded


def _load_models(sport_dir: str, prefix: str, model_names: list[str]) -> dict:
    """Load submodules from sport_dir/models/."""
    models_dir = os.path.join(sport_dir, "models")
    if not os.path.isdir(models_dir):
        return {}

    original_path = sys.path.copy()
    sys.path = [sport_dir, models_dir] + [p for p in sys.path if p not in (sport_dir, models_dir)]

    loaded = {}
    for name in model_names:
        path = os.path.join(models_dir, f"{name}.py")
        if not os.path.exists(path):
            continue
        qualified = f"{prefix}.models.{name}"
        short = f"models.{name}"
        spec = importlib.util.spec_from_file_location(qualified, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = mod
        sys.modules[short] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            print(f"[loader] Warning loading {prefix}.models.{name}: {e}")
        loaded[name] = mod

    sys.path = original_path
    for name in model_names:
        sys.modules.pop(f"models.{name}", None)

    return loaded


# ─── Load NBA ─────────────────────────────────────────────────────────────────
print("Loading NBA modules...")
_nba_models = _load_models(NBA_DIR, "nba", ["ev_engine", "prop_model", "game_model"])

NBA_MODULES = ["config", "database", "nba_data", "trade_log", "kalshi_client", "odds_client", "scanner"]
_nba = _load_sport_modules(NBA_DIR, "nba", NBA_MODULES)

nba_config   = sys.modules["nba.config"]
nba_db       = sys.modules["nba.database"]
nba_scanner  = sys.modules["nba.scanner"]
print(f"  NBA loaded: {list(_nba.keys())}")


# ─── Load PGA ─────────────────────────────────────────────────────────────────
print("Loading PGA modules...")
PGA_MODULES = ["config", "database", "ev_engine", "espn_client", "finishing_scanner",
               "kalshi_client", "odds_client", "scanner"]
_pga = _load_sport_modules(PGA_DIR, "pga", PGA_MODULES)

pga_config   = sys.modules["pga.config"]
pga_db       = sys.modules["pga.database"]
pga_scanner  = sys.modules["pga.scanner"]
print(f"  PGA loaded: {list(_pga.keys())}")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Kalshi Edge Scanner")

_nba_scanning = False
_pga_scanning = False
_nba_scan_lock = threading.Lock()
_pga_scan_lock = threading.Lock()
_nba_latest_parlays: list[dict] = []
_pga_latest_parlays: list[dict] = []


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", response_class=FileResponse)
async def dashboard():
    return FileResponse(os.path.join(BASE, "templates", "index.html"))


# ── NBA API ───────────────────────────────────────────────────────────────────

@app.post("/api/nba/scan")
async def nba_scan():
    global _nba_scanning
    if _nba_scanning:
        return JSONResponse({"status": "in_progress"}, status_code=202)
    _nba_scanning = True
    try:
        result = await asyncio.to_thread(_do_nba_scan)
        return result
    finally:
        _nba_scanning = False


@app.get("/api/nba/latest")
async def nba_latest():
    scan = nba_db.get_latest_scan()
    if not scan:
        return JSONResponse({"error": "no_data"}, status_code=404)
    scan["parlays"] = _nba_latest_parlays
    return scan


@app.get("/api/nba/history")
async def nba_history(limit: int = 30):
    return nba_db.get_scan_history(limit)


@app.get("/api/nba/performance")
async def nba_performance():
    return nba_db.get_performance_stats()


@app.get("/api/nba/status")
async def nba_status():
    return {
        "scanning": _nba_scanning,
        "bankroll": nba_config.config.bankroll,
        "min_edge": nba_config.config.min_edge_to_bet,
        "timestamp": datetime.now().isoformat(),
    }


# ── PGA API ───────────────────────────────────────────────────────────────────

@app.post("/api/pga/scan")
async def pga_scan():
    global _pga_scanning
    if _pga_scanning:
        return JSONResponse({"status": "in_progress"}, status_code=202)
    _pga_scanning = True
    try:
        result = await asyncio.to_thread(_do_pga_scan)
        return result
    finally:
        _pga_scanning = False


@app.get("/api/pga/latest")
async def pga_latest():
    scan = pga_db.get_latest_scan()
    if not scan:
        return JSONResponse({"error": "no_data"}, status_code=404)
    scan["parlays"] = _pga_latest_parlays or scan.get("parlays", [])
    return scan


@app.get("/api/pga/history")
async def pga_history(limit: int = 30):
    return pga_db.get_scan_history(limit)


@app.get("/api/pga/performance")
async def pga_performance():
    return pga_db.get_performance_stats()


@app.get("/api/pga/status")
async def pga_status():
    from datetime import date
    pga_start = date(2026, 5, 14)
    days_to_pga = (pga_start - date.today()).days
    return {
        "scanning": _pga_scanning,
        "bankroll": pga_config.config.bankroll,
        "min_edge": pga_config.config.min_edge_to_bet,
        "timestamp": datetime.now().isoformat(),
        "days_to_pga": max(0, days_to_pga),
        "pga_start_date": "2026-05-14",
    }


@app.get("/api/pga/leaderboard")
async def pga_leaderboard():
    try:
        espn = sys.modules.get("pga.espn_client")
        if not espn:
            return JSONResponse({"error": "espn_client not loaded"}, status_code=500)
        data = espn.get_live_leaderboard()
        if not data:
            return JSONResponse({"error": "no_live_event"}, status_code=404)
        return data
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ─── Scanner cores ────────────────────────────────────────────────────────────

def _do_nba_scan() -> dict:
    global _nba_latest_parlays
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting NBA scan...")
    edges, parlays = nba_scanner.run_scan()
    _nba_latest_parlays = parlays

    total = max(len(set(e.get("ticker", "") for e in edges if e.get("ticker"))), len(edges))
    scan_id = nba_db.save_scan(edges, total)

    actionable = [e for e in edges if e.get("actionable")]
    high = [e for e in edges if e.get("confidence") == "HIGH"]
    summary = {
        "total_edges": len(edges),
        "actionable": len(actionable),
        "high_conf": len(high),
        "game_edges":   len([e for e in actionable if e.get("market_type") == "game"]),
        "prop_edges":   len([e for e in actionable if e.get("market_type") == "prop"]),
        "combo_edges":  len([e for e in actionable if e.get("market_type") in ("combo", "mve")]),
        "futures_edges":len([e for e in actionable if e.get("market_type") in ("champ", "conf", "finals")]),
        "top_ev": round(actionable[0]["best_ev"] * 100, 1) if actionable else 0,
    }
    print(f"[{datetime.now().strftime('%H:%M:%S')}] NBA scan done: {len(edges)} edges, {len(parlays)} parlays")
    return {"id": scan_id, "scanned_at": datetime.now().isoformat(),
            "total_markets": total, "edges": edges, "parlays": parlays, "summary": summary}


def _do_pga_scan() -> dict:
    global _pga_latest_parlays
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting PGA scan...")
    edges, parlays = pga_scanner.run_scan()
    _pga_latest_parlays = parlays

    total = len(edges)
    scan_id = pga_db.save_scan(edges, parlays, total)

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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] PGA scan done: {len(edges)} edges")
    return {"id": scan_id, "scanned_at": datetime.now().isoformat(),
            "total_markets": total, "edges": edges, "parlays": parlays, "summary": summary}


# ─── Schedulers ───────────────────────────────────────────────────────────────

def _schedule_scans():
    from apscheduler.schedulers.background import BackgroundScheduler
    import pytz

    scheduler = BackgroundScheduler(timezone=pytz.timezone("America/New_York"))

    def safe_nba():
        global _nba_scanning
        with _nba_scan_lock:
            if _nba_scanning:
                return
            _nba_scanning = True
        try:
            _do_nba_scan()
        except Exception as e:
            print(f"NBA scheduled scan error: {e}")
        finally:
            _nba_scanning = False

    def safe_pga():
        global _pga_scanning
        with _pga_scan_lock:
            if _pga_scanning:
                return
            _pga_scanning = True
        try:
            _do_pga_scan()
        except Exception as e:
            print(f"PGA scheduled scan error: {e}")
        finally:
            _pga_scanning = False

    # NBA: daily at 3 AM ET
    scheduler.add_job(safe_nba, "cron", hour=3, minute=0)
    # PGA: 7 AM and 5 PM ET
    scheduler.add_job(safe_pga, "cron", hour=7, minute=0)
    scheduler.add_job(safe_pga, "cron", hour=17, minute=0)
    scheduler.start()
    print("Schedulers started — NBA at 3 AM, PGA at 7 AM and 5 PM ET")

    # Startup scans if stale / missing
    for fn, db_mod, label in [(safe_nba, nba_db, "NBA"), (safe_pga, pga_db, "PGA")]:
        existing = db_mod.get_latest_scan()
        if existing:
            age = datetime.now() - datetime.fromisoformat(existing["scanned_at"])
            if age < timedelta(hours=3):
                print(f"  {label}: recent scan ({int(age.total_seconds()/60)}m ago), skipping")
                continue
        threading.Thread(target=fn, daemon=True).start()


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def on_startup():
    nba_db.init_db()
    pga_db.init_db()
    _schedule_scans()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("web_app:app", host="0.0.0.0", port=port, reload=False)
