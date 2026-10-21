"""NBA data layer — every external fetch lives here.

Sources:
  - ESPN's public JSON API (site.api.espn.com / site.web.api.espn.com): schedule,
    scores, box scores, standings, injuries, rosters, player game logs. Free, no
    key, and — unlike stats.nba.com (nba_api) — it isn't blocked from Streamlit
    Cloud's IPs, which is the same trap the MLB app hit with Fangraphs.
  - The Odds API (basketball_nba): game lines and player props, live + historical.

Everything here degrades to an empty result instead of raising, same contract as
mlb_core: a bad fetch should mean "no data for this", never a crashed page.

ESPN ids are kept as STRINGS everywhere (team "13", athlete "3112335") so they
hash cleanly for st.cache_data and never get mangled into floats by pandas.
"""
import re
import time
import unicodedata
from datetime import date, datetime, timedelta

import requests as req
import streamlit as st

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
    _UK = ZoneInfo("Europe/London")
    _UTC = ZoneInfo("UTC")
except Exception:  # pragma: no cover
    _ET = _UK = _UTC = None

ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
ESPN_STANDINGS = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
ESPN_COMMON = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
ODDS_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "basketball_nba"

_HEADERS = {"User-Agent": "Mozilla/5.0 (nba-betting-app)"}


def safe_get(url, params=None, timeout=20):
    """GET JSON with 3 attempts. Returns {} on any failure — never raises."""
    for attempt in range(3):
        try:
            r = req.get(url, params=params, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt == 2:
                return {}
            time.sleep(0.8)
    return {}


# --- Dates & times ----------------------------------------------------------

def season_for(d):
    """ESPN labels a season by the year it ENDS: Oct 2026 - Jun 2027 = 2027."""
    return d.year + 1 if d.month >= 8 else d.year


def season_start(season):
    """Safe lower bound for a season's games. Preseason is filtered out by the
    event's season type, so starting a little early costs nothing."""
    return date(season - 1, 10, 1)


def parse_iso(s):
    """Parse either API's timestamp ('2026-10-22T23:30Z' from ESPN,
    '2026-10-22T23:30:00Z' from The Odds API) into an aware UTC datetime."""
    if not s:
        return None
    s = str(s).strip().replace("Z", "+00:00")
    for fmt in ("%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def to_uk_time(iso):
    """Kick-off in UK local time (handles GMT/BST properly — the MLB app's
    hard-coded +1h would be an hour out from late October)."""
    dt = parse_iso(iso)
    if dt is None or _UK is None:
        return ""
    return dt.astimezone(_UK).strftime("%H:%M")


def to_et_date(iso):
    """US-Eastern calendar date — the NBA's slate date. A 7:30pm ET tip is
    00:30 UK the next morning but belongs to the US date."""
    dt = parse_iso(iso)
    if dt is None or _ET is None:
        return None
    return dt.astimezone(_ET).date()


def et_date_str(iso):
    d = to_et_date(iso)
    return d.strftime("%a %b %d") if d else ""


def today_et():
    if _ET is None:
        return date.today()
    return datetime.now(_ET).date()


# --- Names --------------------------------------------------------------------

def nickname(team_name):
    """Last word of a team name, lower-cased. Unique across the NBA and
    identical between ESPN ('LA Clippers') and The Odds API ('Los Angeles
    Clippers'), which is why matching is done on it."""
    parts = (team_name or "").lower().strip().split()
    return parts[-1] if parts else ""


_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm_name(name):
    """Normalise a player name for matching across sources: strip accents
    (Jokić -> jokic), punctuation (P.J. -> pj) and suffixes (Jr., III)."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " "))
    parts = [p for p in s.split() if p not in _SUFFIXES]
    return " ".join(parts)


def _num(v, default=0.0):
    try:
        if v is None or v == "" or v == "--":
            return default
        return float(v)
    except Exception:
        return default


def _minutes(v):
    """'34', '34:12' or '34.5' -> float minutes."""
    if v is None:
        return 0.0
    s = str(v)
    if ":" in s:
        m, _, sec = s.partition(":")
        return _num(m) + _num(sec) / 60.0
    return _num(s)


def _made(v):
    """'3-7' -> 3.0 (made part of a made-attempted string)."""
    if v is None:
        return 0.0
    s = str(v)
    return _num(s.split("-")[0]) if "-" in s else _num(s)


def _pick(labels, keys, stats, label_opts, key_opts):
    """Look up one stat by display label (stable: 'PTS', 'REB', '3PT') with a
    fallback to ESPN's internal key names. Returns the raw string or None."""
    for lab in label_opts:
        if labels and lab in labels:
            i = labels.index(lab)
            if i < len(stats):
                return stats[i]
    for k in key_opts:
        if keys and k in keys:
            i = keys.index(k)
            if i < len(stats):
                return stats[i]
    return None


def _line_from_stats(labels, keys, stats):
    return {
        "min": _minutes(_pick(labels, keys, stats, ["MIN"], ["minutes"])),
        "pts": _num(_pick(labels, keys, stats, ["PTS"], ["points"])),
        "reb": _num(_pick(labels, keys, stats, ["REB"], ["rebounds", "totalRebounds"])),
        "ast": _num(_pick(labels, keys, stats, ["AST"], ["assists"])),
        "fg3": _made(_pick(labels, keys, stats, ["3PT"],
                           ["threePointFieldGoalsMade-threePointFieldGoalsAttempted"])),
    }


# --- Schedule & scores ----------------------------------------------------------

def _parse_event(ev):
    comp = (ev.get("competitions") or [{}])[0]
    status = comp.get("status") or ev.get("status") or {}
    stype = status.get("type") or {}
    out = {
        "event_id": str(ev.get("id", "")),
        "start_iso": ev.get("date") or comp.get("date") or "",
        "season_type": int(((ev.get("season") or {}).get("type")) or 2),
        "state": stype.get("state", "pre"),          # pre | in | post
        "completed": bool(stype.get("completed")),
        "detail": stype.get("shortDetail") or stype.get("detail", ""),
        "period": status.get("period"),
        "clock": status.get("displayClock", ""),
        "venue": (comp.get("venue") or {}).get("fullName", ""),
    }
    for c in comp.get("competitors", []):
        side = c.get("homeAway")
        if side not in ("home", "away"):
            continue
        t = c.get("team") or {}
        out[f"{side}_id"] = str(t.get("id", ""))
        out[f"{side}_name"] = t.get("displayName", "")
        out[f"{side}_abbr"] = t.get("abbreviation", "")
        sc = c.get("score")
        if isinstance(sc, dict):
            sc = sc.get("value")
        out[f"{side}_score"] = int(_num(sc)) if sc not in (None, "") else None
    return out


def _events_from(data, include_preseason=False):
    out = []
    for ev in data.get("events", []) or []:
        g = _parse_event(ev)
        if not g.get("home_id") or not g.get("away_id"):
            continue
        if g["season_type"] == 1 and not include_preseason:
            continue
        out.append(g)
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_scoreboard(date_str):
    """Every regular-season/playoff game on a US date (YYYY-MM-DD), any state.
    Short TTL so live scores stay current."""
    ds = str(date_str).replace("-", "")
    return _events_from(safe_get(f"{ESPN_SITE}/scoreboard", {"dates": ds, "limit": 100}))


def _month_chunks(start_d, end_d):
    cur = date(start_d.year, start_d.month, 1)
    while cur <= end_d:
        nxt = date(cur.year + (cur.month == 12), cur.month % 12 + 1, 1)
        yield max(cur, start_d), min(nxt - timedelta(days=1), end_d)
        cur = nxt


class FetchFailed(Exception):
    """Raised INSIDE cached functions when a call fails, so st.cache_data
    doesn't store the failure. (Previously a failed call returned [] and that
    empty result was cached — for a week on past months — which is exactly
    how the Backtest page ended up seeing no games at all.)"""


def _get_events_or_raise(params, timeout=30):
    data = safe_get(f"{ESPN_SITE}/scoreboard", params, timeout=timeout)
    if not isinstance(data, dict) or "events" not in data:
        raise FetchFailed(str(params))
    return _events_from(data)


@st.cache_data(ttl=10800, show_spinner=False)
def _fetch_range(start_str, end_str):
    return _get_events_or_raise({"dates": f"{start_str}-{end_str}", "limit": 1000})


@st.cache_data(ttl=604800, show_spinner=False)
def _fetch_range_frozen(start_str, end_str):
    """Months fully in the past — finished games never change, cached a week.
    Only ever holds a SUCCESSFUL response (failures raise and aren't cached)."""
    return _get_events_or_raise({"dates": f"{start_str}-{end_str}", "limit": 1000})


@st.cache_data(ttl=604800, show_spinner=False)
def _fetch_day_frozen(ds):
    """One past day's games — the same request the live slate uses, so it's
    the reliable fallback if ESPN's date-range form returns nothing."""
    return _get_events_or_raise({"dates": ds, "limit": 100}, timeout=20)


def _fetch_days(a, b):
    """All games between dates a and b, one cached call per day, 8 in parallel
    (a full month is ~30 small calls; only slow the first time)."""
    days = [(a + timedelta(days=i)).strftime("%Y%m%d") for i in range((b - a).days + 1)]

    recent = (today_et() - timedelta(days=2)).strftime("%Y%m%d")

    def one(ds):
        try:
            # Last couple of days use the short-TTL fetch, so a game that was
            # still in progress isn't frozen as unfinished for a week.
            return fetch_scoreboard(ds) if ds >= recent else _fetch_day_frozen(ds)
        except Exception:
            return []

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as ex:
        return [g for day in ex.map(one, days) for g in day]


# Visible diagnostics for the Backtest page — how the last season pull went.
LAST_SEASON_FETCH = {}


def fetch_season_results(season, before_date_str):
    """Completed games of `season` played strictly BEFORE before_date (ET).
    Tries one date-range call per month first; any month that fails or comes
    back empty is re-fetched day by day. Capped at the requested date so a
    past slate is never rebuilt with games that hadn't happened yet."""
    end = datetime.strptime(str(before_date_str), "%Y-%m-%d").date() - timedelta(days=1)
    start = season_start(season)
    end = min(end, date(season, 6, 30))
    LAST_SEASON_FETCH.clear()
    LAST_SEASON_FETCH.update(season=season, months_range=0, months_daily=0, games=0)
    if end < start:
        return []
    this_month = date(today_et().year, today_et().month, 1)
    games, seen = [], set()
    for a, b in _month_chunks(start, end):
        fn = _fetch_range_frozen if b < this_month else _fetch_range
        try:
            evs = fn(a.strftime("%Y%m%d"), b.strftime("%Y%m%d"))
        except Exception:
            evs = []
        if evs:
            LAST_SEASON_FETCH["months_range"] += 1
        else:
            evs = _fetch_days(a, b)
            LAST_SEASON_FETCH["months_daily"] += 1
        for g in evs:
            if not g["completed"] or g["event_id"] in seen:
                continue
            gd = to_et_date(g["start_iso"])
            if gd is None or gd > end:
                continue
            g = dict(g, et_date=str(gd))
            seen.add(g["event_id"])
            games.append(g)
    games.sort(key=lambda g: g["start_iso"])
    LAST_SEASON_FETCH["games"] = len(games)
    return games


def fetch_standings(season):
    """Wrapper: {} on failure, but the failure itself is never cached."""
    try:
        return _fetch_standings_cached(season)
    except Exception:
        return {}


@st.cache_data(ttl=604800, show_spinner=False)
def _fetch_standings_cached(season):
    """Season points for/against per team from ESPN standings (one call).
    Used as the PRIOR for early-season team ratings — season = the year the
    season ended. Returns {team_id: {pf, pa, gp, name}}."""
    data = safe_get(ESPN_STANDINGS, {"season": season, "seasontype": 2})
    out = {}

    def walk(node):
        for ent in ((node.get("standings") or {}).get("entries") or []):
            t = ent.get("team") or {}
            stats = {s.get("name") or s.get("type"): s.get("value")
                     for s in ent.get("stats", []) if isinstance(s, dict)}
            pf, pa = _num(stats.get("avgPointsFor")), _num(stats.get("avgPointsAgainst"))
            gp = _num(stats.get("gamesPlayed")) or (_num(stats.get("wins")) + _num(stats.get("losses")))
            if t.get("id") and pf > 0 and pa > 0:
                out[str(t["id"])] = {"pf": pf, "pa": pa, "gp": gp,
                                     "name": t.get("displayName", "")}
        for child in node.get("children", []) or []:
            walk(child)

    walk(data)
    if not out:
        raise FetchFailed(f"standings {season}")
    return out


# --- Injuries, rosters, players -----------------------------------------------

def _athlete_id(a):
    if not a:
        return ""
    if a.get("id"):
        return str(a["id"])
    for ln in a.get("links", []) or []:
        m = re.search(r"/id/(\d+)", ln.get("href", ""))
        if m:
            return m.group(1)
    return ""


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_injuries():
    """Current league injury report. CURRENT ONLY — ESPN doesn't serve a
    historical report, so past-date rebuilds run without it (and say so).
    Returns {team_id: [{id, name, status, detail}]}."""
    data = safe_get(f"{ESPN_SITE}/injuries")
    out = {}
    for team in data.get("injuries", []) or []:
        tid = str(team.get("id") or (team.get("team") or {}).get("id") or "")
        lst = []
        for inj in team.get("injuries", []) or []:
            a = inj.get("athlete") or {}
            lst.append({"id": _athlete_id(a), "name": a.get("displayName", ""),
                        "status": inj.get("status") or (inj.get("type") or {}).get("description", ""),
                        "detail": inj.get("shortComment", "")})
            if not tid:
                tid = str(((a.get("team") or {}).get("id")) or "")
        if tid:
            out.setdefault(tid, []).extend(lst)
    return out


OUT_STATUSES = ("out", "doubtful", "suspension", "suspended")
QUESTIONABLE_STATUSES = ("questionable", "day-to-day", "game time decision", "probable")


def injury_status(team_injuries, athlete_id=None, name=None):
    """'' if healthy, else the reported status string for this player."""
    nn = norm_name(name) if name else None
    for p in team_injuries or []:
        if (athlete_id and p.get("id") == str(athlete_id)) or (nn and norm_name(p.get("name")) == nn):
            return p.get("status", "") or ""
    return ""


def is_out(status):
    s = (status or "").lower()
    return any(k in s for k in OUT_STATUSES)


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_team_roster(team_id):
    """[{id, name}] for a team's current roster."""
    data = safe_get(f"{ESPN_SITE}/teams/{team_id}/roster")
    out = []
    for a in data.get("athletes", []) or []:
        if isinstance(a, dict) and "items" in a:  # grouped by position on some sports
            for x in a.get("items", []):
                out.append({"id": str(x.get("id", "")), "name": x.get("displayName", "")})
        elif isinstance(a, dict):
            out.append({"id": str(a.get("id", "")), "name": a.get("displayName", "")})
    return [p for p in out if p["id"]]


@st.cache_data(ttl=10800, show_spinner=False)
def fetch_player_gamelog(athlete_id, season):
    """One player's game-by-game lines for a season (regular season + playoffs,
    preseason excluded), oldest first. One call per player per season, so it's
    cached hard and only ever called for players actually being modelled.
    Returns [{event_id, et_date, min, pts, reb, ast, fg3}]."""
    if not athlete_id:
        return []
    data = safe_get(f"{ESPN_COMMON}/athletes/{athlete_id}/gamelog", {"season": season})
    labels, keys = data.get("labels") or [], data.get("names") or []
    events = data.get("events") or {}
    games, seen = [], set()
    for stype in data.get("seasonTypes", []) or []:
        if "preseason" in (stype.get("displayName") or "").lower():
            continue
        for cat in stype.get("categories", []) or []:
            for e in cat.get("events", []) or []:
                eid = str(e.get("eventId", ""))
                stats = e.get("stats") or []
                if not eid or eid in seen or not stats:
                    continue
                seen.add(eid)
                meta = events.get(eid) or {}
                d = to_et_date(meta.get("gameDate"))
                line = _line_from_stats(labels, keys, stats)
                line.update({"event_id": eid, "et_date": str(d) if d else ""})
                games.append(line)
    games.sort(key=lambda g: g["et_date"])
    return games


def _parse_box(data):
    out = []
    for team in ((data.get("boxscore") or {}).get("players") or []):
        tid = str((team.get("team") or {}).get("id", ""))
        for block in team.get("statistics", []) or []:
            labels = block.get("labels") or []
            keys = block.get("keys") or block.get("names") or []
            for a in block.get("athletes", []) or []:
                stats = a.get("stats") or []
                ath = a.get("athlete") or {}
                if a.get("didNotPlay") or not stats:
                    continue
                line = _line_from_stats(labels, keys, stats)
                line.update({"athlete_id": _athlete_id(ath), "name": ath.get("displayName", ""),
                             "team_id": tid, "starter": bool(a.get("starter"))})
                if line["min"] > 0:
                    out.append(line)
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_box_final(event_id):
    """Player box score for a FINISHED game — cached a day, it never changes."""
    if not event_id:
        return []
    return _parse_box(safe_get(f"{ESPN_SITE}/summary", {"event": event_id}))


@st.cache_data(ttl=60, show_spinner=False)
def fetch_game_live(event_id):
    """(game_state_dict, box_rows) for a game in any state, 60s TTL so a live
    check mid-game is current."""
    if not event_id:
        return {}, []
    data = safe_get(f"{ESPN_SITE}/summary", {"event": event_id})
    state = {}
    comp = ((data.get("header") or {}).get("competitions") or [{}])[0]
    if comp:
        state = _parse_event({"id": event_id, "competitions": [comp],
                              "date": comp.get("date")})
    return state, _parse_box(data)


# --- The Odds API -----------------------------------------------------------------

def get_secret(name, default=""):
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    import os
    return os.environ.get(name, default)


def _odds_call(path, params, timeout=20):
    key = get_secret("ODDS_API_KEY")
    if not key:
        return None, {"error": "No ODDS_API_KEY found in Streamlit secrets."}
    try:
        r = req.get(f"{ODDS_BASE}{path}", params={"apiKey": key, **params}, timeout=timeout)
    except Exception as e:
        return None, {"error": f"Request failed: {e}"}
    meta = {"status": r.status_code, "remaining": r.headers.get("x-requests-remaining"),
            "used": r.headers.get("x-requests-used"), "last": r.headers.get("x-requests-last"),
            "error": ""}
    if r.status_code != 200:
        meta["error"] = f"HTTP {r.status_code}: {r.text[:300]}"
        return None, meta
    try:
        return r.json(), meta
    except Exception as e:
        meta["error"] = f"Bad JSON: {e}"
        return None, meta


@st.cache_data(ttl=900, show_spinner=False)
def fetch_nba_odds(regions="uk", markets="h2h,spreads,totals"):
    """Live game lines. Costs markets x regions credits (3 on one region).
    Cached 15 min to protect quota."""
    data, meta = _odds_call(f"/sports/{SPORT_KEY}/odds",
                            {"regions": regions, "markets": markets,
                             "oddsFormat": "decimal", "dateFormat": "iso"})
    return (data or []), meta


@st.cache_data(ttl=900, show_spinner=False)
def fetch_nba_events():
    """Upcoming NBA events (ids + teams + tip times). FREE — the events
    endpoint doesn't cost credits, so props use this to find event ids
    instead of paying for a game-lines pull."""
    data, meta = _odds_call(f"/sports/{SPORT_KEY}/events", {"dateFormat": "iso"})
    return (data or []), meta


@st.cache_data(ttl=900, show_spinner=False)
def fetch_event_props(event_id, markets, regions="us"):
    """Player props for one game: 1 credit per market per region."""
    return _odds_call(f"/sports/{SPORT_KEY}/events/{event_id}/odds",
                      {"regions": regions, "markets": markets,
                       "oddsFormat": "decimal", "dateFormat": "iso"})


def snapshot_iso_for_start(start_iso, lead_minutes=60):
    """Timestamp `lead_minutes` before tip, floored to the 5-minute grid
    historical snapshots are stored on."""
    dt = parse_iso(start_iso)
    if dt is None:
        return None
    snap = dt.astimezone(_UTC) - timedelta(minutes=lead_minutes) if _UTC else dt
    snap = snap.replace(minute=(snap.minute // 5) * 5, second=0, microsecond=0)
    return snap.strftime("%Y-%m-%dT%H:%M:%SZ")


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_odds(snapshot_iso, regions="uk", markets="h2h,spreads,totals"):
    """Game lines as they stood at a past moment. COST: 10 credits per market
    per region — 30 for the three game markets on one region."""
    data, meta = _odds_call(f"/historical/sports/{SPORT_KEY}/odds",
                            {"regions": regions, "markets": markets, "oddsFormat": "decimal",
                             "dateFormat": "iso", "date": snapshot_iso})
    if data is None:
        return [], meta
    meta["snapshot_actual"] = data.get("timestamp")
    return data.get("data", []), meta


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_historical_event_props(event_id, markets, snapshot_iso, regions="us"):
    """Player props for one game at a past moment. Historical event odds bill
    at the historical rate (10 per market per region), so with five prop
    markets this can be ~50 credits a game — watch the quota line."""
    data, meta = _odds_call(f"/historical/sports/{SPORT_KEY}/events/{event_id}/odds",
                            {"regions": regions, "markets": markets, "oddsFormat": "decimal",
                             "dateFormat": "iso", "date": snapshot_iso}, timeout=25)
    if data is None:
        return None, meta
    meta["snapshot_actual"] = data.get("timestamp")
    return data.get("data", data), meta


def historical_slate_odds(games, regions="uk", lead_minutes=60):
    """Game lines for a past slate, each game priced at ITS OWN snapshot.
    Games tipping at the same time share one call (NBA slates cluster heavily
    at 7:00/7:30/10:00 ET, so this is usually 2-4 calls, not one per game).
    `games` are ESPN game dicts. Returns (odds_list, meta, snapshots)."""
    groups = {}
    for g in games:
        snap = snapshot_iso_for_start(g.get("start_iso"), lead_minutes)
        if snap:
            groups.setdefault(snap, []).append(g)
    merged, metas, snaps, seen = [], [], [], set()
    for snap in sorted(groups):
        want = {(nickname(g.get("home_name")), nickname(g.get("away_name"))) for g in groups[snap]}
        data, meta = fetch_historical_odds(snap, regions=regions)
        metas.append(meta)
        snaps.append({"requested": snap, "actual": meta.get("snapshot_actual"),
                      "games": len(groups[snap]), "error": meta.get("error", "")})
        for ev in data:
            pair = (nickname(ev.get("home_team")), nickname(ev.get("away_team")))
            if pair in want and ev.get("id") not in seen:
                merged.append(ev)
                seen.add(ev.get("id"))
    combined = {"used": metas[-1].get("used") if metas else None,
                "remaining": metas[-1].get("remaining") if metas else None,
                "error": next((m.get("error") for m in metas if m.get("error")), ""),
                "calls": len(groups), "credits_estimate": len(groups) * 30}
    return merged, combined, snaps
