"""NBA builders — the functions the pages call. Mirrors mlb_core's
build_game_edges / build_prop_edges / build_most_likely / suggested bets /
results / CLV, adapted for basketball."""
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pandas as pd

from nba_data import *   # noqa: F401,F403
from nba_model import *  # noqa: F401,F403

GAME_REGIONS = "uk"   # UK books for game lines (same as the MLB app)
PROP_REGIONS = "us"   # props only really exist on US books
GAME_MARKETS = ["Moneyline", "Spread", "Total"]
PROP_LABELS = ["Points", "Rebounds", "Assists", "Threes", "PRA"]
FORM_GAMES = 10


# --- Shared slate context ---------------------------------------------------------------

def slate_context(sel_date, use_injuries=True):
    """Everything a build needs about a date, fetched once: that day's games,
    as-of team ratings, back-to-backs and (current slates only) injuries."""
    ds = str(sel_date)
    season = season_for(sel_date)
    season_games = fetch_season_results(season, ds)
    prior = fetch_standings(season - 1)
    ratings = compute_team_ratings(season_games, prior)
    live = sel_date >= today_et()
    return {
        "ds": ds, "season": season, "date": sel_date, "live": live,
        "games": fetch_scoreboard(ds), "season_games": season_games,
        "ratings": ratings, "b2b": b2b_teams(season_games, sel_date),
        "injuries": fetch_injuries() if (use_injuries and live) else {},
        "injuries_on": bool(use_injuries and live),
    }


def _rating(ctx, tid):
    lg = ctx["ratings"]["lg"]
    return ctx["ratings"]["teams"].get(tid) or {"off": 1.0, "def": 1.0, "gp": 0, "ppg": lg}


def prefetch_gamelogs(pairs):
    """Warm the game-log cache in parallel — one ESPN call per (player, season),
    which is the slow part of every prop build. 8 at a time is polite and
    turns a ~60s cold load into a few seconds."""
    pairs = list({(str(a), int(s)) for a, s in pairs if a})
    if not pairs:
        return
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda p: fetch_player_gamelog(p[0], p[1]), pairs))


def player_games(aid, season, before_str):
    """(current-season games, last-season games). Last season is only pulled
    while the current sample is small — from ~25 games in it's skipped,
    halving the calls for the rest of the year."""
    cur = fetch_player_gamelog(aid, season)
    n_before = sum(1 for g in cur if g["et_date"] and g["et_date"] < before_str)
    prior = fetch_player_gamelog(aid, season - 1) if n_before < 25 else []
    return cur, prior


def team_injury_impact(ctx, tid):
    """(fraction of offence lost, [descriptions of absent players], [questionable]).
    A player who hasn't played at all this season while his team has 5+ games
    is skipped — his absence is already baked into the team's numbers."""
    inj = ctx["injuries"].get(tid, [])
    if not inj:
        return 0.0, [], []
    r = _rating(ctx, tid)
    lost, outs, qs = 0.0, [], []
    for p in inj:
        st_ = p.get("status", "")
        if not is_out(st_):
            if any(k in st_.lower() for k in QUESTIONABLE_STATUSES):
                qs.append(f"{p['name']} ({st_})")
            continue
        if not p.get("id"):
            continue
        cur, prior = player_games(p["id"], ctx["season"], ctx["ds"])
        cur_b = [g for g in cur if g["et_date"] < ctx["ds"] and g["min"] > 0]
        if r["gp"] >= 5 and not cur_b:
            continue
        pool = cur_b if len(cur_b) >= 3 else (cur_b + [g for g in prior if g["min"] > 0][-40:])
        if not pool:
            continue
        ppg = sum(g["pts"] for g in pool) / len(pool)
        if ppg < 5:
            continue
        lost += ppg
        outs.append(f"{p['name']} ({ppg:.0f} ppg)")
    frac = min(INJURY_LOSS_SHARE * lost / (r["ppg"] or LEAGUE_PPG_DEFAULT), INJURY_CAP)
    return frac, outs, qs


def model_for_game(ctx, g, total_line=None, home_point=None):
    """Run the game model for one ESPN game dict. Returns (model, info)."""
    hid, aid = g["home_id"], g["away_id"]
    if ctx["injuries_on"]:
        prefetch_gamelogs([(p["id"], s) for t in (hid, aid) for p in ctx["injuries"].get(t, [])
                           if is_out(p.get("status")) for s in (ctx["season"], ctx["season"] - 1)])
    h_inj, h_out, h_q = team_injury_impact(ctx, hid)
    a_inj, a_out, a_q = team_injury_impact(ctx, aid)
    rh, ra = _rating(ctx, hid), _rating(ctx, aid)
    mdl = game_model(rh, ra, ctx["ratings"]["lg"], total_line=total_line, home_point=home_point,
                     home_b2b=hid in ctx["b2b"], away_b2b=aid in ctx["b2b"],
                     home_inj=h_inj, away_inj=a_inj)
    notes = []
    if min(rh["gp"], ra["gp"]) < 10:
        notes.append(f"early season — ratings lean on last season ({min(rh['gp'], ra['gp'])} games in)")
    if not ctx["injuries_on"]:
        notes.append("no injury adjustment (past date or toggled off)")
    if h_q or a_q:
        notes.append("questionable: " + ", ".join(h_q + a_q))
    info = {"rh": rh, "ra": ra, "h_out": h_out, "a_out": a_out, "h_q": h_q, "a_q": a_q,
            "h_b2b": hid in ctx["b2b"], "a_b2b": aid in ctx["b2b"], "notes": notes}
    return mdl, info


def game_label(g):
    return f"{g.get('away_abbr') or g.get('away_name')} @ {g.get('home_abbr') or g.get('home_name')}"


def _match_espn(games, home_name, away_name):
    pair = (nickname(home_name), nickname(away_name))
    return next((g for g in games if (nickname(g["home_name"]), nickname(g["away_name"])) == pair), None)


def _team_reason(side, info):
    me, opp = (info["rh"], info["ra"]) if side == "home" else (info["ra"], info["rh"])
    my_out, opp_out = (info["h_out"], info["a_out"]) if side == "home" else (info["a_out"], info["h_out"])
    my_b2b, opp_b2b = (info["h_b2b"], info["a_b2b"]) if side == "home" else (info["a_b2b"], info["h_b2b"])
    bits = []
    if me["off"] - opp["off"] >= 0.03:
        bits.append(f"better offence ({me['off']*100:.0f} vs {opp['off']*100:.0f} index)")
    if opp["def"] - me["def"] >= 0.03:
        bits.append(f"better defence ({me['def']*100:.0f} vs {opp['def']*100:.0f} allowed index)")
    if opp_b2b and not my_b2b:
        bits.append("opponent on a back-to-back")
    if opp_out:
        bits.append("opponent missing " + ", ".join(opp_out[:2]))
    if side == "home" and not bits:
        bits.append("home court")
    if my_b2b:
        bits.append("⚠️ own back-to-back")
    if my_out:
        bits.append("⚠️ missing " + ", ".join(my_out[:2]))
    if not bits:
        return "Edge from market pricing, not a standout matchup"
    s = "; ".join(bits[:3])
    return s[:1].upper() + s[1:]


def _total_reason(direction, mdl, lg, info):
    bits = [f"model projects {mdl['total']:.0f} total ({mdl['exp_away']:.0f}-{mdl['exp_home']:.0f})"]
    pace = mdl["total"] / (2 * lg) if lg else 1
    if direction == "Over" and pace >= 1.02:
        bits.append("both sides score above league average")
    if direction == "Under" and pace <= 0.98:
        bits.append("below-average scoring matchup")
    if info["h_out"] or info["a_out"]:
        bits.append("absences: " + ", ".join((info["h_out"] + info["a_out"])[:2]))
    return "; ".join(bits).capitalize()


# --- Game edges -----------------------------------------------------------------------

def build_game_edges(sel_date, odds_override=None, meta_override=None, use_injuries=True):
    """Match the slate to UK odds, run the model, return (df, note, meta)."""
    ctx = slate_context(sel_date, use_injuries)
    games = ctx["games"]
    if not games:
        return None, "No NBA games scheduled for this date.", {}
    if odds_override is not None:
        odds, meta = odds_override, (meta_override or {})
    else:
        odds, meta = fetch_nba_odds(regions=GAME_REGIONS)
    if meta.get("error"):
        return None, meta["error"], meta
    odds = [o for o in odds if to_et_date(o.get("commence_time")) == sel_date]
    if not odds:
        return None, "No odds posted for this slate yet.", meta

    lg = ctx["ratings"]["lg"]
    rows, unmatched = [], []
    for og in odds:
        g = _match_espn(games, og.get("home_team"), og.get("away_team"))
        if g is None:
            unmatched.append(f"{og.get('away_team')} @ {og.get('home_team')}")
            continue
        cons = consolidate_odds(og)
        mdl, info = model_for_game(ctx, g, cons["total_line"], cons["home_point"])
        gl, ct = game_label(g), og.get("commence_time") or g["start_iso"]
        proj = f"Proj {g['away_abbr']} {mdl['exp_away']:.0f} – {g['home_abbr']} {mdl['exp_home']:.0f}"
        base = {"Game": gl, "EventID": g["event_id"], "_ct": ct, "Start": to_uk_time(ct),
                "US Date": et_date_str(ct), "Proj": proj, "DataNotes": "; ".join(info["notes"])}

        def add(market, sel, p_raw, fair, best, reason, side=None, thr=None, direction=None):
            p = calibrate(p_raw, market)
            e, v = edge_ev(p, fair, best)
            if p is None or fair is None or not best:
                return
            rows.append({**base, "Market": market, "Selection": sel, "Model %": p * 100,
                         "Fair %": fair * 100, "Edge": e, "Odds": best, "EV %": v,
                         "Reason": reason, "Side": side, "Threshold": thr, "Direction": direction})

        fh, fa = devig_two(cons["ml_home"], cons["ml_away"])
        add("Moneyline", g["home_name"], mdl["p_home_ml"], fh, cons["ml_home_best"],
            _team_reason("home", info), "home")
        add("Moneyline", g["away_name"], mdl["p_away_ml"], fa, cons["ml_away_best"],
            _team_reason("away", info), "away")
        hp = cons["home_point"]
        if hp is not None:
            sh, sa = devig_two(cons["sp_home"], cons["sp_away"])
            add("Spread", f"{g['home_name']} {hp:+g}", mdl.get("p_home_cover"), sh,
                cons["sp_home_best"], _team_reason("home", info), "home", hp)
            add("Spread", f"{g['away_name']} {-hp:+g}", mdl.get("p_away_cover"), sa,
                cons["sp_away_best"], _team_reason("away", info), "away", -hp)
        tl = cons["total_line"]
        if tl is not None:
            fo, fu = devig_two(cons["over"], cons["under"])
            add("Total", f"Over {tl:g}", mdl.get("p_over"), fo, cons["over_best"],
                _total_reason("Over", mdl, lg, info), None, tl, "Over")
            add("Total", f"Under {tl:g}", mdl.get("p_under"), fu, cons["under_best"],
                _total_reason("Under", mdl, lg, info), None, tl, "Under")

    if not rows:
        return None, "No matched games with usable odds.", meta
    df = pd.DataFrame(rows)
    for c in ("Model %", "Fair %", "Edge", "EV %"):
        df[c] = df[c].round(1)
    df = df.sort_values(["_ct", "Game"]).reset_index(drop=True)
    note = f"Couldn't match odds for: {', '.join(unmatched)}" if unmatched else ""
    return df, note, meta


# --- Player props -------------------------------------------------------------------------

def _roster_lookup(team_ids):
    """norm_name -> (athlete_id, team_id) across the given rosters, plus a
    last-name index used only when it's unambiguous."""
    full, last = {}, {}
    for tid in team_ids:
        for p in fetch_team_roster(tid):
            nn = norm_name(p["name"])
            full[nn] = (p["id"], tid)
            ln = nn.split()[-1] if nn else ""
            last.setdefault(ln, []).append((p["id"], tid))
    return full, last


def _find_player(name, full, last):
    nn = norm_name(name)
    if nn in full:
        return full[nn]
    cands = last.get(nn.split()[-1] if nn else "", [])
    return cands[0] if len(cands) == 1 else (None, None)


def _prop_context(ctx, g, mdl, tid):
    side = "home" if tid == g["home_id"] else "away"
    exp = mdl["exp_home"] if side == "home" else mdl["exp_away"]
    r = _rating(ctx, tid)
    return side, exp, r["ppg"]


def _prop_reason(stat, mu, proj, ratio, total, lg):
    unit = {"pts": "pts", "reb": "reb", "ast": "ast", "fg3": "threes", "pra": "PRA"}[stat]
    bits = [f"projects {mu:.1f} {unit} in ~{proj['min']:.0f} min"]
    last5 = proj["last_games"][-5:]
    if last5:
        if stat == "pra":
            avg = sum(x["pts"] + x["reb"] + x["ast"] for x in last5) / len(last5)
        else:
            avg = sum(x[stat] for x in last5) / len(last5)
        bits.append(f"last {len(last5)}: {avg:.1f} avg")
    if ratio >= 1.03:
        bits.append(f"soft matchup (team projected {(ratio - 1) * 100:+.0f}% vs norm)")
    elif ratio <= 0.97:
        bits.append(f"tough matchup (team projected {(ratio - 1) * 100:+.0f}% vs norm)")
    if lg and total / (2 * lg) >= 1.03:
        bits.append(f"high-scoring game (proj {total:.0f})")
    s = "; ".join(bits[:3])
    return s[:1].upper() + s[1:]


def _prop_conditions(proj, status, b2b, blowout):
    bits = []
    if status:
        bits.append(f"⚠️ {status}")
    if b2b:
        bits.append("back-to-back")
    if blowout:
        bits.append("blowout risk (minutes trimmed)")
    if proj["n_cur"] < 5:
        bits.append(f"thin current-season sample ({proj['n_cur']} games; leans on last season)")
    return " · ".join(bits)


def build_prop_edges(sel_date, max_games=6, snapshot_by_event=None,
                     odds_override=None, meta_override=None, use_injuries=True):
    """Prop value bets for the slate. Returns (df, meta, note)."""
    ctx = slate_context(sel_date, use_injuries)
    games = ctx["games"]
    if not games:
        return None, {}, "No NBA games scheduled for this date."
    if odds_override is not None:
        events, meta = odds_override, (meta_override or {})
    else:
        events, meta = fetch_nba_events()
    if meta.get("error") and not events:
        return None, meta, f"Odds request failed: {meta['error']}"
    events = sorted([e for e in events if to_et_date(e.get("commence_time")) == sel_date],
                    key=lambda e: e.get("commence_time") or "")
    if not events:
        return None, meta, "No events posted for this slate yet — try closer to tip-off."

    markets = ",".join(PROP_MARKETS)
    cols = ["Market", "Light", "Player", "Game", "Start", "Line", "Model %", "Market %",
            "Edge", "Best over", "Proj", "Reason", "Conditions", "EventID", "PlayerID",
            "MarketKey", "Point", "Team"]
    rows, analysed, last_meta, unmatched = [], 0, meta, []
    lg = ctx["ratings"]["lg"]

    for ev in events:
        if analysed >= max_games:
            break
        g = _match_espn(games, ev.get("home_team"), ev.get("away_team"))
        if g is None:
            unmatched.append(f"{ev.get('away_team')} @ {ev.get('home_team')}")
            continue
        snap = (snapshot_by_event or {}).get(ev.get("id"))
        if snap:
            event, em = fetch_historical_event_props(ev.get("id"), markets, snap, regions=PROP_REGIONS)
        else:
            event, em = fetch_event_props(ev.get("id"), markets, regions=PROP_REGIONS)
        analysed += 1
        if em.get("remaining"):
            last_meta = em
        if em.get("error") or not event:
            continue

        full, last = _roster_lookup([g["home_id"], g["away_id"]])
        per_market = {mk: consolidate_prop(event, mk) for mk in PROP_MARKETS}
        ids = {_find_player(p, full, last) for mk in per_market for p in per_market[mk]}
        prefetch_gamelogs([(aid, s) for aid, _ in ids if aid
                           for s in (ctx["season"], ctx["season"] - 1)])

        mdl, info = model_for_game(ctx, g)
        blowout = abs(mdl["margin"]) >= BLOWOUT_MARGIN
        gl, start = game_label(g), to_uk_time(ev.get("commence_time"))
        proj_cache = {}
        for mk, players in per_market.items():
            label, stat = PROP_LABEL[mk], PROP_STAT[mk]
            for player, od in players.items():
                aid, tid = _find_player(player, full, last)
                if not aid:
                    continue
                status = injury_status(ctx["injuries"].get(tid, []), aid, player)
                if is_out(status):
                    continue
                if aid not in proj_cache:
                    cur, prior = player_games(aid, ctx["season"], ctx["ds"])
                    proj_cache[aid] = project_player(cur, prior, ctx["ds"])
                proj = proj_cache[aid]
                if proj is None:
                    continue
                side, exp, ppg = _prop_context(ctx, g, mdl, tid)
                means = matchup_means(proj, exp, ppg, mdl["total"], lg, blowout)
                mu = means[stat]
                mp = calibrate(prop_p_over(mu, proj["disp"][stat], od["point"]), label)
                bp = market_prob(od["over"], od["under"])
                if mp is None or bp is None:
                    continue
                edge = (mp - bp) * 100
                if not in_value_band(edge, label):
                    continue
                rows.append([label, classify_pick(edge, mp * 100, label), player, gl, start,
                             f"O{od['point']:g}", round(mp * 100, 1), round(bp * 100, 1),
                             round(edge, 1), od["over_best"], round(mu, 1),
                             _prop_reason(stat, mu, proj, exp / ppg if ppg else 1, mdl["total"], lg),
                             _prop_conditions(proj, status, tid in ctx["b2b"], blowout),
                             g["event_id"], aid, mk, od["point"], tid])

    note = f"Analysed {analysed} game(s)."
    if unmatched:
        note += f" Couldn't match: {', '.join(unmatched[:4])}."
    if not ctx["injuries_on"]:
        note += " No injury adjustment for this run."
    return pd.DataFrame(rows, columns=cols), last_meta, note


def _candidate_players(ctx, tid, n=9):
    """Who to project for a team when there are no odds to tell us: the top-n
    by minutes in the team's most recent completed game (falls back to the
    full roster before a team has played)."""
    last = next((g for g in reversed(ctx["season_games"])
                 if tid in (g["home_id"], g["away_id"])), None)
    if last:
        box = [b for b in fetch_box_final(last["event_id"]) if b["team_id"] == tid]
        box.sort(key=lambda b: -b["min"])
        if box:
            return [(b["athlete_id"], b["name"]) for b in box[:n]]
    return [(p["id"], p["name"]) for p in fetch_team_roster(tid)]


def build_most_likely(sel_date, max_games=6, use_injuries=True):
    """Free ranking — no odds, no quota: each rotation player's raw model
    probability of reaching milestones (20+ pts, 8+ reb, ...). Returns (df, note)."""
    ctx = slate_context(sel_date, use_injuries)
    games = [g for g in ctx["games"] if g["state"] != "post"] or ctx["games"]
    if not games:
        return None, "No NBA games scheduled for this date."
    games = sorted(games, key=lambda g: g["start_iso"])[:max_games]
    lg = ctx["ratings"]["lg"]
    rows = []
    for g in games:
        mdl, info = model_for_game(ctx, g)
        blowout = abs(mdl["margin"]) >= BLOWOUT_MARGIN
        cands = {tid: _candidate_players(ctx, tid) for tid in (g["home_id"], g["away_id"])}
        prefetch_gamelogs([(a, s) for lst in cands.values() for a, _ in lst
                           for s in (ctx["season"], ctx["season"] - 1)])
        for tid, players in cands.items():
            for aid, name in players:
                status = injury_status(ctx["injuries"].get(tid, []), aid, name)
                if is_out(status):
                    continue
                cur, prior = player_games(aid, ctx["season"], ctx["ds"])
                proj = project_player(cur, prior, ctx["ds"])
                if proj is None:
                    continue
                side, exp, ppg = _prop_context(ctx, g, mdl, tid)
                means = matchup_means(proj, exp, ppg, mdl["total"], lg, blowout)
                cond = _prop_conditions(proj, status, tid in ctx["b2b"], blowout)
                for label in PROP_LABELS:
                    mk = LABEL_TO_KEY[label]
                    stat = PROP_STAT[mk]
                    for k in MILESTONES[label]:
                        p = p_at_least(means[stat], proj["disp"][stat], k)
                        rows.append([label, k, name, game_label(g), to_uk_time(g["start_iso"]),
                                     round(p * 100, 1), round(means[stat], 1),
                                     round(proj["min"]), cond, aid, mk])
    if not rows:
        return None, "No projectable players yet for these games."
    df = pd.DataFrame(rows, columns=["Market", "Threshold", "Player", "Game", "Start", "Value",
                                     "Expected", "Min", "Conditions", "PlayerID", "MarketKey"])
    return df, f"Projected rotation players across {len(games)} game(s)."


def form_streak(player_id, season, market_key, line, games=FORM_GAMES):
    """Did the player clear `line` in each of his last `games` games?
    Returns (hits, played, symbols) or (None, 0, '')."""
    stat = PROP_STAT.get(market_key)
    if not stat or not player_id:
        return None, 0, ""
    log = [g for g in fetch_player_gamelog(str(player_id), season) if g["min"] > 0]
    if len(log) < games:
        log = [g for g in fetch_player_gamelog(str(player_id), season - 1) if g["min"] > 0] + log
    log = log[-games:]
    if not log:
        return None, 0, ""
    hits, syms = 0, []
    for g in log:
        val = (g["pts"] + g["reb"] + g["ast"]) if stat == "pra" else g[stat]
        ok = val > float(line)
        hits += ok
        syms.append("✅" if ok else "❌")
    return hits, len(log), "".join(syms)


# --- Slate overview (the NBA version of MLB's "Load Today's Slate") ---------------------

def build_slate_overview(sel_date, use_injuries=True):
    ctx = slate_context(sel_date, use_injuries)
    out = []
    for g in sorted(ctx["games"], key=lambda x: x["start_iso"]):
        mdl, info = model_for_game(ctx, g)
        out.append({"game": game_label(g), "start": to_uk_time(g["start_iso"]),
                    "state": g["detail"], "home": g["home_name"], "away": g["away_name"],
                    "proj_home": mdl["exp_home"], "proj_away": mdl["exp_away"],
                    "p_home": mdl["p_home_ml"], "total": mdl["total"],
                    "rh": info["rh"], "ra": info["ra"], "h_b2b": info["h_b2b"],
                    "a_b2b": info["a_b2b"], "h_out": info["h_out"], "a_out": info["a_out"],
                    "h_q": info["h_q"], "a_q": info["a_q"], "notes": info["notes"]})
    return out, ctx


# --- Suggested bets ---------------------------------------------------------------------------
MARKET_TRUST_TIER = {
    # PROVISIONAL — no tracked NBA results yet. Start conservative and re-weight
    # from real slips the way the MLB tiers were. Threes is the most volatile
    # prop (a 2.5 line swings on one shot), so it gets the smallest slice.
    "Moneyline": 2.0, "Spread": 1.5, "Totals": 1.5, "Points": 1.5,
    "Rebounds": 1.0, "Assists": 1.0, "PRA": 1.0, "Threes": 0.5,
}


def _normalize_game_row(r):
    return {"label": r["Selection"], "game": r["Game"], "odds": float(r["Odds"]),
            "model_pct": float(r["Model %"]), "reason": r.get("Reason", ""), "kind": "game",
            "market": r["Market"], "event_id": r.get("EventID"), "side": r.get("Side"),
            "threshold": r.get("Threshold"), "direction": r.get("Direction")}


def _normalize_prop_row(r):
    return {"label": f"{r['Player']} {r['Line']} {r['Market']}", "game": r["Game"],
            "odds": float(r["Best over"]), "model_pct": float(r["Model %"]),
            "reason": r.get("Reason", ""), "kind": "prop", "market": r["Market"],
            "event_id": r.get("EventID"), "player_id": r.get("PlayerID"),
            "market_key": r.get("MarketKey"), "point": r.get("Point")}


def _best_combo(cands, n_legs, normalize):
    """Highest Model % legs from DIFFERENT games."""
    if cands is None or cands.empty:
        return None
    chosen, used = [], set()
    for _, row in cands.sort_values("Model %", ascending=False).iterrows():
        leg = normalize(row)
        if leg["game"] in used:
            continue
        chosen.append(leg)
        used.add(leg["game"])
        if len(chosen) == n_legs:
            break
    if len(chosen) < n_legs:
        return None
    odds = math.prod(l["odds"] for l in chosen)
    prob = math.prod(l["model_pct"] / 100 for l in chosen)
    return {"legs": chosen, "combined_odds": odds, "combined_prob": prob}


def build_suggested_bets(sel_date, prop_max_games=6, use_injuries=True):
    results, quota = {}, {}
    gdf, gnote, gmeta = build_game_edges(sel_date, use_injuries=use_injuries)
    quota["game"] = gmeta
    for label, out_key in (("Moneyline", "Moneyline"), ("Spread", "Spread"), ("Total", "Totals")):
        if gdf is not None and not gdf.empty:
            sub = gdf[(gdf["Market"] == label) & gdf["Edge"].apply(lambda e: in_value_band(e, label))]
            results[out_key] = {"double": _best_combo(sub, 2, _normalize_game_row),
                                "treble": _best_combo(sub, 3, _normalize_game_row),
                                "note": "" if not sub.empty else "No green/amber picks today."}
        else:
            results[out_key] = {"double": None, "treble": None, "note": gnote or "No game odds."}
    pdf, pmeta, pnote = build_prop_edges(sel_date, prop_max_games, use_injuries=use_injuries)
    quota["props"] = pmeta
    for label in PROP_LABELS:
        if pdf is not None and not pdf.empty:
            sub = pdf[pdf["Market"] == label]
            results[label] = {"double": _best_combo(sub, 2, _normalize_prop_row),
                              "treble": _best_combo(sub, 3, _normalize_prop_row),
                              "note": "" if not sub.empty else "No green/amber picks today."}
        else:
            results[label] = {"double": None, "treble": None, "note": pnote or "No prop odds."}
    return results, quota


def suggest_stakes(bankroll, markets):
    total = sum(MARKET_TRUST_TIER.get(m, 1.0) for m in markets)
    return {m: round(bankroll * MARKET_TRUST_TIER.get(m, 1.0) / total, 2) if total else 0.0
            for m in markets}


def evaluate_leg_status(leg):
    """(status, detail): pending | winning | losing | won | lost | unknown."""
    state, box = fetch_game_live(leg.get("event_id"))
    if not state:
        return "unknown", "Live data unavailable."
    if state.get("state") == "pre":
        return "pending", "Game hasn't tipped off yet."
    final = state.get("state") == "post"
    hs, as_ = state.get("home_score"), state.get("away_score")
    where = "Final" if final else f"Live (Q{state.get('period')} {state.get('clock')})"

    if leg["kind"] == "game":
        if hs is None or as_ is None:
            return "unknown", "Score unavailable."
        me, opp = (hs, as_) if leg.get("side") == "home" else (as_, hs)
        if leg["market"] == "Moneyline":
            diff = me - opp
        elif leg["market"] == "Spread":
            diff = me + float(leg.get("threshold") or 0) - opp
        else:  # Total
            tot, ln = hs + as_, float(leg.get("threshold") or 0)
            diff = (tot - ln) if leg.get("direction") == "Over" else (ln - tot)
            if not final and leg.get("direction") == "Over" and diff > 0:
                return "won", f"{where}: total {tot} already over {ln:g}."
        detail = f"{where}: {state.get('away_abbr')} {as_} – {state.get('home_abbr')} {hs}"
        if final:
            if diff == 0:
                return "unknown", detail + " — push, check your book."
            return ("won" if diff > 0 else "lost"), detail
        return ("winning" if diff > 0 else "losing"), detail

    stat = PROP_STAT.get(leg.get("market_key"))
    row = next((b for b in box if b["athlete_id"] == str(leg.get("player_id"))), None)
    if row is None:
        return ("lost" if final else "pending"), ("Didn't play." if final else "Not in the box score yet.")
    val = (row["pts"] + row["reb"] + row["ast"]) if stat == "pra" else row[stat]
    line = float(leg.get("point") or 0)
    if val > line:
        return "won", f"{where}: {val:g} — already over {line:g}."
    if final:
        return "lost", f"Final: {val:g} (needed over {line:g})."
    return "losing", f"{where}: {val:g} so far (needs over {line:g})."


def evaluate_combo_status(combo):
    legs = [evaluate_leg_status(l) for l in combo["legs"]]
    st_ = [s for s, _ in legs]
    if "lost" in st_:
        overall = "lost"
    elif all(s == "won" for s in st_):
        overall = "won"
    elif "unknown" in st_:
        overall = "unknown"
    else:
        overall = "alive"
    return overall, legs


# --- Results ------------------------------------------------------------------------------

def _finals(ctx):
    return [g for g in ctx["games"] if g["completed"] and g["home_score"] is not None]


def build_game_results(sel_date):
    """Model-only Moneyline reads vs real results (free). Spreads/totals need the
    line that was on offer, so they're graded in Priced mode only."""
    ctx = slate_context(sel_date, use_injuries=False)
    finals = _finals(ctx)
    if not finals:
        return None, "No completed games on this date yet."
    rows = []
    for g in finals:
        mdl, _ = model_for_game(ctx, g)
        hs, as_ = g["home_score"], g["away_score"]
        score = f"{g['away_abbr']} {as_} – {g['home_abbr']} {hs}"
        for side, name, p, won in (("home", g["home_name"], mdl["p_home_ml"], hs > as_),
                                   ("away", g["away_name"], mdl["p_away_ml"], as_ > hs)):
            rows.append({"Market": "Moneyline", "Player": name, "Line": "ML",
                         "Model %": round(calibrate(p, "Moneyline") * 100, 1),
                         "Actual": score, "Hit": bool(won), "Game": game_label(g), "Score": score,
                         "Proj": f"{mdl['exp_away']:.0f}-{mdl['exp_home']:.0f}"})
    return pd.DataFrame(rows), f"Scored {len(finals)} completed game(s) — Moneyline only in this mode."


def build_prop_results(sel_date, max_games=None):
    """Model-only prop reads: each rotation player's milestone probabilities
    (15+/25+ pts etc.) vs what he actually did. Free; uses only games before
    the date, so nothing from the future leaks in."""
    ctx = slate_context(sel_date, use_injuries=False)
    finals = _finals(ctx)[:max_games] if max_games else _finals(ctx)
    if not finals:
        return None, "No completed games on this date yet."
    lg = ctx["ratings"]["lg"]
    rows = []
    for g in finals:
        box = [b for b in fetch_box_final(g["event_id"]) if b["min"] >= 10]
        prefetch_gamelogs([(b["athlete_id"], s) for b in box for s in (ctx["season"], ctx["season"] - 1)])
        mdl, _ = model_for_game(ctx, g)
        blowout = abs(mdl["margin"]) >= BLOWOUT_MARGIN
        score = f"{g['away_abbr']} {g['away_score']} – {g['home_abbr']} {g['home_score']}"
        for tid in (g["home_id"], g["away_id"]):
            players = sorted([b for b in box if b["team_id"] == tid], key=lambda b: -b["min"])[:8]
            for b in players:
                cur, prior = player_games(b["athlete_id"], ctx["season"], ctx["ds"])
                proj = project_player(cur, prior, ctx["ds"])
                if proj is None:
                    continue
                _, exp, ppg = _prop_context(ctx, g, mdl, tid)
                means = matchup_means(proj, exp, ppg, mdl["total"], lg, blowout)
                for label in PROP_LABELS:
                    stat = PROP_STAT[LABEL_TO_KEY[label]]
                    actual = (b["pts"] + b["reb"] + b["ast"]) if stat == "pra" else b[stat]
                    for k in RESULTS_MILESTONES[label]:
                        p = p_at_least(means[stat], proj["disp"][stat], k)
                        rows.append({"Market": label, "Player": b["name"], "Line": f"{k}+",
                                     "Model %": round(p * 100, 1), "Actual": actual,
                                     "Hit": bool(actual >= k), "Game": game_label(g), "Score": score})
    if not rows:
        return None, "No box-score data available yet."
    return pd.DataFrame(rows), f"Scored {len(finals)} completed game(s) on {sel_date}."


def build_priced_results(sel_date, max_games=4, lead_minutes=60):
    """Rebuild the value picks the app WOULD have shown, priced at real odds from
    `lead_minutes` before each tip, then grade them. Spends historical credits."""
    ctx = slate_context(sel_date, use_injuries=False)
    finals = _finals(ctx)
    if not finals:
        return None, "No completed games on this date yet.", {}
    odds, ometa, snaps = historical_slate_odds(finals, regions=GAME_REGIONS, lead_minutes=lead_minutes)
    if not odds:
        return None, f"No historical odds returned. {ometa.get('error', '')}".strip(), {"snapshots": snaps}
    snap_by_event = {ev.get("id"): snapshot_iso_for_start(ev.get("commence_time"), lead_minutes)
                     for ev in odds}
    by_id = {g["event_id"]: g for g in finals}

    game_rows = []
    gdf, gnote, _ = build_game_edges(sel_date, odds_override=odds, meta_override=ometa,
                                     use_injuries=False)
    if isinstance(gdf, pd.DataFrame):
        for _, r in gdf.iterrows():
            g = by_id.get(r["EventID"])
            if g is None:
                continue
            hs, as_ = g["home_score"], g["away_score"]
            me, opp = (hs, as_) if r["Side"] == "home" else (as_, hs)
            if r["Market"] == "Moneyline":
                diff = me - opp
            elif r["Market"] == "Spread":
                diff = me + float(r["Threshold"]) - opp
            else:
                diff = (hs + as_ - float(r["Threshold"])) * (1 if r["Direction"] == "Over" else -1)
            if diff == 0:
                continue  # push — stake returned, nothing to grade
            hit = diff > 0
            game_rows.append({"Market": r["Market"], "Light": classify_pick(r["Edge"], r["Model %"], r["Market"]),
                              "Player": r["Selection"], "Game": r["Game"], "Line": r["Threshold"] or "",
                              "Model %": r["Model %"], "Market %": r["Fair %"], "Edge": r["Edge"],
                              "Odds": r["Odds"], "Actual": f"{as_}-{hs}", "Hit": hit,
                              "P/L (1u)": round((r["Odds"] - 1) if hit else -1.0, 2)})

    pdf, pmeta, pnote = build_prop_edges(sel_date, max_games, snapshot_by_event=snap_by_event,
                                         odds_override=odds, meta_override=ometa, use_injuries=False)
    prop_rows = []
    if isinstance(pdf, pd.DataFrame) and not pdf.empty:
        boxes = {eid: {b["athlete_id"]: b for b in fetch_box_final(eid)} for eid in pdf["EventID"].unique()}
        for _, r in pdf.iterrows():
            b = boxes.get(r["EventID"], {}).get(str(r["PlayerID"]))
            if not b:
                continue  # didn't play — most books void the bet
            stat = PROP_STAT[r["MarketKey"]]
            actual = (b["pts"] + b["reb"] + b["ast"]) if stat == "pra" else b[stat]
            if actual == float(r["Point"]):
                continue
            hit = actual > float(r["Point"])
            prop_rows.append({"Market": r["Market"], "Light": r["Light"], "Player": r["Player"],
                              "Game": r["Game"], "Line": r["Line"], "Model %": r["Model %"],
                              "Market %": r["Market %"], "Edge": r["Edge"], "Odds": r["Best over"],
                              "Actual": actual, "Hit": hit,
                              "P/L (1u)": round((r["Best over"] - 1) if hit else -1.0, 2)})
    cost = {"snapshots": snaps, "game_line_credits": ometa.get("credits_estimate", 0),
            "remaining": (pmeta or {}).get("remaining") or ometa.get("remaining")}
    all_rows = prop_rows + game_rows
    if not all_rows:
        return None, (pnote or gnote or "No qualifying value picks for that date."), cost
    return (pd.DataFrame(all_rows),
            f"Reconstructed and graded {len(prop_rows)} prop pick(s) and {len(game_rows)} game-line pick(s).",
            cost)


def build_clv_report(sel_date, bet_lead_minutes=60, close_lead_minutes=5, flagged_only=True):
    """Closing Line Value for the model's game-line picks. Same idea as the MLB
    app: price at bet time vs the closing price. COST: fetches the historical
    slate twice."""
    ctx = slate_context(sel_date, use_injuries=False)
    games = ctx["games"]
    if not games:
        return None, "No games on this date.", {}
    bet_odds, bmeta, bsnaps = historical_slate_odds(games, GAME_REGIONS, bet_lead_minutes)
    if not bet_odds:
        return None, bmeta.get("error") or "No odds for the bet snapshot.", {}
    close_odds, cmeta, csnaps = historical_slate_odds(games, GAME_REGIONS, close_lead_minutes)
    cost = {"total_credits": bmeta.get("credits_estimate", 0) + cmeta.get("credits_estimate", 0),
            "remaining": cmeta.get("remaining") or bmeta.get("remaining")}
    if not close_odds:
        return None, cmeta.get("error") or "No odds for the closing snapshot.", cost
    gdf, gnote, _ = build_game_edges(sel_date, odds_override=bet_odds, meta_override=bmeta,
                                     use_injuries=False)
    if gdf is None or gdf.empty:
        return None, gnote or "No model picks to score.", cost

    close_by_event = {}
    for og in close_odds:
        g = _match_espn(games, og.get("home_team"), og.get("away_team"))
        if g:
            close_by_event[g["event_id"]] = consolidate_odds(og)

    rows, no_close = [], 0
    for _, r in gdf.iterrows():
        light = classify_pick(r["Edge"], r["Model %"], r["Market"])
        if flagged_only and light not in ("🟢", "🟡"):
            continue
        c = close_by_event.get(r["EventID"])
        if not c:
            no_close += 1
            continue
        side, mkt = r["Side"], r["Market"]
        close_price = close_fair = close_line = None
        moved = False
        if mkt == "Moneyline":
            close_price = c["ml_home_best"] if side == "home" else c["ml_away_best"]
            fh, fa = devig_two(c["ml_home"], c["ml_away"])
            close_fair = fh if side == "home" else fa
        elif mkt == "Spread":
            close_line = c["home_point"] if side == "home" else (-c["home_point"] if c["home_point"] is not None else None)
            if close_line is None or abs(close_line - float(r["Threshold"])) > 0.01:
                moved = True
            else:
                close_price = c["sp_home_best"] if side == "home" else c["sp_away_best"]
                sh, sa = devig_two(c["sp_home"], c["sp_away"])
                close_fair = sh if side == "home" else sa
        else:
            close_line = c["total_line"]
            if close_line is None or abs(close_line - float(r["Threshold"])) > 0.01:
                moved = True
            else:
                over = r["Direction"] == "Over"
                close_price = c["over_best"] if over else c["under_best"]
                fo, fu = devig_two(c["over"], c["under"])
                close_fair = fo if over else fu
        clv = beat = None
        if close_price:
            clv = round((r["Odds"] / close_price - 1) * 100, 2)
            beat = bool(r["Odds"] > close_price)
        rows.append({"Light": light, "Market": mkt, "Selection": r["Selection"], "Game": r["Game"],
                     "Model %": r["Model %"], "Edge": r["Edge"], "Bet odds": r["Odds"],
                     "Close odds": round(close_price, 3) if close_price else None,
                     "CLV %": clv, "Beat close": beat, "Bet line": r["Threshold"],
                     "Close line": close_line, "Line moved": moved,
                     "Close fair %": round(close_fair * 100, 1) if close_fair else None,
                     "Model vs close": round(r["Model %"] - close_fair * 100, 1) if close_fair else None})
    if not rows:
        return None, "No flagged picks could be matched to a closing price.", cost
    df = pd.DataFrame(rows)
    note = f"Compared {len(df)} pick(s) against the closing line."
    if no_close:
        note += f" {no_close} had no closing snapshot."
    return df, note, cost


# --- Backtest -------------------------------------------------------------------------------

def run_game_backtest(sel_date, days_back=14):
    """Score the game model on recent completed games, rebuilding team ratings
    AS OF each day (only games before it). Free. Returns (records, residuals,
    days_done) — residuals = (actual - model) for margin and total, whose SDs
    are the fitted replacements for MARGIN_SD / TOTAL_SD."""
    season = season_for(sel_date)
    season_games = fetch_season_results(season, str(sel_date))
    prior = fetch_standings(season - 1)
    recs, resid, days_done = [], {"margin": [], "total": []}, 0
    for i in range(days_back, 0, -1):
        d = sel_date - timedelta(days=i)
        ds = str(d)
        today = [g for g in season_games if g["et_date"] == ds]
        if not today:
            continue
        before = [g for g in season_games if g["et_date"] < ds]
        ratings = compute_team_ratings(before, prior)
        b2b = b2b_teams(before, d)
        days_done += 1
        for g in today:
            t = ratings["teams"]
            rh = t.get(g["home_id"]) or {"off": 1, "def": 1}
            ra = t.get(g["away_id"]) or {"off": 1, "def": 1}
            m = game_model(rh, ra, ratings["lg"], home_b2b=g["home_id"] in b2b,
                           away_b2b=g["away_id"] in b2b)
            margin = g["home_score"] - g["away_score"]
            recs.append(("Moneyline (home win)", m["p_home_ml"], 1 if margin > 0 else 0))
            resid["margin"].append(margin - m["margin"])
            resid["total"].append(g["home_score"] + g["away_score"] - m["total"])
    return recs, resid, days_done


def calib_summary(recs, market):
    rows = [(p, o) for m, p, o in recs if m == market]
    if not rows:
        return None
    n = len(rows)
    buckets = []
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        grp = [(p, o) for p, o in rows if lo <= p < hi or (k == 9 and p >= 1)]
        if grp:
            buckets.append((f"{int(lo*100)}-{int(hi*100)}%", len(grp),
                            round(sum(p for p, _ in grp) / len(grp) * 100, 1),
                            round(sum(o for _, o in grp) / len(grp) * 100, 1)))
    return {"n": n, "brier": round(sum((p - o) ** 2 for p, o in rows) / n, 4),
            "acc": round(sum((p >= 0.5) == (o == 1) for p, o in rows) / n * 100, 1),
            "base_rate": round(sum(o for _, o in rows) / n * 100, 1), "buckets": buckets}
