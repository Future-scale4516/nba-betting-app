"""NBA models — pure maths, no fetching (all I/O lives in nba_data.py).

GAME MODEL
  Each team gets an offensive index (points scored / league average) and a
  defensive index (points allowed / league average). Expected points:

      home = league_ppg * home_off * away_def + HOME_COURT/2 - b2b - injuries
      away = league_ppg * away_off * home_def - HOME_COURT/2 - b2b - injuries

  Same multiplicative shape as the MLB app's expected_runs(). The margin and
  the total are then treated as NORMAL, not Poisson/NB: NBA scores are large
  sums of many small scoring events, so the Normal is the standard and
  well-behaved choice here (Poisson made sense for runs, not for 115 points).

  Early season the current-season sample is tiny, so every team rating is
  blended with LAST season's (regressed toward average for roster turnover),
  the same shrink-to-a-prior idea the MLB app uses for platoon splits.

PROP MODEL
  Per-minute production (current season weighted fully, last season
  down-weighted) x projected minutes x a matchup factor taken from the game
  model (this team's expected points vs its normal output — which already
  folds in opponent defence, pace, home court and rest). Counts are priced
  with a Negative Binomial whose dispersion is the player's OWN observed
  variance shrunk toward a market prior — same reasoning as the MLB app's
  PROP_DISPERSION, just per-player instead of one number per market.

EVERY constant marked "provisional" is a reasoned starting value, not a fit.
The Backtest page reports the observed margin/total residual SD so
MARGIN_SD / TOTAL_SD can be replaced with fitted numbers once games accrue.
"""
import math

# --- League / game constants (provisional) ---------------------------------------
LEAGUE_PPG_DEFAULT = 114.5     # per-team points per game fallback
HOME_COURT = 2.4               # home advantage in points of margin (has shrunk
                               # over the last decade; ~2-2.5 recently)
B2B_PENALTY = 1.8              # points off a team on the 2nd night of a back-to-back
MARGIN_SD = 14.5               # SD of final margin around the model's mean — fitted from
                               # the Backtest page (2025-26 season, ~300 games, was 12.0)
TOTAL_SD = 18.5                # SD of final total around the model's mean — fitted figure
                               # matched the provisional value exactly, so left unchanged
PRIOR_GAMES_WEIGHT = 12        # last season counts as this many current-season games
PRIOR_REGRESSION = 0.70        # keep 70% of last season's distance from average
                               # (offseason roster churn pulls everyone to the middle)
INJURY_LOSS_SHARE = 0.45       # share of an absent player's points the team LOSES
                               # (the rest gets absorbed by teammates' usage)
INJURY_CAP = 0.12              # never cut a team's offence by more than 12%
BLOWOUT_MARGIN = 12.0          # projected margin beyond which starters' minutes trimmed
BLOWOUT_MIN_FACTOR = 0.96


def normal_cdf(x, mu=0.0, sd=1.0):
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def discrete_over_under(mu, sd, line):
    """P(X > line), P(X < line), P(X == line) for an INTEGER outcome (a margin
    or total) modelled as Normal(mu, sd), with continuity correction. A .5 line
    has zero push; a whole-number line returns a real push probability."""
    if line is None:
        return None, None, None
    line = float(line)
    p_over = 1.0 - normal_cdf(math.floor(line) + 0.5, mu, sd)
    p_under = normal_cdf(math.ceil(line) - 0.5, mu, sd)
    p_push = max(0.0, 1.0 - p_over - p_under)
    return p_over, p_under, p_push


def no_push(p_win, p_lose):
    """Bettor's probability once pushes (stake returned) are taken out — the
    comparable number to a de-vigged two-way price."""
    s = (p_win or 0) + (p_lose or 0)
    return (p_win / s) if s > 0 else None


# --- Team ratings -------------------------------------------------------------------

def compute_team_ratings(games, prior):
    """Team offensive/defensive indices from completed games (already filtered
    to before the slate date by the caller) blended with last season.

    games: list of ESPN game dicts with scores. prior: {team_id: {pf, pa}}
    from fetch_standings(season-1). Returns
      {"lg": league ppg, "teams": {tid: {off, def, gp, ppg}}, "lg_cur": .., "gp_total": ..}
    """
    agg = {}
    for g in games:
        hs, as_ = g.get("home_score"), g.get("away_score")
        if hs is None or as_ is None:
            continue
        for tid, f, a in ((g["home_id"], hs, as_), (g["away_id"], as_, hs)):
            t = agg.setdefault(tid, [0, 0.0, 0.0])
            t[0] += 1
            t[1] += f
            t[2] += a
    team_games = sum(v[0] for v in agg.values())
    lg_cur = (sum(v[1] for v in agg.values()) / team_games) if team_games else None
    lg_prior = (sum(p["pf"] for p in prior.values()) / len(prior)) if prior else None

    if lg_cur and lg_prior:
        w = team_games / (team_games + 300.0)   # ~10 games per team before current dominates
        lg = w * lg_cur + (1 - w) * lg_prior
    else:
        lg = lg_cur or lg_prior or LEAGUE_PPG_DEFAULT

    teams = {}
    for tid in set(agg) | set(prior):
        gp, pf, pa = agg.get(tid, [0, 0.0, 0.0])
        if prior.get(tid) and lg_prior:
            p_off = 1 + PRIOR_REGRESSION * (prior[tid]["pf"] / lg_prior - 1)
            p_def = 1 + PRIOR_REGRESSION * (prior[tid]["pa"] / lg_prior - 1)
        else:
            p_off = p_def = 1.0
        if gp and lg_cur:
            c_off, c_def = (pf / gp) / lg_cur, (pa / gp) / lg_cur
        else:
            c_off, c_def = 1.0, 1.0
        k = PRIOR_GAMES_WEIGHT
        off = (gp * c_off + k * p_off) / (gp + k)
        dfn = (gp * c_def + k * p_def) / (gp + k)
        teams[tid] = {"off": off, "def": dfn, "gp": gp, "ppg": lg * off}
    return {"lg": lg, "lg_cur": lg_cur, "teams": teams, "gp_total": team_games}


def b2b_teams(games, slate_date):
    """Team ids that played the day before slate_date (2nd night of a B2B)."""
    from datetime import timedelta
    prev = str(slate_date - timedelta(days=1))
    out = set()
    for g in games:
        if g.get("et_date") == prev:
            out.add(g["home_id"])
            out.add(g["away_id"])
    return out


def game_model(home_r, away_r, lg, total_line=None, home_point=None,
               home_b2b=False, away_b2b=False, home_inj=0.0, away_inj=0.0):
    """home_r/away_r: rating dicts {off, def}. home_inj/away_inj: fraction of
    offence lost to absences (0-INJURY_CAP). home_point: the HOME team's
    signed spread (e.g. -5.5 = home favoured by 5.5)."""
    h_off = home_r["off"] * (1 - min(home_inj, INJURY_CAP))
    a_off = away_r["off"] * (1 - min(away_inj, INJURY_CAP))
    exp_home = lg * h_off * away_r["def"] + HOME_COURT / 2 - (B2B_PENALTY if home_b2b else 0)
    exp_away = lg * a_off * home_r["def"] - HOME_COURT / 2 - (B2B_PENALTY if away_b2b else 0)
    mu_m, mu_t = exp_home - exp_away, exp_home + exp_away

    p_hw, p_aw, _ = discrete_over_under(mu_m, MARGIN_SD, 0)   # margin can't be 0 (OT)
    out = {"exp_home": exp_home, "exp_away": exp_away, "margin": mu_m, "total": mu_t,
           "p_home_ml": no_push(p_hw, p_aw), "p_away_ml": no_push(p_aw, p_hw)}
    if home_point is not None:
        # Home covers when margin + home_point > 0, i.e. margin > -home_point.
        cov, lose, _ = discrete_over_under(mu_m, MARGIN_SD, -float(home_point))
        out["p_home_cover"], out["p_away_cover"] = no_push(cov, lose), no_push(lose, cov)
    if total_line is not None:
        po, pu, _ = discrete_over_under(mu_t, TOTAL_SD, total_line)
        out["p_over"], out["p_under"] = no_push(po, pu), no_push(pu, po)
    return out


# --- Odds consolidation ---------------------------------------------------------------

def _median(xs):
    xs = sorted(x for x in xs if x)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _nick(s):
    p = (s or "").lower().split()
    return p[-1] if p else ""


def consolidate_odds(game):
    """Consensus (median) and best decimal price per outcome across books.
    Spreads and totals are bucketed by line and the most widely quoted line
    wins — never averaging prices from different lines together (the MLB
    Run Line bug). Spread buckets are keyed by the HOME team's signed point,
    so which side is favoured is read from the data, never assumed."""
    hn, an = _nick(game.get("home_team")), _nick(game.get("away_team"))
    ml_h, ml_a, spreads, totals = [], [], {}, {}
    for bk in game.get("bookmakers", []):
        for m in bk.get("markets", []):
            key, outs = m.get("key"), m.get("outcomes", [])
            if key == "h2h":
                for o in outs:
                    if _nick(o.get("name")) == hn:
                        ml_h.append(o.get("price"))
                    elif _nick(o.get("name")) == an:
                        ml_a.append(o.get("price"))
            elif key == "spreads":
                ho = next((o for o in outs if _nick(o.get("name")) == hn), None)
                ao = next((o for o in outs if _nick(o.get("name")) == an), None)
                if ho and ao and ho.get("point") is not None:
                    s = spreads.setdefault(float(ho["point"]), {"h": [], "a": []})
                    s["h"].append(ho.get("price"))
                    s["a"].append(ao.get("price"))
            elif key == "totals":
                for o in outs:
                    if o.get("point") is None:
                        continue
                    s = totals.setdefault(float(o["point"]), {"over": [], "under": []})
                    side = (o.get("name") or "").lower()
                    if side in s:
                        s[side].append(o.get("price"))

    def best_line(d, a, b):
        best, cnt = None, 0
        for pt, s in d.items():
            c = min(len(s[a]), len(s[b]))
            if c > cnt:
                best, cnt = pt, c
        return best

    res = {"ml_home": _median(ml_h), "ml_away": _median(ml_a),
           "ml_home_best": max(ml_h) if ml_h else None,
           "ml_away_best": max(ml_a) if ml_a else None,
           "home_point": None, "sp_home": None, "sp_away": None,
           "sp_home_best": None, "sp_away_best": None,
           "total_line": None, "over": None, "under": None,
           "over_best": None, "under_best": None}
    hp = best_line(spreads, "h", "a")
    if hp is not None:
        s = spreads[hp]
        res.update(home_point=hp, sp_home=_median(s["h"]), sp_away=_median(s["a"]),
                   sp_home_best=max(s["h"]), sp_away_best=max(s["a"]))
    tl = best_line(totals, "over", "under")
    if tl is not None:
        s = totals[tl]
        res.update(total_line=tl, over=_median(s["over"]), under=_median(s["under"]),
                   over_best=max(s["over"]), under_best=max(s["under"]))
    return res


def consolidate_prop(event, market_key):
    """{player_name: {point, over, under, over_best, under_best}} for one prop
    market — picks the line most books quote both sides of (books often
    disagree by a point on NBA props, e.g. 24.5 vs 25.5)."""
    by_player = {}
    for bk in (event or {}).get("bookmakers", []):
        for m in bk.get("markets", []):
            if m.get("key") != market_key:
                continue
            for o in m.get("outcomes", []):
                player, side = o.get("description"), (o.get("name") or "").lower()
                pt, price = o.get("point"), o.get("price")
                if not player or price is None or pt is None or side not in ("over", "under"):
                    continue
                by_player.setdefault(player, {}).setdefault(float(pt), {"over": [], "under": []})[side].append(price)
    out = {}
    for player, lines in by_player.items():
        best, cnt = None, -1
        for pt, s in lines.items():
            c = min(len(s["over"]), len(s["under"])) * 100 + len(s["over"])
            if c > cnt and s["over"]:
                best, cnt = pt, c
        if best is None:
            continue
        s = lines[best]
        out[player] = {"point": best, "over": _median(s["over"]),
                       "under": _median(s["under"]) if s["under"] else None,
                       "over_best": max(s["over"]),
                       "under_best": max(s["under"]) if s["under"] else None}
    return out


def devig_two(a, b):
    if not a or not b:
        return None, None
    ia, ib = 1 / a, 1 / b
    return ia / (ia + ib), ib / (ia + ib)


TYPICAL_PROP_OVERROUND = 0.06


def market_prob(over_odds, under_odds):
    """Fair P(over): de-vig both sides when quoted; otherwise strip a typical
    overround from the single side (same reasoning as the MLB app)."""
    if over_odds and under_odds:
        return devig_two(over_odds, under_odds)[0]
    if over_odds:
        return max(0.001, min((1 / over_odds) / (1 + TYPICAL_PROP_OVERROUND), 0.999))
    return None


def edge_ev(model_p, fair_p, best_odds):
    edge = (model_p - fair_p) * 100 if (model_p is not None and fair_p is not None) else None
    ev = (model_p * best_odds - 1) * 100 if (model_p is not None and best_odds) else None
    return edge, ev


# --- Traffic lights -------------------------------------------------------------------
# (no-signal ceiling, green ceiling, amber ceiling). NBA sides and totals are
# among the sharpest markets anywhere, so game-line bands are TIGHTER than the
# MLB app's: an 8-point edge on an NBA spread almost always means the model is
# missing news (a late scratch, a rest day), not that the market is wrong.
MARKET_EDGE_BANDS = {
    "Moneyline": (2, 6, 12), "Spread": (2, 6, 12), "Total": (2, 6, 12),
    "Points": (3, 9, 15), "Rebounds": (3, 9, 15), "Assists": (3, 9, 15),
    "Threes": (3, 9, 15), "PRA": (3, 9, 15),
}
MARKET_PROB_CEILING = {
    "Moneyline": 95, "Spread": 80, "Total": 78,
    "Points": 85, "Rebounds": 85, "Assists": 85, "Threes": 85, "PRA": 85,
}


def _edge_light(edge, market=None):
    if edge is None:
        return ""
    lo, mid, hi = MARKET_EDGE_BANDS.get(market, (2, 8, 15))
    if edge >= hi:
        return "🔴"
    if edge >= mid:
        return "🟡"
    if edge >= lo:
        return "🟢"
    return "⚪"


def classify_pick(edge, model_pct, market=None):
    """Plausibility first (an implausibly high raw probability is red on its
    own), then the market's edge bands — identical logic to the MLB app."""
    if model_pct is not None and model_pct > MARKET_PROB_CEILING.get(market, 95):
        return "🔴"
    return _edge_light(edge, market)


def in_value_band(edge, market):
    lo, _, hi = MARKET_EDGE_BANDS.get(market, (2, 8, 15))
    return edge is not None and lo <= edge < hi


# --- Calibration ---------------------------------------------------------------------------
# adjusted = raw*(1-shrink) + base*shrink. PROVISIONAL — nothing has been fitted
# yet. Over/under props and spreads are set near 50/50 by the books, so 0.5 is
# the natural base; shrinking toward it damps the false edges a young model
# produces. Replace with fitted numbers once Results/Backtest have enough history
# (same workflow as CALIBRATION_FITS in the MLB app).
CALIBRATION_FITS = {
    "Moneyline": {"base": 0.5, "shrink": 0.05},
    "Spread":    {"base": 0.5, "shrink": 0.10},
    "Total":     {"base": 0.5, "shrink": 0.10},
    "Points":    {"base": 0.5, "shrink": 0.25},
    "Rebounds":  {"base": 0.5, "shrink": 0.25},
    "Assists":   {"base": 0.5, "shrink": 0.25},
    "Threes":    {"base": 0.5, "shrink": 0.25},
    "PRA":       {"base": 0.5, "shrink": 0.25},
}


def calibrate(p, market):
    fit = CALIBRATION_FITS.get(market)
    if p is None or not fit or not fit.get("shrink"):
        return p
    return p * (1 - fit["shrink"]) + fit["base"] * fit["shrink"]


# --- Player props ------------------------------------------------------------------------------
PROP_MARKETS = ["player_points", "player_rebounds", "player_assists",
                "player_threes", "player_points_rebounds_assists"]
PROP_LABEL = {"player_points": "Points", "player_rebounds": "Rebounds",
              "player_assists": "Assists", "player_threes": "Threes",
              "player_points_rebounds_assists": "PRA"}
PROP_STAT = {"player_points": "pts", "player_rebounds": "reb", "player_assists": "ast",
             "player_threes": "fg3", "player_points_rebounds_assists": "pra"}
LABEL_TO_KEY = {v: k for k, v in PROP_LABEL.items()}

# Variance-to-mean ratio priors (provisional). Points come in 2s and 3s and
# swing with minutes, so they're well overdispersed; threes are close to
# Poisson. Each player's OWN observed ratio is shrunk toward these.
DISPERSION_PRIOR = {"pts": 2.3, "reb": 1.5, "ast": 1.6, "fg3": 1.25, "pra": 2.8}
DISPERSION_STABILISE_GAMES = 15
PRIOR_SEASON_GAME_WEIGHT = 0.35   # a last-season game counts 35% of a current one
MIN_GAMES_FOR_PROJECTION = 5
MIN_PROJECTED_MINUTES = 12.0

# Milestones for the free "Most Likely" view and the model-reads Results mode
MILESTONES = {"Points": [10, 15, 20, 25, 30], "Rebounds": [4, 6, 8, 10, 12],
              "Assists": [3, 5, 7, 9], "Threes": [1, 2, 3, 4], "PRA": [20, 25, 30, 35, 40]}
RESULTS_MILESTONES = {"Points": [15, 25], "Rebounds": [6, 10], "Assists": [4, 8],
                      "Threes": [2, 3], "PRA": [25, 35]}


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def project_player(cur_games, prior_games, before_date_str):
    """Baseline per-game projection for one player, BEFORE matchup.

    Only games strictly before `before_date_str` are used, so rebuilding a past
    slate never sees the future. Returns None for thin samples or fringe
    minutes — the NBA equivalent of the MLB app's MIN_PA_FOR_RANKING floor."""
    cur = [g for g in cur_games if g.get("et_date") and g["et_date"] < before_date_str and g["min"] > 0]
    prior = [g for g in prior_games if g["min"] > 0][-40:]
    if len(cur) + len(prior) < MIN_GAMES_FOR_PROJECTION:
        return None

    pool = [(g, 1.0 + (0.5 if i >= len(cur) - 5 else 0.0)) for i, g in enumerate(cur)]
    pool += [(g, PRIOR_SEASON_GAME_WEIGHT) for g in prior]
    wmin = sum(g["min"] * w for g, w in pool)
    if wmin <= 0:
        return None
    rates = {k: sum(g[k] * w for g, w in pool) / wmin for k in ("pts", "reb", "ast", "fg3")}

    n = len(cur)
    m_cur = None
    if n:
        m_all, m_recent = _mean([g["min"] for g in cur]), _mean([g["min"] for g in cur[-5:]])
        m_cur = 0.6 * m_recent + 0.4 * m_all if n >= 5 else m_all
    m_prior = _mean([g["min"] for g in prior[-20:]]) if prior else None
    if m_cur is not None and m_prior is not None:
        proj_min = (n * m_cur + 4 * m_prior) / (n + 4)
    else:
        proj_min = m_cur if m_cur is not None else m_prior
    proj_min = min(proj_min or 0.0, 42.0)
    if proj_min < MIN_PROJECTED_MINUTES:
        return None

    means = {k: rates[k] * proj_min for k in rates}
    means["pra"] = means["pts"] + means["reb"] + means["ast"]

    recent = (cur[-30:] + prior[::-1])[:30]
    disp = {}
    for k in ("pts", "reb", "ast", "fg3", "pra"):
        vals = [(g["pts"] + g["reb"] + g["ast"]) if k == "pra" else g[k] for g in recent]
        mu = _mean(vals)
        d_obs = None
        if len(vals) >= 3 and mu > 0:
            var = sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)
            d_obs = var / mu
        k0, n_obs = DISPERSION_STABILISE_GAMES, len(vals) if d_obs is not None else 0
        disp[k] = ((n_obs * (d_obs or 0)) + k0 * DISPERSION_PRIOR[k]) / (n_obs + k0)

    return {"min": proj_min, "means": means, "disp": disp, "n_cur": n,
            "n_prior": len(prior), "last_games": cur[-10:] if cur else prior[-10:]}


def matchup_means(proj, team_exp_pts, team_ppg, game_total, lg, blowout=False):
    """Apply the game context to a baseline projection.
      - points & threes scale with how many points the TEAM is expected to
        score tonight vs its norm (opponent defence, pace, home court, rest)
      - assists a little less (damped); rebounds only with overall game pace
    Damping exponents are provisional."""
    r = (team_exp_pts / team_ppg) if team_ppg else 1.0
    r = max(0.85, min(r, 1.15))
    pace = max(0.9, min((game_total / (2 * lg)) if lg else 1.0, 1.1))
    m = proj["means"]
    mf = BLOWOUT_MIN_FACTOR if blowout else 1.0
    out = {"pts": m["pts"] * r * mf, "fg3": m["fg3"] * (r ** 0.8) * mf,
           "ast": m["ast"] * (r ** 0.7) * mf, "reb": m["reb"] * (pace ** 0.5) * mf}
    out["pra"] = out["pts"] + out["reb"] + out["ast"]
    return out


def _nb_pmf(k, mu, alpha):
    n = 1.0 / alpha
    p = n / (n + mu)
    return math.exp(math.lgamma(k + n) - math.lgamma(n) - math.lgamma(k + 1)
                    + n * math.log(p) + k * math.log(1 - p))


def _pois_pmf(k, mu):
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1)) if mu > 0 else (1.0 if k == 0 else 0.0)


def count_cdf(k, mu, disp):
    """P(X <= k) for a count with mean mu and variance/mean ratio `disp`."""
    if k < 0:
        return 0.0
    if mu <= 0:
        return 1.0
    if disp <= 1.02:
        return min(1.0, sum(_pois_pmf(i, mu) for i in range(int(k) + 1)))
    alpha = (disp - 1.0) / mu
    return min(1.0, sum(_nb_pmf(i, mu, alpha) for i in range(int(k) + 1)))


def prop_p_over(mu, disp, line):
    """P(over the line), with pushes removed for whole-number lines."""
    if mu is None or line is None:
        return None
    line = float(line)
    p_over = 1.0 - count_cdf(math.floor(line), mu, disp)
    if line == int(line):
        p_under = count_cdf(int(line) - 1, mu, disp)
        return no_push(p_over, p_under)
    return max(0.0, min(1.0, p_over))


def p_at_least(mu, disp, k):
    """P(X >= k) — the milestone framing ('20+ points')."""
    return max(0.0, min(1.0, 1.0 - count_cdf(k - 1, mu, disp)))
