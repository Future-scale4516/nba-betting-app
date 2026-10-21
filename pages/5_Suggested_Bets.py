import streamlit as st
from nba_core import *

setup_page("NBA Prop Analyser — Suggested Bets")
sel_date, use_inj = sidebar_date()

st.title("📋 Suggested Bets")
st.caption("Auto-built doubles and trebles across Moneyline, Spread, Totals, Points, Rebounds, "
           "Assists, Threes and PRA — green/amber picks only, ranked by the model's raw "
           "probability, never two legs from the same game. Stakes are split by a per-market "
           "trust tier (provisional until there are real NBA results to tune it from).")

c1, c2 = st.columns(2)
with c1:
    bankroll = st.number_input("Bankroll to allocate (£)", min_value=1.0, value=40.0, step=1.0)
with c2:
    prop_games = st.slider("Prop games to analyse (5 credits each)", 1, 15, 6)
st.caption(f"Projected cost: up to {prop_games * 5} credits for props + 3 for game lines. "
           "Cached 15 min.")

MARKET_ORDER = ["Moneyline", "Spread", "Totals", "Points", "Rebounds", "Assists", "PRA", "Threes"]
STATUS_BADGE = {"pending": "⏳ Pending", "winning": "🟢 Winning", "losing": "🔴 Losing",
                "won": "✅ Won", "lost": "❌ Lost", "unknown": "❓ Unknown"}
COMBO_BADGE = {"alive": "🟢 Still alive", "won": "✅ Combo won!", "lost": "❌ Combo lost",
               "unknown": "❓ Uncertain — check manually"}

if st.button("Build suggested bets"):
    with st.spinner("Scanning every market and building combos..."):
        results, quota = build_suggested_bets(sel_date, prop_games, use_injuries=use_inj)
    st.session_state.update(suggested_results=results, suggested_bankroll=bankroll,
                            suggested_quota=quota)
    st.session_state.pop("suggested_live", None)

if "suggested_results" in st.session_state:
    results = st.session_state["suggested_results"]
    bank = st.session_state.get("suggested_bankroll", bankroll)
    pmeta = (st.session_state.get("suggested_quota") or {}).get("props") or {}
    if pmeta.get("remaining"):
        st.caption(f"Odds API quota — used {pmeta.get('used')}, remaining {pmeta.get('remaining')}")
    available = [m for m, r in results.items() if r["double"] or r["treble"]]
    if not available:
        st.warning("No markets have enough qualifying picks right now. Try closer to tip-off, "
                   "once injury news and lines have settled.")
    else:
        stakes = suggest_stakes(bank, available)
        l1, l2 = st.columns([2, 3])
        with l1:
            check_live = st.button("🔴 Check live status")
        with l2:
            st.caption("Pending, live winning/losing, or already won/lost. One lost leg kills the "
                       "combo. Over props and Overs settle as won the moment they're cleared.")
        show_form = st.checkbox(f"Show last-{FORM_GAMES} games form on prop legs",
                                key="suggested_form_toggle")
        if check_live:
            with st.spinner("Checking live scores and box scores..."):
                st.session_state["suggested_live"] = {
                    m: {ct: evaluate_combo_status(r[ct]) for ct in ("double", "treble") if r.get(ct)}
                    for m, r in results.items()}
        live = st.session_state.get("suggested_live")

        for market in MARKET_ORDER:
            r = results.get(market)
            st.divider()
            if not r or (not r["double"] and not r["treble"]):
                st.markdown(f"### {market}")
                st.write(f"No qualifying picks today. {r['note'] if r else ''}")
                continue
            stake = stakes.get(market, 0.0)
            st.markdown(f"### {market} — suggested stake £{stake:.2f}")
            if market == "Threes":
                st.caption("Most volatile prop — a line swings on a single shot. Smallest stake slice.")
            cols = st.columns(2)
            for col, ct, label in ((cols[0], "double", "Double"), (cols[1], "treble", "Treble")):
                combo = r[ct]
                with col:
                    st.markdown(f"**{label}**")
                    if combo is None:
                        st.caption("Not enough distinct-game picks for this size.")
                        continue
                    leg_live = None
                    if live and ct in live.get(market, {}):
                        overall, leg_live = live[market][ct]
                        st.markdown(f"**{COMBO_BADGE.get(overall, overall)}**")
                    for i, leg in enumerate(combo["legs"]):
                        st.write(f"• **{leg['label']}** ({leg['game']}) @ {leg['odds']:.2f} — "
                                 f"{leg['model_pct']:.1f}% model")
                        if leg.get("reason"):
                            st.caption(leg["reason"])
                        if show_form and leg["kind"] == "prop":
                            try:
                                h, p, syms = form_streak(leg["player_id"], season_for(sel_date),
                                                         leg["market_key"], leg["point"])
                                if h is not None:
                                    st.caption(f"Last {p}: {syms}  ({h}/{p})")
                            except Exception:
                                st.caption("Form data unavailable for this player right now.")
                        if leg_live:
                            s, detail = leg_live[i]
                            st.caption(f"{STATUS_BADGE.get(s, s)} — {detail}")
                    m1, m2 = st.columns(2)
                    m1.metric("Combined odds", f"{combo['combined_odds']:.2f}")
                    m2.metric("Chance to land", f"{combo['combined_prob'] * 100:.1f}%")
                    st.metric(f"Return on £{stake:.2f}", f"£{stake * combo['combined_odds']:.2f}")
        st.divider()
        st.caption("Starting points, not instructions — confirm live prices and late injury news "
                   "before staking. Live status is a same-day read only; nothing is saved.")
