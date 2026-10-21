import streamlit as st
import pandas as pd
from nba_core import *

setup_page("NBA Prop Analyser — Game Bets")
sel_date, use_inj = sidebar_date()

st.markdown("## 🎯 Game Bets — Money Line · Spread · Totals")
st.caption("Projects each team's points from offensive/defensive ratings (blended with last "
           "season early on), home court, back-to-backs and reported absences, models the "
           "margin and total as Normal distributions, then compares to de-vigged UK odds. "
           "Positive edge = the model rates the bet better than the market. Always confirm "
           "the live price before staking — NBA lines move hard on late injury news.")

if st.button("Analyse game bets (UK odds)"):
    with st.spinner("Fetching schedule, team ratings, injuries and UK odds..."):
        gdf, gnote, gmeta = build_game_edges(sel_date, use_injuries=use_inj)
    if gdf is None:
        st.warning(gnote)
        st.session_state.pop("game_edges", None)
    else:
        # Fetch + store only; display lives in the persistent block below so the
        # Sort-by dropdowns keep working (same fix as the MLB app).
        st.session_state["game_edges"] = gdf
        st.session_state["game_edges_note"] = gnote
        st.session_state["game_edges_meta"] = gmeta

gdf = st.session_state.get("game_edges")
if isinstance(gdf, pd.DataFrame) and not gdf.empty:
    gnote = st.session_state.get("game_edges_note")
    gmeta = st.session_state.get("game_edges_meta") or {}
    if gmeta.get("remaining"):
        st.caption(f"Odds quota — used {gmeta.get('used')}, remaining {gmeta.get('remaining')}")
    if gnote:
        st.info(gnote)
    lights = gdf.apply(lambda r: classify_pick(r["Edge"], r["Model %"], r["Market"]), axis=1)
    st.markdown(f"### 🟢 {int((lights == '🟢').sum())} green · 🟡 {int((lights == '🟡').sum())} "
                f"amber · 🔴 {int((lights == '🔴').sum())} red")
    st.caption("🟢 2–6 pts = believable value · 🟡 6–12 = treat with caution · 🔴 12+ = almost "
               "certainly missing news (a late scratch, a rest day), not a real edge · ⚪ under 2 "
               "= no signal. Bands are tighter than the MLB app's on purpose — NBA sides and "
               "totals are among the sharpest markets there are.")

    def show_market(tab, market):
        with tab:
            sub = gdf[gdf["Market"] == market].copy()
            if sub.empty:
                st.write("No odds available for this market today.")
                return
            sub = sort_picker(sub, [("Start time", "_ct", True),
                                    ("Edge (high to low)", "Edge", False),
                                    ("Model % (high to low)", "Model %", False),
                                    ("Odds (high to low)", "Odds", False)], key=f"sort_{market}")
            for _, r in sub.iterrows():
                cond = " · ".join(x for x in [r["Proj"], r["DataNotes"]] if x)
                render_pick_card(
                    classify_pick(r["Edge"], r["Model %"], market), f"{r['Selection']} @ {r['Odds']:.2f}",
                    f"{r['Game']} · {r['Start']} UK ({r['US Date']})",
                    [("Model %", f"{r['Model %']:.1f}%"), ("Fair %", f"{r['Fair %']:.1f}%"),
                     ("Edge", f"{r['Edge']:.1f} pts"), ("EV %", f"{r['EV %']:.1f}%")],
                    reason=r["Reason"], conditions=cond)

    def show_most_likely(tab):
        with tab:
            st.caption("Pure model confidence, ignoring the market — 'what does the model "
                       "predict', not 'where's the value'. A high % is still not a guarantee.")
            pick = st.multiselect("Markets to include:", ["Moneyline", "Spread", "Total"],
                                  default=["Moneyline", "Spread", "Total"], key="ml_market_pick")
            sub = gdf[gdf["Market"].isin(pick)].copy()
            if sub.empty:
                st.write("No selections match the chosen markets.")
                return
            sub = sort_picker(sub, [("Model % (high to low)", "Model %", False),
                                    ("Start time", "_ct", True)], key="sort_most_likely_gb")
            for _, r in sub.iterrows():
                render_pick_card(None, f"{r['Selection']} · {r['Market']}",
                                 f"{r['Game']} · {r['Start']} UK ({r['US Date']})",
                                 [("Model %", f"{r['Model %']:.1f}%")], conditions=r["Proj"])

    t1, t2, t3, t4 = st.tabs(["💰 Money Line", "📏 Spread", "📊 Totals", "🎯 Most Likely"])
    show_market(t1, "Moneyline")
    show_market(t2, "Spread")
    show_market(t3, "Total")
    show_most_likely(t4)
    st.caption("Model %: our probability (pushes removed on whole-number lines) · Fair %: "
               "book's de-vigged probability · Edge: model minus fair · EV %: expected return "
               "per unit at best odds.")

    greens = gdf[gdf.apply(lambda r: MARKET_EDGE_BANDS[r["Market"]][0] <= r["Edge"]
                           < MARKET_EDGE_BANDS[r["Market"]][1], axis=1)].copy()
    st.markdown("### 🎟️ Accumulator builder")
    st.caption("Builds a multi from 🟢 green selections only — ambers and reds are excluded.")
    if greens.empty:
        st.write("No green selections on the analysed slate to build an accumulator from.")
    else:
        greens = greens.sort_values("Edge", ascending=False).reset_index(drop=True)
        greens["pick"] = greens.apply(lambda r: f"{r['Selection']}  @ {r['Odds']:.2f}  ·  "
                                      f"{r['Market']}  ·  {r['Game']}  (edge {r['Edge']:.1f})", axis=1)
        chosen = st.multiselect("Choose your legs (each is one green selection):", greens["pick"].tolist())
        stake = st.number_input("Stake (£)", min_value=0.0, value=10.0, step=1.0)
        if chosen:
            sel = greens[greens["pick"].isin(chosen)]
            odds = float(sel["Odds"].prod())
            model = float((sel["Model %"] / 100).prod())
            ret = stake * odds
            a1, a2, a3 = st.columns(3)
            a1.metric("Legs", len(sel))
            a2.metric("Combined odds", f"{odds:.2f}")
            a3.metric(f"Return on £{stake:.0f}", f"£{ret:.2f}", f"+£{ret - stake:.2f}")
            b1, b2, b3 = st.columns(3)
            b1.metric("Model: chance it lands", f"{model * 100:.1f}%")
            b2.metric("Market-implied chance", f"{100 / odds:.1f}%")
            b3.metric("Combined EV", f"{(model * odds - 1) * 100:+.1f}%")
            dupes = [g for g, c in sel["Game"].value_counts().items() if c > 1]
            if dupes:
                st.warning("⚠️ Multiple legs from the same game (" + ", ".join(dupes) + "). "
                           "Those outcomes are correlated (a favourite covering and the Over "
                           "often go together), so the combined chance above is optimistic and "
                           "most books need a same-game multi.")
            st.caption("Assumes legs are independent (true across different games). A multi "
                       "needs every leg to win — high variance even with an edge on each leg.")
