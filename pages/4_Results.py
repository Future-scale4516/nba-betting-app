import streamlit as st
import pandas as pd
from nba_core import *

setup_page("NBA Prop Analyser — Results")
sel_date, use_inj = sidebar_date()

st.title("📋 Results")
st.caption("How the model's predictions turned out for the selected date — what it projected, "
           "what happened, and whether it landed. Model-only modes are free.")

bet_type = st.radio("Bet type", ["Player props", "Game bets"], horizontal=True)
is_game = bet_type == "Game bets"
mode = st.radio("Results mode", ["Model reads only (free)",
                                 "Priced-up picks (uses historical odds credits)"])
priced = mode.startswith("Priced")

if priced:
    max_g = st.slider("Games to reconstruct props for", 1, 12, 4)
    st.warning(f"**Credit cost:** historical game lines bill 10 credits per region per market — "
               f"about **30 per distinct tip-off time** (NBA slates usually have 2–4). Historical "
               f"props bill at the same historical rate, which can be ~50 credits per game for the "
               f"five prop markets — so {max_g} games could be up to ~{max_g * 50 + 120} credits. "
               "Check the quota line after the first run. Cached for a day.")
    st.caption("Odds come from a snapshot 1 hour before each game's own tip-off.")
else:
    with st.expander("What this mode can and can't show", expanded=False):
        st.markdown("""
**Props:** every rotation player's probability of reaching fixed milestones (15+/25+ points,
6+/10+ rebounds, 4+/8+ assists, 2+/3+ threes, 25+/35+ PRA), built only from games before
the date, checked against his real box score.

**Game bets:** Moneyline only — spreads and totals need the line that was on offer, so
they're graded in **Priced-up** mode.
""")

if st.button("Load results"):
    with st.spinner("Rebuilding picks and checking them against results..."):
        if priced:
            rdf, rnote, rcost = build_priced_results(sel_date, max_games=max_g)
            if isinstance(rdf, pd.DataFrame) and not rdf.empty:
                mask = rdf["Market"].isin(["Moneyline", "Spread", "Total"])
                rdf = rdf[mask if is_game else ~mask].reset_index(drop=True)
            st.session_state["results_cost"] = rcost
        elif is_game:
            rdf, rnote = build_game_results(sel_date)
        else:
            rdf, rnote = build_prop_results(sel_date)
    st.session_state.update(results_df=rdf, results_note=rnote, results_for_date=str(sel_date),
                            results_priced=priced)

rdf = st.session_state.get("results_df")
if st.session_state.get("results_for_date") == str(sel_date) and isinstance(rdf, pd.DataFrame) and not rdf.empty:
    st.caption(st.session_state.get("results_note", ""))
    floor = st.slider("Only show picks the model rated at least this likely (%)", 0, 90, 50, 5)
    sub_all = rdf[rdf["Model %"] >= floor]
    if sub_all.empty:
        st.warning("No picks at or above that confidence level.")
    else:
        hits, total = int(sub_all["Hit"].sum()), len(sub_all)
        rate, avg_model = hits / total * 100, sub_all["Model %"].mean()
        if "P/L (1u)" in sub_all.columns:
            pl = sub_all["P/L (1u)"].sum()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Landed", f"{hits}/{total}")
            c2.metric("Hit rate", f"{rate:.1f}%")
            c3.metric("P/L (1u stakes)", f"{pl:+.2f}u")
            c4.metric("ROI", f"{pl / total * 100:+.1f}%")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Landed", f"{hits}/{total}")
            c2.metric("Actual hit rate", f"{rate:.1f}%")
            c3.metric("Model predicted", f"{avg_model:.1f}%")
        if abs(rate - avg_model) <= 5:
            st.success(f"Model said {avg_model:.1f}%, reality was {rate:.1f}% — closely calibrated on this slate.")
        elif rate < avg_model:
            st.warning(f"Model said {avg_model:.1f}% but {rate:.1f}% landed — overconfident on this slate.")
        else:
            st.info(f"Model said {avg_model:.1f}% and {rate:.1f}% landed — underconfident on this slate.")
        st.caption("One slate is a small sample — judge calibration over many dates.")

        markets = ["Moneyline", "Spread", "Total"] if is_game else ["Points", "Rebounds", "Assists", "Threes", "PRA"]
        tabs = st.tabs(["📊 All"] + markets)

        def show_table(tab, market=None):
            with tab:
                sub = sub_all if market is None else sub_all[sub_all["Market"] == market]
                if sub.empty:
                    st.write("No picks in this market at that confidence level.")
                    return
                h, t = int(sub["Hit"].sum()), len(sub)
                hdr = f"**{h}/{t} landed ({h / t * 100:.1f}%)** · model predicted {sub['Model %'].mean():.1f}%"
                if "P/L (1u)" in sub.columns:
                    hdr += f" · P/L {sub['P/L (1u)'].sum():+.2f}u"
                st.markdown(hdr)
                disp = sub.copy()
                disp["Result"] = disp["Hit"].map({True: "✅", False: "❌"})
                cols = (["Result", "Light", "Player", "Market", "Line", "Model %", "Market %", "Edge",
                         "Odds", "Actual", "P/L (1u)", "Game"] if "P/L (1u)" in disp.columns else
                        ["Result", "Player", "Market", "Line", "Model %", "Actual", "Game", "Score"])
                cols = [c for c in cols if c in disp.columns and not (market and c == "Market")]
                st.dataframe(disp[cols].sort_values("Model %", ascending=False).reset_index(drop=True),
                             width="stretch", hide_index=True)

        show_table(tabs[0])
        for tab, mk in zip(tabs[1:], markets):
            show_table(tab, mk)
        st.download_button("Download results CSV", sub_all.to_csv(index=False).encode("utf-8"),
                           file_name=f"nba_{'game_bets' if is_game else 'props'}_results_{sel_date}.csv",
                           mime="text/csv")
elif st.session_state.get("results_for_date") == str(sel_date):
    st.warning(st.session_state.get("results_note") or "No results available.")
else:
    st.info("Pick a date in the sidebar and click **Load results**.")

# --- CLV -------------------------------------------------------------------------------------
st.divider()
st.header("📉 Closing Line Value (CLV)")
st.caption("Everything above scores the model against a base rate. This scores it against the "
           "market — the bar that decides whether any of it is worth betting.")
with st.expander("What this measures and why it matters", expanded=False):
    st.markdown("""
The **closing line** is the last price before tip-off — the sharpest number a book produces,
with all the late injury and rest news baked in (which matters even more in the NBA than MLB).

This fetches the slate twice — at the point you'd realistically have bet, and as close to tip as
the data allows — and asks of every flagged game-line pick: **did the price move toward us?**

- **% beat close above 50%** across many dates is the encouraging sign.
- Positive CLV still isn't profit — you need to clear the book's margin first.
- **One slate proves nothing.**
- Picks where the line itself moved (spread −4.5 → −6, total 224.5 → 227) are reported
  separately — they're no longer the same bet.
""")
k1, k2 = st.columns([2, 3])
with k1:
    lead = st.slider("Bet snapshot (minutes before tip)", 15, 240, 60, 15)
with k2:
    st.error("**Most expensive call in the app** — fetches the historical slate twice "
             "(~30 credits per distinct tip time, doubled). Cached for a day.")
flagged = st.checkbox("Only score picks the model flagged 🟢/🟡", value=True, key="clv_flagged_only")
if st.button("Run CLV report"):
    with st.spinner("Fetching bet-time and closing snapshots..."):
        cdf, cnote, ccost = build_clv_report(sel_date, bet_lead_minutes=lead, flagged_only=flagged)
    st.session_state.update(clv_df=cdf, clv_note=cnote, clv_cost=ccost, clv_for_date=str(sel_date))

if st.session_state.get("clv_for_date") == str(sel_date):
    cdf = st.session_state.get("clv_df")
    ccost = st.session_state.get("clv_cost") or {}
    if not isinstance(cdf, pd.DataFrame) or cdf.empty:
        st.warning(st.session_state.get("clv_note") or "No CLV data available.")
    else:
        st.caption(st.session_state.get("clv_note", ""))
        if ccost.get("remaining"):
            st.caption(f"This run used ~{ccost.get('total_credits')} credits, "
                       f"{ccost.get('remaining')} remaining.")
        comp = cdf[cdf["CLV %"].notna()]
        if comp.empty:
            st.warning("No picks were directly comparable — the line moved on all of them.")
        else:
            beat, n = int(comp["Beat close"].sum()), len(comp)
            pct, avg = beat / n * 100, comp["CLV %"].mean()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Picks compared", n)
            m2.metric("Beat the close", f"{beat}/{n}")
            m3.metric("% beat close", f"{pct:.1f}%")
            m4.metric("Average CLV", f"{avg:+.2f}%")
            if n < 20:
                st.info(f"Only {n} comparable pick(s) — far too small to read anything into.")
            elif pct > 50 and avg > 0:
                st.success(f"Beat the close on {pct:.1f}% of picks. Encouraging — but one date.")
            elif pct < 50:
                st.warning(f"Beat the close on only {pct:.1f}% of picks. If this persists across "
                           "many dates, it's the clearest sign the edge isn't there yet.")
        disp = cdf.copy()
        disp["Beat close"] = disp["Beat close"].map({True: "✅", False: "❌"})
        st.dataframe(disp.sort_values("CLV %", ascending=False, na_position="last").reset_index(drop=True),
                     width="stretch", hide_index=True)
        st.download_button("Download CLV CSV", cdf.to_csv(index=False).encode("utf-8"),
                           file_name=f"nba_clv_{sel_date}.csv", mime="text/csv", key="dl_clv")
