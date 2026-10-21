import streamlit as st
import pandas as pd
import statistics
from nba_core import *

setup_page("NBA Prop Analyser — Backtest")
sel_date, use_inj = sidebar_date()

st.title("🧪 Backtest — game model")
st.caption("Re-runs the game model on recent completed games, rebuilding every team rating "
           "AS OF that day (only games before it count), and checks it against real results. "
           "Free ESPN data, no Odds API credits. Injuries aren't included — ESPN has no "
           "historical injury report — so this tests the base model.")
st.info("Early in a season there's little to test. To check the model on a full season, set "
        "the sidebar date to a spring date (e.g. 2026-04-10) — it'll backtest the 2025-26 "
        "season using 2024-25 as the prior.")

days = st.slider("Days to look back", 3, 45, 14)
if st.button("Run backtest"):
    with st.spinner("Rebuilding ratings day by day and scoring games..."):
        recs, resid, days_done = run_game_backtest(sel_date, days)
    st.session_state["bt"] = (recs, resid, days_done, str(sel_date))

bt = st.session_state.get("bt")
if bt and bt[3] == str(sel_date):
    recs, resid, days_done, _ = bt
    if not recs:
        st.warning("No completed games in that window.")
    else:
        s = calib_summary(recs, "Moneyline (home win)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Games", s["n"])
        c2.metric("Days with games", days_done)
        c3.metric("Picked winner", f"{s['acc']:.1f}%")
        c4.metric("Brier score", f"{s['brier']:.3f}", help="Lower is better. 0.25 = a coin flip; "
                  "good NBA models sit around 0.20-0.21.")
        st.caption(f"Home teams won {s['base_rate']:.1f}% of these games.")

        st.markdown("**Calibration — does 70% mean 70%?**")
        bdf = pd.DataFrame(s["buckets"], columns=["Model says", "Games", "Avg model %", "Actual %"])
        st.dataframe(bdf, width="stretch", hide_index=True)
        st.caption("Ignore buckets with only a handful of games — they're noise, not a trend.")

        st.markdown("**Margin & total errors — fitting the spread/total model**")
        msd = statistics.pstdev(resid["margin"]) if len(resid["margin"]) > 1 else None
        tsd = statistics.pstdev(resid["total"]) if len(resid["total"]) > 1 else None
        mbias = statistics.mean(resid["margin"])
        tbias = statistics.mean(resid["total"])
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Margin error SD", f"{msd:.1f}" if msd else "—", f"model uses {MARGIN_SD}",
                  delta_color="off")
        d2.metric("Total error SD", f"{tsd:.1f}" if tsd else "—", f"model uses {TOTAL_SD}",
                  delta_color="off")
        d3.metric("Margin bias", f"{mbias:+.1f}", help="Actual minus model, home perspective. "
                  "Persistently positive = home court is under-rated (raise HOME_COURT).")
        d4.metric("Total bias", f"{tbias:+.1f}", help="Actual minus model. Persistently positive = "
                  "scoring is running above the blended league average.")
        st.caption("Once a few hundred games have been tested, the two SDs above are the fitted "
                   "replacements for MARGIN_SD / TOTAL_SD in nba_model.py — the same way the MLB "
                   "app's calibration numbers were fitted from its backtest.")
