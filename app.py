import streamlit as st
from nba_core import *

setup_page("NBA Prop Analyser")
sel_date, use_inj = sidebar_date()

st.title("🏀 NBA Prop Analyser")
st.caption("Same workflow as the MLB app — pick a slate date in the sidebar, then use the "
           "pages on the left.")

st.markdown("""
**Pages**
- **Game Bets** — Moneyline · Spread · Totals vs UK odds, traffic-light edges, Most Likely, acca builder
- **Player Props** — slate overview (rest, injuries, projections), prop edges vs US books, Most Likely milestones
- **Backtest** — how the game model's win probabilities and projected margins/totals have held up
- **Results** — what the model said vs what happened (free model reads, or priced-up with historical odds) + CLV
- **Suggested Bets** — auto-built doubles/trebles per market with stake split and live status
""")

st.info("**Early-season note:** for the first few weeks every team rating and player "
        "projection leans on last season (regressed toward average), shifting to this "
        "season's numbers as games accrue. Expect the model to be at its least reliable "
        "now — paper-trade until Results/Backtest have some history.")

if st.button("Quick slate check"):
    with st.spinner("Loading slate..."):
        games = fetch_scoreboard(str(sel_date))
    if not games:
        st.warning("No NBA games on this date.")
    else:
        for g in sorted(games, key=lambda x: x["start_iso"]):
            score = (f" — {g['away_score']}-{g['home_score']}"
                     if g["state"] != "pre" and g["home_score"] is not None else "")
            st.write(f"• **{g['away_name']} @ {g['home_name']}** · {to_uk_time(g['start_iso'])} UK "
                     f"· {g['detail']}{score}")
