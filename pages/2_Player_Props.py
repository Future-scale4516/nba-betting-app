import streamlit as st
import pandas as pd
from nba_core import *

setup_page("NBA Prop Analyser — Player Props")
sel_date, use_inj = sidebar_date()

st.title("🎰 Player Props")
st.caption("Load the slate for matchup context, then explore prop edges (best value) and the "
           "most-likely view.")
st.divider()

# --- Slate overview (the NBA version of "Load Today's Slate") -----------------------
c1, c2 = st.columns([2, 3])
with c1:
    load_btn = st.button("Load Today's Slate")
with c2:
    st.markdown("**Auto-loads:** schedule · team ratings · back-to-backs · injury report · "
                "projected scores (ESPN, free)")
if load_btn:
    with st.spinner("Loading slate..."):
        overview, _ctx = build_slate_overview(sel_date, use_inj)
    st.session_state["slate_overview"] = overview
    st.session_state["slate_overview_date"] = str(sel_date)

if st.session_state.get("slate_overview_date") == str(sel_date):
    overview = st.session_state.get("slate_overview") or []
    if not overview:
        st.info("No games on this date.")
    for o in overview:
        rest = " · ".join(x for x in [f"{o['away']} on B2B" if o["a_b2b"] else "",
                                      f"{o['home']} on B2B" if o["h_b2b"] else ""] if x)
        with st.expander(f"🏀 {o['game']}  🕒 {o['start']} UK  |  Proj {o['proj_away']:.0f}-"
                         f"{o['proj_home']:.0f}  |  {o['state']}" + (f"  |  ⚠️ {rest}" if rest else "")):
            ca, ch = st.columns(2)
            for col, side, r, outs, qs in ((ca, o["away"], o["ra"], o["a_out"], o["a_q"]),
                                           (ch, o["home"], o["rh"], o["h_out"], o["h_q"])):
                with col:
                    st.markdown(f"### {side}")
                    st.caption(f"Offence index {r['off']*100:.0f} · Defence (allowed) index "
                               f"{r['def']*100:.0f} · {r['gp']} games this season")
                    st.write("**Out:** " + (", ".join(outs) if outs else "none reported"))
                    if qs:
                        st.write("**Questionable:** " + ", ".join(qs))
            st.caption(f"Home win {o['p_home']*100:.0f}% · projected total {o['total']:.0f}"
                       + (" · " + "; ".join(o["notes"]) if o["notes"] else ""))

# --- Prop edges ---------------------------------------------------------------------------
st.markdown("## 🎰 Player Prop Edges")
st.caption("Pulls US-book props (Points, Rebounds, Assists, Threes, PRA) per game, projects each "
           "player from per-minute production × projected minutes × tonight's matchup, and "
           "surfaces green/amber value. Out/Doubtful players are dropped. Each game costs 5 "
           "credits (one per market); finding the games is free.")
p1, p2 = st.columns([2, 3])
with p1:
    prop_max_games = st.slider("Games to analyse (5 credits each)", 1, 15, 6)
with p2:
    st.caption(f"Projected cost: up to {prop_max_games * 5} credits. Cached 15 min, so "
               "re-viewing the same games is free.")
if st.button("Find player prop edges (US books)"):
    with st.spinner("Pulling props, game logs and matchups..."):
        prop_df, prop_meta, prop_note = build_prop_edges(sel_date, prop_max_games, use_injuries=use_inj)
    if prop_df is None:
        st.warning(prop_note)
        st.session_state.pop("prop_edges", None)
    else:
        prop_df = prop_df.copy()
        if not prop_df.empty:
            prop_df["Game"] = prop_df["Game"] + " · " + prop_df["Start"]
        st.session_state["prop_edges"] = prop_df
        st.session_state["prop_edges_meta"] = prop_meta
        st.session_state["prop_edges_note"] = prop_note

prop_df = st.session_state.get("prop_edges")
if isinstance(prop_df, pd.DataFrame):
    meta = st.session_state.get("prop_edges_meta") or {}
    if meta.get("remaining"):
        st.caption(f"Quota — used {meta.get('used')}, remaining {meta.get('remaining')}")
    if st.session_state.get("prop_edges_note"):
        st.info(st.session_state["prop_edges_note"])
    if prop_df.empty:
        st.write("No green/amber prop edges found in the analysed games.")
    else:
        st.markdown(f"### 🟢 {int((prop_df['Light'] == '🟢').sum())} green · "
                    f"🟡 {int((prop_df['Light'] == '🟡').sum())} amber · "
                    f"🔴 {int((prop_df['Light'] == '🔴').sum())} implausible value props")
        show_form = st.checkbox(f"Show last-{FORM_GAMES} games form", key="prop_form_toggle")
        st.caption(f"Whether the player cleared *this pick's line* in each of his last "
                   f"{FORM_GAMES} games. Uses the same game logs the model already loaded, so "
                   "it's quick.")
        picked = st.multiselect("Filter by game", sorted(prop_df["Game"].unique()), default=[],
                                key="prop_game_filter", help="Leave empty to show every game.")
        view = prop_df[prop_df["Game"].isin(picked)] if picked else prop_df

        def show_prop_market(tab, label):
            with tab:
                sub = view[view["Market"] == label].copy()
                if sub.empty:
                    st.write("No value bets in this market today.")
                    return
                sub = sort_picker(sub, [("Edge (high to low)", "Edge", False),
                                        ("Model % (high to low)", "Model %", False),
                                        ("Odds (high to low)", "Best over", False)],
                                  key=f"sort_prop_{label}")
                for _, r in sub.iterrows():
                    metrics = [("Model %", f"{r['Model %']:.1f}%"), ("Market %", f"{r['Market %']:.1f}%"),
                               ("Edge", f"{r['Edge']:.1f} pts"), ("Odds", f"{r['Best over']:.2f}"),
                               ("Proj", f"{r['Proj']:.1f}")]
                    detail = r["Reason"]
                    if show_form:
                        try:
                            hits, played, syms = form_streak(r["PlayerID"], season_for(sel_date),
                                                             r["MarketKey"], r["Point"])
                            if hits is not None:
                                metrics.append(("Form", f"{hits}/{played}"))
                                detail = f"{syms}  —  {detail}"
                        except Exception:
                            pass
                    render_pick_card(r["Light"], f"{r['Player']} {r['Line']} {label}", r["Game"],
                                     metrics, reason=detail, conditions=r["Conditions"])

        tabs = st.tabs(["🏀 Points", "🙌 Rebounds", "🎯 Assists", "👌 Threes", "📦 PRA"])
        for tab, label in zip(tabs, ["Points", "Rebounds", "Assists", "Threes", "PRA"]):
            show_prop_market(tab, label)
        st.caption("🟢 edge 3–9 · 🟡 9–15. No-signal (<3) and 15+ edges are hidden. Model %: our "
                   "probability · Market %: de-vigged book probability · Proj: model's expected "
                   "stat. Teammate absences don't yet boost a player's usage in the model — when a "
                   "star sits, his teammates' lines move and the model won't fully follow.")

# --- Most likely ----------------------------------------------------------------------
st.markdown("## 🔮 Most Likely — milestones")
st.caption("Each rotation player's raw model probability of reaching a milestone (20+ points, "
           "8+ rebounds...). The 'most likely' lens — ignores odds, so pair it with Prop Edges. "
           "Free: no Odds API credits.")
ml_games = st.slider("Games to project", 1, 15, 6, key="ml_games")
if st.button("Rank most likely"):
    with st.spinner("Reading rotations, game logs and matchups..."):
        ml_df, ml_note = build_most_likely(sel_date, ml_games, use_injuries=use_inj)
    if ml_df is None:
        st.warning(ml_note)
        st.session_state.pop("most_likely_df", None)
    else:
        ml_df = ml_df.copy()
        ml_df["Game"] = ml_df["Game"] + " · " + ml_df["Start"]
        st.session_state["most_likely_df"] = ml_df
        st.session_state["most_likely_note"] = ml_note

ml_df = st.session_state.get("most_likely_df")
if isinstance(ml_df, pd.DataFrame):
    st.caption(st.session_state.get("most_likely_note", ""))
    show_form_ml = st.checkbox(f"Show last-{FORM_GAMES} games form", key="ml_form_toggle")
    picked = st.multiselect("Filter by game", sorted(ml_df["Game"].unique()), default=[],
                            key="ml_game_filter")
    view = ml_df[ml_df["Game"].isin(picked)] if picked else ml_df

    def show_ml(tab, label):
        with tab:
            k = st.selectbox("Milestone", MILESTONES[label], index=len(MILESTONES[label]) // 2,
                             key=f"ml_k_{label}", format_func=lambda x: f"{x}+ {label.lower()}")
            sub = view[(view["Market"] == label) & (view["Threshold"] == k)].copy()
            if sub.empty:
                st.write("No ranked players for this market.")
                return
            sub = sort_picker(sub, [("Model prob % (high to low)", "Value", False),
                                    ("Expected (high to low)", "Expected", False)],
                              key=f"sort_ml_{label}")
            for _, r in sub.head(40).iterrows():
                metrics = [("Model prob", f"{r['Value']:.1f}%"), ("Expected", f"{r['Expected']:.1f}"),
                           ("Min", f"{r['Min']:.0f}")]
                if show_form_ml:
                    try:
                        hits, played, _ = form_streak(r["PlayerID"], season_for(sel_date),
                                                      r["MarketKey"], k - 0.5)
                        if hits is not None:
                            metrics.append(("Form", f"{hits}/{played}"))
                    except Exception:
                        pass
                render_pick_card(None, f"{r['Player']} — {k}+ {label.lower()}", r["Game"],
                                 metrics, conditions=r["Conditions"])

    tabs = st.tabs(["🏀 Points", "🙌 Rebounds", "🎯 Assists", "👌 Threes", "📦 PRA"])
    for tab, label in zip(tabs, ["Points", "Rebounds", "Assists", "Threes", "PRA"]):
        show_ml(tab, label)
    st.caption("Most likely is not the same as best bet: a player can be very likely yet fairly "
               "priced. These are raw (uncalibrated) probabilities — the Results page scores "
               "them against what actually happened.")
