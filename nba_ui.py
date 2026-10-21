"""Shared page chrome — same look and card format as the MLB app."""
import streamlit as st

from nba_data import today_et


def setup_page(title="NBA Prop Analyser"):
    """Per-page config + shared styling. Must be the first Streamlit call on a page."""
    st.set_page_config(page_title=title, page_icon="🏀", layout="wide")
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html,body,[class*="css"]{font-family:'Inter',sans-serif;}
    .stApp{background:#f7f6f2;}
    .metric-card{background:#fff;border-radius:12px;padding:14px 18px;border:1px solid #dcd9d5;
      box-shadow:0 2px 8px rgba(0,0,0,.05);text-align:center;margin-bottom:10px;}
    .metric-value{font-size:1.7rem;font-weight:700;color:#c4511f;}
    .metric-label{font-size:.72rem;color:#7a7974;text-transform:uppercase;letter-spacing:.05em;margin-top:3px;}
    section[data-testid="stSidebar"]{background:#1c1b19;}
    section[data-testid="stSidebar"] *{color:#cdccca !important;}
    </style>
    """, unsafe_allow_html=True)


def sidebar_date():
    """Shared sidebar: slate date (US Eastern — the NBA's own calendar, so a
    UK late-night check still lands on the right slate) + injury toggle +
    clear cache. Returns (date, use_injuries)."""
    with st.sidebar:
        st.markdown("## 🏀 NBA Props")
        sel = st.date_input("Slate Date (US)", value=today_et(), key="sel_date_picker")
        use_inj = st.toggle("Adjust for injuries", value=True, key="use_injuries",
                            help="Uses ESPN's current injury report: Out/Doubtful players are "
                                 "removed from props and their scoring is taken off their "
                                 "team's projection. Current slates only — ESPN has no "
                                 "historical report. Slower on the first load of the day.")
        if st.button("Clear Cache", key="clear_cache_btn"):
            st.cache_data.clear()
            st.rerun()
    return sel, use_inj


def render_pick_card(light, title, subtitle, metrics, reason=None, conditions=None):
    """One pick as a compact, phone-friendly card: header, one dense metrics
    line, then optional reason and conditions captions."""
    with st.container(border=True):
        st.markdown(f"{light} **{title}**" if light else f"**{title}**")
        if subtitle:
            st.caption(subtitle)
        st.caption("  ·  ".join(f"{label}: {value}" for label, value in metrics))
        if reason:
            st.caption(reason)
        if conditions:
            st.caption(f"📍 {conditions}")


def sort_picker(df, sort_options, key):
    """Sort-by selectbox for a card list. sort_options: [(label, column, ascending)]."""
    labels = [lbl for lbl, _, _ in sort_options]
    choice = st.selectbox("Sort by", labels, key=key)
    _, col, asc = next(o for o in sort_options if o[0] == choice)
    return df.sort_values(col, ascending=asc).reset_index(drop=True)
