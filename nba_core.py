"""Single import point for the pages: `from nba_core import *`.

The code is split across four smaller modules (instead of one big file like
mlb_core.py) so no single file gets near the size where the GitHub web editor
starts truncating pastes:
  nba_data.py   — every external fetch (ESPN + The Odds API)
  nba_model.py  — the maths (team ratings, game model, prop model, bands)
  nba_build.py  — builders the pages call (edges, results, CLV, backtest)
  nba_ui.py     — page setup, sidebar, cards
"""
from nba_data import *    # noqa: F401,F403
from nba_model import *   # noqa: F401,F403
from nba_build import *   # noqa: F401,F403
from nba_ui import *      # noqa: F401,F403
