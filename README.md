# NBA Prop Analyser

Streamlit app for NBA betting analysis — same structure and feel as the MLB app.

## Pages
| Page | What it does | Odds API cost |
|---|---|---|
| Game Bets | Moneyline · Spread · Totals vs UK odds, traffic lights, Most Likely, acca builder | 3 credits (cached 15 min) |
| Player Props | Slate overview, prop edges (Points/Rebounds/Assists/Threes/PRA) vs US books, Most Likely milestones | 5 per game; slate overview + Most Likely free |
| Backtest | Game model calibration + margin/total error SDs (for fitting the model) | Free |
| Results | Model reads (free) or priced-up picks (historical odds) + CLV | Free / historical credits |
| Suggested Bets | Auto doubles/trebles per market, stake split, live status | Same as Game Bets + Props |

## Deploy (Streamlit Cloud)
1. Create a new GitHub repo (e.g. `nba-betting-app`) and upload everything in this folder, keeping the `pages/` folder.
2. On Streamlit Cloud: **New app** → pick the repo → Main file path: `app.py`.
3. **Settings → Secrets**, add:
   ```
   ODDS_API_KEY = "your-key"
   ```
4. Deploy. ESPN data needs no key.

## Files
- `nba_data.py` — every fetch (ESPN public API + The Odds API)
- `nba_model.py` — the maths; all tunable constants are at the top
- `nba_build.py` — builders the pages call
- `nba_ui.py` — page setup, sidebar, cards
- `nba_core.py` — one import for the pages

## Model in one paragraph
Team points = league average × own offence index × opponent defence index, ± home court, minus back-to-back and injury adjustments. Margin and total are Normal distributions. Early season, every rating blends with last season (regressed to average) until ~12 games in. Props: per-minute production (this season weighted fully, last season at 35%) × projected minutes × tonight's matchup factor from the game model, priced with a Negative Binomial using each player's own variance.

## Provisional — tune from real results
- `MARGIN_SD`, `TOTAL_SD`, `HOME_COURT`, `B2B_PENALTY` → use the Backtest page's fitted SDs and biases
- `CALIBRATION_FITS` → shrink toward 50%, not yet fitted
- `MARKET_TRUST_TIER` → stake weights, re-weight from tracked slips

## Known gaps
- A teammate's absence doesn't yet boost a player's projected usage
- ESPN has no historical injury report, so past-date rebuilds run without injuries
- Not live-tested against ESPN from the build environment — first run may surface a field-name tweak
