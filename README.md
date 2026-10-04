# Fantasy Matchup Lab

A web page with eight tools:

- **This week's top 10:** projected best QBs, RBs, WRs, and TEs for the upcoming week (injured players left off).
- **Waiver wire:** the best players likely to be available in 8-, 10-, and 12-team leagues.
- **Defense vs position:** points every defense allows per game to each position or depth-chart role (RB1–RB4, WR1–WR6, TE1–TE4), as a heat map.
- **Team offense:** points, fantasy production, yards, and target/carry shares for every offense.
- **Injuries:** the official NFL injury report by player or by team, with who's next up on the depth chart.
- **Team of the week:** the highest-scoring possible lineup for any week since 2015.
- **Team of the season:** the best lineup by average points per game for any season since 2015, with a minimum-games filter.
- **Game log:** every QB/RB/WR/TE game, filterable by position, role, opponent, team, home/away, week range, points, and player name.

Standard, half PPR, and PPR scoring all work. Stats, schedules, and injury reports come from [nflverse](https://github.com/nflverse), which is free and updates nightly during the season. ESPN's public fantasy data is used for % rostered and IR designations when it's reachable; if it isn't, the page estimates availability instead.

## Put it online (about 10 minutes, free)

1. Create a free account at github.com, then click **New repository**. Name it something like `fantasy-lab` and make it **Public**.
2. On the new repo page, click **uploading an existing file** and drag in everything from this folder: `fetch_data.py`, `README.md`, the `site` folder, and the `.github` folder.
   - The `.github` folder is hidden on Mac. In Finder press **Cmd + Shift + .** to show it.
   - If drag-and-drop skips the folder, create the file in the browser instead: **Add file → Create new file**, name it `.github/workflows/update.yml`, and paste in the contents.
3. Go to **Settings → Pages**. Under **Source**, choose **GitHub Actions**.
4. Go to the **Actions** tab, click **Update stats and publish site**, then **Run workflow**.
5. When it finishes (about a minute), your site is at `https://YOUR-USERNAME.github.io/fantasy-lab/`.

After that it updates itself Tuesday, Thursday, Saturday, and Sunday mornings, so injury news is fresh before games. You never need to touch it.

Note: GitHub pauses scheduled jobs on repos with no activity for 60 days. If updates stop, open the Actions tab and re-enable it.

## Run it on your own computer instead

Requires Python 3.

```bash
pip install pandas numpy
python fetch_data.py          # downloads stats, writes site/data.json
cd site
python -m http.server 8000    # then open http://localhost:8000
```

Opening `index.html` by double-clicking won't load the data (browsers block that). The page will let you pick `data.json` manually, or use the local server above.

## Files the update creates

`fetch_data.py` writes two files into `site/`: `data.json` (this season and last, used by most tabs) and `history.json` (every season since 2015, used only by the team of the week and team of the season tabs, and loaded in the background). To change how far back history goes, edit `HISTORY_FROM` at the top of `fetch_data.py`.

## Tweaking the projections

The settings at the top of `fetch_data.py` control the model: how much recent games matter (`DECAY`), how much last season counts early on (`PRIOR_GAMES`), and how far matchups, team totals, and usage can move a projection. The page's "How the picks are made" section explains each piece.

The model doesn't know about injuries, depth-chart changes, or weather. Always check news before setting a lineup.
