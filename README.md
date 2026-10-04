# Fantasy Matchup Lab

A web page with three tools:

- **This week's top 10:** projected best QBs, RBs, WRs, and TEs for the upcoming week.
- **Defense vs position:** points every defense allows per game to each position, as a heat map. Click a cell to see every player who faced that defense.
- **Game log:** every QB/RB/WR/TE game, filterable by position, opponent, team, home/away, week range, points, and player name.

Standard, half PPR, and PPR scoring all work. Stats come from [nflverse](https://github.com/nflverse), which is free and updates nightly during the season.

## Put it online (about 10 minutes, free)

1. Create a free account at github.com, then click **New repository**. Name it something like `fantasy-lab` and make it **Public**.
2. On the new repo page, click **uploading an existing file** and drag in everything from this folder: `fetch_data.py`, `README.md`, the `site` folder, and the `.github` folder.
   - The `.github` folder is hidden on Mac. In Finder press **Cmd + Shift + .** to show it.
   - If drag-and-drop skips the folder, create the file in the browser instead: **Add file → Create new file**, name it `.github/workflows/update.yml`, and paste in the contents.
3. Go to **Settings → Pages**. Under **Source**, choose **GitHub Actions**.
4. Go to the **Actions** tab, click **Update stats and publish site**, then **Run workflow**.
5. When it finishes (about a minute), your site is at `https://YOUR-USERNAME.github.io/fantasy-lab/`.

After that it updates itself every Tuesday and Saturday morning. You never need to touch it.

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

## Tweaking the projections

The settings at the top of `fetch_data.py` control the model: how much recent games matter (`DECAY`), how much last season counts early on (`PRIOR_GAMES`), and how far matchups, team totals, and usage can move a projection. The page's "How the picks are made" section explains each piece.

The model doesn't know about injuries, depth-chart changes, or weather. Always check news before setting a lineup.
