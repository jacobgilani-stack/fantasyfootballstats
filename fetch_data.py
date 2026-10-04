#!/usr/bin/env python3
"""
Builds data.json for the fantasy matchup dashboard.

Data source: nflverse (free, public NFL stats, updated nightly during the season).
  - Weekly player stats (fantasy points already calculated)
  - Schedule with Vegas lines (used for team implied totals)

Usage:
    python fetch_data.py                      # current season -> site/data.json
    python fetch_data.py --season 2025        # a specific season
    python fetch_data.py --out somewhere.json
"""
import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request
from io import StringIO

import numpy as np
import pandas as pd

POSITIONS = ["QB", "RB", "WR", "TE"]
FORMATS = ["std", "half", "ppr"]

STATS_URLS = [
    # Newer nflverse release (2025+)
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.csv",
    # Older release name, kept as a fallback
    "https://github.com/nflverse/nflverse-data/releases/download/player_stats/player_stats_{season}.csv",
]
SCHEDULE_URLS = [
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
    "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv",
]

# ---- Projection model settings (tweak these to change how picks are made) ----
DECAY = 0.85            # weight of each older game vs the next newer one
PRIOR_GAMES = 2.0       # how many "games" last season's average counts for
MATCHUP_SHRINK = 3.0    # pulls defenses with few games back toward average
MATCHUP_CLIP = (0.80, 1.20)
ENV_EXPONENT = 0.6      # how strongly the team implied total moves projections
ENV_CLIP = (0.85, 1.15)
USAGE_EXPONENT = 0.35   # how strongly recent touches/targets trend matters
USAGE_CLIP = (0.90, 1.10)
TOP_N = 25              # players kept per position (page shows 10 by default)

STAT_MAP = {
    "pyd": ["passing_yards"],
    "ptd": ["passing_tds"],
    "int": ["passing_interceptions", "interceptions"],
    "att": ["attempts"],
    "car": ["carries"],
    "ryd": ["rushing_yards"],
    "rtd": ["rushing_tds"],
    "tgt": ["targets"],
    "rec": ["receptions"],
    "reyd": ["receiving_yards"],
    "retd": ["receiving_tds"],
}
FUMBLE_COLS = ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]
TWO_PT_COLS = ["passing_2pt_conversions", "rushing_2pt_conversions", "receiving_2pt_conversions"]
ROW_COLS = ["s", "p", "wk", "tm", "opp", "home", "std", "half", "ppr",
            "pyd", "ptd", "int", "car", "ryd", "rtd", "tgt", "rec", "reyd", "retd"]


# ----------------------------------------------------------------------------
# Downloading
# ----------------------------------------------------------------------------
def current_season(today=None):
    today = today or dt.date.today()
    return today.year if today.month >= 3 else today.year - 1


def fetch_csv(url):
    req = urllib.request.Request(url, headers={"User-Agent": "fantasy-matchup-dashboard"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        text = resp.read().decode("utf-8")
    return pd.read_csv(StringIO(text), low_memory=False)


def load_first(urls, label):
    errors = []
    for url in urls:
        try:
            df = fetch_csv(url)
            print(f"  loaded {label}: {url} ({len(df)} rows)")
            return df
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            errors.append(f"{url}: {e}")
    print(f"  could not load {label}:\n    " + "\n    ".join(errors))
    return None


def load_stats(season):
    return load_first([u.format(season=season) for u in STATS_URLS], f"{season} player stats")


# ----------------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------------
def pick(df, names):
    for n in names:
        if n in df.columns:
            return pd.to_numeric(df[n], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def normalize_stats(raw, season):
    """Turn an nflverse weekly stats file into one tidy row per player-game."""
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if "season_type" in df.columns:
        df = df[df["season_type"].astype(str).str.upper() == "REG"]
    pos_col = "position" if "position" in df.columns else "position_group"
    df = df[df[pos_col].isin(POSITIONS)].copy()
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=df.index)
    out["s"] = season
    out["id"] = df["player_id"].astype(str)
    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    out["name"] = df[name_col].astype(str)
    out["pos"] = df[pos_col].astype(str)
    team_col = "team" if "team" in df.columns else "recent_team"
    out["tm"] = df[team_col].astype(str)
    out["opp"] = df["opponent_team"].astype(str) if "opponent_team" in df.columns else ""
    out["wk"] = pd.to_numeric(df["week"], errors="coerce").fillna(0).astype(int)
    for key, names in STAT_MAP.items():
        out[key] = pick(df, names)

    fumbles = sum(pick(df, [c]) for c in FUMBLE_COLS)
    two_pt = sum(pick(df, [c]) for c in TWO_PT_COLS)
    computed_std = (0.04 * out["pyd"] + 4 * out["ptd"] - 2 * out["int"]
                    + 0.1 * (out["ryd"] + out["reyd"]) + 6 * (out["rtd"] + out["retd"])
                    - 2 * fumbles + 2 * two_pt)
    if "fantasy_points" in df.columns:
        std = pd.to_numeric(df["fantasy_points"], errors="coerce").fillna(computed_std)
    else:
        std = computed_std
    if "fantasy_points_ppr" in df.columns:
        ppr = pd.to_numeric(df["fantasy_points_ppr"], errors="coerce").fillna(std + out["rec"])
    else:
        ppr = std + out["rec"]
    out["std"] = std
    out["ppr"] = ppr
    out["half"] = (std + ppr) / 2
    out = out[out["wk"] > 0]
    return out.reset_index(drop=True)


def normalize_schedule(raw, season):
    if raw is None or raw.empty:
        return pd.DataFrame()
    s = raw[(raw["season"] == season) & (raw["game_type"] == "REG")].copy()
    s["week"] = s["week"].astype(int)
    for c in ["home_score", "away_score", "total_line", "spread_line"]:
        s[c] = pd.to_numeric(s[c], errors="coerce") if c in s.columns else np.nan
    if "gameday" not in s.columns:
        s["gameday"] = ""
    s["final"] = s["home_score"].notna() & s["away_score"].notna()
    return s


def team_games(sched):
    """One row per team per game, with the team's Vegas implied point total."""
    if sched.empty:
        return pd.DataFrame(columns=["week", "team", "opp", "home", "imp", "final", "gameday"])
    total, spread = sched["total_line"], sched["spread_line"]  # spread > 0 means home favored
    home = pd.DataFrame({"week": sched["week"], "team": sched["home_team"], "opp": sched["away_team"],
                         "home": True, "imp": (total + spread) / 2, "final": sched["final"],
                         "gameday": sched["gameday"].astype(str)})
    away = pd.DataFrame({"week": sched["week"], "team": sched["away_team"], "opp": sched["home_team"],
                         "home": False, "imp": (total - spread) / 2, "final": sched["final"],
                         "gameday": sched["gameday"].astype(str)})
    return pd.concat([home, away], ignore_index=True)


def attach_schedule(rows, tg):
    """Add home/away and fill missing opponents from the schedule."""
    if rows.empty:
        return rows
    if tg.empty:
        rows["home"] = None
        return rows
    m = rows.merge(tg[["week", "team", "opp", "home"]].rename(columns={"week": "wk", "team": "tm", "opp": "sched_opp"}),
                   on=["wk", "tm"], how="left")
    m["opp"] = np.where((m["opp"].isin(["", "nan", "None"])) & m["sched_opp"].notna(), m["sched_opp"], m["opp"])
    return m.drop(columns=["sched_opp"])


def week_status(sched):
    if sched.empty:
        return 0, None
    finals = sched.groupby("week")["final"].all()
    done = [w for w, f in finals.items() if f]
    open_ = [w for w, f in finals.items() if not f]
    return (max(done) if done else 0), (min(open_) if open_ else None)


# ----------------------------------------------------------------------------
# Projections
# ----------------------------------------------------------------------------
def defense_factors(hist, fmt):
    """How many points each defense allows to each position, vs league average."""
    g = hist.groupby(["opp", "wk", "pos"])[fmt].sum().reset_index()
    league = g.groupby("pos")[fmt].mean()
    d = g.groupby(["opp", "pos"])[fmt].agg(["mean", "count"]).reset_index()
    d["raw"] = d["mean"] / d["pos"].map(league)
    d["factor"] = (1 + (d["raw"] - 1) * d["count"] / (d["count"] + MATCHUP_SHRINK)).clip(*MATCHUP_CLIP)
    d["rank"] = d.groupby("pos")["mean"].rank(ascending=False, method="min").astype(int)
    d["teams"] = d.groupby("pos")["opp"].transform("count")
    return {(r.opp, r.pos): (r.factor, r.rank, r.teams, r.raw) for r in d.itertuples()}


def build_projections(cur, prev, tg, next_week):
    if next_week is None:
        return {"week": None, "status": "season_over", "players": {}}
    hist = cur[cur["wk"] < next_week]
    if hist.empty:
        return {"week": next_week, "status": "no_games_yet", "players": {}}

    upcoming = tg[tg["week"] == next_week].set_index("team")
    imps = upcoming["imp"].dropna()
    league_imp = imps.mean() if len(imps) else None
    team_played_last = set(tg[(tg["week"] == next_week - 1)]["team"])
    actual_this_week = cur[cur["wk"] == next_week].set_index("id")

    prev_avgs = {}
    if not prev.empty:
        pg = prev.groupby("id")
        prev_counts = pg.size()
        for fmt in FORMATS:
            prev_avgs[fmt] = pg[fmt].mean()[prev_counts >= 4]

    result = {}
    for pos in POSITIONS:
        result[pos] = {}
        ph = hist[hist["pos"] == pos]
        per_game = ph.groupby("id")[FORMATS].mean()
        counts = ph.groupby("id").size()
        for fmt in FORMATS:
            dfac = defense_factors(hist, fmt)
            floor = per_game.loc[counts[counts >= 2].index, fmt].quantile(0.30) if (counts >= 2).any() else 0.0
            entries = []
            for pid, games in ph.sort_values("wk").groupby("id"):
                last = games.iloc[-1]
                team = last["tm"]
                if team not in upcoming.index:
                    continue  # bye week or unknown team
                if last["wk"] < next_week - 3:
                    continue  # hasn't played in a while (likely injured/benched)
                n = len(games)
                vals = games[fmt].to_numpy()
                weights = DECAY ** np.arange(n - 1, -1, -1)
                wavg = float(np.dot(vals, weights) / weights.sum())
                prev_avg = prev_avgs.get(fmt, pd.Series(dtype=float)).get(pid)
                if prev_avg is not None and not np.isnan(prev_avg):
                    base = (n * wavg + PRIOR_GAMES * prev_avg) / (n + PRIOR_GAMES)
                else:
                    base = (n * wavg + floor) / (n + 1)

                if pos == "QB":
                    opps = (games["att"] + games["car"]).to_numpy()
                else:
                    opps = (games["tgt"] + games["car"]).to_numpy()
                use = 1.0
                if n >= 4 and opps.mean() > 0:
                    use = float(np.clip((opps[-3:].mean() / opps.mean()) ** USAGE_EXPONENT, *USAGE_CLIP))

                g = upcoming.loc[team]
                opp = g["opp"]
                mu, drank, dteams, draw = dfac.get((opp, pos), (1.0, None, None, 1.0))
                imp = g["imp"]
                env = 1.0
                if league_imp and not pd.isna(imp):
                    env = float(np.clip((imp / league_imp) ** ENV_EXPONENT, *ENV_CLIP))

                proj = base * mu * env * use
                played = bool(g["final"])
                actual = None
                if played and pid in actual_this_week.index:
                    a = actual_this_week.loc[pid]
                    actual = round(float(a[fmt] if not isinstance(a, pd.DataFrame) else a[fmt].sum()), 1)
                entries.append({
                    "id": pid, "name": last["name"], "tm": team, "opp": opp, "home": bool(g["home"]),
                    "day": g["gameday"], "proj": round(proj, 1), "base": round(base, 1),
                    "mu": round(float(mu), 3), "allow": round(float(draw), 3),
                    "dr": int(drank) if drank is not None else None,
                    "dteams": int(dteams) if dteams is not None else None,
                    "imp": None if pd.isna(imp) else round(float(imp), 1),
                    "env": round(env, 3), "use": round(use, 3), "gp": n,
                    "missed": bool(team in team_played_last and last["wk"] < next_week - 1),
                    "played": played, "actual": actual,
                })
            entries.sort(key=lambda e: e["proj"], reverse=True)
            result[pos][fmt] = entries[:TOP_N]
    return {"week": next_week, "status": "ok", "players": result}


# ----------------------------------------------------------------------------
# Assemble data.json
# ----------------------------------------------------------------------------
def build(cur_raw, prev_raw, sched_raw, season, sample=False):
    cur = normalize_stats(cur_raw, season)
    prev = normalize_stats(prev_raw, season - 1)
    sched = normalize_schedule(sched_raw, season)
    prev_sched = normalize_schedule(sched_raw, season - 1)
    tg = team_games(sched)
    cur = attach_schedule(cur, tg)
    prev = attach_schedule(prev, team_games(prev_sched))
    last_done, next_week = week_status(sched)

    projections = build_projections(cur, prev, tg, next_week) if not cur.empty else \
        {"week": next_week, "status": "no_games_yet", "players": {}}

    both = pd.concat([d for d in (prev, cur) if not d.empty], ignore_index=True)
    # Player table (latest name/position wins)
    players = both.sort_values(["s", "wk"]).groupby("id").last()[["name", "pos"]].reset_index()
    pindex = {pid: i for i, pid in enumerate(players["id"])}

    rows = []
    for r in both.itertuples(index=False):
        home = None if pd.isna(r.home) else int(bool(r.home))
        rows.append([int(r.s), pindex[r.id], int(r.wk), r.tm, r.opp, home,
                     round(float(r.std), 1), round(float(r.half), 2), round(float(r.ppr), 1),
                     int(r.pyd), int(r.ptd), int(r.int), int(r.car), int(r.ryd), int(r.rtd),
                     int(r.tgt), int(r.rec), int(r.reyd), int(r.retd)])

    weeks_with_stats = sorted(cur["wk"].unique().tolist()) if not cur.empty else []
    return {
        "meta": {
            "season": season,
            "seasons": sorted(both["s"].unique().tolist()),
            "updated": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "last_completed_week": int(last_done),
            "latest_stats_week": int(max(weeks_with_stats)) if weeks_with_stats else 0,
            "next_week": next_week,
            "source": "nflverse (github.com/nflverse)",
            "sample": sample,
        },
        "players": players[["id", "name", "pos"]].values.tolist(),
        "cols": ROW_COLS,
        "rows": rows,
        "projections": projections,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--out", default="site/data.json")
    args = ap.parse_args()
    season = args.season or current_season()

    print(f"Building data for the {season} season")
    sched_raw = load_first(SCHEDULE_URLS, "schedule")
    cur_raw = load_stats(season)
    prev_raw = load_stats(season - 1)
    if sched_raw is None:
        sys.exit("Stopped: the schedule could not be downloaded.")
    if cur_raw is None and prev_raw is None:
        sys.exit("Stopped: no player stats could be downloaded.")

    data = build(cur_raw, prev_raw, sched_raw, season)
    if not data["rows"]:
        sys.exit("Stopped: stats downloaded but no QB/RB/WR/TE rows were found.")

    with open(args.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    m = data["meta"]
    print(f"Wrote {args.out}: {len(data['rows'])} player-games, "
          f"stats through week {m['latest_stats_week']}, projections for week {m['next_week']}")


if __name__ == "__main__":
    main()
