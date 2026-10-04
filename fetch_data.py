#!/usr/bin/env python3
"""
Builds data.json for the fantasy matchup dashboard.

Data sources:
  - nflverse (free, public NFL data, updated nightly during the season):
      weekly player stats, schedule with Vegas lines, official injury reports
  - ESPN's public fantasy endpoint (optional): % rostered and injury status.
    If ESPN can't be reached, the page falls back to estimates.

Usage:
    python fetch_data.py                      # current season -> site/data.json
    python fetch_data.py --season 2025        # a specific season
    python fetch_data.py --out somewhere.json
"""
import argparse
import datetime as dt
import json
import os
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

INJURY_URLS = [
    "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.csv",
]
ESPN_URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
            "/segments/0/leaguedefaults/3?view=kona_player_info")
ESPN_FILTER = {"players": {"limit": 2000, "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}}
ESPN_POS = {1: "QB", 2: "RB", 3: "WR", 4: "TE"}
ESPN_TEAM = {1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET", 9: "GB",
             10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LA", 15: "MIA", 16: "MIN", 17: "NE", 18: "NO",
             19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI", 23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB",
             28: "WAS", 29: "CAR", 30: "JAX", 33: "BAL", 34: "HOU"}
ESPN_STATUS = {"QUESTIONABLE": "Questionable", "DOUBTFUL": "Doubtful", "OUT": "Out", "INJURY_RESERVE": "IR",
               "SUSPENSION": "Suspended", "DAY_TO_DAY": "Questionable", "PUP": "PUP"}
OUT_STATUSES = {"Out", "Doubtful", "IR", "Suspended", "PUP"}

# ---- History (team of the week / season tabs) ----
HISTORY_FROM = 2015
HIST_PER_POS = 10   # top players kept per position per week (per scoring format)

# ---- Waiver wire settings ----
LEAGUE_SIZES = [8, 10, 12]
# Typical number of players rostered per fantasy team at each position (16-man rosters)
ROSTER_PER_TEAM = {"QB": 1.6, "RB": 4.6, "WR": 5.2, "TE": 1.6}
# Starters per fantasy team at each position (flex split between RB and WR), for comparing positions
STARTERS_PER_TEAM = {"QB": 1.0, "RB": 2.4, "WR": 2.6, "TE": 1.0}
TEAMMATE_OUT_BOOST = 1.10  # bump when a teammate ahead of him on the depth chart is out
WAIVER_PER_POS = 12

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


def load_injuries(season):
    return load_first([u.format(season=season) for u in INJURY_URLS], f"{season} injury reports")


def norm_name(s):
    s = str(s).lower()
    for ch in ".'’,":
        s = s.replace(ch, "")
    s = s.replace("-", " ")
    return " ".join(p for p in s.split() if p not in {"jr", "sr", "ii", "iii", "iv", "v"})


def load_espn(season):
    """% rostered and injury status from ESPN's public fantasy endpoint. Optional."""
    req = urllib.request.Request(ESPN_URL.format(season=season), headers={
        "User-Agent": "Mozilla/5.0 (fantasy-matchup-dashboard)", "Accept": "application/json",
        "X-Fantasy-Filter": json.dumps(ESPN_FILTER)})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001 - optional source, never fatal
        print(f"  ESPN data unavailable ({e}); using estimates instead")
        return None
    items = data.get("players", []) if isinstance(data, dict) else data
    out = []
    for item in items or []:
        p = item.get("player", item) if isinstance(item, dict) else {}
        pos = ESPN_POS.get(p.get("defaultPositionId"))
        if not pos or not p.get("fullName"):
            continue
        own = (p.get("ownership") or {}).get("percentOwned")
        out.append({"key": norm_name(p["fullName"]), "pos": pos, "tm": ESPN_TEAM.get(p.get("proTeamId")),
                    "own": None if own is None else float(own),
                    "status": ESPN_STATUS.get(str(p.get("injuryStatus", "")).upper())})
    if not out:
        print("  ESPN responded but returned no players; using estimates instead")
        return None
    print(f"  loaded ESPN player info ({len(out)} players)")
    return pd.DataFrame(out)


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
# Injuries, depth charts, ESPN matching
# ----------------------------------------------------------------------------
def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    v = str(v).strip()
    return v or None


def short_practice(p):
    p = clean(p)
    if not p:
        return None
    pl = p.lower()
    if "did not" in pl:
        return "Did not practice"
    if "limited" in pl:
        return "Limited"
    if "full" in pl:
        return "Full"
    return p


def normalize_injuries(raw, season):
    """Latest official NFL injury report in the file (all positions)."""
    if raw is None or raw.empty:
        return pd.DataFrame(), None
    df = raw.copy()
    if "season" in df.columns:
        df = df[df["season"] == season]
    if "game_type" in df.columns:
        df = df[df["game_type"] == "REG"]
    if df.empty:
        return pd.DataFrame(), None
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    wk = int(df["week"].max())
    df = df[df["week"] == wk]
    if "full_name" in df.columns:
        names = df["full_name"]
    else:
        names = df.get("first_name", "").astype(str) + " " + df.get("last_name", "").astype(str)
    out = pd.DataFrame({
        "id": df["gsis_id"] if "gsis_id" in df.columns else None,
        "name": names, "tm": df["team"], "pos": df["position"],
        "status": df["report_status"] if "report_status" in df.columns else None,
        "injury": df["report_primary_injury"] if "report_primary_injury" in df.columns else None,
        "practice": df["practice_status"] if "practice_status" in df.columns else None,
    })
    out = out.map(clean) if hasattr(out, "map") else out.applymap(clean)
    out["practice"] = out["practice"].map(short_practice)
    out = out[out["status"].notna() | out["practice"].isin(["Did not practice", "Limited"])]
    out = out.drop_duplicates(subset=["name", "tm"]).reset_index(drop=True).astype(object)
    return out.where(out.notna(), None), wk


def depth_charts(cur):
    """Rank each team's players at each position by season-to-date usage with that team."""
    if cur.empty:
        return {}
    latest = cur.sort_values("wk").groupby("id").last()
    use = cur.assign(u=np.where(cur["pos"] == "QB", cur["att"] + cur["car"],
                                np.where(cur["pos"] == "RB", cur["car"] + cur["tgt"], cur["tgt"])))
    tot = use.groupby(["id", "tm"])["u"].sum().reset_index()
    tot = tot[tot.apply(lambda r: latest.loc[r["id"], "tm"] == r["tm"], axis=1)]
    tot["pos"] = tot["id"].map(latest["pos"])
    charts = {}
    for (tm, pos), g in tot.sort_values("u", ascending=False).groupby(["tm", "pos"], sort=False):
        charts.setdefault(tm, {})[pos] = g["id"].tolist()
    return charts


def match_espn(espn, players):
    """Map our player ids to ESPN rows by name + position (team breaks ties)."""
    if espn is None or players.empty:
        return {}
    by_key = {}
    for r in espn.itertuples(index=False):
        by_key.setdefault((r.key, r.pos), []).append(r)
    out = {}
    for p in players.itertuples(index=False):
        cands = by_key.get((norm_name(p.name), p.pos))
        if not cands:
            continue
        best = next((c for c in cands if c.tm == p.tm), cands[0])
        own = None if best.own is None or pd.isna(best.own) else float(best.own)
        out[p.id] = {"own": own, "status": clean(best.status)}
    return out


def resolve_status(inj, inj_current, espn_map):
    """One current status per player id: this week's NFL report first, ESPN fills gaps and adds IR."""
    st = {}
    if inj_current and not inj.empty:
        for r in inj.itertuples(index=False):
            if clean(r.id) and clean(r.status):
                st[r.id] = clean(r.status)
    for pid, e in espn_map.items():
        s = e.get("status")
        if s and (pid not in st or s in ("IR", "Suspended", "PUP")):
            st[pid] = s
    return st


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


def evaluate_players(cur, prev, tg, next_week, status, charts):
    """Form, usage, matchup and next-week projection for every player, in every scoring format."""
    hist = cur[cur["wk"] < next_week] if next_week else cur
    if hist.empty:
        return {f: {} for f in FORMATS}
    upcoming = tg[tg["week"] == next_week].set_index("team") if next_week else pd.DataFrame()
    imps = upcoming["imp"].dropna() if not upcoming.empty else pd.Series(dtype=float)
    league_imp = imps.mean() if len(imps) else None
    team_played_last = set(tg[tg["week"] == (next_week or 99) - 1]["team"])
    names = cur.groupby("id")["name"].last()

    prev_avgs = {}
    if not prev.empty:
        counts = prev.groupby("id").size()
        for fmt in FORMATS:
            prev_avgs[fmt] = prev.groupby("id")[fmt].mean()[counts >= 4]

    per_game = hist.groupby("id")[FORMATS].mean()
    gcounts = hist.groupby("id").size()
    pos_of = hist.groupby("id")["pos"].last()
    floors = {}
    for fmt in FORMATS:
        for pos in POSITIONS:
            ids = gcounts[(gcounts >= 2) & (pos_of == pos)].index
            floors[(fmt, pos)] = float(per_game.loc[ids, fmt].quantile(0.30)) if len(ids) else 0.0
    dfacs = {fmt: defense_factors(hist, fmt) for fmt in FORMATS}

    out = {f: {} for f in FORMATS}
    for pid, games in hist.sort_values("wk").groupby("id"):
        last = games.iloc[-1]
        pos, team, n = last["pos"], last["tm"], len(games)
        opps = ((games["att"] if pos == "QB" else games["tgt"]) + games["car"]).to_numpy()
        use = 1.0
        if n >= 4 and opps.mean() > 0:
            use = float(np.clip((opps[-3:].mean() / opps.mean()) ** USAGE_EXPONENT, *USAGE_CLIP))

        chart = charts.get(team, {}).get(pos, [])
        depth = chart.index(pid) + 1 if pid in chart else None
        mate_out = None
        if depth:
            for other in chart[:depth - 1]:
                if status.get(other) in OUT_STATUSES:
                    mate_out = names.get(other)
                    break
        boost = TEAMMATE_OUT_BOOST if mate_out else 1.0

        g = upcoming.loc[team] if (not upcoming.empty and team in upcoming.index) else None
        if isinstance(g, pd.DataFrame):
            g = g.iloc[0]
        env = 1.0
        if g is not None and league_imp and not pd.isna(g["imp"]):
            env = float(np.clip((g["imp"] / league_imp) ** ENV_EXPONENT, *ENV_CLIP))

        for fmt in FORMATS:
            vals = games[fmt].to_numpy()
            weights = DECAY ** np.arange(n - 1, -1, -1)
            wavg = float(np.dot(vals, weights) / weights.sum())
            prev_avg = prev_avgs.get(fmt, pd.Series(dtype=float)).get(pid)
            if prev_avg is not None and not np.isnan(prev_avg):
                base = (n * wavg + PRIOR_GAMES * prev_avg) / (n + PRIOR_GAMES)
            else:
                base = (n * wavg + floors[(fmt, pos)]) / (n + 1)
            e = {
                "id": pid, "name": last["name"], "pos": pos, "tm": team, "gp": n, "last_wk": int(last["wk"]),
                "base": round(base, 1), "last3": round(float(vals[-3:].mean()), 1), "use": round(use, 3),
                "depth": depth, "mate_out": mate_out, "boost": boost, "status": status.get(pid),
                "missed": bool(next_week and team in team_played_last and last["wk"] < next_week - 1),
                "proj": None, "opp": None, "home": None, "imp": None, "env": round(env, 3),
                "mu": 1.0, "allow": 1.0, "dr": None, "dteams": None, "played": False, "actual": None, "day": None,
            }
            if g is not None:
                mu, drank, dteams, draw = dfacs[fmt].get((g["opp"], pos), (1.0, None, None, 1.0))
                e.update({
                    "opp": g["opp"], "home": bool(g["home"]), "day": g["gameday"],
                    "imp": None if pd.isna(g["imp"]) else round(float(g["imp"]), 1),
                    "mu": round(float(mu), 3), "allow": round(float(draw), 3),
                    "dr": int(drank) if drank is not None else None,
                    "dteams": int(dteams) if dteams is not None else None,
                    "proj": round(base * mu * env * use * boost, 1), "played": bool(g["final"]),
                })
            out[fmt][pid] = e
    # Fill in actual points for games already played this week (e.g. Thursday night)
    if next_week:
        actual = cur[cur["wk"] == next_week].groupby("id")[FORMATS].sum()
        for fmt in FORMATS:
            for pid, e in out[fmt].items():
                if e["played"] and pid in actual.index:
                    e["actual"] = round(float(actual.loc[pid, fmt]), 1)
    return out


PROJ_KEYS = ["id", "name", "tm", "opp", "home", "day", "proj", "base", "mu", "allow", "dr", "dteams", "imp",
             "env", "use", "gp", "missed", "played", "actual", "status", "mate_out"]


def build_projections(evals, next_week, has_games):
    if next_week is None:
        return {"week": None, "status": "season_over", "players": {}}
    if not has_games:
        return {"week": next_week, "status": "no_games_yet", "players": {}}
    result = {}
    for pos in POSITIONS:
        result[pos] = {}
        for fmt in FORMATS:
            cand = [e for e in evals[fmt].values()
                    if e["pos"] == pos and e["proj"] is not None
                    and e["last_wk"] >= next_week - 3 and e["status"] not in OUT_STATUSES]
            cand.sort(key=lambda e: e["proj"], reverse=True)
            result[pos][fmt] = [{k: e[k] for k in PROJ_KEYS} for e in cand[:TOP_N]]
    return {"week": next_week, "status": "ok", "players": result}


# ----------------------------------------------------------------------------
# Waiver wire
# ----------------------------------------------------------------------------
def build_waivers(evals, cur, prev, espn_map, last_done, next_week):
    if not evals["half"]:
        return {"status": "no_games_yet", "sizes": LEAGUE_SIZES}
    ids = list(evals["half"].keys())
    own = {pid: espn_map.get(pid, {}).get("own") for pid in ids}
    use_espn = sum(v is not None for v in own.values()) >= 0.5 * len(ids)

    if use_espn:
        roster_score = {pid: (own[pid] or 0.0) for pid in ids}
    else:
        # Estimate: last season's points per game plus this season's, leaving out the latest week
        # (players who just broke out usually haven't been picked up yet).
        before = cur[cur["wk"] < last_done] if last_done else cur.iloc[0:0]
        b_ppg, b_n = before.groupby("id")["half"].mean(), before.groupby("id").size()
        p_ppg = prev.groupby("id")["half"].mean()[prev.groupby("id").size() >= 4] if not prev.empty else pd.Series(dtype=float)
        roster_score = {}
        for pid in ids:
            n = float(b_n.get(pid, 0))
            roster_score[pid] = (n * float(b_ppg.get(pid, 0.0)) + 3 * float(p_ppg.get(pid, 0.0))) / (n + 3)

    recent_cut = (next_week or last_done + 1) - 3
    sizes = {}
    for size in LEAGUE_SIZES:
        sizes[str(size)] = {}
        rostered = set()
        for pos in POSITIONS:
            pos_ids = [pid for pid in ids if evals["half"][pid]["pos"] == pos]
            pos_ids.sort(key=lambda pid: roster_score[pid], reverse=True)
            rostered.update(pos_ids[:int(np.ceil(size * ROSTER_PER_TEAM[pos]))])
        for fmt in FORMATS:
            sizes[str(size)][fmt] = {}
            # Points of the last starter at each position in this league size ("replacement level")
            repl = {}
            for pos in POSITIONS:
                bases = sorted((evals[fmt][pid]["base"] for pid in ids if evals[fmt][pid]["pos"] == pos), reverse=True)
                cut = int(np.ceil(size * STARTERS_PER_TEAM[pos])) - 1
                repl[pos] = bases[min(cut, len(bases) - 1)] if bases else 0.0
            for pos in POSITIONS:
                rows = []
                for pid in ids:
                    e = evals[fmt][pid]
                    if e["pos"] != pos or pid in rostered or e["status"] in OUT_STATUSES or e["last_wk"] < recent_cut:
                        continue
                    nxt = e["proj"] / e["boost"] if e["proj"] is not None else 0.6 * e["base"]
                    score = (0.45 * nxt + 0.35 * e["last3"] + 0.20 * e["base"]) * e["boost"]
                    rows.append({
                        "id": pid, "name": e["name"], "pos": pos, "tm": e["tm"], "score": round(score, 1),
                        "vor": round(score - repl[pos], 1),
                        "proj": e["proj"], "opp": e["opp"], "home": e["home"], "bye": e["opp"] is None,
                        "last3": e["last3"], "base": e["base"], "use": e["use"], "gp": e["gp"],
                        "depth": e["depth"], "mate_out": e["mate_out"], "status": e["status"],
                        "own": None if own[pid] is None else round(own[pid], 1),
                    })
                rows.sort(key=lambda r: r["score"], reverse=True)
                sizes[str(size)][fmt][pos] = rows[:WAIVER_PER_POS]
    return {"status": "ok", "source": "espn" if use_espn else "estimate", "sizes": sizes,
            "roster_per_team": ROSTER_PER_TEAM}


# ----------------------------------------------------------------------------
# Assemble data.json / history.json
# ----------------------------------------------------------------------------
def encode_rows(df, pindex):
    rows = []
    for r in df.itertuples(index=False):
        home = None if pd.isna(r.home) else int(bool(r.home))
        rows.append([int(r.s), pindex[r.id], int(r.wk), r.tm, r.opp, home,
                     round(float(r.std), 1), round(float(r.half), 2), round(float(r.ppr), 1),
                     int(r.pyd), int(r.ptd), int(r.int), int(r.car), int(r.ryd), int(r.rtd),
                     int(r.tgt), int(r.rec), int(r.reyd), int(r.retd)])
    return rows


def season_frame(stats_raw, sched_raw, season):
    df = normalize_stats(stats_raw, season)
    return attach_schedule(df, team_games(normalize_schedule(sched_raw, season)))


def build_history(frames):
    """Compact all-seasons file: each week's top scorers, plus every player's season totals."""
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True).sort_values(["s", "wk"])
    keep = set()
    for fmt in FORMATS:
        keep.update(df.sort_values(fmt, ascending=False).groupby(["s", "wk", "pos"]).head(HIST_PER_POS).index)
    weekly = df.loc[sorted(keep)].sort_values(["s", "wk"])
    g = df.groupby(["s", "id"])
    agg = g.agg(tm=("tm", "last"), g=("wk", "count"),
                std=("std", "sum"), half=("half", "sum"), ppr=("ppr", "sum"),
                bstd=("std", "max"), bhalf=("half", "max"), bppr=("ppr", "max")).reset_index()
    players = df.groupby("id").last()[["name", "pos"]].reset_index()
    pindex = {pid: i for i, pid in enumerate(players["id"])}
    season_rows = [[int(r.s), pindex[r.id], r.tm, int(r.g), round(float(r.std), 1), round(float(r.half), 1),
                    round(float(r.ppr), 1), round(float(r.bstd), 1), round(float(r.bhalf), 1), round(float(r.bppr), 1)]
                   for r in agg.itertuples(index=False)]
    return {
        "seasons": sorted(df["s"].unique().tolist()),
        "players": players[["id", "name", "pos"]].values.tolist(),
        "cols": ROW_COLS,
        "rows": encode_rows(weekly, pindex),
        "season_cols": ["s", "p", "tm", "g", "std", "half", "ppr", "bstd", "bhalf", "bppr"],
        "season_rows": season_rows,
    }


# ----------------------------------------------------------------------------
# Assemble data.json
# ----------------------------------------------------------------------------
def build(cur_raw, prev_raw, sched_raw, season, inj_raw=None, espn=None, sample=False):
    cur = normalize_stats(cur_raw, season)
    prev = normalize_stats(prev_raw, season - 1)
    sched = normalize_schedule(sched_raw, season)
    prev_sched = normalize_schedule(sched_raw, season - 1)
    tg = team_games(sched)
    cur = attach_schedule(cur, tg)
    prev = attach_schedule(prev, team_games(prev_sched))
    last_done, next_week = week_status(sched)

    both = pd.concat([d for d in (prev, cur) if not d.empty], ignore_index=True)
    players = both.sort_values(["s", "wk"]).groupby("id").last()[["name", "pos", "tm"]].reset_index()
    pindex = {pid: i for i, pid in enumerate(players["id"])}
    cur_players = cur.sort_values("wk").groupby("id").last()[["name", "pos", "tm"]].reset_index() if not cur.empty else players.iloc[0:0]

    inj, inj_week = normalize_injuries(inj_raw, season)
    inj_current = inj_week is not None and next_week is not None and inj_week == next_week
    espn_map = match_espn(espn, cur_players)
    status = resolve_status(inj, inj_current, espn_map)
    charts = depth_charts(cur)

    evals = evaluate_players(cur, prev, tg, next_week, status, charts) if not cur.empty else {f: {} for f in FORMATS}
    projections = build_projections(evals, next_week, not cur[cur["wk"] < (next_week or 99)].empty if not cur.empty else False)
    waivers = build_waivers(evals, cur, prev, espn_map, last_done, next_week) if not cur.empty else \
        {"status": "no_games_yet", "sizes": LEAGUE_SIZES}

    rows = encode_rows(both, pindex)

    # Team scores for the team offense tab
    team_scores = []
    for s_, sch in ((season - 1, prev_sched), (season, sched)):
        if sch.empty:
            continue
        for g in sch[sch["final"]].itertuples(index=False):
            team_scores.append([s_, int(g.week), g.home_team, g.away_team, int(g.home_score), int(g.away_score)])
            team_scores.append([s_, int(g.week), g.away_team, g.home_team, int(g.away_score), int(g.home_score)])

    upcoming = []
    if next_week:
        for g in tg[tg["week"] == next_week].itertuples(index=False):
            upcoming.append({"tm": g.team, "opp": g.opp, "home": bool(g.home), "final": bool(g.final),
                             "imp": None if pd.isna(g.imp) else round(float(g.imp), 1), "day": g.gameday})

    # Injury list: this week's NFL report (all positions) plus ESPN statuses it doesn't cover
    injuries = []
    listed = set()
    for r in inj.itertuples(index=False) if not inj.empty else []:
        r = r._replace(**{k: clean(v) for k, v in r._asdict().items()})
        status_now = r.status if inj_current else None
        if r.id in status and r.id in espn_map and status[r.id] in ("IR", "Suspended", "PUP"):
            status_now = status[r.id]
        injuries.append({"p": pindex.get(r.id), "name": r.name, "tm": r.tm, "pos": r.pos,
                         "status": status_now, "report": r.status, "injury": r.injury, "practice": r.practice})
        listed.add(r.id)
    if espn_map:
        cp = cur_players.set_index("id")
        for pid, st in status.items():
            if pid in listed or pid not in cp.index:
                continue
            injuries.append({"p": pindex.get(pid), "name": cp.loc[pid, "name"], "tm": cp.loc[pid, "tm"],
                             "pos": cp.loc[pid, "pos"], "status": st, "report": None, "injury": None, "practice": None})

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
            "espn": bool(espn_map),
            "sample": sample,
        },
        "players": players[["id", "name", "pos"]].values.tolist(),
        "cols": ROW_COLS,
        "rows": rows,
        "team_scores": team_scores,
        "upcoming": upcoming,
        "depth": {tm: {pos: [pindex[i] for i in ids if i in pindex] for pos, ids in d.items()} for tm, d in charts.items()},
        "injuries": {"week": inj_week, "current": inj_current, "list": injuries},
        "projections": projections,
        "waivers": waivers,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--out", default="site/data.json")
    ap.add_argument("--no-espn", action="store_true", help="skip ESPN and estimate roster % instead")
    args = ap.parse_args()
    season = args.season or current_season()

    print(f"Building data for the {season} season")
    sched_raw = load_first(SCHEDULE_URLS, "schedule")
    cur_raw = load_stats(season)
    prev_raw = load_stats(season - 1)
    inj_raw = load_injuries(season)
    espn = None if args.no_espn else load_espn(season)
    if sched_raw is None:
        sys.exit("Stopped: the schedule could not be downloaded.")
    if cur_raw is None and prev_raw is None:
        sys.exit("Stopped: no player stats could be downloaded.")

    data = build(cur_raw, prev_raw, sched_raw, season, inj_raw=inj_raw, espn=espn)
    if not data["rows"]:
        sys.exit("Stopped: stats downloaded but no QB/RB/WR/TE rows were found.")

    dump = lambda obj, path: json.dump(obj, open(path, "w"), separators=(",", ":"), allow_nan=False,
                                       default=lambda o: o.item() if hasattr(o, "item") else str(o))
    dump(data, args.out)

    # History for the team of the week / season tabs (older seasons are only downloaded here)
    frames = [season_frame(cur_raw, sched_raw, season), season_frame(prev_raw, sched_raw, season - 1)]
    for s_ in range(HISTORY_FROM, season - 1):
        raw = load_stats(s_)
        if raw is not None:
            frames.append(season_frame(raw, sched_raw, s_))
    hist_path = os.path.join(os.path.dirname(args.out) or ".", "history.json")
    hist = build_history(frames)
    dump(hist, hist_path)
    print(f"Wrote {hist_path}: seasons {hist['seasons'][0]}-{hist['seasons'][-1]}")
    m = data["meta"]
    print(f"Wrote {args.out}: {len(data['rows'])} player-games, stats through week {m['latest_stats_week']}, "
          f"projections for week {m['next_week']}, {len(data['injuries']['list'])} injury entries, "
          f"roster %: {'ESPN' if m['espn'] else 'estimated'}")


if __name__ == "__main__":
    main()
