import pandas as pd
from glob import glob

files = sorted(glob('data/E0_*.csv'))
dfs = [pd.read_csv(f, dayfirst=True) for f in files]
df = pd.concat(dfs, ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df = df.sort_values('Date').reset_index(drop=True)

df['target'] = df['FTR'].map({'H':0, 'D':1, 'A':2})

import numpy as np

def compute_team_recent_ppg(matches_df, team, as_of_date, window=5):
    # matches_df contains all matches with Date, HomeTeam, AwayTeam, FTHG, FTAG
    past = matches_df[matches_df['Date'] < as_of_date]
    # games where team played
    is_home = past['HomeTeam'] == team
    is_away = past['AwayTeam'] == team
    team_games = past[is_home | is_away].copy().sort_values('Date', ascending=False).head(window)
    if team_games.empty:
        return np.nan
    pts = []
    for _, row in team_games.iterrows():
        if row['HomeTeam'] == team:
            if row['FTR']=='H': pts.append(3)
            elif row['FTR']=='D': pts.append(1)
            else: pts.append(0)
        else:
            if row['FTR']=='A': pts.append(3)
            elif row['FTR']=='D': pts.append(1)
            else: pts.append(0)
    return np.mean(pts)  # ppg in last `window` games
