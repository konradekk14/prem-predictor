# src/utils.py

# some utility functions for the project

import numpy as np
import pandas as pd

def compute_points(row, team):
    if row["HomeTeam"] == team:
        if row["FTR"] == "H": return 3
        elif row["FTR"] == "D": return 1
        else: return 0
    else:
        if row["FTR"] == "A": return 3
        elif row["FTR"] == "D": return 1
        else: return 0

def recent_ppg(df, team, date, window=5):
    past = df[df["Date"] < date]
    team_games = past[(past["HomeTeam"] == team) | (past["AwayTeam"] == team)]
    team_games = team_games.sort_values("Date", ascending=False).head(window)
    if team_games.empty:
        return np.nan
    points = [compute_points(row, team) for _, row in team_games.iterrows()]
    return np.mean(points)

def odds_to_probs(home_odds, draw_odds, away_odds):
    inv = [1/home_odds, 1/draw_odds, 1/away_odds]
    total = sum(inv)
    return [x/total for x in inv]

def recent_stat_avg(df, team, date, stat_col_home, stat_col_away, window=5):
    """Calculate average of a statistic for a team over recent matches"""
    past = df[df["Date"] < date]
    team_games = past[(past["HomeTeam"] == team) | (past["AwayTeam"] == team)]
    team_games = team_games.sort_values("Date", ascending=False).head(window)
    
    if team_games.empty:
        return np.nan
    
    stats = []
    for _, row in team_games.iterrows():
        if row["HomeTeam"] == team:
            stat_value = row.get(stat_col_home, np.nan)
        else:
            stat_value = row.get(stat_col_away, np.nan)
        
        if not pd.isna(stat_value):
            stats.append(stat_value)
    
    return np.mean(stats) if stats else np.nan

def recent_shots_avg(df, team, date, window=5):
    """Calculate average shots for a team over recent matches"""
    return recent_stat_avg(df, team, date, "HS", "AS", window)

def recent_shots_target_avg(df, team, date, window=5):
    """Calculate average shots on target for a team over recent matches"""
    return recent_stat_avg(df, team, date, "HST", "AST", window)

def recent_corners_avg(df, team, date, window=5):
    """Calculate average corners for a team over recent matches"""
    return recent_stat_avg(df, team, date, "HC", "AC", window)

def recent_fouls_avg(df, team, date, window=5):
    """Calculate average fouls for a team over recent matches"""
    return recent_stat_avg(df, team, date, "HF", "AF", window)

def recent_cards_avg(df, team, date, window=5):
    """Calculate average cards (yellow + red) for a team over recent matches"""
    past = df[df["Date"] < date]
    team_games = past[(past["HomeTeam"] == team) | (past["AwayTeam"] == team)]
    team_games = team_games.sort_values("Date", ascending=False).head(window)
    
    if team_games.empty:
        return np.nan
    
    cards = []
    for _, row in team_games.iterrows():
        if row["HomeTeam"] == team:
            home_cards = (row.get("HY", 0) or 0) + (row.get("HR", 0) or 0)
            cards.append(home_cards)
        else:
            away_cards = (row.get("AY", 0) or 0) + (row.get("AR", 0) or 0)
            cards.append(away_cards)
    
    return np.mean(cards) if cards else np.nan

def calculate_odds_variance(row, bookmakers=None):
    """Calculate variance across multiple bookmaker odds for home/draw/away"""
    if bookmakers is None:
        # Default bookmakers available in the data
        bookmakers = [
            ("B365H", "B365D", "B365A"),  # Bet365
            ("BWH", "BWD", "BWA"),        # Betway
            ("BFH", "BFD", "BFA"),        # Betfair
            ("PSH", "PSD", "PSA"),        # Paddy Power
            ("WHH", "WHD", "WHA"),        # William Hill
        ]
    
    home_odds, draw_odds, away_odds = [], [], []
    
    for h_col, d_col, a_col in bookmakers:
        if h_col in row and not pd.isna(row[h_col]):
            home_odds.append(row[h_col])
            draw_odds.append(row[d_col])
            away_odds.append(row[a_col])
    
    if len(home_odds) < 2:  # Need at least 2 bookmakers for variance
        return np.nan, np.nan, np.nan
    
    return np.var(home_odds), np.var(draw_odds), np.var(away_odds)

def calculate_average_odds(row, bookmakers=None):
    """Calculate average odds across multiple bookmakers"""
    if bookmakers is None:
        bookmakers = [
            ("B365H", "B365D", "B365A"),
            ("BWH", "BWD", "BWA"),
            ("BFH", "BFD", "BFA"),
            ("PSH", "PSD", "PSA"),
            ("WHH", "WHD", "WHA"),
        ]
    
    home_odds, draw_odds, away_odds = [], [], []
    
    for h_col, d_col, a_col in bookmakers:
        if h_col in row and not pd.isna(row[h_col]):
            home_odds.append(row[h_col])
            draw_odds.append(row[d_col])
            away_odds.append(row[a_col])
    
    if not home_odds:  # Fallback to B365 if no odds available
        if "B365H" in row and not pd.isna(row["B365H"]):
            return row["B365H"], row["B365D"], row["B365A"]
        else:
            return np.nan, np.nan, np.nan
    
    return np.mean(home_odds), np.mean(draw_odds), np.mean(away_odds)
