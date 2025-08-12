import pandas as pd
from pathlib import Path
from utils import (
    recent_ppg, odds_to_probs, 
    recent_shots_avg, recent_shots_target_avg, recent_corners_avg,
    recent_fouls_avg, recent_cards_avg,
    calculate_odds_variance, calculate_average_odds
)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

def load_all_data():
    files = sorted(RAW_DIR.glob("E0_*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def build_features(df):
    feature_rows = []
    for idx, row in df.iterrows():
        date = row["Date"]
        home, away = row["HomeTeam"], row["AwayTeam"]
        
        # Historical data for feature calculation (only past matches)
        past_df = df.iloc[:idx]

        # Basic performance features
        home_recent_ppg = recent_ppg(past_df, home, date)
        away_recent_ppg = recent_ppg(past_df, away, date)
        
        # Enhanced shot statistics
        home_shots_avg = recent_shots_avg(past_df, home, date)
        away_shots_avg = recent_shots_avg(past_df, away, date)
        home_shots_target_avg = recent_shots_target_avg(past_df, home, date)
        away_shots_target_avg = recent_shots_target_avg(past_df, away, date)
        
        # Possession and control indicators
        home_corners_avg = recent_corners_avg(past_df, home, date)
        away_corners_avg = recent_corners_avg(past_df, away, date)
        home_fouls_avg = recent_fouls_avg(past_df, home, date)
        away_fouls_avg = recent_fouls_avg(past_df, away, date)
        
        # Disciplinary statistics
        home_cards_avg = recent_cards_avg(past_df, home, date)
        away_cards_avg = recent_cards_avg(past_df, away, date)

        # Enhanced odds features - use average across multiple bookmakers
        avg_home_odds, avg_draw_odds, avg_away_odds = calculate_average_odds(row)
        
        # Calculate odds variance as market confidence indicator
        odds_var_home, odds_var_draw, odds_var_away = calculate_odds_variance(row)
        
        # Convert average odds to probabilities
        if not pd.isna(avg_home_odds):
            probs = odds_to_probs(avg_home_odds, avg_draw_odds, avg_away_odds)
        else:
            probs = [0.33, 0.33, 0.33]

        feature_rows.append({
            # Basic features
            "home_recent_ppg": home_recent_ppg,
            "away_recent_ppg": away_recent_ppg,
            
            # Shot statistics
            "home_shots_avg": home_shots_avg,
            "away_shots_avg": away_shots_avg,
            "home_shots_target_avg": home_shots_target_avg,
            "away_shots_target_avg": away_shots_target_avg,
            
            # Possession indicators
            "home_corners_avg": home_corners_avg,
            "away_corners_avg": away_corners_avg,
            "home_fouls_avg": home_fouls_avg,
            "away_fouls_avg": away_fouls_avg,
            
            # Disciplinary
            "home_cards_avg": home_cards_avg,
            "away_cards_avg": away_cards_avg,
            
            # Enhanced odds features
            "odds_home": probs[0],
            "odds_draw": probs[1],
            "odds_away": probs[2],
            "odds_variance_home": odds_var_home,
            "odds_variance_draw": odds_var_draw,
            "odds_variance_away": odds_var_away,
            
            # Target
            "target": {"H":0, "D":1, "A":2}.get(row["FTR"], None)
        })

    feat_df = pd.DataFrame(feature_rows)
    return feat_df

def ensure_features_generated():
    """Ensure features are generated, only creating them if they don't exist"""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    features_file = PROCESSED_DIR / "features.csv"
    
    if features_file.exists():
        print("✓ Processed features already exist - loading from file")
        feat_df = pd.read_csv(features_file)
        print(f"✓ Loaded {len(feat_df)} feature rows")
        return feat_df
    else:
        print("⚠ Processed features not found - generating from raw data...")
        df = load_all_data()
        feat_df = build_features(df)
        feat_df.to_csv(features_file, index=False)
        print(f"✓ Generated and saved {len(feat_df)} feature rows")
        return feat_df

if __name__ == "__main__":
    ensure_features_generated()
