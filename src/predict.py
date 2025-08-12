import argparse
import pandas as pd
import joblib
from pathlib import Path
from utils import (
    recent_ppg, recent_shots_avg, recent_shots_target_avg,
    recent_corners_avg, recent_fouls_avg, recent_cards_avg
)
from features import load_all_data
from train_model import get_enhanced_features

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def predict_match(home, away, date):
    """Make a prediction for a specific match"""
    fixture_date = pd.to_datetime(date)
    
    # load model and data
    model = joblib.load(MODELS_DIR / "rf_model.joblib")
    df = load_all_data()
    
    # calculate all features for prediction
    home_recent_ppg = recent_ppg(df, home, fixture_date)
    away_recent_ppg = recent_ppg(df, away, fixture_date)
    
    # shot statistics
    home_shots_avg = recent_shots_avg(df, home, fixture_date)
    away_shots_avg = recent_shots_avg(df, away, fixture_date)
    home_shots_target_avg = recent_shots_target_avg(df, home, fixture_date)
    away_shots_target_avg = recent_shots_target_avg(df, away, fixture_date)
    
    # possession indicators
    home_corners_avg = recent_corners_avg(df, home, fixture_date)
    away_corners_avg = recent_corners_avg(df, away, fixture_date)
    home_fouls_avg = recent_fouls_avg(df, home, fixture_date)
    away_fouls_avg = recent_fouls_avg(df, away, fixture_date)
    
    # disciplinary
    home_cards_avg = recent_cards_avg(df, home, fixture_date)
    away_cards_avg = recent_cards_avg(df, away, fixture_date)
    
    # default odds for future matches
    probs = [0.33, 0.33, 0.33]

    # get features list from train_model to ensure consistency
    features = get_enhanced_features()
    
    # create prediction DataFrame with all features using the same logic as main.py
    pred_data = {}
    for feature in features:
        if "home_recent_ppg" in feature:
            pred_data[feature] = home_recent_ppg
        elif "away_recent_ppg" in feature:
            pred_data[feature] = away_recent_ppg
        elif "home_shots_avg" in feature:
            pred_data[feature] = home_shots_avg
        elif "away_shots_avg" in feature:
            pred_data[feature] = away_shots_avg
        elif "home_shots_target_avg" in feature:
            pred_data[feature] = home_shots_target_avg
        elif "away_shots_target_avg" in feature:
            pred_data[feature] = away_shots_target_avg
        elif "home_corners_avg" in feature:
            pred_data[feature] = home_corners_avg
        elif "away_corners_avg" in feature:
            pred_data[feature] = away_corners_avg
        elif "home_fouls_avg" in feature:
            pred_data[feature] = home_fouls_avg
        elif "away_fouls_avg" in feature:
            pred_data[feature] = away_fouls_avg
        elif "home_cards_avg" in feature:
            pred_data[feature] = home_cards_avg
        elif "away_cards_avg" in feature:
            pred_data[feature] = away_cards_avg
        elif "odds_home" in feature:
            pred_data[feature] = probs[0]
        elif "odds_draw" in feature:
            pred_data[feature] = probs[1]
        elif "odds_away" in feature:
            pred_data[feature] = probs[2]
        elif "odds_variance" in feature:
            pred_data[feature] = 0.0

    X_pred = pd.DataFrame([pred_data]).fillna(0)

    # make prediction
    pred = model.predict(X_pred)[0]
    pred_proba = model.predict_proba(X_pred)[0]
    outcome_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}

    # display results with enhanced formatting
    print(f"\n" + "=" * 50)
    print("MATCH PREDICTION")
    print("=" * 50)
    print(f"\nPrediction for {home} vs {away} on {date}: {outcome_map[pred]}")
    print(f"Probabilities: Home={pred_proba[0]:.2f}, Draw={pred_proba[1]:.2f}, Away={pred_proba[2]:.2f}")
    
    # display some features for context
    print(f"\nTeam Statistics (last 5 matches):")
    print(f"{home} - PPG: {home_recent_ppg:.2f}, Shots: {home_shots_avg:.1f}, Shots on Target: {home_shots_target_avg:.1f}")
    print(f"{away} - PPG: {away_recent_ppg:.2f}, Shots: {away_shots_avg:.1f}, Shots on Target: {away_shots_target_avg:.1f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Premier League match outcomes")
    parser.add_argument("--home", type=str, required=True, help="Home team name")
    parser.add_argument("--away", type=str, required=True, help="Away team name") 
    parser.add_argument("--date", type=str, required=True, help="Fixture date YYYY-MM-DD")
    args = parser.parse_args()

    predict_match(args.home, args.away, args.date)
