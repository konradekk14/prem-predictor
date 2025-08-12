import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from features import load_all_data
from utils import recent_ppg, odds_to_probs
import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def get_enhanced_features():
    """Get the complete enhanced feature set"""
    return [
        # basic performance
        "home_recent_ppg", "away_recent_ppg",
        
        # shot statistics
        "home_shots_avg", "away_shots_avg",
        "home_shots_target_avg", "away_shots_target_avg",
        
        # possession indicators
        "home_corners_avg", "away_corners_avg",
        "home_fouls_avg", "away_fouls_avg",
        
        # disciplinary
        "home_cards_avg", "away_cards_avg",

        # enhanced odds
        "odds_home", "odds_draw", "odds_away",
        "odds_variance_home", "odds_variance_draw", "odds_variance_away"
    ]

def rolling_prediction_backtest(df, window_size=500, prediction_horizon=50):
    """
    Perform rolling window backtesting to simulate real-world prediction accuracy
    
    Args:
        df: Full dataset
        window_size: Number of matches to train on
        prediction_horizon: Number of future matches to predict
    """
    features = get_enhanced_features()
    results = []
    
    print(f"Starting rolling backtest with {window_size} training matches, predicting {prediction_horizon} ahead...")
    
    for start_idx in range(window_size, len(df) - prediction_horizon, prediction_horizon):
        # training data: window_size matches before start_idx
        train_start = start_idx - window_size
        train_df = df.iloc[train_start:start_idx].dropna(subset=["target"])
        
        # test data: next prediction_horizon matches
        test_df = df.iloc[start_idx:start_idx + prediction_horizon].dropna(subset=["target"])
        
        if len(train_df) < 100 or len(test_df) < 10:  # minimum data requirements
            continue
            
        # prepare training and test data
        X_train = train_df[features].fillna(0)
        y_train = train_df["target"]
        
        X_test = test_df[features].fillna(0)
        y_test = test_df["target"]
        
        # train model
        model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        
        # make predictions
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        
        # calculate accuracy for this window
        accuracy = (predictions == y_test).mean()
        
        # betting odds baseline for this window
        odds_predictions = []
        for _, row in X_test.iterrows():
            odds_probs = [row['odds_home'], row['odds_draw'], row['odds_away']]
            odds_pred = np.argmax(odds_probs)
            odds_predictions.append(odds_pred)
        
        odds_accuracy = (np.array(odds_predictions) == y_test).mean()
        
        results.append({
            'period_start': start_idx,
            'period_end': start_idx + prediction_horizon,
            'accuracy': accuracy,
            'odds_baseline': odds_accuracy,
            'improvement': accuracy - odds_accuracy,
            'n_predictions': len(y_test),
            'train_size': len(train_df)
        })
        
        print(f"Period {len(results)}: Matches {start_idx}-{start_idx + prediction_horizon}, "
              f"Accuracy: {accuracy:.3f}, Odds Baseline: {odds_accuracy:.3f}")
    
    return pd.DataFrame(results)

def season_by_season_analysis(df):
    """Analyze performance season by season"""
    # assuming we have date information to split by seasons
    # this is a simplified version for now not using actual date ranges
    
    results = []
    total_matches = len(df)
    season_size = total_matches // 4  # roughly 4 seasons
    
    features = get_enhanced_features()
    
    for season in range(3):  # use first 3 seasons to predict 4th
        train_start = season * season_size
        train_end = (season + 1) * season_size
        test_start = 3 * season_size
        test_end = len(df)
        
        # training data from this season
        train_df = df.iloc[train_start:train_end].dropna(subset=["target"])
        # test on final season
        test_df = df.iloc[test_start:test_end].dropna(subset=["target"])
        
        if len(train_df) < 50 or len(test_df) < 50:
            continue
        
        X_train = train_df[features].fillna(0)
        y_train = train_df["target"]
        X_test = test_df[features].fillna(0)
        y_test = test_df["target"]
        
        # train model
        model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
        model.fit(X_train, y_train)
        
        # predictions
        predictions = model.predict(X_test)
        accuracy = (predictions == y_test).mean()
        
        results.append({
            'training_season': season + 1,
            'test_season': 4,
            'accuracy': accuracy,
            'train_matches': len(train_df),
            'test_matches': len(test_df)
        })
    
    return pd.DataFrame(results)

def class_specific_analysis(df):
    """Analyze how well the model predicts each outcome type"""
    features = get_enhanced_features()
    
    # use temporal split
    cutoff = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff].dropna(subset=["target"])
    test_df = df.iloc[cutoff:].dropna(subset=["target"])
    
    X_train = train_df[features].fillna(0)
    y_train = train_df["target"]
    X_test = test_df[features].fillna(0)
    y_test = test_df["target"]
    
    # train model
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    class_names = ["Home Win", "Draw", "Away Win"]
    results = []
    
    for class_idx, class_name in enumerate(class_names):
        # get subset where true class is class_idx
        class_mask = y_test == class_idx
        class_predictions = predictions[class_mask]
        class_true = y_test[class_mask]
        class_probs = probabilities[class_mask]
        
        if len(class_true) == 0:
            continue
            
        # accuracy for this class
        class_accuracy = (class_predictions == class_true).mean()
        
        # average confidence when predicting this class correctly
        correct_mask = class_predictions == class_true
        avg_confidence = class_probs[correct_mask, class_idx].mean() if correct_mask.sum() > 0 else 0
        
        # count of predictions
        n_actual = len(class_true)
        n_predicted_as = (predictions == class_idx).sum()
        
        results.append({
            'outcome': class_name,
            'recall': class_accuracy,  # how often we correctly identify this class
            'n_actual': n_actual,
            'n_predicted_as': n_predicted_as,
            'avg_confidence': avg_confidence
        })
    
    return pd.DataFrame(results)

def print_backtest_report(rolling_results, season_results, class_results):
    """Print comprehensive backtest report"""
    print("=" * 80)
    print("BACKTESTING REPORT")
    print("=" * 80)
    
    print(f"\nROLLING WINDOW BACKTEST")
    print(f"Number of test periods: {len(rolling_results)}")
    print(f"Average accuracy: {rolling_results['accuracy'].mean():.3f} ± {rolling_results['accuracy'].std():.3f}")
    print(f"Average odds baseline: {rolling_results['odds_baseline'].mean():.3f} ± {rolling_results['odds_baseline'].std():.3f}")
    print(f"Average improvement: {rolling_results['improvement'].mean():.3f}")
    print(f"Periods with positive improvement: {(rolling_results['improvement'] > 0).sum()}/{len(rolling_results)}")
    
    best_period = rolling_results.loc[rolling_results['accuracy'].idxmax()]
    worst_period = rolling_results.loc[rolling_results['accuracy'].idxmin()]
    print(f"Best period: {best_period['accuracy']:.3f} (matches {best_period['period_start']}-{best_period['period_end']})")
    print(f"Worst period: {worst_period['accuracy']:.3f} (matches {worst_period['period_start']}-{worst_period['period_end']})")
    
    if len(season_results) > 0:
        print(f"\nSEASON-BY-SEASON ANALYSIS")
        for _, row in season_results.iterrows():
            print(f"Season {row['training_season']} → Season {row['test_season']}: "
                  f"{row['accuracy']:.3f} accuracy ({row['test_matches']} test matches)")
    
    print(f"\n🎯 CLASS-SPECIFIC PERFORMANCE")
    for _, row in class_results.iterrows():
        print(f"{row['outcome']}: {row['recall']:.3f} recall, "
              f"{row['n_actual']} actual, {row['n_predicted_as']} predicted, "
              f"{row['avg_confidence']:.3f} avg confidence")
    
    print(f"\nCONSISTENCY METRICS")
    accuracy_consistency = 1 - rolling_results['accuracy'].std()
    print(f"Accuracy consistency: {accuracy_consistency:.3f} (1.0 = perfectly consistent)")
    
    improvement_consistency = (rolling_results['improvement'] > 0).mean()
    print(f"Improvement consistency: {improvement_consistency:.3f} (fraction of periods beating odds)")

def run_backtest():
    """Run comprehensive backtesting analysis"""
    print("Loading data for backtesting...")
    
    # Load data from features.py
    from features import load_all_data, build_features
    raw_df = load_all_data()
    df = build_features(raw_df)
    df = df.dropna(subset=["target"])
    
    print(f"Loaded {len(df)} matches for backtesting")
    
    print("\nRunning rolling window backtest...")
    rolling_results = rolling_prediction_backtest(df, window_size=400, prediction_horizon=50)
    
    print("\nRunning season-by-season analysis...")
    season_results = season_by_season_analysis(df)
    
    print("\nRunning class-specific analysis...")
    class_results = class_specific_analysis(df)
    
    print_backtest_report(rolling_results, season_results, class_results)
    
    return {
        'rolling_results': rolling_results,
        'season_results': season_results,
        'class_results': class_results
    }

if __name__ == "__main__":
    results = run_backtest()
