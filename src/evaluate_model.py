import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    precision_recall_fscore_support,
    log_loss
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# load the trained model and processed data
def load_model_and_data():
    model = joblib.load(MODELS_DIR / "rf_model.joblib")
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    df = df.dropna(subset=["target"])
    return model, df

# calculate baseline accuracies for comparison
def baseline_accuracy(y_test):
    # most frequent class baseline
    most_frequent = pd.Series(y_test).mode()[0]
    most_frequent_acc = (y_test == most_frequent).mean()
    
    # random baseline (expected accuracy for random guessing)
    random_acc = 1.0 / len(np.unique(y_test))
    
    return {
        "most_frequent_baseline": most_frequent_acc,
        "random_baseline": random_acc
    }

def detailed_evaluation(model, X_test, y_test):
    """Perform detailed model evaluation"""
    # predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
    
    # log loss (probability-based metric)
    logloss = log_loss(y_test, y_pred_proba)
    
    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "log_loss": logloss,
        "class_precision": precision,
        "class_recall": recall,
        "class_f1": f1,
        "predictions": y_pred,
        "probabilities": y_pred_proba
    }

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

def cross_validation_scores(df, n_folds=5):
    """Perform cross-validation to get more robust accuracy estimates"""
    features = get_enhanced_features()
    X = df[features].fillna(0)
    y = df["target"]
    
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    
    # stratified K-Fold to maintain class distribution
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    return {
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cv_scores": cv_scores
    }

def feature_importance_analysis(model):
    """Analyze feature importance to understand what drives predictions"""
    features = get_enhanced_features()
    importance = model.feature_importances_
    
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return feature_importance

def betting_odds_baseline(X_test, y_test):
    """Test how well betting odds alone predict outcomes"""
    # use odds as direct probabilities for prediction
    odds_predictions = []
    for _, row in X_test.iterrows():
        odds_probs = [row['odds_home'], row['odds_draw'], row['odds_away']]
        prediction = np.argmax(odds_probs)
        odds_predictions.append(prediction)
    
    odds_accuracy = accuracy_score(y_test, odds_predictions)
    return odds_accuracy

def temporal_split_evaluation(df, test_fraction=0.2):
    """Evaluate model using temporal split (more realistic for time series)"""
    features = get_enhanced_features()
    
    # sort by index (which should be chronological)
    df_sorted = df.sort_index()
    
    # temporal split
    cutoff = int(len(df_sorted) * (1 - test_fraction))
    train_df = df_sorted.iloc[:cutoff]
    test_df = df_sorted.iloc[cutoff:]
    
    X_train, y_train = train_df[features].fillna(0), train_df["target"]
    X_test, y_test = test_df[features].fillna(0), test_df["target"]
    
    # train model
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    
    # evaluate
    evaluation = detailed_evaluation(model, X_test, y_test)
    baseline = baseline_accuracy(y_test)
    odds_baseline = betting_odds_baseline(X_test, y_test)
    
    return {
        "temporal_evaluation": evaluation,
        "temporal_baseline": baseline,
        "temporal_odds_baseline": odds_baseline
    }

def print_evaluation_report(results):
    """Print comprehensive evaluation report"""
    print("=" * 80)
    print("ENHANCED MODEL EVALUATION REPORT")
    print("=" * 80)
    
    # enhanced vs Original Comparison
    if 'comparison' in results:
        comp = results['comparison']
        print(f"\nENHANCED FEATURES IMPACT")
        print(f"Original Model ({comp['old_features']} features): {comp['old_accuracy']:.3f}")
        print(f"Enhanced Model ({comp['new_features']} features): {comp['new_accuracy']:.3f}")
        print(f"Improvement: +{comp['improvement']:.3f} ({comp['improvement_percentage']:+.1f}%)")
        
        if comp['improvement'] > 0:
            print(f"Enhanced features provide {comp['improvement_percentage']:.1f}% better accuracy!")
        else:
            print(f"Enhanced features show {comp['improvement_percentage']:.1f}% change (may need tuning)")
    
    
    # cross-validation results
    print(f"\nCROSS-VALIDATION RESULTS (5-fold)")
    print(f"Mean Accuracy: {results['cv_results']['cv_mean']:.3f} ± {results['cv_results']['cv_std']:.3f}")
    print(f"Individual fold scores: {[f'{score:.3f}' for score in results['cv_results']['cv_scores']]}")
    
    # temporal split results
    print(f"\nTEMPORAL SPLIT EVALUATION")
    temporal = results['temporal_results']['temporal_evaluation']
    print(f"Accuracy: {temporal['accuracy']:.3f}")
    print(f"Macro Precision: {temporal['macro_precision']:.3f}")
    print(f"Macro Recall: {temporal['macro_recall']:.3f}")
    print(f"Macro F1: {temporal['macro_f1']:.3f}")
    print(f"Log Loss: {temporal['log_loss']:.3f}")
    
    # baseline comparisons
    print(f"\nBASELINE COMPARISONS")
    temporal_baseline = results['temporal_results']['temporal_baseline']
    print(f"Random Baseline: {temporal_baseline['random_baseline']:.3f}")
    print(f"Most Frequent Class: {temporal_baseline['most_frequent_baseline']:.3f}")
    print(f"Betting Odds Baseline: {results['temporal_results']['temporal_odds_baseline']:.3f}")
    print(f"Our Model: {temporal['accuracy']:.3f}")
    
    # performance improvement
    improvement_over_random = temporal['accuracy'] - temporal_baseline['random_baseline']
    improvement_over_odds = temporal['accuracy'] - results['temporal_results']['temporal_odds_baseline']
    print(f"\n🚀 PERFORMANCE IMPROVEMENTS")
    print(f"Improvement over random: +{improvement_over_random:.3f} ({improvement_over_random/temporal_baseline['random_baseline']*100:.1f}%)")
    print(f"Improvement over betting odds: +{improvement_over_odds:.3f} ({improvement_over_odds/results['temporal_results']['temporal_odds_baseline']*100:.1f}%)")
    
    # class-wise performance
    print(f"\n🎯 CLASS-WISE PERFORMANCE")
    class_names = ["Home Win", "Draw", "Away Win"]
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: Precision={temporal['class_precision'][i]:.3f}, "
              f"Recall={temporal['class_recall'][i]:.3f}, F1={temporal['class_f1'][i]:.3f}")
    
    # feature importance - show top 10 for clarity
    print(f"\nTOP FEATURE IMPORTANCE")
    top_features = results['feature_importance'].head(10)
    for _, row in top_features.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")
    
    if len(results['feature_importance']) > 10:
        print(f"... and {len(results['feature_importance']) - 10} more features")
    
    # confusion matrix
    print(f"\nCONFUSION MATRIX")
    print("Predicted ->")
    print("Actual ↓  Home  Draw  Away")
    cm = confusion_matrix(results['temporal_results']['temporal_evaluation']['predictions'], 
                         results['test_y'])
    for i, class_name in enumerate(["Home", "Draw", "Away"]):
        print(f"{class_name:6}", end="")
        for j in range(3):
            print(f"{cm[i,j]:6}", end="")
        print()

def compare_old_vs_new_features(df):
    """Compare performance between old 5-feature model and new enhanced model"""
    print("Comparing old vs enhanced feature sets...")
    
    # old feature set
    old_features = ["home_recent_ppg", "away_recent_ppg", "odds_home", "odds_draw", "odds_away"]
    enhanced_features = get_enhanced_features()
    
    # temporal split
    cutoff = int(len(df) * 0.8)
    train_df = df.iloc[:cutoff]
    test_df = df.iloc[cutoff:]
    
    # evaluate old model
    X_train_old = train_df[old_features].fillna(0)
    X_test_old = test_df[old_features].fillna(0)
    y_train, y_test = train_df["target"], test_df["target"]
    
    old_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    old_model.fit(X_train_old, y_train)
    old_acc = accuracy_score(y_test, old_model.predict(X_test_old))
    
    # evaluate enhanced model
    X_train_new = train_df[enhanced_features].fillna(0)
    X_test_new = test_df[enhanced_features].fillna(0)
    
    new_model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    new_model.fit(X_train_new, y_train)
    new_acc = accuracy_score(y_test, new_model.predict(X_test_new))
    
    improvement = new_acc - old_acc
    improvement_pct = (improvement / old_acc) * 100
    
    return {
        "old_accuracy": old_acc,
        "new_accuracy": new_acc,
        "improvement": improvement,
        "improvement_percentage": improvement_pct,
        "old_features": len(old_features),
        "new_features": len(enhanced_features)
    }

def run_comprehensive_evaluation():
    """Run all evaluation tests"""
    print("Loading model and data...")
    model, df = load_model_and_data()
    
    print("Running enhanced vs baseline comparison...")
    comparison = compare_old_vs_new_features(df)
    
    print("Running cross-validation...")
    cv_results = cross_validation_scores(df)
    
    print("Running temporal split evaluation...")
    temporal_results = temporal_split_evaluation(df)
    
    print("Analyzing feature importance...")
    feature_importance = feature_importance_analysis(model)
    
    # prepare test data for confusion matrix
    cutoff = int(len(df) * 0.8)
    test_df = df.iloc[cutoff:]
    features = get_enhanced_features()
    X_test = test_df[features].fillna(0)
    y_test = test_df["target"]
    
    results = {
        'cv_results': cv_results,
        'temporal_results': temporal_results,
        'feature_importance': feature_importance,
        'comparison': comparison,
        'test_y': y_test
    }
    
    print_evaluation_report(results)
    
    return results

if __name__ == "__main__":
    results = run_comprehensive_evaluation()
