# src/train_model.py
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

def get_enhanced_features():
    """Get the list of enhanced features"""
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

def ensure_model_trained():
    """Ensure model is trained, only training if it doesn't exist"""
    model_file = MODELS_DIR / "rf_model.joblib"
    
    if model_file.exists():
        print("✓ Trained model already exists - loading from file")
        model = joblib.load(model_file)
        print("✓ Model loaded successfully")
        return model
    
    print("⚠ Trained model not found - training new model...")
    
    # load and prepare data
    df = pd.read_csv(PROCESSED_DIR / "features.csv")
    df = df.dropna(subset=["target"])
    cutoff = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cutoff], df.iloc[cutoff:]
    
    features = get_enhanced_features()
    X_train, y_train = train_df[features].fillna(0), train_df["target"]
    X_test, y_test = test_df[features].fillna(0), test_df["target"]
    
    # train model
    model = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)
    print("✓ Model training completed")
    
    # save model
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, model_file)
    print(f"✓ Model saved to {model_file}")
    
    # evaluate model
    evaluate_model_performance(model, X_test, y_test, features)
    
    return model

def evaluate_model_performance(model, X_test, y_test, features):
    """Evaluate and display model performance"""
    preds = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)
    
    print("=" * 50)
    print("TRAINING RESULTS")
    print("=" * 50)
    print(f"Training samples: {len(X_test)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, preds):.3f}")
    
    print("\nConfusion Matrix:")
    print("Predicted ->")
    print("Actual ↓  Home  Draw  Away")
    cm = confusion_matrix(y_test, preds)
    class_names = ["Home", "Draw", "Away"]
    for i, class_name in enumerate(class_names):
        print(f"{class_name:6}", end="")
        for j in range(3):
            print(f"{cm[i,j]:6}", end="")
        print()
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, preds, target_names=class_names))
    
    print(f"\nFeature Importance:")
    for name, importance in zip(features, model.feature_importances_):
        print(f"{name}: {importance:.3f}")
    
    print(f"\n💡 TIP: Run 'python evaluate_model.py' for comprehensive accuracy testing!")
    print(f"💡 TIP: Run 'python backtest.py' for historical performance analysis!")
    
    return preds, pred_proba

if __name__ == "__main__":
    ensure_model_trained()
