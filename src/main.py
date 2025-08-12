import argparse
from download_data import ensure_all_seasons_downloaded
from features import ensure_features_generated, RAW_DIR
from train_model import ensure_model_trained
from predict import predict_match

# the main pipeline for the football prediction model
def run_pipeline(home=None, away=None, date=None):
    """Set up the complete data pipeline: download data, process features, train model"""
    
    # 1. download data if it doesn't already exist
    ensure_all_seasons_downloaded(RAW_DIR)

    # 2. check if features need processing
    ensure_features_generated()

    # 3. check if a model exists or needs to be trained
    ensure_model_trained()
    
    # 4. make prediction if teams and date provided
    if home and away and date:
        predict_match(home, away, date)
    else:
        print(f"TIP: Run with --home --away --date to make a prediction!")
        print(f"Example: python main.py --home Arsenal --away Chelsea --date 2024-12-25")
    
    print(f"\n" + "=" * 50)
    print(f"\nRun 'python evaluate_model.py' for comprehensive accuracy testing!")
    print(f"Run 'python backtest.py' for historical performance analysis!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Premier League Prediction Pipeline")
    parser.add_argument("--home", type=str, help="Home team name")
    parser.add_argument("--away", type=str, help="Away team name") 
    parser.add_argument("--date", type=str, help="Fixture date YYYY-MM-DD")
    args = parser.parse_args()

    run_pipeline(home=args.home, away=args.away, date=args.date)
