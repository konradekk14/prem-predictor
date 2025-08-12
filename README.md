# Prem Predictor

A machine learning system for predicting Premier League football match outcomes using historical data and betting odds.

## Overview

Prem Predictor analyzes Premier League match data from the last 4 seasons to predict match outcomes (Home Win, Draw, Away Win). The system combines recent team performance metrics with betting market information to make predictions.

## Features

- **Recent Form Analysis**: Teams' performance over their last 5 matches
- **Betting Market Integration**: Incorporates bookmaker odds as predictive features
- **Temporal Awareness**: Only uses data available before the prediction date
- **Machine Learning**: Random Forest Classifier with balanced class weights

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd prem-predictor
```

2. Create and activate a virtual environment:

```bash
python -m venv prem-predictor-env
source prem-predictor-env/bin/activate  # On Windows: prem-predictor-env\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run the complete pipeline with prediction

```bash
cd src
python main.py --home "Arsenal" --away "Chelsea" --date "2024-12-25"
```

### Option 2: Run individual components

1. **Download data**:

```bash
cd src
python download_data.py
```

2. **Build features**:

```bash
cd src
python features.py
```

3. **Train model**:

```bash
cd src
python train_model.py
```

4. **Make prediction** (after training):

```bash
cd src
python predict.py "Arsenal" "Chelsea" "2024-12-25"
```

5. **Evaluate model accuracy** (comprehensive testing):

```bash
cd src
python evaluate_model.py
```

6. **Run backtesting** (historical performance analysis):

```bash
cd src
python backtest.py
```

## Data Sources

- **Match Data**: football-data.co.uk (E0.csv files)
- **Seasons**: 2021/22, 2022/23, 2023/24, 2024/25
- **Features**: Recent PPG, betting odds, match outcomes

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features**: 5 features (home_recent_ppg, away_recent_ppg, odds_home, odds_draw, odds_away)
- **Target**: 3 classes (Home Win=0, Draw=1, Away Win=2)
- **Training**: 80/20 train/test split with balanced class weights

## Project Structure

```
prem-predictor/
├── data/
│   ├── raw/          # Downloaded CSV files
│   └── processed/    # Feature-engineered data
├── models/           # Trained ML models
├── src/              # Source code
│   ├── main.py       # Main pipeline
│   ├── download_data.py
│   ├── features.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
└── requirements.txt
```

## Model Testing & Evaluation

The project includes comprehensive accuracy testing tools:

### 📊 `evaluate_model.py`

- **Cross-validation**: 5-fold validation for robust accuracy estimates
- **Temporal evaluation**: Realistic time-series split testing
- **Baseline comparisons**: Against random predictions and betting odds
- **Feature importance**: Understanding what drives predictions
- **Class-wise performance**: How well each outcome is predicted

### 🔄 `backtest.py`

- **Rolling window testing**: Simulates real-world prediction scenarios
- **Season-by-season analysis**: Performance across different time periods
- **Consistency metrics**: How stable the predictions are over time
- **Class-specific analysis**: Deep dive into each outcome type

## Output

The system provides:

- **Predicted Outcome**: Home Win, Draw, or Away Win
- **Probability Distribution**: Confidence scores for each outcome
- **Model Performance**: Comprehensive accuracy metrics and comparisons
- **Evaluation Reports**: Detailed analysis of prediction quality

## Notes

- The system automatically downloads data from the internet
- Predictions for future matches use neutral odds (33.3% each outcome)
- All data processing respects temporal boundaries (no data leakage)
