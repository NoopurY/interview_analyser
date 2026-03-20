"""
Communication Score Model Training Script
Trains a RandomForest regressor to predict communication score.

Usage:
    python train_model.py

Output:
    models/communication_model.pkl
    models/evaluation_report.txt
"""

import os
import pickle
import json
import numpy as np

# Try importing sklearn; guide user if missing
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not installed. Run: pip install scikit-learn")
    print("    The app will fall back to rule-based scoring without the ML model.")


def generate_synthetic_dataset(n_samples: int = 500, seed: int = 42):
    """
    Generate synthetic training data based on domain expertise rules.

    Features:
        filler_rate, filler_count, ttr, conf_ratio, star_score,
        prof_terms, has_numbers, mean_sentence_len,
        speaking_rate, pause_frequency, pitch_variation

    Target:
        communication_score (0–100)
    """
    rng = np.random.RandomState(seed)

    n = n_samples
    filler_rate = rng.beta(2, 20, n)                    # mostly low
    filler_count = (filler_rate * 200).astype(int)
    ttr = rng.beta(5, 3, n)                              # vocabulary richness
    conf_ratio = rng.beta(4, 3, n)                       # confidence
    star_score = rng.beta(3, 2, n)                       # STAR coverage
    prof_terms = rng.randint(0, 10, n)
    has_numbers = rng.binomial(1, 0.45, n)
    mean_sentence_len = rng.normal(14, 4, n).clip(5, 35)
    speaking_rate = rng.normal(140, 25, n).clip(80, 220)
    pause_frequency = rng.exponential(5, n).clip(0, 30)
    pitch_variation = rng.normal(18, 8, n).clip(2, 50)

    X = np.column_stack([
        filler_rate, filler_count, ttr, conf_ratio, star_score,
        prof_terms, has_numbers, mean_sentence_len,
        speaking_rate, pause_frequency, pitch_variation
    ])

    # Target: composite communication score (ground truth formula)
    score = (
        50
        - filler_rate * 150              # penalty for fillers
        + ttr * 20                       # reward vocabulary richness
        + (conf_ratio - 0.5) * 30        # confidence bonus/penalty
        + star_score * 20                # reward STAR structure
        + prof_terms * 1.5               # reward professional terms
        + has_numbers * 8                # reward quantified results
        - np.abs(speaking_rate - 140) * 0.1  # penalty for extreme pace
        - pause_frequency * 0.5          # penalty for many pauses
        + (pitch_variation - 10) * 0.2   # reward vocal variety
    )

    # Add noise
    score += rng.normal(0, 3, n)
    score = np.clip(score, 0, 100)

    return X, score


def train_and_save():
    if not SKLEARN_AVAILABLE:
        print("❌ Cannot train: scikit-learn not available.")
        return

    print("🔄 Generating synthetic training dataset...")
    X, y = generate_synthetic_dataset(n_samples=1000)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: StandardScaler + RandomForest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])

    print("🧠 Training RandomForest model...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="r2")

    print(f"\n📊 Evaluation Results:")
    print(f"   MAE  : {mae:.2f} points")
    print(f"   R²   : {r2:.4f}")
    print(f"   CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    os.makedirs("../models", exist_ok=True)
    model_path = "../models/communication_model.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print(f"\n✅ Model saved to: {model_path}")

    # Save evaluation report
    report = {
        "model": "RandomForestRegressor",
        "n_estimators": 200,
        "mae": round(float(mae), 4),
        "r2_score": round(float(r2), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "features": [
            "filler_rate", "filler_count", "ttr", "conf_ratio",
            "star_score", "prof_terms", "has_numbers", "mean_sentence_len",
            "speaking_rate", "pause_frequency", "pitch_variation"
        ],
        "training_samples": 800,
        "test_samples": 200
    }

    with open("../models/evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("📄 Evaluation report saved to: models/evaluation_report.json")
    return pipeline


if __name__ == "__main__":
    train_and_save()
