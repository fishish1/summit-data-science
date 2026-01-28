import numpy as np
import pandas as pd
from typing import Tuple, Dict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import os

class IntervalPredictor:
    """
    Predicts ranges (confidence intervals) using Quantile Regression.
    """
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.q10_model = None
        self.q90_model = None

    def train_intervals(self, X: pd.DataFrame, y: np.array, numeric_features: list, categorical_features: list):
        """
        Trains 10th and 90th percentile models.
        """
        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])

        # Train 10th Percentile
        self.q10_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', HistGradientBoostingRegressor(loss='quantile', quantile=0.1, random_state=42, max_iter=200))
        ])
        
        # Train 90th Percentile
        self.q90_model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', HistGradientBoostingRegressor(loss='quantile', quantile=0.9, random_state=42, max_iter=200))
        ])

        print("Training Quantile Models (10th/90th)...")
        self.q10_model.fit(X, np.log1p(y))
        self.q90_model.fit(X, np.log1p(y))

    def predict_interval(self, X: pd.DataFrame) -> Tuple[np.array, np.array]:
        """
        Returns (lower_bound, upper_bound)
        """
        if self.q10_model is None or self.q90_model is None:
            return None, None
            
        low = np.expm1(self.q10_model.predict(X))
        high = np.expm1(self.q90_model.predict(X))
        
        return low, high

    def save(self, run_id: int):
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.q10_model, f"models/quantile_q10_v{run_id}.pkl")
        joblib.dump(self.q90_model, f"models/quantile_q90_v{run_id}.pkl")

    def load(self, run_id: int):
        self.q10_model = joblib.load(f"models/quantile_q10_v{run_id}.pkl")
        self.q90_model = joblib.load(f"models/quantile_q90_v{run_id}.pkl")

class ConfidenceEngine:
    """
    Scores the 'reliability' of a prediction based on data proximity.
    """
    def calculate_score(self, features: dict, training_summary: dict) -> float:
        # Simplistic implementation: decrease score if feature is outside 2 std devs of training
        # In production this would use GMM or Mahalanobis distance
        score = 100.0
        # Placeholder for real density logic
        return score
