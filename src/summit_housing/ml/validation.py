import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
import os

class TemporalFoldSplitter:
    """
    Splitter that respects time. Ensures that training data always precedes testing data.
    """
    def __init__(self, date_col: str = 'tx_date', test_months: int = 6, n_folds: int = 3):
        self.date_col = date_col
        self.test_months = test_months
        self.n_folds = n_folds

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates (train, test) folds.
        Each fold's test set is a window of `test_months` ending at a specific point,
        and the train set is everything before it.
        """
        df = df.sort_values(self.date_col).copy()
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_col]):
            df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        max_date = df[self.date_col].max()
        folds = []
        
        for i in range(self.n_folds):
            # Calculate testing window for this fold
            # Fold 0: Last X months
            # Fold 1: X to 2X months ago, and so on
            test_end = max_date - pd.DateOffset(months=i * self.test_months)
            test_start = test_end - pd.DateOffset(months=self.test_months)
            
            test_df = df[(df[self.date_col] > test_start) & (df[self.date_col] <= test_end)]
            train_df = df[df[self.date_col] <= test_start]
            
            if len(test_df) > 0 and len(train_df) > 0:
                folds.append((train_df, test_df))
                
        return folds[::-1] # Return in chronological order (oldest fold first)

class ResidualAnalyzer:
    """
    Analyzes model errors to identify segments with high bias or variance.
    """
    def __init__(self, results_dir: str = "docs/validation"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def analyze(self, y_true: np.array, y_pred: np.array, metadata: pd.DataFrame, run_name: str = "run"):
        """
        Generates error breakdown by metadata categories (e.g., city, prop_type).
        """
        analysis_df = metadata.copy()
        analysis_df['actual'] = y_true
        analysis_df['predicted'] = y_pred
        analysis_df['error'] = analysis_df['predicted'] - analysis_df['actual']
        analysis_df['abs_error'] = analysis_df['error'].abs()
        analysis_df['error_pct'] = (analysis_df['abs_error'] / analysis_df['actual']) * 100

        # 1. Summary Metrics
        overall_mae = analysis_df['abs_error'].mean()
        overall_mape = analysis_df['error_pct'].mean()
        
        print(f"--- Validation Report: {run_name} ---")
        print(f"Overall MAE: ${overall_mae:,.0f}")
        print(f"Overall MAPE: {overall_mape:.2f}%")

        # 2. Slice Analysis
        report = []
        for col in ['city', 'prop_type']:
            if col in analysis_df.columns:
                slice_stats = analysis_df.groupby(col).agg({
                    'abs_error': 'mean',
                    'error_pct': 'mean',
                    'actual': 'count'
                }).rename(columns={'abs_error': 'MAE', 'error_pct': 'MAPE', 'actual': 'Count'})
                report.append(f"\nError by {col}:\n{slice_stats.to_string()}")

        # 3. High Value Segment Analysis
        luxury_mask = analysis_df['actual'] > 2_000_000
        if luxury_mask.any():
            luxury_mae = analysis_df[luxury_mask]['abs_error'].mean()
            print(f"Luxury Segment (> $2M) MAE: ${luxury_mae:,.0f} (N={luxury_mask.sum()})")

        # Save Text Report
        with open(os.path.join(self.results_dir, f"{run_name}_report.txt"), "w") as f:
            f.write(f"Validation Report: {run_name}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Overall MAE: ${overall_mae:,.0f}\n")
            f.write(f"Overall MAPE: {overall_mape:.2f}%\n")
            f.write("\n".join(report))

        return analysis_df
