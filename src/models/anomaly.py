"""
Anomaly detection utilities and ensemble methods.

Combines outputs from multiple models for robust detection.
"""
import numpy as np
import pandas as pd
from typing import List, Dict


class AnomalyEnsemble:
    """
    Ensemble multiple anomaly detection models.

    Combines predictions from:
    - Tabular baseline (XGBoost)
    - GNN model
    - (Optional) Unsupervised methods (Isolation Forest, One-Class SVM)
    """

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize ensemble.

        Args:
            weights: Model weights for weighted voting (default: equal weights)
        """
        self.weights = weights or {}
        self.models = {}

    def add_model(self, name: str, model):
        """Register a model in the ensemble."""
        self.models[name] = model
        if name not in self.weights:
            self.weights[name] = 1.0

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ensemble prediction using weighted voting.

        Args:
            X: Input features

        Returns:
            Ensemble anomaly scores (0-1)
        """
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred * self.weights[name])

        # Weighted average
        ensemble_scores = np.mean(predictions, axis=0)
        return ensemble_scores

    def predict_with_breakdown(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return predictions with per-model breakdown.

        Useful for explainability and debugging.
        """
        results = {}
        for name, model in self.models.items():
            results[f"{name}_score"] = model.predict(X)

        results["ensemble_score"] = self.predict(X)
        return pd.DataFrame(results)


class AnomalyThresholdOptimizer:
    """
    Optimize anomaly threshold for desired precision/recall.

    TODO: Implement threshold optimization based on business constraints:
    - Max false positive rate
    - Min recall
    - Cost-sensitive thresholds
    """

    def __init__(self, target_precision: float = 0.9):
        self.target_precision = target_precision
        self.optimal_threshold = 0.5

    def fit(self, y_true: np.ndarray, y_scores: np.ndarray):
        """Find optimal threshold."""
        from sklearn.metrics import precision_recall_curve

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

        # Find threshold that achieves target precision
        idx = np.argmax(precisions >= self.target_precision)
        self.optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5

        print(f"Optimal threshold: {self.optimal_threshold:.4f}")
        print(f"Precision: {precisions[idx]:.4f}, Recall: {recalls[idx]:.4f}")

        return self.optimal_threshold

    def predict(self, y_scores: np.ndarray) -> np.ndarray:
        """Apply optimized threshold."""
        return (y_scores >= self.optimal_threshold).astype(int)


def detect_anomalous_patterns(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based anomaly detection for known patterns.

    TODO: Implement pattern detection:
    - Circular transfers (A -> B -> C -> A)
    - Structuring (many transactions just below $10k)
    - Rapid movement (money bounces through accounts quickly)
    - Geographic anomalies (unusual country patterns)
    """
    anomalies = []

    # Example: Flag high-value international transfers
    high_value_intl = transactions_df[
        (transactions_df["amount"] > 50000) & (transactions_df["is_international"])
    ]
    anomalies.append(
        high_value_intl.assign(pattern="high_value_international")
    )

    # TODO: Add more pattern detection logic

    if anomalies:
        return pd.concat(anomalies, ignore_index=True)
    else:
        return pd.DataFrame()


if __name__ == "__main__":
    print("Anomaly detection utilities - import and use in other modules")

