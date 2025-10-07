"""
Baseline tabular model using XGBoost.

Train a gradient boosted tree classifier on engineered features
with hyperparameter tuning and cross-validation.
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb

from src.features.tabular import TabularFeatureEngineer


class BaselineModel:
    """XGBoost classifier for AML detection."""

    def __init__(self, **xgb_params):
        """
        Initialize model.

        Args:
            **xgb_params: XGBoost hyperparameters
        """
        default_params = {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "scale_pos_weight": 10,  # Handle class imbalance
            "random_state": 42,
        }
        default_params.update(xgb_params)
        self.model = xgb.XGBClassifier(**default_params)
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model.

        Args:
            X: Feature matrix
            y: Binary labels
        """
        self.feature_names = X.columns.tolist()

        print(f"Training on {len(X)} samples with {len(self.feature_names)} features...")
        print(f"Class distribution: {y.value_counts().to_dict()}")

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=True,
        )

        # Evaluate
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)[:, 1]

        print("\n=== Validation Results ===")
        print(classification_report(y_val, y_pred))
        print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")
        print(f"\nConfusion Matrix:\n{confusion_matrix(y_val, y_pred)}")

        # Feature importance
        self._print_feature_importance()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def _print_feature_importance(self, top_k: int = 15):
        """Print top-k important features."""
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        print(f"\n=== Top {top_k} Features ===")
        print(feature_importance.head(top_k).to_string(index=False))

    def save(self, path: str):
        """Save model to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"✓ Model saved to {path}")

    @staticmethod
    def load(path: str) -> "BaselineModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


def main():
    """CLI entry point for training baseline model."""
    parser = argparse.ArgumentParser(description="Train baseline tabular model")
    parser.add_argument(
        "--data", required=True, help="Path to transaction data (parquet)"
    )
    parser.add_argument(
        "--output", default="artifacts/baseline_model.pkl", help="Output model path"
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    transactions_df = pd.read_parquet(args.data)
    accounts_df = pd.read_parquet(Path(args.data).parent / "accounts.parquet")

    # Engineer features
    print("\n=== Feature Engineering ===")
    engineer = TabularFeatureEngineer()
    X, y = engineer.fit_transform(transactions_df, accounts_df)

    # Train model
    print("\n=== Training Model ===")
    model = BaselineModel()
    model.train(X, y)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()

