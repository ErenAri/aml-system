"""
SHAP-based explanations for tabular models.

Provides feature importance and individual prediction explanations.
"""
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import Optional, List, Dict
from pathlib import Path
import json


class SHAPExplainer:
    """SHAP explainer for tree-based models."""

    def __init__(self, model, X_background: pd.DataFrame):
        """
        Initialize SHAP explainer.

        Args:
            model: Trained model (XGBoost, etc.)
            X_background: Background dataset for SHAP (sample of training data)
        """
        self.model = model
        self.explainer = shap.TreeExplainer(model.model)
        self.X_background = X_background
        self.feature_names = X_background.columns.tolist()

    def explain_prediction(self, X: pd.DataFrame, idx: int = 0) -> dict:
        """
        Explain a single prediction.

        Args:
            X: Input features
            idx: Index of instance to explain

        Returns:
            Dictionary with SHAP values and base value
        """
        shap_values = self.explainer.shap_values(X.iloc[idx : idx + 1])

        return {
            "shap_values": shap_values[0],
            "base_value": self.explainer.expected_value,
            "feature_values": X.iloc[idx].values,
            "feature_names": self.feature_names,
            "prediction": self.model.predict(X.iloc[idx : idx + 1])[0],
        }

    def plot_waterfall(self, X: pd.DataFrame, idx: int = 0, save_path: Optional[str] = None):
        """
        Generate SHAP waterfall plot for a single prediction.

        Shows how each feature contributes to pushing the prediction
        from base value to final prediction.
        """
        explanation = self.explain_prediction(X, idx)

        # Create explanation object for plotting
        shap_exp = shap.Explanation(
            values=explanation["shap_values"],
            base_values=explanation["base_value"],
            data=explanation["feature_values"],
            feature_names=self.feature_names,
        )

        plt.figure(figsize=(10, 8))
        shap.waterfall_plot(shap_exp, show=False)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Waterfall plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_force(self, X: pd.DataFrame, idx: int = 0, save_path: Optional[str] = None):
        """
        Generate SHAP force plot for a single prediction.

        Shows forces pushing prediction higher (red) or lower (blue).
        """
        shap_values = self.explainer.shap_values(X.iloc[idx : idx + 1])

        shap.force_plot(
            self.explainer.expected_value,
            shap_values[0],
            X.iloc[idx],
            matplotlib=True,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Force plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def get_global_importance(self, X: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
        """
        Compute global feature importance across all predictions.

        Args:
            X: Input features
            top_k: Number of top features to return

        Returns:
            DataFrame with feature importance scores
        """
        shap_values = self.explainer.shap_values(X)

        # Mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame(
            {"feature": self.feature_names, "importance": importance}
        ).sort_values("importance", ascending=False)

        return importance_df.head(top_k)

    def plot_summary(self, X: pd.DataFrame, save_path: Optional[str] = None):
        """
        Generate SHAP summary plot showing feature importance and effects.

        Combines feature importance with feature values (red = high, blue = low).
        """
        shap_values = self.explainer.shap_values(X)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Summary plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_dependence(
        self, X: pd.DataFrame, feature: str, interaction_feature: Optional[str] = None, save_path: Optional[str] = None
    ):
        """
        Generate SHAP dependence plot for a single feature.

        Shows how the model output changes with feature value.

        Args:
            X: Input features
            feature: Feature to plot
            interaction_feature: Optional feature to color by (shows interactions)
            save_path: Path to save plot
        """
        shap_values = self.explainer.shap_values(X)

        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature,
            shap_values,
            X,
            interaction_index=interaction_feature,
            show=False,
        )

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Dependence plot saved to {save_path}")
        else:
            plt.show()

        plt.close()


def explain_top_predictions(
    model, X: pd.DataFrame, top_k: int = 10
) -> pd.DataFrame:
    """
    Explain top-k most suspicious predictions.

    Returns a DataFrame with predictions and top contributing features.
    """
    # Get predictions
    scores = model.predict(X)

    # Get top-k
    top_indices = np.argsort(scores)[-top_k:][::-1]

    # Create explainer
    explainer = SHAPExplainer(model, X.sample(100))

    results = []
    for idx in top_indices:
        explanation = explainer.explain_prediction(X, idx)

        # Top contributing features
        shap_vals = explanation["shap_values"]
        top_features_idx = np.argsort(np.abs(shap_vals))[-3:][::-1]

        results.append(
            {
                "index": idx,
                "score": scores[idx],
                "top_feature_1": explainer.feature_names[top_features_idx[0]],
                "top_feature_1_shap": shap_vals[top_features_idx[0]],
                "top_feature_2": explainer.feature_names[top_features_idx[1]],
                "top_feature_2_shap": shap_vals[top_features_idx[1]],
                "top_feature_3": explainer.feature_names[top_features_idx[2]],
                "top_feature_3_shap": shap_vals[top_features_idx[2]],
            }
        )

    return pd.DataFrame(results)


# --- Utility functions requested ---
def _unwrap_model(model_or_artifact):
    """Return an estimator usable by SHAP from various saved formats.

    Supports:
    - plain estimator or pipeline
    - joblib artifact dict with key 'model'
    - BaselineModel (has attribute 'model')
    """
    # joblib artifact dict from training script
    if isinstance(model_or_artifact, dict) and "model" in model_or_artifact:
        return model_or_artifact["model"], model_or_artifact.get("feature_names")

    # BaselineModel wrapper
    try:
        from src.models.baseline_tabular import BaselineModel  # type: ignore

        if isinstance(model_or_artifact, BaselineModel):
            return model_or_artifact.model, getattr(model_or_artifact, "feature_names", None)
    except Exception:
        pass

    # Fallback: assume it's already an estimator/pipeline
    return model_or_artifact, None


def _build_explainer(model, background: pd.DataFrame):
    """Create a SHAP explainer with a reasonable background sample.

    Special handling for sklearn Pipeline(LogisticRegression) to use LinearExplainer
    on the transformed feature space for performance and correctness.
    """
    # Use a modest background sample for performance
    if len(background) > 2000:
        bg = background.sample(n=2000, random_state=42)
    else:
        bg = background

    # Try to detect sklearn Pipeline with final LogisticRegression
    try:
        from sklearn.pipeline import Pipeline as _SkPipeline  # type: ignore
        from sklearn.linear_model import LogisticRegression as _LogReg  # type: ignore

        if isinstance(model, _SkPipeline) and isinstance(model.named_steps.get("clf"), _LogReg):
            scaler = model.named_steps.get("scaler")
            clf = model.named_steps.get("clf")
            X_bg = scaler.transform(bg) if scaler is not None else bg.to_numpy()
            explainer = shap.LinearExplainer(clf, X_bg)

            def transform_fn(X_df: pd.DataFrame):
                return scaler.transform(X_df) if scaler is not None else X_df.to_numpy()

            return explainer, transform_fn
    except Exception:
        pass

    # Generic path
    try:
        return shap.Explainer(model, bg), (lambda X_df: X_df)
    except Exception:
        # Last-resort fallback to KernelExplainer using probability of class 1
        def predict_fn(X_):
            # Handle numpy or DataFrame inputs gracefully
            import numpy as _np
            import pandas as _pd

            X_df = _pd.DataFrame(X_, columns=background.columns) if not isinstance(X_, _pd.DataFrame) else X_
            if hasattr(model, "predict_proba"):
                return model.predict_proba(X_df)[:, 1]
            if hasattr(model, "decision_function"):
                vals = model.decision_function(X_df)
                # Map decision to [0,1]
                return 1 / (1 + _np.exp(-vals))
            preds = model.predict(X_df)
            return preds.astype(float)

        bg_np = bg.to_numpy()
        explainer = shap.KernelExplainer(predict_fn, bg_np)
        return explainer, (lambda X_df: X_df.to_numpy())


def global_feature_importance(model, X: pd.DataFrame) -> pd.DataFrame:
    """Compute mean absolute SHAP values per feature.

    Args:
        model: Estimator, pipeline, or saved artifact dict with key 'model'.
        X: Feature matrix as DataFrame (columns are feature names).

    Returns:
        DataFrame with columns ['feature', 'importance'] sorted descending.
    """
    estimator, saved_feature_names = _unwrap_model(model)

    # Align columns if feature names were saved with the model
    if saved_feature_names is not None:
        missing = [c for c in saved_feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"X is missing expected feature columns: {missing[:5]}{'...' if len(missing)>5 else ''}")
        X_local = X.loc[:, saved_feature_names]
    else:
        X_local = X

    explainer, transform_fn = _build_explainer(estimator, X_local)

    X_eval = transform_fn(X_local)
    shap_exp = explainer(X_eval)
    values = getattr(shap_exp, "values", shap_exp)  # KernelExplainer returns np.ndarray
    importance = np.abs(values).mean(axis=0)

    feature_names = list(getattr(shap_exp, "feature_names", X_local.columns))
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importance})
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)


def local_explanations(model, X: pd.DataFrame, idx_list: List[int]) -> Dict[int, List[dict]]:
    """Compute local SHAP explanations for specified row indices.

    Args:
        model: Estimator, pipeline, or saved artifact dict with key 'model'.
        X: Feature matrix as DataFrame.
        idx_list: List of integer row indices to explain (positional, not labels).

    Returns:
        Mapping of row index -> list of {feature, shap_value} contributions.
    """
    if len(idx_list) == 0:
        return {}

    estimator, saved_feature_names = _unwrap_model(model)
    if saved_feature_names is not None:
        X_local = X.loc[:, saved_feature_names]
    else:
        X_local = X

    # Subset rows for explanation
    rows = X_local.iloc[idx_list]

    explainer, transform_fn = _build_explainer(estimator, X_local)
    shap_exp = explainer(transform_fn(rows))
    values = getattr(shap_exp, "values", shap_exp)
    feature_names = list(getattr(shap_exp, "feature_names", X_local.columns))

    out: Dict[int, List[dict]] = {}
    for pos, idx in enumerate(idx_list):
        row_vals = values[pos]
        out[int(idx)] = [
            {"feature": feature_names[j], "shap_value": float(row_vals[j])}
            for j in range(len(feature_names))
        ]
    return out


def save_global_to_csv(importance_df: pd.DataFrame, artifacts_dir: Path, top_k: int = 20) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "shap_global.csv"
    importance_df.head(top_k).to_csv(out_path, index=False)
    return out_path


def save_locals_to_json(locals_map: Dict[int, List[dict]], index_to_id: Dict[int, str], artifacts_dir: Path) -> List[Path]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for idx, contribs in locals_map.items():
        acc_id = index_to_id.get(idx, str(idx))
        out_path = artifacts_dir / f"shap_local_{acc_id}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(contribs, f, ensure_ascii=False, indent=2)
        written.append(out_path)
    return written

