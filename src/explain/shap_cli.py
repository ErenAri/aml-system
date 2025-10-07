"""
CLI to export SHAP global feature importance and local explanations.

Defaults:
- Loads artifacts/baseline_model.joblib (from training script)
- Loads features parquet (e.g., artifacts/account_features.parquet or provided)
- Writes:
  - artifacts/shap_global.csv (top-20)
  - artifacts/shap_local_<account_id>.json for top-20 flagged accounts
"""
import argparse
from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np
import joblib

from src.explain.shap_tools import (
    global_feature_importance,
    local_explanations,
    save_global_to_csv,
    save_locals_to_json,
)


def load_features_default(features_path: Path) -> pd.DataFrame:
    if features_path.exists():
        return pd.read_parquet(features_path)
    # Fallbacks commonly used
    alt = [
        features_path.parent / "account_features.parquet",
        Path("artifacts") / "account_features.parquet",
    ]
    for p in alt:
        if p.exists():
            return pd.read_parquet(p)
    raise FileNotFoundError(f"Features parquet not found. Tried: {[str(features_path)] + [str(p) for p in alt]}")


def load_predictions_default(artifacts_dir: Path) -> pd.DataFrame:
    preds_path = artifacts_dir / "preds_baseline.csv"
    if preds_path.exists():
        return pd.read_csv(preds_path)
    raise FileNotFoundError(f"Missing predictions file at {preds_path}. Run training to produce it.")


def main():
    parser = argparse.ArgumentParser(description="Export SHAP global and local explanations")
    parser.add_argument("--model", default="artifacts/baseline_model.joblib", help="Path to model artifact joblib")
    parser.add_argument("--features", default="artifacts/account_features.parquet", help="Path to features parquet")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--topk", type=int, default=20, help="Top-k global features and flagged accounts")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load model artifact (dict with 'model' and 'feature_names' per training script)
    model_artifact = joblib.load(args.model)

    # Load features
    features_df = load_features_default(Path(args.features))

    # Ensure we keep id columns for mapping to accounts
    if "account_id" not in features_df.columns:
        raise ValueError("Features parquet must contain 'account_id' column for local artifact naming.")

    # Extract X in the same order as used for training
    X = (
        features_df.drop(columns=["account_id"]) if "label_node" not in features_df.columns else features_df.drop(columns=["account_id", "label_node"])
    )

    # Global importance
    imp_df = global_feature_importance(model_artifact, X)
    global_path = save_global_to_csv(imp_df, artifacts_dir, top_k=args.topk)
    print(f"Saved global importances to {global_path}")

    # Local explanations for top-k flagged accounts from predictions
    preds_df = load_predictions_default(artifacts_dir)
    # Join to get positional indices aligned with features_df
    merged = preds_df.merge(
        features_df[["account_id"]].reset_index().rename(columns={"index": "row_idx"}),
        on="account_id",
        how="inner",
    )
    merged = merged.sort_values("score", ascending=False).head(args.topk)
    idx_list = merged["row_idx"].astype(int).tolist()
    index_to_id = {int(r["row_idx"]): str(r["account_id"]) for _, r in merged.iterrows()}

    locals_map = local_explanations(model_artifact, X, idx_list)
    written = save_locals_to_json(locals_map, index_to_id, artifacts_dir)
    print(f"Saved {len(written)} local explanations to {artifacts_dir}")


if __name__ == "__main__":
	main()


