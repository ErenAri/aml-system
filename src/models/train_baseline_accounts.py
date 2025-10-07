"""
Train a baseline classifier on account-level features.

- Reads account_features.parquet with columns: account_id, label_node, feature_*
- Stratified train/val/test split on label_node
- Baseline: LogisticRegression (liblinear/saga), class_weight='balanced'
- Optional: XGBoost if available
- Metrics: PR-AUC, Precision@K (K=1%), FPR at 95% precision
- Saves: artifacts/baseline_model.joblib and artifacts/preds_baseline.csv

Designed for reproducibility and fast runtime.
"""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib


def set_seeds(seed: int):
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k_ratio: float) -> float:
    n = len(y_true)
    k = max(1, int(np.ceil(k_ratio * n)))
    order = np.argsort(-y_scores)
    top_k = y_true[order][:k]
    return float(top_k.mean()) if k > 0 else 0.0


def compute_fpr_at_precision(y_true: np.ndarray, y_scores: np.ndarray, target_precision: float) -> float:
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    # Find first threshold achieving at least target precision
    idx = np.argmax(precisions >= target_precision)
    if idx == 0 and precisions[0] < target_precision:
        return 1.0  # cannot achieve target precision, treat as worst-case
    thr = thresholds[min(idx, len(thresholds) - 1)] if len(thresholds) > 0 else 1.0
    y_pred = (y_scores >= thr).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = tn + fp
    return float(fp / denom) if denom > 0 else 0.0


def try_xgboost(seed: int):
    try:
        import xgboost as xgb  # type: ignore

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="aucpr",
            random_state=seed,
            tree_method="hist",
            max_bin=256,
            # scale_pos_weight will be set based on training labels later
        )
        return model
    except Exception:
        return None


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # Basic validation
    required = {"account_id", "label_node"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def make_splits(df: pd.DataFrame, seed: int):
    features = df.drop(columns=["account_id", "label_node"])\
        .select_dtypes(include=[np.number])
    labels = df["label_node"].astype(int).to_numpy()

    X_temp, X_test, y_temp, y_test, temp_idx, test_idx = train_test_split(
        features.to_numpy(), labels, np.arange(len(df)),
        test_size=0.2, stratify=labels, random_state=seed
    )
    y_temp_series = y_temp
    X_train, X_val, y_train, y_val, train_idx, val_idx = train_test_split(
        X_temp, y_temp_series, temp_idx, test_size=0.2, stratify=y_temp_series, random_state=seed
    )
    return (
        X_train, y_train, train_idx,
        X_val, y_val, val_idx,
        X_test, y_test, test_idx,
        features.columns.tolist(),
    )


def build_lr_pipeline(seed: int) -> Pipeline:
    solver = "liblinear"  # good for small/medium, supports class_weight
    lr = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver=solver,
        max_iter=200,
        random_state=seed,
        n_jobs=None,
    )
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", lr),
    ])
    return pipe


def train_and_eval(model, X_train, y_train, X_val, y_val):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # If XGBoost, set scale_pos_weight to handle imbalance
        try:
            import xgboost as xgb  # type: ignore

            if isinstance(model, xgb.XGBClassifier):
                pos = np.sum(y_train == 1)
                neg = np.sum(y_train == 0)
                spw = float(neg / max(1, pos))
                model.set_params(scale_pos_weight=spw)
        except Exception:
            pass
        model.fit(X_train, y_train)
    val_scores = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_val)
    pr_auc = float(average_precision_score(y_val, val_scores))
    p_at_1 = float(compute_precision_at_k(y_val, val_scores, 0.01))
    fpr_at_p95 = float(compute_fpr_at_precision(y_val, val_scores, 0.95))
    return {
        "pr_auc": pr_auc,
        "precision_at_1pct": p_at_1,
        "fpr_at_95p": fpr_at_p95,
        "scores": val_scores,
    }


def train_full(model, X_tr_val, y_tr_val):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_tr_val, y_tr_val)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train baseline classifier on account features")
    parser.add_argument("--data", default="account_features.parquet", help="Path to account_features.parquet")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", choices=["lr", "xgb"], default="lr", help="Model type")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts output directory")
    args = parser.parse_args()

    set_seeds(args.seed)

    data_path = Path(args.data)
    df = load_data(data_path)

    (
        X_train, y_train, train_idx,
        X_val, y_val, val_idx,
        X_test, y_test, test_idx,
        feature_names,
    ) = make_splits(df, seed=args.seed)

    # Choose model
    use_xgb = args.model == "xgb"
    xgb_model = try_xgboost(args.seed) if use_xgb else None
    if use_xgb and xgb_model is None:
        print("XGBoost not available; falling back to LogisticRegression.")
        use_xgb = False

    if use_xgb:
        model = xgb_model
    else:
        model = build_lr_pipeline(args.seed)

    # Train and evaluate on validation for reporting
    metrics_val = train_and_eval(model, X_train, y_train, X_val, y_val)

    # Fit on train+val for final model, evaluate on test for predictions export
    X_tr_val = np.vstack([X_train, X_val])
    y_tr_val = np.concatenate([y_train, y_val])
    final_model = train_full(model, X_tr_val, y_tr_val)

    test_scores = final_model.predict_proba(X_test)[:, 1] if hasattr(final_model, "predict_proba") else final_model.predict(X_test)
    # Compute test metrics (for information)
    metrics_test = {
        "pr_auc": float(average_precision_score(y_test, test_scores)),
        "precision_at_1pct": float(compute_precision_at_k(y_test, test_scores, 0.01)),
        "fpr_at_95p": float(compute_fpr_at_precision(y_test, test_scores, 0.95)),
    }

    # Prepare artifacts
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "baseline_model.joblib"
    preds_path = artifacts_dir / "preds_baseline.csv"

    # Save model
    joblib.dump({
        "model": final_model,
        "feature_names": feature_names,
        "seed": args.seed,
        "model_type": "xgb" if use_xgb else "lr",
    }, model_path)

    # Save predictions on test set
    out_df = pd.DataFrame({
        "account_id": df.iloc[test_idx]["account_id"].to_numpy(),
        "score": test_scores,
        "label": y_test,
    })
    out_df.to_csv(preds_path, index=False)

    # Compact metrics table to stdout
    header = ["split", "model", "PR-AUC", "Prec@1%", "FPR@95P"]
    row_val = ["val", "xgb" if use_xgb else "lr", f"{metrics_val['pr_auc']:.4f}", f"{metrics_val['precision_at_1pct']:.4f}", f"{metrics_val['fpr_at_95p']:.4f}"]
    row_test = ["test", "xgb" if use_xgb else "lr", f"{metrics_test['pr_auc']:.4f}", f"{metrics_test['precision_at_1pct']:.4f}", f"{metrics_test['fpr_at_95p']:.4f}"]
    widths = [max(len(h), len(r_val), len(r_test)) for h, r_val, r_test in zip(header, row_val, row_test)]
    def fmt(cols):
        return " | ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(fmt(header))
    print("-+-".join("-" * w for w in widths))
    print(fmt(row_val))
    print(fmt(row_test))

    print(f"Saved model to {model_path}")
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()


