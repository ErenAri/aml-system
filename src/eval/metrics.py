"""
Evaluation metrics for AML detection models.

Comprehensive metrics beyond accuracy:
- Precision, Recall, F1 (especially at high precision)
- ROC-AUC, PR-AUC
- Confusion matrix analysis
- Cost-sensitive metrics
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from typing import Optional, Dict
import argparse


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the precision-recall curve (average precision).

    Args:
        y_true: Ground-truth binary labels (0/1) with shape (n_samples,).
        y_score: Predicted scores/probabilities with shape (n_samples,).

    Returns:
        Average precision (PR-AUC) as a float in [0, 1].
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    return float(average_precision_score(y_true, y_score))


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: float = 0.01) -> float:
    """Compute precision among the top-k fraction ranked by score.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_score: Predicted scores/probabilities.
        k: Fraction in (0, 1], e.g., 0.01 for top 1%.

    Returns:
        Precision in [0, 1] over the top-k fraction.
    """
    if not (0 < k <= 1):
        raise ValueError("k must be in the interval (0, 1].")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = y_true.shape[0]
    if n == 0:
        return 0.0

    top_n = int(np.ceil(k * n))
    top_n = max(1, min(top_n, n))

    # Indices of scores sorted descending
    order = np.argsort(-y_score, kind="mergesort")
    top_idx = order[:top_n]
    positives_in_top_k = np.sum(y_true[top_idx] == 1)
    return float(positives_in_top_k / top_n)


def fpr_at_precision(
    y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.95
) -> float:
    """Compute the false positive rate at the smallest threshold achieving target precision.

    This finds a score threshold such that precision >= target_precision, then computes
    FPR at that threshold.

    Args:
        y_true: Ground-truth binary labels (0/1).
        y_score: Predicted scores/probabilities.
        target_precision: Desired minimum precision, in (0, 1].

    Returns:
        False positive rate (FP / (FP + TN)) at the chosen threshold. If no threshold
        achieves the target precision, returns np.nan.
    """
    if not (0 < target_precision <= 1):
        raise ValueError("target_precision must be in the interval (0, 1].")

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_true.size == 0:
        return float("nan")

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # precision_recall_curve returns precision/recall of length n_points and thresholds of length n_points-1
    # Align thresholds with precision[1:] and recall[1:]. We seek any index i with precision[i] >= target.
    candidate_indices = np.where(precision[1:] >= target_precision)[0]
    if candidate_indices.size == 0:
        return float("nan")

    # Choose the highest threshold among candidates to be most conservative
    chosen_idx = candidate_indices[np.argmax(thresholds[candidate_indices])]
    threshold = thresholds[chosen_idx]

    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    denom = fp + tn
    if denom == 0:
        return float("nan")
    return float(fp / denom)


def _format_markdown_table(rows: Dict[str, float]) -> str:
    """Return a markdown table given a mapping of metric->value."""
    lines = ["| Metric | Value |", "|---|---|"]
    for key, value in rows.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            val_str = "NaN"
        else:
            val_str = f"{value:.6f}"
        lines.append(f"| {key} | {val_str} |")
    return "\n".join(lines)


def _cli():
    parser = argparse.ArgumentParser(description="Compute AML evaluation metrics from predictions CSV.")
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV with columns: account_id, score, label",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.01,
        help="Top-k fraction for precision_at_k (default: 0.01)",
    )
    parser.add_argument(
        "--target_precision",
        type=float,
        default=0.95,
        help="Target precision for fpr_at_precision (default: 0.95)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    expected_cols = {"account_id", "score", "label"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV: {sorted(missing)}")

    y_true = df["label"].to_numpy()
    y_score = df["score"].to_numpy()

    metrics_map = {
        "PR-AUC": pr_auc(y_true, y_score),
        f"Precision@{args.k:.4f}": precision_at_k(y_true, y_score, k=args.k),
        f"FPR@Precision≥{args.target_precision:.2f}": fpr_at_precision(
            y_true, y_score, target_precision=args.target_precision
        ),
    }

    print(_format_markdown_table(metrics_map))


class AMLMetrics:
    """Comprehensive metrics for AML detection."""

    def __init__(self, y_true: np.ndarray, y_pred: np.ndarray, y_scores: np.ndarray):
        """
        Initialize metrics calculator.

        Args:
            y_true: True labels (0/1)
            y_pred: Predicted labels (0/1)
            y_scores: Predicted probabilities
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_scores = y_scores

    def compute_all(self) -> Dict:
        """Compute all metrics."""
        return {
            "classification_report": classification_report(self.y_true, self.y_pred),
            "confusion_matrix": confusion_matrix(self.y_true, self.y_pred),
            "roc_auc": roc_auc_score(self.y_true, self.y_scores),
            "pr_auc": average_precision_score(self.y_true, self.y_scores),
            "cost_sensitive": self.compute_cost_sensitive_metrics(),
        }

    def compute_cost_sensitive_metrics(
        self, fp_cost: float = 1.0, fn_cost: float = 10.0
    ) -> Dict:
        """
        Compute cost-sensitive metrics.

        In AML, false negatives (missed fraud) are typically more costly
        than false positives (false alarms).

        Args:
            fp_cost: Cost of false positive
            fn_cost: Cost of false negative

        Returns:
            Dictionary with cost metrics
        """
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()

        total_cost = fp * fp_cost + fn * fn_cost
        avg_cost_per_sample = total_cost / len(self.y_true)

        return {
            "total_cost": total_cost,
            "avg_cost_per_sample": avg_cost_per_sample,
            "false_positives": fp,
            "false_negatives": fn,
            "fp_cost": fp_cost,
            "fn_cost": fn_cost,
        }

    def plot_confusion_matrix(self, save_path: Optional[str] = None):
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(self.y_true, self.y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Normal", "Suspicious"],
            yticklabels=["Normal", "Suspicious"],
        )
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.title("Confusion Matrix")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Confusion matrix saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_roc_curve(self, save_path: Optional[str] = None):
        """Plot ROC curve."""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        auc = roc_auc_score(self.y_true, self.y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ ROC curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_precision_recall_curve(self, save_path: Optional[str] = None):
        """Plot Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_scores)
        pr_auc = average_precision_score(self.y_true, self.y_scores)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f"PR (AUC={pr_auc:.4f})", linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ PR curve saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_threshold_analysis(self, save_path: Optional[str] = None):
        """
        Plot how precision/recall/F1 change with threshold.

        Helps choose optimal operating point.
        """
        thresholds = np.linspace(0, 1, 100)
        precisions, recalls, f1s = [], [], []

        for thresh in thresholds:
            y_pred_thresh = (self.y_scores >= thresh).astype(int)

            if y_pred_thresh.sum() == 0:
                precisions.append(0)
                recalls.append(0)
                f1s.append(0)
                continue

            cm = confusion_matrix(self.y_true, y_pred_thresh)
            tn, fp, fn, tp = cm.ravel()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label="Precision", linewidth=2)
        plt.plot(thresholds, recalls, label="Recall", linewidth=2)
        plt.plot(thresholds, f1s, label="F1 Score", linewidth=2)
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold Analysis")
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Threshold analysis saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def print_summary(self):
        """Print comprehensive metrics summary."""
        print("=" * 60)
        print("AML DETECTION METRICS SUMMARY")
        print("=" * 60)

        print("\n--- Classification Report ---")
        print(classification_report(self.y_true, self.y_pred))

        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(self.y_true, self.y_pred))

        print("\n--- ROC & PR Metrics ---")
        print(f"ROC-AUC: {roc_auc_score(self.y_true, self.y_scores):.4f}")
        print(f"PR-AUC:  {average_precision_score(self.y_true, self.y_scores):.4f}")

        print("\n--- Cost-Sensitive Metrics ---")
        cost_metrics = self.compute_cost_sensitive_metrics()
        for key, value in cost_metrics.items():
            print(f"{key}: {value}")

        print("=" * 60)


def compare_models(
    y_true: np.ndarray, predictions: Dict[str, np.ndarray], save_path: Optional[str] = None
):
    """
    Compare multiple models on ROC curve.

    Args:
        y_true: True labels
        predictions: Dictionary {model_name: y_scores}
        save_path: Path to save comparison plot
    """
    plt.figure(figsize=(10, 8))

    for model_name, y_scores in predictions.items():
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model Comparison - ROC Curves")
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"✓ Model comparison saved to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    _cli()

