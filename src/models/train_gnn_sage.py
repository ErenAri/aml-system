"""
Train a GraphSAGE node classifier for AML detection.

- Loads artifacts/graph_data.pt (x, edge_index, y, id_map)
- Stratified train/val/test split on labeled nodes
- GraphSAGE architecture: 2-3 SageConv layers with dropout
- Metrics: PR-AUC, Precision@K (K=1%), FPR at 95% precision
- Saves: artifacts/gnn_sage.pt and artifacts/preds_gnn.csv

Designed for fast training with best checkpoint saving.
"""
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, precision_recall_curve


def set_seeds(seed: int):
    import os
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k_ratio: float) -> float:
    """Precision in top K% of predictions."""
    n = len(y_true)
    k = max(1, int(np.ceil(k_ratio * n)))
    order = np.argsort(-y_scores)
    top_k = y_true[order][:k]
    return float(top_k.mean()) if k > 0 else 0.0


def compute_fpr_at_precision(y_true: np.ndarray, y_scores: np.ndarray, target_precision: float) -> float:
    """FPR at specified precision threshold."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    idx = np.argmax(precisions >= target_precision)
    if idx == 0 and precisions[0] < target_precision:
        return 1.0
    thr = thresholds[min(idx, len(thresholds) - 1)] if len(thresholds) > 0 else 1.0
    y_pred = (y_scores >= thr).astype(int)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    denom = tn + fp
    return float(fp / denom) if denom > 0 else 0.0


class GraphSAGEClassifier(torch.nn.Module):
    """GraphSAGE node classifier with 2-3 conv layers."""

    def __init__(self, in_channels: int, hidden_channels: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))

        self.classifier = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Final classification layer (logits)
        x = self.classifier(x)
        return x.squeeze(-1)  # [num_nodes]


def load_graph_data(path: Path):
    """Load graph data from .pt file."""
    data = torch.load(path)
    required = {"x", "edge_index", "y", "id_map"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys in graph_data.pt: {sorted(missing)}")
    return data["x"], data["edge_index"], data["y"], data["id_map"]


def create_stratified_masks(y, seed: int):
    """Create train/val/test masks with stratification on labeled nodes."""
    # Find labeled nodes (y != -1)
    labeled_mask = (y != -1).numpy()
    labeled_indices = np.where(labeled_mask)[0]
    labeled_y = y[labeled_mask].numpy()

    # Stratified split: 60% train, 20% val, 20% test
    train_idx, temp_idx, _, temp_y = train_test_split(
        labeled_indices, labeled_y, test_size=0.4, stratify=labeled_y, random_state=seed
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=temp_y, random_state=seed
    )

    num_nodes = len(y)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def compute_metrics(y_true, y_scores):
    """Compute PR-AUC, Precision@1%, FPR@95%."""
    y_true_np = y_true.cpu().numpy()
    y_scores_np = y_scores.cpu().numpy()

    pr_auc = float(average_precision_score(y_true_np, y_scores_np))
    p_at_1 = float(compute_precision_at_k(y_true_np, y_scores_np, 0.01))
    fpr_at_p95 = float(compute_fpr_at_precision(y_true_np, y_scores_np, 0.95))

    return {
        "pr_auc": pr_auc,
        "precision_at_1pct": p_at_1,
        "fpr_at_95p": fpr_at_p95,
    }


def train_epoch(model, x, edge_index, y, train_mask, optimizer, device):
    """Single training epoch."""
    model.train()
    optimizer.zero_grad()

    logits = model(x, edge_index)
    # BCE loss on labeled training nodes only
    loss = F.binary_cross_entropy_with_logits(
        logits[train_mask], y[train_mask].float()
    )

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, x, edge_index, y, mask, device):
    """Evaluate model on given mask."""
    model.eval()
    logits = model(x, edge_index)
    probs = torch.sigmoid(logits)

    # Extract predictions for masked nodes
    y_masked = y[mask]
    probs_masked = probs[mask]

    metrics = compute_metrics(y_masked, probs_masked)
    return metrics, probs


def main():
    parser = argparse.ArgumentParser(description="Train GraphSAGE node classifier")
    parser.add_argument("--data", default="artifacts/graph_data.pt", help="Path to graph_data.pt")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of SAGE layers (2 or 3)")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Artifacts directory")
    args = parser.parse_args()

    set_seeds(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load graph data
    data_path = Path(args.data)
    x, edge_index, y, id_map = load_graph_data(data_path)

    # Move to device
    x = x.to(device)
    edge_index = edge_index.to(device)
    y = y.to(device)

    # Create masks
    train_mask, val_mask, test_mask = create_stratified_masks(y, args.seed)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    print(f"Nodes: {len(y)} | Train: {train_mask.sum()} | Val: {val_mask.sum()} | Test: {test_mask.sum()}")
    print(f"Features: {x.shape[1]} | Edges: {edge_index.shape[1]}")

    # Initialize model
    model = GraphSAGEClassifier(
        in_channels=x.shape[1],
        hidden_channels=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_val_pr_auc = 0.0
    best_epoch = 0
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifacts_dir / "gnn_sage.pt"

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, x, edge_index, y, train_mask, optimizer, device)

        if epoch % 10 == 0 or epoch == 1:
            val_metrics, _ = evaluate(model, x, edge_index, y, val_mask, device)
            print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val PR-AUC: {val_metrics['pr_auc']:.4f}")

            # Save best checkpoint
            if val_metrics["pr_auc"] > best_val_pr_auc:
                best_val_pr_auc = val_metrics["pr_auc"]
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_pr_auc": best_val_pr_auc,
                    "args": vars(args),
                }, checkpoint_path)

    print(f"\nBest validation PR-AUC: {best_val_pr_auc:.4f} at epoch {best_epoch}")

    # Load best checkpoint for final evaluation
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate on validation and test
    val_metrics, _ = evaluate(model, x, edge_index, y, val_mask, device)
    test_metrics, test_probs = evaluate(model, x, edge_index, y, test_mask, device)

    # Save predictions on test set
    test_indices = test_mask.cpu().numpy().nonzero()[0]
    test_probs_np = test_probs[test_mask].cpu().numpy()
    test_y_np = y[test_mask].cpu().numpy()

    # Map node indices back to account IDs
    # id_map is dict: account_id -> node_idx, need reverse
    idx_to_id = {v: k for k, v in id_map.items()}
    test_account_ids = [idx_to_id[int(idx)] for idx in test_indices]

    preds_df = pd.DataFrame({
        "account_id": test_account_ids,
        "score": test_probs_np,
        "label": test_y_np,
    })
    preds_path = artifacts_dir / "preds_gnn.csv"
    preds_df.to_csv(preds_path, index=False)

    # Print compact metrics table
    header = ["split", "model", "PR-AUC", "Prec@1%", "FPR@95P"]
    row_val = ["val", "gnn", f"{val_metrics['pr_auc']:.4f}", f"{val_metrics['precision_at_1pct']:.4f}", f"{val_metrics['fpr_at_95p']:.4f}"]
    row_test = ["test", "gnn", f"{test_metrics['pr_auc']:.4f}", f"{test_metrics['precision_at_1pct']:.4f}", f"{test_metrics['fpr_at_95p']:.4f}"]
    widths = [max(len(h), len(r_val), len(r_test)) for h, r_val, r_test in zip(header, row_val, row_test)]

    def fmt(cols):
        return " | ".join(c.ljust(w) for c, w in zip(cols, widths))

    print("\n" + fmt(header))
    print("-+-".join("-" * w for w in widths))
    print(fmt(row_val))
    print(fmt(row_test))

    print(f"\nSaved checkpoint to {checkpoint_path}")
    print(f"Saved predictions to {preds_path}")


if __name__ == "__main__":
    main()

