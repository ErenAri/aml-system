"""
Graph Neural Network models for AML detection.

Implements GraphSAGE and GAT architectures with PyTorch Geometric.
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool
from torch_geometric.data import Data
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

from src.features.graph import GraphFeatureEngineer


class GraphSAGEModel(torch.nn.Module):
    """GraphSAGE model for node classification."""

    def __init__(self, num_features: int, hidden_dim: int = 64, num_classes: int = 2):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # Graph convolutions
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Classification
        x = self.classifier(x)
        return x


class GATModel(torch.nn.Module):
    """Graph Attention Network model for node classification."""

    def __init__(
        self, num_features: int, hidden_dim: int = 64, num_classes: int = 2, heads: int = 4
    ):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=0.6)
        self.classifier = torch.nn.Linear(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = self.classifier(x)
        return x


class GNNTrainer:
    """Trainer for GNN models."""

    def __init__(self, model, lr: float = 0.001, weight_decay: float = 5e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train_epoch(self, data: Data, train_mask):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        out = self.model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], data.y[train_mask])

        # Backward pass
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def evaluate(self, data: Data, mask):
        """Evaluate on validation/test set."""
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            proba = F.softmax(out, dim=1)[:, 1]

            correct = pred[mask] == data.y[mask]
            accuracy = correct.sum().item() / mask.sum().item()

            # Detailed metrics
            y_true = data.y[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
            y_proba = proba[mask].cpu().numpy()

            return {
                "accuracy": accuracy,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_proba": y_proba,
            }

    def train(self, data: Data, epochs: int = 100):
        """
        Full training loop with train/val split.

        TODO: Implement proper train/val/test split on graph data
        """
        # Create random train/val split
        num_nodes = data.num_nodes
        indices = torch.randperm(num_nodes)
        train_size = int(0.8 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:]] = True

        data = data.to(self.device)

        print(f"Training on {train_mask.sum()} nodes, validating on {val_mask.sum()} nodes")

        best_val_acc = 0
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(data, train_mask)

            if epoch % 10 == 0:
                val_results = self.evaluate(data, val_mask)
                val_acc = val_results["accuracy"]

                print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc

        # Final evaluation
        print("\n=== Final Validation Results ===")
        val_results = self.evaluate(data, val_mask)
        print(classification_report(val_results["y_true"], val_results["y_pred"]))
        print(f"ROC-AUC: {roc_auc_score(val_results['y_true'], val_results['y_proba']):.4f}")

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(self.model.state_dict(), path)
        print(f"✓ Model saved to {path}")

    @staticmethod
    def load(model_class, path: str, **model_kwargs):
        """Load model from checkpoint."""
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(path))
        return model


def main():
    """CLI entry point for training GNN."""
    parser = argparse.ArgumentParser(description="Train GNN model")
    parser.add_argument("--data", required=True, help="Path to transaction data")
    parser.add_argument("--output", default="artifacts/gnn_model.pt", help="Output model path")
    parser.add_argument("--model", choices=["graphsage", "gat"], default="graphsage")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()

    # Load and prepare data
    print("Loading data...")
    transactions_df = pd.read_parquet(args.data)
    accounts_df = pd.read_parquet(Path(args.data).parent / "accounts.parquet")

    print("\n=== Building Graph ===")
    engineer = GraphFeatureEngineer()
    data = engineer.build_pyg_data(transactions_df, accounts_df)

    # Initialize model
    print(f"\n=== Training {args.model.upper()} Model ===")
    if args.model == "graphsage":
        model = GraphSAGEModel(
            num_features=data.num_node_features, hidden_dim=args.hidden_dim
        )
    else:
        model = GATModel(num_features=data.num_node_features, hidden_dim=args.hidden_dim)

    # Train
    trainer = GNNTrainer(model, lr=args.lr)
    trainer.train(data, epochs=args.epochs)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    trainer.save(args.output)


if __name__ == "__main__":
    main()

