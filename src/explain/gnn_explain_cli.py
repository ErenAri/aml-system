"""
CLI to precompute GNN explanations for top flagged nodes.

Defaults:
- Loads artifacts/gnn_model.pt (trained GNN model)
- Loads graph data (x, edge_index) from artifacts/graph_data.pt or rebuilds from transactions
- Loads predictions from artifacts/preds_gnn.csv
- Precomputes explanations for top-10 highest risk nodes
- Caches results to artifacts/explanations/node_<id>.json
"""
import argparse
from pathlib import Path
import sys

import torch
import pandas as pd
import numpy as np

from src.explain.gnn_explain import explain_node, cache_explanation, load_cached_explanation
from src.features.graph import GraphFeatureEngineer


def load_graph_data(graph_data_path: Path, tx_path: Path = None, accounts_path: Path = None):
    """Load graph tensors (x, edge_index) from saved file or rebuild."""
    if graph_data_path.exists():
        print(f"Loading graph data from {graph_data_path}...")
        graph_data = torch.load(graph_data_path)
        return graph_data["x"], graph_data["edge_index"], graph_data.get("edge_attr"), graph_data.get("id_map")
    
    # Fallback: rebuild from transactions
    if tx_path is None or accounts_path is None:
        raise FileNotFoundError(
            f"Graph data not found at {graph_data_path} and no transaction/account paths provided for rebuild."
        )
    
    print("Rebuilding graph data from transactions...")
    tx_df = pd.read_parquet(tx_path)
    accounts_df = pd.read_parquet(accounts_path)
    
    engineer = GraphFeatureEngineer()
    data = engineer.build_pyg_data(tx_df, accounts_df)
    
    # Save for future use
    graph_data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "x": data.x,
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr if hasattr(data, "edge_attr") else None,
        "id_map": engineer.node_id_map,
    }, graph_data_path)
    print(f"Saved graph data to {graph_data_path}")
    
    return data.x, data.edge_index, getattr(data, "edge_attr", None), engineer.node_id_map


def load_model(model_path: Path, num_features: int, model_type: str = "graphsage"):
    """Load trained GNN model."""
    print(f"Loading {model_type} model from {model_path}...")
    
    if model_type == "graphsage":
        from src.models.gnn import GraphSAGEModel
        model = GraphSAGEModel(num_features=num_features, hidden_dim=64, num_classes=2)
    elif model_type == "gat":
        from src.models.gnn import GATModel
        model = GATModel(num_features=num_features, hidden_dim=64, num_classes=2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def load_predictions(preds_path: Path) -> pd.DataFrame:
    """Load GNN predictions CSV."""
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")
    return pd.read_csv(preds_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute GNN explanations for top-k flagged nodes")
    parser.add_argument("--model", default="artifacts/gnn_model.pt", help="Path to trained GNN model")
    parser.add_argument("--model-type", default="graphsage", choices=["graphsage", "gat"], help="Model architecture")
    parser.add_argument("--graph-data", default="artifacts/graph_data.pt", help="Path to cached graph data")
    parser.add_argument("--tx", default="src/data/data/transactions.parquet", help="Path to transactions (fallback)")
    parser.add_argument("--accounts", default="src/data/data/accounts.parquet", help="Path to accounts (fallback)")
    parser.add_argument("--preds", default="artifacts/preds_gnn.csv", help="Path to GNN predictions CSV")
    parser.add_argument("--artifacts", default="artifacts", help="Artifacts output directory")
    parser.add_argument("--topk", type=int, default=10, help="Number of top flagged nodes to explain")
    parser.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimization epochs")
    parser.add_argument("--force", action="store_true", help="Force recompute even if cached")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Load graph data
    try:
        x, edge_index, edge_attr, id_map = load_graph_data(
            Path(args.graph_data),
            tx_path=Path(args.tx) if Path(args.tx).exists() else None,
            accounts_path=Path(args.accounts) if Path(args.accounts).exists() else None,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load model
    try:
        model = load_model(Path(args.model), num_features=x.shape[1], model_type=args.model_type)
    except FileNotFoundError:
        print(f"Error: Model not found at {args.model}")
        sys.exit(1)

    # Load predictions
    try:
        preds_df = load_predictions(Path(args.preds))
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Get top-k flagged nodes
    # Assuming preds_gnn.csv has columns: node_id or account_id, score, prediction
    if "node_id" in preds_df.columns:
        node_col = "node_id"
    elif "account_id" in preds_df.columns:
        node_col = "account_id"
        # Need to map account_id to node_id
        if id_map is None:
            print("Error: id_map required to map account_id to node indices")
            sys.exit(1)
        # Reverse id_map
        reverse_map = {v: k for k, v in id_map.items()}
        preds_df["node_id"] = preds_df["account_id"].map(reverse_map)
        preds_df = preds_df.dropna(subset=["node_id"])
        preds_df["node_id"] = preds_df["node_id"].astype(int)
        node_col = "node_id"
    else:
        print("Error: preds_gnn.csv must have 'node_id' or 'account_id' column")
        sys.exit(1)

    # Sort by score (assuming higher = more suspicious)
    score_col = "score" if "score" in preds_df.columns else "prediction_score"
    top_nodes = preds_df.sort_values(score_col, ascending=False).head(args.topk)

    print(f"\nPrecomputing explanations for top-{args.topk} flagged nodes...")
    print(f"Using {args.epochs} GNNExplainer epochs per node")

    for i, row in enumerate(top_nodes.itertuples(), 1):
        node_id = int(getattr(row, node_col))
        score = getattr(row, score_col)

        # Check cache
        if not args.force:
            cached = load_cached_explanation(node_id, artifacts_dir)
            if cached is not None:
                print(f"  [{i}/{args.topk}] Node {node_id} (score: {score:.3f}) - Using cached explanation")
                continue

        print(f"  [{i}/{args.topk}] Node {node_id} (score: {score:.3f}) - Computing explanation...")

        # Explain node
        try:
            explanation = explain_node(
                model=model,
                x=x,
                edge_index=edge_index,
                node_id=node_id,
                epochs=args.epochs,
                edge_attr=edge_attr,
            )

            # Cache
            out_path = cache_explanation(explanation, artifacts_dir)
            print(f"       → {explanation['textual_reason']}")
            print(f"       → Saved to {out_path}")

        except Exception as e:
            print(f"       → Error: {e}")
            continue

    print(f"\n✓ Completed explanation precomputation")
    print(f"  Explanations saved to {artifacts_dir / 'explanations'}")


if __name__ == "__main__":
    main()

