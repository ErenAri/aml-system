"""
Graph feature engineering for GNN models.

Converts transaction data into PyTorch Geometric format with:
- Node features (account attributes + graph stats)
- Edge features (transaction attributes)
- Graph structure (edge_index)
"""
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .tabular import build_account_features


class GraphFeatureEngineer:
    """Convert tabular data to PyTorch Geometric graph format."""

    def __init__(self):
        self.node_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.node_id_map = {}

    def build_pyg_data(
        self, transactions_df: pd.DataFrame, accounts_df: pd.DataFrame
    ) -> Data:
        """
        Build PyTorch Geometric Data object.

        Args:
            transactions_df: Transaction data
            accounts_df: Account data

        Returns:
            PyG Data object with node/edge features and labels
        """
        print("Encoding nodes...")
        node_features, node_labels = self._build_node_features(accounts_df)

        print("Building edge index and features...")
        edge_index, edge_features = self._build_edge_features(transactions_df)

        # Convert to tensors
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_features)
        y = torch.LongTensor(node_labels)

        # Create PyG data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        print(f"✓ Built PyG graph: {data.num_nodes} nodes, {data.num_edges} edges")
        return data

    def _build_node_features(self, accounts_df: pd.DataFrame):
        """
        Build node feature matrix.

        TODO: Add more node features:
        - Embeddings from account metadata
        - Temporal features (account age, activity patterns)
        """
        # Create node ID mapping
        account_ids = accounts_df["account_id"].tolist()
        self.node_id_map = {acc_id: idx for idx, acc_id in enumerate(account_ids)}

        # Extract features
        features = []
        labels = []

        for _, row in accounts_df.iterrows():
            # Numerical features
            feat = [
                1 if row["account_type"].name == "BUSINESS" else 0,
                1 if row["kyc_verified"] else 0,
                row["transaction_limit"],
                row["risk_score"],
                # TODO: Add graph statistics (degree, clustering, etc.)
            ]
            features.append(feat)
            labels.append(1 if row["is_suspicious"] else 0)

        features = np.array(features, dtype=np.float32)

        # Normalize features
        features = self.scaler.fit_transform(features)

        return features, labels

    def _build_edge_features(self, transactions_df: pd.DataFrame):
        """
        Build edge index and edge features.

        Returns:
            edge_index: [2, num_edges] array
            edge_features: [num_edges, num_features] array
        """
        edge_index = [[], []]
        edge_features = []

        for _, row in transactions_df.iterrows():
            src = row["source_account"]
            tgt = row["target_account"]

            # Skip if account not in node map
            if src not in self.node_id_map or tgt not in self.node_id_map:
                continue

            src_idx = self.node_id_map[src]
            tgt_idx = self.node_id_map[tgt]

            edge_index[0].append(src_idx)
            edge_index[1].append(tgt_idx)

            # Edge features
            feat = [
                row["amount"],
                1 if row["is_international"] else 0,
                1 if row["is_cash_intensive"] else 0,
                # TODO: Add timestamp features (hour of day, day of week)
            ]
            edge_features.append(feat)

        edge_index = np.array(edge_index, dtype=np.int64)
        edge_features = np.array(edge_features, dtype=np.float32)

        # Normalize edge features
        edge_features = StandardScaler().fit_transform(edge_features)

        return edge_index, edge_features


def build_graph_tensors(accounts_df: pd.DataFrame, tx_df: pd.DataFrame, labels_path: str = None):
    """
    Build graph tensors for GNN training.
    
    Args:
        accounts_df: Account data with account_id column
        tx_df: Transaction data with source_account, target_account columns
        labels_path: Optional path to labels_nodes.parquet (with account_id and label columns)
    
    Returns:
        dict with keys: 'x', 'edge_index', 'y', 'id_map'
            - x: Node feature matrix [N, F] (normalized)
            - edge_index: Edge indices [2, E] (bidirectional)
            - y: Label vector [N] (unknown labels → -1 for semi-supervised)
            - id_map: Dict mapping account_id → node_index
    """
    print("Building graph tensors...")
    
    # Step 1: Build tabular features
    print("  - Generating node features...")
    features_df = build_account_features(accounts_df, tx_df)
    
    # Step 2: Create account_id → node_index mapping
    print("  - Creating node index mapping...")
    account_ids = features_df['account_id'].tolist()
    id_map = {acc_id: idx for idx, acc_id in enumerate(account_ids)}
    n_nodes = len(account_ids)
    
    # Step 3: Extract and normalize node features
    print("  - Normalizing features...")
    feature_cols = [col for col in features_df.columns if col != 'account_id']
    X = features_df[feature_cols].values.astype(np.float32)
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = torch.FloatTensor(X)
    
    print(f"    Node features shape: {X.shape}")
    
    # Step 4: Build bidirectional edge_index
    print("  - Building edge index (bidirectional)...")
    edge_list = []
    
    for _, row in tx_df.iterrows():
        src = row['source_account']
        tgt = row['target_account']
        
        # Only include edges where both nodes exist in our node set
        if src in id_map and tgt in id_map:
            src_idx = id_map[src]
            tgt_idx = id_map[tgt]
            
            # Add both directions for undirected graph
            edge_list.append([src_idx, tgt_idx])
            edge_list.append([tgt_idx, src_idx])
    
    if len(edge_list) > 0:
        edge_index = torch.LongTensor(edge_list).t().contiguous()
    else:
        # Empty graph
        edge_index = torch.LongTensor([[], []]).contiguous()
    
    print(f"    Edge index shape: {edge_index.shape} ({edge_index.shape[1]} directed edges)")
    
    # Step 5: Load labels (if available)
    print("  - Loading labels...")
    if labels_path and pd.io.common.file_exists(labels_path):
        labels_df = pd.read_parquet(labels_path)
        
        # Initialize all labels to -1 (unknown for semi-supervised)
        y = torch.ones(n_nodes, dtype=torch.long) * -1
        
        # Fill in known labels
        for _, row in labels_df.iterrows():
            acc_id = row['account_id']
            if acc_id in id_map:
                node_idx = id_map[acc_id]
                label = int(row['label']) if 'label' in row else int(row.get('is_suspicious', -1))
                y[node_idx] = label
        
        n_labeled = (y != -1).sum().item()
        print(f"    Labels: {n_labeled}/{n_nodes} labeled ({n_labeled/n_nodes*100:.1f}%)")
    else:
        # No labels available, set all to -1
        y = torch.ones(n_nodes, dtype=torch.long) * -1
        print(f"    No labels file found, using -1 for all nodes (semi-supervised)")
    
    # Step 6: Package results
    graph_data = {
        'x': X,
        'edge_index': edge_index,
        'y': y,
        'id_map': id_map,
    }
    
    print(f"✓ Built graph: {n_nodes} nodes, {edge_index.shape[1]} edges, {X.shape[1]} features")
    
    return graph_data


def main():
    """Standalone graph building script."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Build graph tensors for GNN")
    parser.add_argument("--accounts", required=True, help="Path to accounts parquet")
    parser.add_argument("--tx", required=True, help="Path to transactions parquet")
    parser.add_argument("--labels", default=None, help="Path to labels_nodes.parquet (optional)")
    parser.add_argument("--out", required=True, help="Output path for graph tensors (.pt file)")
    parser.add_argument("--legacy", action="store_true", help="Use legacy GraphFeatureEngineer")

    args = parser.parse_args()

    # Load data
    print(f"Loading data...")
    accounts_df = pd.read_parquet(args.accounts)
    tx_df = pd.read_parquet(args.tx)
    
    if args.legacy:
        # Use legacy feature engineer
        print("Using legacy GraphFeatureEngineer...")
        engineer = GraphFeatureEngineer()
        data = engineer.build_pyg_data(tx_df, accounts_df)
        torch.save(data, args.out)
    else:
        # Use new build_graph_tensors function
        graph_data = build_graph_tensors(accounts_df, tx_df, labels_path=args.labels)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
        
        # Save tensors
        torch.save(graph_data, args.out)
    
    print(f"\n✓ Saved graph data to {args.out}")


if __name__ == "__main__":
    main()

