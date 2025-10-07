"""
Tabular feature engineering for baseline models.

Extracts graph-based and statistical features from transaction data:
- Node features: degree, clustering coefficient, PageRank, betweenness
- Transaction features: velocity, amount statistics, time patterns
- Behavioral features: deviation from normal patterns
"""
import pandas as pd
import networkx as nx
import numpy as np
from typing import Tuple


class TabularFeatureEngineer:
    """Extract tabular features from transaction graph."""

    def __init__(self):
        self.graph = None
        self.feature_names = []

    def fit_transform(
        self, transactions_df: pd.DataFrame, accounts_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate features for each account.

        Args:
            transactions_df: Transaction data
            accounts_df: Account data

        Returns:
            features_df: Feature matrix (one row per account)
            labels: Binary labels (0: normal, 1: suspicious)
        """
        print("Building transaction graph...")
        self._build_graph(transactions_df)

        print("Extracting graph features...")
        graph_features = self._extract_graph_features()

        print("Extracting transaction features...")
        txn_features = self._extract_transaction_features(transactions_df)

        print("Extracting account features...")
        account_features = self._extract_account_features(accounts_df)

        # Merge all features
        features_df = pd.concat(
            [account_features, graph_features, txn_features], axis=1
        )

        # Fill missing values
        features_df = features_df.fillna(0)

        labels = accounts_df.set_index("account_id")["is_suspicious"].astype(int)

        print(f"✓ Generated {len(features_df.columns)} features for {len(features_df)} accounts")
        self.feature_names = features_df.columns.tolist()

        return features_df, labels

    def _build_graph(self, transactions_df: pd.DataFrame):
        """Build directed graph from transactions."""
        self.graph = nx.DiGraph()

        for _, row in transactions_df.iterrows():
            self.graph.add_edge(
                row["source_account"],
                row["target_account"],
                weight=row["amount"],
                timestamp=row["timestamp"],
            )

    def _extract_graph_features(self) -> pd.DataFrame:
        """
        Extract graph topology features.

        TODO: Add more sophisticated graph features:
        - Community detection (Louvain)
        - Shortest path lengths
        - K-core decomposition
        """
        features = {}

        # Degree centrality
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        features["in_degree"] = in_degree
        features["out_degree"] = out_degree
        features["total_degree"] = {
            k: in_degree.get(k, 0) + out_degree.get(k, 0) for k in self.graph.nodes()
        }

        # Clustering coefficient
        clustering = nx.clustering(self.graph.to_undirected())
        features["clustering_coef"] = clustering

        # PageRank
        pagerank = nx.pagerank(self.graph)
        features["pagerank"] = pagerank

        # Betweenness centrality (expensive, sample for large graphs)
        # TODO: Implement sampling for large graphs
        # betweenness = nx.betweenness_centrality(self.graph)
        # features["betweenness"] = betweenness

        return pd.DataFrame(features)

    def _extract_transaction_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract transaction statistics per account.

        TODO: Add time-based features:
        - Transaction velocity (txns per hour/day)
        - Burstiness
        - Weekend/night activity patterns
        """
        features = {}

        # Source account features (outgoing)
        outgoing = transactions_df.groupby("source_account").agg(
            {
                "amount": ["sum", "mean", "std", "count"],
                "transaction_id": "count",
            }
        )
        outgoing.columns = ["out_" + "_".join(col).strip() for col in outgoing.columns.values]

        # Target account features (incoming)
        incoming = transactions_df.groupby("target_account").agg(
            {
                "amount": ["sum", "mean", "std", "count"],
                "transaction_id": "count",
            }
        )
        incoming.columns = ["in_" + "_".join(col).strip() for col in incoming.columns.values]

        # Merge
        features_df = pd.concat([outgoing, incoming], axis=1)

        return features_df

    def _extract_account_features(self, accounts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract static account features.

        TODO: Encode categorical features properly
        """
        features = accounts_df.set_index("account_id")[
            ["kyc_verified", "transaction_limit"]
        ].copy()

        # Encode account type
        features["is_business"] = (
            accounts_df.set_index("account_id")["account_type"].astype(str) == "AccountType.BUSINESS"
        ).astype(int)

        # Account age in days
        features["account_age_days"] = (
            pd.Timestamp.now() - accounts_df.set_index("account_id")["creation_date"]
        ).dt.days

        return features


def build_account_features(accounts_df: pd.DataFrame, tx_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build comprehensive account features from transactions.
    
    Args:
        accounts_df: Account data with account_id column
        tx_df: Transaction data with source_account, target_account, amount, timestamp columns
    
    Returns:
        features_df: DataFrame with one row per account and computed features
    """
    print("Building account features...")
    
    # Get all account IDs
    all_accounts = set(accounts_df['account_id'].unique())
    
    # Initialize features dict
    features = {account_id: {} for account_id in all_accounts}
    
    # Build directed graph for degree and triangle count
    print("  - Computing graph metrics...")
    G = nx.DiGraph()
    for _, row in tx_df.iterrows():
        src = row['source_account']
        tgt = row['target_account']
        amt = row['amount']
        
        if G.has_edge(src, tgt):
            G[src][tgt]['weight'] += amt
            G[src][tgt]['count'] += 1
        else:
            G.add_edge(src, tgt, weight=amt, count=1)
    
    # Compute degree features
    for account_id in all_accounts:
        in_deg = G.in_degree(account_id) if account_id in G else 0
        out_deg = G.out_degree(account_id) if account_id in G else 0
        
        features[account_id]['degree_in'] = in_deg
        features[account_id]['degree_out'] = out_deg
        
        # Weighted degree (sum of amounts)
        weighted_in = sum(G[u][account_id]['weight'] for u in G.predecessors(account_id)) if account_id in G else 0
        weighted_out = sum(G[account_id][v]['weight'] for v in G.successors(account_id)) if account_id in G else 0
        
        features[account_id]['weighted_in'] = weighted_in
        features[account_id]['weighted_out'] = weighted_out
        
        # out_in_amount_ratio
        if weighted_in > 0:
            features[account_id]['out_in_amount_ratio'] = weighted_out / weighted_in
        else:
            features[account_id]['out_in_amount_ratio'] = 0.0
    
    # Triangle count using undirected clustering
    print("  - Computing triangle counts...")
    G_undirected = G.to_undirected()
    clustering_coef = nx.clustering(G_undirected)
    
    for account_id in all_accounts:
        # Triangle count = clustering_coefficient * degree * (degree - 1) / 2
        if account_id in G_undirected:
            deg = G_undirected.degree(account_id)
            cc = clustering_coef.get(account_id, 0)
            triangle_count = cc * deg * (deg - 1) / 2 if deg > 1 else 0
        else:
            triangle_count = 0
        features[account_id]['triangle_count'] = triangle_count
    
    # Compute transaction-level features
    print("  - Computing transaction features...")
    
    # Parse timestamps if needed
    if not pd.api.types.is_datetime64_any_dtype(tx_df['timestamp']):
        tx_df = tx_df.copy()
        tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'])
    
    # Add hour of day for night transactions (e.g., 22:00 - 06:00)
    tx_df = tx_df.copy()
    tx_df['hour'] = tx_df['timestamp'].dt.hour
    tx_df['is_night'] = ((tx_df['hour'] >= 22) | (tx_df['hour'] < 6)).astype(int)
    
    # Check for cross-border transactions
    if 'is_international' in tx_df.columns:
        tx_df['is_cross_border'] = tx_df['is_international'].astype(int)
    elif 'is_cross_border' in tx_df.columns:
        tx_df['is_cross_border'] = tx_df['is_cross_border'].astype(int)
    else:
        # Default to 0 if not available
        tx_df['is_cross_border'] = 0
    
    # Process outgoing transactions
    for account_id in all_accounts:
        out_txs = tx_df[tx_df['source_account'] == account_id]
        in_txs = tx_df[tx_df['target_account'] == account_id]
        all_txs = pd.concat([out_txs, in_txs])
        
        if len(all_txs) == 0:
            # No transactions
            features[account_id]['unique_counterparties'] = 0
            features[account_id]['pct_cross_border'] = 0.0
            features[account_id]['pct_night_tx'] = 0.0
            features[account_id]['avg_tx_amount'] = 0.0
            features[account_id]['tx_amount_zscore'] = 0.0
            features[account_id]['mean_time_gap'] = 0.0
            features[account_id]['std_time_gap'] = 0.0
        else:
            # Unique counterparties
            counterparties = set(out_txs['target_account'].unique()) | set(in_txs['source_account'].unique())
            counterparties.discard(account_id)
            features[account_id]['unique_counterparties'] = len(counterparties)
            
            # Percentage cross-border
            features[account_id]['pct_cross_border'] = all_txs['is_cross_border'].mean()
            
            # Percentage night transactions
            features[account_id]['pct_night_tx'] = all_txs['is_night'].mean()
            
            # Average transaction amount
            features[account_id]['avg_tx_amount'] = all_txs['amount'].mean()
            
            # Transaction amount z-score (standardized within account)
            if len(all_txs) > 1 and all_txs['amount'].std() > 0:
                features[account_id]['tx_amount_zscore'] = (
                    (all_txs['amount'] - all_txs['amount'].mean()) / all_txs['amount'].std()
                ).abs().mean()
            else:
                features[account_id]['tx_amount_zscore'] = 0.0
            
            # Time gaps between consecutive transactions
            if len(all_txs) > 1:
                sorted_txs = all_txs.sort_values('timestamp')
                time_gaps = sorted_txs['timestamp'].diff().dt.total_seconds().dropna()
                
                if len(time_gaps) > 0:
                    features[account_id]['mean_time_gap'] = time_gaps.mean()
                    features[account_id]['std_time_gap'] = time_gaps.std() if len(time_gaps) > 1 else 0.0
                else:
                    features[account_id]['mean_time_gap'] = 0.0
                    features[account_id]['std_time_gap'] = 0.0
            else:
                features[account_id]['mean_time_gap'] = 0.0
                features[account_id]['std_time_gap'] = 0.0
    
    # Convert to DataFrame
    features_df = pd.DataFrame.from_dict(features, orient='index')
    features_df.index.name = 'account_id'
    features_df = features_df.reset_index()
    
    print(f"✓ Generated {len(features_df.columns) - 1} features for {len(features_df)} accounts")
    
    return features_df


def main():
    """Standalone feature engineering script."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate account features")
    parser.add_argument("--accounts", required=True, help="Path to accounts parquet")
    parser.add_argument("--tx", required=True, help="Path to transactions parquet")
    parser.add_argument("--out", required=True, help="Output path for features")
    parser.add_argument("--legacy", action="store_true", help="Use legacy TabularFeatureEngineer")

    args = parser.parse_args()

    # Load data
    print(f"Loading data...")
    accounts_df = pd.read_parquet(args.accounts)
    tx_df = pd.read_parquet(args.tx)
    
    if args.legacy:
        # Use legacy feature engineer
        print("Using legacy TabularFeatureEngineer...")
        engineer = TabularFeatureEngineer()
        features_df, labels = engineer.fit_transform(tx_df, accounts_df)
        features_df["label"] = labels
    else:
        # Use new build_account_features function
        features_df = build_account_features(accounts_df, tx_df)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Save
    features_df.to_parquet(args.out, index=False)
    print(f"\n✓ Saved features to {args.out}")


if __name__ == "__main__":
    main()

