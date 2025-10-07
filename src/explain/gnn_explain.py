"""
GNN explainability using GNNExplainer.

Identifies important subgraphs and features for GNN predictions.
"""
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.data import Data
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import json
import plotly.graph_objects as go


class GNNExplainerWrapper:
    """Wrapper for GNNExplainer with visualization utilities."""

    def __init__(self, model, num_hops: int = 2):
        """
        Initialize GNN explainer.

        Args:
            model: Trained GNN model
            num_hops: Number of hops for explanation subgraph
        """
        self.model = model
        self.num_hops = num_hops

        # Initialize explainer
        self.explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="raw",
            ),
        )

    def explain_node(self, data: Data, node_idx: int) -> dict:
        """
        Explain prediction for a single node.

        Args:
            data: PyG Data object
            node_idx: Index of node to explain

        Returns:
            Dictionary with explanation (node/edge masks)
        """
        # Get explanation
        explanation = self.explainer(
            x=data.x,
            edge_index=data.edge_index,
            index=node_idx,
        )

        # Extract masks
        node_mask = explanation.node_mask
        edge_mask = explanation.edge_mask

        return {
            "node_idx": node_idx,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
            "prediction": self.model(data.x, data.edge_index)[node_idx].argmax().item(),
            "explanation": explanation,
        }

    def visualize_explanation(
        self,
        data: Data,
        node_idx: int,
        node_id_map: dict,
        save_path: Optional[str] = None,
    ):
        """
        Visualize explanation subgraph.

        Args:
            data: PyG Data object
            node_idx: Node to explain
            node_id_map: Mapping from node indices to account IDs
            save_path: Path to save visualization
        """
        explanation = self.explain_node(data, node_idx)
        edge_mask = explanation["edge_mask"].cpu().detach().numpy()

        # Build subgraph from important edges
        G = nx.DiGraph()
        edge_index = data.edge_index.cpu().numpy()

        # Add edges weighted by importance
        threshold = edge_mask.mean()
        for i, (src, tgt) in enumerate(edge_index.T):
            if edge_mask[i] > threshold:
                G.add_edge(src, tgt, weight=edge_mask[i])

        # Add target node if isolated
        if node_idx not in G.nodes():
            G.add_node(node_idx)

        # Visualize
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(G, seed=42)

        # Node colors (red = target, blue = neighbors)
        node_colors = ["red" if n == node_idx else "lightblue" for n in G.nodes()]

        # Edge widths based on importance
        edge_widths = [G[u][v]["weight"] * 5 for u, v in G.edges()]

        nx.draw(
            G,
            pos,
            node_color=node_colors,
            width=edge_widths,
            with_labels=True,
            node_size=500,
            font_size=8,
            arrows=True,
        )

        plt.title(f"GNN Explanation for Node {node_idx}\nPrediction: {explanation['prediction']}")

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            print(f"✓ Explanation visualization saved to {save_path}")
        else:
            plt.show()

        plt.close()

        return G

    def get_important_neighbors(
        self, data: Data, node_idx: int, top_k: int = 5
    ) -> list:
        """
        Get top-k most important neighbors for a node's prediction.

        Returns list of (neighbor_idx, importance_score) tuples.
        """
        explanation = self.explain_node(data, node_idx)
        edge_mask = explanation["edge_mask"].cpu().detach().numpy()
        edge_index = data.edge_index.cpu().numpy()

        # Find edges connected to target node
        neighbor_importance = []
        for i, (src, tgt) in enumerate(edge_index.T):
            if src == node_idx or tgt == node_idx:
                neighbor = tgt if src == node_idx else src
                neighbor_importance.append((neighbor, edge_mask[i]))

        # Sort by importance
        neighbor_importance.sort(key=lambda x: x[1], reverse=True)

        return neighbor_importance[:top_k]


def explain_subgraph(
    model, data: Data, suspicious_nodes: list, save_path: Optional[str] = None
):
    """
    Explain a subgraph containing multiple suspicious nodes.

    Useful for understanding connected fraud rings.

    TODO: Implement multi-node explanation
    - Identify common patterns across suspicious nodes
    - Find shared important features
    - Detect fraud ring structures
    """
    # TODO: Implement subgraph explanation
    pass


def attention_analysis(model, data: Data, node_idx: int):
    """
    Analyze attention weights for GAT models.

    Shows which neighbors the model focuses on.

    TODO: Extract and visualize attention weights from GAT layers
    """
    # TODO: Implement attention extraction and visualization
    pass


# --- New GNNExplainer-based explain_node implementation ---

# Feature names for node features (must match GraphFeatureEngineer order)
NODE_FEATURE_NAMES = [
    "is_business",
    "kyc_verified",
    "transaction_limit",
    "risk_score",
]


def explain_node(
    model,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    node_id: int,
    epochs: int = 200,
    edge_attr: Optional[torch.Tensor] = None,
    feature_names: Optional[List[str]] = None,
) -> Dict:
    """
    Explain a single node prediction using GNNExplainer.

    Args:
        model: Trained GNN model (GraphSAGE, GAT, etc.)
        x: Node feature matrix [num_nodes, num_features]
        edge_index: Edge indices [2, num_edges]
        node_id: Target node index to explain
        epochs: Number of optimization epochs for GNNExplainer
        edge_attr: Optional edge features
        feature_names: Optional list of feature names for textual reason

    Returns:
        Dictionary with:
            - node_id: Target node ID
            - subgraph_nodes: List of important neighbor nodes
            - subgraph_edges: List of (src, tgt, importance) tuples
            - feature_mask: Array of feature importances
            - textual_reason: Human-readable explanation string
            - prediction: Model prediction for this node
            - node_mask: Importance scores for all nodes
            - edge_mask: Importance scores for all edges
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    edge_index = edge_index.to(device)
    if edge_attr is not None:
        edge_attr = edge_attr.to(device)

    # Initialize explainer
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="raw",
        ),
    )

    # Get explanation
    explanation = explainer(
        x=x,
        edge_index=edge_index,
        index=node_id,
        edge_attr=edge_attr,
    )

    # Extract masks
    node_mask = explanation.node_mask.cpu().detach().numpy() if explanation.node_mask is not None else None
    edge_mask = explanation.edge_mask.cpu().detach().numpy()

    # Get model prediction
    with torch.no_grad():
        out = model(x, edge_index)
        pred_class = out[node_id].argmax().item()
        pred_score = torch.softmax(out[node_id], dim=0)[1].item()

    # Extract important subgraph
    threshold = edge_mask.mean() + 0.5 * edge_mask.std()
    important_edges = []
    subgraph_nodes = set([node_id])

    edge_index_np = edge_index.cpu().numpy()
    for i, (src, tgt) in enumerate(edge_index_np.T):
        if edge_mask[i] > threshold:
            important_edges.append((int(src), int(tgt), float(edge_mask[i])))
            subgraph_nodes.add(int(src))
            subgraph_nodes.add(int(tgt))

    # Feature importance (average across node mask if available, else use target node features)
    if node_mask is not None and node_mask.ndim == 2:
        # node_mask is [num_nodes, num_features]
        feature_importance = node_mask[node_id]
    elif node_mask is not None and node_mask.ndim == 1:
        # Single feature mask for target node
        feature_importance = node_mask
    else:
        # Fallback: uniform importance
        feature_importance = np.ones(x.shape[1]) / x.shape[1]

    # Generate textual reason
    if feature_names is None:
        feature_names = NODE_FEATURE_NAMES[:x.shape[1]] if x.shape[1] <= len(NODE_FEATURE_NAMES) else [f"feat_{i}" for i in range(x.shape[1])]

    textual_reason = _generate_textual_reason(
        node_id=node_id,
        feature_importance=feature_importance,
        feature_names=feature_names,
        node_features=x[node_id].cpu().numpy(),
        num_neighbors=len(subgraph_nodes) - 1,
        pred_score=pred_score,
    )

    return {
        "node_id": int(node_id),
        "subgraph_nodes": sorted(list(subgraph_nodes)),
        "subgraph_edges": important_edges,
        "feature_mask": feature_importance.tolist() if isinstance(feature_importance, np.ndarray) else feature_importance,
        "textual_reason": textual_reason,
        "prediction": pred_class,
        "prediction_score": pred_score,
        "node_mask": node_mask.tolist() if node_mask is not None else None,
        "edge_mask": edge_mask.tolist(),
    }


def _generate_textual_reason(
    node_id: int,
    feature_importance: np.ndarray,
    feature_names: List[str],
    node_features: np.ndarray,
    num_neighbors: int,
    pred_score: float,
) -> str:
    """Generate human-readable explanation from feature importance."""
    # Find top contributing features
    top_k = min(3, len(feature_importance))
    top_indices = np.argsort(np.abs(feature_importance))[-top_k:][::-1]

    reasons = []

    # Feature-based reasons
    for idx in top_indices:
        feat_name = feature_names[idx]
        feat_val = node_features[idx]
        importance = feature_importance[idx]

        if importance > 0.1:  # Only include significant features
            if "risk" in feat_name.lower():
                reasons.append(f"high risk score ({feat_val:.2f})")
            elif "cross" in feat_name.lower() or "international" in feat_name.lower():
                if feat_val > 0.5:
                    reasons.append("frequent cross-border transactions")
            elif "business" in feat_name.lower():
                if feat_val > 0.5:
                    reasons.append("business account")
            elif "kyc" in feat_name.lower():
                if feat_val < 0.5:
                    reasons.append("unverified KYC")
            elif "out_in" in feat_name.lower() or "ratio" in feat_name.lower():
                if abs(feat_val) > 1.0:
                    reasons.append(f"abnormal out/in ratio ({feat_val:.2f})")

    # Graph structure reason
    if num_neighbors > 5:
        reasons.append(f"connected to {num_neighbors} flagged nodes")
    elif num_neighbors > 0:
        reasons.append(f"{num_neighbors}-hop link to suspicious cluster")

    # Combine reasons
    if not reasons:
        reasons.append(f"anomalous pattern (score: {pred_score:.2f})")

    return "; ".join(reasons).capitalize() + "."


def subgraph_to_plotly(
    subgraph_nodes: List[int],
    subgraph_edges: List[Tuple[int, int, float]],
    node_id: int,
    node_scores: Optional[Dict[int, float]] = None,
    tx_df: Optional[pd.DataFrame] = None,
    id_map: Optional[Dict[int, str]] = None,
) -> go.Figure:
    """
    Convert subgraph to plotly interactive visualization.

    Args:
        subgraph_nodes: List of node indices in subgraph
        subgraph_edges: List of (src, tgt, importance) tuples
        node_id: Target node being explained
        node_scores: Optional dict mapping node_idx -> risk_score for coloring
        tx_df: Optional transaction dataframe to get edge amounts
        id_map: Optional mapping from node_idx -> account_id

    Returns:
        Plotly Figure with interactive graph visualization
    """
    # Build NetworkX graph for layout
    G = nx.DiGraph()
    for node in subgraph_nodes:
        G.add_node(node)

    edge_data = {}
    for src, tgt, importance in subgraph_edges:
        G.add_edge(src, tgt, weight=importance)
        edge_data[(src, tgt)] = importance

    # Compute layout
    pos = nx.spring_layout(G, seed=42, k=2)

    # Create edge traces
    edge_traces = []
    for (src, tgt), importance in edge_data.items():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]

        # Get edge amount if available
        edge_width = 1 + importance * 5  # Scale by importance
        if tx_df is not None and id_map is not None:
            src_acc = id_map.get(src, src)
            tgt_acc = id_map.get(tgt, tgt)
            # Find matching transaction
            matching_tx = tx_df[
                (tx_df["source_account"] == src_acc) & (tx_df["target_account"] == tgt_acc)
            ]
            if len(matching_tx) > 0:
                total_amount = matching_tx["amount"].sum()
                edge_width = 1 + np.log1p(total_amount) * 0.5

        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode="lines",
            line=dict(width=edge_width, color="lightgray"),
            hoverinfo="none",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_sizes = []

    for node in subgraph_nodes:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Color by risk score if available
        if node_scores and node in node_scores:
            score = node_scores[node]
            node_colors.append(score)
        else:
            node_colors.append(0.5)

        # Size: larger for target node
        if node == node_id:
            node_sizes.append(30)
        else:
            node_sizes.append(15)

        # Text label
        label = id_map.get(node, str(node)) if id_map else str(node)
        node_text.append(f"Node: {label}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Risk Score"),
            line=dict(width=2, color="white"),
        ),
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f"Explanation Subgraph for Node {node_id}",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
        ),
    )

    return fig


def cache_explanation(explanation: Dict, artifacts_dir: Path) -> Path:
    """
    Cache explanation to artifacts/explanations/node_<id>.json.

    Args:
        explanation: Explanation dict from explain_node()
        artifacts_dir: Base artifacts directory

    Returns:
        Path to saved JSON file
    """
    exp_dir = artifacts_dir / "explanations"
    exp_dir.mkdir(parents=True, exist_ok=True)

    node_id = explanation["node_id"]
    out_path = exp_dir / f"node_{node_id}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(explanation, f, ensure_ascii=False, indent=2)

    return out_path


def load_cached_explanation(node_id: int, artifacts_dir: Path) -> Optional[Dict]:
    """Load cached explanation if it exists."""
    exp_path = artifacts_dir / "explanations" / f"node_{node_id}.json"
    if exp_path.exists():
        with open(exp_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    print("GNN explainability tools - import and use in other modules")

