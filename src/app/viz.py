"""
Visualization utilities for transaction networks.

Uses Plotly for interactive network graphs.
"""
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List


def create_network_graph(
    transactions_df: pd.DataFrame,
    accounts_df: Optional[pd.DataFrame] = None,
    suspicious_accounts: Optional[List[str]] = None,
    max_nodes: int = 100,
) -> go.Figure:
    """
    Create interactive Plotly network graph from transactions.

    Args:
        transactions_df: Transaction data
        accounts_df: Account data (optional, for node attributes)
        suspicious_accounts: List of suspicious account IDs to highlight
        max_nodes: Maximum number of nodes to display

    Returns:
        Plotly Figure object
    """
    # Build NetworkX graph
    G = nx.DiGraph()

    for _, row in transactions_df.iterrows():
        G.add_edge(
            row["source_account"],
            row["target_account"],
            weight=row["amount"],
        )

    # Limit to largest connected component if too large
    if len(G.nodes()) > max_nodes:
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

        # Further limit if still too large
        if len(G.nodes()) > max_nodes:
            # Take highest degree nodes
            degrees = dict(G.degree())
            top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_nodes]
            G = G.subgraph(top_nodes).copy()

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Create edges
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=0.5, color="#888"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    suspicious_accounts = suspicious_accounts or []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        # Node info
        degree = G.degree(node)
        node_text.append(f"Account: {node}<br>Degree: {degree}")

        # Color: red if suspicious, blue otherwise
        if node in suspicious_accounts:
            node_color.append("red")
        else:
            node_color.append("lightblue")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            color=node_color,
            size=10,
            line=dict(width=1, color="white"),
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])

    fig.update_layout(
        title="Transaction Network Graph",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )

    return fig


def create_subgraph_explanation(
    transactions_df: pd.DataFrame,
    center_account: str,
    hop_radius: int = 2,
    edge_importance: Optional[dict] = None,
) -> go.Figure:
    """
    Create subgraph visualization for GNN explanation.

    Args:
        transactions_df: Transaction data
        center_account: Account to explain
        hop_radius: Number of hops for neighborhood
        edge_importance: Dictionary mapping edge -> importance score

    Returns:
        Plotly Figure with highlighted important edges
    """
    # Build graph
    G = nx.DiGraph()
    for _, row in transactions_df.iterrows():
        G.add_edge(row["source_account"], row["target_account"], weight=row["amount"])

    # Extract subgraph (ego graph)
    if center_account not in G.nodes():
        # Return empty figure
        return go.Figure()

    subgraph = nx.ego_graph(G.to_undirected(), center_account, radius=hop_radius)

    # Layout
    pos = nx.spring_layout(subgraph, seed=42)

    # Edges with importance coloring
    edge_traces = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]

        # Edge importance
        importance = edge_importance.get(edge, 0.5) if edge_importance else 0.5
        width = importance * 5
        color = f"rgba(255, 0, 0, {importance})"

        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color=color),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Nodes
    node_x = [pos[node][0] for node in subgraph.nodes()]
    node_y = [pos[node][1] for node in subgraph.nodes()]
    node_text = [f"Account: {node}" for node in subgraph.nodes()]
    node_color = ["red" if node == center_account else "lightblue" for node in subgraph.nodes()]

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[node if node == center_account else "" for node in subgraph.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            color=node_color,
            size=[20 if node == center_account else 10 for node in subgraph.nodes()],
            line=dict(width=2, color="white"),
        ),
        showlegend=False,
    )

    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])

    fig.update_layout(
        title=f"Explanation Subgraph for {center_account}",
        showlegend=False,
        hovermode="closest",
        margin=dict(b=0, l=0, r=0, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600,
    )

    return fig


if __name__ == "__main__":
    print("Visualization utilities - import and use in Streamlit app")

