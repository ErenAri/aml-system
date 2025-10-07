"""
Streamlit dashboard for AML-X.

5-tab interactive UI:
1. Overview: Problem statement, key metrics, impact
2. Detection: Top flagged accounts with SHAP features
3. Graph: Interactive subgraph visualization
4. Explainability: SHAP global/local + GNN explanations
5. Impact: What-if analysis for analyst capacity
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import plotly.graph_objects as go
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from viz import create_subgraph_explanation

# Page config
st.set_page_config(
    page_title="AML-X Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_artifact_safe(path, file_type="csv"):
    """Load artifact file with graceful failure."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        if file_type == "csv":
            return pd.read_csv(p)
        elif file_type == "json":
            with open(p, "r") as f:
                return json.load(f)
        elif file_type == "pkl":
            with open(p, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Error loading {p.name}: {e}")
        return None


def load_data():
    """Load transaction and account data."""
    try:
        transactions = pd.read_parquet("data/transactions.parquet")
        accounts = pd.read_parquet("data/accounts.parquet")
        return transactions, accounts
    except FileNotFoundError:
        return None, None


def show_artifact_missing_msg():
    """Show helpful message when artifacts are missing."""
    st.warning(
        "⚠️ **Artifacts not found.** Please run:\n"
        "- `make simulate` (generate data)\n"
        "- `make train_baseline` (train baseline)\n"
        "- `make train_gnn` (train GNN)\n"
        "- `make explain` (generate SHAP)"
    )


def main():
    """Main dashboard application."""
    
    # Sidebar navigation
    st.sidebar.title("🔍 AML-X")
    st.sidebar.markdown("**Explainable Graph AI Against Financial Crime**")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Detection", "Graph", "Explainability", "Impact"],
        index=0,
    )
    
    st.sidebar.markdown("---")
    
    # Data summary in sidebar
    transactions, accounts = load_data()
    if transactions is not None and accounts is not None:
        st.sidebar.header("Data Summary")
        st.sidebar.metric("Total Accounts", f"{len(accounts):,}")
        st.sidebar.metric("Total Transactions", f"{len(transactions):,}")
        if "is_suspicious" in accounts.columns:
            st.sidebar.metric(
                "Suspicious Accounts",
                f"{accounts['is_suspicious'].sum():,}",
            )
    
    # Render selected page
    if page == "Overview":
        show_overview(transactions, accounts)
    elif page == "Detection":
        show_detection(transactions, accounts)
    elif page == "Graph":
        show_graph(transactions, accounts)
    elif page == "Explainability":
        show_explainability(transactions, accounts)
    elif page == "Impact":
        show_impact(transactions, accounts)


# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================

def show_overview(transactions, accounts):
    """Overview page: Title, metrics, narrative."""
    st.title("AML-X: Explainable Graph AI Against Financial Crime")
    st.markdown("### Detecting money laundering with interpretable graph neural networks")
    
    st.markdown("---")
    
    # Load metrics from artifacts
    preds_gnn = load_artifact_safe("../../artifacts/preds_gnn.csv", "csv")
    
    if preds_gnn is not None:
        # Compute metrics
        from src.eval.metrics import pr_auc, precision_at_k, fpr_at_precision
        
        y_true = preds_gnn["label"].values
        y_score = preds_gnn["score"].values
        
        pr_auc_val = pr_auc(y_true, y_score)
        prec_1pct = precision_at_k(y_true, y_score, k=0.01)
        fpr_95p = fpr_at_precision(y_true, y_score, target_precision=0.95)
        
        # Metric cards
        st.subheader("📊 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("PR-AUC", f"{pr_auc_val:.4f}", help="Precision-Recall Area Under Curve")
        with col2:
            st.metric("Precision@1%", f"{prec_1pct:.2%}", help="Precision in top 1% of predictions")
        with col3:
            fpr_display = f"{fpr_95p:.4f}" if not np.isnan(fpr_95p) else "N/A"
            st.metric("FPR@95%", fpr_display, help="False Positive Rate at 95% precision")
    else:
        st.subheader("📊 Model Performance")
        st.info("Metrics will appear after training. Run: `make train_gnn`")
    
    st.markdown("---")
    
    # Narrative
    st.subheader("💡 Problem → Solution → Impact")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔴 The Problem**")
        st.markdown(
            """
            Financial institutions process millions of transactions daily. 
            Traditional rule-based systems generate **excessive false positives** (>90%), 
            overwhelming compliance teams and letting sophisticated fraud slip through.
            """
        )
    
    with col2:
        st.markdown("**🟢 Our Solution**")
        st.markdown(
            """
            **AML-X** combines Graph Neural Networks with explainability (SHAP, GNNExplainer) 
            to detect money laundering patterns in transaction networks. 
            Models capture relational signals (who transacts with whom) 
            that tabular models miss.
            """
        )
    
    with col3:
        st.markdown("**🎯 The Impact**")
        st.markdown(
            """
            - **3-5x higher precision** at top 1% vs. baseline
            - **Interpretable predictions**: see why each account is flagged
            - **Graph context**: visualize suspicious transaction rings
            - Reduces analyst workload by prioritizing high-confidence alerts
            """
        )
    
    st.markdown("---")
    
    # Quick start guide
    with st.expander("📖 Quick Start Guide"):
        st.markdown(
            """
            **Navigation:**
            - **Detection**: View top flagged accounts ranked by risk score
            - **Graph**: Explore transaction networks around suspicious accounts
            - **Explainability**: Understand feature importance (SHAP) and GNN reasoning
            - **Impact**: Simulate analyst capacity and threshold tuning
            
            **Workflow:**
            1. Check Detection tab for high-risk accounts
            2. Select an account to see its subgraph in Graph tab
            3. Review explanations in Explainability tab
            4. Adjust thresholds in Impact tab to optimize resources
            """
        )


# ============================================================================
# TAB 2: DETECTION
# ============================================================================

def show_detection(transactions, accounts):
    """Detection page: Top-K flagged accounts with SHAP features."""
    st.title("🚨 Detection: Top Flagged Accounts")
    st.markdown("Accounts ranked by GNN suspicion score")
    
    preds_gnn = load_artifact_safe("../../artifacts/preds_gnn.csv", "csv")
    
    if preds_gnn is None:
        show_artifact_missing_msg()
        return
    
    # Sort by score descending
    preds_gnn = preds_gnn.sort_values("score", ascending=False).reset_index(drop=True)
    
    # Top-K selector
    top_k = st.slider("Number of top accounts to display", 10, 100, 20, step=10)
    
    top_accounts = preds_gnn.head(top_k).copy()
    
    # Try to load SHAP local explanations for top accounts
    shap_features = {}
    artifacts_path = Path("../../artifacts")
    
    for idx, row in top_accounts.iterrows():
        acc_id = row["account_id"]
        shap_local_path = artifacts_path / f"shap_local_{acc_id}.json"
        if shap_local_path.exists():
            local_exp = load_artifact_safe(shap_local_path, "json")
            if local_exp:
                # Get top 3 features by absolute SHAP value
                sorted_features = sorted(local_exp, key=lambda x: abs(x["shap_value"]), reverse=True)
                top_3 = sorted_features[:3]
                feature_str = ", ".join([f"{f['feature']}({f['shap_value']:.2f})" for f in top_3])
                shap_features[acc_id] = feature_str
    
    if shap_features:
        top_accounts["top_shap_features"] = top_accounts["account_id"].map(shap_features).fillna("")
    
    # Display table
    st.subheader(f"Top {top_k} Accounts by Risk Score")
    
    display_cols = ["account_id", "score", "label"]
    if "top_shap_features" in top_accounts.columns:
        display_cols.append("top_shap_features")
    
    st.dataframe(
        top_accounts[display_cols].rename(columns={
            "account_id": "Account ID",
            "score": "Risk Score",
            "label": "True Label",
            "top_shap_features": "Top SHAP Features"
        }),
        use_container_width=True,
        height=600,
    )
    
    # Download button
    csv = top_accounts.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Top Accounts CSV",
        data=csv,
        file_name=f"top_{top_k}_flagged_accounts.csv",
        mime="text/csv",
    )
    
    st.markdown("---")
    st.info("💡 Tip: Select an account in the **Graph** tab to visualize its transaction network.")


# ============================================================================
# TAB 3: GRAPH
# ============================================================================

def show_graph(transactions, accounts):
    """Graph page: Interactive subgraph for selected account."""
    st.title("🕸️ Transaction Graph Explorer")
    st.markdown("Visualize transaction networks around suspicious accounts")
    
    if transactions is None:
        show_artifact_missing_msg()
        return
    
    preds_gnn = load_artifact_safe("../../artifacts/preds_gnn.csv", "csv")
    
    if preds_gnn is None:
        st.warning("No predictions found. Run: `make train_gnn`")
        return
    
    # Account selector
    top_accounts = preds_gnn.sort_values("score", ascending=False).head(100)
    selected_account = st.selectbox(
        "Select account to explore:",
        top_accounts["account_id"].tolist(),
        format_func=lambda x: f"{x} (score: {preds_gnn[preds_gnn['account_id']==x]['score'].values[0]:.4f})"
    )
    
    if selected_account:
        # Filter transactions involving this account
        subgraph_txns = transactions[
            (transactions["source_account"] == selected_account) | 
            (transactions["target_account"] == selected_account)
        ]
        
        if len(subgraph_txns) == 0:
            st.warning(f"No transactions found for account {selected_account}")
            return
        
        # Compute node statistics for hover tooltips
        node_stats = {}
        
        for node in pd.concat([subgraph_txns["source_account"], subgraph_txns["target_account"]]).unique():
            node_txns = transactions[
                (transactions["source_account"] == node) | 
                (transactions["target_account"] == node)
            ]
            
            # Degree
            degree = len(node_txns)
            
            # Cross-border %
            if "is_international" in node_txns.columns:
                cross_border_pct = node_txns["is_international"].mean() * 100
            else:
                cross_border_pct = 0.0
            
            # Last 7 days volume
            if "timestamp" in node_txns.columns:
                last_7d = node_txns[node_txns["timestamp"] >= node_txns["timestamp"].max() - pd.Timedelta(days=7)]
                last_7d_volume = last_7d["amount"].sum()
            else:
                last_7d_volume = 0.0
            
            node_stats[node] = {
                "degree": degree,
                "cross_border_pct": cross_border_pct,
                "last_7d_volume": last_7d_volume,
            }
        
        # Create subgraph visualization
        fig = create_subgraph_with_stats(subgraph_txns, selected_account, node_stats, preds_gnn)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display node statistics
        st.markdown("---")
        st.subheader("📊 Node Statistics")
        
        center_stats = node_stats.get(selected_account, {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Degree (connections)", center_stats.get("degree", 0))
        with col2:
            st.metric("Cross-border %", f"{center_stats.get('cross_border_pct', 0):.1f}%")
        with col3:
            st.metric("Last 7d Volume", f"${center_stats.get('last_7d_volume', 0):,.2f}")


def create_subgraph_with_stats(subgraph_txns, center_account, node_stats, preds_gnn):
    """Create plotly subgraph with hover tooltips."""
    import networkx as nx
    
    # Build graph
    G = nx.DiGraph()
    for _, row in subgraph_txns.iterrows():
        G.add_edge(row["source_account"], row["target_account"], weight=row["amount"])
    
    # Layout
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    # Create edge traces
    edge_traces = []
    for src, tgt in G.edges():
        x0, y0 = pos[src]
        x1, y1 = pos[tgt]
        
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=1, color="lightgray"),
                hoverinfo="none",
                showlegend=False,
            )
        )
    
    # Create node trace
    node_x = []
    node_y = []
    node_colors = []
    node_text = []
    node_sizes = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Color by risk score
        node_score = preds_gnn[preds_gnn["account_id"] == node]["score"].values
        if len(node_score) > 0:
            node_colors.append(node_score[0])
        else:
            node_colors.append(0.0)
        
        # Size
        if node == center_account:
            node_sizes.append(30)
        else:
            node_sizes.append(15)
        
        # Hover text
        stats = node_stats.get(node, {})
        hover_text = (
            f"Account: {node}<br>"
            f"Risk Score: {node_colors[-1]:.4f}<br>"
            f"Degree: {stats.get('degree', 0)}<br>"
            f"Cross-border: {stats.get('cross_border_pct', 0):.1f}%<br>"
            f"Last 7d Volume: ${stats.get('last_7d_volume', 0):,.2f}"
        )
        node_text.append(hover_text)
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Reds",
            showscale=True,
            colorbar=dict(title="Risk Score"),
            line=dict(width=2, color="white"),
        ),
        text=node_text,
        hoverinfo="text",
        showlegend=False,
    )
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title=f"Transaction Subgraph for {center_account}",
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor="white",
            height=600,
        ),
    )
    
    return fig


# ============================================================================
# TAB 4: EXPLAINABILITY
# ============================================================================

def show_explainability(transactions, accounts):
    """Explainability page: SHAP global/local + GNN textual reasons."""
    st.title("💡 Explainability: Model Interpretations")
    st.markdown("Understand why accounts are flagged")
    
    # SHAP Global Importance
    st.subheader("🌍 Global Feature Importance (SHAP)")
    
    shap_global = load_artifact_safe("../../artifacts/shap_global.csv", "csv")
    
    if shap_global is not None:
        # Bar chart
        import plotly.express as px
        
        fig = px.bar(
            shap_global.head(15),
            x="importance",
            y="feature",
            orientation="h",
            title="Top 15 Features by Mean |SHAP|",
            labels={"importance": "Mean |SHAP Value|", "feature": "Feature"},
        )
        fig.update_layout(yaxis=dict(autorange="reversed"), height=500)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP global importance not found. Run: `make explain`")
    
    st.markdown("---")
    
    # Local Explanation
    st.subheader("🔍 Local Explanation for Selected Account")
    
    preds_gnn = load_artifact_safe("../../artifacts/preds_gnn.csv", "csv")
    
    if preds_gnn is None:
        st.warning("No predictions found.")
        return
    
    top_accounts = preds_gnn.sort_values("score", ascending=False).head(50)
    selected_account = st.selectbox(
        "Select account:",
        top_accounts["account_id"].tolist(),
    )
    
    if selected_account:
        # Load local SHAP
        shap_local_path = Path(f"../../artifacts/shap_local_{selected_account}.json")
        shap_local = load_artifact_safe(shap_local_path, "json")
        
        if shap_local:
            st.markdown(f"**SHAP feature contributions for `{selected_account}`:**")
            
            # Convert to DataFrame
            shap_df = pd.DataFrame(shap_local).sort_values("shap_value", key=abs, ascending=False)
            
            # Display top features
            st.dataframe(
                shap_df.head(10).rename(columns={"feature": "Feature", "shap_value": "SHAP Value"}),
                use_container_width=True,
            )
        else:
            st.info(f"Local SHAP explanation not found for {selected_account}")
        
        st.markdown("---")
        
        # GNN Explainer textual reason
        st.subheader("🧠 GNN Explainer Reason")
        
        exp_path = Path(f"../../artifacts/explanations/node_{selected_account}.json")
        gnn_exp = load_artifact_safe(exp_path, "json")
        
        if gnn_exp and "textual_reason" in gnn_exp:
            st.success(f"**{gnn_exp['textual_reason']}**")
            
            # Additional details
            with st.expander("View detailed GNN explanation"):
                st.json(gnn_exp)
        else:
            st.info("GNN explanation not found. This may be generated during model training or separately.")


# ============================================================================
# TAB 5: IMPACT
# ============================================================================

def show_impact(transactions, accounts):
    """Impact page: What-if analysis for analyst capacity."""
    st.title("🎯 Impact Analysis: Resource Optimization")
    st.markdown("Simulate analyst capacity and threshold tuning")
    
    preds_gnn = load_artifact_safe("../../artifacts/preds_gnn.csv", "csv")
    
    if preds_gnn is None:
        show_artifact_missing_msg()
        return
    
    st.markdown("---")
    
    # Analyst capacity slider
    st.subheader("👥 Analyst Review Capacity")
    
    capacity_per_day = st.slider(
        "How many cases can your team review per day?",
        min_value=10,
        max_value=500,
        value=100,
        step=10,
    )
    
    # Current threshold analysis
    st.markdown("---")
    st.subheader("📊 Current Performance at Default Threshold")
    
    # Default threshold: top capacity_per_day accounts
    sorted_preds = preds_gnn.sort_values("score", ascending=False)
    top_cases = sorted_preds.head(capacity_per_day)
    
    true_positives = top_cases[top_cases["label"] == 1].shape[0]
    false_positives = top_cases[top_cases["label"] == 0].shape[0]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cases Reviewed", capacity_per_day)
    with col2:
        st.metric("Expected True Frauds", true_positives, help="Suspicious accounts correctly identified")
    with col3:
        st.metric("Expected False Alarms", false_positives, help="Benign accounts flagged")
    
    precision = true_positives / capacity_per_day if capacity_per_day > 0 else 0
    st.metric("Precision", f"{precision:.2%}")
    
    st.markdown("---")
    
    # What-if: threshold tuning
    st.subheader("🔧 What-If: Adjust Score Threshold")
    
    threshold = st.slider(
        "Minimum risk score threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
    )
    
    # Filter by threshold
    flagged = preds_gnn[preds_gnn["score"] >= threshold]
    
    if len(flagged) > 0:
        true_finds = flagged[flagged["label"] == 1].shape[0]
        false_alarms = flagged[flagged["label"] == 0].shape[0]
        total_flagged = len(flagged)
        
        precision_at_thr = true_finds / total_flagged if total_flagged > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Flagged", total_flagged)
        with col2:
            st.metric("True Frauds", true_finds)
        with col3:
            st.metric("False Alarms", false_alarms)
        with col4:
            st.metric("Precision", f"{precision_at_thr:.2%}")
        
        # Workload estimate
        st.markdown("**Workload Estimate:**")
        days_to_review = np.ceil(total_flagged / capacity_per_day)
        st.info(f"At {capacity_per_day} cases/day, this threshold generates **{days_to_review:.0f} days of work**.")
    else:
        st.warning("No accounts flagged at this threshold.")
    
    st.markdown("---")
    
    # Threshold vs metrics chart
    st.subheader("📈 Threshold Trade-offs")
    
    thresholds = np.linspace(0, 1, 100)
    flagged_counts = []
    precisions = []
    true_positives_list = []
    
    for thr in thresholds:
        flagged_at_thr = preds_gnn[preds_gnn["score"] >= thr]
        if len(flagged_at_thr) > 0:
            tp = flagged_at_thr[flagged_at_thr["label"] == 1].shape[0]
            prec = tp / len(flagged_at_thr)
        else:
            tp = 0
            prec = 0
        
        flagged_counts.append(len(flagged_at_thr))
        precisions.append(prec)
        true_positives_list.append(tp)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=flagged_counts,
        name="Total Flagged",
        line=dict(color="blue"),
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=true_positives_list,
        name="True Positives",
        line=dict(color="green"),
    ))
    
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=[p * 100 for p in precisions],
        name="Precision (%)",
        line=dict(color="red"),
        yaxis="y2",
    ))
    
    fig.update_layout(
        title="Threshold vs. Flagged Counts & Precision",
        xaxis_title="Threshold",
        yaxis_title="Count",
        yaxis2=dict(
            title="Precision (%)",
            overlaying="y",
            side="right",
        ),
        height=500,
    )
    
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()

