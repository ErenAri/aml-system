# AML-X: Anti-Money Laundering Detection & Explanation System

**Build an explainable fraud detection system that analysts trust — in hours, not months.**

## 🎯 Overview

AML-X demonstrates end-to-end anomaly detection on synthetic transaction graphs with:
- **Two modeling tracks**: Tabular baseline (XGBoost + SHAP) and Graph Neural Networks (GraphSAGE/GAT + GNNExplainer)
- **Explainable AI**: SHAP values for tabular models, GNNExplainer for graph models
- **Interactive Dashboard**: Streamlit app with Plotly network visualizations
- **Production-ready**: Structured monorepo, reproducible pipelines, comprehensive metrics

## ⚡ 60-Second Value Prop

- **Catch more fraud with fewer false positives** using a hybrid of tabular ML and GNNs.
- **Explain every alert**: SHAP for features, GNNExplainer for subgraphs — investigators see the why.
- **Ready for demo and deployment**: Makefile pipelines, artifacts, metrics, Streamlit UI.
- **Bank-ready metrics**: PR-AUC, Precision@1%, FPR@95% to prove impact quickly.

## 🏗️ Architecture (ASCII)

```
          ┌─────────────────────┐        ┌────────────────────┐
          │  Data Simulator     │        │   Historical Data  │
          └─────────┬───────────┘        └─────────┬──────────┘
                    │                                │
                    ▼                                ▼
           ┌────────────────┐               ┌──────────────────┐
           │  Feature Eng   │<──────────────│  Feature Store    │ (future)
           │ (tabular+GNN)  │               └──────────────────┘
           └───────┬────────┘
                   │  artifacts/(features, graphs)
                   ▼
      ┌───────────────────────┐     ┌─────────────────────────┐
      │  Baseline (XGBoost)   │     │   GNN (GraphSAGE/GAT)   │
      └───────────┬───────────┘     └───────────┬─────────────┘
                  │ predictions, explns                      │
                  ├──────────────┐             ┌──────────────┘
                  ▼              ▼             ▼
        ┌────────────────┐  ┌──────────────┐  ┌─────────────────────┐
        │   SHAP (tab)   │  │ GNNExplainer │  │  Metrics (PR/ROC)   │
        └────────┬───────┘  └──────┬───────┘  └──────────┬──────────┘
                 │                  │                     │
                 ▼                  ▼                     ▼
                      ┌────────────────────────────────┐
                      │   Streamlit Analyst Dashboard   │
                      └────────────────────────────────┘
```

## 🚀 Quickstart (5 minutes)

```bash
# 1) Setup environment
make setup

# 2) Generate synthetic data
make simulate

# 3) Build features (tabular + graph)
make features

# 4) Train models
make train_baseline
make train_gnn

# 5) Generate explanations
make explain

# 6) Launch dashboard
make app
```

Visit `http://localhost:8501` to explore detected anomalies and explanations.

## 📦 Features

### Data Generation
- Synthetic transaction graph with normal and anomalous patterns
- Configurable fraud scenarios (circular transfers, structuring, rapid movement)
- Realistic account and transaction attributes

### Modeling Approaches

**Track A: Tabular Baseline**
- Feature engineering from graph structure (degree, clustering, PageRank)
- XGBoost classifier for anomaly detection
- SHAP explanations for individual predictions

**Track B: Graph Neural Networks**
- GraphSAGE/GAT architectures with PyTorch Geometric
- End-to-end learning on transaction graph
- GNNExplainer for subgraph explanations

### Evaluation
- **Banking metrics we optimize**:
  - **PR-AUC**: Area under Precision-Recall curve (robust to class imbalance)
  - **Precision@1%**: Precision among top 1% highest scores (analyst queue)
  - **FPR@95%**: False Positive Rate at 95% precision (regulatory-friendly)
  - Plus: Precision, Recall, F1, ROC-AUC, threshold analysis
  - CLI support: `python -m src.eval.metrics --csv artifacts/preds_gnn.csv`

### Visualization
- Interactive network graphs (Plotly)
- Explainability dashboards (SHAP waterfall, force plots)
- GNN attention/importance heatmaps

## 📁 Project Structure

```
aml-x/
├── data/            # Generated datasets (parquet/csv)
├── artifacts/       # Trained models, predictions
├── src/
│   ├── data/        # Data generation and schema
│   ├── features/    # Feature engineering
│   ├── models/      # ML models (baseline, GNN)
│   ├── explain/     # Explainability tools
│   ├── eval/        # Evaluation metrics
│   └── app/         # Streamlit dashboard
├── notebooks/       # Jupyter notebooks for exploration
└── requirements.txt
```

## 🧭 Path to Production

1. **Ingestion**: Kafka topics (`transactions`, `accounts`) with schema registry
2. **Feature Serving**: Stream processing → materialize rolling features (velocity, ratios)
3. **Scoring API**: Containerized model service (REST/gRPC) with A/B routing and canary
4. **Alerting**: Scores > threshold → publish to `alerts` topic with explanation payload
5. **Case Management**: Push to analyst queue (Actimize/Mantas/custom) with audit trail
6. **Feedback Loop**: Analyst decisions → label store → nightly retraining + drift checks

## 🛠️ Development

```bash
make lint          # Run code quality checks
make format        # Auto-format code with black
make test          # Run test suite (TODO)
```

## ✅ Responsible AI / Compliance

- **Auditability**: Persist model version, features, thresholds, and explanations with each alert
- **Traceable Explanations**: SHAP artifacts (CSV/JSON) and GNN subgraph masks are stored per decision
- **Governance**: Config-driven thresholds; change logs for approvals; reproducible training with seeds
- **Fairness & Drift**: Hooks for bias checks and data/model drift; alert when performance degrades
- **Human-in-the-Loop**: Analyst feedback captured and fed into retraining with approval workflows

## 📊 5-Minute Pitch

**Script (5 minutes)**

1) **Problem (45s)** — Banks spend millions triaging alerts. Rules miss evolving patterns and flood analysts with false positives. We need models that are both accurate and explainable.

2) **Solution (60s)** — AML-X fuses two worlds: a tabular baseline for stability and a GNN for relational patterns. Every alert comes with a reason: SHAP shows which features drove the score; GNNExplainer highlights the suspicious subgraph.

3) **Demo (90s)** — Generate data, train both models, and open the dashboard. We filter top-risk accounts, reveal SHAP features, and visualize the transaction ring around an entity with edge importance. Metrics show PR-AUC, Precision@1%, and FPR@95%.

4) **Impact (60s)** — In pilots, teams care about analyst efficiency. We optimize Precision@1% to prioritize queues and minimize FPR@95% for compliance. Result: fewer false alarms, faster investigations, and clearer audits.

5) **Moats & Fit (45s)** — Hybrid modeling, built-in explainability, clean path to Kafka + scoring API + case management. Extensible to sanctions screening, KYC, trade surveillance.

6) **Ask (30s)** — Data access and SME feedback to tune features and thresholds. We’ll integrate with your case management in week 2.

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

*Built for production ML systems that matter.*

