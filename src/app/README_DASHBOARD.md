# AML-X Streamlit Dashboard

## Overview

This is a comprehensive 5-tab Streamlit dashboard for **AML-X: Explainable Graph AI Against Financial Crime**, designed for the TiDB AgentX Hackathon 2025.

## Features

### 1. Overview Tab 📊
- **Title**: "AML-X: Explainable Graph AI Against Financial Crime"
- **Performance Metrics**: PR-AUC, Precision@1%, FPR@95%
- **Narrative**: Problem → Solution → Impact
- **Quick Start Guide**: Navigation instructions

### 2. Detection Tab 🚨
- **Top-K Flagged Accounts**: Table sorted by risk score
- **SHAP Features**: Top 3 SHAP features for each account (if available)
- **Download Button**: Export flagged accounts to CSV
- Adjustable slider to control number of accounts displayed (10-100)

### 3. Graph Tab 🕸️
- **Interactive Subgraph Visualization**: Select an account to explore its transaction network
- **Hover Tooltips**: 
  - Account ID and risk score
  - Degree (number of connections)
  - Cross-border transaction percentage
  - Last 7-day transaction volume
- **Node Coloring**: Red gradient based on risk score
- **Node Sizing**: Larger for selected account

### 4. Explainability Tab 💡
- **Global SHAP Importance**: Horizontal bar chart of top 15 features
- **Local SHAP Explanation**: Feature contributions for selected account
- **GNN Explainer**: Textual reason for why an account was flagged
- Detailed JSON view of GNN explanations (expandable)

### 5. Impact Tab 🎯
- **Analyst Capacity Slider**: Simulate team workload (10-500 cases/day)
- **Current Performance**: Expected true frauds vs. false alarms
- **What-If Analysis**: Threshold tuning with real-time metrics update
- **Threshold Trade-offs Chart**: Visualize precision/recall at different thresholds
- **Workload Estimation**: Calculate days of work generated

## Running the Dashboard

### Prerequisites

Make sure you have the required data and artifacts:

```bash
# 1. Generate synthetic data
make simulate

# 2. Train baseline model
make train_baseline

# 3. Train GNN model
make train_gnn

# 4. Generate SHAP explanations
make explain
```

### Launch Dashboard

From the project root:

```bash
streamlit run src/app/streamlit_app.py
```

Or from the `src/app` directory:

```bash
cd src/app
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## Graceful Degradation

The dashboard handles missing artifacts gracefully:

- **No Data**: Shows helpful message to run `make simulate`
- **No Predictions**: Displays "Run: `make train_gnn`"
- **No SHAP**: Shows "Run: `make explain`"
- **Missing Files**: Warns user and continues with available data

Each tab checks for required artifacts and provides clear instructions when files are missing.

## Expected Artifact Files

The dashboard expects the following files in `artifacts/`:

- `preds_gnn.csv` - GNN predictions (columns: account_id, score, label)
- `shap_global.csv` - Global feature importance (columns: feature, importance)
- `shap_local_{account_id}.json` - Per-account SHAP contributions
- `explanations/node_{account_id}.json` - GNN explainer results

## Data Files

Located in `data/`:

- `transactions.parquet` - Transaction records
- `accounts.parquet` - Account metadata

## Technical Stack

- **Streamlit**: Web framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **NetworkX**: Graph construction and layout
- **NumPy**: Numerical operations

## Tips

1. **Navigation**: Use the sidebar to switch between tabs
2. **Interactivity**: Hover over graph nodes for detailed statistics
3. **Export**: Download flagged accounts from Detection tab
4. **Threshold Tuning**: Experiment with Impact tab to optimize for your team's capacity
5. **Account Selection**: Accounts are ranked by risk score across all tabs

## Architecture

```
streamlit_app.py
├── Tab 1: Overview (show_overview)
│   ├── Metric cards from preds_gnn.csv
│   └── Problem/Solution/Impact narrative
├── Tab 2: Detection (show_detection)
│   ├── Top-K table with SHAP features
│   └── CSV download
├── Tab 3: Graph (show_graph)
│   ├── Subgraph visualization (create_subgraph_with_stats)
│   └── Node statistics with tooltips
├── Tab 4: Explainability (show_explainability)
│   ├── SHAP global bar chart
│   ├── SHAP local contributions
│   └── GNN textual reason
└── Tab 5: Impact (show_impact)
    ├── Analyst capacity simulation
    ├── Threshold what-if analysis
    └── Trade-off visualization
```

## Customization

To customize metrics or thresholds:

- **Default Capacity**: Line 585 - change default from 100
- **Top-K Accounts**: Line 234 - adjust slider range
- **SHAP Features**: Line 495 - change number from 15
- **Graph Layout**: Line 387 - adjust spring layout parameter `k`

## Troubleshooting

**Error: "Module not found"**
- Ensure you're running from project root or `src/app`
- Check that `viz.py` is in the same directory

**Error: "File not found"**
- Run the data generation and training steps
- Check that artifact paths match (relative to app directory)

**Slow Performance**
- Large transaction datasets may slow graph rendering
- Consider limiting subgraph depth or node count

## Future Enhancements

- Real-time data refresh
- User authentication
- Alert configuration
- Model retraining from UI
- TiDB Cloud integration
- Export reports to PDF

