.PHONY: help setup simulate features train_baseline train_gnn explain app lint format clean test

help:
	@echo "AML-X Makefile Commands:"
	@echo "  make setup          - Install dependencies and setup environment"
	@echo "  make simulate       - Generate synthetic transaction data"
	@echo "  make features       - Generate tabular and graph features"
	@echo "  make train_baseline - Train tabular baseline model (XGBoost + SHAP)"
	@echo "  make train_gnn      - Train GNN model (GraphSAGE/GAT)"
	@echo "  make explain        - Generate SHAP and GNN explanations"
	@echo "  make app            - Launch Streamlit dashboard"
	@echo "  make lint           - Run code quality checks"
	@echo "  make format         - Auto-format code with black"
	@echo "  make clean          - Remove generated artifacts"
	@echo "  make test           - Run test suite"

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

simulate:
	. .venv/bin/activate && python -m src.data.simulate --n_accounts 5000 --n_tx 120000 --days 90 --seed 42 --pct_suspicious 0.02 --out data/

features:
	. .venv/bin/activate && python -m src.features.tabular --accounts data/accounts.parquet --tx data/transactions.parquet --out artifacts/account_features.parquet && python -m src.features.graph --accounts data/accounts.parquet --tx data/transactions.parquet --out artifacts/graph_data.pt

train_baseline:
	. .venv/bin/activate && python -m src.models.baseline_tabular

train_gnn:
	. .venv/bin/activate && python -m src.models.gnn --epochs 40

explain:
	. .venv/bin/activate && python -m src.explain.gnn_explain --topk 10 && python -m src.explain.shap_tools --topk 20

app:
	. .venv/bin/activate && streamlit run src/app/streamlit_app.py

lint:
	flake8 src/
	isort --check-only src/
	black --check src/

format:
	isort src/
	black src/
	@echo "✓ Code formatted"

clean:
	rm -rf data/*.parquet data/*.csv
	rm -rf artifacts/*.pkl artifacts/*.pt artifacts/*.json
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned generated files"

test:
	pytest tests/ -v
	@echo "✓ Tests passed"

