"""Test that the generator can be used as a module."""
from data.simulate import RealisticTransactionGenerator

# Create generator
generator = RealisticTransactionGenerator(
    n_accounts=50,
    n_tx=200,
    days=30,
    seed=42,
    pct_suspicious=0.1
)

# Generate data
accounts_df, transactions_df, labels_df = generator.generate()

# Verify
print(f"\nModule test successful!")
print(f"Generated: {len(accounts_df)} accounts, {len(transactions_df)} transactions")
print(f"Suspicious nodes: {labels_df['label_node'].sum()}")
print(f"Suspicious edges: {transactions_df['label_edge'].sum()}")

