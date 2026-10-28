
from pathlib import Path

import numpy as np
import pandas as pd

RUN_NAME = "xi_2027_63d_p50"
TOP_WALLETS = 100

ROOT = Path("data/processed") / RUN_NAME

trades = pd.read_parquet(ROOT / "trades.parquet")
edges = pd.read_parquet(ROOT / "temporal_edges.parquet")
features = pd.read_parquet(ROOT / "wallet_features.parquet")

# 1. Summarise repeated interactions for each unordered wallet pair.
edges = edges.copy()
edges["wallet_a"] = edges[["source", "target"]].min(axis=1)
edges["wallet_b"] = edges[["source", "target"]].max(axis=1)

pairs = (
    edges.groupby(["wallet_a", "wallet_b"], as_index=False)
    .agg(
        interaction_events=("timestamp", "size"),
        distinct_timestamps=("timestamp", "nunique"),
        median_time_diff=("time_diff", "median"),
        same_behavior_rate=("same_behavior", "mean"),
        same_outcome_rate=("same_outcome", "mean"),
    )
)

# 2. Heuristic candidate score, NOT a probability of coordination.
pairs["pair_score"] = (
    np.log1p(pairs["interaction_events"])
    * (0.5 + pairs["same_behavior_rate"].fillna(0))
    * np.log1p(pairs["distinct_timestamps"])
    / (1 + pairs["median_time_diff"].clip(lower=0).fillna(300) / 60)
)

pairs = pairs.sort_values("pair_score", ascending=False)
pairs.to_csv(ROOT / "ranked_wallet_pairs.csv", index=False)

# 3. Aggregate pair evidence to each wallet.
wallet_scores = pd.concat([
    pairs[["wallet_a", "pair_score"]].rename(
        columns={"wallet_a": "wallet"}
    ),
    pairs[["wallet_b", "pair_score"]].rename(
        columns={"wallet_b": "wallet"}
    ),
])

wallet_scores = wallet_scores.groupby("wallet").agg(
    pair_score_sum=("pair_score", "sum"),
    connected_wallets=("pair_score", "size"),
    strongest_pair_score=("pair_score", "max"),
).reset_index()

# 4. Add the existing wallet activity features.
ranked = wallet_scores.merge(
    features.drop(columns=["market_id"], errors="ignore"),
    on="wallet",
    how="left",
)

# Keep both behavioral evidence and activity visible.
ranked = ranked.sort_values(
    ["pair_score_sum", "connected_wallets"],
    ascending=False,
).reset_index(drop=True)

ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
ranked.head(TOP_WALLETS).to_csv(
    ROOT / "top_wallets.csv", index=False
)

print(f"Wallets ranked: {len(ranked)}")
print(f"Top wallets saved: {ROOT / 'top_wallets.csv'}")
print(f"Pair rankings saved: {ROOT / 'ranked_wallet_pairs.csv'}")
print(ranked.head(20).to_string(index=False))
