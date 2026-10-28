"""
Polymap — Wallet Relationship Globe (prototype)

Run with:
    streamlit run app.py

Expects, in ../data/processed/ (relative to wherever you launch this):
    temporal_edges.parquet          (required)
    baseline_clusters.parquet       (required)
    wallet_features.parquet         (required)
    candidate_relationships.parquet (required)
    pair_relationship_scores.parquet (optional — see export snippet in chat;
                                       without it, only "Candidate relationships"
                                       view is available)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Polymap — Wallet Relationship Globe", layout="wide")

DATA_DIR = Path("../data/processed")


# --------------------------------------------------------------------------
# 1. Load
# --------------------------------------------------------------------------
@st.cache_data
def load_data():
    required = {
        "temporal_edges": DATA_DIR / "temporal_edges.parquet",
        "baseline_clusters": DATA_DIR / "baseline_clusters.parquet",
        "wallet_features": DATA_DIR / "wallet_features.parquet",
        "candidate_relationships": DATA_DIR / "candidate_relationships.parquet",
    }
    optional = {"pair_scores": DATA_DIR / "pair_relationship_scores.parquet"}

    data, missing = {}, []
    for name, path in required.items():
        if path.exists():
            data[name] = pd.read_parquet(path)
        else:
            missing.append(str(path))
    for name, path in optional.items():
        data[name] = pd.read_parquet(path) if path.exists() else None

    return data, missing


data, missing_files = load_data()

if missing_files:
    st.error(
        "Missing required data file(s):\n\n" + "\n".join(f"- {m}" for m in missing_files)
    )
    st.stop()

temporal_edges = data["temporal_edges"]
baseline_clusters = data["baseline_clusters"]
wallet_features = data["wallet_features"]
candidate_relationships = data["candidate_relationships"]
pair_scores = data["pair_scores"]

has_full_scores = pair_scores is not None
if not has_full_scores:
    st.sidebar.warning(
        "pair_relationship_scores.parquet not found — showing candidate "
        "relationships only. Export the full pair-score table from "
        "wallet_relationship_analysis.ipynb to unlock 'All' / 'Strong' views."
    )


# --------------------------------------------------------------------------
# 2. Build the relationship graph — deduplicate, aggregate, merge scores
# --------------------------------------------------------------------------
@st.cache_data
def build_pair_table(temporal_edges, candidate_relationships, pair_scores):
    edges = temporal_edges.copy()
    edges["pair_key"] = edges.apply(lambda r: tuple(sorted((r["source"], r["target"]))), axis=1)
    edges = (
        edges.sort_values("time_diff")
        .drop_duplicates(subset=["pair_key", "timestamp"], keep="first")
        .reset_index(drop=True)
    )

    pair_agg = (
        edges.groupby("pair_key")
        .agg(n_events=("timestamp", "size"),
             first_interaction=("timestamp", "min"),
             last_interaction=("timestamp", "max"))
        .reset_index()
    )
    pair_agg["wallet_a"] = pair_agg["pair_key"].apply(lambda p: p[0])
    pair_agg["wallet_b"] = pair_agg["pair_key"].apply(lambda p: p[1])

    if pair_scores is not None:
        ps = pair_scores.copy()
        ps["pair_key"] = ps.apply(lambda r: tuple(sorted((r["wallet_a"], r["wallet_b"]))), axis=1)
        pair_agg = pair_agg.merge(
            ps[["pair_key", "mean_margin"]], on="pair_key", how="left"
        )
    else:
        pair_agg["mean_margin"] = np.nan

    cand = candidate_relationships.copy()
    cand["pair_key"] = cand.apply(lambda r: tuple(sorted((r["wallet_a"], r["wallet_b"]))), axis=1)
    cand_cols = ["pair_key", "relationship_type"]
    if "mean_margin" in cand.columns:
        cand = cand.rename(columns={"mean_margin": "candidate_margin"})
        cand_cols.append("candidate_margin")
    pair_agg = pair_agg.merge(cand[cand_cols], on="pair_key", how="left")
    pair_agg["is_candidate"] = pair_agg["relationship_type"].notna()

    # fall back score if the full pair-score table isn't available
    pair_agg["display_score"] = pair_agg["mean_margin"]
    if pair_agg["display_score"].isna().all():
        pair_agg["display_score"] = pair_agg["candidate_margin"]

    return edges, pair_agg


edges, pair_table = build_pair_table(temporal_edges, candidate_relationships, pair_scores)


@st.cache_data
def build_node_table(wallet_features, baseline_clusters):
    nodes = wallet_features.merge(baseline_clusters, on="proxyWallet", how="left")
    return nodes


node_table = build_node_table(wallet_features, baseline_clusters)
node_lookup = node_table.set_index("proxyWallet")


# --------------------------------------------------------------------------
# 3. Sidebar controls
# --------------------------------------------------------------------------
st.sidebar.header("Filters")

view_options = ["Candidate relationships"]
if has_full_scores:
    view_options += ["Strong relationships", "All relationships"]

relationship_view = st.sidebar.radio("Relationship view", view_options, index=0)

min_interactions = st.sidebar.slider(
    "Minimum interactions", 1, int(pair_table["n_events"].max()), 3
)

if relationship_view != "Candidate relationships" and has_full_scores:
    score_min, score_max = float(pair_table["display_score"].min()), float(pair_table["display_score"].max())
    default_min = score_max * 0.5 if relationship_view == "Strong relationships" else score_min
    min_score = st.sidebar.slider("Minimum relationship score", score_min, score_max, default_min)
else:
    min_score = None

st.sidebar.markdown("---")
st.sidebar.subheader("Focus (keeps the view small)")

filtered = pair_table[pair_table["n_events"] >= min_interactions].copy()
if relationship_view == "Candidate relationships":
    filtered = filtered[filtered["is_candidate"]]
elif min_score is not None:
    filtered = filtered[filtered["display_score"] >= min_score]

# progressive disclosure: default to a SMALL seed, not the whole graph
filtered = filtered.sort_values("display_score", ascending=False, na_position="last")

all_wallets_in_view = sorted(set(filtered["wallet_a"]) | set(filtered["wallet_b"]))
seed_options = ["(top-scoring relationships)"] + all_wallets_in_view
seed_wallet = st.sidebar.selectbox("Focus wallet (optional)", seed_options)

n_top = st.sidebar.slider("Number of relationships to show", 1, min(50, max(len(filtered), 1)), min(6, len(filtered)) if len(filtered) else 1)
hops = st.sidebar.slider("Hops from focus wallet", 1, 2, 1)

if seed_wallet == "(top-scoring relationships)":
    view_pairs = filtered.head(n_top)
else:
    # expand outward from the seed wallet by N hops, still capped by n_top
    frontier = {seed_wallet}
    collected_keys = set()
    remaining = filtered.copy()
    for _ in range(hops):
        hop_edges = remaining[remaining["wallet_a"].isin(frontier) | remaining["wallet_b"].isin(frontier)]
        collected_keys |= set(hop_edges["pair_key"])
        frontier |= set(hop_edges["wallet_a"]) | set(hop_edges["wallet_b"])
    view_pairs = filtered[filtered["pair_key"].isin(collected_keys)].head(n_top)

st.sidebar.markdown(f"**Showing {len(view_pairs)} relationships / "
                    f"{len(set(view_pairs['wallet_a']) | set(view_pairs['wallet_b']))} wallets**")

# --------------------------------------------------------------------------
# 4. Timeline slider
# --------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Timeline")

if len(view_pairs):
    t_min = view_pairs["first_interaction"].min()
    t_max = view_pairs["last_interaction"].max()
else:
    t_min, t_max = edges["timestamp"].min(), edges["timestamp"].max()

as_of = st.sidebar.slider(
    "As of", min_value=t_min.to_pydatetime(), max_value=t_max.to_pydatetime(),
    value=t_max.to_pydatetime(), format="MMM D, YYYY HH:mm"
)

as_of_ts = pd.Timestamp(as_of)
if as_of_ts.tzinfo is None:
    as_of_ts = as_of_ts.tz_localize("UTC")

time_filtered_pairs = view_pairs[view_pairs["first_interaction"] <= as_of_ts]


# --------------------------------------------------------------------------
# 5. Layout on a sphere ("globe") + market hub at the center
# --------------------------------------------------------------------------
def fibonacci_sphere(n, radius=10):
    points = []
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = 1 - (i / max(n - 1, 1)) * 2
        r = np.sqrt(max(0, 1 - y * y))
        theta = phi * i
        x, z = np.cos(theta) * r, np.sin(theta) * r
        points.append((x * radius, y * radius, z * radius))
    return points


wallets_shown = sorted(set(time_filtered_pairs["wallet_a"]) | set(time_filtered_pairs["wallet_b"]))
positions = dict(zip(wallets_shown, fibonacci_sphere(len(wallets_shown))))


# --------------------------------------------------------------------------
# 6. Build the plot
# --------------------------------------------------------------------------
fig = go.Figure()

# Market hub, at the center — the top of the requested hierarchy
fig.add_trace(go.Scatter3d(
    x=[0], y=[0], z=[0], mode="markers+text",
    marker=dict(size=14, color="gold", symbol="diamond"),
    text=["MARKET"], textposition="top center", textfont=dict(size=12, color="white"),
    hoverinfo="text", hovertext=["Selected market (prototype hub node)"],
    name="Market"
))

# Market -> wallet spokes
spoke_x, spoke_y, spoke_z = [], [], []
for w in wallets_shown:
    x, y, z = positions[w]
    spoke_x += [0, x, None]
    spoke_y += [0, y, None]
    spoke_z += [0, z, None]

fig.add_trace(go.Scatter3d(
    x=spoke_x, y=spoke_y, z=spoke_z, mode="lines",
    line=dict(color="rgba(150,150,150,0.25)", width=1),
    hoverinfo="skip", showlegend=False
))

# Wallet-wallet relationship edges
edge_x, edge_y, edge_z, edge_hover = [], [], [], []
for _, row in time_filtered_pairs.iterrows():
    if row["wallet_a"] not in positions or row["wallet_b"] not in positions:
        continue
    xa, ya, za = positions[row["wallet_a"]]
    xb, yb, zb = positions[row["wallet_b"]]
    edge_x += [xa, xb, None]
    edge_y += [ya, yb, None]
    edge_z += [za, zb, None]

score_display = row.get("display_score", np.nan)
if len(time_filtered_pairs):
    for _, row in time_filtered_pairs.iterrows():
        status = row["relationship_type"] if pd.notna(row["relationship_type"]) else "ordinary relationship"
        edge_hover.append(
            f"{row['wallet_a'][:10]}.. <-> {row['wallet_b'][:10]}..<br>"
            f"Interactions: {row['n_events']}<br>"
            f"Score: {row['display_score']:.2f}<br>Status: {status}<br>"
            f"First: {row['first_interaction']}<br>Last: {row['last_interaction']}"
        )

candidate_mask = time_filtered_pairs["is_candidate"].fillna(False)
for label, mask, color in [
    ("Candidate relationships", candidate_mask, "crimson"),
    ("Other relationships", ~candidate_mask, "rgba(100,180,255,0.6)"),
]:
    subset = time_filtered_pairs[mask]
    sx, sy, sz = [], [], []
    for _, row in subset.iterrows():
        if row["wallet_a"] not in positions or row["wallet_b"] not in positions:
            continue
        xa, ya, za = positions[row["wallet_a"]]
        xb, yb, zb = positions[row["wallet_b"]]
        sx += [xa, xb, None]
        sy += [ya, yb, None]
        sz += [za, zb, None]
    if sx:
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz, mode="lines",
            line=dict(color=color, width=4 if label == "Candidate relationships" else 2),
            name=label, hoverinfo="skip"
        ))

# Wallet nodes
node_x, node_y, node_z, node_color, node_size, node_hover = [], [], [], [], [], []
for w in wallets_shown:
    x, y, z = positions[w]
    node_x.append(x); node_y.append(y); node_z.append(z)
    info = node_lookup.loc[w] if w in node_lookup.index else None
    if info is not None:
        cluster = info.get("cluster_10", "?")
        trade_count = info.get("trade_count", 0)
        duration = info.get("active_duration_hours", 0)
        buy_ratio = info.get("buy_ratio", np.nan)
        node_color.append(cluster if pd.notna(cluster) else -1)
        node_size.append(6 + min(np.log1p(trade_count) * 2, 14))
        node_hover.append(
            f"{w[:12]}..<br>Cluster: {cluster}<br>Trades: {trade_count}<br>"
            f"Active: {duration:.1f}h<br>Buy ratio: {buy_ratio:.2f}"
        )
    else:
        node_color.append(-1); node_size.append(6)
        node_hover.append(f"{w[:12]}..")

fig.add_trace(go.Scatter3d(
    x=node_x, y=node_y, z=node_z, mode="markers",
    marker=dict(size=node_size, color=node_color, colorscale="Viridis", showscale=True,
                colorbar=dict(title="Baseline<br>cluster")),
    text=node_hover, hoverinfo="text", name="Wallets"
))

fig.update_layout(
    height=750,
    scene=dict(
        xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
        bgcolor="rgb(10,10,20)",
    ),
    paper_bgcolor="rgb(10,10,20)",
    font=dict(color="white"),
    legend=dict(bgcolor="rgba(0,0,0,0.4)"),
    margin=dict(l=0, r=0, t=30, b=0),
    title=f"Wallet relationship globe — as of {pd.Timestamp(as_of).strftime('%b %d, %Y %H:%M UTC')}",
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    "Drag to rotate, scroll to zoom. Node color = baseline behavioral cluster, "
    "node size = log(trade count). Red edges = flagged candidate relationships. "
    "This is a relationship layout (nodes placed evenly on a sphere for visual clarity) "
    "— not a geographic map."
)
