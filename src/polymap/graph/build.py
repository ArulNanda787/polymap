"""Turn edges and wallet features into tensors.

Kept free of torch-geometric so the package imports on a machine without it;
`to_pyg_data()` is the only function that needs it and it imports lazily.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..data.schema import EDGE_FEATURE_COLUMNS, NODE_FEATURE_COLUMNS

log = logging.getLogger(__name__)


@dataclass
class GraphBundle:
    """Everything downstream stages need, in one object.

    Notebooks used to pass ten loose variables between cells, which is how a
    stale `wallet_to_id` silently corrupts a later result. One object with a
    known shape removes that whole class of bug.
    """
    x: np.ndarray                  # (num_nodes, num_node_features), standardised
    edge_index: np.ndarray         # (2, num_edges)
    edge_attr: np.ndarray          # (num_edges, num_edge_features)
    timestamps: np.ndarray         # (num_edges,) unix seconds, ascending
    wallet_to_id: dict[str, int]
    id_to_wallet: dict[int, str]
    edges: pd.DataFrame
    scaler: StandardScaler

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def summary(self) -> dict:
        return {"num_nodes": self.num_nodes, "num_edges": self.num_edges,
                "node_features": self.x.shape[1], "edge_features": self.edge_attr.shape[1]}


def build_graph(edges: pd.DataFrame, wallet_features: pd.DataFrame) -> GraphBundle:
    if edges.empty:
        raise ValueError("cannot build a graph from zero edges")

    edges = edges.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    wallets = sorted(set(edges["source"]) | set(edges["target"]))
    wallet_to_id = {w: i for i, w in enumerate(wallets)}
    id_to_wallet = {i: w for w, i in wallet_to_id.items()}

    # Collapse to one row per wallet. Multi-market runs give a wallet one row
    # per market, so average them rather than letting a merge pick arbitrarily.
    feats = (wallet_features.groupby("wallet", as_index=False)[NODE_FEATURE_COLUMNS]
             .mean())
    feats = feats[feats["wallet"].isin(wallet_to_id)].copy()
    feats["node_id"] = feats["wallet"].map(wallet_to_id)
    feats = feats.sort_values("node_id")

    if len(feats) != len(wallets):
        missing = set(wallets) - set(feats["wallet"])
        raise ValueError(
            f"{len(missing)} wallets appear in edges but have no features "
            f"(e.g. {list(missing)[:3]}). Features and edges came from "
            f"different trade sets.")

    scaler = StandardScaler()
    x = scaler.fit_transform(feats[NODE_FEATURE_COLUMNS].to_numpy(np.float64)).astype(np.float32)

    edge_index = np.vstack([edges["source"].map(wallet_to_id).to_numpy(np.int64),
                            edges["target"].map(wallet_to_id).to_numpy(np.int64)])
    edge_attr = edges[EDGE_FEATURE_COLUMNS].to_numpy(np.float32)
    timestamps = edges["timestamp"].to_numpy(np.int64)

    log.info("graph: %d nodes, %d edges, %d node feats, %d edge feats",
             len(wallets), edge_index.shape[1], x.shape[1], edge_attr.shape[1])
    return GraphBundle(x, edge_index, edge_attr, timestamps,
                       wallet_to_id, id_to_wallet, edges, scaler)


def to_pyg_data(bundle: GraphBundle):
    """Optional torch-geometric view, for reusing PyG utilities."""
    import torch
    from torch_geometric.data import Data
    d = Data(x=torch.tensor(bundle.x), 
             edge_index=torch.tensor(bundle.edge_index),
             edge_attr=torch.tensor(bundle.edge_attr))
    d.timestamp = torch.tensor(bundle.timestamps)
    return d
