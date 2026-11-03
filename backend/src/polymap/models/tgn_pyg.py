"""TGN (Rossi et al., 2020) via torch_geometric, scored on the project's protocol.

Optional: needs `pip install torch_geometric`. It follows the reference PyG
example (TGNMemory + LastAggregator + TransformerConv over last-10 neighbours),
trained with the same chronological train prefix and evaluated on the SAME fixed
negatives and timestamp-grouped batches as every other method.

Scope, stated plainly:
  * This is TGN. TGAT is NOT included: it needs temporal neighbour sampling with
    attention at every layer and is better taken from DyGLib than re-implemented.
  * The PyG example is written against a particular torch_geometric API. If a
    newer release moves `LastNeighborLoader`/`IdentityMessage`, adjust the import
    below; the logic does not change.
  * The bundle must be chronologically sorted with train = prefix and
    test = the contiguous suffix (true for `temporal_split` and for prospective
    splits). This is asserted, because the neighbour loader's edge ids are the
    running insert counter.
"""
from __future__ import annotations

import logging

import numpy as np

from ..evaluation.protocol import time_grouped_slices
from ..graph.build import GraphBundle
from ..training.negatives import build_sampler
from ..training.splits import TemporalSplit

log = logging.getLogger(__name__)


def run_tgn(bundle: GraphBundle, split: TemporalSplit, neg_ids, *,
            seed: int = 42, epochs: int = 5, lr: float = 1e-3, dim: int = 32,
            batch_size: int = 200, eval_batch: int = 32, n_neighbors: int = 10,
            train_negative: str = "hard", device: str = "cpu"):
    """Train on split.train_idx, score split.test_idx.

    `neg_ids` is either one (n_test, K) array -> returns (pos, neg), or a dict
    {name: array} -> returns {name: (pos, neg)}. With a dict the model is trained
    ONCE and every negative set is scored from an identical copy of the state at
    the train cutoff, so samplers are compared on the same trained model.
    """
    import copy

    import torch
    from torch import nn

    try:
        from torch_geometric.nn import TGNMemory, TransformerConv
        from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
    except ImportError as e:  # pragma: no cover
        raise ImportError("TGN needs torch_geometric: pip install torch_geometric") from e
    try:                                   # torch_geometric >= 2.3
        from torch_geometric.loader import LastNeighborLoader
    except ImportError:                    # older releases
        from torch_geometric.nn.models.tgn import LastNeighborLoader

    from ..training.trainer import normalise_edge_attr

    n_train, n_total = len(split.train_idx), bundle.num_edges
    if not (np.array_equal(split.train_idx, np.arange(n_train))
            and np.array_equal(split.test_idx, np.arange(n_train, n_train + len(split.test_idx)))):
        raise ValueError("TGN needs train = prefix and test = contiguous suffix of the sorted edges")

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    src_np, tgt_np = bundle.edge_index
    ts_np = bundle.timestamps
    edge_attr, _ = normalise_edge_attr(bundle.edge_attr, split.train_idx)
    msg_all = torch.tensor(edge_attr, dtype=torch.float32, device=dev)
    t_all = torch.tensor(ts_np - ts_np.min(), dtype=torch.long, device=dev)
    src_all = torch.tensor(src_np, dtype=torch.long, device=dev)
    dst_all = torch.tensor(tgt_np, dtype=torch.long, device=dev)
    msg_dim = msg_all.size(-1)
    n_nodes = bundle.num_nodes

    memory = TGNMemory(n_nodes, msg_dim, dim, dim,
                       message_module=IdentityMessage(msg_dim, dim, dim),
                       aggregator_module=LastAggregator()).to(dev)

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_enc = memory.time_enc
            edge_dim = msg_dim + self.time_enc.out_channels
            self.conv = TransformerConv(dim, dim // 2, heads=2, dropout=0.1, edge_dim=edge_dim)

        def forward(self, x, last_update, edge_index, t, msg):
            rel_t = last_update[edge_index[0]] - t
            enc = self.time_enc(rel_t.to(x.dtype))
            return self.conv(x, edge_index, torch.cat([enc, msg], dim=-1))

    class Link(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b, self.out = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, 1)

        def forward(self, zs, zd):
            return self.out((self.a(zs) + self.b(zd)).relu())

    gnn, link = Attn().to(dev), Link().to(dev)
    loader = LastNeighborLoader(n_nodes, size=n_neighbors, device=dev)
    params = set(memory.parameters()) | set(gnn.parameters()) | set(link.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    crit = nn.BCEWithLogitsLoss()
    assoc = torch.empty(n_nodes, dtype=torch.long, device=dev)
    sampler = build_sampler(train_negative, bundle.edge_index, split.train_idx, n_nodes)

    def embed(n_id):
        n_id, edge_index, e_id = loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=dev)
        z, last_update = memory(n_id)
        return gnn(z, last_update, edge_index, t_all[e_id], msg_all[e_id])

    for epoch in range(epochs):
        memory.reset_state()
        loader.reset_state()
        memory.train(); gnn.train(); link.train()
        total = 0.0
        for st in range(0, n_train, batch_size):
            sel = np.arange(st, min(st + batch_size, n_train))
            s, d = src_all[sel], dst_all[sel]
            neg = torch.as_tensor(sampler.sample_many(src_np[sel], rng, 1)[:, 0], device=dev)
            z = embed(torch.cat([s, d, neg]).unique())
            po = link(z[assoc[s]], z[assoc[d]])
            no = link(z[assoc[s]], z[assoc[neg]])
            loss = crit(po, torch.ones_like(po)) + crit(no, torch.zeros_like(no))
            memory.update_state(s, d, t_all[sel], msg_all[sel])
            loader.insert(s, d)
            opt.zero_grad(); loss.backward(); opt.step()
            memory.detach()
            total += float(loss) * len(sel)
        log.info("tgn epoch %d/%d loss %.4f", epoch + 1, epochs, total / n_train)

    # Memory and neighbour state now end exactly at the train cutoff. Continue
    # chronologically through the test events; never reset. Each negative set
    # gets its own copy of that state.
    memory.eval(); gnn.eval(); link.eval()
    test = split.test_idx
    single = not isinstance(neg_ids, dict)
    sets = {"_": neg_ids} if single else dict(neg_ids)
    mem0, loader0 = copy.deepcopy(memory.state_dict()), copy.deepcopy(loader.__dict__)
    store0 = {k: copy.deepcopy(getattr(memory, k)) for k in ("msg_s_store", "msg_d_store")
              if hasattr(memory, k)}
    out = {}
    for name, ids in sets.items():
        memory.load_state_dict(mem0)
        for k, v in store0.items():
            setattr(memory, k, copy.deepcopy(v))
        loader.__dict__.update(copy.deepcopy(loader0))
        K = ids.shape[1]
        pos_out = np.empty(len(test))
        neg_out = np.empty((len(test), K))
        with torch.no_grad():
            for sl in time_grouped_slices(ts_np[test], eval_batch):
                eids = test[sl]
                s, d = src_all[eids], dst_all[eids]
                neg = torch.as_tensor(ids[sl], device=dev)
                z = embed(torch.cat([s, d, neg.reshape(-1)]).unique())
                zs = z[assoc[s]]
                pos_out[sl] = link(zs, z[assoc[d]]).view(-1).cpu().numpy()
                zs_rep = zs.unsqueeze(1).expand(-1, K, -1).reshape(-1, zs.size(-1))
                neg_out[sl] = link(zs_rep, z[assoc[neg.reshape(-1)]]).view(-1, K).cpu().numpy()
                memory.update_state(s, d, t_all[eids], msg_all[eids])
                loader.insert(s, d)
        out[name] = (pos_out, neg_out)
    return out["_"] if single else out
