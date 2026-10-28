"""Training and scoring loop (batched).

The ordering rule that matters is unchanged: events in a batch are scored
against negatives using memory as it was BEFORE the batch, and only then is the
batch absorbed into memory. Reverse those two and the model scores interactions
it has already seen, which inflates every number.

What changed from the per-event version
---------------------------------------
Events are processed in batches of `cfg.batch_size` (default 256), the way the
original TGN does it, instead of one at a time. That turns ~40 tiny torch ops
per event into ~40 ops per *batch*, which is where the speed-up comes from.

Trade-offs, stated plainly:
  * Inside one batch, memory is stale: event 200 does not see event 3's update.
  * If a wallet appears several times in a batch, only the LAST update to it is
    kept (scatter is last-write-wins). Both are standard TGN approximations and
    both shrink as `batch_size` shrinks. Set batch_size=1 to recover the old
    behaviour, at the old speed.
  * Each batch is absorbed into memory lazily, at the start of the NEXT batch,
    inside the autograd graph. That is what lets every batch's loss send gradient
    into the message/GRU weights (the TGN trick). Memory still only ever contains
    events strictly earlier than the batch being scored.
  * The optimizer steps every `bptt_batches` batches, then memory is detached.
Numbers from this trainer are NOT comparable to runs made with the per-event
trainer (the config fingerprint changes because of the new fields).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import ModelConfig
from ..graph.build import GraphBundle
from ..models.pair_history import PairHistory
from ..training.negatives import NegativeSampler, build_sampler
from ..training.splits import TemporalSplit

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    losses: list[float] = field(default_factory=list)
    memory: object = None
    model: object = None
    device: str = "cpu"


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for training. Install it with "
            "`pip install torch`, or use the baselines, which need only numpy."
        ) from e


def normalise_edge_attr(edge_attr: np.ndarray, train_idx: np.ndarray
                        ) -> tuple[np.ndarray, dict]:
    """Standardise time_diff using TRAIN statistics only.

    Using full-dataset statistics leaks test information into training, which
    is subtle enough to survive review and invalidate the result anyway.
    """
    arr = edge_attr.astype(np.float32).copy()
    minutes = arr[:, 0] / 60.0
    mean = float(minutes[train_idx].mean())
    std = float(minutes[train_idx].std())
    arr[:, 0] = (minutes - mean) / (std + 1e-8)
    return arr, {"time_mean_min": mean, "time_std_min": std}


def train(bundle: GraphBundle, split: TemporalSplit, cfg: ModelConfig,
          negative: str = "random") -> TrainResult:
    _require_torch()
    import torch
    import torch.nn.functional as F

    from ..models.temporal import PolymapTGN, resolve_device

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)

    edge_attr, stats = normalise_edge_attr(bundle.edge_attr, split.train_idx)
    log.info("edge time normalisation (train only): %s", stats)

    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    ei = torch.tensor(bundle.edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    model = PolymapTGN(
        bundle.x.shape[1], bundle.edge_attr.shape[1], cfg.hidden_dim, cfg.temperature,
        pair_features=cfg.pair_features, memory_updates=cfg.memory_updates).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    sampler = build_sampler(negative, bundle.edge_index, split.train_idx, bundle.num_nodes)

    order = split.train_idx
    B = max(1, cfg.batch_size)
    K = max(1, cfg.bptt_batches)
    n_batches = (len(order) + B - 1) // B
    log.info("training: %d events/epoch, batch=%d, %d batches/epoch, device=%s",
             len(order), B, n_batches, device)
    log_every = max(1, n_batches // 5)

    losses: list[float] = []
    memory = None
    for epoch in range(cfg.epochs):
        model.train()
        memory = model.initial_memory(x)      # not detached: reaches node_encoder
        pair_history = PairHistory(bundle.num_nodes)
        pending = None     # previous batch, absorbed lazily so its update carries grad
        window: list = []
        recent: list[tuple[float, float, float]] = []

        for bi in range(n_batches):
            sel = order[bi * B:(bi + 1) * B]
            idx = torch.as_tensor(sel, device=device)
            s, t, attr = ei[0, idx], ei[1, idx], ea[idx]
            src_np = bundle.edge_index[0][sel]
            tgt_np = bundle.edge_index[1][sel]
            ts_np = bundle.timestamps[sel]
            neg_np = sampler.sample_many(src_np, rng, 1)[:, 0]
            negs = torch.as_tensor(neg_np, device=device)

            # Absorb the PREVIOUS batch now (inside the autograd graph), so the
            # message/GRU weights get gradient from this batch's loss. Memory
            # still contains only events strictly before this batch.
            if pending is not None:
                memory = model.advance(memory, *pending)
            pair_pos = pair_history.features(src_np, tgt_np, ts_np)
            pair_neg = pair_history.features(src_np, neg_np, ts_np)
            pair_pos_t = torch.as_tensor(pair_pos, dtype=torch.float32, device=device)
            pair_neg_t = torch.as_tensor(pair_neg, dtype=torch.float32, device=device)
            pos = model.score(memory, s, t, pair_pos_t if cfg.pair_features else None)
            neg = model.score(memory, s, negs, pair_neg_t if cfg.pair_features else None)
            window.append((F.softplus(-pos) + F.softplus(neg)).mean())
            pending = (s, t, attr)
            pair_history.update(src_np, tgt_np, ts_np)

            recent.append((float(window[-1].detach()), float(pos.detach().mean()),
                           float(neg.detach().mean())))
            if len(window) == K or bi == n_batches - 1:
                loss = torch.stack(window).mean()
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                memory = memory.detach()
                losses.append(float(loss.detach()))
                window = []
            if (bi + 1) % log_every == 0:
                r = np.mean(recent, axis=0)
                log.info("epoch %d/%d batch %d/%d | loss %.4f | pos %.2f neg %.2f",
                         epoch + 1, cfg.epochs, bi + 1, n_batches, r[0], r[1], r[2])
                recent.clear()
        if pending is not None:               # leave memory current for scoring
            with torch.no_grad():
                memory = model.advance(memory, *pending)

    return TrainResult(losses=losses, memory=memory.detach(), model=model, device=str(device))


def score_events(result: TrainResult, bundle: GraphBundle, split: TemporalSplit,
                 sampler: NegativeSampler, seed: int = 42,
                 n_negatives: int = 50, batch_size: int = 1024) -> dict:
    """Score test events against n_negatives each, in batches.

    Negatives equal to the source or true target are redrawn (a "negative" that
    is the true target is a tie, which distorts AUC under the hard sampler).
    Each event also records `gap_s`: seconds since the same wallet pair last
    interacted (NaN if never), so metrics can be stratified by recency.
    """
    _require_torch()
    import pandas as pd
    import torch

    model, memory = result.model, result.memory
    model.eval()
    rng = np.random.default_rng(seed)
    device = torch.device(result.device)

    edge_attr, _ = normalise_edge_attr(bundle.edge_attr, split.train_idx)
    ei = torch.tensor(bundle.edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    src_np, tgt_np = bundle.edge_index
    ts_np = bundle.timestamps
    test = split.test_idx
    n_nodes = bundle.num_nodes

    # gap_s: time since the same undirected pair last interacted (train + test
    # events, in index order). Vectorised; identical to the old dict walk.
    a, b = np.minimum(src_np, tgt_np).astype(np.int64), np.maximum(src_np, tgt_np).astype(np.int64)
    prev = pd.Series(ts_np.astype(np.float64)).groupby(a * n_nodes + b).shift(1).to_numpy()
    gap_all = ts_np - prev                      # NaN where never seen
    gap = gap_all[test]

    def draw(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
        c = sampler.sample_many(src, rng, n_negatives)
        for _ in range(10):                     # redraw collisions, bounded
            bad = (c == src[:, None]) | (c == tgt[:, None])
            if not bad.any():
                break
            c[bad] = sampler.sample_many(np.zeros(int(bad.sum()), np.int64), rng, 1)[:, 0] \
                if sampler.name != "hard" else \
                sampler.sample_many(np.repeat(src[:, None], n_negatives, 1)[bad], rng, 1)[:, 0]
        return c

    pos_all, neg_all, neg_id_all = [], [], []
    mem = memory.clone()
    pair_history = PairHistory(bundle.num_nodes)
    pair_history.replay(bundle.edge_index, bundle.timestamps, split.train_idx)
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            idx_np = test[start:start + batch_size]
            idx = torch.as_tensor(idx_np, device=device)
            s, t = ei[0, idx], ei[1, idx]
            neg_np = draw(src_np[idx_np], tgt_np[idx_np])
            neg_id_all.append(neg_np)
            neg_ids = torch.as_tensor(neg_np, device=device)
            event_times = ts_np[idx_np]
            pair_pos = pair_history.features(src_np[idx_np], tgt_np[idx_np], event_times)
            pair_neg = pair_history.features(
                np.repeat(src_np[idx_np], n_negatives),
                neg_np.reshape(-1),
                np.repeat(event_times, n_negatives))
            pair_pos_t = torch.as_tensor(pair_pos, dtype=torch.float32, device=device)
            pair_neg_t = torch.as_tensor(pair_neg, dtype=torch.float32, device=device)
            p = model.score(mem, s, t, pair_pos_t if model.pair_features else None)
            ns = model.score(
                mem, s[:, None].expand(-1, n_negatives).reshape(-1), neg_ids.reshape(-1),
                pair_neg_t if model.pair_features else None).reshape(len(idx_np), n_negatives)
            pos_all.append(p.cpu().numpy())
            neg_all.append(ns.cpu().numpy())
            mem = model.advance(mem, s, t, ea[idx])
            pair_history.update(src_np[idx_np], tgt_np[idx_np], event_times)

    pos = np.concatenate(pos_all) if pos_all else np.empty(0)
    negs = np.concatenate(neg_all) if neg_all else np.empty((0, n_negatives))
    neg_ids_out = (np.concatenate(neg_id_all) if neg_id_all
                   else np.empty((0, n_negatives), dtype=np.int64))
    ev = pd.DataFrame({
        "edge_idx": test.astype(np.int64),
        "source": [bundle.id_to_wallet[int(i)] for i in src_np[test]],
        "target": [bundle.id_to_wallet[int(i)] for i in tgt_np[test]],
        "timestamp": ts_np[test].astype(np.int64),
        "positive_score": pos.astype(np.float64),
        "negative_score": negs.mean(axis=1).astype(np.float64),
        "margin": (pos - negs.mean(axis=1)).astype(np.float64),
        "gap_s": gap,
    })
    # neg_ids: the exact negatives scored, so baselines can be evaluated on the same ones.
    return {"positive": pos, "negative": negs, "events": ev, "neg_ids": neg_ids_out}
