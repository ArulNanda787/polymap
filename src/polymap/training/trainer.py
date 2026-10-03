"""Training and scoring loop.

The ordering rule below is the one that matters: for every event the model
scores the true target and a negative **before** the event is revealed, and
only then advances memory. Reverse those two and the model is scoring an
interaction it has already absorbed, which inflates every number.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import ModelConfig
from ..graph.build import GraphBundle
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
    from ..models.temporal import PolymapTGN, resolve_device

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = resolve_device(cfg.device)

    edge_attr, stats = normalise_edge_attr(bundle.edge_attr, split.train_idx)
    log.info("edge time normalisation (train only): %s", stats)

    x = torch.tensor(bundle.x, dtype=torch.float32, device=device)
    ei = torch.tensor(bundle.edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    model = PolymapTGN(bundle.x.shape[1], bundle.edge_attr.shape[1],
                       cfg.hidden_dim, cfg.temperature).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                           weight_decay=cfg.weight_decay)
    sampler = build_sampler(negative, bundle.edge_index, split.train_idx, bundle.num_nodes)

    memory = model.initial_memory(x).detach().clone()
    losses = []
    order = split.train_idx

    for start in range(0, len(order), cfg.bptt_chunk):
        chunk = order[start:start + cfg.bptt_chunk]
        model.train()
        batch_losses = []
        for i in chunk:
            s = ei[0, i].unsqueeze(0)
            t = ei[1, i].unsqueeze(0)
            neg = torch.tensor([sampler.sample(int(bundle.edge_index[0][i]), rng)],
                               device=device)

            # score BEFORE the event is absorbed into memory
            pos_score = model.score(memory, s, t)
            neg_score = model.score(memory, s, neg)
            batch_losses.append(
                torch.nn.functional.softplus(-pos_score)
                + torch.nn.functional.softplus(neg_score))

            memory = model.advance(memory, s, t, ea[i].unsqueeze(0))

        loss = torch.stack(batch_losses).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        memory = memory.detach()
        losses.append(float(loss.item()))
        log.info("events %d-%d | loss %.4f", start, start + len(chunk), losses[-1])

    return TrainResult(losses=losses, memory=memory, model=model, device=str(device))


def score_events(result: TrainResult, bundle: GraphBundle, split: TemporalSplit,
                 sampler: NegativeSampler, seed: int = 42,
                 n_negatives: int = 1) -> dict:
    """Score test events, returning positive and negative scores plus margins.

    Margin (positive minus negative, per event) is the quantity the training
    objective directly shapes, which is why relationship ranking uses it rather
    than static embedding cosine.
    """
    _require_torch()
    import torch

    model, memory = result.model, result.memory
    model.eval()
    rng = np.random.default_rng(seed)
    device = torch.device(result.device)

    edge_attr, _ = normalise_edge_attr(bundle.edge_attr, split.train_idx)
    ei = torch.tensor(bundle.edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)

    pos, negs, rows = [], [], []
    mem = memory.clone()
    with torch.no_grad():
        for i in split.test_idx:
            s = ei[0, i].unsqueeze(0)
            t = ei[1, i].unsqueeze(0)
            p = float(model.score(mem, s, t).item())
            ns = [float(model.score(
                mem, s, torch.tensor([sampler.sample(int(bundle.edge_index[0][i]), rng)],
                                     device=device)).item())
                  for _ in range(n_negatives)]
            pos.append(p)
            negs.append(ns)
            rows.append({
                "edge_idx": int(i),
                "source": bundle.id_to_wallet[int(bundle.edge_index[0][i])],
                "target": bundle.id_to_wallet[int(bundle.edge_index[1][i])],
                "timestamp": int(bundle.timestamps[i]),
                "positive_score": p,
                "negative_score": float(np.mean(ns)),
                "margin": p - float(np.mean(ns)),
            })
            mem = model.advance(mem, s, t, ea[i].unsqueeze(0))

    import pandas as pd
    return {"positive": np.array(pos), "negative": np.array(negs),
            "events": pd.DataFrame(rows)}
