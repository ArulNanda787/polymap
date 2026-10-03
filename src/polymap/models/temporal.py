"""The temporal memory model.

This is a prototype in the temporal-memory family, not a faithful
reimplementation of a published temporal graph network. It has node memory,
GRU updates, chronological event processing and truncated BPTT. It does not
have per-node time encoding or an explicit last-update-time mechanism. Saying
so plainly costs nothing and protects every other claim you make.

Two stabilisation choices below are load-bearing rather than cosmetic. Without
them the representation collapsed: one principal component carried 96.7% of
variance, arbitrary wallet pairs sat at cosine 0.988, and isolated wallets were
bitwise identical. With L2-normalised memory and a temperature-scaled cosine
head, PC1 drops to 41.1% and isolated wallets separate.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalMemory(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.memory_update = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, memory, source, target, edge_attr):
        src_mem, tgt_mem = memory[source], memory[target]
        message = self.message_mlp(torch.cat([src_mem, tgt_mem, edge_attr], dim=1))

        new_src = F.normalize(self.memory_update(message, src_mem), p=2, dim=-1)
        new_tgt = F.normalize(self.memory_update(message, tgt_mem), p=2, dim=-1)

        memory = memory.clone()
        memory[source] = new_src
        memory[target] = new_tgt
        return memory


class CosineLinkPredictor(nn.Module):
    """Temperature-scaled cosine similarity.

    A raw dot product lets the model raise scores by growing memory norms,
    which is the collapse described above. Cosine forces direction to carry
    the signal.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, source_memory, target_memory):
        return F.cosine_similarity(source_memory, target_memory, dim=-1) / self.temperature


class PolymapTGN(nn.Module):
    """Node encoder, memory module and predictor as one object.

    Bundling them means `initial_memory` has exactly one definition. The
    original notebooks created several competing zero-initialised tensors,
    which is a silent inconsistency nobody notices until results drift.
    """

    def __init__(self, num_node_features: int, num_edge_features: int,
                 hidden_dim: int = 32, temperature: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Linear(num_node_features, hidden_dim)
        self.memory_module = TemporalMemory(hidden_dim, num_edge_features, hidden_dim)
        self.predictor = CosineLinkPredictor(temperature)

    def initial_memory(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.node_encoder(x), p=2, dim=-1)

    def score(self, memory, source, target):
        return self.predictor(memory[source], memory[target])

    def advance(self, memory, source, target, edge_attr):
        return self.memory_module(memory, source, target, edge_attr)


def resolve_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
