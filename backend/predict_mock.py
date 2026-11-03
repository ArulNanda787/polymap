#!/usr/bin/env python
"""Run the real PolymapTGN model on mock trades and export its predictions.

What this does, per market in the frontend's demo dataset:

  1. Simulate 60 days of trades for ~36 wallets. Wallet pairs listed as
     relationships in the demo data trade together in short bursts (a few
     seconds apart, usually on the same side); everyone also trades on their
     own, and a few market-wide "news" moments make many wallets trade at once.
     Some relationships stop before the end and some only start late, so the
     model has to cope with change, not just repetition.
  2. Build temporal co-trade edges with the backend's own `build_temporal_edges`.
  3. Split chronologically (70% train / 30% test) with `temporal_split`.
  4. Train PolymapTGN with the tuned settings (node memory + pair-history head,
     train-only node features) using the backend's `train`. Training negatives
     are a 50/50 mix of hard and random by default (see `use_mixed_negatives`).
  5. Evaluate on the held-out test events with `score_events` (MRR, Hits@10,
     AUC) and compare with the recency baseline on the same negatives.
  6. Replay the test window event by event. Each time a wallet trades, the
     trained model scores every other wallet as its likely partner (before
     seeing the answer). A pair's prediction score is the average probability
     the model gave it, and is compared with whether the pair actually
     co-traded in the test window.

The output JSON is what the frontend's Predictions tab draws.

Usage (from the backend folder, in the environment that has torch):

    python predict_mock.py
    python predict_mock.py --markets 3 --epochs 10     # quicker
    python predict_mock.py --out /path/to/frontend/data/model_predictions.json
    python predict_mock.py --real --epochs 5           # the tuned run's REAL trades

The trades are synthetic. The model, training and scoring code are the real
backend code, unchanged.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from polymap.features.edges import build_temporal_edges  # noqa: E402
from polymap.features.wallets import build_wallet_features  # noqa: E402
from polymap.graph.build import build_graph  # noqa: E402
from polymap.training.splits import temporal_split  # noqa: E402

log = logging.getLogger("predict_mock")

DAY = 86_400
START_TS = 1_790_000_000           # arbitrary fixed start so runs are reproducible
WINDOW_SECONDS = 300               # same co-trade window as the real pipeline
DEFAULT_FRONTEND = ROOT.parent / "frontend"

TRADES_PER_WALLET = {              # expected independent trades over the period
    "Whale": 14,
    "Frequent trader": 24,
    "Occasional trader": 10,
    "New wallet": 6,
}


# --------------------------------------------------------------------------
# Mock data
# --------------------------------------------------------------------------

def pick_wallets(relationships: list[dict], max_wallets: int) -> tuple[list[str], list[dict]]:
    """A connected, readable slice of one market: breadth-first from the busiest wallets."""
    adj: dict[str, set[str]] = defaultdict(set)
    for r in relationships:
        adj[r["wallet_a_id"]].add(r["wallet_b_id"])
        adj[r["wallet_b_id"]].add(r["wallet_a_id"])

    def degree(w: str) -> int:
        return len(adj[w])

    order = sorted(adj, key=lambda w: (-degree(w), w))
    selected: list[str] = []
    seen: set[str] = set()
    for start in order:
        if len(selected) >= max_wallets:
            break
        if start in seen:
            continue
        queue = [start]
        while queue and len(selected) < max_wallets:
            w = queue.pop(0)
            if w in seen:
                continue
            seen.add(w)
            selected.append(w)
            queue.extend(sorted(adj[w] - seen, key=lambda x: (-degree(x), x)))

    keep = set(selected)
    rels = [r for r in relationships if r["wallet_a_id"] in keep and r["wallet_b_id"] in keep]
    return selected, rels


def simulate_trades(market_id: str, wallet_ids: list[str], rels: list[dict],
                    wallets: dict[str, dict], rng: np.random.Generator,
                    days: int = 60) -> pd.DataFrame:
    T = days * DAY
    rows: list[dict] = []

    def trade(wid: str, t: float, outcome: str, side: str, price: float, size: float):
        rows.append({
            "market_id": market_id,
            "wallet": wallets[wid]["address"],
            "token_id": f"{market_id}-{outcome}",
            "outcome": outcome,
            "side": side,
            "size": float(max(size, 1.0)),
            "price": float(np.clip(price, 0.01, 0.99)),
            "timestamp": int(START_TS + t),
            "tx_hash": f"0x{len(rows):064x}",
        })

    def own_trade(wid: str, t: float):
        w = wallets[wid]
        outcome = "Yes" if rng.random() < w.get("yes_share", 0.5) else "No"
        base = w.get("average_entry_price", 0.5)
        price = base if outcome == "Yes" else 1 - base
        side = "BUY" if rng.random() < 0.8 else "SELL"
        size = rng.lognormal(np.log(max(w.get("median_trade_size_usd", 100), 5)), 0.6)
        trade(wid, t, outcome, side, price + rng.normal(0, 0.04), size)
        return outcome, side, price, size

    # 1. Independent trading.
    for wid in wallet_ids:
        lam = TRADES_PER_WALLET.get(wallets[wid]["wallet_type"], 20)
        for t in rng.uniform(0, T, rng.poisson(lam)):
            own_trade(wid, t)

    # 2. Market-wide news moments: many wallets react within a few minutes.
    for t0 in rng.uniform(0, T, 5):
        crowd = rng.choice(wallet_ids, size=max(3, int(len(wallet_ids) * rng.uniform(0.15, 0.3))),
                           replace=False)
        for wid in crowd:
            own_trade(str(wid), t0 + rng.uniform(0, 240))

    # 3. Related pairs trade together. Some relationships stop early, some
    #    start late, so the held-out window is not a copy of the past.
    for r in rels:
        a, b = r["wallet_a_id"], r["wallet_b_id"]
        s = float(r["relationship_strength"])
        phase = rng.random()
        lo, hi = (0.0, 0.55) if phase < 0.2 else (0.5, 1.0) if phase < 0.35 else (0.0, 1.0)
        sessions = rng.poisson((3 + 12 * s) * (hi - lo) + 2)
        median_gap = max(float(r.get("median_time_gap_seconds", 20.0)), 1.0)
        for t in rng.uniform(lo * T, hi * T, sessions):
            leader, follower = (a, b) if rng.random() < 0.6 else (b, a)
            outcome, side, price, size = own_trade(leader, t)
            gap = min(rng.exponential(median_gap / np.log(2)), WINDOW_SECONDS - 20)
            if rng.random() < r.get("same_side_rate", 0.7):
                f_outcome, f_side, f_price = outcome, side, price + rng.normal(0, 0.01)
            else:
                f_outcome = "No" if outcome == "Yes" else "Yes"
                f_side, f_price = side, 1 - price
            trade(follower, t + gap, f_outcome, f_side, f_price, size * rng.uniform(0.6, 1.4))

    return pd.DataFrame(rows).sort_values("timestamp", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# Graph preparation (numpy / pandas only)
# --------------------------------------------------------------------------

def prepare(trades: pd.DataFrame, train_fraction: float):
    edges = build_temporal_edges(trades, window_seconds=WINDOW_SECONDS)
    bundle = build_graph(edges, build_wallet_features(trades))
    split = temporal_split(bundle, train_fraction)

    # Train-only node features, exactly as `prepare_model_bundle` does for the
    # tuned model: nothing from the test window leaks into the node encoder.
    fit = trades[trades["timestamp"] <= split.cutoff_ts]
    features = build_wallet_features(fit, wallets=bundle.wallet_to_id)
    bundle = build_graph(bundle.edges, features)
    return bundle, split


def pair_counts(bundle, idx: np.ndarray) -> dict[tuple[int, int], int]:
    src, tgt = bundle.edge_index[0][idx], bundle.edge_index[1][idx]
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for s, t in zip(src.tolist(), tgt.tolist()):
        counts[(min(s, t), max(s, t))] += 1
    return counts


# --------------------------------------------------------------------------
# Model (needs torch)
# --------------------------------------------------------------------------

def replay_pair_scores(result, bundle, split, batch_size: int):
    """Score every candidate partner for every test event, chronologically.

    Mirrors `trainer.score_events`: a batch is scored with the memory and pair
    history from before it, then absorbed. Returns an (n, n) symmetric matrix of
    average partner probabilities, plus top-1 accuracy and MRR when the true
    partner is ranked against every other wallet (not just sampled negatives).
    """
    import torch

    from polymap.models.pair_history import PairHistory
    from polymap.training.trainer import normalise_edge_attr

    model = result.model
    model.eval()
    device = torch.device(result.device)
    n = bundle.num_nodes
    src, tgt = bundle.edge_index
    ts = bundle.timestamps
    test = split.test_idx
    edge_attr, _ = normalise_edge_attr(bundle.edge_attr, split.train_idx)
    ea = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    memory = result.memory.clone().to(device)
    history = PairHistory(n)
    history.replay(bundle.edge_index, ts, split.train_idx)

    def t_long(a):
        return torch.as_tensor(np.asarray(a), dtype=torch.long, device=device)

    prob_sum = np.zeros((n, n))
    events = np.zeros(n)
    top1 = 0
    rr_sum = 0.0
    candidates = np.arange(n)
    with torch.no_grad():
        for start in range(0, len(test), batch_size):
            idx = test[start:start + batch_size]
            s, t, tm = src[idx], tgt[idx], ts[idx]
            B = len(idx)
            S = np.repeat(s, n)
            C = np.tile(candidates, B)
            feats = history.features(S, C, np.repeat(tm, n))
            logits = model.score(memory, t_long(S), t_long(C),
                                 torch.as_tensor(feats, dtype=torch.float32, device=device)).view(B, n)
            logits[torch.arange(B, device=device), t_long(s)] = float("-inf")   # never yourself
            p = torch.softmax(logits, dim=1).cpu().numpy()
            np.add.at(prob_sum, s, p)
            np.add.at(events, s, 1)
            top1 += int((p.argmax(axis=1) == t).sum())
            true_p = p[np.arange(B), t]
            rr_sum += float((1.0 / (1 + (p > true_p[:, None]).sum(axis=1))).sum())
            memory = model.advance(memory, t_long(s), t_long(t), ea[t_long(idx)])
            history.update(s, t, tm)

    denom = events[:, None] + events[None, :]
    score = np.where(denom > 0, (prob_sum + prob_sum.T) / np.maximum(denom, 1), 0.0)
    k = max(len(test), 1)
    return score, {"top1_all_wallets": top1 / k, "mrr_all_wallets": rr_sum / k}


def use_mixed_negatives(p_hard: float = 0.5) -> None:
    """Register a "mixed" training sampler: each negative is hard or uniformly random.

    The backend's trainer picks its sampler by name through `build_sampler`, so
    the mixed sampler is added there. Evaluation still uses hard negatives.

    Why: with hard negatives only, every wrong answer the model sees in training
    is a past partner, so it learns "never traded together = likely partner".
    That is invisible under the hard-negative benchmark but makes the model
    favour strangers when it must choose among ALL wallets, which is what the
    prediction graph asks of it. Random negatives remove that shortcut.
    """
    import polymap.training.trainer as trainer_mod
    from polymap.training import negatives as neg

    class MixedNegativeSampler(neg.NegativeSampler):
        name = "mixed"

        def __init__(self, edge_index, train_idx, num_nodes):
            self.hard = neg.HardNegativeSampler(edge_index, train_idx, num_nodes)
            self.random = neg.RandomNegativeSampler(num_nodes)

        def sample_many(self, sources, rng, n=1):
            h = self.hard.sample_many(sources, rng, n)
            r = self.random.sample_many(sources, rng, n)
            return np.where(rng.random(h.shape) < p_hard, h, r)

    original = neg.build_sampler

    def build_sampler(kind, edge_index, train_idx, num_nodes):
        if kind == "mixed":
            return MixedNegativeSampler(edge_index, train_idx, num_nodes)
        return original(kind, edge_index, train_idx, num_nodes)

    trainer_mod.build_sampler = build_sampler


def run_model(bundle, split, epochs: int, seed: int, n_negatives: int,
              train_negatives: str = "mixed") -> dict:

    from polymap.config import ModelConfig
    from polymap.evaluation.baseline_eval import score_baselines
    from polymap.evaluation.metrics import classification_metrics, ranking_metrics
    from polymap.training.negatives import build_sampler
    from polymap.training.trainer import score_events, train

    cfg = ModelConfig(
        pair_features=True,
        train_only_features=True,
        learning_rate=0.003,
        epochs=epochs,
        seed=seed,
        eval_negatives=n_negatives,
        eval_batch_size=16,
        device="cpu",
    )

    if train_negatives == "mixed":
        use_mixed_negatives()
    result = train(bundle, split, cfg, negative=train_negatives)

    # Held-out ranking evaluation on test events (same protocol as the paper numbers).
    sampler = build_sampler("hard", bundle.edge_index, split.train_idx, bundle.num_nodes)
    scored = score_events(result, bundle, split, sampler, seed=seed,
                          n_negatives=n_negatives, batch_size=cfg.eval_batch_size)
    tgnn = {**ranking_metrics(scored["positive"], scored["negative"]),
            **classification_metrics(scored["positive"], scored["negative"].ravel())}
    base = score_baselines(bundle, split, scored["neg_ids"], cfg.eval_batch_size, ("recency",))
    recency = ranking_metrics(base["recency"]["positive"], base["recency"]["negative"])

    # Turn the model into a graph the way it is actually used: replay the test
    # window event by event. Each time a wallet trades, the model scores EVERY
    # other wallet as its possible partner (softmax over candidates), using
    # memory and pair history as they were just before that event. Only then is
    # the event absorbed. A pair's score is the average probability the model
    # gave each wallet as the other's partner across their test-window trades.
    pair_score, all_wallets = replay_pair_scores(result, bundle, split, cfg.eval_batch_size)
    n = bundle.num_nodes
    ii, jj = np.triu_indices(n, k=1)
    prob = pair_score[ii, jj]

    return {
        "ii": ii, "jj": jj, "prob": prob, "matrix": pair_score,
        "metrics": {
            "test_events": int(len(scored["positive"])),
            "negatives_per_event": n_negatives,
            "mrr": round(float(tgnn["mrr"]), 4),
            "hits@10": round(float(tgnn["hits@10"]), 4),
            "roc_auc": round(float(tgnn["roc_auc"]), 4),
            "recency_mrr": round(float(recency["mrr"]), 4),
            "recency_hits@10": round(float(recency["hits@10"]), 4),
            "top1_all_wallets": round(all_wallets["top1_all_wallets"], 4),
            "mrr_all_wallets": round(all_wallets["mrr_all_wallets"], 4),
            "final_train_loss": round(float(result.losses[-1]), 4) if result.losses else None,
        },
    }


# --------------------------------------------------------------------------
# Real data (the tuned run's candidate graph)
# --------------------------------------------------------------------------

def load_real(run: str, train_fraction: float):
    """Rebuild exactly the graph the tuned model was trained on.

    Candidate-only edges (both wallets in candidate_wallets.txt), chronological
    split, then train-only node features: the same steps as `stage_train` with
    candidates_only=True and model.train_only_features=True.
    """
    d = ROOT / "data" / "processed" / run
    for f in ("trades.parquet", "temporal_edges.parquet", "candidate_wallets.txt"):
        if not (d / f).exists():
            raise SystemExit(f"{d / f} not found. Run the backend pipeline for '{run}' first.")
    trades = pd.read_parquet(d / "trades.parquet")
    edges = pd.read_parquet(d / "temporal_edges.parquet")
    keep = set((d / "candidate_wallets.txt").read_text().split())
    sub = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]
    bundle = build_graph(sub, build_wallet_features(trades))
    split = temporal_split(bundle, train_fraction)
    fit = trades[trades["timestamp"] <= split.cutoff_ts]
    bundle = build_graph(bundle.edges, build_wallet_features(fit, wallets=bundle.wallet_to_id))
    return trades, bundle, split


def real_wallet_types(trades: pd.DataFrame, bundle, split) -> list[str]:
    """Rough wallet labels for colouring, from pre-cutoff trading only."""
    fit = trades[trades["timestamp"] <= split.cutoff_ts]
    f = build_wallet_features(fit, per_market=False).set_index("wallet")
    big = f["avg_trade_size"].quantile(0.85)
    busy = f["trade_count"].quantile(0.70)
    start, cutoff = int(bundle.timestamps.min()), int(split.cutoff_ts)
    late = start + 0.6 * (cutoff - start)
    out = []
    for nid in range(bundle.num_nodes):
        w = bundle.id_to_wallet[nid]
        if w not in f.index:
            out.append("New wallet")
            continue
        r = f.loc[w]
        if r["avg_trade_size"] >= big:
            out.append("Whale")
        elif r["first_trade"] >= late:
            out.append("New wallet")
        elif r["trade_count"] >= busy:
            out.append("Frequent trader")
        else:
            out.append("Occasional trader")
    return out


def market_record(market_id, name, n_trades, bundle, split, out, wallet_meta, display, truth_min):
    """Package one market for the frontend, restricted to the `display` node ids."""
    train_counts = pair_counts(bundle, split.train_idx)
    test_counts = pair_counts(bundle, split.test_idx)
    display = list(display)
    M = out["matrix"]
    pairs = []
    for x in range(len(display)):
        for y in range(x + 1, len(display)):
            i, j = sorted((display[x], display[y]))
            pairs.append([x, y, round(float(M[i, j]), 4),
                          test_counts.get((i, j), 0), train_counts.get((i, j), 0)])
    ts = bundle.timestamps
    n_true = sum(1 for p in pairs if p[3] >= truth_min)
    log.info("%s: MRR %.3f vs hard negs (recency %.3f) | MRR %.3f, top-1 %.3f vs all %d wallets | "
             "showing %d wallets, %d pairs, %d co-traded in test",
             name, out["metrics"]["mrr"], out["metrics"]["recency_mrr"],
             out["metrics"]["mrr_all_wallets"], out["metrics"]["top1_all_wallets"],
             bundle.num_nodes, len(display), len(pairs), n_true)
    return {
        "market_id": market_id,
        "market_name": name,
        "trades": int(n_trades),
        "edges": int(bundle.num_edges),
        "wallets_in_model": int(bundle.num_nodes),
        "train_events": int(len(split.train_idx)),
        "test_events": int(len(split.test_idx)),
        "train_window": [int(ts.min()), int(split.cutoff_ts)],
        "test_window": [int(split.cutoff_ts), int(ts.max())],
        "metrics": out["metrics"],
        "wallets": [wallet_meta[i] for i in display],
        # [wallet index a, wallet index b, prediction score (avg partner probability),
        #  co-trade edges in test window, co-trade edges in train window]
        "pairs": pairs,
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--demo", type=Path, default=DEFAULT_FRONTEND / "data" / "polymap_demo.json",
                    help="frontend demo dataset (wallets, markets, relationships)")
    ap.add_argument("--out", type=Path, default=DEFAULT_FRONTEND / "data" / "model_predictions.json")
    ap.add_argument("--markets", type=int, default=12, help="how many demo markets to run")
    ap.add_argument("--wallets", type=int, default=36, help="wallets per market")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--negatives", type=int, default=20)
    ap.add_argument("--train-fraction", type=float, default=0.7)
    ap.add_argument("--truth-min-events", type=int, default=None,
                    help="a pair counts as 'co-traded in the test window' with at least this many "
                         "co-trade edges (default 2 for mock, 5 for --real)")
    ap.add_argument("--real", nargs="?", const="xi_2027_63d_p50", default=None, metavar="RUN",
                    help="use a real backend run instead of mock trades (default run: xi_2027_63d_p50)")
    ap.add_argument("--show-wallets", type=int, default=48,
                    help="with --real: how many of the most active wallets to draw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-negatives", choices=["mixed", "hard", "random"], default="mixed",
                    help="negatives used in TRAINING (evaluation always uses hard negatives)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("polymap.features.edges", "polymap.graph.build", "polymap.training.splits"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    truth_min = args.truth_min_events or (5 if args.real else 2)
    out_markets = []

    if args.real:
        t0 = time.time()
        trades, bundle, split = load_real(args.real, args.train_fraction)
        log.info("real run %s: %d wallets, %d edges (%d train / %d test)", args.real,
                 bundle.num_nodes, bundle.num_edges, len(split.train_idx), len(split.test_idx))
        out = run_model(bundle, split, args.epochs, args.seed, args.negatives, args.train_negatives)
        types = real_wallet_types(trades, bundle, split)
        meta = [{"wallet_id": f"w{nid:03d}", "address": bundle.id_to_wallet[nid],
                 "wallet_type": types[nid]} for nid in range(bundle.num_nodes)]
        # Draw the wallets with the most co-trades before the cutoff (no test info used).
        activity = np.bincount(bundle.edge_index[0][split.train_idx], minlength=bundle.num_nodes) \
            + np.bincount(bundle.edge_index[1][split.train_idx], minlength=bundle.num_nodes)
        display = np.argsort(-activity, kind="stable")[: args.show_wallets]
        out_markets.append(market_record(
            args.real, "Xi 2027 market (real trades)", len(trades), bundle, split, out, meta,
            sorted(display.tolist()), truth_min))
        log.info("done in %.1f min", (time.time() - t0) / 60)
    else:
        demo = json.loads(args.demo.read_text())
        wallets = {w["wallet_id"]: w for w in demo["wallets"]}
        by_address = {w["address"]: w for w in demo["wallets"]}
        for m_i, market in enumerate(demo["markets"][: args.markets]):
            mid = market["market_id"]
            rels_all = [r for r in demo["candidate_relationships"] if mid in r["market_ids"]]
            wallet_ids, rels = pick_wallets(rels_all, args.wallets)
            rng = np.random.default_rng(args.seed + m_i)
            trades = simulate_trades(mid, wallet_ids, rels, wallets, rng)
            bundle, split = prepare(trades, args.train_fraction)
            out = run_model(bundle, split, args.epochs, args.seed, args.negatives, args.train_negatives)
            meta = []
            for nid in range(bundle.num_nodes):
                w = by_address[bundle.id_to_wallet[nid]]
                meta.append({"wallet_id": w["wallet_id"], "address": w["address"],
                             "wallet_type": w["wallet_type"]})
            out_markets.append(market_record(
                mid, market["name"], len(trades), bundle, split, out, meta,
                range(bundle.num_nodes), truth_min))

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "backend/predict_mock.py",
        "data": (f"REAL trades from backend run {args.real}" if args.real else
                 "MOCK trades simulated from the frontend demo dataset; model code is the real PolymapTGN"),
        "is_mock_data": not args.real,
        "model": {
            "name": "PolymapTGN (tuned settings)",
            "hidden_dim": 32,
            "temperature": 0.1,
            "pair_history_head": True,
            "train_only_features": True,
            "train_negatives": args.train_negatives,
            "epochs": args.epochs,
            "learning_rate": 0.003,
            "seed": args.seed,
        },
        "window_seconds": WINDOW_SECONDS,
        "train_fraction": args.train_fraction,
        "truth_min_events": truth_min,
        "markets": out_markets,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, separators=(",", ":")))
    log.info("wrote %s (%d markets)", args.out, len(out_markets))


if __name__ == "__main__":
    main()
