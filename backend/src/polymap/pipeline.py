"""Stage orchestration.

Each stage reads from disk and writes to disk. That is the whole design, and
it is what makes the project debuggable: any stage can be rerun alone, its
output inspected in a notebook, and a failure localised to one step rather
than to "somewhere in a twelve-notebook chain".

    fetch  -> data/raw/trades_<market>.parquet          (+ .manifest.json)
    build  -> processed/<run>/trades.parquet
                               wallet_features.parquet
                               temporal_edges.parquet
    train  -> artifacts/<run>/event_scores.parquet, model.pt, metrics.json
    screen -> processed/<run>/candidate_relationships.parquet
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import Config
from .data.trades import fetch_many
from .features.edges import build_edges_multi_market
from .features.wallets import build_wallet_features
from .graph.build import build_graph

log = logging.getLogger(__name__)


def _write(df: pd.DataFrame, path, label: str) -> None:
    df.to_parquet(path, index=False)
    log.info("wrote %s: %d rows -> %s", label, len(df), path.name)


def _manifest(cfg: Config, stage: str, payload: dict) -> None:
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.artifacts_dir / f"{stage}_manifest.json"
    path.write_text(json.dumps({
        "stage": stage,
        "run_name": cfg.run_name,
        "config_fingerprint": cfg.fingerprint(),
        "written_at_utc": datetime.now(timezone.utc).isoformat(),
        **payload,
    }, indent=2, default=str))


def stage_fetch(cfg: Config, refresh: bool = False) -> pd.DataFrame:
    cfg.ensure_dirs()
    trades = fetch_many(cfg, refresh=refresh)
    _manifest(cfg, "fetch", {
        "markets": cfg.markets,
        "n_trades": len(trades),
        "n_wallets": int(trades["wallet"].nunique()) if len(trades) else 0,
    })
    return trades


def stage_build(cfg: Config, trades: pd.DataFrame | None = None) -> dict:
    cfg.ensure_dirs()
    if trades is None:
        trades = fetch_many(cfg, refresh=False)
    if trades.empty:
        raise ValueError("no trades. Run `polymap fetch` first.")

    _write(trades, cfg.processed_dir / "trades.parquet", "trades")

    features = build_wallet_features(trades)
    _write(features, cfg.processed_dir / "wallet_features.parquet", "wallet_features")

    edges = build_edges_multi_market(
        trades,
        window_seconds=cfg.graph.window_seconds,
        max_pairs_per_trade=cfg.graph.max_pairs_per_trade,
        drop_crowd_timestamps=cfg.graph.drop_crowd_timestamps,
        chunk_size=cfg.graph.chunk_size,
    )
    if edges.empty:
        raise ValueError(
            f"zero edges at window_seconds={cfg.graph.window_seconds}. "
            "Either the trades are too sparse or the window is too small.")
    _write(edges, cfg.processed_dir / "temporal_edges.parquet", "temporal_edges")

    bundle = build_graph(edges, features)
    _manifest(cfg, "build", {
        "window_seconds": cfg.graph.window_seconds,
        "n_trades": len(trades),
        "n_edges": len(edges),
        **bundle.summary(),
    })
    return {"trades": trades, "features": features, "edges": edges, "bundle": bundle}


def stage_candidates(cfg: Config, categories: tuple[str, ...] | None = None,
                     max_pairs: int = 0) -> pd.DataFrame:
    """Model-free candidate search. Needs only `build` output; no torch.

    Writes, under processed/<run>/:
        prescreen_pairs.parquet     every pair with >= min_events edges, categorised
        candidate_pairs.parquet     the selected subset
        candidate_wallets.txt       one address per line (feed this to the on-chain collector)
    """
    from .analysis.relationships import (
        DEFAULT_CANDIDATE_CATEGORIES,
        candidate_wallets,
        prescreen_pairs,
        select_candidates,
    )

    edges_path = cfg.processed_dir / "temporal_edges.parquet"
    if not edges_path.exists():
        raise FileNotFoundError(f"{edges_path} not found; run `polymap build` first")
    edges = pd.read_parquet(edges_path)
    screened = prescreen_pairs(edges, cfg.screening)
    _write(screened, cfg.processed_dir / "prescreen_pairs.parquet", "prescreen_pairs")

    cats = tuple(categories) if categories else DEFAULT_CANDIDATE_CATEGORIES
    chosen = select_candidates(screened, cats, max_pairs)
    _write(chosen, cfg.processed_dir / "candidate_pairs.parquet", "candidate_pairs")

    wallets = candidate_wallets(chosen)
    (cfg.processed_dir / "candidate_wallets.txt").write_text("\n".join(wallets) + ("\n" if wallets else ""))
    all_wallets = len(set(edges["source"]) | set(edges["target"]))
    log.info("candidates: %d pairs, %d wallets (of %d in the graph) -> candidate_wallets.txt",
             len(chosen), len(wallets), all_wallets)
    _manifest(cfg, "candidates", {
        "categories_selected": list(cats), "max_pairs": max_pairs,
        "n_pairs_screened": len(screened), "n_candidate_pairs": len(chosen),
        "n_candidate_wallets": len(wallets), "n_wallets_in_graph": all_wallets,
        "category_counts": screened["category"].value_counts().to_dict() if len(screened) else {},
    })
    return chosen


def _candidate_bundle(cfg: Config):
    """Graph restricted to edges whose BOTH endpoints are candidate wallets."""
    wallets_path = cfg.processed_dir / "candidate_wallets.txt"
    if not wallets_path.exists():
        raise FileNotFoundError(f"{wallets_path} not found; run `polymap candidates` first")
    keep = set(wallets_path.read_text().split())
    if not keep:
        raise ValueError("candidate_wallets.txt is empty; loosen --categories or the screening thresholds")
    edges = pd.read_parquet(cfg.processed_dir / "temporal_edges.parquet")
    features = pd.read_parquet(cfg.processed_dir / "wallet_features.parquet")
    sub = edges[edges["source"].isin(keep) & edges["target"].isin(keep)]
    log.info("candidates-only graph: %d of %d edges, %d candidate wallets",
             len(sub), len(edges), len(keep))
    if sub.empty:
        raise ValueError("no edges between candidate wallets")
    return build_graph(sub, features)


def prepare_model_bundle(cfg: Config, bundle, feature_indices: np.ndarray | None = None):
    """Return the graph bundle used by training.

    When requested, node features are rebuilt from a causal event prefix. The
    edge set and wallet IDs stay fixed, including wallets that only appear
    after the prefix; those wallets receive neutral fit-period features.
    """
    if not cfg.model.train_only_features:
        return bundle
    trades_path = cfg.processed_dir / "trades.parquet"
    if not trades_path.exists():
        raise FileNotFoundError(f"{trades_path} not found; run `polymap build` first")
    if feature_indices is None or len(feature_indices) == 0:
        raise ValueError("train-only features require a non-empty feature index")
    from .features.wallets import build_wallet_features

    trades = pd.read_parquet(trades_path)
    cutoff = int(bundle.timestamps[np.asarray(feature_indices, dtype=np.int64)].max())
    fit_trades = trades[trades["timestamp"] <= cutoff]
    features = build_wallet_features(fit_trades, wallets=bundle.wallet_to_id)
    log.info("train-only node features: %d trades through timestamp %d", len(fit_trades), cutoff)
    return build_graph(bundle.edges, features)


def stage_train(cfg: Config, bundle=None, negative: str = "random",
                candidates_only: bool = False,
                eval_samplers: tuple[str, ...] | None = None,
                baselines: tuple[str, ...] | None = None) -> dict:
    """Train the TGNN and evaluate it.

    `baselines`: names from `models.baselines.ALL_BASELINES` to score on the very
    same test events and negatives as the TGNN. None/empty = TGNN only.
    """
    from .evaluation.baseline_eval import evaluate_scores, paired_mrr_gap, score_baselines
    from .training.negatives import build_sampler
    from .training.splits import temporal_split
    from .training.trainer import score_events, train

    if candidates_only:
        bundle = _candidate_bundle(cfg)
    elif bundle is None:
        bundle = stage_build(cfg)["bundle"]

    split = temporal_split(bundle, cfg.split.train_fraction, cfg.split.validation_split)
    bundle = prepare_model_bundle(cfg, bundle, split.train_idx)
    result = train(bundle, split, cfg.model, negative=negative)

    metrics = {}
    comparison_rows = []
    w = cfg.graph.window_seconds
    kinds = tuple(eval_samplers) if eval_samplers else ("random", "hard", "popularity")
    if negative not in kinds:        # event_scores.parquet comes from this one
        kinds = (*kinds, negative)
    for kind in kinds:
        sampler = build_sampler(kind, bundle.edge_index, split.train_idx, bundle.num_nodes)
        scored = score_events(result, bundle, split, sampler, seed=cfg.model.seed,
                              n_negatives=cfg.model.eval_negatives,
                              batch_size=cfg.model.eval_batch_size)
        pos, neg, ev = scored["positive"], scored["negative"], scored["events"]

        gap = ev["gap_s"].to_numpy()
        metrics[kind] = evaluate_scores(pos, neg, gap, w)
        by_recency = metrics[kind]["by_recency"]
        log.info("[%s] AUC %.3f MRR %.3f hits@10 %.3f | by recency: %s", kind,
                 metrics[kind]["roc_auc"], metrics[kind]["mrr"], metrics[kind]["hits@10"],
                 {k: (v["n"], round(v["roc_auc"], 3)) for k, v in by_recency.items()})

        if baselines:
            base_scores = score_baselines(bundle, split, scored["neg_ids"],
                                          cfg.model.eval_batch_size, tuple(baselines))
            metrics[kind]["baselines"] = {}
            for name, sc in base_scores.items():
                m = evaluate_scores(sc["positive"], sc["negative"], gap, w)
                m["tgnn_minus_baseline_mrr"] = paired_mrr_gap(
                    pos, neg, sc["positive"], sc["negative"], seed=cfg.model.seed)
                metrics[kind]["baselines"][name] = m
                log.info("[%s] baseline %-18s AUC %.3f MRR %.3f hits@10 %.3f | TGNN-MRR gap %+.3f "
                         "(%+.3f..%+.3f)", kind, name, m["roc_auc"], m["mrr"], m["hits@10"],
                         m["tgnn_minus_baseline_mrr"]["mrr_gap"],
                         m["tgnn_minus_baseline_mrr"]["ci95_low"],
                         m["tgnn_minus_baseline_mrr"]["ci95_high"])
            for name, m in [("tgnn", metrics[kind]), *metrics[kind]["baselines"].items()]:
                comparison_rows.append({
                    "sampler": kind, "model": name, "n_test_events": len(pos),
                    **{k: m[k] for k in ("roc_auc", "average_precision", "mrr",
                                         "hits@1", "hits@5", "hits@10")},
                    **({"mrr_gap_tgnn_minus_this": m["tgnn_minus_baseline_mrr"]["mrr_gap"],
                        "gap_ci95_low": m["tgnn_minus_baseline_mrr"]["ci95_low"],
                        "gap_ci95_high": m["tgnn_minus_baseline_mrr"]["ci95_high"]}
                       if name != "tgnn" else {})})
        if kind == negative:
            _write(ev, cfg.processed_dir / "event_scores.parquet", "event_scores")

    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (cfg.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if comparison_rows:
        table = pd.DataFrame(comparison_rows)
        table.to_csv(cfg.artifacts_dir / "baseline_comparison.csv", index=False)
        log.info("wrote baseline_comparison.csv (%d rows)", len(table))
    _manifest(cfg, "train", {"negative_sampler": negative,
                             "candidates_only": candidates_only,
                             "baselines": list(baselines) if baselines else [],
                             "graph": bundle.summary(),
                             "split": split.summary(), "metrics": metrics})
    return {"bundle": bundle, "split": split, "result": result, "metrics": metrics}


def stage_screen(cfg: Config) -> pd.DataFrame:
    from .analysis.relationships import run_screening

    events = pd.read_parquet(cfg.processed_dir / "event_scores.parquet")
    edges = pd.read_parquet(cfg.processed_dir / "temporal_edges.parquet")
    screened = run_screening(events, edges, cfg.screening)
    _write(screened, cfg.processed_dir / "candidate_relationships.parquet",
           "candidate_relationships")
    _manifest(cfg, "screen", {"categories": screened["category"].value_counts().to_dict()})
    return screened


def stage_null(cfg: Config, n_null: int = 20, modes: tuple[str, ...] | None = None) -> dict:
    """Permutation baseline for the tight-pair screen. See analysis/null_model.py."""
    from .analysis.null_model import DEFAULT_MODES, run_null_model

    trades = pd.read_parquet(cfg.processed_dir / "trades.parquet")
    result = run_null_model(cfg, trades, n_null=n_null, modes=modes or DEFAULT_MODES)
    path = cfg.artifacts_dir / "null_model.json"
    cfg.artifacts_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    log.info("wrote null model results -> %s", path.name)
    _manifest(cfg, "null", {"n_null": n_null, "results": result})
    return result


def run_all(cfg: Config, refresh: bool = False, negative: str = "random") -> dict:
    trades = stage_fetch(cfg, refresh=refresh)
    built = stage_build(cfg, trades)
    trained = stage_train(cfg, built["bundle"], negative=negative)
    screened = stage_screen(cfg)
    return {**built, **trained, "screened": screened}

def stage_validate_funding(cfg: Config, funding_path: str,
                           controls_per_candidate: int = 3,
                           candidates_file: str | None = None,
                           max_funder_fanout: int = 100,
                           n_permutations: int = 2000) -> dict:
    """Compare candidate pairs with co-timing-matched controls using transfers.

    Candidates default to `candidate_pairs.parquet` (from `polymap candidates`)
    when it exists, else `candidate_relationships.parquet` (from `screen`).
    """
    from pathlib import Path

    from .analysis.funding_validation import run_funding_validation

    if candidates_file:
        candidates = Path(candidates_file)
    elif (cfg.processed_dir / "candidate_pairs.parquet").exists():
        candidates = cfg.processed_dir / "candidate_pairs.parquet"
    else:
        candidates = cfg.processed_dir / "candidate_relationships.parquet"
    log.info("funding validation candidates: %s", candidates)
    edges = cfg.processed_dir / "temporal_edges.parquet"
    if not candidates.exists():
        raise FileNotFoundError(f"{candidates} not found; run `polymap screen` first")
    if not edges.exists():
        raise FileNotFoundError(f"{edges} not found; run `polymap build` first")
    result = run_funding_validation(
        candidates, edges, funding_path, cfg.artifacts_dir,
        controls_per_candidate=controls_per_candidate, seed=cfg.model.seed,
        max_funder_fanout=max_funder_fanout, n_permutations=n_permutations)
    _manifest(cfg, "funding_validation", result)
    log.info("funding validation summary written to %s",
             cfg.artifacts_dir / "funding_validation.json")
    return result


def stage_funding_robustness(cfg: Config, funding_path: str,
                             candidates_file: str | None = None,
                             exclude_wallets_file: str | None = None,
                             caps: tuple[int, ...] | None = None) -> dict:
    """Anchor-fixed controls, hub sweeps, deposit-only evidence, bootstrap intervals."""
    from pathlib import Path

    from .analysis.funding_robustness import DEFAULT_CAPS, run_funding_robustness

    cand = Path(candidates_file) if candidates_file else cfg.processed_dir / "candidate_pairs.parquet"
    for need, hint in ((cand, "polymap candidates"),
                       (cfg.processed_dir / "temporal_edges.parquet", "polymap build"),
                       (cfg.processed_dir / "trades.parquet", "polymap build")):
        if not Path(need).exists():
            raise FileNotFoundError(f"{need} not found; run `{hint}` first")
    result = run_funding_robustness(
        cand, cfg.processed_dir / "temporal_edges.parquet", cfg.processed_dir / "trades.parquet",
        funding_path, cfg.artifacts_dir, exclude_wallets_file=exclude_wallets_file,
        caps=tuple(caps) if caps else DEFAULT_CAPS, seed=cfg.model.seed)
    _manifest(cfg, "funding_robustness", {k: v for k, v in result.items() if k != "table"})
    return result
