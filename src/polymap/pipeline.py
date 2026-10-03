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


def stage_train(cfg: Config, bundle=None, negative: str = "random") -> dict:
    from .training.negatives import build_sampler
    from .training.splits import temporal_split
    from .training.trainer import score_events, train
    from .evaluation.metrics import classification_metrics, ranking_metrics

    if bundle is None:
        bundle = stage_build(cfg)["bundle"]

    split = temporal_split(bundle, cfg.split.train_fraction)
    result = train(bundle, split, cfg.model, negative=negative)

    metrics = {}
    for kind in ("random", "hard", "popularity"):
        sampler = build_sampler(kind, bundle.edge_index, split.train_idx, bundle.num_nodes)
        scored = score_events(result, bundle, split, sampler, seed=cfg.model.seed)
        metrics[kind] = {
            **classification_metrics(scored["positive"], scored["negative"].ravel()),
            **ranking_metrics(scored["positive"], scored["negative"]),
        }
        if kind == negative:
            _write(scored["events"], cfg.processed_dir / "event_scores.parquet",
                   "event_scores")

    (cfg.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _manifest(cfg, "train", {"negative_sampler": negative,
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


def run_all(cfg: Config, refresh: bool = False, negative: str = "random") -> dict:
    trades = stage_fetch(cfg, refresh=refresh)
    built = stage_build(cfg, trades)
    trained = stage_train(cfg, built["bundle"], negative=negative)
    screened = stage_screen(cfg)
    return {**built, **trained, "screened": screened}
