"""FastAPI app."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .. import pipeline
from ..config import Config, load_config
from .jobs import registry
from .schemas import JobCreated, JobRequest, JobView, MarketRow, RunSummary

log = logging.getLogger(__name__)

app = FastAPI(title="polymap", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_cfg(req: JobRequest) -> Config:
    return load_config(*req.config_paths, **req.overrides)


def _json_safe(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if limit is not None:
        df = df.head(limit)
    clean = df.where(pd.notnull(df), None)
    return clean.to_dict(orient="records")


def _summarize_build(value: dict) -> dict:
    bundle = value.get("bundle")
    return {
        "n_trades": len(value["trades"]),
        "n_wallets_features": len(value["features"]),
        "n_edges": len(value["edges"]),
        "graph": bundle.summary() if bundle is not None else None,
    }


def _summarize_train(value: dict) -> dict:
    split = value["split"]
    result = value["result"]
    return {
        "metrics": value["metrics"],
        "split": split.summary(),
        "loss_curve_tail": result.losses[-20:],
        "n_loss_steps": len(result.losses),
        "device": result.device,
    }


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/markets", response_model=list[MarketRow])
def markets(
    min_volume: float = 0.0,
    closed: bool | None = True,
    limit: int = 40,
) -> list[dict]:
    from ..data.trades import discover_markets

    cfg = load_config()
    df = discover_markets(cfg, min_volume=min_volume,
                          closed=closed, limit=limit)
    if df.empty:
        return []
    keep = ["market_id", "gamma_id", "slug", "question",
            "volume", "resolved", "closed", "end_date"]
    return _json_safe(df[keep])


@app.get("/runs", response_model=list[RunSummary])
def runs() -> list[dict]:
    base = Config().processed_dir.parent
    if not base.exists():
        return []
    out = []
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        artifacts = Config().artifacts_dir.parent / d.name
        out.append(
            {
                "run_name": d.name,
                "path": str(d),
                "has_trades": (d / "trades.parquet").exists(),
                "has_features": (d / "wallet_features.parquet").exists(),
                "has_edges": (d / "temporal_edges.parquet").exists(),
                "has_events": (d / "event_scores.parquet").exists(),
                "has_screening": (d / "candidate_relationships.parquet").exists(),
                "has_metrics": (artifacts / "metrics.json").exists(),
            }
        )
    return out


def _run_dir(run_name: str) -> Path:
    cfg = load_config(run_name=run_name)
    if not cfg.processed_dir.exists():
        raise HTTPException(
            404, f"run '{run_name}' has no processed directory")
    return cfg.processed_dir


@app.get("/runs/{run_name}/metrics")
def run_metrics(run_name: str) -> dict:
    path = load_config(run_name=run_name).artifacts_dir / "metrics.json"
    if not path.exists():
        raise HTTPException(404, f"no metrics.json for run '{run_name}'")
    return json.loads(path.read_text())


@app.get("/runs/{run_name}/screening")
def run_screening(
    run_name: str,
    category: str | None = Query(None),
    limit: int = Query(500, le=5000),
) -> list[dict]:
    path = _run_dir(run_name) / "candidate_relationships.parquet"
    if not path.exists():
        raise HTTPException(
            404, "no candidate_relationships.parquet; run `screen` first")
    df = pd.read_parquet(path)
    if category:
        df = df[df["category"] == category]
    return _json_safe(df, limit=limit)


@app.get("/runs/{run_name}/edges")
def run_edges(run_name: str, limit: int = Query(200, le=2000)) -> list[dict]:
    path = _run_dir(run_name) / "temporal_edges.parquet"
    if not path.exists():
        raise HTTPException(
            404, "no temporal_edges.parquet; run `build` first")
    df = pd.read_parquet(path)
    return _json_safe(df, limit=limit)


@app.get("/runs/{run_name}/wallets")
def run_wallets(run_name: str) -> list[dict]:
    path = _run_dir(run_name) / "wallet_features.parquet"
    if not path.exists():
        raise HTTPException(
            404, "no wallet_features.parquet; run `build` first")
    df = pd.read_parquet(path)
    keep = [
        "wallet", "market_id",
        "trade_count", "buy_count", "sell_count",
        "avg_trade_size", "median_trade_size", "total_size",
        "avg_price", "price_std", "unique_outcomes",
        "buy_ratio", "sell_ratio",
        "first_trade", "last_trade", "active_duration_hours", "trade_frequency",
        "log_trade_count", "log_avg_trade_size", "log_median_trade_size", "log_active_duration",
    ]
    keep = [c for c in keep if c in df.columns]
    return _json_safe(df[keep])


@app.get("/runs/{run_name}/config")
def run_config(run_name: str) -> dict:
    cfg = load_config(run_name=run_name)
    return {"config": cfg.to_dict(), "fingerprint": cfg.fingerprint()}


@app.post("/jobs/fetch", response_model=JobCreated)
def job_fetch(req: JobRequest) -> dict:
    cfg = _build_cfg(req)
    job = registry.submit(
        "fetch",
        cfg.run_name,
        fn=lambda: pipeline.stage_fetch(cfg, refresh=req.refresh),
        summarize=lambda df: {
            "n_trades": int(len(df)),
            "n_wallets": int(df["wallet"].nunique()) if len(df) else 0,
            "refreshed": req.refresh,
        },
    )
    return {"job_id": job.id}


@app.post("/jobs/build", response_model=JobCreated)
def job_build(req: JobRequest) -> dict:
    cfg = _build_cfg(req)
    job = registry.submit("build", cfg.run_name, fn=lambda: pipeline.stage_build(
        cfg), summarize=_summarize_build)
    return {"job_id": job.id}


@app.post("/jobs/train", response_model=JobCreated)
def job_train(req: JobRequest) -> dict:
    cfg = _build_cfg(req)
    job = registry.submit(
        "train",
        cfg.run_name,
        fn=lambda: pipeline.stage_train(cfg, negative=req.negative),
        summarize=_summarize_train,
    )
    return {"job_id": job.id}


@app.post("/jobs/screen", response_model=JobCreated)
def job_screen(req: JobRequest) -> dict:
    cfg = _build_cfg(req)
    job = registry.submit(
        "screen",
        cfg.run_name,
        fn=lambda: pipeline.stage_screen(cfg),
        summarize=lambda df: {
            "n_candidates": int(len(df)),
            "categories": df["category"].value_counts().to_dict(),
        },
    )
    return {"job_id": job.id}


@app.post("/jobs/all", response_model=JobCreated)
def job_all(req: JobRequest) -> dict:
    cfg = _build_cfg(req)
    job = registry.submit(
        "all",
        cfg.run_name,
        fn=lambda: pipeline.run_all(
            cfg, refresh=req.refresh, negative=req.negative),
        summarize=lambda out: {
            "n_trades": len(out["trades"]),
            "n_edges": len(out["edges"]),
            "graph": out["bundle"].summary(),
            "metrics": out["metrics"],
            "n_candidates": len(out["screened"]),
        },
    )
    return {"job_id": job.id}


@app.get("/jobs", response_model=list[JobView])
def jobs() -> list[dict]:
    return [j.view() for j in registry.list()]


@app.get("/jobs/{job_id}", response_model=JobView)
def job(job_id: str) -> dict:
    j = registry.get(job_id)
    if j is None:
        raise HTTPException(404, f"no job '{job_id}'")
    return j.view()
