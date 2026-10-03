"""Request and response models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Stage = Literal["fetch", "build", "train", "screen", "all"]
JobStatus = Literal["queued", "running", "done", "error"]


class JobRequest(BaseModel):
    config_paths: list[str] = Field(default_factory=list)
    overrides: dict[str, Any] = Field(default_factory=dict)
    refresh: bool = False
    negative: Literal["random", "hard", "popularity"] = "random"


class JobCreated(BaseModel):
    job_id: str


class JobView(BaseModel):
    id: str
    stage: Stage
    status: JobStatus
    run_name: str
    started_at: str | None
    finished_at: str | None
    logs: list[str]
    error: str | None
    result: dict[str, Any] | None


class RunSummary(BaseModel):
    run_name: str
    path: str
    has_trades: bool
    has_features: bool
    has_edges: bool
    has_events: bool
    has_screening: bool
    has_metrics: bool


class MarketRow(BaseModel):
    market_id: str | None
    gamma_id: str
    slug: str | None
    question: str | None
    volume: float
    resolved: bool
    closed: bool
    end_date: str | None
