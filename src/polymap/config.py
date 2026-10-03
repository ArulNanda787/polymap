"""Configuration.

Every number that changes a result lives here, not scattered through notebooks.
This matters more than it looks: the 300-second window, the crowd threshold and
the screening criteria are all research choices, and a reviewer asking "what
happens at 60 seconds?" should be answered by editing one line, not by hunting
through twelve notebooks for hardcoded constants.

Load order: dataclass defaults, then `configs/default.yaml`, then a market file,
then CLI overrides. Later wins.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # yaml is optional; JSON configs work without it
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ApiConfig:
    gamma_base: str = "https://gamma-api.polymarket.com"
    data_base: str = "https://data-api.polymarket.com"
    clob_base: str = "https://clob.polymarket.com"
    timeout_sec: int = 30
    max_retries: int = 4
    sleep_sec: float = 0.2
    page_size: int = 100
    # Gamma rejects offsets past roughly this value with HTTP 422.
    max_offset: int = 2_000


@dataclass(frozen=True)
class GraphConfig:
    window_seconds: int = 300
    # 0 disables. Caps how many partners a single trade may generate.
    max_pairs_per_trade: int = 0
    # 0 disables. Drops timestamps where the whole market reacts at once;
    # those moments produce co-timing with no relational meaning.
    drop_crowd_timestamps: int = 0
    chunk_size: int = 2_000_000


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 32
    temperature: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    bptt_chunk: int = 2_000
    seed: int = 42
    device: str = "auto"      # auto | cpu | cuda | mps


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.7


@dataclass(frozen=True)
class ScreeningConfig:
    min_events: int = 3
    persistent_min_span_days: float = 7.0
    persistent_min_timestamps: int = 3
    tight_max_median_time_diff: float = 5.0
    tight_max_median_price_diff: float = 0.01


@dataclass(frozen=True)
class Config:
    markets: list[str] = field(default_factory=list)
    max_trades_per_market: int = 10_000
    api: ApiConfig = field(default_factory=ApiConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    data_dir: str = "data"
    run_name: str = "default"

    # ------------------------------------------------------------- paths --
    @property
    def raw_dir(self) -> Path:
        return REPO_ROOT / self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return REPO_ROOT / self.data_dir / "processed" / self.run_name

    @property
    def artifacts_dir(self) -> Path:
        return REPO_ROOT / self.data_dir / "artifacts" / self.run_name

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.artifacts_dir):
            d.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Stable hash of every setting that affects a result.

        Written into each run manifest so you can tell whether two result sets
        are comparable without diffing config files by eye.
        """
        import hashlib
        payload = {k: v for k, v in self.to_dict().items()
                   if k not in ("data_dir", "run_name")}
        blob = json.dumps(payload, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


_SECTIONS = {"api": ApiConfig, "graph": GraphConfig, "model": ModelConfig,
             "split": SplitConfig, "screening": ScreeningConfig}


def _merge(base: Config, data: dict[str, Any]) -> Config:
    updates: dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTIONS and isinstance(value, dict):
            updates[key] = replace(getattr(base, key), **value)
        elif hasattr(base, key):
            updates[key] = value
        else:
            raise ValueError(f"unknown config key: {key}")
    return replace(base, **updates)


def load_config(*paths: str | Path, **overrides: Any) -> Config:
    """Build a Config from defaults, then files in order, then keyword overrides.

        cfg = load_config("configs/default.yaml", "configs/markets/iran.yaml",
                          run_name="window60")
    """
    cfg = Config()
    for p in paths:
        if p is None:
            continue
        path = Path(p)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"config not found: {path}")
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            if yaml is None:
                raise ImportError("PyYAML is needed for .yaml configs: pip install pyyaml")
            data = yaml.safe_load(text) or {}
        else:
            data = json.loads(text)
        cfg = _merge(cfg, data)
    if overrides:
        cfg = _merge(cfg, overrides)
    return cfg
