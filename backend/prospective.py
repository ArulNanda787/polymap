"""Frozen, prospective test: freeze -> build -> evaluate (once).

The problem this solves: the development test window was inspected, and the
wallet universe was chosen with the full observation period. Here the universe,
node features, screening statistics and all model choices depend ONLY on data
with timestamp <= T (the freeze time). The evaluation period is strictly later.

Stages
------
freeze    Hash the spec (hyperparameters, thresholds, hypotheses, success rule)
          and the code, and write freeze.json. Do this BEFORE the forward data
          exists (or at least before you look at it).
build     Merge dev trades (<= T) with forward trades (> T). Build edges, the
          candidate universe and node features from <= T ONLY. Writes the files
          the funding scripts expect (candidate_pairs.parquet etc.).
evaluate  Train on edges <= T, score edges > T once, with the frozen seeds,
          collision-free fixed negatives, timestamp-grouped batches, the
          pre-registered comparator and decision rule. Writes a ledger; a second
          evaluation is refused unless --allow-reevaluation (and is then flagged).

What is and is not causal here
------------------------------
* Edge timestamp = time of the LATER trade, so every edge with ts <= T involves
  only trades <= T. Building edges on the merged trades and splitting by edge
  timestamp is therefore leak-free; `max_pairs_per_trade` and
  `drop_crowd_timestamps` act per trade / per timestamp and are also causal.
* Candidate pairs, tightness and persistence statistics: edges <= T only.
* Node features: trades <= T only; wallets first seen after T get neutral
  zeros (build_wallet_features(wallets=...)).
* The scaler in build_graph is fitted over the node rows, which hold only <= T
  information.
* Simultaneous timestamps: see evaluation/protocol.py.
* Wallets that never appear before T can still appear in test edges only if
  they were selected as candidate wallets, which requires pre-T activity, so
  cold-start wallets are excluded by construction. State this in the paper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from polymap.analysis.relationships import candidate_wallets, prescreen_pairs, select_candidates
from polymap.config import GraphConfig, ScreeningConfig, load_config
from polymap.evaluation.experiment import (VariantSpec, paired, paired_table, run_baselines,
                                           run_variant, summarize)
from polymap.evaluation.protocol import build_eval_negatives, dedupe_mask
from polymap.features.edges import build_edges_multi_market
from polymap.features.wallets import build_wallet_features
from polymap.graph.build import build_graph
from polymap.models import baselines_extra  # noqa: F401
from polymap.training.splits import TemporalSplit

ROOT = Path(__file__).resolve().parent
CODE_GLOBS = ["src/polymap/**/*.py", "run_*.py", "prospective.py", "funding_confirmatory.py",
              "tune_tgnn.py"]
log = logging.getLogger("prospective")


# ------------------------------------------------------------------ hashing
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hashes() -> dict[str, str]:
    files = sorted({p for g in CODE_GLOBS for p in ROOT.glob(g) if p.is_file()})
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in files}


def spec_hash(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=str).encode()).hexdigest()


def pdir(name: str) -> Path:
    d = ROOT / "data" / "prospective" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_freeze(path: Path, allow_code_drift: bool) -> dict:
    fz = json.loads(Path(path).read_text())
    if spec_hash(fz["spec"]) != fz["spec_sha256"]:
        sys.exit("FREEZE VIOLATION: the spec stored in freeze.json no longer matches its hash.")
    now, then = code_hashes(), fz["code"]
    changed = sorted(f for f in set(now) | set(then) if now.get(f) != then.get(f))
    if changed and not allow_code_drift:
        sys.exit("FREEZE VIOLATION: code changed since freeze:\n  " + "\n  ".join(changed) +
                 "\nRe-run with --allow-code-drift only if the change cannot affect results, "
                 "and say so in the paper.")
    if changed:
        log.warning("code drift allowed for: %s", changed)
    fz["code_drift"] = changed
    return fz


# ------------------------------------------------------------------- freeze
def cmd_freeze(a) -> int:
    spec = yaml.safe_load(Path(a.spec).read_text())
    for key in ("name", "cutoff_ts", "graph", "universe", "model", "seeds", "eval", "primary"):
        if key not in spec:
            sys.exit(f"spec missing '{key}'")
    dev = pd.read_parquet(a.dev_trades)
    if dev["timestamp"].max() < spec["cutoff_ts"]:
        sys.exit("cutoff_ts is later than the last development trade")
    out = pdir(spec["name"]) / "freeze.json"
    if out.exists() and not a.overwrite:
        sys.exit(f"{out} exists. A freeze is not edited; choose a new name.")
    fz = {"spec": spec, "spec_sha256": spec_hash(spec), "code": code_hashes(),
          "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
          "dev_trades_file": str(a.dev_trades), "dev_trades_sha256": sha256_file(Path(a.dev_trades)),
          "dev_trades_max_ts": int(dev["timestamp"].max()),
          "dev_trades_used_n": int((dev["timestamp"] <= spec["cutoff_ts"]).sum())}
    out.write_text(json.dumps(fz, indent=2))
    print(f"frozen -> {out}\nspec sha256: {fz['spec_sha256']}\n"
          "Commit freeze.json (and the code) to version control NOW, before fetching forward data.")
    return 0


# -------------------------------------------------------------------- build
def cmd_build(a) -> int:
    fz = load_freeze(a.freeze, a.allow_code_drift)
    spec = fz["spec"]
    T = int(spec["cutoff_ts"])
    d = pdir(spec["name"])

    dev = pd.read_parquet(a.dev_trades)
    fwd = pd.read_parquet(a.forward_trades)
    if sha256_file(Path(a.dev_trades)) != fz["dev_trades_sha256"]:
        sys.exit("dev trades file differs from the one frozen")
    dev = dev[dev["timestamp"] <= T]
    fwd = fwd[fwd["timestamp"] > T]
    if fwd.empty:
        sys.exit("no forward trades after the cutoff")
    gap_days = (fwd["timestamp"].min() - T) / 86400
    if gap_days > 1:
        log.warning("first forward trade is %.1f days after the cutoff; the gap is missing data", gap_days)
    trades = pd.concat([dev, fwd], ignore_index=True)

    g = GraphConfig(**{**GraphConfig().__dict__, **spec["graph"]})
    edges = build_edges_multi_market(
        trades, window_seconds=g.window_seconds, max_pairs_per_trade=g.max_pairs_per_trade,
        drop_crowd_timestamps=g.drop_crowd_timestamps, chunk_size=g.chunk_size)
    e_dev = edges[edges["timestamp"] <= T]
    e_fwd = edges[edges["timestamp"] > T]

    # candidate universe from information available at T
    scr = ScreeningConfig(**{**ScreeningConfig().__dict__, **spec.get("screening", {}),
                             "min_events": spec["universe"].get("min_events", 3)})
    screened = prescreen_pairs(e_dev, scr)
    cats = tuple(spec["universe"].get("categories", ("persistent_tight", "one_off_tight")))
    chosen = select_candidates(screened, cats, int(spec["universe"].get("max_pairs", 0)))
    wallets = candidate_wallets(chosen)
    if not wallets:
        sys.exit("empty candidate universe at the cutoff")
    wset = set(wallets)

    sub_dev = e_dev[e_dev.source.isin(wset) & e_dev.target.isin(wset)]
    sub_fwd = e_fwd[e_fwd.source.isin(wset) & e_fwd.target.isin(wset)]
    feats = build_wallet_features(dev, wallets=sorted(wset))   # <= T only

    trades.to_parquet(d / "trades.parquet", index=False)
    e_dev.to_parquet(d / "temporal_edges.parquet", index=False)         # <= T, for control pools
    e_fwd.to_parquet(d / "temporal_edges_forward.parquet", index=False)
    feats.to_parquet(d / "wallet_features.parquet", index=False)
    chosen.to_parquet(d / "candidate_pairs.parquet", index=False)
    (d / "candidate_wallets.txt").write_text("\n".join(wallets) + "\n")
    manifest = {"cutoff_ts": T, "n_trades_dev": len(dev), "n_trades_forward": len(fwd),
                "n_edges_dev": len(e_dev), "n_edges_forward": len(e_fwd),
                "n_candidate_pairs": len(chosen), "n_candidate_wallets": len(wallets),
                "n_universe_edges_dev": len(sub_dev), "n_universe_edges_forward": len(sub_fwd),
                "forward_first_ts": int(fwd.timestamp.min()), "forward_last_ts": int(fwd.timestamp.max()),
                "forward_trades_sha256": sha256_file(Path(a.forward_trades)),
                "spec_sha256": fz["spec_sha256"], "code_drift": fz["code_drift"]}
    (d / "build_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    if len(sub_fwd) < 200:
        print("WARNING: very few forward events inside the universe; the test will be underpowered.")
    return 0


# ----------------------------------------------------------------- evaluate
def _load_bundle(d: Path, T: int):
    e = pd.concat([pd.read_parquet(d / "temporal_edges.parquet"),
                   pd.read_parquet(d / "temporal_edges_forward.parquet")], ignore_index=True)
    wset = set((d / "candidate_wallets.txt").read_text().split())
    e = e[e.source.isin(wset) & e.target.isin(wset)]
    feats = pd.read_parquet(d / "wallet_features.parquet")
    bundle = build_graph(e, feats)
    ts = bundle.timestamps
    train_idx, test_idx = np.where(ts <= T)[0], np.where(ts > T)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        sys.exit("empty train or test side of the split")
    src, tgt = bundle.edge_index
    deg = (np.bincount(src[train_idx], minlength=bundle.num_nodes)
           + np.bincount(tgt[train_idx], minlength=bundle.num_nodes))
    seen = (deg[src[test_idx]] > 0) & (deg[tgt[test_idx]] > 0)
    return bundle, TemporalSplit(train_idx, test_idx, int(T), seen, deg)


def cmd_evaluate(a) -> int:
    fz = load_freeze(a.freeze, a.allow_code_drift)
    spec = fz["spec"]
    T = int(spec["cutoff_ts"])
    d = pdir(spec["name"])
    ledger = d / "ledger.json"
    reeval = False
    if ledger.exists():
        if not a.allow_reevaluation:
            sys.exit(f"{ledger} exists: this prospective test was already evaluated. "
                     "A second look is not prospective. Use --allow-reevaluation to override; "
                     "the output will be flagged.")
        reeval = True

    cfg = load_config(run_name=spec["name"], model=dict(spec["model"]), graph=dict(spec["graph"]))
    bundle, split = _load_bundle(d, T)
    ev = spec["eval"]
    dedupe = bool(ev.get("dedupe_same_timestamp_pairs", True))
    eval_sets = {k: build_eval_negatives(bundle, split, k, int(ev.get("n_negatives", 50)),
                                         int(ev.get("eval_seed", 777)),
                                         int(ev.get("active_window_seconds", 3600)))
                 for k in ev["samplers"]}
    print(f"train {len(split.train_idx)} | test {len(split.test_idx)} "
          f"(after dedupe {int(dedupe_mask(bundle, split).sum())}) | "
          f"cold-start test events {split.n_cold_start}")

    m = spec["model"]
    vs = VariantSpec(name="frozen_model", pair_features=bool(m.get("pair_features", True)),
                     memory_updates=bool(m.get("memory_updates", True)),
                     train_negative=spec.get("train_negative", "hard"),
                     epochs=int(m["epochs"]), learning_rate=float(m["learning_rate"]),
                     hidden_dim=m.get("hidden_dim"), train_only_features=False)
    rows, rr = run_baselines(cfg, bundle, split, tuple(spec["baselines"]), eval_sets, dedupe=dedupe)
    r, rr_m = run_variant(cfg, bundle, split, vs, [int(s) for s in spec["seeds"]], eval_sets,
                          dedupe=dedupe, device=m.get("device", "cpu"))
    rows += r
    rr.update(rr_m)

    df = pd.DataFrame(rows)
    summ = summarize(df)
    pt = paired_table(rr, list(ev["samplers"]), ["frozen_model"], list(spec["baselines"]))
    df.to_csv(d / "per_seed.csv", index=False)
    summ.to_csv(d / "summary.csv", index=False)
    pt.to_csv(d / "paired.csv", index=False)

    prim = spec["primary"]
    res = paired(rr, prim["sampler"], "frozen_model", prim["comparator"])
    supported = bool(res["ci95_low"] > float(prim.get("min_gap", 0.0)))
    verdict = {"primary_endpoint": prim, "result": res, "hypothesis_supported": supported,
               "rule": f"supported iff 95% block-bootstrap CI lower bound of MRR gap > "
                       f"{prim.get('min_gap', 0.0)}",
               "reevaluation": reeval, "n_seeds": len(spec["seeds"]),
               "test_events": int(len(split.test_idx)),
               "test_events_after_dedupe": int(dedupe_mask(bundle, split).sum()),
               "negatives": {k: v.meta for k, v in eval_sets.items()},
               "spec_sha256": fz["spec_sha256"], "code_drift": fz["code_drift"]}
    (d / "verdict.json").write_text(json.dumps(verdict, indent=2))
    ledger.write_text(json.dumps({"evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "spec_sha256": fz["spec_sha256"], "reevaluation": reeval}, indent=2))
    print(summ.round(3).to_string(index=False))
    print("\nPRIMARY:", json.dumps(res), "->", "SUPPORTED" if supported else "NOT SUPPORTED")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--spec", required=True)
    f.add_argument("--dev-trades", required=True, help="processed/<dev_run>/trades.parquet")
    f.add_argument("--overwrite", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("--freeze", required=True)
    b.add_argument("--dev-trades", required=True)
    b.add_argument("--forward-trades", required=True, help="trades.parquet collected AFTER the freeze")
    b.add_argument("--allow-code-drift", action="store_true")
    e = sub.add_parser("evaluate")
    e.add_argument("--freeze", required=True)
    e.add_argument("--allow-code-drift", action="store_true")
    e.add_argument("--allow-reevaluation", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return {"freeze": cmd_freeze, "build": cmd_build, "evaluate": cmd_evaluate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
