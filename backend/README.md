# Polymap

Temporal graph learning for wallet relationship discovery in prediction markets.

---

## The one idea behind the structure

**Every stage reads from disk and writes to disk.** Nothing is passed in memory
between steps.

```
fetch  →  data/raw/trades_<market>.parquet        + .manifest.json (sha256)
build  →  data/processed/<run>/trades.parquet
                                wallet_features.parquet
                                temporal_edges.parquet
train  →  data/artifacts/<run>/metrics.json, *_manifest.json
          data/processed/<run>/event_scores.parquet
screen →  data/processed/<run>/candidate_relationships.parquet
```

That single decision is what makes the project debuggable. Any stage can be
rerun alone, its output opened in a notebook, and a failure localised to one
step. The old notebook chain passed ten loose variables between cells, so a
stale `wallet_to_id` could silently corrupt a result three notebooks later and
nothing would tell you.

---

## Running it

```bash
pip install -e ".[model,dev]"

polymap markets --min-volume 200000          # find condition IDs
# put those IDs in configs/markets/<name>.yaml, then:
polymap all -c configs/default.yaml -c configs/markets/iran_2027.yaml
```

Or one stage at a time, which is how you should work while debugging:

```bash
polymap fetch  -c configs/markets/iran_2027.yaml
polymap build  -c configs/markets/iran_2027.yaml
polymap train  -c configs/markets/iran_2027.yaml --negative hard
polymap screen -c configs/markets/iran_2027.yaml
```

For the leak-free pair-history model, use explicit overrides so the experiment
is visible in the command and included in the run fingerprint:

```bash
polymap train -c configs/default.yaml -c configs/markets/xi_2027.yaml \
  --run-name xi_2027_63d_p50 --negative hard \
  --set model.pair_features=true \
  --set model.train_only_features=true \
  --set model.learning_rate=0.003 --set model.epochs=20
```

Tune only on a validation window; `tune_tgnn.py` never scores the final test
window:

```bash
python tune_tgnn.py -c configs/default.yaml -c configs/markets/xi_2027.yaml \
  --run-name xi_2027_63d_p50 --candidates-only --negative hard \
  --set model.pair_features=true --set model.train_only_features=true \
  --set split.validation_split=0.15
```

Run a variant without touching the original:

```bash
polymap build -c configs/markets/iran_2027.yaml --window 60 --run-name window60
```

Outputs land in `data/processed/window60/`. Nothing is overwritten, and the
config fingerprint in each manifest tells you whether two result sets are
comparable.

---

## What lives where

| Path | Responsibility |
| --- | --- |
| `config.py` | Every number that changes a result |
| `data/schema.py` | Canonical column names, API field mapping, validation |
| `data/client.py` | HTTP: retry, pagination, honest error surfacing |
| `data/trades.py` | Market discovery, trade fetching, **immutable caching** |
| `features/edges.py` | Temporal interaction edges (vectorised) |
| `features/wallets.py` | Node features |
| `graph/build.py` | Tensors + `GraphBundle` |
| `models/temporal.py` | The custom memory-based temporal model |
| `models/pair_history.py` | Causal pair-level history features |
| `models/baselines.py` | Frequency, recency, popularity |
| `training/negatives.py` | Random / hard / popularity samplers |
| `training/splits.py` | Chronological split |
| `training/trainer.py` | Training and scoring loop |
| `evaluation/metrics.py` | AUC, AP, MRR, Hits@k |
| `analysis/relationships.py` | Margin aggregation and screening |
| `pipeline.py` | Stage orchestration |
| `cli.py` | `polymap <stage>` |

---

## Five things that changed, and why

**1. Edge construction is ~150x faster.** The original used nested `for` loops
with `.iloc` on every row access. Measured against a literal transcription of
that loop: 1,600 trades took 68.1s before and 0.44s now, with byte-identical
output (`tests/test_edges.py` asserts this). 100,000 trades now build 601,749
edges in 0.32s. The old version would have taken roughly 45 minutes on the
10,000-trade slice alone.

**2. Memory is bounded.** The first rewrite still materialised every pair index
at once and died on 5 markets × 8,000 trades. Row-block chunking fixed it:
40,000 trades → 3,972,708 edges in 1.8s inside a fixed envelope.

**3. Raw slices are cached and hashed.** This is the reproducibility fix, not a
speed optimisation. The Data API caps a market at ~10,000 trades and returns a
*different* slice each call — an earlier pull of market 665374 covered 19 Jul to
20 Sep 2026 with 2,358 wallets; a later identical request returned 10 to 26 Sep
with ~1,351. Every fetch now writes a manifest with a SHA-256, row count and
observed time span, and later runs read the cache unless you pass `--refresh`.
You can now state in a paper which artifact produced a result.

**4. Markets never join.** `build_edges_multi_market` builds each market
separately and refuses to link wallets across them. Within one market all
wallets face the same question and resolution, so co-timing is interpretable.
Across markets, two wallets trading simultaneously usually share nothing but a
news cycle.

**5. A popularity baseline exists.** Under hard negatives the negative is drawn
from the source's own interaction history, so any history-based baseline is
evaluated on a set built adversarially against it and can score below chance.
`metrics.py` flags AUC < 0.5 explicitly rather than letting it read as a clean
win. `PopularityBaseline` is the reference the protocol does not disadvantage.

**6. Pair history is an explicit ablation.** The node memory model has no
pair-level state or clock, so the optional six-feature causal history head adds
pair count, three decayed counts, a seen flag and pair age. It is evaluated on
the same negatives as the model. Call the result a custom memory-based temporal
model; it is not a reimplementation of TGN, TGAT or JODIE.

---

## How to add things

**A new market.** Add the condition ID to a config file's `markets:` list and
rerun. Nothing else changes — the whole pipeline is already multi-market.

**A new node or edge feature.** Add the column in `features/wallets.py` or
`features/edges.py`, then add its name to `NODE_FEATURE_COLUMNS` or
`EDGE_FEATURE_COLUMNS` in `data/schema.py`. The graph builder picks it up
automatically. *Do not* add anything derived from market resolution to node
features: `test_features_have_no_outcome_leakage` will fail, and it should.

**A new baseline.** Subclass `Baseline` in `models/baselines.py`, implement
`fit` and `score`, register it in `ALL_BASELINES`.

**A new negative sampler.** Subclass `NegativeSampler`, add it to
`build_sampler`. Document what it advantages and disadvantages — that is the
part that decides whether a comparison is fair.

**A new model.** Anything exposing `initial_memory`, `score` and `advance` works
with the existing trainer.

**A new pipeline stage.** Add `stage_x(cfg)` in `pipeline.py` reading from
`cfg.processed_dir` and writing back, then add it to the CLI choices.

**An API field renamed upstream.** Add the new name to `TRADE_FIELD_MAP` in
`data/schema.py`. One file, nothing else.

---

## Tests

```bash
pytest -q          # 23 tests
```

The important one is `test_matches_naive`: it runs the original nested-loop
implementation alongside the vectorised one and asserts identical output. If
the fast path ever drifts, that fails. Keep it.

Also enforced: no self-loops, the window is respected, markets are never
joined, features contain no outcome leakage, splits are chronological, the
config fingerprint changes when a research parameter changes, and below-chance
AUC is flagged.

---

## Known limits

- **`models/`, `training/trainer.py` are syntax-checked but not executed here**
  (no PyTorch in the build environment). They are transcriptions of verified
  notebook code, but run `polymap train` on a small market first.
- **Simultaneous trades are linked with `time_diff == 0`**, matching the
  original loop. Direction between them reflects sort order, not causality.
- **Tightness is measured inside a window we imposed.** Edge construction
  truncates at `window_seconds` before the 5-second threshold is applied.
- **Screening output is candidates, not findings.** Two wallets reacting
  independently to the same news are indistinguishable here from two wallets
  under one operator.
- **Proxy wallets are not people.** One person may run several.

## Funding validation (Task 2)

The TGNN remains the behavioural candidate generator. To test whether its
candidate pairs have more observable funding links than pairs with comparable
co-timing, run a separate validation stage after `build`, `train`, and `screen`:

```bash
polymap validate-funding --run-name xi_2027 --funding-file /path/to/transfers.csv --controls-per-candidate 3
```

The funding input can be CSV, TSV, or Parquet and must contain normalized
columns `from_wallet`, `to_wallet`, and `timestamp`. `timestamp` may be Unix
seconds or a parseable datetime. Each row represents an observed on-chain
transfer. Keep the chain, token/asset filters, and collection time window in
your research notes; the validator cannot infer coverage from the file alone.

Outputs are written to `data/artifacts/<run-name>/`:

- `funding_validation_pairs.parquet`: candidate/control labels, co-timing counts,
  match distance, direct-transfer evidence, shared-funder evidence, and combined flag.
- `funding_validation.json`: candidate vs matched-control rates, risk differences,
  sample sizes, a descriptive permutation p-value, and limitations.

Controls are sampled from observed wallet pairs that are not TGNN candidates,
matched to each candidate by nearest `log1p(co-timing count)`. The report includes
matching distance and sample sizes; do not present the p-value as causal evidence.
The plain co-timing baseline is the observed interaction count, and must also be
compared with TGNN held-out ranking metrics from `metrics.json` rather than being
silently folded into the TGNN score.

**Interpretation constraint:** a direct transfer or shared funding source is a
candidate financial link, not proof of shared ownership, control, intent, or
coordination. Exchange, bridge, router, and custodial addresses can create
misleading shared-funder links; label/filter known service addresses before
making substantive claims. No ownership claims are produced by this stage.
