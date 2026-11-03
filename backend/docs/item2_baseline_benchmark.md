# Item 2 — Fair comparison against strong baselines

**Status:** complete (both markets, 5 seeds, 4 negative-sampling protocols)
**Run:** 30 Oct 2026, `run_benchmark.py`, outputs in `data/artifacts/benchmark/`
**Question:** does the learned temporal model add value beyond simple baselines, consistently, across markets?

## 1. Bottom line

**No, not in general.** The tuned model beats every baseline (including TGN) in
both markets, every seed, **only under the hard-negative protocol it was
trained with** — by about +0.15 MRR over recency, with tight intervals. Under
the three other protocols (active, random, popularity) the simple **recency
baseline wins in both markets**, and the tuned model is significantly worse than
recency in all six market × protocol cells.

Part of the hard-negative advantage is a protocol shortcut (Section 5): hard
negatives are, by construction, wallets the source has *already* traded with, so
a positive pair that has *never* been seen before is easy to pick out from a
"seen before" feature. Removing those never-seen events shrinks but does not
eliminate the hard-negative advantage (Xi +0.10, Iran +0.06 over recency).

What the paper can claim:

> Under hard (history-based) negatives, the memory model with causal pair
> history ranks future co-trading partners better than recency, EdgeBank, TGN and
> neighbourhood heuristics in two markets (MRR gap vs recency +0.149 and +0.156,
> 95% CIs excluding zero; 5/5 seeds in each market). The advantage does not
> transfer to random, popularity-weighted or concurrently-active negatives,
> where a 24-hour decayed co-trade count (recency) is the strongest method.
> The model's gains are therefore protocol-specific.

What it cannot claim: that the learned model consistently outperforms simple
methods.

## 2. What was compared

Every method saw exactly the same, per market:

| Held fixed | Value |
|---|---|
| Graph | candidate-wallet induced co-trading graph (300 s window) |
| Split | chronological 70/30, cut between distinct timestamps |
| Negatives | one fixed 50-negative matrix per protocol, drawn once (seed 12345), independent of model seeds |
| Collision rule | a negative is never the source, nor any wallet that co-traded with the source at that same second |
| Batching | timestamp-grouped (an event never sees another event from the same second) |
| Scored events | one test row per (pair, second) — duplicate rows collapsed |
| Seeds | 42–46 for every learned model; non-learned baselines are deterministic |

Methods:

| Method | Type | Notes |
|---|---|---|
| Tuned (ours) | learned | node memory + 6 causal pair-history features, 20 epochs, lr 0.003, hard training negatives |
| Plain (ours) | learned | node memory only, 10 epochs, lr 0.003 |
| TGN | learned | PyG reference TGN (memory + graph attention), 5 epochs, hard training negatives |
| Recency | heuristic | co-trade count decayed with a 24 h half-life, updated online |
| EdgeBank ∞ / 24 h / 1 h | heuristic | 1 if the pair co-traded ever / in the last 24 h / in the last hour |
| Pair count | heuristic | prior co-trade count |
| Common neighbours, Adamic–Adar | heuristic | shared prior partners (Adamic–Adar down-weights busy partners) |
| Popularity | heuristic | candidate's prior degree |

**TGAT was not run.** It needs per-layer temporal neighbour attention; a
faithful version should be taken from DyGLib rather than reimplemented. TGN is
the closer and stronger comparator for a memory-based model. State this in the paper.

Negative-sampling protocols:

| Protocol | Negatives are… | Advantages |
|---|---|---|
| hard | wallets the source co-traded with during training | models that know who is *new*; penalises history baselines |
| active | wallets in any trade in the hour before the event | the realistic "who else was trading right then?" question |
| random | uniform over the candidate universe | easiest; rewards anything activity-aware |
| popularity | proportional to training degree | does not disadvantage frequency baselines |

Data:

| | Xi | Iran |
|---|---:|---:|
| Trade window (UTC) | 18 Aug – 20 Oct 2026 | 28 Aug – 30 Oct 2026 (**new pull**) |
| Candidate wallets / pairs | 163 / 397 | 388 / 1,365 |
| Train / test events | 54,813 / 23,493 | 165,151 / 70,801 |
| Test events scored (after dedupe) | 9,638 | 38,163 |
| Test events with a train-unseen wallet | 3,205 | 19,882 |
| Active protocol fell back to uniform | 13.0% of negatives | 8.9% |

The Iran sample is a fresh 63-day pull (the earlier `rep_iran` artifacts were
deleted). It is a different, later window than the report's Iran run, so Iran
numbers here are not comparable to the earlier report's Iran table.

## 3. Why these numbers differ from the earlier report

Earlier Xi: tuned MRR 0.586, recency 0.433. Now: 0.498 and 0.349. The protocol
is stricter:

1. **Same-second leakage removed.** Previously, two events at the same second
   could fall in different batches, so the second was scored after the first had
   updated memory and pair history.
2. **Duplicate rows collapsed.** Repeated rows of the same pair at the same
   second counted as many correct predictions; 59% (Xi) and 46% (Iran) of test
   rows were such duplicates.
3. **Collision-free negatives.** A "negative" can no longer be a true partner at
   that second.
4. **One negative set for all methods and seeds.** Previously each seed drew its
   own negatives.

Do not mix old and new numbers in one table.

## 4. Results

### Xi


**Negatives: hard**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| **Tuned (ours)** | 0.498 ± 0.023 | 0.707 | 0.791 | 0.478 | 0.717 |
| Recency | 0.349 | 0.578 | 0.738 | 0.377 | 0.027 |
| TGN | 0.344 ± 0.044 | 0.587 | 0.765 | 0.360 | 0.169 |
| EdgeBank 1 h | 0.277 | 0.528 | 0.706 | 0.295 | 0.076 |
| Plain (ours) | 0.231 ± 0.016 | 0.391 | 0.633 | 0.242 | 0.111 |
| EdgeBank 24 h | 0.226 | 0.531 | 0.737 | 0.241 | 0.064 |
| Adamic–Adar | 0.146 | 0.251 | 0.480 | 0.153 | 0.060 |
| Pair count | 0.144 | 0.215 | 0.479 | 0.155 | 0.027 |
| Common neighbours | 0.141 | 0.252 | 0.477 | 0.148 | 0.062 |
| Popularity | 0.125 | 0.210 | 0.497 | 0.130 | 0.079 |
| EdgeBank ∞ | 0.099 | 0.224 | 0.492 | 0.105 | 0.027 |

**Negatives: active**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| Recency | 0.326 | 0.411 | 0.550 | 0.353 | 0.033 |
| TGN | 0.298 ± 0.025 | 0.402 | 0.563 | 0.309 | 0.179 |
| Pair count | 0.259 | 0.360 | 0.553 | 0.279 | 0.033 |
| **Tuned (ours)** | 0.258 ± 0.024 | 0.316 | 0.506 | 0.246 | 0.393 |
| Plain (ours) | 0.256 ± 0.004 | 0.327 | 0.533 | 0.262 | 0.186 |
| Popularity | 0.243 | 0.349 | 0.538 | 0.253 | 0.137 |
| Adamic–Adar | 0.233 | 0.328 | 0.530 | 0.243 | 0.120 |
| Common neighbours | 0.222 | 0.326 | 0.529 | 0.231 | 0.118 |
| EdgeBank 24 h | 0.143 | 0.277 | 0.518 | 0.151 | 0.057 |
| EdgeBank 1 h | 0.115 | 0.216 | 0.464 | 0.120 | 0.066 |
| EdgeBank ∞ | 0.107 | 0.230 | 0.521 | 0.114 | 0.033 |

**Negatives: random**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| Recency | 0.505 | 0.862 | 0.868 | 0.547 | 0.038 |
| EdgeBank 1 h | 0.387 | 0.604 | 0.743 | 0.414 | 0.080 |
| EdgeBank 24 h | 0.373 | 0.748 | 0.818 | 0.400 | 0.071 |
| TGN | 0.356 ± 0.065 | 0.694 | 0.822 | 0.373 | 0.163 |
| Pair count | 0.345 | 0.754 | 0.792 | 0.373 | 0.038 |
| Adamic–Adar | 0.302 | 0.616 | 0.680 | 0.322 | 0.077 |
| Common neighbours | 0.299 | 0.604 | 0.674 | 0.318 | 0.080 |
| Popularity | 0.297 | 0.690 | 0.796 | 0.313 | 0.115 |
| Plain (ours) | 0.233 ± 0.019 | 0.415 | 0.656 | 0.245 | 0.092 |
| **Tuned (ours)** | 0.208 ± 0.079 | 0.357 | 0.604 | 0.213 | 0.153 |
| EdgeBank ∞ | 0.137 | 0.346 | 0.644 | 0.146 | 0.038 |

**Negatives: popularity**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| Recency | 0.341 | 0.610 | 0.769 | 0.369 | 0.030 |
| TGN | 0.328 ± 0.041 | 0.589 | 0.774 | 0.343 | 0.154 |
| **Tuned (ours)** | 0.273 ± 0.056 | 0.463 | 0.675 | 0.278 | 0.219 |
| EdgeBank 1 h | 0.270 | 0.540 | 0.710 | 0.287 | 0.075 |
| Plain (ours) | 0.219 ± 0.015 | 0.380 | 0.634 | 0.230 | 0.093 |
| EdgeBank 24 h | 0.218 | 0.540 | 0.744 | 0.232 | 0.062 |
| Pair count | 0.160 | 0.298 | 0.602 | 0.172 | 0.030 |
| Adamic–Adar | 0.131 | 0.239 | 0.501 | 0.139 | 0.044 |
| Common neighbours | 0.128 | 0.242 | 0.497 | 0.136 | 0.046 |
| EdgeBank ∞ | 0.104 | 0.244 | 0.561 | 0.111 | 0.030 |
| Popularity | 0.099 | 0.149 | 0.478 | 0.104 | 0.039 |

### Iran


**Negatives: hard**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| **Tuned (ours)** | 0.502 ± 0.010 | 0.693 | 0.786 | 0.463 | 0.711 |
| Recency | 0.346 | 0.558 | 0.701 | 0.403 | 0.030 |
| TGN | 0.332 ± 0.009 | 0.539 | 0.727 | 0.371 | 0.122 |
| EdgeBank 1 h | 0.252 | 0.484 | 0.673 | 0.285 | 0.070 |
| EdgeBank 24 h | 0.245 | 0.531 | 0.708 | 0.278 | 0.062 |
| Pair count | 0.159 | 0.238 | 0.450 | 0.182 | 0.030 |
| Adamic–Adar | 0.150 | 0.251 | 0.461 | 0.166 | 0.062 |
| Common neighbours | 0.145 | 0.247 | 0.460 | 0.160 | 0.063 |
| Plain (ours) | 0.130 ± 0.005 | 0.266 | 0.548 | 0.136 | 0.100 |
| EdgeBank ∞ | 0.109 | 0.232 | 0.477 | 0.123 | 0.030 |
| Popularity | 0.100 | 0.180 | 0.429 | 0.103 | 0.082 |

**Negatives: active**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| Recency | 0.294 | 0.370 | 0.522 | 0.342 | 0.032 |
| Pair count | 0.253 | 0.350 | 0.547 | 0.293 | 0.032 |
| TGN | 0.249 ± 0.004 | 0.348 | 0.504 | 0.275 | 0.103 |
| Adamic–Adar | 0.218 | 0.313 | 0.523 | 0.243 | 0.079 |
| **Tuned (ours)** | 0.217 ± 0.016 | 0.281 | 0.466 | 0.180 | 0.418 |
| Common neighbours | 0.213 | 0.312 | 0.524 | 0.237 | 0.081 |
| Popularity | 0.204 | 0.326 | 0.537 | 0.221 | 0.111 |
| Plain (ours) | 0.167 ± 0.002 | 0.250 | 0.499 | 0.172 | 0.139 |
| EdgeBank 24 h | 0.132 | 0.241 | 0.469 | 0.148 | 0.046 |
| EdgeBank ∞ | 0.108 | 0.232 | 0.508 | 0.122 | 0.032 |
| EdgeBank 1 h | 0.096 | 0.185 | 0.426 | 0.104 | 0.052 |

**Negatives: random**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| Recency | 0.493 | 0.760 | 0.830 | 0.576 | 0.042 |
| TGN | 0.395 ± 0.010 | 0.636 | 0.788 | 0.443 | 0.134 |
| Pair count | 0.372 | 0.660 | 0.777 | 0.432 | 0.042 |
| EdgeBank 24 h | 0.352 | 0.644 | 0.761 | 0.404 | 0.069 |
| EdgeBank 1 h | 0.337 | 0.541 | 0.704 | 0.385 | 0.075 |
| Adamic–Adar | 0.305 | 0.590 | 0.705 | 0.344 | 0.088 |
| Common neighbours | 0.299 | 0.585 | 0.701 | 0.337 | 0.090 |
| Popularity | 0.220 | 0.532 | 0.739 | 0.241 | 0.107 |
| EdgeBank ∞ | 0.174 | 0.438 | 0.705 | 0.198 | 0.042 |
| **Tuned (ours)** | 0.150 ± 0.032 | 0.194 | 0.516 | 0.142 | 0.194 |
| Plain (ours) | 0.131 ± 0.006 | 0.286 | 0.570 | 0.138 | 0.091 |

**Negatives: popularity**

| Method | MRR (mean ± SD) | Hits@10 | AUC | Seen-pair MRR | Never-seen MRR |
|---|---:|---:|---:|---:|---:|
| Recency | 0.372 | 0.631 | 0.750 | 0.434 | 0.033 |
| TGN | 0.339 ± 0.010 | 0.568 | 0.744 | 0.382 | 0.108 |
| EdgeBank 1 h | 0.269 | 0.510 | 0.684 | 0.306 | 0.070 |
| EdgeBank 24 h | 0.258 | 0.567 | 0.724 | 0.294 | 0.061 |
| Pair count | 0.207 | 0.382 | 0.598 | 0.238 | 0.033 |
| **Tuned (ours)** | 0.205 ± 0.029 | 0.396 | 0.629 | 0.194 | 0.266 |
| Adamic–Adar | 0.146 | 0.268 | 0.474 | 0.165 | 0.040 |
| Common neighbours | 0.141 | 0.261 | 0.470 | 0.159 | 0.041 |
| Plain (ours) | 0.125 ± 0.004 | 0.264 | 0.551 | 0.132 | 0.091 |
| EdgeBank ∞ | 0.118 | 0.274 | 0.574 | 0.133 | 0.033 |
| Popularity | 0.053 | 0.075 | 0.320 | 0.057 | 0.035 |

## Paired gaps: tuned minus each comparator (MRR, 95% block-bootstrap CI)

| Market | Negatives | vs Recency | vs TGN | vs EdgeBank 1 h | vs Pair count | vs Plain |
|---|---|---|---|---|---|---|
| Xi | hard | +0.149 [+0.116, +0.184] | +0.153 [+0.122, +0.184] | +0.221 [+0.190, +0.252] | +0.353 [+0.322, +0.384] | +0.267 [+0.239, +0.295] |
| Xi | active | -0.068 [-0.098, -0.040] | -0.040 [-0.074, -0.006] | +0.143 [+0.113, +0.175] | -0.001 [-0.041, +0.040] | +0.003 [-0.028, +0.033] |
| Xi | random | -0.297 [-0.326, -0.268] | -0.148 [-0.175, -0.123] | -0.179 [-0.208, -0.150] | -0.137 [-0.167, -0.106] | -0.025 [-0.044, -0.005] |
| Xi | popularity | -0.068 [-0.095, -0.041] | -0.054 [-0.079, -0.031] | +0.003 [-0.024, +0.029] | +0.113 [+0.089, +0.138] | +0.055 [+0.033, +0.075] |
| Iran | hard | +0.156 [+0.120, +0.187] | +0.169 [+0.144, +0.194] | +0.250 [+0.220, +0.277] | +0.343 [+0.306, +0.379] | +0.371 [+0.349, +0.393] |
| Iran | active | -0.077 [-0.109, -0.047] | -0.032 [-0.060, -0.004] | +0.121 [+0.100, +0.142] | -0.036 [-0.070, -0.001] | +0.050 [+0.031, +0.069] |
| Iran | random | -0.343 [-0.376, -0.311] | -0.245 [-0.273, -0.220] | -0.187 [-0.221, -0.155] | -0.222 [-0.256, -0.189] | +0.019 [+0.002, +0.039] |
| Iran | popularity | -0.168 [-0.203, -0.136] | -0.135 [-0.161, -0.109] | -0.065 [-0.099, -0.032] | -0.002 [-0.039, +0.031] | +0.080 [+0.064, +0.098] |

## Paired gaps: TGN minus recency

| Market | hard | active | random | popularity |
|---|---|---|---|---|
| Xi | -0.004 [-0.036, +0.028] | -0.028 [-0.065, +0.010] | -0.149 [-0.182, -0.118] | -0.013 [-0.045, +0.017] |
| Iran | -0.013 [-0.034, +0.006] | -0.045 [-0.071, -0.019] | -0.098 [-0.113, -0.084] | -0.033 [-0.051, -0.016] |

Per-seed: tuned beats recency in **5/5 seeds in both markets under hard
negatives and 0/5 seeds in every other market × protocol cell**. TGN beats
recency in 2/5 (Xi) and 1/5 (Iran) seeds under hard negatives.

Intervals are contiguous time-block bootstrap intervals (50 blocks, 2,000
draws) on the seed-averaged per-event reciprocal rank. They capture
event-sampling uncertainty, not seed-to-seed training variance; seed SDs are in
the tables above.

## 5. The never-seen-pair shortcut

"Never-seen" test events are pairs with no earlier co-trade. Under the hard
protocol, every negative is a previous partner of the source, so a never-seen
positive differs from all 50 negatives in one obvious way: its pair-history
features are all zero, including the "seen" flag.

| Never-seen MRR, hard negatives | Xi | Iran |
|---|---:|---:|
| Tuned | 0.717 | 0.711 |
| TGN | 0.169 | 0.122 |
| Recency | 0.027 | 0.030 |

Tuned ranks these brand-new pairs first most of the time. That is not
predictive skill. It learned during hard-negative training that "unseen pair
= positive", which holds under this protocol only. The same model drops to
0.15–0.19 on never-seen events under random negatives, where most negatives are
also unseen.

On previously seen pairs only (recent + stale), the hard-negative advantage
shrinks but remains:

| Seen-pair MRR, hard | Tuned | Recency | Gap |
|---|---:|---:|---:|
| Xi | 0.478 | 0.377 | +0.101 |
| Iran | 0.463 | 0.403 | +0.060 |

(Point estimates; no interval was computed for this post-hoc stratum.)

**Implication for item 3:** the training-negative ablation (`neg_train` block,
which trains with random and popularity negatives) and the `no_count_features`
ablation (removes the count and seen-flag features) directly test this
explanation.

## 6. Interpretation

1. **The model's advantage is protocol-specific.** It is large and robust under
   hard negatives and absent or reversed elsewhere. The paper should report all
   four protocols, not only hard.
2. **Recency is the strongest general baseline.** It wins under active, random
   and popularity negatives in both markets. A 24-hour decayed co-trade count is
   the benchmark to beat.
3. **TGN does not beat recency either.** It is statistically indistinguishable
   from recency under hard negatives, level with or below it under active and
   popularity, and clearly below it under random. This agrees with the
   temporal-graph benchmarking literature: simple memorisation baselines are hard
   to beat on recurrent-interaction data.
4. **Pair history is what lifts tuned over plain**, mainly under hard and
   popularity negatives (and Iran active, +0.050). Under random negatives the two are close (Xi −0.025,
   Iran +0.019). Note that the tuned-vs-plain comparison here still differs in
   epochs (20 vs 10); the controlled version is in the item 3 ablations.
5. **The results replicate across markets.** Every sign in the tuned-vs-recency
   row is the same in Xi and Iran. The finding is consistent, it is just not the
   finding the original report hoped for.

## 7. Recommended framing for the paper

* Present the model as competitive with simple baselines, with an advantage
  confined to history-based negatives, rather than as superior.
* Lead the evaluation with the active protocol: it matches the real question
  ("of the wallets trading at that moment, which one is the partner?").
* Report never-seen and seen strata separately for every method.
* Retrain the model with random or mixed negatives (item 3 `neg_train`) before
  the frozen prospective test (item 4), and choose the training protocol on
  validation data, not on these test numbers.

## 8. Limitations

* Two markets, both geopolitical "before 2027" markets.
* The candidate universe is selected on the full observation window (fixed in
  item 4).
* Xi's test period was already inspected in earlier work; this benchmark adds no
  tuning on it, but it is not a pristine holdout.
* The active protocol fell back to uniform for 9–13% of negatives (too few other
  wallets active in the preceding hour).
* TGN ran for 5 epochs with default PyG hyperparameters and was not tuned; the
  tuned model received more tuning effort. This favours our model, which
  strengthens the negative result.

## 9. Reproduce

```bash
python run_benchmark.py -c configs/default.yaml \
  --run-names xi_2027_63d_p50 rep_iran \
  --samplers hard active random popularity \
  --variants tuned plain --seeds 42 43 44 45 46 \
  --tgn --tgn-epochs 5 --outdir data/artifacts/benchmark
```

Files: `per_seed.csv` (every model × seed × protocol, with strata),
`summary.csv`, `paired.csv` (all pairwise gaps with CIs), `cross_market.csv`,
`protocol.json` (negative-set diagnostics). Code changes made before this run:
fixed the memory-off training path, fixed the TGN import for current PyG, and
TGN is now trained once per seed and scored on all four negative sets from
the same trained state.
