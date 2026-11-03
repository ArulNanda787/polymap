# Item 3 — Controlled ablations

**Status:** Xi complete. Iran components and training-negative blocks complete;
Iran window and minimum-event sweeps were still running when this was written
and are marked *pending* (regenerate the tables when they finish).
**Run:** 30 Oct 2026, `run_ablations.py`; outputs in `data/artifacts/<run>/ablations/`
**Question:** which components produce the model's performance, and are those
contributions robust?

## 1. Bottom line

1. **Pair history is the main contribution.** With everything else fixed,
   removing all six pair-history features costs −0.287 (Xi) and −0.360 (Iran)
   MRR under hard negatives. Without it the model falls well below recency.
2. **Node memory also helps under hard negatives.** Removing it costs −0.098
   (Xi) and −0.085 (Iran). Under active negatives it helps in Iran (−0.038
   without it) but makes no detectable difference in Xi (+0.015, CI includes 0).
3. **The decayed-count and age features carry the useful signal.** Removing
   them is almost as damaging as removing all of pair history. Removing the
   count and seen-flag features costs only −0.08/−0.07 under hard negatives and
   nothing under active negatives (Xi −0.003, n.s.; Iran +0.009).
4. **The training-negative sampler is the biggest single lever.** Training on
   hard negatives makes the model learn a "never-seen pair → positive" shortcut
   (Section 5). Training on random negatives removes most of it and raises
   active-protocol MRR by +0.063 (Xi) and +0.088 (Iran). It costs some
   hard-protocol MRR (−0.049 / −0.083), but stays above recency there.
5. **Robustness:** the hard-protocol advantage over recency holds across
   minimum-interaction thresholds of 5–20, though it shrinks as the universe
   gets smaller. In Xi it disappears at short trading windows (30 s and 60 s).
   Under active negatives the full model is below recency in every
   configuration tested.

**What changed compared with the earlier report:** the old tuned-vs-plain
comparison changed pair history *and* epochs (20 vs 10), and the old memory-off
run also changed the learning rate and had a training bug. Here every variant
has the same 10 epochs, learning rate 0.003, batch size, seeds, split,
negatives and evaluation. That same bug, in which the static encoder stopped
learning after the first optimizer step, was fixed before these runs.

## 2. Design

| Held fixed in every row | Value |
|---|---|
| Training budget | 10 epochs, lr 0.003, batch 128, hidden 32, weight decay 1e-5 |
| Seeds | 42–46 (components, training negatives); 42–44 (window, min-events sweeps) |
| Split | chronological 70/30 between distinct timestamps |
| Training negatives | hard (except in the training-negative block) |
| Evaluation | fixed collision-free negatives (50 per event, seed 12345), timestamp-grouped batches, one row per (pair, second) |
| Evaluation protocols | hard and active (see the item 2 report) |
| Node features | rebuilt from training-period trades only |

Variants (one factor each):

| Block | Variant | What changes |
|---|---|---|
| Components | − memory | memory updates switched off; the model keeps its static node encoder + pair head |
| | − pair history | pair-history head removed (all 6 features) |
| | − memory − pair history | both off; only the static encoder remains |
| | − recency features | decayed counts (5 min, 1 h, 24 h) and pair age zeroed |
| | − count features | log pair count and seen flag zeroed |
| Training negatives | hard / popularity / random | the sampler used during training only |
| Window | W = 30, 60, 120, 300 s | only edges with time difference ≤ W |
| Min events | 5, 10, 15, 20 | the candidate-universe threshold (10 = main study) |

Components and training-negative variants are scored on **identical events**,
so their differences are paired, with block-bootstrap 95% CIs. The window and
min-events sweeps change which events exist, so absolute MRR is not comparable
across rows. The comparable quantity is each row's **gap to recency on the same
events**.

Feature removal zeroes the masked inputs to the pair-head MLP during both
training and scoring; the head's bias absorbs the constant. Network size and
training budget are unchanged.

## 3. Note on "full" vs the earlier "tuned"

"Full" here is the tuned architecture at 10 epochs instead of 20. On the same
events it scores Xi 0.516 vs tuned (20 epochs) 0.498 under hard negatives, and
Iran 0.489 vs 0.502. The extra 10 epochs were not needed, so use 10 in the
frozen item 4 configuration unless validation says otherwise.

## 4. Results

### 4.1 Components (identical events, 10 epochs, lr 0.003, 5 seeds)


**Xi**

| Variant | Hard MRR | Δ vs full (hard) | Active MRR | Δ vs full (active) | Δ vs recency (active) | Never-seen MRR (hard) |
|---|---:|---:|---:|---:|---:|---:|
| Full model | 0.516 ± 0.012 | — | 0.261 ± 0.016 | — | -0.065 [-0.095, -0.039] | 0.704 |
| − memory | 0.418 ± 0.028 | -0.098 [-0.123, -0.075] | 0.276 ± 0.024 | +0.015 [-0.010, +0.042] | -0.050 [-0.080, -0.021] | 0.424 |
| − pair history (all 6) | 0.229 ± 0.007 | -0.287 [-0.312, -0.261] | 0.245 ± 0.006 | -0.016 [-0.043, +0.010] | -0.081 [-0.112, -0.052] | 0.114 |
| − memory − pair history | 0.133 ± 0.005 | -0.383 [-0.420, -0.349] | 0.230 ± 0.003 | -0.031 [-0.074, +0.009] | -0.096 [-0.138, -0.056] | 0.129 |
| − recency features (decayed counts, age) | 0.173 ± 0.019 | -0.343 [-0.381, -0.307] | 0.191 ± 0.012 | -0.070 [-0.104, -0.039] | -0.135 [-0.179, -0.093] | 0.763 |
| − count features (count, seen flag) | 0.434 ± 0.013 | -0.083 [-0.106, -0.061] | 0.258 ± 0.013 | -0.003 [-0.019, +0.012] | -0.068 [-0.097, -0.040] | 0.731 |
| Recency baseline | 0.349 | — | 0.326 | — | — | 0.027 |

**Iran**

| Variant | Hard MRR | Δ vs full (hard) | Active MRR | Δ vs full (active) | Δ vs recency (active) | Never-seen MRR (hard) |
|---|---:|---:|---:|---:|---:|---:|
| Full model | 0.489 ± 0.028 | — | 0.222 ± 0.016 | — | -0.072 [-0.101, -0.045] | 0.619 |
| − memory | 0.404 ± 0.008 | -0.085 [-0.105, -0.065] | 0.184 ± 0.009 | -0.038 [-0.053, -0.023] | -0.109 [-0.137, -0.084] | 0.491 |
| − pair history (all 6) | 0.129 ± 0.003 | -0.360 [-0.383, -0.337] | 0.165 ± 0.004 | -0.058 [-0.078, -0.040] | -0.129 [-0.154, -0.105] | 0.103 |
| − memory − pair history | 0.132 ± 0.001 | -0.357 [-0.383, -0.330] | 0.181 ± 0.001 | -0.041 [-0.065, -0.019] | -0.112 [-0.138, -0.087] | 0.125 |
| − recency features (decayed counts, age) | 0.228 ± 0.013 | -0.261 [-0.292, -0.232] | 0.174 ± 0.006 | -0.048 [-0.068, -0.029] | -0.120 [-0.156, -0.085] | 0.651 |
| − count features (count, seen flag) | 0.420 ± 0.042 | -0.069 [-0.077, -0.060] | 0.232 ± 0.017 | +0.009 [+0.002, +0.018] | -0.062 [-0.090, -0.036] | 0.578 |
| Recency baseline | 0.346 | — | 0.294 | — | — | 0.030 |

### 4.2 Training-negative sampler (identical events, 10 epochs, 5 seeds)

| Market | Trained with | Hard MRR | Δ vs train-hard (hard) | Active MRR | Δ vs train-hard (active) | Never-seen MRR (hard) | Recency (hard / active) |
|---|---|---:|---:|---:|---:|---:|---:|
| Xi | hard (default) | 0.514 ± 0.020 | — | 0.272 ± 0.017 | — | 0.720 | 0.349 / 0.326 |
| Xi | popularity | 0.511 ± 0.011 | -0.003 [-0.021, +0.015] | 0.295 ± 0.024 | +0.023 [+0.008, +0.039] | 0.469 | 0.349 / 0.326 |
| Xi | random | 0.466 ± 0.008 | -0.049 [-0.069, -0.029] | 0.335 ± 0.002 | +0.063 [+0.043, +0.084] | 0.383 | 0.349 / 0.326 |
| Iran | hard (default) | 0.488 ± 0.019 | — | 0.224 ± 0.013 | — | 0.585 | 0.346 / 0.294 |
| Iran | popularity | 0.480 ± 0.004 | -0.008 [-0.027, +0.013] | 0.280 ± 0.007 | +0.056 [+0.037, +0.077] | 0.317 | 0.346 / 0.294 |
| Iran | random | 0.405 ± 0.008 | -0.083 [-0.106, -0.057] | 0.312 ± 0.003 | +0.088 [+0.062, +0.117] | 0.166 | 0.346 / 0.294 |

### 4.3 Trading-time window W (edges with time difference ≤ W; 3 seeds)

| Market | window | Wallets | Scored events | Model hard MRR | Recency hard | Gap vs recency, hard [95% CI] | Model active MRR | Recency active | Gap vs recency, active [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Xi | 30 | 163 | 7772 | 0.367 | 0.343 | +0.025 [-0.051, +0.096] | 0.177 | 0.341 | -0.163 [-0.219, -0.109] |
| Xi | 60 | 163 | 7320 | 0.363 | 0.374 | -0.011 [-0.091, +0.061] | 0.188 | 0.349 | -0.161 [-0.224, -0.106] |
| Xi | 120 | 163 | 8321 | 0.468 | 0.355 | +0.113 [+0.086, +0.138] | 0.271 | 0.337 | -0.067 [-0.092, -0.043] |
| Xi | 300 | 163 | 9638 | 0.522 | 0.349 | +0.173 [+0.144, +0.204] | 0.274 | 0.326 | -0.052 [-0.082, -0.026] |
| Iran | *pending* | | | | | | | | |

### 4.4 Minimum interaction count for the candidate universe (3 seeds)

| Market | min_events | Wallets | Scored events | Model hard MRR | Recency hard | Gap vs recency, hard [95% CI] | Model active MRR | Recency active | Gap vs recency, active [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Xi | 5 | 355 | 13629 | 0.532 | 0.360 | +0.171 [+0.140, +0.205] | 0.269 | 0.319 | -0.050 [-0.080, -0.024] |
| Xi | 10 | 163 | 9638 | 0.514 | 0.349 | +0.165 [+0.134, +0.200] | 0.261 | 0.326 | -0.065 [-0.095, -0.038] |
| Xi | 15 | 87 | 6717 | 0.442 | 0.350 | +0.092 [+0.049, +0.136] | 0.226 | 0.324 | -0.098 [-0.138, -0.057] |
| Xi | 20 | 48 | 4627 | 0.379 | 0.312 | +0.066 [+0.032, +0.104] | 0.272 | 0.331 | -0.059 [-0.095, -0.019] |
| Iran | *pending* | | | | | | | | |

Intervals are paired contiguous-time-block bootstrap CIs (50 blocks, 2,000
draws) on seed-averaged per-event reciprocal ranks. Seed-to-seed SDs are shown
next to the means.

## 5. The shortcut, and what removes it

The item 2 benchmark showed that the hard-trained model ranks brand-new pairs
first ("never-seen MRR" ≈ 0.70). The ablations locate the cause:

* **Any pair-history feature reveals novelty.** Removing the count and seen
  flag leaves never-seen MRR at 0.731 (Xi), because the decayed counts and age
  are also zero for an unseen pair. Removing the decayed counts and age leaves it
  at 0.763, because the seen flag remains. Only removing the whole head brings
  it down, to 0.114, and that also removes the model's real signal.
* **The training sampler creates the shortcut.** Hard training negatives are
  always previous partners, so "unseen" perfectly separates positive from
  negative during training. With random training negatives most negatives are
  also unseen, the shortcut no longer pays, and never-seen MRR falls to 0.383
  (Xi) and 0.166 (Iran).
* **Fixing the training sampler helps where it matters.** Random-trained models
  are the best learned variants under active negatives in both markets, and they
  remain above recency under hard negatives.

| Point estimates, 10 epochs, 5 seeds | Xi hard | Xi active | Iran hard | Iran active |
|---|---:|---:|---:|---:|
| Full, trained on hard negatives | 0.514 | 0.272 | 0.488 | 0.224 |
| Full, trained on random negatives | 0.466 | **0.335** | 0.405 | **0.312** |
| Recency baseline | 0.349 | 0.326 | 0.346 | 0.294 |

The random-trained model is the only learned variant whose point estimates are
at or above recency under **both** protocols in **both** markets. Its active-protocol
margin over recency is small (+0.009 Xi, +0.018 Iran), and no paired interval
against recency was computed for it.

**This is a test-set observation and must not be reported as a result.**
Choosing the training sampler because of these numbers would be selecting on
the test window. The correct use is as a pre-registered hypothesis for item 4:
choose the training sampler on a validation window, freeze it, and test it
prospectively against recency on the active protocol.

## 6. Interpretation by component

* **Pair history:** essential, and its value comes mostly from the time-decayed
  counts and pair age, i.e. *how recently and how often* the pair co-traded.
  The model is learning a richer, learned version of the recency baseline. This
  matches the finding that recency is the strongest heuristic.
* **Node memory:** a moderate, replicated gain under hard negatives (−0.098 Xi,
  −0.085 Iran when removed). Under active negatives it is market-dependent
  (no effect in Xi, −0.038 in Iran). Claim "memory contributes under
  history-based negatives", not "memory is essential".
* **Memory alone (no pair history)** is at or below recency in every cell. A
  memory-only temporal model is not competitive here.
* **Training negatives:** the dominant factor for generalisation across
  evaluation protocols. Report it as a design choice with a measured trade-off,
  not a tuning detail.
* **Trading window (Xi):** the model's hard-protocol advantage needs the full
  300 s window. At 30 s and 60 s it is indistinguishable from recency, and under
  active negatives its deficit roughly triples (−0.16 vs −0.05). Shorter windows give sparser graphs and
  fewer pair histories to learn from.
* **Minimum events (Xi):** the hard-protocol gap over recency is stable from
  5 to 10 events (+0.17) and shrinks with stricter universes (+0.09 at 15,
  +0.07 at 20), as the graph drops to 87 and 48 wallets. It never reverses.

## 7. What to write in the paper

> Controlled single-factor ablations (fixed 10-epoch budget, five seeds,
> identical evaluation events) show that the causal pair-history head accounts
> for most of the model's performance (removing it: −0.29 and −0.36 MRR under
> hard negatives in Xi and Iran), driven by its time-decayed counts and pair age.
> Node memory adds a smaller, replicated gain under hard negatives (−0.10,
> −0.09 when removed). The training negative sampler determines how performance
> transfers across evaluation protocols: hard-negative training induces a
> novelty shortcut, while random-negative training improves active-protocol MRR
> by +0.06 and +0.09 at a cost of −0.05 and −0.08 under hard negatives.

## 8. Limitations

* Two markets. Iran window and min-events sweeps pending at the time of writing.
* Three seeds for the sweeps; seed SD is visible but small-sample.
* The min-events sweep re-runs the model-free screen on the full observation
  window (as the original study did). It is a sensitivity analysis, not a
  leak-free universe; item 4 fixes this.
* Feature removal is by zero-masking, not by retraining a smaller head. This is
  standard practice, but a zeroed feature is not identical to a never-existing
  one if the remaining features are correlated with it.
* Window sweeps filter the 300 s edge table. With `max_pairs_per_trade = 50`
  this is equivalent to rebuilding at W, because the cap keeps the earliest
  partners.
* Xi's test window was inspected in earlier work; these ablations do no tuning,
  but item 4 is the clean test.

## 9. Reproduce

```bash
for RUN in xi_2027_63d_p50 rep_iran; do
  python run_ablations.py -c configs/default.yaml --run-name $RUN \
    --blocks components neg_train --seeds 42 43 44 45 46 \
    --epochs 10 --lr 0.003 --eval-negatives hard active
  python run_ablations.py -c configs/default.yaml --run-name $RUN \
    --blocks window min_events --seeds 42 43 44 \
    --epochs 10 --lr 0.003 --eval-negatives hard active \
    --windows 30 60 120 300 --min-events-list 5 10 15 20
done
```

Files per market in `data/artifacts/<run>/ablations/`: `components_*`,
`neg_train_*`, `window_*`, `min_events_*` (per-seed rows, summaries, paired
gaps).
