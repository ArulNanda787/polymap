# Polymap: candidate wallet pairs and on-chain funding evidence

Runs: `xi_2027_63d_p50` (first market, sections 1 to 12) and `rep_iran` (replication, sections 13 to 15). Both: one Polymarket market, 63 days of trades.
Status: a small, consistent positive gap on direct transfers in both markets; the pre-agreed criterion is not met on the primary (all-wallet) analysis in either market, and is met only in the sensitivity analysis that excludes truncated wallets. Exploratory, not confirmed. Written 2026-10-26.

---

## 1. Question

Some pairs of wallets trade at nearly the same moment and price, repeatedly. That can be coincidence (two people reacting to the same news) or it can mean the wallets are related, for example controlled by one party.

Trading timing alone cannot separate those. This project looks for a second, independent kind of evidence on the blockchain: **do candidate pairs show more money links than comparable ordinary pairs do?**

Two money links are measured:

- **Direct transfer:** one wallet sent funds straight to the other, in either direction.
- **Shared funder:** a third address sent funds to both wallets.

A funding link is never proof of common ownership. Exchanges, bridges, routers and market makers create links between unrelated wallets.

---

## 2. Data

| Item | Value |
|---|---|
| Market | `0xa467b14d51f01b957109d9cbb1d6c124fab2a089d52ed8f471d23c2812e743b7` |
| Trades | 23,839 (cached Data API pull, 2026-08-18 to 2026-10-20) |
| Wallets in the trade graph | 2,461 |
| Temporal edges | 340,752 (`window_seconds` = 300, `max_pairs_per_trade` = 50) |
| Config layers | `configs/default.yaml`, `markets/xi_2027_63d.yaml`, `markets/xi_2027_63d_p50.yaml` |

A temporal edge links two wallets that traded within 300 seconds of each other in this market. Wallets are never linked across markets.

---

## 3. How candidate pairs are chosen

Candidates come from `polymap candidates`, a **model-free** screen computed directly from the edges. The trained temporal model is not used to select them. (Older output files label these pairs `tgnn_candidate`; that label is historical and misleading.)

For each wallet pair with at least `min_events` edges:

- **Persistent:** the pair's activity spans more than 7 days and at least 3 distinct timestamps.
- **Tight:** median time gap of 5 seconds or less and median price gap of 0.01 or less.

Selected for this analysis: `persistent_tight` and `one_off_tight`, with `--min-events 10`.

| | Count |
|---|---|
| Candidate pairs | 397 (299 persistent_tight, 98 one_off_tight) |
| Candidate wallets | 163 |

`--min-events` sets the sample size: 3 gives 625 wallets, 10 gives 163, 20 gives 48.

---

## 4. On-chain data collection

`collect_polygon_transfers.py`, Etherscan V2, Polygon (chain 137). Only the candidate wallets are queried, using three address-scoped endpoints per wallet: `txlist`, `txlistinternal`, `tokentx`. It is not market-wide.

- Window: earliest market trade minus 24 hours, up to the chain head; rows outside the padded window are dropped.
- Everything those wallets did on Polygon in the window is kept, not only Polymarket activity.
- One hop only: counterparties' own histories are not fetched.
- Result: 463,739 raw rows, 462,777 unique after removing errors and duplicates; **160 of 163 wallets returned data**.

**Known data problems**

1. **Truncation.** The collector caps each endpoint at 10 pages x 1,000 rows, oldest first. **16 wallets** hit the cap; their ERC-20 history stops 50 to 64 days before the window ends. The collector's own flag caught only 1 of them because de-duplication pushed counts just below 10,000. List: `truncated_wallets.csv`. `recover_erc20.py` covers 5 of the 16 in its hardcoded list; it now also accepts `--wallets-file` and `--outdir`. Recovery has not been run on the other 11.
2. **Window start.** Funding that happened before the window starts (one day before the market's first trade) is invisible.
3. **Hub addresses.** The top funders reach most wallets. 11,709 of 15,014 distinct funders reached exactly one fetched wallet, but 69 reached 20 or more and 4 reached over 100. The two biggest reached 143 of 160 wallets with 57k transfers each, which looks like settlement flow through protocol contracts rather than deposits (addresses not independently labelled). The zero address (token mints) reached 127 wallets and is excluded.

---

## 5. How the validation was designed, and what was wrong at first

### Version 1 (superseded)

Controls were any non-candidate pair matched only on co-timing count. A shared funder counted if any address had sent to both wallets.

Problems found:

- **Saturation:** 99.0% of candidates and 98.8% of controls "shared a funder" because of the hub addresses. The measure carried no information.
- **Weak matching:** controls differed from candidates in wallet activity.
- **Reported direct-transfer result (15.1% vs 7.2%) is superseded** by version 2: that comparison was partly flattered by the control design.

### An invalid run that was discarded

A separate `xi_2027` run reported 613 candidates vs 376 unique controls (41.6% vs 100% sharing a raw funder). It is **invalid**: `xi_2027` is a different, small slice (2,100 trades, 196 wallets, 2026-09-30 to 10-03), but the transfers file covered the 160 wallets of the 63-day run. Only 54 of its 196 wallets had data, so only 38.8% of candidate pairs had both wallets fetched, while controls were restricted to fetched wallets (100%). The gap measured data coverage, not relatedness. This run is excluded from all conclusions.

### Version 2 (current): `polymap funding-robustness`

- **Anchor-fixed controls.** For a candidate pair (A, B), controls are (A, C) and (B, D): same anchor wallet, a different partner who also co-traded with it, matched on log co-timing count, never reused, never a candidate pair. Each wallet's own history is identical on both sides.
- **Coverage enforced.** Candidate pairs where either wallet has no on-chain data are dropped and counted.
- **Hub sweep.** Shared-funder results are reported with funders ignored if they reach more than 5, 10, 20, 50 or 100 fetched wallets.
- **Deposit-only definition.** Stablecoin transfers (pUSD, USDC.e, USDC, USDT0) received at or before the wallet's first trade in this market, excluding the zero address. This removes trading settlement and gas top-ups.
- **Rarity-weighted score.** Rare shared funders count more than common ones.
- **Uncertainty.** A wallet-level bootstrap (2,000 reweightings; a pair's weight is the product of its two wallets' weights) gives 95% intervals, because pairs share wallets and are not independent.

---

## 6. Results

### Primary run: all fetched wallets (391 candidate pairs, 782 controls; 6 candidate pairs dropped for no coverage)

| Measure | Candidates | Controls | Gap (points) | 95% interval |
|---|---|---|---|---|
| Direct transfer (any time) | 15.3% | 8.6% | +6.8 | -0.3 to +16.7 |
| Shared funder, any transfer, cap 5 | 24.8% | 20.7% | +4.1 | -7.3 to +15.7 |
| Shared funder, any transfer, cap 10 | 38.1% | 33.6% | +4.5 | -8.9 to +16.6 |
| Shared funder, any transfer, cap 20 | 50.6% | 49.5% | +1.1 | -12.6 to +12.3 |
| Shared funder, any transfer, cap 50 | 58.8% | 57.5% | +1.3 | -12.5 to +12.7 |
| Shared funder, any transfer, cap 100 | 85.7% | 83.1% | +2.6 | -8.3 to +11.2 |
| Shared funder, deposit-only, cap 5 | 4.6% | 4.3% | +0.3 | -3.7 to +4.6 |
| Shared funder, deposit-only, cap 10 | 8.7% | 7.2% | +1.5 | -4.8 to +8.0 |
| Shared funder, deposit-only, cap 20 | 14.8% | 12.3% | +2.6 | -5.9 to +10.3 |
| Shared funder, deposit-only, cap 50 | 19.2% | 17.4% | +1.8 | -6.9 to +10.2 |
| Shared funder, deposit-only, cap 100 | 52.2% | 45.9% | +6.3 | -4.3 to +16.6 |
| Rarity-weighted shared deposit funders (mean score) | 0.767 | 0.823 | -0.056 | -0.422 to +0.299 |

### Sensitivity run: excluding the 16 truncated wallets (253 candidate pairs, 506 controls; 144 candidate pairs dropped)

| Measure | Candidates | Controls | Gap (points) | 95% interval |
|---|---|---|---|---|
| Direct transfer (any time) | 13.4% | 4.2% | +9.3 | +0.6 to +23.1 |
| Shared funder, any transfer, cap 10 | 36.4% | 29.0% | +7.3 | -8.8 to +21.4 |
| Shared funder, deposit-only, cap 10 | 4.7% | 1.4% | +3.4 | -1.7 to +10.1 |
| Shared funder, deposit-only, cap 20 | 11.5% | 4.9% | +6.5 | -1.3 to +16.1 |
| Rarity-weighted shared deposit funders (mean score) | 0.435 | 0.392 | +0.043 | -0.087 to +0.164 |

(Full tables: `data/artifacts/xi_2027_63d_p50/funding_robustness_table.csv`. The sensitivity run overwrote the primary table unless it was copied first; the primary numbers above are from the first run.)

### Sensitivity run, rerun with the same truncation rule as Iran (21 wallets excluded; 242 candidate pairs, 484 controls; 155 candidate pairs dropped)

The 16-wallet list above came from the row-count heuristic only. `find_truncated_wallets.py` run on Xi with the manifest plus that heuristic flags **21 wallets** (0 from the manifest, 21 from the heuristic), touching 152 of 397 candidate pairs (38%). This is the same rule used for Iran, so the two sensitivity analyses are now comparable. The 16-wallet version above is kept for the record; **this version replaces it in sections 14 and 15.**

| Measure | Candidates | Controls | Gap (points) | 95% interval |
|---|---|---|---|---|
| Direct transfer (any time) | 14.1% | 4.1% | +9.9 | +0.3 to +24.0 |
| Shared funder, any transfer, cap 10 | 34.7% | 27.3% | +7.4 | -9.6 to +22.8 |
| Shared funder, deposit-only, cap 10 | 5.0% | 0.6% | +4.3 | -0.4 to +11.8 |
| Shared funder, deposit-only, cap 20 | 9.5% | 3.9% | +5.6 | -1.8 to +13.6 |
| Shared funder, deposit-only, cap 100 | 39.3% | 29.3% | +9.9 | -6.1 to +23.9 |
| Rarity-weighted shared deposit funders (mean score) | 0.416 | 0.371 | +0.045 | -0.087 to +0.163 |

Same picture as the 16-wallet version: direct transfer gap positive with a lower bound just above zero; every shared-funder row includes zero.

---

## 7. What can and cannot be concluded

**Supported**

- The direction is consistent: across both runs and nearly every definition, candidates show somewhat more funding evidence than comparable pairs.
- Direct transfers show the clearest gap: roughly 2 to 3 times as often for candidates. In the primary run the interval just touches zero; in the sensitivity run it just excludes it.

**Not supported**

- Shared-funder evidence is not distinguishable from zero in either run, under any hub cap.
- Nothing here shows common ownership or coordination.
- The sensitivity run describes lighter wallets only: it drops 36% of candidate pairs, and those involve the heaviest wallets.

**On the choice of primary measure.** Direct transfers (all-wallets run) is designated the primary result because it does not depend on a hub threshold. That choice was made after seeing the version 1 result, so this analysis is exploratory. A replication on another market, with the primary measure and thresholds fixed in advance, is the confirmatory test.

---

## 8. Limitations

- Small sample: 160 wallets, a few hundred pairs; intervals are wide (about ±12 points).
- 16 wallets with truncated history (11 not yet recovered) account for 36% of candidate pairs.
- Funding before the window start is not observed.
- Only Polygon PoS; no cross-chain or exchange-internal funding.
- Shared funders may be exchanges, bridges, relayers or custodians; addresses are not labelled.
- Many measures are reported (12 per run). A single significant row among them would be weak evidence on its own.
- Candidate screening thresholds (`min-events`, tightness, persistence) are research choices that change the sample.
- Intervals are descriptive resampling intervals, not causal tests.

---

## 9. Pipeline performance (background)

Training and scoring were rewritten as batched, TGN-style processing (batch 128, evaluation batch 4):

| | Before | After |
|---|---|---|
| Training, 100k edges, 1 epoch | 94 s (about 745 events/s) | about 4 s |
| Full run, 340k edges, 3 epochs, 3 samplers | about 20+ min (extrapolated) | 76 s |
| Candidate-subgraph training | not run | about 30 s |

Metrics shifted slightly versus the old per-event trainer (on this run, AUC against random negatives 0.79 to 0.82, hard 0.76 to 0.73, popularity 0.73 to 0.69). Set `model: eval_batch_size: 1` for exact sequential evaluation. The trained model does not feed the candidate screen or this validation.

---

## 10. How to reproduce (zsh, from `backend/`)

Type each command on one line.

```bash
polymap build -c configs/default.yaml -c configs/markets/xi_2027_63d.yaml -c configs/markets/xi_2027_63d_p50.yaml --run-name xi_2027_63d_p50

polymap candidates -c configs/default.yaml -c configs/markets/xi_2027_63d.yaml -c configs/markets/xi_2027_63d_p50.yaml --run-name xi_2027_63d_p50 --min-events 10

export ETHERSCAN_API_KEY=your_key
python collect_polygon_transfers.py --trades data/raw/trades_0xa467b14d51f01b957109d9cbb1d6c124fab2a089d52ed8f471d23c2812e743_d63.parquet --wallets-file data/processed/xi_2027_63d_p50/candidate_wallets.txt --outdir data/raw/onchain_candidates --resume

python export_funding_summary.py data/raw/onchain_candidates/polygon_transfers.parquet --outdir data/raw/onchain_candidates/summary

polymap funding-robustness -c configs/default.yaml -c configs/markets/xi_2027_63d.yaml -c configs/markets/xi_2027_63d_p50.yaml --run-name xi_2027_63d_p50 --funding-file data/raw/onchain_candidates/polygon_transfers.parquet

polymap funding-robustness -c configs/default.yaml -c configs/markets/xi_2027_63d.yaml -c configs/markets/xi_2027_63d_p50.yaml --run-name xi_2027_63d_p50 --funding-file data/raw/onchain_candidates/polygon_transfers.parquet --exclude-wallets-file truncated_wallets.csv
```

Note: zsh does not word-split an unquoted shell variable, so do not store the `-c ...` options in `$C`; type them out or use an array.

---

## 11. Files

| File | Purpose |
|---|---|
| `src/polymap/analysis/relationships.py` | `prescreen_pairs`, `select_candidates` (model-free candidates) |
| `src/polymap/analysis/funding_robustness.py` | Version 2 validation: controls, hub sweep, deposit-only, bootstrap |
| `src/polymap/analysis/funding_validation.py` | Version 1 validator (kept for comparison; superseded) |
| `collect_polygon_transfers.py` | On-chain fetch (`--wallets-file` for candidates only) |
| `recover_erc20.py` | Re-fetches capped ERC-20 histories (`--wallets-file`, `--outdir`) |
| `export_funding_summary.py` | Shrinks raw transfers to shareable edge, fan-out and coverage tables |
| `truncated_wallets.csv` | The 16 truncated wallets |
| `data/processed/xi_2027_63d_p50/` | `candidate_pairs.parquet`, `candidate_wallets.txt`, `prescreen_pairs.parquet` |
| `data/artifacts/xi_2027_63d_p50/` | `funding_robustness.json`, `funding_robustness_table.csv`, `funding_robustness_pairs.parquet` |

---

## 12. Replication plan (fixed before the Iran run; results in sections 13 to 15)

Fix these before looking at any new result:

1. **Primary measure:** direct transfer (any time), all fetched wallets.
2. **Candidate definition:** `persistent_tight` + `one_off_tight`, same tightness and persistence thresholds. Choose `--min-events` so that the number of candidate wallets is at least as large as here (about 160).
3. **Control design:** anchor-fixed, matched on co-timing count, no reuse.
4. **Sensitivity checks, reported alongside:** the full hub sweep, deposit-only definition, and exclusion of truncated wallets.
5. **Market choice:** similar size and duration (about 20k trades over about 60 days) so the screen and the data volume are comparable. Choose it by volume, not by whether it looks promising.
6. **Success criterion:** same direction on direct transfers, and an interval that excludes zero.
7. **Data hygiene:** run the coverage check first. Every candidate wallet must be in the fetched set before comparing (this is what invalidated the `xi_2027` run), and truncated wallets should be recovered or excluded up front.
8. Report the result whether or not it agrees with this one.

Template (replace the market config and run name; the funding step needs the new run's trades file):

```bash
polymap fetch      -c configs/default.yaml -c configs/markets/<market>.yaml --run-name <run>
polymap build      -c configs/default.yaml -c configs/markets/<market>.yaml --run-name <run>
polymap candidates -c configs/default.yaml -c configs/markets/<market>.yaml --run-name <run> --min-events 10
# then: collect_polygon_transfers.py with --wallets-file data/processed/<run>/candidate_wallets.txt,
# export_funding_summary.py, and polymap funding-robustness as in section 10.
```


---

## 13. Replication: Iran market (`rep_iran`)

Everything below used the settings fixed in section 12. Only `run_name` and the market id differ from the Xi config (checked: the loaded configs are identical apart from those two fields).

**Market choice rule (set before looking at any funding result).** Take open markets (`polymap markets --open --min-volume 500000`), skip those flagged `resolved = True` (prices pinned near 0 or 1, mostly long shots), take the highest-volume remaining one, and require 15,000 to 40,000 trades in the last 63 days. First eligible market: `will-the-us-invade-iran-before-2027`, condition id `0x5db999fad322cea2914535aae5517060c3f80ad6d8c0231cde2124a434d16846`, about $75M volume. It passed the trade-count check.

| Item | Xi (first run) | Iran (replication) |
|---|---|---|
| Trades (63 days) | 23,839 | 35,135 (2026-08-24 to 2026-10-26) |
| Wallets in the graph | 2,461 | 3,761 |
| Temporal edges | 340,752 | 460,091 |
| Candidate pairs (`--min-events 10`) | 397 | 1,512 (998 persistent_tight, 514 one_off_tight) |
| Candidate wallets | 163 | 420 |
| On-chain rows | 463,739 | 1,293,799 |
| Wallets with truncated history | 16 | 54 (all in the collector manifest; 49 also by the row heuristic) |
| Candidate pairs involving a truncated wallet | 144 of 397 (36%) | 632 of 1,512 (42%) |

Truncated wallets were found with `find_truncated_wallets.py`: a wallet is flagged if the collector manifest lists it, or if it has at least 9,000 ERC-20 rows and its history ends more than 7 days before the end of the data.

### Iran primary run: all fetched wallets (1,512 candidate pairs, 3,023 controls, 0 dropped for coverage)

| Measure | Candidates | Controls | Gap (points) | 95% interval |
|---|---|---|---|---|
| Direct transfer (any time) | 21.9% | 17.3% | +4.6 | -0.8 to +10.5 |
| Shared funder, any transfer, cap 5 | 13.4% | 15.2% | -1.8 | -6.2 to +3.2 |
| Shared funder, any transfer, cap 10 | 18.6% | 22.0% | -3.4 | -8.7 to +2.5 |
| Shared funder, any transfer, cap 20 | 26.3% | 28.3% | -2.1 | -8.2 to +4.2 |
| Shared funder, any transfer, cap 50 | 40.5% | 43.3% | -2.8 | -10.1 to +5.2 |
| Shared funder, any transfer, cap 100 | 46.2% | 51.0% | -4.8 | -11.8 to +2.7 |
| Shared funder, deposit-only, cap 5 | 1.3% | 1.5% | -0.2 | -1.3 to +1.0 |
| Shared funder, deposit-only, cap 10 | 2.1% | 2.5% | -0.4 | -1.9 to +1.2 |
| Shared funder, deposit-only, cap 20 | 5.2% | 3.8% | +1.3 | -1.3 to +4.8 |
| Shared funder, deposit-only, cap 50 | 6.4% | 5.4% | +1.0 | -2.1 to +4.5 |
| Shared funder, deposit-only, cap 100 | 8.0% | 6.6% | +1.4 | -2.1 to +5.1 |
| Rarity-weighted shared deposit funders (mean score) | 0.435 | 0.474 | -0.038 | -0.177 to +0.062 |

### Iran sensitivity run: excluding the 54 truncated wallets (880 candidate pairs, 1,758 controls; 632 candidate pairs dropped)

| Measure | Candidates | Controls | Gap (points) | 95% interval |
|---|---|---|---|---|
| Direct transfer (any time) | 13.3% | 7.7% | +5.6 | +0.2 to +12.3 |
| Shared funder, any transfer, cap 5 | 7.3% | 6.8% | +0.5 | -3.3 to +5.6 |
| Shared funder, any transfer, cap 10 | 10.8% | 11.7% | -0.9 | -6.1 to +5.2 |
| Shared funder, any transfer, cap 20 | 18.5% | 16.7% | +1.8 | -5.9 to +10.3 |
| Shared funder, any transfer, cap 50 | 33.3% | 31.7% | +1.5 | -7.6 to +11.5 |
| Shared funder, any transfer, cap 100 | 39.9% | 41.0% | -1.1 | -9.9 to +8.4 |
| Shared funder, deposit-only, cap 5 | 0.3% | 0.1% | +0.3 | -0.2 to +1.4 |
| Shared funder, deposit-only, cap 10 | 0.7% | 0.2% | +0.4 | -0.4 to +1.8 |
| Shared funder, deposit-only, cap 20 | 4.0% | 0.7% | +3.3 | +0.3 to +7.6 |
| Shared funder, deposit-only, cap 50 | 4.8% | 1.3% | +3.5 | +0.5 to +7.8 |
| Shared funder, deposit-only, cap 100 | 5.9% | 2.0% | +3.9 | +0.6 to +8.3 |
| Rarity-weighted shared deposit funders (mean score) | 0.291 | 0.273 | +0.018 | -0.041 to +0.081 |

---

## 14. Outcome against the pre-agreed criterion

Criterion (section 12): the primary measure, direct transfer on all fetched wallets, shows the same direction as Xi with a 95% interval that excludes zero.

| Analysis | Xi gap (interval) | Iran gap (interval) |
|---|---|---|
| Direct transfer, all wallets (primary) | +6.8 (-0.3 to +16.7) | +4.6 (-0.8 to +10.5) |
| Direct transfer, truncated wallets excluded (sensitivity; same rule in both markets) | +9.9 (+0.3 to +24.0) | +5.6 (+0.2 to +12.3) |

- **Direction:** replicated. Both markets, both analyses, positive.
- **Interval excluding zero on the primary:** **not met** in either market. Both intervals include zero by a small margin.
- **Sensitivity analysis:** the interval excludes zero in both markets, but only just (lower bounds +0.3 and +0.2 points).
- **Rough combination** (fixed-effect inverse-variance pooling from the reported intervals, treating them as symmetric): primary about +5.3 points (about +0.6 to +10.0); sensitivity about +6.5 points (about +1.1 to +11.9; recomputed with the 21-wallet Xi rerun). This was computed after seeing both results and is not the pre-agreed criterion; treat it as descriptive.

**Shared funders.**

- Under the "any transfer, any time" definition, there is no usable signal in either market or either analysis; the signs flip between rows.
- Under the deposit-only definition, the Iran sensitivity run shows gaps of +3.3 to +3.9 points at caps 20, 50 and 100 with intervals that just exclude zero. Those three rows are the same measure at nearby thresholds, not independent results, and the underlying rates are very small (0.7% to 5.9%). Xi's deposit-only gaps (including the 21-wallet rerun) were positive but their intervals included zero. Do not read this as established.

**Base rates differ by market.** Ordinary control pairs have direct transfers 17.3% of the time in Iran (primary run) against 8.6% in Xi. Excluding the truncated wallets lowers Iran's control rate to 7.7%. One possible explanation is that heavy wallets with truncated histories have many counterparties and so inflate the baseline, hiding a gap. That explanation is a hypothesis; it was not tested.

---

## 15. Updated conclusions and caveats

**What the evidence supports**

- Candidate pairs (tight, persistent co-trading) have direct transfers between them somewhat more often than comparable ordinary pairs, in both markets. The gap is about 5 to 9 points on a base of 4% to 17%.
- Shared-funder evidence does not add support.

**What it does not support**

- A confirmed effect. The pre-agreed primary analysis did not meet its criterion in either market.
- Anything about common ownership or coordination. Direct transfers can come from market makers, custodians or exchanges.

**Cautions specific to the replication**

- The sensitivity analysis restricts the sample (Xi: 242 of 397 pairs with the 21-wallet rule; Iran: 880 of 1,512). It describes wallets with complete history, which are lighter traders than those excluded.
- Many measures are reported (12 per run, 4 runs). The few rows whose intervals exclude zero are weak evidence individually; the consistent direction of the direct-transfer gap across four analyses is the more informative pattern.
- The truncated-wallet lists were first built differently (Xi: row-count heuristic only, 16 wallets; Iran: manifest plus heuristic, 54 wallets). Xi was rerun with the same script (21 wallets); the result barely changed (see section 6). Both sensitivity analyses now use the same rule.
- The sensitivity analysis excludes 39% (Xi) and 42% (Iran) of candidate pairs, and the lower bounds sit at +0.3 and +0.2 points. A different but reasonable exclusion rule could move either bound across zero; this was not tested.
- No multiplicity correction was applied across the 12 measures per run.
- Only two markets, both geopolitical "before 2027" markets. Generalisation beyond that type is untested.
- Truncation is a measured data limit: 42% of Iran candidate pairs involve a wallet whose history stopped early. Recovering that history (`recover_erc20.py --wallets-file truncated_rep_iran.csv --outdir data/raw/onchain_rep_iran`) would make the primary analysis cleaner; it has not been run.
- The first-run Xi result (15.1% vs 7.2% under the old controls) and the discarded `xi_2027` run remain superseded (section 5).

**Suggested next steps**

1. Recover the truncated wallets' history in both markets, then rerun the primary analysis so the heavy wallets are in the sample with complete data.
2. Add a third market chosen by the same rule, to see whether the direction holds.
3. Report both the primary and sensitivity analyses together, with the criterion outcome stated as in section 14.

## Files added in the replication

| File | Purpose |
|---|---|
| `configs/markets/replication_template.yaml` | Xi settings in one file; change only `run_name` and the market id |
| `configs/markets/rep_iran.yaml` | The Iran config actually used |
| `find_truncated_wallets.py` | Flags capped wallets (manifest list plus row heuristic) |
| `truncated_rep_iran.csv` | The 54 truncated Iran wallets |
| `data/raw/onchain_rep_iran/` | Iran transfers (parquet, CSV, manifest, raw JSON) |
| `data/artifacts/rep_iran/` | `funding_robustness.json`, `funding_robustness_table.csv`, `funding_robustness_pairs.parquet`; the sensitivity run overwrote the primary table unless it was copied to `funding_robustness_table_all_wallets.csv` first |

Reproduction commands for Iran:

```bash
polymap fetch      -c configs/default.yaml -c configs/markets/rep_iran.yaml
polymap build      -c configs/default.yaml -c configs/markets/rep_iran.yaml
polymap candidates -c configs/default.yaml -c configs/markets/rep_iran.yaml --min-events 10
caffeinate -i python collect_polygon_transfers.py --trades data/raw/trades_0x5db999fad322cea2914535aae5517060c3f80ad6d8c0231cde2124a434d168_d63.parquet --wallets-file data/processed/rep_iran/candidate_wallets.txt --outdir data/raw/onchain_rep_iran --resume
python find_truncated_wallets.py data/raw/onchain_rep_iran/polygon_transfers.parquet --manifest data/raw/onchain_rep_iran/manifest.json --candidates data/processed/rep_iran/candidate_pairs.parquet --out truncated_rep_iran.csv
polymap funding-robustness -c configs/default.yaml -c configs/markets/rep_iran.yaml --run-name rep_iran --funding-file data/raw/onchain_rep_iran/polygon_transfers.parquet
polymap funding-robustness -c configs/default.yaml -c configs/markets/rep_iran.yaml --run-name rep_iran --funding-file data/raw/onchain_rep_iran/polygon_transfers.parquet --exclude-wallets-file truncated_rep_iran.csv
```
