# Polymap

Temporal graph learning for finding related wallets in Polymarket prediction markets, with a Next.js dashboard for exploring the results.

> **Snapshot date: 3 November 2026.** Everything below describes the project as it stood on that day. If something here disagrees with the code, trust the code.

GitHub: <https://github.com/ArulNanda787/polymap> (branch `main`)

> **© 2026 Arul Nanda. All rights reserved.** This repository is public for viewing only. It may not be used, copied, modified or redistributed for any purpose, including research and commercial use, without written permission. See [`LICENSE`](LICENSE).

---

## Contents

1. [What this project is](#1-what-this-project-is)
2. [Folder layout](#2-folder-layout)
3. [First-time setup (or after moving the folder)](#3-first-time-setup-or-after-moving-the-folder)
4. [Running the frontend](#4-running-the-frontend)
5. [Running the backend pipeline](#5-running-the-backend-pipeline)
6. [Regenerating the Predictions tab data](#6-regenerating-the-predictions-tab-data)
7. [The backend API](#7-the-backend-api)
8. [Research scripts in `backend/`](#8-research-scripts-in-backend)
9. [Runs that exist and headline results](#9-runs-that-exist-and-headline-results)
10. [Data: where it lives and how to share it](#10-data-where-it-lives-and-how-to-share-it)
11. [Git notes and known problems](#11-git-notes-and-known-problems)
12. [Things to remember](#12-things-to-remember)
13. [License](#13-license)

---

## 1. What this project is

Polymap looks at trades in a single Polymarket market and asks: **which wallets repeatedly trade within seconds of each other?** Those pairs could be one operator running several wallets or a coordinated group. They could also be two people reacting to the same news.

The pipeline:

1. **Fetches** every trade for a market from the Polymarket Data API, caches it and hashes the file so the result is reproducible.
2. **Builds** a temporal graph. Wallets are nodes, and an edge joins two wallets that traded within `window_seconds` (default 300 s) of each other.
3. **Trains** a custom memory-based temporal graph model. It has per-wallet memory and an optional causal pair-history head, and it predicts which wallet will co-trade with which next.
4. **Screens** wallet pairs into categories (persistent/one-off × tight/loose) and keeps the tight ones as *candidates*.
5. **Validates** candidates against on-chain Polygon funding links: direct transfers between them, or a shared funder. It compares them with matched control pairs and uses a pre-registered test.

The frontend presents all of this as a landing page plus a dashboard. Its views are Markets, Network, Pairs, Wallets, Charts, Architecture, Model Results and Predictions.

**Interpretation rule (important for any write-up):** screening output is *candidates, not findings*. A funding link is a possible financial connection, not proof of shared ownership or coordination. Exchanges, bridges and routers create false shared-funder links.

---

## 2. Folder layout

```
polymap-updated/
├── README.md                ← this file
├── .gitignore               ← see §11
├── backend/                 Python package `polymap` + research scripts
│   ├── src/polymap/         the library (pipeline, model, features, API)
│   ├── configs/             YAML configs (default + one per market/run)
│   ├── data/                ALL data (raw trades, processed, artifacts) — NOT in git, many GB
│   ├── docs/                research notes (tgnn_tuning_results.md)
│   ├── tests/               pytest suite
│   ├── notebooks/           old exploratory notebook code
│   ├── *.py                 top-level research scripts (see §8)
│   ├── pyproject.toml       dependencies and the `polymap` command
│   ├── README.md            detailed backend design notes (read this too)
│   └── .venv/               local Python env — machine-specific, rebuild after moving
├── frontend/                Next.js 16 + React 19 + Tailwind 4 dashboard
│   ├── app/                 routes: / (landing), /dashboard, /runs/[name]
│   │   └── components/      landing/ and polymap/ (the dashboard views)
│   ├── components/          shared visual components (kokonutui, ASCII, etc.)
│   ├── lib/                 data loading, types, formatting, API client
│   ├── data/                polymap_demo.json + model_predictions.json (the app's data)
│   └── .env.local           NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000
├── Claude outputs/          POLYMAP_Industry_Presentation.pptx
├── backup_before_baselines/ old backup (gitignored)
└── untitled folder(.zip)    scratch (gitignored)
```

The `frontend/` folder in this project was **replaced on 3 Nov 2026** with a newer version that had been sitting loose in `~/Downloads/frontend`.

---

## 3. First-time setup (or after moving the folder)

You can move the whole `polymap-updated` folder anywhere. The code locates files relative to itself. The **only** thing that breaks is the backend's virtual environment, which records its own absolute path.

### Backend (Python 3.11, managed with `uv`)

```bash
cd backend
rm -rf .venv                              # only if the folder was moved or the env is broken
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[model,api,dev]"      # add ,viz for matplotlib/plotly/streamlit
```

Dependency groups in `pyproject.toml`:

| Extra | What it adds | Needed for |
|---|---|---|
| *(core)* | numpy, pandas, pyarrow, requests, scikit-learn, pyyaml, scipy | fetch, build, screen, funding validation |
| `model` | torch, torch-geometric | `train`, `predict_mock.py`, benchmarks |
| `api` | fastapi, uvicorn | the HTTP API (§7) |
| `dev` | pytest, ruff | tests and linting |
| `viz` | matplotlib, networkx, plotly, streamlit | plots and notebooks |

If you don't use `uv`, `python3.11 -m venv .venv && pip install -e ".[model,api,dev]"` works too.

Check it worked:

```bash
polymap --help
pytest -q
```

### Frontend (Node 20+)

```bash
cd frontend
npm install           # node_modules is not in the folder or in git
```

### Secrets / environment variables

| Variable | Where | Used by |
|---|---|---|
| `ETHERSCAN_API_KEY` | export in your terminal before running | `collect_polygon_transfers.py`, `recover_erc20.py` (Etherscan API V2, Polygon chain 137) |
| `NEXT_PUBLIC_API_BASE` | `frontend/.env.local` | frontend API client; default `http://127.0.0.1:8000` |

`.env*` files are gitignored, so recreate `frontend/.env.local` after a fresh clone:

```bash
echo "NEXT_PUBLIC_API_BASE=http://127.0.0.1:8000" > frontend/.env.local
```

---

## 4. Running the frontend

```bash
cd frontend
npm run dev        # http://localhost:3000
npm run build && npm start    # production build
npm run lint
```

- `/` is the landing page (navbar, crystal-ball hero, feature cards).
- `/dashboard` is the main app (`AppShell`). Its views are home, markets, network, pairs, wallets, charts, architecture, results and predictions.
- The dashboard reads **static JSON** from `frontend/data/`:
  - `polymap_demo.json` (~575 KB): the demo wallets, markets and relationships.
  - `model_predictions.json`: output of `backend/predict_mock.py`, drawn by the Predictions tab.
- `lib/api.ts` is a client for the live backend API, but the current dashboard views load the JSON files above. The backend does not need to be running to view the site.

**Next.js note:** this is Next.js 16, which has breaking changes from older versions (see `frontend/AGENTS.md`). Check `node_modules/next/dist/docs/` before changing routing or config.

---

## 5. Running the backend pipeline

All commands run from `backend/` with the venv active. Every stage reads from disk and writes to disk, so any stage can be re-run on its own.

```bash
polymap markets --min-volume 200000 --open        # find market condition IDs
# put an ID in configs/markets/<name>.yaml, then either:
polymap all -c configs/default.yaml -c configs/markets/<name>.yaml
# or stage by stage:
polymap fetch      -c configs/default.yaml -c configs/markets/<name>.yaml
polymap build      -c configs/default.yaml -c configs/markets/<name>.yaml
polymap train      -c configs/default.yaml -c configs/markets/<name>.yaml --negative hard --baselines
polymap candidates -c configs/default.yaml -c configs/markets/<name>.yaml
polymap screen     -c configs/default.yaml -c configs/markets/<name>.yaml
polymap null       -c configs/default.yaml -c configs/markets/<name>.yaml
polymap validate-funding --run-name <run> --funding-file <transfers.parquet>
polymap funding-robustness --run-name <run> --funding-file <transfers.parquet>
```

Stages: `fetch, build, train, candidates, screen, null, validate-funding, funding-robustness, all, markets`.

Useful flags:

- `-c FILE`: layer config files; later ones win.
- `--run-name NAME`: output goes to `data/processed/NAME/` and `data/artifacts/NAME/`, and nothing else is overwritten.
- `--set section.field=value`: override any config value, e.g. `--set model.epochs=20`.
- `--window N`: override `graph.window_seconds`. This is the key research parameter.
- `--negative random|hard|popularity`: training negative sampler.
- `--candidates-only`: train only on edges between candidate wallets.
- `--refresh`: **re-fetch and replace** cached trades. This changes all downstream results, so avoid it unless you mean it.

**The tuned model configuration** (the one in the results table, §9):

```bash
polymap train -c configs/default.yaml -c configs/markets/xi_2027.yaml \
  --run-name xi_2027_63d_p50 --negative hard --candidates-only \
  --set model.pair_features=true --set model.train_only_features=true \
  --set model.learning_rate=0.003 --set model.epochs=20
```

**Adding a new market for replication:** copy `configs/markets/replication_template.yaml` to `configs/markets/<name>.yaml`. Change only `run_name` and the market ID. `history_days: 63` means the market must be **open and active now**, because it fetches the last 63 days.

Output locations:

```
data/raw/trades_<market>_d63.parquet  + .manifest.json (sha256, rows, time span)
data/raw/onchain_<run>/               on-chain transfers from Etherscan
data/processed/<run>/                 trades, wallet_features, temporal_edges, candidates, event_scores
data/artifacts/<run>/                 metrics.json, *_manifest.json, funding_validation*.json, CSV results
```

Tests: `pytest -q`. The key one is `test_matches_naive`, which checks the fast edge builder against the original loop.

---

## 6. Regenerating the Predictions tab data

`backend/predict_mock.py` simulates 60 days of trades for the demo markets and trains the **real** PolymapTGN model on them. It then writes `frontend/data/model_predictions.json`.

```bash
cd backend && source .venv/bin/activate     # needs the `model` extra (torch)
python predict_mock.py                      # default: 12 markets, 36 wallets, 20 epochs
python predict_mock.py --markets 3 --epochs 10      # quicker
python predict_mock.py --real --epochs 5            # use the real xi_2027_63d_p50 trades
```

By default it reads `../frontend/data/polymap_demo.json` and writes `../frontend/data/model_predictions.json`, relative to `backend/`. This was fixed on 3 Nov 2026; it previously pointed one level too high. Override with `--demo` and `--out`. The trades are synthetic; the model, training and scoring code are unchanged.

---

## 7. The backend API

```bash
cd backend && source .venv/bin/activate
uvicorn polymap.api.main:app --reload --port 8000     # docs at http://127.0.0.1:8000/docs
```

CORS allows `localhost:3000` and `localhost:5173`.

| Method | Path | Returns |
|---|---|---|
| GET | `/health` | status |
| GET | `/markets` | market list from Polymarket |
| GET | `/runs` | runs found under `data/` |
| GET | `/runs/{run}/metrics` | `metrics.json` |
| GET | `/runs/{run}/screening` | candidate relationships |
| GET | `/runs/{run}/funding-validation` (+ `/pairs`) | funding validation summary / pairs |
| GET | `/runs/{run}/edges`, `/wallets`, `/config` | graph edges, wallet features, run config |
| POST | `/jobs/fetch`, `/jobs/build`, `/jobs/train`, … | start a pipeline stage in the background |

---

## 8. Research scripts in `backend/`

Run each from `backend/` with the venv active. Several use paths like `data/processed/...` relative to the current folder, so **always `cd backend` first**.

| Script | Purpose |
|---|---|
| `tune_tgnn.py` | Hyperparameter tuning on a **validation** window only; never touches the test window. |
| `run_final_test.py` | The pre-committed final test: 3 variants × 5 fixed seeds, run once. No tuning logic. |
| `run_benchmark.py` | Fair multi-market benchmark: model vs. baselines vs. (optional) TGN on identical splits and negatives. |
| `run_ablations.py` | Controlled ablations, changing one factor at a time (components, negative sampler, window, min_events…). |
| `collect_polygon_transfers.py` | Pulls Polygon transactions and ERC-20 transfers for candidate wallets from Etherscan. Needs `ETHERSCAN_API_KEY`. |
| `recover_erc20.py` | Re-fetches ERC-20 histories that hit Etherscan's 10,000-row cap. |
| `find_truncated_wallets.py` | Lists wallets whose on-chain history was cut off by that cap (→ `truncated_*.csv`). |
| `export_funding_summary.py` | Shrinks the raw transfer table into small shareable CSVs (edges, funder fan-out, coverage). |
| `funding_confirmatory.py` | **Pre-registered**, coverage-matched funding test. The spec in `configs/prereg/*.yaml` is hashed into a `.lock.json` before analysis; `run` refuses without a matching lock. |
| `prospective.py` | Frozen prospective test (freeze → build → evaluate once) using only data before a freeze time. |
| `rank_wallets.py` | Ranks wallets for a run. |
| `predict_mock.py` | Generates the frontend Predictions data (§6). |

---

## 9. Runs that exist and headline results

Runs with artifacts in `backend/data/artifacts/`: `_smoke`, `benchmark`, `iran_2027`, `iran_w30`, `iran_w60`, `rep_iran`, `xi_2027`, `xi_2027_63d`, `xi_2027_63d_p50`.

- **`xi_2027_63d_p50`** is the main tuned run (Xi market, 63 days, max 50 pairs per trade, candidate-only graph).
- **`rep_iran`** is the replication on an Iran market.
- **`iran_w30` / `iran_w60`** are window-size sweeps.

Final test on `xi_2027_63d_p50` (hard negatives, 5 fixed seeds, from `backend/docs/tgnn_tuning_results.md`):

| Variant | MRR (mean ± SD) | AUC | Recency-baseline MRR | MRR gap vs recency |
|---|---:|---:|---:|---:|
| Plain, leak-free | 0.341 ± 0.019 | 0.689 | 0.433 | −0.093 |
| **Tuned: memory + pair history** | **0.586 ± 0.015** | **0.865** | 0.433 | **+0.153** |
| Memory off, pair history only | 0.182 ± 0.033 | 0.563 | 0.433 | −0.252 |

Only the tuned model beats the recency baseline, and removing node memory destroys the gain. Credit the improvement to the *combination* of memory and pair history. Call it a custom memory-based temporal model, not a reimplementation of TGN, TGAT or JODIE.

---

## 10. Data: where it lives and how to share it

`backend/data/` is many GB, mostly `data/raw/onchain_*/raw_by_endpoint/` Etherscan dumps and Parquet files. It is gitignored and will not go to GitHub.

Options for sharing it:

1. **Hugging Face Datasets (recommended).** It's free, versioned, can be private and handles large sizes.
   ```bash
   pip install huggingface_hub && huggingface-cli login
   huggingface-cli upload <hf-username>/polymap-data backend/data --repo-type=dataset
   # to get it back on a new machine:
   huggingface-cli download <hf-username>/polymap-data --repo-type=dataset --local-dir backend/data
   ```
2. **GitHub Release assets.** Zip it into archives of 2 GB or less and attach them to a release.
3. **Google Drive / Dropbox zip** with the link in this README. Simple, but not versioned.

Avoid Git LFS for this. GitHub's free LFS quota is small for multi-GB data.

Worth committing even though `data/` is ignored: the small `*_manifest.json` and `metrics.json` files in `data/artifacts/`. They record exactly which input produced each result. Raw trades and on-chain data can be re-fetched with the scripts, but **Polymarket's API returns different slices over time**, so the cached copies are the only way to reproduce exact numbers. Back them up.

> Data hosted at: _(fill in the link once uploaded)_

---

## 11. Git notes and known problems

**Remote:** `origin` → `https://github.com/ArulNanda787/polymap.git`, branch `main`.

**Nested repo:** `frontend/` contains its own `.git/` folder (left over from `create-next-app`). Git treats it as a separate repository, so the frontend files **won't be committed** in the main repo. If the frontend should be part of this repo:

```bash
rm -rf frontend/.git
git add frontend
```

**`.gitignore` (fixed 3 Nov 2026):** the old file used bare patterns (`data/`, `**/data/`, `models/`, `runs/`) that also matched source code: `backend/src/polymap/data/`, `backend/src/polymap/models/`, `frontend/data/` and `frontend/app/runs/`. Those rules are now anchored to `/backend/data/`, `/backend/models/` and similar. If you ever add a new ignore rule for a folder name, start it with `/` so it can't catch a same-named source folder deeper in the tree. Run `git status` after changing `.gitignore` to confirm source files still appear.

**Other known issues:**

- `frontend/app/components/polymap/predictions-view.tsx` (around line 456) shows the instruction `cd ~/Downloads/polymap-updated/backend`. It's only on-screen text, so update it if the folder moves.
- `notebooks/test.py` uses `Path("../data/processed")`, so run it from inside `backend/notebooks/`.
- `frontend/app/runs/[name]/page.tsx` is essentially empty (3 bytes).
- Etherscan's free tier caps list endpoints at 10,000 rows per wallet, so some funding histories are truncated. `truncated_*.csv` and `funding-robustness --exclude-wallets-file` handle this as a sensitivity check.

---

## 12. Things to remember

- **Never edit `configs/default.yaml` to run an experiment.** Make a new config or use `--set` so runs stay comparable. Each manifest stores a config fingerprint.
- **Don't use `--refresh` casually.** It replaces cached trades and changes every downstream number.
- **Don't look at the test window when tuning.** Use `tune_tgnn.py` for tuning, then run `run_final_test.py` once.
- **Pre-registered tests:** don't edit `configs/prereg/*.yaml` after its `.lock.json` exists. That defeats the point and the script will refuse to run.
- **No outcome leakage:** never add anything derived from market resolution to node features. `test_features_have_no_outcome_leakage` guards this.
- **Markets are never joined.** Edges are only built within a single market.
- **Moved the folder?** Rebuild `backend/.venv` (§3). Everything else keeps working.
- **New machine?** Clone the repo, download the data (§10), set up the backend and frontend (§3), recreate `frontend/.env.local`, export `ETHERSCAN_API_KEY`.
- More backend design detail (the edge-builder speed-up, caching, how to add features, baselines, samplers and stages) is in `backend/README.md`.

---

## 13. License

**All rights reserved** (added 3 Nov 2026, see `LICENSE`). The repo is public only so it can be viewed. Nobody may use the code, models, data or findings for research, commercial or any other purpose without written permission. Do **not** add an open-source license (MIT, Apache, GPL, etc.) or a Creative Commons license unless you intend to give those rights away; once granted, they can't be taken back from people who already received the code under them.
