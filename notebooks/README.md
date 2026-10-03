# Notebooks

Notebooks are for **looking at things**, never for defining them. Every
function lives in `src/polymap/`; a notebook imports it, runs it, and plots.

The rule exists because the old chain defined `build_temporal_edges` inside a
notebook, which meant it could not be tested, could not be reused, and existed
in three slightly different versions across three files.

Start every notebook with:

```python
%load_ext autoreload
%autoreload 2

from polymap.config import load_config
cfg = load_config("../configs/default.yaml", "../configs/markets/iran_2027.yaml")
```

Then read stage outputs rather than recomputing them:

```python
import pandas as pd
edges = pd.read_parquet(cfg.processed_dir / "temporal_edges.parquet")
pairs = pd.read_parquet(cfg.processed_dir / "candidate_relationships.parquet")
```

No `sys.path.append("../src")` needed once you have run `pip install -e .`.

Suggested notebooks, all read-only over pipeline outputs:

- `01_market_survey.ipynb` — pick markets, check volume and wallet counts
- `02_data_quality.ipynb` — trade-count distribution, time span, API-cap check
- `03_graph_structure.ipynb` — degree, edge stats, window sensitivity
- `04_model_results.ipynb` — metrics across negative samplers
- `05_embedding_validation.ipynb` — embedding vs behavioural similarity
- `06_relationships.ipynb` — screening categories, top pairs
- `07_visualisation.ipynb` — the interface prototype
