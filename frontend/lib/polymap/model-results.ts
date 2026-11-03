/**
 * Real outputs of the PolymapTGN model, copied from the backend run artifacts.
 * Generated from polymap-updated/backend/data/artifacts/ (metrics.json,
 * train_manifest.json, benchmark/summary.csv, benchmark/protocol.json,
 * xi_2027_63d_p50/ablations/components_summary.csv, screen_manifest.json,
 * funding_validation.json). All rankings use hard negatives, 50 per event.
 * Regenerate rather than hand-edit if the backend is re-run.
 */

export const modelResults = {
  "source": {
    "run": "xi_2027_63d_p50",
    "writtenAt": "2026-10-24T08:48:35.844812+00:00",
    "fingerprint": "2412c5350ad0",
    "negatives": 50,
    "seeds": [
      42,
      43,
      44,
      45,
      46
    ]
  },
  "graph": {
    "num_nodes": 163,
    "num_edges": 78306,
    "node_features": 9,
    "edge_features": 4
  },
  "split": {
    "train_events": 54813,
    "test_events": 23493,
    "cutoff_ts": 1790709909,
    "seen_events": 20288,
    "cold_start_events": 3205,
    "train_isolated_wallets": 8
  },
  "runs": {
    "xi_2027_63d_p50": {
      "trainEvents": 54813,
      "testEvents": 23493,
      "testEventsScored": 9638,
      "coldStartEvents": 3205
    },
    "rep_iran": {
      "trainEvents": 165151,
      "testEvents": 70801,
      "testEventsScored": 38163,
      "coldStartEvents": 19882
    }
  },
  "benchmark": {
    "xi_2027_63d_p50": [
      {
        "id": "tuned",
        "label": "PolymapTGN (tuned)",
        "kind": "ours",
        "seeds": 5,
        "mrr": 0.4978,
        "mrrStd": 0.0229,
        "hits10": 0.7073,
        "auc": 0.7913
      },
      {
        "id": "recency",
        "label": "Recency-weighted count",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.3485,
        "mrrStd": null,
        "hits10": 0.578,
        "auc": 0.7376
      },
      {
        "id": "tgn",
        "label": "TGN (PyG reference)",
        "kind": "learned",
        "seeds": 5,
        "mrr": 0.3444,
        "mrrStd": 0.0441,
        "hits10": 0.5871,
        "auc": 0.7653
      },
      {
        "id": "edgebank_1h",
        "label": "EdgeBank 1h",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.2771,
        "mrrStd": null,
        "hits10": 0.5282,
        "auc": 0.706
      },
      {
        "id": "plain",
        "label": "PolymapTGN (plain)",
        "kind": "ours",
        "seeds": 5,
        "mrr": 0.2311,
        "mrrStd": 0.0161,
        "hits10": 0.3906,
        "auc": 0.6326
      },
      {
        "id": "edgebank_24h",
        "label": "EdgeBank 24h",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.2263,
        "mrrStd": null,
        "hits10": 0.5308,
        "auc": 0.7375
      },
      {
        "id": "adamic_adar",
        "label": "Adamic-Adar",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1459,
        "mrrStd": null,
        "hits10": 0.2515,
        "auc": 0.4797
      },
      {
        "id": "pair_count",
        "label": "Pair count",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1444,
        "mrrStd": null,
        "hits10": 0.2154,
        "auc": 0.4791
      },
      {
        "id": "common_neighbours",
        "label": "Common neighbours",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1413,
        "mrrStd": null,
        "hits10": 0.2521,
        "auc": 0.477
      },
      {
        "id": "popularity",
        "label": "Popularity",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1254,
        "mrrStd": null,
        "hits10": 0.2099,
        "auc": 0.4973
      },
      {
        "id": "edgebank_unlimited",
        "label": "EdgeBank unlimited",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.0989,
        "mrrStd": null,
        "hits10": 0.2239,
        "auc": 0.4918
      }
    ],
    "rep_iran": [
      {
        "id": "tuned",
        "label": "PolymapTGN (tuned)",
        "kind": "ours",
        "seeds": 5,
        "mrr": 0.5015,
        "mrrStd": 0.0098,
        "hits10": 0.6935,
        "auc": 0.7856
      },
      {
        "id": "recency",
        "label": "Recency-weighted count",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.3457,
        "mrrStd": null,
        "hits10": 0.5584,
        "auc": 0.7011
      },
      {
        "id": "tgn",
        "label": "TGN (PyG reference)",
        "kind": "learned",
        "seeds": 5,
        "mrr": 0.3323,
        "mrrStd": 0.0092,
        "hits10": 0.5387,
        "auc": 0.7273
      },
      {
        "id": "edgebank_1h",
        "label": "EdgeBank 1h",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.2518,
        "mrrStd": null,
        "hits10": 0.4838,
        "auc": 0.6732
      },
      {
        "id": "edgebank_24h",
        "label": "EdgeBank 24h",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.2448,
        "mrrStd": null,
        "hits10": 0.5306,
        "auc": 0.7083
      },
      {
        "id": "pair_count",
        "label": "Pair count",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1589,
        "mrrStd": null,
        "hits10": 0.2376,
        "auc": 0.4503
      },
      {
        "id": "adamic_adar",
        "label": "Adamic-Adar",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1501,
        "mrrStd": null,
        "hits10": 0.2513,
        "auc": 0.4612
      },
      {
        "id": "common_neighbours",
        "label": "Common neighbours",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1448,
        "mrrStd": null,
        "hits10": 0.2471,
        "auc": 0.4596
      },
      {
        "id": "plain",
        "label": "PolymapTGN (plain)",
        "kind": "ours",
        "seeds": 5,
        "mrr": 0.1305,
        "mrrStd": 0.0052,
        "hits10": 0.2656,
        "auc": 0.5479
      },
      {
        "id": "edgebank_unlimited",
        "label": "EdgeBank unlimited",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.1088,
        "mrrStd": null,
        "hits10": 0.2323,
        "auc": 0.477
      },
      {
        "id": "popularity",
        "label": "Popularity",
        "kind": "baseline",
        "seeds": 1,
        "mrr": 0.0998,
        "mrrStd": null,
        "hits10": 0.1801,
        "auc": 0.4292
      }
    ]
  },
  "ablation": [
    {
      "id": "full",
      "label": "Full model",
      "seeds": 5,
      "mrr": 0.5162,
      "mrrStd": 0.0124,
      "hits10": 0.7173,
      "auc": 0.7975
    },
    {
      "id": "no_count_features",
      "label": "No pair-count feature",
      "seeds": 5,
      "mrr": 0.4336,
      "mrrStd": 0.0132,
      "hits10": 0.6537,
      "auc": 0.7842
    },
    {
      "id": "no_memory",
      "label": "No node memory",
      "seeds": 5,
      "mrr": 0.4179,
      "mrrStd": 0.0276,
      "hits10": 0.6407,
      "auc": 0.7169
    },
    {
      "id": "recency",
      "label": "Recency baseline",
      "seeds": 1,
      "mrr": 0.3485,
      "mrrStd": null,
      "hits10": 0.578,
      "auc": 0.7376
    },
    {
      "id": "no_pair_history",
      "label": "No pair-history head",
      "seeds": 5,
      "mrr": 0.2294,
      "mrrStd": 0.007,
      "hits10": 0.3905,
      "auc": 0.6396
    },
    {
      "id": "no_recency_features",
      "label": "No decayed-recency features",
      "seeds": 5,
      "mrr": 0.1733,
      "mrrStd": 0.0191,
      "hits10": 0.2839,
      "auc": 0.5442
    },
    {
      "id": "no_memory_no_pair_history",
      "label": "No memory, no pair history",
      "seeds": 5,
      "mrr": 0.1332,
      "mrrStd": 0.0048,
      "hits10": 0.2351,
      "auc": 0.5142
    }
  ],
  "recency": [
    {
      "id": "never_seen_pair",
      "n": 791,
      "auc": 0.5974
    },
    {
      "id": "recent_pair",
      "n": 18242,
      "auc": 0.7359
    },
    {
      "id": "stale_pair",
      "n": 4460,
      "auc": 0.6355
    }
  ],
  "screening": {
    "persistent_loose": 375,
    "one_off_loose": 257,
    "persistent_tight": 117,
    "one_off_tight": 77
  },
  "funding": {
    "candidates": 397,
    "controls": 1090,
    "directTransfer": {
      "candidate": 0.1511,
      "control": 0.0716
    },
    "sharedFunder": {
      "candidate": 0.9899,
      "control": 0.9881
    },
    "pValueAnyLink": 0.7999
  }
} as const;

export type BenchmarkRun = keyof typeof modelResults.benchmark;
