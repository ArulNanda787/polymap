"""Extra non-learned baselines. Importing this module registers them.

    edgebank_unlimited   1 if the pair has EVER co-traded, else 0 (EdgeBank, unlimited memory)
    edgebank_24h         1 if the pair co-traded within the last 24 hours (EdgeBank, time window)
    edgebank_1h          same with 1 hour
    adamic_adar          common neighbours weighted by 1/log(degree)

EdgeBank (Poursafaei et al., 2022) is the standard "memorise what you have seen"
reference for temporal link prediction. It is binary, so it produces many ties;
`metrics.per_event_ranking` resolves ties in expectation, which neither rewards
nor punishes it.

Protocol is identical to `models/baselines.py`: `fit` on the train events, then
score a batch BEFORE `update`-ing it.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .baselines import ALL_BASELINES, Baseline, _key


class EdgeBankUnlimited(Baseline):
    name = "edgebank_unlimited"
    label = "EdgeBank (unlimited)"

    def _init_state(self):
        self.seen: set[int] = set()

    def update(self, sources, targets, times):
        n = self._n
        for s, t in zip(np.asarray(sources).tolist(), np.asarray(targets).tolist()):
            self.seen.add(_key(s, t, n))

    def score(self, source, target, at_time=0):
        return 1.0 if _key(int(source), int(target), self._n) in self.seen else 0.0


class EdgeBankWindow(Baseline):
    name = "edgebank_24h"
    label = "EdgeBank (24h window)"
    window_hours = 24.0

    def _init_state(self):
        self.last: dict[int, int] = {}

    def update(self, sources, targets, times):
        n = self._n
        for s, t, ts in zip(np.asarray(sources).tolist(), np.asarray(targets).tolist(),
                            np.asarray(times).tolist()):
            k = _key(s, t, n)
            if ts > self.last.get(k, -1):
                self.last[k] = ts

    def score(self, source, target, at_time=0):
        last = self.last.get(_key(int(source), int(target), self._n))
        if last is None:
            return 0.0
        return 1.0 if (int(at_time) - last) <= self.window_hours * 3600.0 else 0.0


class EdgeBank1h(EdgeBankWindow):
    name = "edgebank_1h"
    label = "EdgeBank (1h window)"
    window_hours = 1.0


class AdamicAdarBaseline(Baseline):
    name = "adamic_adar"
    label = "Adamic-Adar"

    def _init_state(self):
        self.nbrs: dict[int, set[int]] = defaultdict(set)

    def update(self, sources, targets, times):
        for s, t in zip(np.asarray(sources).tolist(), np.asarray(targets).tolist()):
            if s != t:
                self.nbrs[s].add(t)
                self.nbrs[t].add(s)

    def score(self, source, target, at_time=0):
        a, b = self.nbrs.get(int(source)), self.nbrs.get(int(target))
        if not a or not b:
            return 0.0
        total = 0.0
        for z in a & b:
            deg = len(self.nbrs[z])
            if deg > 1:
                total += 1.0 / np.log(deg)
        return float(total)


ALL_BASELINES.update({
    "edgebank_unlimited": EdgeBankUnlimited,
    "edgebank_24h": EdgeBankWindow,
    "edgebank_1h": EdgeBank1h,
    "adamic_adar": AdamicAdarBaseline,
})
