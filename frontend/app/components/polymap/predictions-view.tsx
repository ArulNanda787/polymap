"use client";

import { useMemo, useState } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";

import raw from "@/data/model_predictions.json";
import { addr } from "@/lib/polymap/data";
import type { WalletType } from "@/lib/polymap/types";

import {
  Badge,
  Button,
  Card,
  CardBody,
  PageHeader,
  Segmented,
  Select,
  Stat,
  cn,
} from "./ui";

/* --------------------------------------------------
   Data: real PolymapTGN predictions on mock trades

   Written by `polymap-updated/backend/predict_mock.py`, which simulates
   trades from the demo dataset, trains the actual PolymapTGN code on the
   first 70% of time, and scores every wallet pair at the train/test cutoff.
-------------------------------------------------- */

type PairRow = [number, number, number, number, number]; // a, b, probability, test co-trades, train co-trades

interface RawMarket {
  market_id: string;
  market_name: string;
  trades: number;
  edges: number;
  train_events: number;
  test_events: number;
  wallets_in_model?: number;
  train_window: [number, number];
  test_window: [number, number];
  metrics: {
    test_events: number;
    negatives_per_event: number;
    mrr: number;
    "hits@10": number;
    roc_auc: number;
    recency_mrr: number;
    "recency_hits@10": number;
    final_train_loss: number | null;
    top1_all_wallets?: number;
    mrr_all_wallets?: number;
  };
  wallets: { wallet_id: string; address: string; wallet_type: WalletType }[];
  pairs: PairRow[];
}

interface RawPredictions {
  generated_at?: string;
  model?: { name: string; epochs: number; learning_rate: number; seed: number };
  truth_min_events?: number;
  is_mock_data?: boolean;
  markets: RawMarket[];
}

const DATA = raw as unknown as RawPredictions;
const TRUTH_MIN = DATA.truth_min_events ?? 2;

const W = 900;
const H = 600;

const WALLET_COLOURS: Record<WalletType, string> = {
  Whale: "#7C3AED",
  "Frequent trader": "#2563EB",
  "Occasional trader": "#16A34A",
  "New wallet": "#EA580C",
};

type Outcome = "hit" | "false_alarm" | "missed" | "true_negative";

const OUTCOME_STYLE: Record<
  Exclude<Outcome, "true_negative">,
  { label: string; color: string; dash: string; help: string }
> = {
  hit: {
    label: "Correct prediction",
    color: "#0D9488",
    dash: "",
    help: "The model predicted a link and the pair did co-trade in the test window.",
  },
  false_alarm: {
    label: "False alarm",
    color: "#EA580C",
    dash: "6 4",
    help: "The model predicted a link but the pair did not co-trade.",
  },
  missed: {
    label: "Missed",
    color: "#A3A3A3",
    dash: "2 4",
    help: "The pair co-traded, but the model scored it below the threshold.",
  },
};

interface Prediction {
  key: string;
  a: number;
  b: number;
  score: number;
  testEvents: number;
  trainEvents: number;
  actual: boolean;
}

function toPredictions(m: RawMarket): Prediction[] {
  return m.pairs.map(([a, b, score, testEvents, trainEvents]) => ({
    key: `${a}|${b}`,
    a,
    b,
    score,
    testEvents,
    trainEvents,
    actual: testEvents >= TRUTH_MIN,
  }));
}

/** Predict as many links as were active before the cutoff (uses no test data). */
function defaultThreshold(preds: Prediction[]) {
  const k = Math.max(1, preds.filter((p) => p.trainEvents >= TRUTH_MIN).length);
  const sorted = preds.map((p) => p.score).sort((x, y) => y - x);
  const t = sorted[Math.min(k, sorted.length) - 1] ?? 0.5;
  return Math.max(0.005, Math.floor(t * 1000) / 1000);
}

function outcomeOf(p: Prediction, threshold: number): Outcome {
  const predicted = p.score >= threshold;
  if (predicted && p.actual) return "hit";
  if (predicted) return "false_alarm";
  if (p.actual) return "missed";
  return "true_negative";
}

const fmtDate = (unix: number) =>
  new Date(unix * 1000).toLocaleDateString("en-GB", { day: "numeric", month: "short" });

/* --------------------------------------------------
   Force-directed layout (same approach as the Network tab)
-------------------------------------------------- */

/**
 * Spread-out force layout.
 *
 * Laying out every link pulls a dense graph into one ball, so only a sparse
 * backbone (each wallet's few strongest links) acts as springs. Repulsion is
 * strong, steps are capped so nothing flies off, and the result is stretched
 * to fill the canvas.
 */
function computeLayout(n: number, preds: Prediction[]) {
  if (n === 0) return [] as { x: number; y: number }[];

  // Backbone: each wallet's 3 strongest links (most co-trades before the
  // cutoff, then highest score).
  const backbone = new Map<string, { a: number; b: number }>();
  const byWallet: Prediction[][] = Array.from({ length: n }, () => []);
  for (const p of preds) {
    byWallet[p.a].push(p);
    byWallet[p.b].push(p);
  }
  for (const list of byWallet) {
    list
      .sort((x, y) => y.trainEvents - x.trainEvents || y.score - x.score)
      .slice(0, 3)
      .forEach((p) => backbone.set(p.key, { a: p.a, b: p.b }));
  }
  const links = [...backbone.values()];

  const r0 = Math.min(W, H) * 0.35;
  const pos = Array.from({ length: n }, (_, i) => ({
    x: W / 2 + Math.cos((i / n) * Math.PI * 2) * r0,
    y: H / 2 + Math.sin((i / n) * Math.PI * 2) * r0,
  }));
  const rest = 160;
  const charge = 40000;
  const spring = 0.01;
  const iterations = 400;
  const maxStep = 30;

  for (let it = 0; it < iterations; it++) {
    const cool = 1 - it / iterations;
    const dx = new Array(n).fill(0);
    const dy = new Array(n).fill(0);

    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const vx = pos[i].x - pos[j].x;
        const vy = pos[i].y - pos[j].y;
        const d2 = Math.max(vx * vx + vy * vy, 1);
        const d = Math.sqrt(d2);
        const f = charge / d2;
        dx[i] += (vx / d) * f;
        dy[i] += (vy / d) * f;
        dx[j] -= (vx / d) * f;
        dy[j] -= (vy / d) * f;
      }
    }

    for (const l of links) {
      const vx = pos[l.b].x - pos[l.a].x;
      const vy = pos[l.b].y - pos[l.a].y;
      const d = Math.max(Math.sqrt(vx * vx + vy * vy), 1);
      const f = (d - rest) * spring;
      dx[l.a] += (vx / d) * f;
      dy[l.a] += (vy / d) * f;
      dx[l.b] -= (vx / d) * f;
      dy[l.b] -= (vy / d) * f;
    }

    for (let i = 0; i < n; i++) {
      dx[i] += (W / 2 - pos[i].x) * 0.008;
      dy[i] += (H / 2 - pos[i].y) * 0.008;
      pos[i].x += Math.max(-maxStep, Math.min(maxStep, dx[i] * cool));
      pos[i].y += Math.max(-maxStep, Math.min(maxStep, dy[i] * cool));
    }
  }

  // Stretch to fill the canvas, leaving room for labels at the edges.
  const xs = pos.map((p) => p.x);
  const ys = pos.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const spanX = Math.max(Math.max(...xs) - minX, 1);
  const spanY = Math.max(Math.max(...ys) - minY, 1);
  const padX = 70;
  const padY = 50;
  return pos.map((p) => ({
    x: padX + ((p.x - minX) / spanX) * (W - 2 * padX),
    y: padY + ((p.y - minY) / spanY) * (H - 2 * padY),
  }));
}

/* --------------------------------------------------
   Side panel
-------------------------------------------------- */

function ScoreBar({ score, threshold, max = 1 }: { score: number; threshold: number; max?: number }) {
  return (
    <div className="relative h-2 w-full rounded-full bg-neutral-100">
      <div
        className={cn(
          "absolute inset-y-0 left-0 rounded-full",
          score >= threshold ? "bg-teal-600" : "bg-neutral-400"
        )}
        style={{ width: `${Math.min(score / max, 1) * 100}%` }}
      />
      <div
        className="absolute inset-y-[-3px] w-px bg-neutral-900"
        style={{ left: `${Math.min(threshold / max, 1) * 100}%` }}
      />
    </div>
  );
}

function WalletRanking({
  market,
  walletIdx,
  predictions,
  threshold,
  max,
  onPick,
}: {
  market: RawMarket;
  walletIdx: number;
  predictions: Prediction[];
  threshold: number;
  max: number;
  onPick: (key: string) => void;
}) {
  const wallet = market.wallets[walletIdx];
  const ranked = predictions
    .filter((p) => p.a === walletIdx || p.b === walletIdx)
    .sort((x, y) => y.score - x.score);
  const firstTrue = ranked.findIndex((p) => p.actual);
  const actualCount = ranked.filter((p) => p.actual).length;

  return (
    <div className="space-y-5">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Wallet</p>
        <h3 className="mt-1 font-mono text-sm font-semibold text-neutral-950">{addr(wallet.address)}</h3>
        <p className="mt-1 text-sm text-neutral-500">{wallet.wallet_type}</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Stat label="Actual partners" value={<span className="tabular-nums">{actualCount}</span>} />
        <Stat
          label="First correct guess"
          value={<span className="tabular-nums">{firstTrue >= 0 ? `Rank ${firstTrue + 1}` : "—"}</span>}
        />
      </div>

      <div>
        <h4 className="text-sm font-semibold text-neutral-950">Model&apos;s ranking of partners</h4>
        <p className="mt-1 text-xs text-neutral-500">
          Average probability the model gave each wallet as this one&apos;s partner, each time it traded after the cutoff. Top 10 shown.
        </p>
        <ol className="mt-3 space-y-1">
          {ranked.slice(0, 10).map((p, i) => {
            const other = market.wallets[p.a === walletIdx ? p.b : p.a];
            return (
              <li key={p.key}>
                <button
                  type="button"
                  onClick={() => onPick(p.key)}
                  className="grid w-full grid-cols-[1.5rem_minmax(0,1fr)_3rem] items-center gap-x-3 gap-y-1 rounded-lg px-2 py-2 text-left transition-colors hover:bg-teal-50/60"
                >
                  <span className="text-xs tabular-nums text-neutral-500">{i + 1}</span>
                  <span className="flex min-w-0 items-center gap-2">
                    <span className="truncate font-mono text-xs text-neutral-950">{addr(other.address, 4)}</span>
                    {p.actual && (
                      <Badge className="bg-teal-50 px-1.5 py-0 text-[10px] text-teal-800">co-traded</Badge>
                    )}
                    {p.actual && p.trainEvents === 0 && (
                      <Badge className="px-1.5 py-0 text-[10px]">new pair</Badge>
                    )}
                  </span>
                  <span className="text-right text-xs font-semibold tabular-nums text-neutral-950">
                    {p.score.toFixed(3)}
                  </span>
                  <span />
                  <span className="col-span-2">
                    <ScoreBar score={p.score} threshold={threshold} max={max} />
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      </div>
    </div>
  );
}

function PairDetail({
  market,
  p,
  threshold,
  max,
}: {
  market: RawMarket;
  p: Prediction;
  threshold: number;
  max: number;
}) {
  const outcome = outcomeOf(p, threshold);
  const style = outcome === "true_negative" ? null : OUTCOME_STYLE[outcome];

  return (
    <div className="space-y-5">
      <div>
        <p className="text-xs font-medium uppercase tracking-wide text-neutral-500">Wallet pair</p>
        <h3 className="mt-1 font-mono text-sm font-semibold text-neutral-950">
          {addr(market.wallets[p.a].address, 4)} ↔ {addr(market.wallets[p.b].address, 4)}
        </h3>
        {style && (
          <p className="mt-2 inline-flex items-center gap-2 text-sm font-medium" style={{ color: style.color }}>
            <span className="h-2.5 w-2.5 rounded-full" style={{ background: style.color }} />
            {style.label}
          </p>
        )}
      </div>

      <div>
        <div className="flex items-baseline justify-between text-sm">
          <span className="text-neutral-600">Prediction score</span>
          <span className="text-lg font-semibold tabular-nums text-neutral-950">{p.score.toFixed(3)}</span>
        </div>
        <div className="mt-2">
          <ScoreBar score={p.score} threshold={threshold} max={max} />
        </div>
        <p className="mt-1 text-xs leading-5 text-neutral-500">Average probability the model gave each wallet as the other&apos;s next partner, over their trades after the cutoff. Black tick = threshold {threshold.toFixed(3)}.</p>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <Stat
          label="Co-trades before cutoff"
          value={<span className="tabular-nums">{p.trainEvents}</span>}
          hint="What the model could see."
        />
        <Stat
          label="Co-trades after cutoff"
          value={<span className="tabular-nums">{p.testEvents}</span>}
          hint={`Counts as co-traded at ${TRUTH_MIN}+.`}
        />
      </div>

      {style && <p className="text-xs leading-5 text-neutral-500">{style.help}</p>}
    </div>
  );
}

function Summary({ market, counts }: { market: RawMarket; counts: Record<Outcome, number> }) {
  return (
    <div className="space-y-5">
      <div>
        <h3 className="text-base font-semibold text-neutral-950">How to read this</h3>
        <p className="mt-2 text-sm leading-relaxed text-neutral-600">
          The model was trained on trades from {fmtDate(market.train_window[0])} to{" "}
          {fmtDate(market.train_window[1])}. Then, every time a wallet traded between{" "}
          {fmtDate(market.test_window[0])} and {fmtDate(market.test_window[1])}, it guessed
          which other wallet it was trading with, before seeing the answer. Pairs the
          model kept picking are its predicted links, checked against which pairs really
          co-traded.
        </p>
      </div>

      <ul className="space-y-3">
        {(Object.keys(OUTCOME_STYLE) as (keyof typeof OUTCOME_STYLE)[]).map((k) => (
          <li key={k} className="flex items-start gap-3 text-sm">
            <svg width="30" height="10" aria-hidden className="mt-1.5 shrink-0">
              <line x1="0" y1="5" x2="30" y2="5" stroke={OUTCOME_STYLE[k].color} strokeWidth="2.5" strokeDasharray={OUTCOME_STYLE[k].dash} />
            </svg>
            <span>
              <span className="font-medium text-neutral-950">
                {OUTCOME_STYLE[k].label} · <span className="tabular-nums">{counts[k]}</span>
              </span>
              <span className="block text-neutral-500">{OUTCOME_STYLE[k].help}</span>
            </span>
          </li>
        ))}
      </ul>

      <p className="text-xs leading-5 text-neutral-500">
        Click a wallet to see the model&apos;s ranked guesses for it, or a line to inspect
        one prediction.
      </p>
    </div>
  );
}

/* --------------------------------------------------
   Empty state: predictions not generated yet
-------------------------------------------------- */

function NotGenerated() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Predictions"
        description="The trained model's predicted wallet links, drawn as a graph."
      />
      <Card>
        <CardBody className="space-y-3 text-sm text-neutral-700">
          <p className="font-semibold text-neutral-950">No model predictions yet</p>
          <p>Run the model once from the backend folder, in the environment that has PyTorch:</p>
          <pre className="overflow-x-auto rounded-lg bg-neutral-950 px-4 py-3 font-mono text-xs text-neutral-100">
            cd ~/Downloads/polymap-updated/backend{"\n"}python predict_mock.py
          </pre>
          <p className="text-neutral-500">
            It writes <code className="font-mono">frontend/data/model_predictions.json</code>; reload this page afterwards.
          </p>
        </CardBody>
      </Card>
    </div>
  );
}

/* --------------------------------------------------
   Page
-------------------------------------------------- */

type Filter = "all" | "hit" | "false_alarm" | "missed";

function PredictionsGraph({
  market,
  onMarketChange,
}: {
  market: RawMarket;
  onMarketChange: (marketId: string) => void;
}) {
  const predictions = useMemo(() => toPredictions(market), [market]);
  const suggested = useMemo(() => defaultThreshold(predictions), [predictions]);

  // Fixed threshold: predict as many links as were active before the cutoff.
  const threshold = suggested;
  const sliderMax = useMemo(
    () => Math.max(0.05, Math.ceil(Math.max(...predictions.map((p) => p.score)) * 100) / 100),
    [predictions]
  );
  const [filter, setFilter] = useState<Filter>("all");
  const [selectedWallet, setSelectedWallet] = useState<number | null>(null);
  const [selectedPair, setSelectedPair] = useState<string | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [zoom, setZoom] = useState(1);

  const layout = useMemo(
    () =>
      computeLayout(market.wallets.length, predictions),
    [market, predictions]
  );

  const withOutcome = useMemo(
    () => predictions.map((p) => ({ p, outcome: outcomeOf(p, threshold) })),
    [predictions, threshold]
  );

  const counts = useMemo(() => {
    const c: Record<Outcome, number> = { hit: 0, false_alarm: 0, missed: 0, true_negative: 0 };
    for (const { outcome } of withOutcome) c[outcome]++;
    return c;
  }, [withOutcome]);

  const predicted = counts.hit + counts.false_alarm;
  const precision = predicted ? counts.hit / predicted : 0;
  const recall = counts.hit + counts.missed ? counts.hit / (counts.hit + counts.missed) : 0;

  const drawn = useMemo(() => {
    const rank: Record<Outcome, number> = { true_negative: -1, missed: 0, false_alarm: 1, hit: 2 };
    return withOutcome
      .filter(({ outcome }) => outcome !== "true_negative")
      .filter(({ outcome }) => filter === "all" || outcome === filter)
      .sort((x, y) => rank[x.outcome] - rank[y.outcome]);
  }, [withOutcome, filter]);

  const pairSel = selectedPair ? predictions.find((p) => p.key === selectedPair) : undefined;
  const focus = hover ?? selectedWallet;

  const bright = useMemo(() => {
    if (pairSel) return new Set([pairSel.a, pairSel.b]);
    if (focus === null) return null;
    const s = new Set([focus]);
    for (const { p } of drawn) {
      if (p.a === focus) s.add(p.b);
      if (p.b === focus) s.add(p.a);
    }
    return s;
  }, [pairSel, focus, drawn]);

  // Node size: co-trading partners seen before the cutoff (what the model knew).
  const history = useMemo(() => {
    const d = new Array(market.wallets.length).fill(0);
    for (const p of predictions) {
      if (p.trainEvents >= TRUTH_MIN) {
        d[p.a]++;
        d[p.b]++;
      }
    }
    return d as number[];
  }, [market, predictions]);

  const labeled = useMemo(() => {
    const top = market.wallets
      .map((_, i) => i)
      .sort((x, y) => history[y] - history[x])
      .slice(0, 6);
    const s = new Set(top);
    bright?.forEach((i) => s.add(i));
    return s;
  }, [market, history, bright]);

  const clearSelection = () => {
    setSelectedPair(null);
    setSelectedWallet(null);
  };

  const m = market.metrics;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Predictions"
        description="PolymapTGN's predicted wallet links at the train/test cutoff, drawn as a graph and checked against what actually happened afterwards."
        action={
          <Select
            value={market.market_id}
            onChange={(e) => onMarketChange(e.target.value)}
            aria-label="Market"
            className="min-w-64"
          >
            {DATA.markets.map((item) => (
              <option key={item.market_id} value={item.market_id}>
                {item.market_name}
              </option>
            ))}
          </Select>
        }
      />

      {/* Model metrics */}
      <Card>
        <CardBody className="grid grid-cols-2 gap-5 sm:grid-cols-3 lg:grid-cols-6">
          <Stat
            label="Test MRR"
            value={<span className="text-xl tabular-nums">{m.mrr.toFixed(3)}</span>}
            hint={`Recency baseline ${m.recency_mrr.toFixed(3)}`}
          />
          <Stat
            label="Test Hits@10"
            value={<span className="text-xl tabular-nums">{(m["hits@10"] * 100).toFixed(0)}%</span>}
            hint={`vs ${m.negatives_per_event} hard negatives, ${m.test_events} test events`}
          />
          {m.top1_all_wallets !== undefined ? (
            <Stat
              label="Top-1 guess right"
              value={<span className="text-xl tabular-nums">{(m.top1_all_wallets * 100).toFixed(0)}%</span>}
              hint={`Picking from all ${(market.wallets_in_model ?? market.wallets.length) - 1} other wallets (chance ${(100 / ((market.wallets_in_model ?? market.wallets.length) - 1)).toFixed(1)}%)`}
            />
          ) : (
            <Stat
              label="Test ROC AUC"
              value={<span className="text-xl tabular-nums">{m.roc_auc.toFixed(3)}</span>}
            />
          )}
          <Stat
            label="Predicted links"
            value={<span className="text-xl tabular-nums">{predicted}</span>}
            hint={`of ${predictions.length} pairs scored`}
          />
          <Stat
            label="Precision"
            value={<span className="text-xl tabular-nums">{(precision * 100).toFixed(0)}%</span>}
            hint="Predicted links that happened"
          />
          <Stat
            label="Recall"
            value={<span className="text-xl tabular-nums">{(recall * 100).toFixed(0)}%</span>}
            hint="Real co-trading pairs found"
          />
        </CardBody>
      </Card>

      {/* Controls */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <Segmented<Filter>
          value={filter}
          onChange={setFilter}
          options={[
            { value: "all", label: "All" },
            { value: "hit", label: "Correct" },
            { value: "false_alarm", label: "False alarms" },
            { value: "missed", label: "Missed" },
          ]}
        />
      </div>

      {/* Graph and side panel */}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card className="overflow-hidden">
          <div className="relative">
            <svg viewBox={`0 0 ${W} ${H}`} className="block h-auto w-full" role="img" aria-label="Predicted wallet link graph">
              <rect width={W} height={H} fill="#FFFFFF" onClick={clearSelection} />

              <g transform={`translate(${W / 2} ${H / 2}) scale(${zoom}) translate(${-W / 2} ${-H / 2})`}>
                {drawn.map(({ p, outcome }) => {
                  const a = layout[p.a];
                  const b = layout[p.b];
                  if (!a || !b || outcome === "true_negative") return null;
                  const style = OUTCOME_STYLE[outcome];
                  const isSel = p.key === selectedPair;
                  const lit = !bright || isSel || (!pairSel && bright.has(p.a) && bright.has(p.b));
                  const width = 1 + (p.score / sliderMax) * 4;

                  return (
                    <g key={p.key} className="cursor-pointer" onClick={() => setSelectedPair(p.key)}>
                      <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke="transparent" strokeWidth={14} />
                      {isSel && (
                        <line x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={style.color} strokeOpacity={0.25} strokeWidth={width + 7} strokeLinecap="round" />
                      )}
                      <line
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke={style.color}
                        strokeWidth={width}
                        strokeDasharray={style.dash}
                        strokeLinecap="round"
                        opacity={lit ? 0.9 : 0.1}
                      />
                    </g>
                  );
                })}

                {market.wallets.map((wallet, i) => {
                  const pos = layout[i];
                  if (!pos) return null;
                  const r = 8 + Math.min(history[i], 6) * 1.6;
                  const sel = i === selectedWallet;
                  const dim = bright ? !bright.has(i) : false;

                  return (
                    <g
                      key={wallet.wallet_id}
                      className="cursor-pointer"
                      opacity={dim ? 0.2 : 1}
                      onClick={() => {
                        setSelectedPair(null);
                        setSelectedWallet(i);
                      }}
                      onMouseEnter={() => setHover(i)}
                      onMouseLeave={() => setHover(null)}
                    >
                      <circle cx={pos.x} cy={pos.y} r={r + (sel ? 4 : 0)} fill={WALLET_COLOURS[wallet.wallet_type]} fillOpacity={sel ? 0.18 : 0} />
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={r}
                        fill={WALLET_COLOURS[wallet.wallet_type]}
                        stroke={sel ? "#111827" : "#FFFFFF"}
                        strokeWidth={sel ? 3 : 2}
                      />
                      {labeled.has(i) && (
                        <text
                          x={pos.x}
                          y={pos.y + r + 15}
                          textAnchor="middle"
                          fontSize="12"
                          fontWeight={600}
                          fill="#171717"
                          stroke="#FFFFFF"
                          strokeWidth={3}
                          paintOrder="stroke"
                        >
                          {addr(wallet.address, 4)}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>

            <div className="absolute right-3 top-3 flex flex-col gap-1 rounded-xl border border-neutral-200 bg-white p-1 shadow-sm">
              <Button size="icon" variant="ghost" aria-label="Zoom in" onClick={() => setZoom((z) => Math.min(2, +(z + 0.2).toFixed(1)))}>
                <Plus className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="ghost" aria-label="Zoom out" onClick={() => setZoom((z) => Math.max(0.6, +(z - 0.2).toFixed(1)))}>
                <Minus className="h-4 w-4" />
              </Button>
              <Button size="icon" variant="ghost" aria-label="Reset zoom" onClick={() => setZoom(1)}>
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>
          </div>

          <CardBody className="border-t border-neutral-100">
            <div className="grid gap-6 text-sm sm:grid-cols-2">
              <div>
                <p className="mb-3 font-semibold text-neutral-900">Lines · Prediction outcome</p>
                <ul className="space-y-2">
                  {(Object.keys(OUTCOME_STYLE) as (keyof typeof OUTCOME_STYLE)[]).map((k) => (
                    <li key={k} className="flex items-center gap-2 text-neutral-600">
                      <svg width="30" height="8" aria-hidden>
                        <line x1="0" y1="4" x2="30" y2="4" stroke={OUTCOME_STYLE[k].color} strokeWidth="2.5" strokeDasharray={OUTCOME_STYLE[k].dash} />
                      </svg>
                      {OUTCOME_STYLE[k].label}
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs leading-5 text-neutral-500">Thicker lines mean a higher prediction score.</p>
              </div>
              <div>
                <p className="mb-3 font-semibold text-neutral-900">Wallets · Wallet type</p>
                <ul className="space-y-2">
                  {Object.entries(WALLET_COLOURS).map(([type, colour]) => (
                    <li key={type} className="flex items-center gap-2 text-neutral-600">
                      <span className="h-3 w-3 rounded-full" style={{ background: colour }} />
                      {type}
                    </li>
                  ))}
                </ul>
                <p className="mt-3 text-xs leading-5 text-neutral-500">Larger circles had more co-trading partners before the cutoff.</p>
              </div>
            </div>
          </CardBody>
        </Card>

        <Card className="xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)] xl:self-start xl:overflow-y-auto">
          <CardBody>
            {pairSel ? (
              <PairDetail market={market} p={pairSel} threshold={threshold} max={sliderMax} />
            ) : selectedWallet !== null ? (
              <WalletRanking
                market={market}
                walletIdx={selectedWallet}
                predictions={predictions}
                threshold={threshold}
                max={sliderMax}
                onPick={setSelectedPair}
              />
            ) : (
              <Summary market={market} counts={counts} />
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}

export function PredictionsView() {
  const [marketId, setMarketId] = useState(DATA.markets?.[0]?.market_id ?? "");
  if (!DATA.markets?.length) return <NotGenerated />;
  const market = DATA.markets.find((m) => m.market_id === marketId) ?? DATA.markets[0];
  // Keyed by market so selection and zoom reset when the market changes.
  return <PredictionsGraph key={market.market_id} market={market} onMarketChange={setMarketId} />;
}
