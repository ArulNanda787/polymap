"use client";

import { useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  modelResults,
  type BenchmarkRun,
} from "@/lib/polymap/model-results";

import {
  ChartContainer,
  ChartLegend,
  TooltipCard,
  type ChartConfig,
} from "./chart-kit";

import {
  Card,
  CardBody,
  CardHeader,
  PageHeader,
  Segmented,
  Stat,
} from "./ui";

/* --------------------------------------------------
   Helpers
-------------------------------------------------- */

type TipProps = {
  active?: boolean;
  payload?: readonly any[];
};

const fmt3 = (v: number) => v.toFixed(3);
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
const int = (v: number) => v.toLocaleString("en-US");

const KIND_COLOR = {
  ours: "#0D9488",
  learned: "#0284C7",
  baseline: "#A3A3A3",
} as const;

const RUN_OPTIONS: { value: BenchmarkRun; label: string }[] = [
  { value: "xi_2027_63d_p50", label: "Xi 2027 market" },
  { value: "rep_iran", label: "Iran market" },
];

/* --------------------------------------------------
   Headline numbers
-------------------------------------------------- */

function Headline({ run }: { run: BenchmarkRun }) {
  const rows = modelResults.benchmark[run];
  const ours = rows.find((r) => r.id === "tuned")!;
  const bestBaseline = rows.find((r) => r.kind === "baseline")!;
  const lift = ours.mrr / bestBaseline.mrr - 1;
  const split = modelResults.runs[run];

  return (
    <Card>
      <CardBody className="grid grid-cols-2 gap-6 lg:grid-cols-5">
        <Stat
          label="MRR"
          value={
            <span className="text-2xl tabular-nums">
              {fmt3(ours.mrr)}
              {ours.mrrStd !== null && (
                <span className="ml-1 text-sm font-medium text-neutral-500">
                  ± {fmt3(ours.mrrStd)}
                </span>
              )}
            </span>
          }
          hint="Mean reciprocal rank of the true partner among 50 hard negatives, 5 seeds."
        />
        <Stat
          label="vs. best baseline"
          value={
            <span className="text-2xl tabular-nums text-teal-700">
              +{(lift * 100).toFixed(0)}%
            </span>
          }
          hint={`${bestBaseline.label} scores ${fmt3(bestBaseline.mrr)} MRR.`}
        />
        <Stat
          label="Hits@10"
          value={<span className="text-2xl tabular-nums">{pct(ours.hits10)}</span>}
          hint="True partner ranked in the top 10 of 51 candidates."
        />
        <Stat
          label="ROC AUC"
          value={<span className="text-2xl tabular-nums">{fmt3(ours.auc)}</span>}
          hint="Positive vs. hard-negative separation."
        />
        <Stat
          label="Test events scored"
          value={<span className="text-2xl tabular-nums">{int(split.testEventsScored)}</span>}
          hint={`Trained on ${int(split.trainEvents)} earlier events; chronological split.`}
        />
      </CardBody>
    </Card>
  );
}

/* --------------------------------------------------
   Benchmark chart
-------------------------------------------------- */

// Methods shown in the benchmark chart: everything down to PolymapTGN (plain)
// on the Xi 2027 market. The same set is used for both markets.
const BENCHMARK_SHOWN = ["tuned", "recency", "tgn", "edgebank_1h", "plain"] as const;

function BenchmarkChart({ run }: { run: BenchmarkRun }) {
  const data = modelResults.benchmark[run]
    .filter((r) => (BENCHMARK_SHOWN as readonly string[]).includes(r.id))
    .map((r) => ({ ...r }));
  const config: ChartConfig = {
    mrr: { label: "MRR", color: KIND_COLOR.ours },
  };

  return (
    <Card>
      <CardHeader
        title="Benchmark: ranking the true co-trading partner"
        description="Every method ranks the same held-out events against the same 50 hard negatives. Higher MRR is better."
      />
      <CardBody>
        <ChartContainer config={config} className="h-60">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ left: 0, right: 48, top: 4, bottom: 0 }}
          >
            <CartesianGrid horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 0.6]}
              ticks={[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]}
              tickFormatter={(v) => v.toFixed(1)}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={170}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              cursor={{ fill: "#F0FDFA" }}
              content={({ active, payload }: TipProps) =>
                active && payload?.length ? (
                  <TooltipCard
                    title={payload[0].payload.label}
                    rows={[
                      {
                        label: "MRR",
                        value:
                          payload[0].payload.mrrStd !== null
                            ? `${fmt3(payload[0].payload.mrr)} ± ${fmt3(payload[0].payload.mrrStd)}`
                            : fmt3(payload[0].payload.mrr),
                      },
                      { label: "Hits@10", value: pct(payload[0].payload.hits10) },
                      { label: "ROC AUC", value: fmt3(payload[0].payload.auc) },
                      { label: "Seeds", value: String(payload[0].payload.seeds) },
                    ]}
                  />
                ) : null
              }
            />
            <Bar dataKey="mrr" radius={[0, 5, 5, 0]} barSize={18}>
              {data.map((d) => (
                <Cell key={d.id} fill={KIND_COLOR[d.kind]} />
              ))}
              <LabelList
                dataKey="mrr"
                position="right"
                formatter={(v: unknown) => fmt3(Number(v))}
                className="fill-neutral-600 text-[11px] tabular-nums"
              />
            </Bar>
          </BarChart>
        </ChartContainer>
        <div className="mt-4">
          <ChartLegend
            items={[
              { label: "PolymapTGN", color: KIND_COLOR.ours },
              { label: "Reference TGN", color: KIND_COLOR.learned },
              { label: "Non-learned baseline", color: KIND_COLOR.baseline },
            ]}
          />
        </div>
      </CardBody>
    </Card>
  );
}

/* --------------------------------------------------
   Ablation
-------------------------------------------------- */

function AblationChart() {
  const data = modelResults.ablation.map((r) => ({
    ...r,
    color:
      r.id === "full"
        ? KIND_COLOR.ours
        : r.id === "recency"
          ? KIND_COLOR.baseline
          : "#5EEAD4",
  }));
  const config: ChartConfig = { mrr: { label: "MRR", color: KIND_COLOR.ours } };

  return (
    <Card>
      <CardHeader
        title="Ablation: which parts matter"
        description="Remove one component, retrain on 5 seeds, re-score. Xi 2027 market."
      />
      <CardBody>
        <ChartContainer config={config} className="h-72">
          <BarChart
            data={data}
            layout="vertical"
            margin={{ left: 0, right: 48, top: 4, bottom: 0 }}
          >
            <CartesianGrid horizontal={false} />
            <XAxis
              type="number"
              domain={[0, 0.6]}
              ticks={[0, 0.2, 0.4, 0.6]}
              tickFormatter={(v) => v.toFixed(1)}
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              type="category"
              dataKey="label"
              width={180}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              cursor={{ fill: "#F0FDFA" }}
              content={({ active, payload }: TipProps) =>
                active && payload?.length ? (
                  <TooltipCard
                    title={payload[0].payload.label}
                    rows={[
                      {
                        label: "MRR",
                        value:
                          payload[0].payload.mrrStd !== null
                            ? `${fmt3(payload[0].payload.mrr)} ± ${fmt3(payload[0].payload.mrrStd)}`
                            : fmt3(payload[0].payload.mrr),
                      },
                      { label: "Hits@10", value: pct(payload[0].payload.hits10) },
                      { label: "ROC AUC", value: fmt3(payload[0].payload.auc) },
                    ]}
                  />
                ) : null
              }
            />
            <Bar dataKey="mrr" radius={[0, 5, 5, 0]} barSize={16}>
              {data.map((d) => (
                <Cell key={d.id} fill={d.color} />
              ))}
              <LabelList
                dataKey="mrr"
                position="right"
                formatter={(v: unknown) => fmt3(Number(v))}
                className="fill-neutral-600 text-[11px] tabular-nums"
              />
            </Bar>
          </BarChart>
        </ChartContainer>
        <p className="mt-3 text-sm leading-relaxed text-neutral-600">
          Removing the decayed-recency inputs or the whole pair-history head
          costs the most. Removing node memory also hurts (0.516 → 0.418).
          Without either, MRR falls to 0.133, close to chance.
        </p>
      </CardBody>
    </Card>
  );
}

/* --------------------------------------------------
   Accuracy by pair history
-------------------------------------------------- */

const RECENCY_LABEL: Record<string, string> = {
  recent_pair: "Pair traded together recently",
  stale_pair: "Pair traded together long ago",
  never_seen_pair: "Pair never traded together",
};

function RecencyBreakdown() {
  const rows = [...modelResults.recency].sort((a, b) => b.auc - a.auc);

  return (
    <Card>
      <CardHeader
        title="Where the model is strongest"
        description="ROC AUC on the pipeline's test run, split by the pair's past history. 0.5 is chance."
      />
      <CardBody className="space-y-4">
        {rows.map((r) => (
          <div key={r.id}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span className="text-neutral-950">{RECENCY_LABEL[r.id] ?? r.id}</span>
              <span className="font-semibold tabular-nums text-neutral-950">{fmt3(r.auc)}</span>
            </div>
            <div className="relative mt-1.5 h-2.5 rounded-full bg-neutral-100">
              <div
                className="absolute inset-y-0 left-0 rounded-full bg-teal-600"
                style={{ width: `${r.auc * 100}%` }}
              />
              <div className="absolute inset-y-[-3px] left-1/2 w-px bg-neutral-400" />
            </div>
            <div className="mt-1 text-xs tabular-nums text-neutral-500">
              {int(r.n)} test events
            </div>
          </div>
        ))}
        <p className="text-xs text-neutral-500">Grey tick marks AUC 0.5.</p>
      </CardBody>
    </Card>
  );
}

/* --------------------------------------------------
   Screening output and funding validation
-------------------------------------------------- */

const SCREEN_ROWS = [
  { id: "persistent_tight", label: "Persistent · tight", color: "#7C3AED" },
  { id: "persistent_loose", label: "Persistent · loose", color: "#0D9488" },
  { id: "one_off_tight", label: "One-off · tight", color: "#EA580C" },
  { id: "one_off_loose", label: "One-off · loose", color: "#A3A3A3" },
] as const;

function ScreeningCard() {
  const cats = modelResults.screening;
  const total = Object.values(cats).reduce<number>((a, b) => a + b, 0);

  return (
    <Card>
      <CardHeader
        title="Wallet pairs screened"
        description="Pairs with 3+ scored test events, ranked by the model's margin over negatives and grouped by how persistently and tightly they co-trade."
      />
      <CardBody className="space-y-4">
        <div className="text-3xl font-semibold tabular-nums tracking-tight text-neutral-950">
          {int(total)}
          <span className="ml-2 text-sm font-medium text-neutral-500">scored pairs</span>
        </div>
        <div className="flex h-3 overflow-hidden rounded-full">
          {SCREEN_ROWS.map((r) => (
            <div
              key={r.id}
              style={{ width: `${(cats[r.id] / total) * 100}%`, background: r.color }}
            />
          ))}
        </div>
        <dl className="grid grid-cols-2 gap-3">
          {SCREEN_ROWS.map((r) => (
            <div key={r.id} className="flex items-center justify-between gap-2 text-sm">
              <dt className="flex items-center gap-2 text-neutral-600">
                <span className="h-2.5 w-2.5 rounded-full" style={{ background: r.color }} />
                {r.label}
              </dt>
              <dd className="font-semibold tabular-nums text-neutral-950">{int(cats[r.id])}</dd>
            </div>
          ))}
        </dl>
        <p className="text-xs leading-relaxed text-neutral-500">
          Tight: median gap ≤ 5 s and median price difference ≤ 0.01. Persistent:
          3+ distinct timestamps spread over more than 7 days. These are
          screening labels for human review, not evidence of coordination.
        </p>
      </CardBody>
    </Card>
  );
}

function CompareBar({
  label,
  candidate,
  control,
}: {
  label: string;
  candidate: number;
  control: number;
}) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-medium text-neutral-950">{label}</div>
      {[
        { name: "Model candidates", v: candidate, cls: "bg-teal-600" },
        { name: "Matched controls", v: control, cls: "bg-neutral-400" },
      ].map((b) => (
        <div key={b.name} className="grid grid-cols-[8.5rem_1fr_3.5rem] items-center gap-3 text-sm">
          <span className="text-neutral-600">{b.name}</span>
          <div className="h-2.5 rounded-full bg-neutral-100">
            <div className={`h-full rounded-full ${b.cls}`} style={{ width: `${b.v * 100}%` }} />
          </div>
          <span className="text-right font-semibold tabular-nums text-neutral-950">{pct(b.v)}</span>
        </div>
      ))}
    </div>
  );
}

function FundingCard() {
  const f = modelResults.funding;

  return (
    <Card>
      <CardHeader
        title="On-chain funding check"
        description={`${int(f.candidates)} model candidates vs. ${int(f.controls)} control pairs with matched co-timing.`}
      />
      <CardBody className="space-y-5">
        <CompareBar
          label="Direct transfer between the two wallets"
          candidate={f.directTransfer.candidate}
          control={f.directTransfer.control}
        />
        <CompareBar
          label="Shared funding source"
          candidate={f.sharedFunder.candidate}
          control={f.sharedFunder.control}
        />
        <p className="text-xs leading-relaxed text-neutral-500">
          Candidates are about twice as likely to have a direct transfer. Shared
          funders are near-universal for both groups (exchanges and bridges), so
          that signal does not separate them (permutation p = {f.pValueAnyLink.toFixed(2)}).
          A funding link is not proof of common ownership.
        </p>
      </CardBody>
    </Card>
  );
}

/* --------------------------------------------------
   Page
-------------------------------------------------- */

export function ModelResultsView() {
  const [run, setRun] = useState<BenchmarkRun>("xi_2027_63d_p50");
  const g = modelResults.graph;
  const s = modelResults.split;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Model results"
        description="Held-out performance of the trained PolymapTGN model on real Polymarket trades, compared with a reference TGN and non-learned baselines."
        action={<Segmented value={run} onChange={setRun} options={RUN_OPTIONS} />}
      />

      <Headline run={run} />

      <BenchmarkChart run={run} />

      <div className="grid gap-6 lg:grid-cols-2">
        <AblationChart />
        <RecencyBreakdown />
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <ScreeningCard />
        <FundingCard />
      </div>

      <p className="text-xs leading-relaxed text-neutral-500">
        Xi 2027 run: {int(g.num_nodes)} wallets, {int(g.num_edges)} co-trade edges,{" "}
        {int(s.train_events)} train / {int(s.test_events)} test events. Benchmark
        de-duplicates same-timestamp events before scoring. Ablation, accuracy
        breakdown, screening and funding check are for the Xi 2027 run only. Source:
        backend/data/artifacts, run fingerprint {modelResults.source.fingerprint}.
      </p>
    </div>
  );
}
