"use client";

import type { ReactNode } from "react";

import { Card, PageHeader } from "./ui";

/* --------------------------------------------------
   Small SVG building blocks
-------------------------------------------------- */

function Box({
  x,
  y,
  w,
  h = 60,
  title,
  sub,
  variant = "default",
}: {
  x: number;
  y: number;
  w: number;
  h?: number;
  title: string;
  sub?: string;
  variant?: "default" | "norm" | "optional";
}) {
  const cx = x + w / 2;

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={10}
        strokeWidth={variant === "norm" ? 1.8 : 1.4}
        strokeDasharray={variant === "optional" ? "6 4" : undefined}
        className={
          variant === "norm"
            ? "fill-teal-50 stroke-teal-600"
            : "fill-white stroke-neutral-300"
        }
      />

      <text
        x={cx}
        y={sub ? y + h / 2 - 4 : y + h / 2 + 5}
        textAnchor="middle"
        className={
          variant === "norm"
            ? "fill-teal-800 text-[14px] font-semibold"
            : "fill-neutral-950 text-[14px] font-semibold"
        }
      >
        {title}
      </text>

      {sub && (
        <text
          x={cx}
          y={y + h / 2 + 15}
          textAnchor="middle"
          className="fill-neutral-500 font-mono text-[11.5px]"
        >
          {sub}
        </text>
      )}
    </g>
  );
}

function Arrow({ d, accent = false }: { d: string; accent?: boolean }) {
  return (
    <path
      d={d}
      fill="none"
      strokeWidth={accent ? 1.8 : 1.5}
      markerEnd={accent ? "url(#arch-arrow-accent)" : "url(#arch-arrow)"}
      className={accent ? "stroke-teal-600" : "stroke-neutral-500"}
    />
  );
}

function Label({
  x,
  y,
  children,
  anchor = "start",
  accent = false,
}: {
  x: number;
  y: number;
  children: ReactNode;
  anchor?: "start" | "middle" | "end";
  accent?: boolean;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      className={
        accent
          ? "fill-teal-700 font-mono text-[11.5px]"
          : "fill-neutral-500 font-mono text-[11.5px]"
      }
    >
      {children}
    </text>
  );
}

function Lane({ x, y, children }: { x: number; y: number; children: string }) {
  return (
    <text
      x={x}
      y={y}
      className="fill-neutral-500 text-[11.5px] font-semibold uppercase tracking-[0.08em]"
    >
      {children}
    </text>
  );
}

/* --------------------------------------------------
   Flowchart
-------------------------------------------------- */

function ArchitectureDiagram() {
  return (
    <svg
      viewBox="0 0 900 580"
      role="img"
      aria-label="PolymapTGN flowchart: wallet features are encoded and L2-normalised into a shared wallet memory. Each co-trade updates the two wallets' memories through a message MLP, a GRU cell and L2 normalisation. A pair is scored with cosine similarity divided by temperature plus an optional pair-history MLP."
      className="block h-auto w-full min-w-[720px]"
    >
      <defs>
        <marker
          id="arch-arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="8"
          markerHeight="8"
          orient="auto-start-reverse"
        >
          <path d="M0,0 L10,5 L0,10 z" className="fill-neutral-500" />
        </marker>

        <marker
          id="arch-arrow-accent"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="8"
          markerHeight="8"
          orient="auto-start-reverse"
        >
          <path d="M0,0 L10,5 L0,10 z" className="fill-teal-600" />
        </marker>
      </defs>

      {/* 1. Encode */}
      <Lane x={20} y={26}>1 · Encode</Lane>

      <Box x={20} y={40} w={260} title="Wallet features x" sub="trades, size, buy ratio…" />
      <Box x={320} y={40} w={260} title="Linear encoder" sub="W·x → 32-d" />
      <Box x={620} y={40} w={260} title="L2 normalise" sub="h⁰ = W·x / ‖W·x‖" variant="norm" />

      <Arrow d="M280 70 H316" />
      <Arrow d="M580 70 H616" />
      <Arrow d="M750 100 V156" />
      <Label x={762} y={134}>h⁰</Label>

      {/* Shared memory */}
      <rect
        x={20}
        y={160}
        width={860}
        height={60}
        rx={12}
        strokeWidth={1.6}
        className="fill-sky-50 stroke-sky-600"
      />
      <text x={40} y={188} className="fill-neutral-950 text-[15px] font-semibold">
        Wallet memory
      </text>
      <text x={40} y={208} className="fill-neutral-600 text-[12.5px]">
        one 32-d unit vector per wallet, updated only when that wallet trades
      </text>

      {/* 2. Update */}
      <Lane x={80} y={270}>2 · Update</Lane>

      <Arrow d="M240 220 V316" />
      <Label x={252} y={262}>mem_src, mem_tgt</Label>
      <Label x={252} y={278}>+ edge features</Label>

      <Box x={80} y={320} w={320} title="Message MLP" sub="Linear → ReLU → Linear" />
      <Box x={80} y={410} w={320} title="GRUCell" sub="mem′ = GRU(message, mem)" />
      <Box x={80} y={500} w={320} title="L2 normalise" sub="mem′ / ‖mem′‖" variant="norm" />

      <Arrow d="M240 380 V406" />
      <Arrow d="M240 470 V496" />

      <Arrow d="M80 530 H44 V224" accent />
      <text
        x={34}
        y={375}
        textAnchor="middle"
        transform="rotate(-90 34 375)"
        className="fill-teal-700 font-mono text-[11.5px]"
      >
        write back
      </text>

      {/* 3. Score */}
      <Lane x={480} y={270}>3 · Score</Lane>

      <Arrow d="M580 220 V316" />
      <Label x={592} y={270}>mem_A, mem_B</Label>

      <Box x={480} y={320} w={200} title="Cosine ÷ τ" sub="cos(A, B) / 0.1" />
      <Box x={720} y={320} w={160} title="Pair history" sub="MLP 6 → 16 → 1" variant="optional" />

      <Arrow d="M580 380 V440 H661" />
      <Arrow d="M800 380 V440 H699" />

      <circle cx={680} cy={440} r={17} strokeWidth={1.5} className="fill-white stroke-neutral-500" />
      <text x={680} y={446} textAnchor="middle" className="fill-neutral-950 text-[16px] font-semibold">
        +
      </text>

      <Arrow d="M680 457 V496" />
      <Box x={560} y={500} w={240} title="Link score(A, B)" sub="higher = likely to co-trade" />
    </svg>
  );
}

/* --------------------------------------------------
   Page
-------------------------------------------------- */

function LegendSwatch({ className }: { className: string }) {
  return <span className={`inline-block h-3.5 w-6 rounded ${className}`} />;
}

export function ArchitectureView() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="PolymapTGN architecture"
        description="The temporal graph model behind Polymap's candidate pairs. Every wallet keeps a memory vector that changes each time it co-trades, and two wallets are scored by how closely their memories point the same way."
      />

      <Card className="overflow-x-auto p-4">
        <ArchitectureDiagram />
      </Card>

      <div className="flex flex-wrap gap-x-6 gap-y-2 text-sm text-neutral-600">
        <span className="inline-flex items-center gap-2">
          <LegendSwatch className="border border-neutral-300 bg-white" />
          Always on
        </span>
        <span className="inline-flex items-center gap-2">
          <LegendSwatch className="border border-dashed border-neutral-400 bg-white" />
          Optional pair-history head
        </span>
        <span className="inline-flex items-center gap-2">
          <LegendSwatch className="border border-teal-600 bg-teal-50" />
          L2 normalisation
        </span>
        <span className="inline-flex items-center gap-2">
          <LegendSwatch className="border border-sky-600 bg-sky-50" />
          Shared wallet memory
        </span>
      </div>

      <Card className="max-w-3xl p-5">
        <h2 className="text-sm font-semibold text-teal-800">
          Why the memory is L2-normalised
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-neutral-600">
          With a plain dot-product score, the model raised every score by
          making vectors longer, and the embeddings collapsed: one principal
          component carried <span className="font-semibold tabular-nums text-neutral-950">96.7%</span> of
          the variance. Keeping every memory at length 1 and scoring with
          cosine ÷ τ means only direction counts. That dropped it to{" "}
          <span className="font-semibold tabular-nums text-neutral-950">41.1%</span>.
        </p>
      </Card>
    </div>
  );
}
