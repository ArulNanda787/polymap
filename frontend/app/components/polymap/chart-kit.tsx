"use client";

import * as React from "react";
import { ResponsiveContainer } from "recharts";
import { cn } from "./ui";

/**
 * Small shadcn-style chart helpers on top of recharts.
 * Same idea as shadcn's chart.tsx: colors come from a config and are exposed as CSS variables
 * (use fill="var(--color-<key>)" inside your charts).
 */

export type ChartConfig = Record<string, { label: string; color: string }>;

export function ChartContainer({
  config,
  className,
  children,
}: {
  config: ChartConfig;
  className?: string;
  children: React.ReactElement;
}) {
  const style: Record<string, string> = {};
  for (const [key, v] of Object.entries(config)) style[`--color-${key}`] = v.color;
  return (
    <div
      style={style as React.CSSProperties}
      className={cn(
        "h-72 w-full text-xs",
        "[&_.recharts-cartesian-axis-tick_text]:fill-neutral-600",
        "[&_.recharts-cartesian-grid_line]:stroke-neutral-200",
        "[&_.recharts-surface]:outline-none [&_*]:outline-none",
        className
      )}
    >
      <ResponsiveContainer width="100%" height="100%">
        {children}
      </ResponsiveContainer>
    </div>
  );
}

export function TooltipCard({
  title,
  rows,
}: {
  title?: React.ReactNode;
  rows: { label: string; value: string; color?: string }[];
}) {
  return (
    <div className="min-w-40 rounded-lg border border-neutral-200 bg-white px-3 py-2 text-sm text-neutral-950 shadow-md">
      {title && <div className="mb-1.5 font-medium">{title}</div>}
      <div className="space-y-1">
        {rows.map((r) => (
          <div key={r.label} className="flex items-center justify-between gap-6">
            <span className="flex items-center gap-2 text-neutral-600">
              {r.color && <span className="h-2 w-2 rounded-full" style={{ background: r.color }} />}
              {r.label}
            </span>
            <span className="font-medium">{r.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ChartLegend({ items }: { items: { label: string; color: string }[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-neutral-950">
      {items.map((i) => (
        <span key={i.label} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full border border-neutral-300" style={{ background: i.color }} />
          {i.label}
        </span>
      ))}
    </div>
  );
}
