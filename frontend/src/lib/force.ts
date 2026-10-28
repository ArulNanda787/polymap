import {
    forceCenter,
    forceCollide,
    forceLink,
    forceManyBody,
    forceSimulation,
    forceX,
    forceY,
    type Simulation,
    type SimulationLinkDatum,
    type SimulationNodeDatum,
} from "d3-force";

import type { Category, MapLink, MapNode } from "./types";

const CATEGORY_COLOR_HEX: Record<Category, string> = {
    persistent_tight: "#ef4444",
    persistent_loose: "#f59e0b",
    one_off_tight: "#3b82f6",
    one_off_loose: "#52525b",
};

export function buyRatioColor(r: number): string {
    if (!Number.isFinite(r)) return "#71717a";
    // Blue (0, all sells) -> zinc (0.5) -> red (1, all buys).
    const t = Math.max(0, Math.min(1, r));
    if (t < 0.5) {
        const u = t / 0.5;
        return lerpHex("#3b82f6", "#71717a", u);
    }
    const u = (t - 0.5) / 0.5;
    return lerpHex("#71717a", "#ef4444", u);
}

export function categoryColor(c: Category): string {
    return CATEGORY_COLOR_HEX[c] ?? "#52525b";
}

export function radiusFor(logTradeCount: number): number {
    // log_trade_count is log1p(trade_count); typical range 0..8.
    const t = Math.max(0, Math.min(1, logTradeCount / 8));
    return 4 + t * 20;
}

function lerpHex(a: string, b: string, t: number): string {
    const pa = hexToRgb(a);
    const pb = hexToRgb(b);
    const r = Math.round(pa[0] + (pb[0] - pa[0]) * t);
    const g = Math.round(pa[1] + (pb[1] - pa[1]) * t);
    const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t);
    return `#${[r, g, bl].map((x) => x.toString(16).padStart(2, "0")).join("")}`;
}

function hexToRgb(hex: string): [number, number, number] {
    const h = hex.replace("#", "");
    return [
        parseInt(h.slice(0, 2), 16),
        parseInt(h.slice(2, 4), 16),
        parseInt(h.slice(4, 6), 16),
    ];
}

export interface LayoutInput {
    wallets: import("./types").WalletFeature[];
    links: MapLink[];
    width: number;
    height: number;
}

export interface LayoutOutput {
    nodes: MapNode[];
    links: MapLink[];
}

export function computeLayout(input: LayoutInput): LayoutOutput {
    const { wallets, links, width, height } = input;

    // Only wallets that appear in at least one screened pair are drawn.
    const referenced = new Set<string>();
    for (const l of links) {
        referenced.add(typeof l.source === "string" ? l.source : (l.source as MapNode).id);
        referenced.add(typeof l.target === "string" ? l.target : (l.target as MapNode).id);
    }

    const nodes: MapNode[] = wallets
        .filter((w) => referenced.has(w.wallet))
        .map((w) => ({
            id: w.wallet,
            wallet: w.wallet,
            radius: radiusFor(w.log_trade_count),
            color: buyRatioColor(w.buy_ratio),
            trade_count: w.trade_count,
            buy_ratio: w.buy_ratio,
            avg_trade_size: w.avg_trade_size,
            active_duration_hours: w.active_duration_hours,
            inCategory: false,
        }));

    const nodeIds = new Set(nodes.map((n) => n.id));
    const simLinks: (MapLink & SimulationLinkDatum<MapNode>)[] = links
        .filter((l) => {
            const s = typeof l.source === "string" ? l.source : (l.source as MapNode).id;
            const t = typeof l.target === "string" ? l.target : (l.target as MapNode).id;
            return nodeIds.has(s) && nodeIds.has(t);
        })
        .map((l) => ({ ...l }));

    const sim: Simulation<MapNode, SimulationLinkDatum<MapNode>> = forceSimulation(nodes)
        .force(
            "link",
            forceLink<MapNode, SimulationLinkDatum<MapNode>>(simLinks as never)
                .id((d: MapNode) => d.id)
                .distance(60)
                .strength(0.3),
        )
        .force("charge", forceManyBody<MapNode>().strength(-120))
        .force("center", forceCenter(width / 2, height / 2))
        .force("collide", forceCollide<MapNode>().radius((d) => d.radius + 4))
        .force("x", forceX<MapNode>(width / 2).strength(0.03))
        .force("y", forceY<MapNode>(height / 2).strength(0.03))
        .stop();

    // Freeze after a bounded number of ticks so the snapshot is deterministic
    // and 400+ bubbles do not churn forever on the main thread.
    const ticks = Math.min(300, Math.ceil(Math.log(sim.alphaMin()) / Math.log(1 - sim.alphaDecay())));
    for (let i = 0; i < ticks; i++) sim.tick();

    return { nodes, links: simLinks as MapLink[] };
}