// Turns the API's wallet + screening rows into the shape the globe renders.
import type { Category, ScreeningRow, WalletFeature } from "./types";
import type { Edge, Kind, WalletNode } from "../graphData";
import { fmtSeconds, fmtVolume, shortWallet } from "./format";

export interface Graph {
    nodes: WalletNode[];
    edges: Edge[];
    adjacency: number[][];
    defaultSelected: number;
}

const KIND_OF: Record<Category, Kind> = {
    persistent_tight: "both",
    persistent_loose: "persistent",
    one_off_tight: "tight",
    one_off_loose: "temporal",
};

// mean_margin is an unbounded model margin, not a probability. Squash it so the
// existing 0-1 "TGNN score" slider and confidence labels keep working.
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

type V3 = [number, number, number];
const norm = (v: V3): V3 => {
    const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l];
};
const cross = (a: V3, b: V3): V3 => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
];

export function buildGraph(wallets: WalletFeature[], rows: ScreeningRow[]): Graph {
    // /wallets is per (market, wallet) by default. Collapse to one row per wallet.
    const agg = new Map<string, { trades: number; size: number }>();
    for (const w of wallets) {
        const a = agg.get(w.wallet) ?? { trades: 0, size: 0 };
        a.trades += w.trade_count;
        a.size += w.total_size;
        agg.set(w.wallet, a);
    }

    // Only wallets that appear in a screened pair are drawn.
    const ids: string[] = [];
    const index = new Map<string, number>();
    const touch = (id: string) => {
        if (!index.has(id)) {
            index.set(id, ids.length);
            ids.push(id);
        }
    };
    rows.forEach((r) => {
        touch(r.wallet_a);
        touch(r.wallet_b);
    });

    const edges: Edge[] = rows.map((r, i) => ({
        a: index.get(r.wallet_a)!,
        b: index.get(r.wallet_b)!,
        interactions: r.n_events,
        score: Math.round(sigmoid(r.mean_margin) * 100) / 100,
        kind: KIND_OF[r.category] ?? "temporal",
        arc: false,
        gap: fmtSeconds(r.median_time_diff),
        shared: r.distinct_timestamps,
        seed: i * 7919,
    }));

    const adjacency: number[][] = ids.map(() => []);
    edges.forEach((e, i) => {
        adjacency[e.a].push(i);
        adjacency[e.b].push(i);
    });

    // Connected components -> one cluster each (union-find).
    const parent = ids.map((_, i) => i);
    const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
    edges.forEach((e) => {
        parent[find(e.a)] = find(e.b);
    });
    const comps = new Map<number, number[]>();
    ids.forEach((_, i) => {
        const r = find(i);
        comps.set(r, [...(comps.get(r) ?? []), i]);
    });
    // Biggest clusters first; the globe opens on the first one.
    const clusters = [...comps.values()].sort((p, q) => q.length - p.length);

    const nodes: WalletNode[] = new Array(ids.length);
    const golden = Math.PI * (3 - Math.sqrt(5));
    clusters.forEach((members, ci) => {
        // Fibonacci point on the unit sphere as the cluster centre.
        const y = 1 - (2 * (ci + 0.5)) / clusters.length;
        const rad = Math.sqrt(1 - y * y);
        const c: V3 = [Math.cos(ci * golden) * rad, y, Math.sin(ci * golden) * rad];
        const h: V3 = Math.abs(c[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
        const u = norm(cross(h, c));
        const v = cross(c, u);
        // Most-connected member sits in the middle, the rest spiral around it.
        const sorted = [...members].sort((p, q) => adjacency[q].length - adjacency[p].length);
        const ring = 0.05 + 0.012 * Math.sqrt(members.length);
        sorted.forEach((ni, k) => {
            const ang = k * golden;
            const d = k === 0 ? 0 : ring * Math.sqrt(k);
            const cs = d * Math.cos(ang);
            const sn = d * Math.sin(ang);
            const p = norm([
                c[0] + u[0] * cs + v[0] * sn,
                c[1] + u[1] * cs + v[1] * sn,
                c[2] + u[2] * cs + v[2] * sn,
            ]);
            const a = agg.get(ids[ni]);
            const logT = Math.log1p(a?.trades ?? 0);
            nodes[ni] = {
                id: shortWallet(ids[ni], 3),
                r: 4 + Math.min(1, logT / 8) * 12,
                trades: a?.trades ?? 0,
                volume: fmtVolume(a?.size ?? 0),
                score: Math.max(12, Math.min(99, Math.round(30 + adjacency[ni].length * 6 + logT * 4))),
                cluster: ci,
                x: p[0],
                y: p[1],
                z: p[2],
            };
        });
    });

    const first = clusters[0]?.slice().sort((p, q) => adjacency[q].length - adjacency[p].length)[0] ?? 0;
    return { nodes, edges, adjacency, defaultSelected: first };
}
