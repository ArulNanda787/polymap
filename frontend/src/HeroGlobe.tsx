import { useMemo } from "react";
import { KIND_COLOR, type Kind } from "./graphData";

// Static (non-interactive) snapshot of the explorer globe, drawn as SVG so the
// landing page matches the real thing without running the canvas engine.

const W = 1000;
const H = 410;
const CX = W / 2;
const CY = H / 2 + 2;
const UNIT = 186; // pixels per sphere radius
const CAMERA = 3.4; // camera distance in sphere radii, same as GlobeGraph
const GREEN = "#3ddc97";
const TAU = Math.PI * 2;
const YAW = 0.52;
const PITCH = 0.26;
const COUNT = 54;

const mulberry32 = (seed: number) => () => {
  seed |= 0;
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};

const project = (x: number, y: number, z: number) => {
  const cy = Math.cos(YAW), sy = Math.sin(YAW);
  const cp = Math.cos(PITCH), sp = Math.sin(PITCH);
  const x1 = x * cy + z * sy;
  const z1 = -x * sy + z * cy;
  const y2 = y * cp - z1 * sp;
  const z2 = y * sp + z1 * cp;
  const s = CAMERA / (CAMERA - z2);
  return { x: CX + x1 * s * UNIT, y: CY - y2 * s * UNIT, z: z2, s };
};

const fmt = (n: number) => n.toFixed(1);

type PNode = { x: number; y: number; z: number; sx: number; sy: number; sz: number; s: number; r: number; kind: Kind | null; id: string };
type PEdge = { a: number; b: number; arc: boolean; kind: Kind };

const KINDS: Kind[] = ["persistent", "tight", "temporal", "both"];

const build = () => {
  const rand = mulberry32(7);

  // Fibonacci sphere with a little jitter so it reads as organic, not a lattice.
  const raw: { x: number; y: number; z: number }[] = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < COUNT; i += 1) {
    const yy = 1 - ((i + 0.5) / COUNT) * 2;
    const rad = Math.sqrt(1 - yy * yy);
    const th = golden * i + rand() * 0.5;
    let x = Math.cos(th) * rad + (rand() - 0.5) * 0.12;
    let y = yy + (rand() - 0.5) * 0.12;
    let z = Math.sin(th) * rad + (rand() - 0.5) * 0.12;
    const l = Math.hypot(x, y, z);
    x /= l; y /= l; z /= l;
    raw.push({ x, y, z });
  }

  // The hub is the front-facing wallet closest to the middle of the view.
  let hub = 0;
  let best = -Infinity;
  raw.forEach((p, i) => {
    const q = project(p.x, p.y, p.z);
    const score = q.z - Math.hypot(q.x - CX, q.y - CY) / 260;
    if (score > best) { best = score; hub = i; }
  });

  const nodes: PNode[] = raw.map((p, i) => {
    const q = project(p.x, p.y, p.z);
    const big = rand() < 0.12;
    const base = big ? 9 + rand() * 4 : 3.4 + rand() * 4.2;
    const r = (i === hub ? 17 : base) * q.s;
    const kind = rand() < 0.42 ? KINDS[Math.floor(rand() * KINDS.length)] : null;
    const hex = Math.floor(rand() * 0xfffff).toString(16).padStart(5, "0");
    const id = `0x${hex.slice(0, 2).toUpperCase()}${hex.slice(2, 3).toUpperCase()}…${hex.slice(3).toUpperCase()}`;
    return { ...p, sx: q.x, sy: q.y, sz: q.z, s: q.s, r, kind, id };
  });
  nodes[hub].kind = null;

  const dist = (i: number, j: number) => Math.hypot(raw[i].x - raw[j].x, raw[i].y - raw[j].y, raw[i].z - raw[j].z);
  const edges: PEdge[] = [];
  const seen = new Set<string>();
  const add = (a: number, b: number, arc = false) => {
    const key = a < b ? `${a}-${b}` : `${b}-${a}`;
    if (a === b || seen.has(key)) return;
    seen.add(key);
    edges.push({ a, b, arc, kind: KINDS[Math.floor(rand() * KINDS.length)] });
  };

  // hub neighbours first, then each wallet's nearest couple of neighbours
  const byHub = raw.map((_, i) => i).filter((i) => i !== hub).sort((p, q) => dist(hub, p) - dist(hub, q));
  const hubLinks = byHub.slice(0, 5);
  hubLinks.forEach((n, k) => (k % 2 === 0 ? add(hub, n) : add(n, hub)));
  raw.forEach((_, i) => {
    const near = raw.map((__, j) => j).filter((j) => j !== i).sort((p, q) => dist(i, p) - dist(i, q));
    near.slice(0, 2).forEach((j) => { if (dist(i, j) < 0.62) add(i, j); });
  });
  // a handful of long lifted arcs, like the explorer's cross-cluster links
  let arcs = 0;
  for (let tries = 0; tries < 400 && arcs < 7; tries += 1) {
    const a = Math.floor(rand() * COUNT), b = Math.floor(rand() * COUNT);
    if (a === hub || b === hub) continue;
    if (dist(a, b) > 1.15 && dist(a, b) < 1.7 && nodes[a].sz > -0.2 && nodes[b].sz > -0.2) { add(a, b, true); arcs += 1; }
  }

  return { nodes, edges, hub, hubLinks: new Set(hubLinks) };
};

const graticule = () => {
  const out: string[] = [];
  const steps = 72;
  const run = (fn: (u: number) => [number, number, number]) => {
    let d = "";
    let pen = false;
    for (let k = 0; k <= steps; k += 1) {
      const [x, y, z] = fn(k / steps);
      const p = project(x, y, z);
      if (p.z > 0) { d += `${pen ? "L" : "M"}${fmt(p.x)} ${fmt(p.y)}`; pen = true; } else pen = false;
    }
    if (d) out.push(d);
  };
  for (let lat = -60; lat <= 60; lat += 30) {
    const phi = (lat * Math.PI) / 180;
    run((u) => [Math.cos(phi) * Math.cos(u * TAU), Math.sin(phi), Math.cos(phi) * Math.sin(u * TAU)]);
  }
  for (let m = 0; m < 12; m += 1) {
    const lam = (m / 12) * Math.PI;
    run((u) => {
      const a = u * TAU;
      return [Math.cos(a) * Math.cos(lam), Math.sin(a), Math.cos(a) * Math.sin(lam)];
    });
  }
  return out.join(" ");
};

export default function HeroGlobe() {
  const { nodes, edges, hub, hubLinks } = useMemo(build, []);
  const grid = useMemo(graticule, []);
  const sil = UNIT * (CAMERA / Math.sqrt(CAMERA * CAMERA - 1));

  // depth-sorted draw list, same ordering idea as the live globe
  const items = useMemo(() => {
    const list: { z: number; kind: 0 | 1; i: number }[] = [];
    edges.forEach((e, i) => list.push({ z: (nodes[e.a].sz + nodes[e.b].sz) / 2 - 0.0005, kind: 0, i }));
    nodes.forEach((n, i) => list.push({ z: n.sz, kind: 1, i }));
    return list.sort((p, q) => p.z - q.z);
  }, [nodes, edges]);

  const drawEdge = (ei: number) => {
    const e = edges[ei];
    const a = nodes[e.a], b = nodes[e.b];
    const t = (Math.min(1, Math.max(-1, (a.sz + b.sz) / 2)) + 1) / 2;
    const incident = e.a === hub || e.b === hub;

    let color = "#e4e9ff", alpha = 0.22 + 0.6 * t, width = 1.1;
    if (incident) { color = GREEN; alpha = 0.55 + 0.45 * t; width = 1.6; }
    else if (e.arc) { color = KIND_COLOR[e.kind]; alpha = 0.16 + 0.34 * t; width = 1; }

    let cxp = (a.sx + b.sx) / 2, cyp = (a.sy + b.sy) / 2;
    if (e.arc) {
      const mx = a.x + b.x, my = a.y + b.y, mz = a.z + b.z;
      const ml = Math.hypot(mx, my, mz) || 1;
      const chord = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      const lift = 1.03 + chord * 0.2;
      const c = project((mx / ml) * lift, (my / ml) * lift, (mz / ml) * lift);
      cxp = c.x; cyp = c.y;
    }
    let sdx = (e.arc ? cxp : b.sx) - a.sx, sdy = (e.arc ? cyp : b.sy) - a.sy;
    let edx = b.sx - (e.arc ? cxp : a.sx), edy = b.sy - (e.arc ? cyp : a.sy);
    const sl = Math.hypot(sdx, sdy), el = Math.hypot(edx, edy);
    if (sl < 1 || el < 1) return null;
    sdx /= sl; sdy /= sl; edx /= el; edy /= el;
    const x0 = a.sx + sdx * a.r, y0 = a.sy + sdy * a.r;
    const x1 = b.sx - edx * (b.r + 1.5), y1 = b.sy - edy * (b.r + 1.5);
    if (!e.arc && (x1 - x0) * edx + (y1 - y0) * edy < 0) return null;

    const size = 5 * Math.max(0.7, Math.min(1.25, b.s));
    const head = `${fmt(x1)},${fmt(y1)} ${fmt(x1 - edx * size - edy * size * 0.5)},${fmt(y1 - edy * size + edx * size * 0.5)} ${fmt(x1 - edx * size + edy * size * 0.5)},${fmt(y1 - edy * size - edx * size * 0.5)}`;
    const d = e.arc ? `M${fmt(x0)} ${fmt(y0)} Q${fmt(cxp)} ${fmt(cyp)} ${fmt(x1)} ${fmt(y1)}` : `M${fmt(x0)} ${fmt(y0)} L${fmt(x1)} ${fmt(y1)}`;
    const lw = width * Math.max(0.7, Math.min(1.3, (a.s + b.s) / 2));
    return (
      <g key={`e${ei}`} opacity={alpha}>
        <path d={d} fill="none" stroke={color} strokeWidth={lw} />
        <polygon points={head} fill={color} />
      </g>
    );
  };

  const drawNode = (i: number) => {
    const n = nodes[i];
    if (i === hub) return null; // drawn on top
    const t = (Math.min(1, Math.max(-1, n.sz)) + 1) / 2;
    const r = Math.max(2, n.r);
    const neighbour = hubLinks.has(i);
    const colour = neighbour ? GREEN : n.kind ? KIND_COLOR[n.kind] : null;
    if (colour) {
      const grad = neighbour ? "hero-g-green" : `hero-g-${n.kind}`;
      return <circle key={`n${i}`} cx={n.sx} cy={n.sy} r={r} fill={`url(#${grad})`} stroke={colour} strokeWidth={1.6} opacity={0.34 + 0.66 * t} />;
    }
    return <circle key={`n${i}`} cx={n.sx} cy={n.sy} r={r} fill="url(#hero-g-dim)" stroke="rgba(128,146,214,0.42)" strokeWidth={1} opacity={0.14 + 0.78 * Math.pow(t, 1.5)} />;
  };

  const hubNode = nodes[hub];
  const labelled = [...hubLinks].filter((i) => nodes[i].sz > 0).slice(0, 3);

  return (
    <svg className="network-preview" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid slice" role="img" aria-label="Preview of the wallet relationship globe">
      <defs>
        <radialGradient id="hero-halo" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0.642" stopColor="rgba(70,100,255,0.2)" />
          <stop offset="1" stopColor="rgba(70,100,255,0)" />
        </radialGradient>
        <radialGradient id="hero-body" cx="0.5" cy="0.5" r="0.5" fx="0.34" fy="0.31">
          <stop offset="0" stopColor="rgba(54,74,160,0.34)" />
          <stop offset="0.65" stopColor="rgba(20,30,78,0.2)" />
          <stop offset="1" stopColor="rgba(10,16,48,0.06)" />
        </radialGradient>
        {(["both", "persistent", "tight", "temporal"] as Kind[]).map((k) => (
          <radialGradient key={k} id={`hero-g-${k}`} cx="0.5" cy="0.5" r="0.5" fx="0.38" fy="0.34">
            <stop offset="0" stopColor={KIND_COLOR[k]} stopOpacity="0.42" />
            <stop offset="1" stopColor={KIND_COLOR[k]} stopOpacity="0.1" />
          </radialGradient>
        ))}
        <radialGradient id="hero-g-green" cx="0.5" cy="0.5" r="0.5" fx="0.38" fy="0.34">
          <stop offset="0" stopColor={GREEN} stopOpacity="0.45" />
          <stop offset="1" stopColor={GREEN} stopOpacity="0.12" />
        </radialGradient>
        <radialGradient id="hero-g-dim" cx="0.5" cy="0.5" r="0.5" fx="0.33" fy="0.3">
          <stop offset="0" stopColor="rgb(84,100,166)" stopOpacity="0.5" />
          <stop offset="0.6" stopColor="rgb(34,44,92)" stopOpacity="0.4" />
          <stop offset="1" stopColor="rgb(18,25,58)" stopOpacity="0.5" />
        </radialGradient>
        <radialGradient id="hero-g-sel" cx="0.5" cy="0.5" r="0.5" fx="0.35" fy="0.35">
          <stop offset="0" stopColor="rgb(40,110,84)" stopOpacity="0.95" />
          <stop offset="1" stopColor="rgb(14,44,36)" stopOpacity="0.95" />
        </radialGradient>
        <filter id="hero-glow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>

      <circle cx={CX} cy={CY} r={sil * 1.34} fill="url(#hero-halo)" />
      <circle cx={CX} cy={CY} r={sil} fill="url(#hero-body)" stroke="rgba(118,146,255,0.22)" strokeWidth={1.2} />
      <path d={grid} fill="none" stroke="rgba(110,140,255,0.1)" strokeWidth={1} />

      {items.map((it) => (it.kind === 0 ? drawEdge(it.i) : drawNode(it.i)))}

      {labelled.map((i) => (
        <text key={`l${i}`} x={nodes[i].sx} y={nodes[i].sy + nodes[i].r + 14} textAnchor="middle" className="preview-label">{nodes[i].id}</text>
      ))}
      <circle cx={hubNode.sx} cy={hubNode.sy} r={hubNode.r} fill="url(#hero-g-sel)" stroke="#ffffff" strokeWidth={2.4} filter="url(#hero-glow)" />
      <text x={hubNode.sx} y={hubNode.sy + hubNode.r + 16} textAnchor="middle" className="preview-label strong">0x71F…92A</text>
    </svg>
  );
}