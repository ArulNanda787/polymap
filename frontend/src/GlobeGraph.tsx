import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { bestKind, edgePasses, KIND_COLOR, type Filters, type Kind } from "./graphData";
import type { Graph } from "./lib/adapter";

export interface GlobeHandle {
  focus: (node: number) => void;
  zoomBy: (factor: number) => void;
  reset: () => void;
}

interface Props {
  graph: Graph;
  selected: number;
  selectedEdge: number | null;
  filters: Filters;
  autoRotate: boolean;
  onSelectNode: (node: number) => void;
  onSelectEdge: (edge: number) => void;
}

const CAMERA = 3.4; // camera distance in sphere radii
const PITCH_LIMIT = 1.4;
const NODE_REF = 410; // reference radius used when laying out node sizes
const GREEN = "#3ddc97";
const TAU = Math.PI * 2;

const hexToRgb = (hex: string): [number, number, number] => {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};
const rgba = (hex: string, a: number) => {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r},${g},${b},${a})`;
};
const wrapPi = (a: number) => {
  while (a > Math.PI) a -= TAU;
  while (a < -Math.PI) a += TAU;
  return a;
};
const shortId = (id: string) => id.replace("...", "…");

const GlobeGraph = forwardRef<GlobeHandle, Props>(function GlobeGraph(props, handleRef) {
  const { nodes, edges, adjacency, defaultSelected } = props.graph;
  const focusAngles = (i: number) => {
    const { x, y, z } = nodes[i];
    const yaw = Math.atan2(-x, z);
    const h = Math.hypot(x, z);
    return { yaw, pitch: Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, Math.atan2(y, h))) };
  };
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const propsRef = useRef(props);
  propsRef.current = props;
  const [tip, setTip] = useState<{ node: number; x: number; y: number } | null>(null);

  // Which edges survive the current filters, which wallets light up, and who neighbours the selection.
  const derived = useMemo(() => {
    const pass = edges.map((e) => edgePasses(e, props.filters));
    const kind: (Kind | null)[] = nodes.map(() => null);
    edges.forEach((e, i) => {
      if (!pass[i] || e.arc) return;
      kind[e.a] = bestKind(kind[e.a], e.kind);
      kind[e.b] = bestKind(kind[e.b], e.kind);
    });
    const neighbours = new Set<number>();
    const incident = new Set<number>();
    adjacency[props.selected].forEach((ei) => {
      if (!pass[ei]) return;
      incident.add(ei);
      neighbours.add(edges[ei].a === props.selected ? edges[ei].b : edges[ei].a);
    });
    return { pass, kind, neighbours, incident };
  }, [props.filters, props.selected]);
  const derivedRef = useRef(derived);
  derivedRef.current = derived;

  const engine = useRef({
    yaw: 0,
    pitch: 0,
    vYaw: 0,
    vPitch: 0,
    zoom: 1,
    panX: 0,
    panY: 0,
    focus: null as { yaw: number; pitch: number } | null,
    hoverNode: -1,
    hoverEdge: -1,
    dragNode: -1,
    rotating: false,
    offset: nodes.map(() => [0, 0, 0] as [number, number, number]),
    w: 800,
    h: 600,
    // projected buffers
    sx: new Float32Array(nodes.length),
    sy: new Float32Array(nodes.length),
    sz: new Float32Array(nodes.length),
    ss: new Float32Array(nodes.length),
    sr: new Float32Array(nodes.length),
    init: false,
  }).current;

  if (!engine.init) {
    const a = focusAngles(defaultSelected);
    engine.yaw = a.yaw;
    engine.pitch = a.pitch;
    engine.init = true;
  }

  useImperativeHandle(handleRef, () => ({
    focus: (node: number) => {
      engine.focus = focusAngles(node);
      engine.vYaw = 0;
      engine.vPitch = 0;
    },
    zoomBy: (factor: number) => {
      const next = Math.max(0.55, Math.min(4, engine.zoom * factor));
      const k = next / engine.zoom;
      engine.panX *= k;
      engine.panY *= k;
      engine.zoom = next;
    },
    reset: () => {
      engine.zoom = 1;
      engine.panX = 0;
      engine.panY = 0;
      engine.focus = focusAngles(propsRef.current.selected);
    },
  }));

  useEffect(() => {
    const canvas = canvasRef.current!;
    const wrap = wrapRef.current!;
    const ctx = canvas.getContext("2d")!;
    let raf = 0;
    let dpr = 1;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    const resize = () => {
      const rect = wrap.getBoundingClientRect();
      dpr = Math.min(window.devicePixelRatio || 1, 2);
      engine.w = rect.width;
      engine.h = rect.height;
      canvas.width = Math.max(1, Math.round(rect.width * dpr));
      canvas.height = Math.max(1, Math.round(rect.height * dpr));
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(wrap);

    // ---- projection -------------------------------------------------------
    const ppu = () => Math.min(engine.w, engine.h) * 0.385 * engine.zoom;
    const project = () => {
      const cy = Math.cos(engine.yaw), sy = Math.sin(engine.yaw);
      const cp = Math.cos(engine.pitch), sp = Math.sin(engine.pitch);
      const unit = ppu();
      const cx0 = engine.w / 2 + engine.panX;
      const cy0 = engine.h / 2 + engine.panY;
      const dragged = engine.dragNode;
      for (let i = 0; i < nodes.length; i += 1) {
        const n = nodes[i];
        const off = engine.offset[i];
        let ox = off[0], oy = off[1], oz = off[2];
        if (dragged >= 0 && i !== dragged) {
          // neighbours of the wallet being pulled follow along a little
          const d = engine.offset[dragged];
          if (adjacency[dragged].some((ei) => edges[ei].a === i || edges[ei].b === i)) {
            ox += d[0] * 0.28; oy += d[1] * 0.28; oz += d[2] * 0.28;
          }
        }
        const x = n.x + ox, y = n.y + oy, z = n.z + oz;
        const x1 = x * cy + z * sy;
        const z1 = -x * sy + z * cy;
        const y2 = y * cp - z1 * sp;
        const z2 = y * sp + z1 * cp;
        const s = CAMERA / (CAMERA - z2);
        engine.sx[i] = cx0 + x1 * s * unit;
        engine.sy[i] = cy0 - y2 * s * unit;
        engine.sz[i] = z2;
        engine.ss[i] = s;
        engine.sr[i] = (n.r * s * unit) / NODE_REF;
      }
    };

    // world-space point (unit sphere coords) -> screen
    const projectPoint = (x: number, y: number, z: number) => {
      const cy = Math.cos(engine.yaw), sy = Math.sin(engine.yaw);
      const cp = Math.cos(engine.pitch), sp = Math.sin(engine.pitch);
      const x1 = x * cy + z * sy;
      const z1 = -x * sy + z * cy;
      const y2 = y * cp - z1 * sp;
      const z2 = y * sp + z1 * cp;
      const s = CAMERA / (CAMERA - z2);
      const unit = ppu();
      return { x: engine.w / 2 + engine.panX + x1 * s * unit, y: engine.h / 2 + engine.panY - y2 * s * unit, z: z2 };
    };

    // control point for arc edges: midpoint pushed out above the surface
    const arcControl = (ei: number) => {
      const e = edges[ei];
      const a = nodes[e.a], b = nodes[e.b];
      const mx = a.x + b.x, my = a.y + b.y, mz = a.z + b.z;
      const ml = Math.hypot(mx, my, mz) || 1;
      const chord = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      const lift = 1.03 + chord * 0.2;
      return projectPoint((mx / ml) * lift, (my / ml) * lift, (mz / ml) * lift);
    };

    // ---- drawing helpers ----------------------------------------------------
    const arrowHead = (x: number, y: number, ux: number, uy: number, size: number) => {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x - ux * size - uy * size * 0.5, y - uy * size + ux * size * 0.5);
      ctx.lineTo(x - ux * size + uy * size * 0.5, y - uy * size - ux * size * 0.5);
      ctx.closePath();
      ctx.fill();
    };

    const drawEdge = (ei: number) => {
      const e = edges[ei];
      const d = derivedRef.current;
      const ps = propsRef.current;
      const passes = d.pass[ei];
      const isSel = ps.selectedEdge === ei;
      const isHover = engine.hoverEdge === ei;
      const isIncident = d.incident.has(ei);
      const t = (Math.min(1, Math.max(-1, (engine.sz[e.a] + engine.sz[e.b]) / 2)) + 1) / 2;
      if (e.arc && !passes) return;

      let color: string;
      let alpha: number;
      let width: number;
      if (isSel || isHover) { color = "#ffffff"; alpha = 1; width = 2.2; }
      else if (isIncident) { color = GREEN; alpha = 0.55 + 0.45 * t; width = 1.6; }
      else if (e.arc) { color = KIND_COLOR[e.kind]; alpha = 0.16 + 0.34 * t; width = 1; }
      else if (passes) { color = "#e4e9ff"; alpha = 0.22 + 0.6 * t; width = 1.1; }
      else { color = "#7e8fd0"; alpha = 0.1 + 0.3 * t; width = 0.9; }

      const ax = engine.sx[e.a], ay = engine.sy[e.a], bx = engine.sx[e.b], by = engine.sy[e.b];
      const ra = engine.sr[e.a], rb = engine.sr[e.b];
      let cxp = (ax + bx) / 2, cyp = (ay + by) / 2;
      if (e.arc) { const c = arcControl(ei); cxp = c.x; cyp = c.y; }
      // direction at start / end (towards control point for arcs)
      let sdx = (e.arc ? cxp : bx) - ax, sdy = (e.arc ? cyp : by) - ay;
      let edx = bx - (e.arc ? cxp : ax), edy = by - (e.arc ? cyp : ay);
      const sl = Math.hypot(sdx, sdy), el = Math.hypot(edx, edy);
      if (sl < 1 || el < 1) return;
      sdx /= sl; sdy /= sl; edx /= el; edy /= el;
      const x0 = ax + sdx * ra, y0 = ay + sdy * ra;
      const x1 = bx - edx * (rb + 1.5), y1 = by - edy * (rb + 1.5);
      if ((x1 - x0) * edx + (y1 - y0) * edy < 0 && !e.arc) return;

      ctx.globalAlpha = alpha;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = width * Math.max(0.7, Math.min(1.3, (engine.ss[e.a] + engine.ss[e.b]) / 2));
      if (isSel) { ctx.shadowColor = "#9db4ff"; ctx.shadowBlur = 12; }
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      if (e.arc) ctx.quadraticCurveTo(cxp, cyp, x1, y1);
      else ctx.lineTo(x1, y1);
      ctx.stroke();
      const size = (isSel || isHover ? 7 : 5) * Math.max(0.7, Math.min(1.25, engine.ss[e.b]));
      arrowHead(x1, y1, edx, edy, size);
      ctx.shadowBlur = 0;
      ctx.globalAlpha = 1;
    };

    const drawNode = (i: number) => {
      const d = derivedRef.current;
      const ps = propsRef.current;
      const x = engine.sx[i], y = engine.sy[i], r = Math.max(2, engine.sr[i]);
      const t = (Math.min(1, Math.max(-1, engine.sz[i])) + 1) / 2;
      const kind = d.kind[i];
      const isNeighbour = d.neighbours.has(i);
      const isSelected = i === ps.selected;
      const colour = isNeighbour ? GREEN : kind ? KIND_COLOR[kind] : null;
      if (colour) {
        const a = 0.34 + 0.66 * t;
        ctx.globalAlpha = a;
        const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.35, r * 0.1, x, y, r);
        g.addColorStop(0, rgba(colour, r > 9 ? 0.5 : 0.28));
        g.addColorStop(1, rgba(colour, r > 9 ? 0.16 : 0.07));
        ctx.fillStyle = g;
        ctx.strokeStyle = colour;
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, TAU);
        ctx.fill();
        ctx.stroke();
      } else {
        const a = 0.14 + 0.78 * Math.pow(t, 1.5);
        ctx.globalAlpha = a;
        const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.05, x, y, r);
        g.addColorStop(0, "rgba(84,100,166,0.5)");
        g.addColorStop(0.6, "rgba(34,44,92,0.4)");
        g.addColorStop(1, "rgba(18,25,58,0.5)");
        ctx.fillStyle = g;
        ctx.strokeStyle = "rgba(128,146,214,0.42)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, TAU);
        ctx.fill();
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
      if (isSelected) return; // drawn in the overlay pass
    };

    const label = (i: number, strong = false) => {
      const text = shortId(nodes[i].id);
      ctx.font = `500 ${strong ? 11 : 10}px "DM Mono", monospace`;
      ctx.textAlign = "center";
      ctx.fillStyle = strong ? "rgba(255,255,255,0.95)" : "rgba(214,224,255,0.78)";
      ctx.fillText(text, engine.sx[i], engine.sy[i] + engine.sr[i] + 14);
    };

    const graticule = (cx: number, cy: number) => {
      ctx.save();
      ctx.lineWidth = 1;
      ctx.strokeStyle = "rgba(110,140,255,0.085)";
      const steps = 72;
      const run = (fn: (u: number) => [number, number, number]) => {
        ctx.beginPath();
        let pen = false;
        for (let k = 0; k <= steps; k += 1) {
          const [x, y, z] = fn(k / steps);
          const p = projectPoint(x, y, z);
          if (p.z > 0) { if (pen) ctx.lineTo(p.x, p.y); else ctx.moveTo(p.x, p.y); pen = true; }
          else pen = false;
        }
        ctx.stroke();
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
      ctx.restore();
      void cx; void cy;
    };

    // ---- main loop ----------------------------------------------------------
    const frame = () => {
      raf = requestAnimationFrame(frame);
      const ps = propsRef.current;

      // physics: inertia, focus easing, idle spin, spring-back for dragged wallets
      if (engine.focus) {
        const dy = wrapPi(engine.focus.yaw - engine.yaw);
        const dp = engine.focus.pitch - engine.pitch;
        engine.yaw += dy * 0.11;
        engine.pitch += dp * 0.11;
        if (Math.abs(dy) < 0.002 && Math.abs(dp) < 0.002) engine.focus = null;
      } else if (!engine.rotating && engine.dragNode < 0) {
        engine.yaw += engine.vYaw;
        engine.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, engine.pitch + engine.vPitch));
        engine.vYaw *= 0.94;
        engine.vPitch *= 0.94;
        if (ps.autoRotate && !reduceMotion && engine.hoverNode < 0 && Math.abs(engine.vYaw) < 0.002) engine.yaw += 0.0016;
      }
      for (let i = 0; i < engine.offset.length; i += 1) {
        if (i === engine.dragNode) continue;
        const o = engine.offset[i];
        if (o[0] === 0 && o[1] === 0 && o[2] === 0) continue;
        o[0] *= 0.84; o[1] *= 0.84; o[2] *= 0.84;
        if (Math.abs(o[0]) + Math.abs(o[1]) + Math.abs(o[2]) < 1e-4) { o[0] = 0; o[1] = 0; o[2] = 0; }
      }

      project();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, engine.w, engine.h);

      // globe body
      const unit = ppu();
      const cx = engine.w / 2 + engine.panX;
      const cy = engine.h / 2 + engine.panY;
      const sil = unit * (CAMERA / Math.sqrt(CAMERA * CAMERA - 1));
      const halo = ctx.createRadialGradient(cx, cy, sil * 0.86, cx, cy, sil * 1.34);
      halo.addColorStop(0, "rgba(70,100,255,0.16)");
      halo.addColorStop(1, "rgba(70,100,255,0)");
      ctx.fillStyle = halo;
      ctx.beginPath(); ctx.arc(cx, cy, sil * 1.34, 0, TAU); ctx.fill();
      const body = ctx.createRadialGradient(cx - sil * 0.32, cy - sil * 0.38, sil * 0.05, cx, cy, sil);
      body.addColorStop(0, "rgba(54,74,160,0.34)");
      body.addColorStop(0.65, "rgba(20,30,78,0.2)");
      body.addColorStop(1, "rgba(10,16,48,0.06)");
      ctx.fillStyle = body;
      ctx.beginPath(); ctx.arc(cx, cy, sil, 0, TAU); ctx.fill();
      ctx.strokeStyle = "rgba(118,146,255,0.22)";
      ctx.lineWidth = 1.2;
      ctx.stroke();
      graticule(cx, cy);

      // depth-sorted edges and wallets
      const items: { z: number; kind: 0 | 1; i: number }[] = [];
      for (let i = 0; i < edges.length; i += 1) {
        const e = edges[i];
        items.push({ z: (engine.sz[e.a] + engine.sz[e.b]) / 2 - 0.0005, kind: 0, i });
      }
      for (let i = 0; i < nodes.length; i += 1) items.push({ z: engine.sz[i], kind: 1, i });
      items.sort((p, q) => p.z - q.z);
      for (const it of items) (it.kind === 0 ? drawEdge(it.i) : drawNode(it.i));

      // overlay: selection, hover and labels
      const d = derivedRef.current;
      d.neighbours.forEach((n) => label(n));
      if (ps.selectedEdge !== null) {
        label(edges[ps.selectedEdge].a, true);
        label(edges[ps.selectedEdge].b, true);
      }
      label(ps.selected, true);
      const sel = ps.selected;
      {
        const x = engine.sx[sel], y = engine.sy[sel], r = Math.max(3, engine.sr[sel]);
        const g = ctx.createRadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r);
        g.addColorStop(0, "rgba(40,110,84,0.95)");
        g.addColorStop(1, "rgba(14,44,36,0.95)");
        ctx.shadowColor = "rgba(255,255,255,0.55)";
        ctx.shadowBlur = 14;
        ctx.fillStyle = g;
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2.4;
        ctx.beginPath(); ctx.arc(x, y, r, 0, TAU); ctx.fill(); ctx.stroke();
        ctx.shadowBlur = 0;
      }
      if (engine.hoverNode >= 0 && engine.hoverNode !== sel) {
        const i = engine.hoverNode;
        ctx.strokeStyle = "rgba(255,255,255,0.9)";
        ctx.lineWidth = 1.6;
        ctx.beginPath(); ctx.arc(engine.sx[i], engine.sy[i], engine.sr[i] + 3, 0, TAU); ctx.stroke();
        label(i, true);
      }
    };
    raf = requestAnimationFrame(frame);

    // ---- hit testing ------------------------------------------------------
    const hitNode = (mx: number, my: number) => {
      let best = -1, bestZ = -2;
      for (let i = 0; i < nodes.length; i += 1) {
        if (engine.sz[i] < -0.15) continue;
        const dx = mx - engine.sx[i], dy = my - engine.sy[i];
        const rad = Math.max(5, engine.sr[i]) + 3;
        if (dx * dx + dy * dy <= rad * rad && engine.sz[i] > bestZ) { best = i; bestZ = engine.sz[i]; }
      }
      return best;
    };
    const segDist = (px: number, py: number, ax: number, ay: number, bx: number, by: number) => {
      const dx = bx - ax, dy = by - ay;
      const l2 = dx * dx + dy * dy || 1;
      const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / l2));
      return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
    };
    const hitEdge = (mx: number, my: number) => {
      let best = -1, bestD = 6;
      for (let i = 0; i < edges.length; i += 1) {
        const e = edges[i];
        if (engine.sz[e.a] < -0.15 || engine.sz[e.b] < -0.15) continue;
        if (e.arc && !derivedRef.current.pass[i]) continue;
        let dist: number;
        if (e.arc) {
          const c = arcControl(i);
          dist = 1e9;
          let px = engine.sx[e.a], py = engine.sy[e.a];
          for (let k = 1; k <= 14; k += 1) {
            const u = k / 14;
            const qx = (1 - u) * (1 - u) * engine.sx[e.a] + 2 * (1 - u) * u * c.x + u * u * engine.sx[e.b];
            const qy = (1 - u) * (1 - u) * engine.sy[e.a] + 2 * (1 - u) * u * c.y + u * u * engine.sy[e.b];
            dist = Math.min(dist, segDist(mx, my, px, py, qx, qy));
            px = qx; py = qy;
          }
        } else {
          dist = segDist(mx, my, engine.sx[e.a], engine.sy[e.a], engine.sx[e.b], engine.sy[e.b]);
        }
        if (dist < bestD) { bestD = dist; best = i; }
      }
      return best;
    };

    // ---- pointer interaction ---------------------------------------------------
    const pointers = new Map<number, { x: number; y: number }>();
    let mode: "none" | "rotate" | "pan" | "node" = "none";
    let downNode = -1;
    let downEdge = -1;
    let moved = 0;
    let startX = 0, startY = 0;
    let lastX = 0, lastY = 0;
    let pinchDist = 0;

    const local = (ev: PointerEvent | WheelEvent) => {
      const r = canvas.getBoundingClientRect();
      return { x: ev.clientX - r.left, y: ev.clientY - r.top };
    };

    const setCursor = () => {
      canvas.style.cursor = mode === "rotate" || mode === "pan" || mode === "node" ? "grabbing" : engine.hoverNode >= 0 || engine.hoverEdge >= 0 ? "pointer" : "grab";
    };

    const onDown = (ev: PointerEvent) => {
      canvas.setPointerCapture(ev.pointerId);
      const p = local(ev);
      pointers.set(ev.pointerId, p);
      setTip(null);
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        pinchDist = Math.hypot(a.x - b.x, a.y - b.y);
        mode = "none";
        engine.rotating = false;
        engine.dragNode = -1;
        return;
      }
      startX = lastX = p.x; startY = lastY = p.y;
      moved = 0;
      engine.focus = null;
      engine.vYaw = 0; engine.vPitch = 0;
      downNode = hitNode(p.x, p.y);
      downEdge = downNode < 0 ? hitEdge(p.x, p.y) : -1;
      if (ev.shiftKey || ev.button === 2 || ev.ctrlKey || ev.metaKey) mode = "pan";
      else if (downNode >= 0) { mode = "node"; engine.dragNode = downNode; }
      else { mode = "rotate"; engine.rotating = true; }
      setCursor();
    };

    const onMove = (ev: PointerEvent) => {
      const p = local(ev);
      if (pointers.has(ev.pointerId)) pointers.set(ev.pointerId, p);

      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinchDist > 0) zoomAt((a.x + b.x) / 2, (a.y + b.y) / 2, dist / pinchDist);
        pinchDist = dist;
        return;
      }

      if (mode === "none") {
        if (ev.pointerType !== "mouse") return;
        const n = hitNode(p.x, p.y);
        const e = n < 0 ? hitEdge(p.x, p.y) : -1;
        if (n !== engine.hoverNode || e !== engine.hoverEdge) {
          engine.hoverNode = n; engine.hoverEdge = e;
          setTip(n >= 0 ? { node: n, x: p.x, y: p.y } : null);
          setCursor();
        } else if (n >= 0) setTip({ node: n, x: p.x, y: p.y });
        return;
      }

      const dx = p.x - lastX, dy = p.y - lastY;
      moved += Math.abs(dx) + Math.abs(dy);
      lastX = p.x; lastY = p.y;

      if (mode === "rotate") {
        const k = 1 / Math.max(180, ppu() * 0.95);
        engine.vYaw = dx * k * 1.6;
        engine.vPitch = dy * k * 1.6;
        engine.yaw += dx * k * 1.6;
        engine.pitch = Math.max(-PITCH_LIMIT, Math.min(PITCH_LIMIT, engine.pitch + dy * k * 1.6));
      } else if (mode === "pan") {
        engine.panX += dx; engine.panY += dy;
      } else if (mode === "node" && moved > 4) {
        // Move the wallet in the view plane, then convert back to world space.
        const i = engine.dragNode;
        const unit = ppu();
        const s = engine.ss[i];
        const vx = dx / (unit * s), vy = -dy / (unit * s);
        const cyw = Math.cos(engine.yaw), syw = Math.sin(engine.yaw);
        const cp = Math.cos(engine.pitch), sp = Math.sin(engine.pitch);
        // inverse of pitch then yaw (view-space delta, z = 0)
        const y1 = vy * cp;
        const z1 = -vy * sp;
        const wx = vx * cyw - z1 * syw;
        const wz = vx * syw + z1 * cyw;
        const o = engine.offset[i];
        o[0] += wx; o[1] += y1; o[2] += wz;
        const len = Math.hypot(o[0], o[1], o[2]);
        if (len > 0.55) { o[0] *= 0.55 / len; o[1] *= 0.55 / len; o[2] *= 0.55 / len; }
      }
    };

    const onUp = (ev: PointerEvent) => {
      pointers.delete(ev.pointerId);
      if (canvas.hasPointerCapture(ev.pointerId)) canvas.releasePointerCapture(ev.pointerId);
      if (pointers.size > 0) return;
      pinchDist = 0;
      const wasClick = moved < 5;
      if (wasClick && ev.type === "pointerup") {
        if (downNode >= 0) {
          engine.focus = focusAngles(downNode);
          propsRef.current.onSelectNode(downNode);
        } else if (downEdge >= 0) propsRef.current.onSelectEdge(downEdge);
      }
      mode = "none";
      engine.rotating = false;
      engine.dragNode = -1;
      downNode = -1; downEdge = -1;
      setCursor();
    };

    const onLeave = () => {
      if (mode !== "none") return;
      engine.hoverNode = -1; engine.hoverEdge = -1;
      setTip(null);
    };

    const zoomAt = (mx: number, my: number, factor: number) => {
      const next = Math.max(0.55, Math.min(4, engine.zoom * factor));
      const k = next / engine.zoom;
      const ox = mx - engine.w / 2, oy = my - engine.h / 2;
      engine.panX = ox - (ox - engine.panX) * k;
      engine.panY = oy - (oy - engine.panY) * k;
      engine.zoom = next;
    };
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const p = local(ev);
      zoomAt(p.x, p.y, Math.exp(-ev.deltaY * 0.0014));
    };
    const onDouble = () => {
      engine.zoom = 1; engine.panX = 0; engine.panY = 0;
      engine.focus = focusAngles(propsRef.current.selected);
    };
    const onContext = (ev: Event) => ev.preventDefault();

    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("wheel", onWheel, { passive: false });
    canvas.addEventListener("dblclick", onDouble);
    canvas.addEventListener("contextmenu", onContext);
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("wheel", onWheel);
      canvas.removeEventListener("dblclick", onDouble);
      canvas.removeEventListener("contextmenu", onContext);
    };
  }, [engine]);

  const tipNode = tip ? nodes[tip.node] : null;
  return (
    <div ref={wrapRef} className="globe-wrap">
      <canvas
        ref={canvasRef}
        className="globe-canvas"
        role="img"
        aria-label="Interactive 3D wallet network globe. Drag to rotate, drag a wallet to pull it, shift-drag to pan, scroll to zoom."
      />
      {tip && tipNode && (
        <div className="globe-tip" style={{ left: tip.x + 16, top: tip.y + 14 }}>
          <b>{tipNode.id}</b>
          <span>{tipNode.trades} trades · {tipNode.volume}</span>
          <span>{adjacency[tip.node].length} relationship{adjacency[tip.node].length === 1 ? "" : "s"}</span>
        </div>
      )}
    </div>
  );
});

export default GlobeGraph;
