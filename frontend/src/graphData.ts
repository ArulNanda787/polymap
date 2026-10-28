// Procedural wallet network, laid out on the surface of a sphere.
// Everything is deterministic (seeded) so the globe looks the same on every load.

export type Kind = "both" | "persistent" | "tight" | "temporal";

export interface WalletNode {
  id: string;
  /** visual radius in reference pixels (sphere radius = 340) */
  r: number;
  trades: number;
  volume: string;
  /** network influence 0-99 */
  score: number;
  cluster: number;
  /** unit-sphere position */
  x: number;
  y: number;
  z: number;
}

export interface Edge {
  /** source wallet (arrow starts here) */
  a: number;
  /** target wallet (arrow points here) */
  b: number;
  interactions: number;
  /** model confidence 0-1 */
  score: number;
  kind: Kind;
  /** long-range link between clusters, drawn as a lifted arc */
  arc: boolean;
  gap: string;
  shared: number;
  seed: number;
}

export const KIND_COLOR: Record<Kind, string> = {
  both: "#f58a3d",
  persistent: "#b45cf5",
  tight: "#4fc0d4",
  temporal: "#dccb3c",
};

export const KIND_LABEL: Record<Kind, string> = {
  both: "Persistent + tight",
  persistent: "Persistent relationship",
  tight: "Tight synchronization",
  temporal: "Temporal link",
};

const KIND_RANK: Record<Kind, number> = { both: 3, persistent: 2, tight: 1, temporal: 0 };

export interface Filters {
  type: "all" | "persistent" | "tight";
  minInteractions: number;
  /** 0-100 */
  minScore: number;
}

export const defaultFilters: Filters = { type: "all", minInteractions: 3, minScore: 65 };

export const edgePasses = (e: Edge, f: Filters) =>
  e.interactions >= f.minInteractions &&
  e.score * 100 >= f.minScore &&
  (f.type === "all" || e.kind === "both" || e.kind === f.type);

export const bestKind = (a: Kind | null, b: Kind): Kind => (a === null || KIND_RANK[b] > KIND_RANK[a] ? b : a);
