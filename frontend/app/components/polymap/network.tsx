
"use client";

import { useMemo, useState } from "react";
import { Minus, Plus, RotateCcw } from "lucide-react";

import {
  addr,
  marketById,
  marketStats,
  markets,
  money,
  num,
  pairName,
  pairs,
  walletById,
} from "@/lib/polymap/data";

import type {
  CandidateRelationship,
  WalletType,
} from "@/lib/polymap/types";

import { Inspector } from "./panels";
import { usePolymap, type Level } from "./store";

import {
  Card,
  CardBody,
  PageHeader,
  Segmented,
  Select,
  Button,
  Stat,
} from "./ui";

const W = 900;
const H = 600;

const WALLET_COLOURS: Record<WalletType, string> = {
  Whale: "#7C3AED",
  "Frequent trader": "#2563EB",
  "Occasional trader": "#16A34A",
  "New wallet": "#EA580C",
};

const CATEGORY_STYLE: Record<
  string,
  { label: string; color: string; dash: string }
> = {
  persistent_tight: {
    label: "Persistent · Tight",
    color: "#059669",
    dash: "",
  },
  persistent_loose: {
    label: "Persistent · Loose",
    color: "#2563EB",
    dash: "6 3",
  },
  one_off_loose: {
    label: "One-off · Loose",
    color: "#EA580C",
    dash: "3 4",
  },
};

/* --------------------------------------------------
   Force-directed layout
-------------------------------------------------- */

function computeLayout(
  ids: string[],
  links: { a: string; b: string }[]
) {
  const n = ids.length;

  if (n === 0) return new Map();

  const idx = new Map(ids.map((id, i) => [id, i]));

  const r0 = Math.min(W, H) * 0.35;

  const pos = ids.map((_, i) => ({
    x: W / 2 + Math.cos((i / n) * Math.PI * 2) * r0,
    y: H / 2 + Math.sin((i / n) * Math.PI * 2) * r0,
  }));

  const ls = links
    .map((link) => ({
      a: idx.get(link.a),
      b: idx.get(link.b),
    }))
    .filter(
      (link): link is { a: number; b: number } =>
        link.a !== undefined && link.b !== undefined
    );

  const rest = n > 25 ? 80 : 115;
  const charge = n > 25 ? 8000 : 14000;

  for (let it = 0; it < 350; it++) {
    const cool = 1 - it / 350;

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

    for (const link of ls) {
      const vx = pos[link.b].x - pos[link.a].x;
      const vy = pos[link.b].y - pos[link.a].y;

      const d = Math.max(
        Math.sqrt(vx * vx + vy * vy),
        1
      );

      const f = (d - rest) * 0.03;

      dx[link.a] += (vx / d) * f;
      dy[link.a] += (vy / d) * f;

      dx[link.b] -= (vx / d) * f;
      dy[link.b] -= (vy / d) * f;
    }

    for (let i = 0; i < n; i++) {
      dx[i] += (W / 2 - pos[i].x) * 0.012;
      dy[i] += (H / 2 - pos[i].y) * 0.012;

      pos[i].x = Math.min(
        W - 60,
        Math.max(60, pos[i].x + dx[i] * cool)
      );

      pos[i].y = Math.min(
        H - 50,
        Math.max(40, pos[i].y + dy[i] * cool)
      );
    }
  }

  return new Map(ids.map((id, i) => [id, pos[i]]));
}

/* --------------------------------------------------
   Market summary
-------------------------------------------------- */

function MarketSummary({
  marketId,
  visible,
}: {
  marketId: string;
  visible: CandidateRelationship[];
}) {
  const { selectPair } = usePolymap();

  const market = marketById.get(marketId);

  if (!market) return null;

  const ids = Array.from(
    new Set(
      visible.flatMap((relationship) => [
        relationship.wallet_a_id,
        relationship.wallet_b_id,
      ])
    )
  );

  const walletTypes: WalletType[] = [
    "Whale",
    "Frequent trader",
    "Occasional trader",
    "New wallet",
  ];

  const top = [...visible]
    .sort(
      (a, b) =>
        b.relationship_strength - a.relationship_strength
    )
    .slice(0, 6);

  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-base font-semibold text-neutral-950">
          {market.name}
        </h3>

        <p className="mt-1 text-sm text-neutral-500">
          {market.category}
        </p>
      </div>

      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          Wallet types in this network
        </h4>

        <ul className="mt-3 space-y-2">
          {walletTypes.map((type) => {
            const count = ids.filter(
              (id) => walletById.get(id)?.wallet_type === type
            ).length;

            return (
              <li
                key={type}
                className="flex items-center justify-between text-sm"
              >
                <span className="flex items-center gap-2 text-neutral-700">
                  <span
                    className="h-2.5 w-2.5 rounded-full"
                    style={{
                      background: WALLET_COLOURS[type],
                    }}
                  />
                  {type}
                </span>

                <span className="font-semibold text-neutral-900">
                  {count}
                </span>
              </li>
            );
          })}
        </ul>
      </div>

      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          Leading relationships
        </h4>

        {top.length === 0 ? (
          <p className="mt-2 text-sm text-neutral-500">
            No relationships match these filters.
          </p>
        ) : (
          <ul className="mt-2 divide-y divide-neutral-100 overflow-hidden rounded-xl border border-neutral-200">
            {top.map((relationship) => (
              <li key={relationship.pair_id}>
                <button
                  type="button"
                  onClick={() => selectPair(relationship.pair_id)}
                  className="flex w-full items-center justify-between gap-3 px-3 py-3 text-left transition-colors hover:bg-cyan-50/60"
                >
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-semibold text-neutral-900">
                      {pairName(relationship)}
                    </span>

                    <span className="mt-1 block text-xs text-neutral-500">
                      {relationship.events} events ·{" "}
                      {money(relationship.matched_volume_usd)}
                    </span>
                  </span>

                  <span className="shrink-0 text-xs font-semibold text-neutral-600">
                    {Math.round(
                      relationship.relationship_strength * 100
                    )}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <p className="text-xs leading-5 text-neutral-500">
        Click a wallet or connection to inspect its observed evidence.
        A connection represents temporal co-occurrence, not proof of
        coordination.
      </p>
    </div>
  );
}

/* --------------------------------------------------
   Network
-------------------------------------------------- */

export function Network() {
  const {
    marketId,
    setMarketId,
    level,
    setLevel,
    filteredPairs,
    selectedPairId,
    selectedWalletId,
    selectPair,
    selectWallet,
  } = usePolymap();

  const mid = marketId ?? markets[0]?.market_id;

  const market = mid ? marketById.get(mid) : undefined;
  const stats = mid ? marketStats.get(mid) : undefined;

  const [hover, setHover] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);

  const allHere = useMemo(
    () =>
      mid
        ? pairs.filter((relationship) =>
          relationship.market_ids.includes(mid)
        )
        : [],
    [mid]
  );

  const layout = useMemo(() => {
    const ids = Array.from(
      new Set(
        allHere.flatMap((relationship) => [
          relationship.wallet_a_id,
          relationship.wallet_b_id,
        ])
      )
    );

    return computeLayout(
      ids,
      allHere.map((relationship) => ({
        a: relationship.wallet_a_id,
        b: relationship.wallet_b_id,
      }))
    );
  }, [allHere]);

  const visible = useMemo(
    () =>
      mid
        ? filteredPairs.filter((relationship) =>
          relationship.market_ids.includes(mid)
        )
        : [],
    [filteredPairs, mid]
  );

  const nodeIds = useMemo(
    () =>
      Array.from(
        new Set(
          visible.flatMap((relationship) => [
            relationship.wallet_a_id,
            relationship.wallet_b_id,
          ])
        )
      ),
    [visible]
  );

  const maxVol = Math.max(
    ...nodeIds.map(
      (id) => walletById.get(id)?.traded_volume_30d_usd ?? 0
    ),
    1
  );

  const selectedPair = selectedPairId
    ? visible.find(
      (relationship) => relationship.pair_id === selectedPairId
    )
    : undefined;

  const focusWallet = hover ?? selectedWalletId;

  /* Wallet highlighting */
  const bright = useMemo(() => {
    if (selectedPair) {
      return new Set([
        selectedPair.wallet_a_id,
        selectedPair.wallet_b_id,
      ]);
    }

    if (focusWallet) {
      const connected = new Set<string>([focusWallet]);

      for (const relationship of visible) {
        if (relationship.wallet_a_id === focusWallet) {
          connected.add(relationship.wallet_b_id);
        }

        if (relationship.wallet_b_id === focusWallet) {
          connected.add(relationship.wallet_a_id);
        }
      }

      return connected;
    }

    return null;
  }, [selectedPair, focusWallet, visible]);

  /* Labels */
  const labeled = useMemo(() => {
    const top = [...nodeIds]
      .sort(
        (a, b) =>
          (walletById.get(b)?.traded_volume_30d_usd ?? 0) -
          (walletById.get(a)?.traded_volume_30d_usd ?? 0)
      )
      .slice(0, nodeIds.length > 20 ? 8 : nodeIds.length);

    const result = new Set(top);

    bright?.forEach((id) => result.add(id));

    return result;
  }, [nodeIds, bright]);

  /* Relationship ordering */
  const ordered = useMemo(
    () =>
      [...visible].sort(
        (a, b) =>
          Number(b.is_tight) - Number(a.is_tight) ||
          Number(b.is_persistent) - Number(a.is_persistent) ||
          b.relationship_strength - a.relationship_strength
      ),
    [visible]
  );

  if (!mid || !market) {
    return (
      <Card>
        <CardBody>
          <p className="text-sm text-neutral-500">
            No market data available.
          </p>
        </CardBody>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Network"
        description="Explore wallets connected by temporally close trades. Select a wallet or connection to inspect its evidence."
        action={
          <Select
            value={mid}
            onChange={(e) => {
              setMarketId(e.target.value || null);
              selectPair(null);
              selectWallet(null);
            }}
            aria-label="Market"
            className="min-w-64"
          >
            {markets.map((item) => (
              <option
                key={item.market_id}
                value={item.market_id}
              >
                {item.name}
              </option>
            ))}
          </Select>
        }
      />

      {/* Market metrics */}
      <Card>
        <CardBody className="grid grid-cols-2 gap-5 sm:grid-cols-4">
          <Stat
            label="Market volume"
            value={
              <span className="text-xl">
                {money(market.total_volume_usd)}
              </span>
            }
          />

          <Stat
            label="Market traders"
            value={
              <span className="text-xl">
                {num(market.total_traders)}
              </span>
            }
          />

          <Stat
            label="Reported candidate pairs"
            value={
              <span className="text-xl">
                {stats?.reportedCandidatePairs ?? "—"}
              </span>
            }
          />

          <Stat
            label="Wallets in relationships"
            value={
              <span className="text-xl">
                {stats?.reportedWalletsInRelationships ?? "—"}
              </span>
            }
          />
        </CardBody>
      </Card>

      {/* Filters */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Segmented<Level>
          value={level}
          onChange={setLevel}
          options={[
            { value: "all", label: "All relationships" },
            { value: "persistent", label: "Persistent" },
            { value: "one_off", label: "One-off" },
            { value: "tight", label: "Tight" },
          ]}
        />

        <p className="text-sm text-neutral-500">
          Showing {visible.length} of {allHere.length} relationships
        </p>
      </div>

      {/* Graph and inspector */}
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card className="overflow-hidden">
          <div className="relative">
            <svg
              viewBox={`0 0 ${W} ${H}`}
              className="block h-auto w-full"
              role="img"
              aria-label="Wallet relationship network"
            >
              <rect
                width={W}
                height={H}
                fill="#FFFFFF"
                onClick={() => {
                  selectPair(null);
                  selectWallet(null);
                }}
              />

              <g
                transform={`translate(${W / 2} ${H / 2}) scale(${zoom}) translate(${-W / 2} ${-H / 2})`}
              >
                {/* Relationship lines */}
                {ordered.map((relationship) => {
                  const a = layout.get(relationship.wallet_a_id);
                  const b = layout.get(relationship.wallet_b_id);

                  if (!a || !b) return null;

                  const style =
                    CATEGORY_STYLE[relationship.category] ??
                    CATEGORY_STYLE.one_off_loose;

                  const isSelected =
                    relationship.pair_id === selectedPairId;

                  const lit =
                    !bright ||
                    isSelected ||
                    (!selectedPair &&
                      bright.has(relationship.wallet_a_id) &&
                      bright.has(relationship.wallet_b_id));

                  const width =
                    1 + Math.sqrt(relationship.events) * 0.45;

                  return (
                    <g
                      key={relationship.pair_id}
                      className="cursor-pointer"
                      onClick={() =>
                        selectPair(relationship.pair_id)
                      }
                    >
                      <line
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke="transparent"
                        strokeWidth={14}
                      />

                      {isSelected && (
                        <line
                          x1={a.x}
                          y1={a.y}
                          x2={b.x}
                          y2={b.y}
                          stroke="#06B6D4"
                          strokeOpacity={0.25}
                          strokeWidth={width + 7}
                          strokeLinecap="round"
                        />
                      )}

                      <line
                        x1={a.x}
                        y1={a.y}
                        x2={b.x}
                        y2={b.y}
                        stroke={isSelected ? "#0891B2" : style.color}
                        strokeWidth={width}
                        strokeDasharray={style.dash}
                        strokeLinecap="round"
                        opacity={lit ? 0.9 : 0.12}
                      />
                    </g>
                  );
                })}

                {/* Wallet nodes */}
                {nodeIds.map((id) => {
                  const wallet = walletById.get(id);
                  const position = layout.get(id);

                  if (!wallet || !position) return null;

                  const radius =
                    7 +
                    Math.sqrt(
                      wallet.traded_volume_30d_usd / maxVol
                    ) *
                    15;

                  const selected = id === selectedWalletId;
                  const dim = bright ? !bright.has(id) : false;

                  return (
                    <g
                      key={id}
                      className="cursor-pointer"
                      opacity={dim ? 0.2 : 1}
                      onClick={() => selectWallet(id)}
                      onMouseEnter={() => setHover(id)}
                      onMouseLeave={() => setHover(null)}
                    >
                      <circle
                        cx={position.x}
                        cy={position.y}
                        r={radius + (selected ? 4 : 0)}
                        fill={WALLET_COLOURS[wallet.wallet_type]}
                        fillOpacity={selected ? 0.18 : 0}
                      />

                      <circle
                        cx={position.x}
                        cy={position.y}
                        r={radius}
                        fill={WALLET_COLOURS[wallet.wallet_type]}
                        stroke={selected ? "#111827" : "#FFFFFF"}
                        strokeWidth={selected ? 3 : 2}
                      />

                      {labeled.has(id) && (
                        <text
                          x={position.x}
                          y={position.y + radius + 15}
                          textAnchor="middle"
                          fontSize="12"
                          fontWeight={600}
                          fill="#171717"
                          stroke="#FFFFFF"
                          strokeWidth={3}
                          paintOrder="stroke"
                        >
                          {addr(wallet.address)}
                        </text>
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>

            {/* Zoom controls */}
            <div className="absolute right-3 top-3 flex flex-col gap-1 rounded-xl border border-neutral-200 bg-white p-1 shadow-sm">
              <Button
                size="icon"
                variant="ghost"
                aria-label="Zoom in"
                onClick={() =>
                  setZoom((z) =>
                    Math.min(2, +(z + 0.2).toFixed(1))
                  )
                }
              >
                <Plus className="h-4 w-4" />
              </Button>

              <Button
                size="icon"
                variant="ghost"
                aria-label="Zoom out"
                onClick={() =>
                  setZoom((z) =>
                    Math.max(0.6, +(z - 0.2).toFixed(1))
                  )
                }
              >
                <Minus className="h-4 w-4" />
              </Button>

              <Button
                size="icon"
                variant="ghost"
                aria-label="Reset zoom"
                onClick={() => setZoom(1)}
              >
                <RotateCcw className="h-4 w-4" />
              </Button>
            </div>

            {visible.length === 0 && (
              <div className="absolute inset-x-0 top-1/2 mx-auto w-fit -translate-y-1/2 rounded-xl border border-neutral-200 bg-white px-4 py-3 text-sm text-neutral-600 shadow-sm">
                No relationships match these filters.
              </div>
            )}
          </div>

          {/* Legend */}
          <CardBody className="border-t border-neutral-100">
            <div className="grid gap-6 text-sm sm:grid-cols-2">
              <div>
                <p className="mb-3 font-semibold text-neutral-900">
                  Connections · Relationship category
                </p>

                <ul className="space-y-2">
                  {Object.entries(CATEGORY_STYLE).map(
                    ([key, style]) => (
                      <li
                        key={key}
                        className="flex items-center gap-2 text-neutral-600"
                      >
                        <svg width="30" height="8" aria-hidden>
                          <line
                            x1="0"
                            y1="4"
                            x2="30"
                            y2="4"
                            stroke={style.color}
                            strokeWidth="2.5"
                            strokeDasharray={style.dash}
                          />
                        </svg>

                        {style.label}
                      </li>
                    )
                  )}
                </ul>

                <p className="mt-3 text-xs leading-5 text-neutral-500">
                  Thicker connections represent more observed events.
                </p>
              </div>

              <div>
                <p className="mb-3 font-semibold text-neutral-900">
                  Wallets · Wallet type
                </p>

                <ul className="space-y-2">
                  {Object.entries(WALLET_COLOURS).map(
                    ([type, colour]) => (
                      <li
                        key={type}
                        className="flex items-center gap-2 text-neutral-600"
                      >
                        <span
                          className="h-3 w-3 rounded-full"
                          style={{ background: colour }}
                        />
                        {type}
                      </li>
                    )
                  )}
                </ul>

                <p className="mt-3 text-xs leading-5 text-neutral-500">
                  Larger circles represent higher 30-day traded volume.
                </p>
              </div>
            </div>
          </CardBody>
        </Card>

        {/* Inspector */}
        <Card className="xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)] xl:self-start xl:overflow-y-auto">
          <CardBody>
            <Inspector
              fallback={
                <MarketSummary
                  marketId={mid}
                  visible={visible}
                />
              }
            />
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
