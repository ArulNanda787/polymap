
"use client";

import { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Pie,
  PieChart,
  ReferenceArea,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import {
  marketById,
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

import {
  ChartContainer,
  ChartLegend,
  TooltipCard,
  type ChartConfig,
} from "./chart-kit";

import { usePolymap } from "./store";

import {
  Card,
  CardBody,
  CardHeader,
  EmptyState,
  PageHeader,
  Select,
  Stat,
} from "./ui";

/* --------------------------------------------------
   Shared styles and helpers
-------------------------------------------------- */

type TipProps = {
  active?: boolean;
  payload?: readonly any[];
  label?: any;
};

const ink = "#171717";

const CATEGORY_STYLE: Record<
  string,
  { label: string; color: string }
> = {
  persistent_tight: {
    label: "Persistent · Tight",
    color: "#7C3AED",
  },
  persistent_loose: {
    label: "Persistent",
    color: "#0D9488",
  },
  one_off_loose: {
    label: "One-off",
    color: "#EA580C",
  },
};

const WALLET_STYLE: Record<
  WalletType,
  { label: string; color: string }
> = {
  Whale: {
    label: "Whale",
    color: "#7C3AED",
  },
  "Frequent trader": {
    label: "Frequent trader",
    color: "#2563EB",
  },
  "Occasional trader": {
    label: "Occasional trader",
    color: "#16A34A",
  },
  "New wallet": {
    label: "New wallet",
    color: "#EA580C",
  },
};

const WALLET_ORDER: WalletType[] = [
  "Whale",
  "Frequent trader",
  "Occasional trader",
  "New wallet",
];

const CATEGORY_ORDER = [
  "persistent_tight",
  "persistent_loose",
  "one_off_loose",
];

function categoryLabel(category: string) {
  return (
    CATEGORY_STYLE[category]?.label ??
    category.replaceAll("_", " ")
  );
}

function categoryColor(category: string) {
  return CATEGORY_STYLE[category]?.color ?? "#737373";
}

function relationshipsForMarket(
  relationships: CandidateRelationship[],
  marketId: string | null
) {
  if (!marketId) return relationships;

  return relationships.filter((relationship) =>
    relationship.market_ids.includes(marketId)
  );
}

/* --------------------------------------------------
   1. Biggest markets
-------------------------------------------------- */

function BiggestMarkets() {
  const [measure, setMeasure] = useState<
    "total_volume_usd" | "total_traders"
  >("total_volume_usd");

  const data = useMemo(
    () =>
      [...markets]
        .sort((a, b) => b[measure] - a[measure])
        .slice(0, 8)
        .map((market) => ({
          name: market.name,
          value: market[measure],
          formatted:
            measure === "total_volume_usd"
              ? money(market.total_volume_usd)
              : num(market.total_traders),
        })),
    [measure]
  );

  const config: ChartConfig = {
    value: {
      label:
        measure === "total_volume_usd"
          ? "Total volume"
          : "Traders",
      color: "#0D9488",
    },
  };

  return (
    <Card>
      <CardHeader
        title="Which markets are the biggest?"
        description="The eight largest markets by volume or trader count."
        action={
          <Select
            value={measure}
            onChange={(event) =>
              setMeasure(
                event.target.value as typeof measure
              )
            }
            aria-label="Market measurement"
            className="h-8"
          >
            <option value="total_volume_usd">
              Total volume
            </option>
            <option value="total_traders">
              Total traders
            </option>
          </Select>
        }
      />

      <CardBody>
        <ChartContainer config={config} className="h-80">
          <BarChart
            data={data}
            layout="vertical"
            margin={{
              left: 0,
              right: 16,
              top: 4,
              bottom: 0,
            }}
          >
            <CartesianGrid horizontal={false} />

            <XAxis
              type="number"
              tickFormatter={(value) =>
                measure === "total_volume_usd"
                  ? money(value)
                  : num(value)
              }
              tickLine={false}
              axisLine={false}
            />

            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11 }}
            />

            <Tooltip
              cursor={{ fill: "#F0FDFA" }}
              content={({ active, payload }: TipProps) =>
                active && payload?.length ? (
                  <TooltipCard
                    title={payload[0].payload.name}
                    rows={[
                      {
                        label:
                          measure === "total_volume_usd"
                            ? "Total volume"
                            : "Total traders",
                        value: payload[0].payload.formatted,
                      },
                    ]}
                  />
                ) : null
              }
            />

            <Bar
              dataKey="value"
              fill="var(--color-value)"
              radius={[0, 5, 5, 0]}
              barSize={20}
            />
          </BarChart>
        </ChartContainer>
      </CardBody>
    </Card>
  );
}

/* --------------------------------------------------
   2. Relationships by market
-------------------------------------------------- */

function RelationshipsByMarket() {
  const data = useMemo(
    () =>
      markets
        .map((market) => {
          const relationships = relationshipsForMarket(
            pairs,
            market.market_id
          );

          const counts = {
            persistent_tight: 0,
            persistent_loose: 0,
            one_off_loose: 0,
          };

          relationships.forEach((relationship) => {
            if (relationship.category in counts) {
              counts[
                relationship.category as keyof typeof counts
              ]++;
            }
          });

          return {
            name: market.name,
            ...counts,
            total: relationships.length,
          };
        })
        .sort((a, b) => b.total - a.total)
        .slice(0, 8),
    []
  );

  const config: ChartConfig = {
    persistent_tight: {
      label: "Persistent · Tight",
      color: categoryColor("persistent_tight"),
    },
    persistent_loose: {
      label: "Persistent",
      color: categoryColor("persistent_loose"),
    },
    one_off_loose: {
      label: "One-off",
      color: categoryColor("one_off_loose"),
    },
  };

  return (
    <Card>
      <CardHeader
        title="Where do wallet relationships appear?"
        description="Relationships involving each market, grouped by category."
      />

      <CardBody className="space-y-3">
        <ChartContainer config={config} className="h-80">
          <BarChart
            data={data}
            layout="vertical"
            margin={{
              left: 0,
              right: 16,
              top: 4,
              bottom: 0,
            }}
          >
            <CartesianGrid horizontal={false} />

            <XAxis
              type="number"
              allowDecimals={false}
              tickLine={false}
              axisLine={false}
            />

            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11 }}
            />

            <Tooltip
              cursor={{ fill: "#F0FDFA" }}
              content={({ active, payload }: TipProps) =>
                active && payload?.length ? (
                  <TooltipCard
                    title={payload[0].payload.name}
                    rows={CATEGORY_ORDER.map((category) => ({
                      label: categoryLabel(category),
                      value: String(
                        payload[0].payload[category]
                      ),
                      color: categoryColor(category),
                    }))}
                  />
                ) : null
              }
            />

            {CATEGORY_ORDER.map((category, index) => (
              <Bar
                key={category}
                dataKey={category}
                stackId="relationships"
                fill={`var(--color-${category})`}
                barSize={20}
                radius={
                  index === CATEGORY_ORDER.length - 1
                    ? [0, 5, 5, 0]
                    : undefined
                }
              />
            ))}
          </BarChart>
        </ChartContainer>

        <ChartLegend
          items={CATEGORY_ORDER.map((category) => ({
            label: categoryLabel(category),
            color: categoryColor(category),
          }))}
        />
      </CardBody>
    </Card>
  );
}

/* --------------------------------------------------
   3. Main charts view
-------------------------------------------------- */

export function ChartsView() {
  const { marketId, setMarketId } = usePolymap();

  const selectedMarkets = marketId
    ? markets.filter(
      (market) => market.market_id === marketId
    )
    : markets;

  const selectedRelationships = useMemo(
    () => relationshipsForMarket(pairs, marketId),
    [marketId]
  );

  const totalVolume = selectedMarkets.reduce(
    (sum, market) => sum + market.total_volume_usd,
    0
  );

  const matchedVolume = selectedRelationships.reduce(
    (sum, relationship) =>
      sum + relationship.matched_volume_usd,
    0
  );

  const persistentCount = selectedRelationships.filter(
    (relationship) => relationship.is_persistent
  ).length;

  const tightCount = selectedRelationships.filter(
    (relationship) => relationship.is_tight
  ).length;

  /* ----------------------------------------------
     Wallet type distribution
  ---------------------------------------------- */

  const kinds = useMemo(() => {
    const walletIds = new Set(
      selectedRelationships.flatMap((relationship) => [
        relationship.wallet_a_id,
        relationship.wallet_b_id,
      ])
    );

    return WALLET_ORDER.map((kind) => ({
      kind,
      label: WALLET_STYLE[kind].label,
      color: WALLET_STYLE[kind].color,
      value: Array.from(walletIds).filter(
        (id) => walletById.get(id)?.wallet_type === kind
      ).length,
    }));
  }, [selectedRelationships]);

  const totalWallets = kinds.reduce(
    (sum, kind) => sum + kind.value,
    0
  );

  /* ----------------------------------------------
     Relationship category distribution
  ---------------------------------------------- */

  const categoryData = useMemo(
    () =>
      CATEGORY_ORDER.map((category) => ({
        category,
        label: categoryLabel(category),
        color: categoryColor(category),
        value: selectedRelationships.filter(
          (relationship) =>
            relationship.category === category
        ).length,
      })),
    [selectedRelationships]
  );

  /* ----------------------------------------------
     Relationship scatter plot
  ---------------------------------------------- */

  const scatter = useMemo(
    () =>
      CATEGORY_ORDER.map((category) => ({
        category,
        label: categoryLabel(category),
        color: categoryColor(category),
        points: selectedRelationships
          .filter(
            (relationship) =>
              relationship.category === category
          )
          .map((relationship) => ({
            x: relationship.median_time_gap_seconds,
            y: relationship.events,
            z: relationship.matched_volume_usd,
            name: pairName(relationship),
            market: relationship.market_ids
              .map(
                (id) =>
                  marketById.get(id)?.name ?? id
              )
              .join(", "),
            category: relationship.category,
            priceDifference:
              relationship.median_price_difference,
            sameSideRate: relationship.same_side_rate,
            persistent: relationship.is_persistent,
            tight: relationship.is_tight,
          })),
      })),
    [selectedRelationships]
  );

  const walletConfig: ChartConfig = {};

  const scatterConfig: ChartConfig = {
    persistent_tight: {
      label: "Persistent · Tight",
      color: categoryColor("persistent_tight"),
    },
    persistent_loose: {
      label: "Persistent",
      color: categoryColor("persistent_loose"),
    },
    one_off_loose: {
      label: "One-off",
      color: categoryColor("one_off_loose"),
    },
  };

  const categoryConfig: ChartConfig = {
    value: {
      label: "Relationships",
      color: "#0D9488",
    },
  };

  return (
    <div className="space-y-8">
      <PageHeader
        title="Charts"
        description="Explore market activity, wallet types, and observed relationships."
      />

      <div className="grid gap-6 xl:grid-cols-2">
        <BiggestMarkets />
        <RelationshipsByMarket />
      </div>

      {/* ------------------------------------------
          Detailed analysis
      ------------------------------------------ */}

      <div className="space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-lg font-semibold tracking-tight text-neutral-950">
              Look closer
            </h2>

            <p className="mt-1 text-sm text-neutral-600">
              Select a market or explore all markets together.
            </p>
          </div>

          <Select
            value={marketId ?? ""}
            onChange={(event) =>
              setMarketId(event.target.value || null)
            }
            aria-label="Market"
            className="min-w-64"
          >
            <option value="">All markets</option>

            {markets.map((market) => (
              <option
                key={market.market_id}
                value={market.market_id}
              >
                {market.name}
              </option>
            ))}
          </Select>
        </div>

        {/* Summary statistics */}

        <Card>
          <CardBody className="grid grid-cols-2 gap-5 lg:grid-cols-4">
            <Stat
              label="Total market volume"
              value={
                <span className="text-xl">
                  {money(totalVolume)}
                </span>
              }
            />

            <Stat
              label="Wallet relationships"
              value={
                <span className="text-xl">
                  {num(selectedRelationships.length)}
                </span>
              }
            />

            <Stat
              label="Persistent relationships"
              value={
                <span className="text-xl">
                  {num(persistentCount)}
                </span>
              }
            />

            <Stat
              label="Tight relationships"
              value={
                <span className="text-xl">
                  {num(tightCount)}
                </span>
              }
            />
          </CardBody>
        </Card>

        {/* Wallet distribution and categories */}

        <div className="grid gap-6 xl:grid-cols-2">
          <Card>
            <CardHeader
              title="What kind of wallets are involved?"
              description="Distinct wallets appearing in the selected relationships."
            />

            <CardBody>
              {totalWallets === 0 ? (
                <EmptyState
                  title="No wallets"
                  description="There are no relationships in this selection."
                />
              ) : (
                <div className="space-y-4">
                  <div className="relative">
                    <ChartContainer
                      config={walletConfig}
                      className="h-56"
                    >
                      <PieChart>
                        <Tooltip
                          content={({
                            active,
                            payload,
                          }: TipProps) =>
                            active && payload?.length ? (
                              <TooltipCard
                                title={payload[0].payload.label}
                                rows={[
                                  {
                                    label: "Wallets",
                                    value: String(
                                      payload[0].value
                                    ),
                                    color:
                                      payload[0].payload.color,
                                  },
                                ]}
                              />
                            ) : null
                          }
                        />

                        <Pie
                          data={kinds}
                          dataKey="value"
                          nameKey="label"
                          innerRadius={58}
                          outerRadius={88}
                          paddingAngle={3}
                          stroke="#fff"
                          strokeWidth={2}
                        >
                          {kinds.map((kind) => (
                            <Cell
                              key={kind.kind}
                              fill={kind.color}
                            />
                          ))}
                        </Pie>
                      </PieChart>
                    </ChartContainer>

                    <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center">
                      <span className="text-2xl font-semibold text-neutral-950">
                        {totalWallets}
                      </span>
                      <span className="text-xs text-neutral-600">
                        distinct wallets
                      </span>
                    </div>
                  </div>

                  <ul className="space-y-2">
                    {kinds.map((kind) => (
                      <li
                        key={kind.kind}
                        className="flex items-center justify-between text-sm text-neutral-950"
                      >
                        <span className="flex items-center gap-2">
                          <span
                            className="h-3 w-3 rounded-full"
                            style={{
                              background: kind.color,
                            }}
                          />
                          {kind.label}
                        </span>

                        <span className="font-medium">
                          {kind.value}{" "}
                          <span className="font-normal text-neutral-600">
                            (
                            {Math.round(
                              (kind.value / totalWallets) * 100
                            )}
                            %)
                          </span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </CardBody>
          </Card>

          <Card>
            <CardHeader
              title="What types of relationships were found?"
              description="Distribution of the three relationship categories."
            />

            <CardBody>
              {selectedRelationships.length === 0 ? (
                <EmptyState
                  title="No relationships"
                  description="There are no relationships in this selection."
                />
              ) : (
                <ChartContainer
                  config={categoryConfig}
                  className="h-64"
                >
                  <PieChart>
                    <Tooltip
                      content={({
                        active,
                        payload,
                      }: TipProps) =>
                        active && payload?.length ? (
                          <TooltipCard
                            title={payload[0].payload.label}
                            rows={[
                              {
                                label: "Relationships",
                                value: String(
                                  payload[0].value
                                ),
                                color:
                                  payload[0].payload.color,
                              },
                            ]}
                          />
                        ) : null
                      }
                    />

                    <Pie
                      data={categoryData}
                      dataKey="value"
                      nameKey="label"
                      innerRadius={55}
                      outerRadius={90}
                      paddingAngle={3}
                      stroke="#fff"
                      strokeWidth={2}
                    >
                      {categoryData.map((category) => (
                        <Cell
                          key={category.category}
                          fill={category.color}
                        />
                      ))}
                    </Pie>
                  </PieChart>
                </ChartContainer>
              )}

              <ChartLegend
                items={categoryData.map((category) => ({
                  label: category.label,
                  color: category.color,
                }))}
              />
            </CardBody>
          </Card>
        </div>

        {/* Relationship scatter plot */}

        <Card>
          <CardHeader
            title="Which relationships are worth investigating?"
            description="Each dot represents a wallet pair. Left-side dots have smaller median time gaps; higher dots have more observed events. Larger dots indicate higher matched volume."
          />

          <CardBody className="space-y-3">
            {selectedRelationships.length === 0 ? (
              <EmptyState
                title="No relationships"
                description="There are no relationships in this market selection."
              />
            ) : (
              <ChartContainer
                config={scatterConfig}
                className="h-96"
              >
                <ScatterChart
                  margin={{
                    left: 8,
                    right: 16,
                    top: 8,
                    bottom: 28,
                  }}
                >
                  <CartesianGrid />

                  <ReferenceArea
                    x1={0}
                    x2={5}
                    fill="#CCFBF1"
                    fillOpacity={0.45}
                    ifOverflow="extendDomain"
                  />

                  <XAxis
                    type="number"
                    dataKey="x"
                    name="Median time gap"
                    domain={[0, "dataMax"]}
                    tickLine={false}
                    axisLine={false}
                    label={{
                      value: "Median time gap (seconds)",
                      position: "insideBottom",
                      offset: -16,
                      fill: ink,
                      fontSize: 12,
                    }}
                  />

                  <YAxis
                    type="number"
                    dataKey="y"
                    name="Events"
                    tickLine={false}
                    axisLine={false}
                    width={48}
                    label={{
                      value: "Observed events",
                      angle: -90,
                      position: "insideLeft",
                      fill: ink,
                      fontSize: 12,
                    }}
                  />

                  <ZAxis
                    type="number"
                    dataKey="z"
                    range={[50, 380]}
                  />

                  <Tooltip
                    cursor={{
                      strokeDasharray: "3 3",
                      stroke: "#A3A3A3",
                    }}
                    content={({
                      active,
                      payload,
                    }: TipProps) => {
                      if (!active || !payload?.length) {
                        return null;
                      }

                      const point = payload[0].payload;

                      return (
                        <TooltipCard
                          title={point.name}
                          rows={[
                            {
                              label: "Markets",
                              value: point.market,
                            },
                            {
                              label: "Category",
                              value: categoryLabel(
                                point.category
                              ),
                              color: categoryColor(
                                point.category
                              ),
                            },
                            {
                              label: "Median time gap",
                              value: `${point.x} sec`,
                            },
                            {
                              label: "Observed events",
                              value: String(point.y),
                            },
                            {
                              label: "Matched volume",
                              value: money(point.z),
                            },
                            {
                              label: "Median price difference",
                              value: String(
                                point.priceDifference
                              ),
                            },
                            {
                              label: "Same-side rate",
                              value: `${(
                                point.sameSideRate * 100
                              ).toFixed(1)}%`,
                            },
                          ]}
                        />
                      );
                    }}
                  />

                  {scatter.map((group) => (
                    <Scatter
                      key={group.category}
                      name={group.label}
                      data={group.points}
                      fill={`var(--color-${group.category})`}
                      fillOpacity={0.8}
                      stroke="#fff"
                    />
                  ))}
                </ScatterChart>
              </ChartContainer>
            )}

            <ChartLegend
              items={CATEGORY_ORDER.map((category) => ({
                label: categoryLabel(category),
                color: categoryColor(category),
              }))}
            />
          </CardBody>
        </Card>

        <p className="text-sm text-neutral-600">
          These are observed wallet relationships, not proof of
          coordination. Matched volume is relationship-level
          volume and should not be interpreted as a verified
          share of total market trading.
        </p>
      </div>
    </div>
  );
}
