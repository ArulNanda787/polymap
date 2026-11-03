
"use client";

import {
  overview,
  marketById,
  markets,
  money,
  num,
  pairName,
  pairs,
  wallets,
} from "@/lib/polymap/data";

import { MarketsTable } from "./markets-view";
import { usePolymap } from "./store";

import {
  Button,
  Card,
  CardBody,
  CardHeader,
  PageHeader,
  Stat,
} from "./ui";

/* --------------------------------------------------
   Relationship categories
-------------------------------------------------- */

const CATEGORY_STYLE: Record<
  string,
  { label: string; color: string; background: string }
> = {
  persistent_tight: {
    label: "Persistent · Tight",
    color: "#6D28D9",
    background: "#EDE9FE",
  },
  persistent_loose: {
    label: "Persistent",
    color: "#0F766E",
    background: "#CCFBF1",
  },
  one_off_loose: {
    label: "One-off",
    color: "#C2410C",
    background: "#FFEDD5",
  },
};

function CategoryBadge({ category }: { category: string }) {
  const style = CATEGORY_STYLE[category] ?? {
    label: category.replaceAll("_", " "),
    color: "#525252",
    background: "#F5F5F5",
  };

  return (
    <span
      className="inline-flex whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-semibold"
      style={{
        color: style.color,
        background: style.background,
      }}
    >
      {style.label}
    </span>
  );
}

/* --------------------------------------------------
   Overview
-------------------------------------------------- */

export function Home() {
  const { setView, selectPair, setMarketId } = usePolymap();

  const best = [...pairs]
    .sort((a, b) => {
      if (a.is_tight !== b.is_tight) {
        return Number(b.is_tight) - Number(a.is_tight);
      }

      return (
        Number(b.relationship_strength ?? 0) -
        Number(a.relationship_strength ?? 0)
      );
    })
    .slice(0, 6);

  const biggestMarkets = [...markets]
    .sort(
      (a, b) =>
        b.total_volume_usd - a.total_volume_usd
    )
    .slice(0, 6);

  return (
    <div className="space-y-6">
      <PageHeader
        title="Overview"
        description={`Polymap scanned ${num(overview.unique_wallets)} unique wallets across ${num(overview.markets_scanned)} markets, identifying ${num(overview.candidate_pairs)} candidate relationships.`}
      />

      {/* ------------------------------------------
          Main statistics
      ------------------------------------------ */}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[
          {
            label: "Markets scanned",
            value: num(overview.markets_scanned),
            hint: "Markets included in this investigation.",
          },
          {
            label: "Trades fetched",
            value: num(overview.trades_fetched),
            hint: "Raw trading activity processed.",
          },
          {
            label: "Unique wallets",
            value: num(overview.unique_wallets),
            hint: "Distinct wallet addresses observed.",
          },
          {
            label: "Candidate relationships",
            value: num(overview.candidate_pairs),
            hint: `${num(overview.persistent_pairs)} persistent relationships.`,
          },
        ].map((stat) => (
          <Card key={stat.label}>
            <CardBody>
              <Stat
                label={stat.label}
                value={
                  <span className="text-2xl">
                    {stat.value}
                  </span>
                }
                hint={stat.hint}
              />
            </CardBody>
          </Card>
        ))}
      </div>

      {/* ------------------------------------------
          Secondary statistics
      ------------------------------------------ */}

      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        <Card>
          <CardBody>
            <Stat
              label="Temporal edges"
              value={
                <span className="text-xl">
                  {num(overview.temporal_edges)}
                </span>
              }
              hint="Different-wallet trades within the time window."
            />
          </CardBody>
        </Card>

        <Card>
          <CardBody>
            <Stat
              label="Cross-market pairs"
              value={
                <span className="text-xl">
                  {num(overview.cross_market_pairs)}
                </span>
              }
              hint="Pairs appearing in multiple markets."
            />
          </CardBody>
        </Card>

        <Card>
          <CardBody>
            <Stat
              label="Tight relationships"
              value={
                <span className="text-xl">
                  {num(overview.tight_pairs)}
                </span>
              }
              hint="Very small time and price differences."
            />
          </CardBody>
        </Card>

        <Card>
          <CardBody>
            <Stat
              label="Persistent relationships"
              value={
                <span className="text-xl">
                  {num(overview.persistent_pairs)}
                </span>
              }
              hint="Observed over more than seven days."
            />
          </CardBody>
        </Card>
      </div>

      {/* ------------------------------------------
          Biggest markets
      ------------------------------------------ */}

      <Card className="overflow-hidden">
        <CardHeader
          title="Biggest markets"
          description="Markets ranked by reported total trading volume."
          action={
            <Button
              size="sm"
              onClick={() => setView("markets")}
            >
              See all markets
            </Button>
          }
        />

        <div className="mt-4 border-t border-neutral-200">
          <MarketsTable rows={biggestMarkets} />
        </div>
      </Card>

      {/* ------------------------------------------
          Strongest relationships
      ------------------------------------------ */}

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader
            title="Relationships worth investigating"
            description="Tight relationships are prioritised, followed by the experimental relationship-strength score."
            action={
              <Button
                size="sm"
                variant="outline"
                onClick={() => setView("pairs")}
              >
                View all pairs
              </Button>
            }
          />

          <CardBody className="pt-3">
            {best.length === 0 ? (
              <p className="py-6 text-center text-sm text-neutral-600">
                No relationships available.
              </p>
            ) : (
              <ul className="divide-y divide-neutral-200">
                {best.map((relationship) => {
                  const marketNames = relationship.market_ids
                    .map(
                      (id) => marketById.get(id)?.name ?? id
                    )
                    .join(", ");

                  return (
                    <li key={relationship.pair_id}>
                      <button
                        className="flex w-full items-center justify-between gap-3 rounded-lg px-2 py-3 text-left transition-colors hover:bg-teal-50/60"
                        onClick={() => {
                          setMarketId(
                            relationship.market_ids[0] ?? null
                          );
                          selectPair(relationship.pair_id);
                          setView("network");
                        }}
                      >
                        <span className="min-w-0">
                          <span className="block font-medium text-neutral-950">
                            {pairName(relationship)}
                          </span>

                          <span className="mt-1 block truncate text-xs text-neutral-600">
                            {marketNames}
                          </span>

                          <span className="mt-1 block text-xs text-neutral-600">
                            {num(relationship.events)} events
                            {" · "}
                            {money(relationship.matched_volume_usd)}
                            {" · "}
                            {relationship.span_days} days
                          </span>
                        </span>

                        <span className="flex shrink-0 flex-col items-end gap-2">
                          <CategoryBadge
                            category={relationship.category}
                          />

                          {relationship.is_tight && (
                            <span className="text-xs font-semibold text-violet-700">
                              Tight match
                            </span>
                          )}
                        </span>
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </CardBody>
        </Card>

        {/* ----------------------------------------
            How matching works
        ---------------------------------------- */}

        <Card>
          <CardHeader title="How matching works" />

          <CardBody className="space-y-4 text-sm text-neutral-950">
            <p className="leading-relaxed text-neutral-700">
              Polymap looks for different wallets trading
              in the same market within a short time window.
              Repeated observations help identify
              relationships worth investigating.
            </p>

            <div className="space-y-3">
              <div className="rounded-lg bg-teal-50 p-3">
                <p className="font-semibold text-teal-900">
                  Temporal edge
                </p>
                <p className="mt-1 text-xs leading-relaxed text-teal-800">
                  Two different wallets trade in the same
                  market within 300 seconds.
                </p>
              </div>

              <div className="rounded-lg bg-orange-50 p-3">
                <p className="font-semibold text-orange-900">
                  Candidate relationship
                </p>
                <p className="mt-1 text-xs leading-relaxed text-orange-800">
                  A wallet pair has at least three
                  qualifying temporal events.
                </p>
              </div>

              <div className="rounded-lg bg-violet-50 p-3">
                <p className="font-semibold text-violet-900">
                  Tight relationship
                </p>
                <p className="mt-1 text-xs leading-relaxed text-violet-800">
                  Median time gap is at most five seconds
                  and median price difference is at most
                  $0.01.
                </p>
              </div>

              <div className="rounded-lg bg-blue-50 p-3">
                <p className="font-semibold text-blue-900">
                  Persistent relationship
                </p>
                <p className="mt-1 text-xs leading-relaxed text-blue-800">
                  Observed over more than seven days and
                  at least three distinct timestamps.
                </p>
              </div>
            </div>

            <div className="rounded-lg border border-neutral-200 bg-neutral-50 p-3">
              <p className="font-semibold text-neutral-900">
                Important
              </p>
              <p className="mt-1 text-xs leading-relaxed text-neutral-600">
                A match is a signal to investigate, not
                proof of coordination. Wallets can trade
                similarly because they react to the same
                information or market conditions.
              </p>
            </div>
          </CardBody>
        </Card>
      </div>

      {/* ------------------------------------------
          Wallet categories
      ------------------------------------------ */}

      <Card>
        <CardHeader
          title="Wallet classification"
          description="The four wallet types used in the network visualisation."
        />

        <CardBody>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {[
              {
                label: "Whale",
                color: "#7C3AED",
                description: "High-volume wallet activity.",
              },
              {
                label: "Frequent trader",
                color: "#2563EB",
                description: "Regular trading activity.",
              },
              {
                label: "Occasional trader",
                color: "#16A34A",
                description: "Less frequent participation.",
              },
              {
                label: "New wallet",
                color: "#EA580C",
                description: "Recently observed wallet activity.",
              },
            ].map((kind) => (
              <div
                key={kind.label}
                className="flex items-start gap-3 rounded-xl border border-neutral-200 bg-white p-4"
              >
                <span
                  className="mt-1 h-3.5 w-3.5 shrink-0 rounded-full"
                  style={{ background: kind.color }}
                />

                <div>
                  <p className="font-semibold text-neutral-950">
                    {kind.label}
                  </p>
                  <p className="mt-1 text-xs text-neutral-600">
                    {kind.description}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </CardBody>
      </Card>

      <p className="text-xs leading-relaxed text-neutral-500">
        Dataset: {wallets.length} wallets · {pairs.length} candidate
        relationships · {markets.length} markets.
      </p>
    </div>
  );
}
