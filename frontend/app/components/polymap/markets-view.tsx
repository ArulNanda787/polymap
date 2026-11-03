
"use client";

import {
  markets,
  money,
  num,
} from "@/lib/polymap/data";

import type { Market } from "@/lib/polymap/types";

import { usePolymap } from "./store";

import {
  Badge,
  Card,
  PageHeader,
} from "./ui";

/* --------------------------------------------------
   Markets table
-------------------------------------------------- */

export function MarketsTable({
  rows,
}: {
  rows: Market[];
}) {
  const {
    setMarketId,
    setView,
    selectPair,
    selectWallet,
  } = usePolymap();

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-neutral-950">
        <thead className="border-b border-neutral-200 bg-neutral-50 text-left text-xs text-neutral-600">
          <tr>
            <th className="px-4 py-3 font-semibold">
              Market
            </th>

            <th className="px-4 py-3 text-right font-semibold">
              Total volume
            </th>

            <th className="px-4 py-3 text-right font-semibold">
              Traders
            </th>

            <th className="px-4 py-3 text-right font-semibold">
              Candidate pairs
            </th>

            <th className="px-4 py-3 text-right font-semibold">
              Wallets involved
            </th>

            <th className="px-4 py-3 text-right font-semibold">
              Matched volume
            </th>

            <th className="px-4 py-3 text-right font-semibold">
              Volume share
            </th>
          </tr>
        </thead>

        <tbody className="divide-y divide-neutral-200">
          {rows.map((market) => (
            <tr
              key={market.market_id}
              onClick={() => {
                selectPair(null);
                selectWallet(null);
                setMarketId(market.market_id);
                setView("network");
              }}
              className="cursor-pointer transition-colors hover:bg-teal-50/60"
            >
              {/* Market */}

              <td className="min-w-56 px-4 py-4">
                <div className="font-semibold text-neutral-950">
                  {market.name}
                </div>

                <div className="mt-1.5">
                  <Badge className="px-2 py-0.5 font-medium">
                    {market.category}
                  </Badge>
                </div>
              </td>

              {/* Total volume */}

              <td className="whitespace-nowrap px-4 py-4 text-right font-medium tabular-nums">
                {money(market.total_volume_usd)}
              </td>

              {/* Traders */}

              <td className="whitespace-nowrap px-4 py-4 text-right tabular-nums">
                {num(market.total_traders)}
              </td>

              {/* Candidate pairs */}

              <td className="whitespace-nowrap px-4 py-4 text-right font-medium tabular-nums">
                {num(market.candidate_pairs)}
              </td>

              {/* Wallets involved */}

              <td className="whitespace-nowrap px-4 py-4 text-right tabular-nums">
                {num(market.wallets_in_relationships)}
              </td>

              {/* Matched volume */}

              <td className="whitespace-nowrap px-4 py-4 text-right font-medium tabular-nums">
                {money(market.matched_pair_volume_usd)}
              </td>

              {/* Matched volume share */}

              <td className="whitespace-nowrap px-4 py-4 text-right">
                <span className="inline-flex rounded-full bg-teal-50 px-2.5 py-1 font-semibold text-teal-800">
                  {(market.matched_pair_volume_share * 100).toFixed(2)}%
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* --------------------------------------------------
   Markets page
-------------------------------------------------- */

export function MarketsView() {
  return (
    <div className="space-y-6">
      <PageHeader
        title="Markets"
        description="Explore the markets scanned by Polymap. Select a market to inspect its wallet relationships."
      />

      <Card className="overflow-hidden">
        <MarketsTable rows={markets} />
      </Card>

      <div className="flex flex-wrap items-center justify-between gap-3 text-sm text-neutral-600">
        <span>
          Showing {markets.length} markets
        </span>

        <span>
          Matched volume share is the reported market-level metric.
        </span>
      </div>
    </div>
  );
}
