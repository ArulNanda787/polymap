
"use client";

import { useMemo, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  Download,
} from "lucide-react";

import {
  addr,
  marketById,
  markets,
  money,
  pairName,
  secondsText,
  walletById,
} from "@/lib/polymap/data";

import type { CandidateRelationship } from "@/lib/polymap/types";

import { PairPanel } from "./panels";
import { usePolymap, type Level } from "./store";
import {
  Button,
  Card,
  CardBody,
  EmptyState,
  Input,
  KindDot,
  PageHeader,
  Segmented,
  Select,
} from "./ui";

type SortKey = "match" | "trades" | "money" | "days" | "gap";

const PAGE = 25;

const COLUMNS: {
  key: SortKey | null;
  label: string;
  right?: boolean;
}[] = [
    { key: null, label: "Wallets" },
    { key: null, label: "Markets" },
    { key: "match", label: "Relationship" },
    { key: "trades", label: "Events", right: true },
    { key: "money", label: "Matched volume", right: true },
    { key: "days", label: "Observed over", right: true },
    { key: "gap", label: "Median time gap", right: true },
  ];

const CATEGORY_STYLE: Record<
  string,
  { label: string; className: string }
> = {
  persistent_tight: {
    label: "Persistent · Tight",
    className: "bg-violet-100 text-violet-900 ring-violet-200",
  },
  persistent_loose: {
    label: "Persistent",
    className: "bg-teal-100 text-teal-900 ring-teal-200",
  },
  one_off_loose: {
    label: "One-off",
    className: "bg-orange-100 text-orange-900 ring-orange-200",
  },
};

function CategoryBadge({ category }: { category: string }) {
  const style = CATEGORY_STYLE[category] ?? {
    label: category.replaceAll("_", " "),
    className: "bg-neutral-100 text-neutral-800 ring-neutral-200",
  };

  return (
    <span
      className={`inline-flex whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${style.className}`}
    >
      {style.label}
    </span>
  );
}

function sortValue(p: CandidateRelationship, key: SortKey): number {
  switch (key) {
    case "match":
      return Number(p.relationship_strength ?? 0);
    case "trades":
      return p.events;
    case "money":
      return p.matched_volume_usd;
    case "days":
      return p.span_days;
    case "gap":
      return p.median_time_gap_seconds;
  }
}

function marketNames(marketIds: string[]) {
  return marketIds
    .map((id) => marketById.get(id)?.name ?? id)
    .join(", ");
}

export function PairsView() {
  const {
    filteredPairs,
    marketId,
    setMarketId,
    level,
    setLevel,
    search,
    setSearch,
    selectedPairId,
    selectPair,
  } = usePolymap();

  const [sortKey, setSortKey] = useState<SortKey>("match");
  const [dir, setDir] = useState<"asc" | "desc">("desc");
  const [shown, setShown] = useState(PAGE);

  const rows = useMemo(() => {
    const out = [...filteredPairs];

    out.sort((a, b) => {
      const difference = sortValue(a, sortKey) - sortValue(b, sortKey);
      return difference * (dir === "asc" ? 1 : -1);
    });

    return out;
  }, [filteredPairs, sortKey, dir]);

  function onSort(key: SortKey) {
    if (key === sortKey) {
      setDir((current) => (current === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setDir(key === "gap" ? "asc" : "desc");
    }
  }

  function exportCsv() {
    const header = [
      "Pair ID",
      "Wallet A",
      "Wallet B",
      "Markets",
      "Category",
      "Events",
      "Matched volume USD",
      "Observed span days",
      "Median time gap seconds",
      "Median price difference",
      "Same side rate",
      "Relationship strength",
      "Tight",
      "Persistent",
    ];

    const lines = rows.map((p) => {
      const walletA = walletById.get(p.wallet_a_id);
      const walletB = walletById.get(p.wallet_b_id);

      return [
        p.pair_id,
        walletA?.address ?? p.wallet_a_id,
        walletB?.address ?? p.wallet_b_id,
        marketNames(p.market_ids),
        p.category,
        p.events,
        p.matched_volume_usd,
        p.span_days,
        p.median_time_gap_seconds,
        p.median_price_difference,
        p.same_side_rate,
        p.relationship_strength,
        p.is_tight,
        p.is_persistent,
      ]
        .map((value) => `"${String(value ?? "").replace(/"/g, '""')}"`)
        .join(",");
    });

    const csv = [header.join(","), ...lines].join("\n");
    const blob = new Blob([csv], {
      type: "text/csv;charset=utf-8;",
    });

    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");

    anchor.href = url;
    anchor.download = "polymap-pairs.csv";
    anchor.click();

    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Pairs"
        description="Wallet relationships identified through repeated temporal activity. Select a pair to inspect its evidence."
        action={
          <Button onClick={exportCsv} disabled={rows.length === 0}>
            <Download className="h-4 w-4" />
            Download CSV
          </Button>
        }
      />

      <div className="flex flex-wrap items-center gap-3">
        <Select
          value={marketId ?? ""}
          onChange={(event) => {
            setMarketId(event.target.value || null);
            setShown(PAGE);
          }}
          aria-label="Market"
          className="min-w-56"
        >
          <option value="">All markets</option>
          {markets.map((market) => (
            <option key={market.market_id} value={market.market_id}>
              {market.name}
            </option>
          ))}
        </Select>

        <Segmented<Level>
          value={level}
          onChange={(nextLevel) => {
            setLevel(nextLevel);
            setShown(PAGE);
          }}
          options={[
            { value: "all", label: "All relationships" },
            { value: "persistent", label: "Persistent" },
            { value: "one_off", label: "One-off" },
            { value: "tight", label: "Tight" },
          ]}
        />

        <Input
          value={search}
          onChange={(event) => {
            setSearch(event.target.value);
            setShown(PAGE);
          }}
          placeholder="Search wallet address"
          aria-label="Search a wallet"
          className="w-56"
        />

        <span className="text-sm font-medium text-neutral-600">
          {rows.length} relationships
        </span>
      </div>

      <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
        <Card className="overflow-hidden">
          {rows.length === 0 ? (
            <EmptyState
              title="No relationships found"
              description="Try another market, wallet address, or relationship filter."
            />
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-neutral-950">
                  <thead className="border-b border-neutral-200 bg-neutral-50 text-xs text-neutral-600">
                    <tr>
                      {COLUMNS.map((column) => (
                        <th
                          key={column.label}
                          className={`whitespace-nowrap px-4 py-3 font-semibold ${column.right ? "text-right" : "text-left"
                            }`}
                        >
                          {column.key ? (
                            <button
                              onClick={() => onSort(column.key!)}
                              className={`inline-flex items-center gap-1 hover:text-neutral-950 ${sortKey === column.key
                                  ? "text-neutral-950"
                                  : ""
                                }`}
                            >
                              {column.label}
                              {sortKey === column.key ? (
                                dir === "asc" ? (
                                  <ArrowUp className="h-3 w-3" />
                                ) : (
                                  <ArrowDown className="h-3 w-3" />
                                )
                              ) : (
                                <ArrowUpDown className="h-3 w-3" />
                              )}
                            </button>
                          ) : (
                            column.label
                          )}
                        </th>
                      ))}
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-neutral-200">
                    {rows.slice(0, shown).map((pair) => {
                      const walletA = walletById.get(pair.wallet_a_id);
                      const walletB = walletById.get(pair.wallet_b_id);

                      return (
                        <tr
                          key={pair.pair_id}
                          onClick={() => selectPair(pair.pair_id)}
                          className={`cursor-pointer transition-colors hover:bg-teal-50/60 ${pair.pair_id === selectedPairId
                              ? "bg-teal-50"
                              : ""
                            }`}
                        >
                          <td className="whitespace-nowrap px-4 py-3">
                            <div className="flex items-center gap-2 font-medium">
                              {walletA && (
                                <KindDot kind={walletA.wallet_type} />
                              )}
                              {addr(walletA?.address ?? pair.wallet_a_id)}
                            </div>

                            <div className="mt-2 flex items-center gap-2 font-medium">
                              {walletB && (
                                <KindDot kind={walletB.wallet_type} />
                              )}
                              {addr(walletB?.address ?? pair.wallet_b_id)}
                            </div>
                          </td>

                          <td className="max-w-52 px-4 py-3">
                            <span
                              className="block truncate"
                              title={marketNames(pair.market_ids)}
                            >
                              {pair.market_ids.length === 1
                                ? marketById.get(pair.market_ids[0])?.name ??
                                pair.market_ids[0]
                                : `${pair.market_count} markets`}
                            </span>
                          </td>

                          <td className="px-4 py-3">
                            <CategoryBadge category={pair.category} />
                          </td>

                          <td className="px-4 py-3 text-right tabular-nums">
                            {pair.events.toLocaleString()}
                          </td>

                          <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums">
                            {money(pair.matched_volume_usd)}
                          </td>

                          <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums">
                            {pair.span_days.toLocaleString()} days
                          </td>

                          <td className="whitespace-nowrap px-4 py-3 text-right tabular-nums">
                            {secondsText(pair.median_time_gap_seconds)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {shown < rows.length && (
                <div className="flex justify-center border-t border-neutral-200 p-3">
                  <Button
                    size="sm"
                    onClick={() => setShown((current) => current + PAGE)}
                  >
                    Show {Math.min(PAGE, rows.length - shown)} more
                  </Button>
                </div>
              )}
            </>
          )}
        </Card>

        <Card className="xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)] xl:self-start xl:overflow-y-auto">
          <CardBody>
            {selectedPairId ? (
              <PairPanel pairId={selectedPairId} />
            ) : (
              <EmptyState
                title="No relationship selected"
                description="Select a row to inspect the wallets, markets, and evidence behind the relationship."
              />
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}
