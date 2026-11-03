"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUpDown, Search, Wallet } from "lucide-react";

import {
  markets,
  money,
  pairs,
  pairsForWallet,
  wallets,
  addr,
  num,
} from "@/lib/polymap/data";

import { WalletPanel } from "./panels";
import { usePolymap } from "./store";
import {
  Card,
  CardBody,
  EmptyState,
  Input,
  PageHeader,
  Select,
} from "./ui";

type SortKey = "volume" | "trades" | "matches";

type WalletKind = (typeof wallets)[number]["wallet_type"];

const PAGE = 25;

const WALLET_TYPES: WalletKind[] = [
  "Whale",
  "Frequent trader",
  "Occasional trader",
  "New wallet",
];

const WALLET_STYLE: Record<
  WalletKind,
  { label: string; color: string; background: string }
> = {
  "Whale": {
    label: "Whale",
    color: "#7C3AED",
    background: "#F3E8FF",
  },
  "Frequent trader": {
    label: "Frequent trader",
    color: "#2563EB",
    background: "#DBEAFE",
  },
  "Occasional trader": {
    label: "Occasional trader",
    color: "#16A34A",
    background: "#DCFCE7",
  },
  "New wallet": {
    label: "New wallet",
    color: "#EA580C",
    background: "#FFEDD5",
  },
};

function WalletTypeBadge({ kind }: { kind: WalletKind }) {
  const style = WALLET_STYLE[kind];

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-semibold"
      style={{
        color: style.color,
        backgroundColor: style.background,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ backgroundColor: style.color }}
      />
      {style.label}
    </span>
  );
}

export function WalletsView() {
  const {
    marketId,
    setMarketId,
    search,
    setSearch,
    selectedWalletId,
    selectWallet,
  } = usePolymap();

  const [kind, setKind] = useState<WalletKind | "">("");
  const [sortKey, setSortKey] = useState<SortKey>("volume");
  const [shown, setShown] = useState(PAGE);

  const rows = useMemo(() => {
    const query = search.trim().toLowerCase();

    const marketWalletIds = marketId
      ? new Set(
        pairs
          .filter((relationship) =>
            relationship.market_ids.includes(marketId)
          )
          .flatMap((relationship) => [
            relationship.wallet_a_id,
            relationship.wallet_b_id,
          ])
      )
      : null;

    return wallets
      .filter((wallet) => {
        const matchesType = !kind || wallet.wallet_type === kind;

        const matchesMarket =
          !marketWalletIds ||
          marketWalletIds.has(wallet.wallet_id);

        const matchesSearch =
          !query ||
          wallet.address.toLowerCase().includes(query) ||
          wallet.wallet_id.toLowerCase().includes(query);

        return matchesType && matchesMarket && matchesSearch;
      })
      .map((wallet) => ({
        wallet,
        matches: pairsForWallet(wallet.wallet_id).length,
      }))
      .sort((a, b) => {
        if (sortKey === "volume") {
          return (
            b.wallet.traded_volume_30d_usd -
            a.wallet.traded_volume_30d_usd
          );
        }

        if (sortKey === "trades") {
          return b.wallet.trades_30d - a.wallet.trades_30d;
        }

        return b.matches - a.matches;
      });
  }, [marketId, search, kind, sortKey]);

  const sortButton = (key: SortKey, label: string) => (
    <button
      type="button"
      onClick={() => setSortKey(key)}
      className={`inline-flex items-center gap-1.5 whitespace-nowrap transition-colors ${sortKey === key
        ? "font-semibold text-neutral-950"
        : "text-neutral-500 hover:text-neutral-900"
        }`}
    >
      {label}
      {sortKey === key ? (
        <ArrowDown className="h-3.5 w-3.5" />
      ) : (
        <ArrowUpDown className="h-3.5 w-3.5" />
      )}
    </button>
  );

  return (
    <div className="space-y-6">
      <PageHeader
        title="Wallets"
        description="Compare wallet activity, market participation and trading profiles."
      />

      <div className="flex flex-wrap items-center gap-3">
        <Select
          value={marketId ?? ""}
          onChange={(e) => {
            setMarketId(e.target.value || null);
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

        <Select
          value={kind}
          onChange={(e) => {
            setKind(e.target.value as WalletKind | "");
            setShown(PAGE);
          }}
          aria-label="Wallet type"
        >
          <option value="">All wallet types</option>
          {WALLET_TYPES.map((type) => (
            <option key={type} value={type}>
              {type}
            </option>
          ))}
        </Select>

        <div className="relative">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-neutral-400" />
          <Input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setShown(PAGE);
            }}
            placeholder="Search wallets..."
            aria-label="Search wallets"
            className="w-56 pl-9"
          />
        </div>

        <span className="ml-auto text-sm font-medium text-neutral-500">
          {num(rows.length)} wallets
        </span>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Card className="overflow-hidden">
          {rows.length === 0 ? (
            <EmptyState
              title="No wallets found"
              description="Try changing the market, wallet type, or search."
            />
          ) : (
            <>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="border-b border-neutral-200 bg-neutral-50/80 text-left text-xs text-neutral-500">
                    <tr>
                      <th className="px-4 py-3 font-semibold">Wallet</th>
                      <th className="px-4 py-3 font-semibold">Type</th>
                      <th className="px-4 py-3 text-right font-semibold">
                        {sortButton("volume", "Volume (30d)")}
                      </th>
                      <th className="px-4 py-3 text-right font-semibold">
                        {sortButton("trades", "Trades")}
                      </th>
                      <th className="px-4 py-3 text-right font-semibold">
                        Markets
                      </th>
                      <th className="px-4 py-3 text-right font-semibold">
                        {sortButton("matches", "Relationships")}
                      </th>
                    </tr>
                  </thead>

                  <tbody className="divide-y divide-neutral-100">
                    {rows.slice(0, shown).map(({ wallet, matches }) => {
                      const selected =
                        wallet.wallet_id === selectedWalletId;

                      return (
                        <tr
                          key={wallet.wallet_id}
                          onClick={() =>
                            selectWallet(wallet.wallet_id)
                          }
                          className={`cursor-pointer transition-colors ${selected
                            ? "bg-cyan-50/80"
                            : "hover:bg-orange-50/40"
                            }`}
                        >
                          <td className="whitespace-nowrap px-4 py-3.5">
                            <div className="flex items-center gap-2.5">
                              <div
                                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl"
                                style={{
                                  color:
                                    WALLET_STYLE[wallet.wallet_type].color,
                                  backgroundColor:
                                    WALLET_STYLE[wallet.wallet_type].background,
                                }}
                              >
                                <Wallet className="h-4 w-4" />
                              </div>

                              <div>
                                <div className="font-semibold text-neutral-900">
                                  {addr(wallet.address)}
                                </div>
                                <div className="text-[11px] text-neutral-400">
                                  {wallet.wallet_id}
                                </div>
                              </div>
                            </div>
                          </td>

                          <td className="whitespace-nowrap px-4 py-3.5">
                            <WalletTypeBadge kind={wallet.wallet_type} />
                          </td>

                          <td className="whitespace-nowrap px-4 py-3.5 text-right font-medium tabular-nums text-neutral-800">
                            {money(wallet.traded_volume_30d_usd)}
                          </td>

                          <td className="px-4 py-3.5 text-right tabular-nums text-neutral-700">
                            {num(wallet.trades_30d)}
                          </td>

                          <td className="px-4 py-3.5 text-right tabular-nums text-neutral-700">
                            {wallet.markets_traded}
                          </td>

                          <td className="px-4 py-3.5 text-right">
                            <span className="inline-flex min-w-8 justify-center rounded-lg bg-cyan-50 px-2 py-1 font-semibold tabular-nums text-cyan-800">
                              {matches}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>

              {shown < rows.length && (
                <div className="flex justify-center border-t border-neutral-100 p-4">
                  <button
                    type="button"
                    onClick={() =>
                      setShown((current) => current + PAGE)
                    }
                    className="rounded-xl border border-neutral-200 bg-white px-4 py-2 text-sm font-semibold text-neutral-700 transition-colors hover:border-cyan-300 hover:bg-cyan-50 hover:text-cyan-800"
                  >
                    Show {Math.min(PAGE, rows.length - shown)} more
                  </button>
                </div>
              )}
            </>
          )}
        </Card>

        <Card className="xl:sticky xl:top-20 xl:max-h-[calc(100vh-6rem)] xl:self-start xl:overflow-y-auto">
          <CardBody>
            {selectedWalletId ? (
              <WalletPanel walletId={selectedWalletId} />
            ) : (
              <EmptyState
                title="Select a wallet"
                description="Choose a wallet to inspect its profile and observed relationships."
              />
            )}
          </CardBody>
        </Card>
      </div>
    </div>
  );
}