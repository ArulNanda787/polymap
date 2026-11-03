"use client";

import { Check, X, ExternalLink } from "lucide-react";

import {
  relationshipById,
  walletById,
  marketById,
  pairsForWallet,
  money,
  addr,
  pct,
  secondsText,
  pairName,
} from "@/lib/polymap/data";

import { usePolymap } from "./store";
import { Button, Stat } from "./ui";

import type {
  CandidateRelationship,
  Wallet,
  WalletType,
} from "@/lib/polymap/types";

/* --------------------------------------------------
   Wallet styling
-------------------------------------------------- */

const WALLET_STYLE: Record<
  WalletType,
  { label: string; color: string; background: string }
> = {
  Whale: {
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

const CATEGORY_STYLE: Record<
  string,
  { label: string; color: string; background: string }
> = {
  persistent_tight: {
    label: "Persistent · Tight",
    color: "#047857",
    background: "#D1FAE5",
  },
  persistent_loose: {
    label: "Persistent · Loose",
    color: "#2563EB",
    background: "#DBEAFE",
  },
  one_off_loose: {
    label: "One-off · Loose",
    color: "#C2410C",
    background: "#FFEDD5",
  },
};

function WalletTypeBadge({ kind }: { kind: WalletType }) {
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

function CategoryBadge({ category }: { category: string }) {
  const style = CATEGORY_STYLE[category] ?? {
    label: category.replaceAll("_", " "),
    color: "#525252",
    background: "#F5F5F5",
  };

  return (
    <span
      className="inline-flex rounded-full px-2.5 py-1 text-xs font-semibold capitalize"
      style={{
        color: style.color,
        backgroundColor: style.background,
      }}
    >
      {style.label}
    </span>
  );
}

/* --------------------------------------------------
   Shared wallet information
-------------------------------------------------- */

function WalletSummary({
  wallet,
  onClick,
}: {
  wallet: Wallet;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="flex w-full items-center justify-between gap-3 px-3 py-3 text-left transition-colors hover:bg-cyan-50/60"
    >
      <span className="min-w-0">
        <span className="block truncate text-sm font-semibold text-neutral-900">
          {addr(wallet.address)}
        </span>

        <span className="mt-1 block text-xs text-neutral-500">
          {wallet.wallet_id} · {money(wallet.traded_volume_30d_usd)} traded
        </span>
      </span>

      <ExternalLink className="h-4 w-4 shrink-0 text-neutral-400" />
    </button>
  );
}

/* --------------------------------------------------
   Pair / relationship panel
-------------------------------------------------- */

export function PairPanel({ pairId }: { pairId: string }) {
  const {
    selectPair,
    selectWallet,
    setView,
    setMarketId,
  } = usePolymap();

  const pair = relationshipById.get(pairId);

  if (!pair) return null;

  const walletA = walletById.get(pair.wallet_a_id);
  const walletB = walletById.get(pair.wallet_b_id);

  if (!walletA || !walletB) return null;

  const firstMarketId = pair.market_ids[0];

  const firstMarket = firstMarketId
    ? marketById.get(firstMarketId)
    : undefined;

  const checks = [
    {
      label: "Median time gap is 5 seconds or less",
      pass: pair.median_time_gap_seconds <= 5,
      value: secondsText(pair.median_time_gap_seconds),
    },
    {
      label: "Median price difference is 1¢ or less",
      pass: pair.median_price_difference <= 0.01,
      value: `${(pair.median_price_difference * 100).toFixed(2)}¢`,
    },
    {
      label: "Observed over more than 7 days",
      pass: pair.span_days > 7,
      value: `${pair.span_days.toFixed(1)} days`,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-neutral-950">
              {pairName(pair)}
            </h3>

            <p className="mt-1 text-xs text-neutral-500">
              {pair.pair_id}
            </p>
          </div>

          <Button
            variant="ghost"
            size="icon"
            aria-label="Close"
            onClick={() => selectPair(null)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <CategoryBadge category={pair.category} />

          {pair.market_count > 1 && (
            <span className="rounded-full bg-purple-50 px-2.5 py-1 text-xs font-semibold text-purple-700">
              {pair.market_count} markets
            </span>
          )}
        </div>

        <p className="mt-3 text-sm leading-6 text-neutral-600">
          These wallets have {pair.events} observed temporally close trade
          events across {pair.market_count} market
          {pair.market_count === 1 ? "" : "s"}. This is a candidate
          relationship, not proof of coordination.
        </p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <Stat
          label="Observed events"
          value={pair.events.toLocaleString("en-US")}
        />

        <Stat
          label="Matched volume"
          value={money(pair.matched_volume_usd)}
        />

        <Stat
          label="Median time gap"
          value={secondsText(pair.median_time_gap_seconds)}
        />

        <Stat
          label="Same-side rate"
          value={pct(pair.same_side_rate)}
        />

        <Stat
          label="Price difference"
          value={`${(pair.median_price_difference * 100).toFixed(2)}¢`}
        />

        <Stat
          label="Observed span"
          value={`${pair.span_days.toFixed(1)} days`}
        />
      </div>

      {/* Relationship flags */}
      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          Relationship signals
        </h4>

        <div className="mt-2 space-y-2 rounded-xl border border-neutral-200 p-3">
          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="text-neutral-600">Tight relationship</span>

            <span
              className={
                pair.is_tight
                  ? "font-semibold text-emerald-700"
                  : "font-medium text-neutral-500"
              }
            >
              {pair.is_tight ? "Yes" : "No"}
            </span>
          </div>

          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="text-neutral-600">
              Persistent relationship
            </span>

            <span
              className={
                pair.is_persistent
                  ? "font-semibold text-emerald-700"
                  : "font-medium text-neutral-500"
              }
            >
              {pair.is_persistent ? "Yes" : "No"}
            </span>
          </div>
        </div>
      </div>

      {/* Threshold checks */}
      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          Threshold checks
        </h4>

        <ul className="mt-2 divide-y divide-neutral-100 rounded-xl border border-neutral-200">
          {checks.map((check) => (
            <li
              key={check.label}
              className="flex items-center justify-between gap-3 px-3 py-3 text-sm"
            >
              <span className="flex min-w-0 items-center gap-2 text-neutral-700">
                {check.pass ? (
                  <Check className="h-4 w-4 shrink-0 text-emerald-600" />
                ) : (
                  <X className="h-4 w-4 shrink-0 text-neutral-400" />
                )}

                {check.label}
              </span>

              <span className="shrink-0 font-semibold tabular-nums text-neutral-900">
                {check.value}
              </span>
            </li>
          ))}
        </ul>
      </div>

      {/* Wallets */}
      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          The two wallets
        </h4>

        <div className="mt-2 divide-y divide-neutral-100 overflow-hidden rounded-xl border border-neutral-200">
          <WalletSummary
            wallet={walletA}
            onClick={() => selectWallet(walletA.wallet_id)}
          />

          <WalletSummary
            wallet={walletB}
            onClick={() => selectWallet(walletB.wallet_id)}
          />
        </div>
      </div>

      {/* Evidence */}
      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          Supporting evidence
        </h4>

        <ul className="mt-2 space-y-2">
          {pair.evidence.map((item, index) => (
            <li
              key={`${pair.pair_id}-evidence-${index}`}
              className="flex gap-2 text-sm leading-5 text-neutral-700"
            >
              <span className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-cyan-500" />
              {item}
            </li>
          ))}
        </ul>
      </div>

      {/* Markets */}
      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          Markets involved
        </h4>

        <div className="mt-2 flex flex-wrap gap-2">
          {pair.market_ids.map((id) => {
            const market = marketById.get(id);

            return (
              <button
                key={id}
                type="button"
                onClick={() => {
                  setMarketId(id);
                  setView("network");
                }}
                className="rounded-lg border border-neutral-200 bg-white px-3 py-2 text-xs font-medium text-neutral-700 transition-colors hover:border-cyan-300 hover:bg-cyan-50 hover:text-cyan-800"
              >
                {market?.name ?? id}
              </button>
            );
          })}
        </div>
      </div>

      {/* Caveat */}
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-3 text-sm leading-5 text-amber-900">
        Temporal proximity is a signal for investigation, not proof that
        wallets are controlled by the same person or are coordinating.
      </div>

      <Button
        size="sm"
        onClick={() => {
          if (firstMarketId) setMarketId(firstMarketId);
          setView("network");
        }}
      >
        Show in network
      </Button>
    </div>
  );
}

/* --------------------------------------------------
   Wallet panel
-------------------------------------------------- */

export function WalletPanel({ walletId }: { walletId: string }) {
  const {
    selectPair,
    selectWallet,
    setMarketId,
    setView,
  } = usePolymap();

  const wallet = walletById.get(walletId);

  if (!wallet) return null;

  const relationships = pairsForWallet(wallet.wallet_id).sort(
    (a, b) => b.relationship_strength - a.relationship_strength
  );

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <div className="flex items-start justify-between gap-2">
          <div className="min-w-0">
            <h3 className="break-all text-base font-semibold text-neutral-950">
              {wallet.address}
            </h3>

            <p className="mt-1 text-xs text-neutral-500">
              {wallet.wallet_id}
            </p>
          </div>

          <Button
            variant="ghost"
            size="icon"
            aria-label="Close"
            onClick={() => selectWallet(null)}
          >
            <X className="h-4 w-4" />
          </Button>
        </div>

        <div className="mt-3">
          <WalletTypeBadge kind={wallet.wallet_type} />
        </div>

        <p className="mt-2 text-sm text-neutral-600">
          {wallet.activity_label} activity profile
        </p>
      </div>

      {/* Metrics */}
      <div className="grid grid-cols-2 gap-4">
        <Stat
          label="Traded (30 days)"
          value={money(wallet.traded_volume_30d_usd)}
        />

        <Stat
          label="Trades (30 days)"
          value={wallet.trades_30d.toLocaleString("en-US")}
        />

        <Stat
          label="Markets traded"
          value={wallet.markets_traded}
        />

        <Stat
          label="Relationships"
          value={relationships.length}
        />

        <Stat
          label="Average entry price"
          value={`${(wallet.average_entry_price * 100).toFixed(1)}¢`}
        />

        <Stat
          label="Median trade size"
          value={money(wallet.median_trade_size_usd)}
        />
      </div>

      {/* Trading profile */}
      <div>
        <h4 className="text-sm font-semibold text-neutral-950">
          Trading profile
        </h4>

        <div className="mt-2 space-y-3 rounded-xl border border-neutral-200 p-3">
          <div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-neutral-600">YES share</span>

              <span className="font-semibold text-neutral-900">
                {pct(wallet.yes_share)}
              </span>
            </div>

            <div className="mt-2 h-2 overflow-hidden rounded-full bg-neutral-100">
              <div
                className="h-full rounded-full bg-cyan-500"
                style={{
                  width: `${Math.max(
                    0,
                    Math.min(100, wallet.yes_share * 100)
                  )}%`,
                }}
              />
            </div>
          </div>

          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="text-neutral-600">
              Relative size index
            </span>

            <span className="font-semibold text-neutral-900">
              {wallet.relative_size_index.toFixed(2)}
            </span>
          </div>

          <div className="flex items-center justify-between gap-3 text-sm">
            <span className="text-neutral-600">
              Activity pattern
            </span>

            <span className="font-semibold text-neutral-900">
              {wallet.activity_label}
            </span>
          </div>
        </div>
      </div>

      {/* Relationships */}
      <div>
        <div className="flex items-center justify-between gap-2">
          <h4 className="text-sm font-semibold text-neutral-950">
            Observed relationships
          </h4>

          <span className="text-xs text-neutral-500">
            {relationships.length} found
          </span>
        </div>

        {relationships.length === 0 ? (
          <p className="mt-2 rounded-xl border border-dashed border-neutral-300 p-4 text-sm text-neutral-500">
            No candidate relationships found for this wallet.
          </p>
        ) : (
          <ul className="mt-2 max-h-96 divide-y divide-neutral-100 overflow-y-auto rounded-xl border border-neutral-200">
            {relationships.map((relationship: CandidateRelationship) => {
              const otherId =
                relationship.wallet_a_id === wallet.wallet_id
                  ? relationship.wallet_b_id
                  : relationship.wallet_a_id;

              const otherWallet = walletById.get(otherId);

              return (
                <li key={relationship.pair_id}>
                  <button
                    type="button"
                    className="flex w-full items-center justify-between gap-3 px-3 py-3 text-left transition-colors hover:bg-cyan-50/60"
                    onClick={() => selectPair(relationship.pair_id)}
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-sm font-semibold text-neutral-900">
                        {otherWallet
                          ? addr(otherWallet.address)
                          : otherId}
                      </span>

                      <span className="mt-1 block truncate text-xs text-neutral-500">
                        {relationship.events} events ·{" "}
                        {relationship.market_count} markets ·{" "}
                        {money(relationship.matched_volume_usd)}
                      </span>

                      <span className="mt-1.5 block">
                        <CategoryBadge category={relationship.category} />
                      </span>
                    </span>

                    <span className="shrink-0 text-xs font-semibold tabular-nums text-neutral-500">
                      {Math.round(
                        relationship.relationship_strength * 100
                      )}
                    </span>
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      {/* Network action */}
      <Button
        size="sm"
        onClick={() => {
          const firstRelationship = relationships[0];
          const firstMarket = firstRelationship?.market_ids[0];

          if (firstMarket) setMarketId(firstMarket);

          setView("network");
        }}
      >
        Show in network
      </Button>
    </div>
  );
}

/* --------------------------------------------------
   Inspector wrapper
-------------------------------------------------- */

export function Inspector({
  fallback,
}: {
  fallback: React.ReactNode;
}) {
  const { selectedPairId, selectedWalletId } = usePolymap();

  if (selectedPairId) {
    return <PairPanel pairId={selectedPairId} />;
  }

  if (selectedWalletId) {
    return <WalletPanel walletId={selectedWalletId} />;
  }

  return <>{fallback}</>;
}