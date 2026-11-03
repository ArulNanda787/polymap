
import raw from "@/data/polymap_demo.json";

import type {
  CandidateRelationship,
  Market,
  OverviewStats,
  Pipeline,
  PolymapData,
  PolymapMeta,
  Wallet,
  WalletKind,
  WalletType,
} from "@/lib/polymap/types";

/* --------------------------------------------------
   1. Raw dataset and normalized data
-------------------------------------------------- */

type RawPolymapData = Omit<PolymapData, "meta"> & {
  meta: Omit<PolymapMeta, "note">;
};

const source = raw as unknown as RawPolymapData;

export const data: PolymapData = {
  ...source,
  meta: {
    ...source.meta,
    note: source.meta.warning,
  },
};

export const meta = data.meta;
export const investigation = data.investigation;
export const overview = data.overview;
export const pipeline: Pipeline = data.pipeline;

export const markets: Market[] = data.markets;
export const wallets: Wallet[] = data.wallets;
export const pairs: CandidateRelationship[] =
  data.candidate_relationships;

export const candidateRelationships = pairs;
export const relationshipTypes = data.ui.relationship_types;
export const networkLegend = data.ui.network_legend;
export const navigation = data.ui.navigation;
export const pageHeaders = data.ui.recommended_page_headers;

/* --------------------------------------------------
   2. Lookup maps
-------------------------------------------------- */

export const marketById = new Map(
  markets.map((market) => [market.market_id, market])
);

export const walletById = new Map(
  wallets.map((wallet) => [wallet.wallet_id, wallet])
);

export const relationshipById = new Map(
  pairs.map((pair) => [pair.pair_id, pair])
);

/* --------------------------------------------------
   3. Wallet categories and colours
-------------------------------------------------- */

export const KIND: Record<
  WalletKind,
  {
    label: WalletType;
    color: string;
    background: string;
  }
> = {
  whale: {
    label: "Whale",
    color: "#7C3AED",
    background: "#F3E8FF",
  },
  frequent: {
    label: "Frequent trader",
    color: "#2563EB",
    background: "#DBEAFE",
  },
  occasional: {
    label: "Occasional trader",
    color: "#16A34A",
    background: "#DCFCE7",
  },
  new: {
    label: "New wallet",
    color: "#EA580C",
    background: "#FFEDD5",
  },
};

export const KIND_ORDER: WalletKind[] = [
  "whale",
  "frequent",
  "occasional",
  "new",
];

export function walletKind(walletType: WalletType): WalletKind {
  switch (walletType) {
    case "Whale":
      return "whale";
    case "Frequent trader":
      return "frequent";
    case "Occasional trader":
      return "occasional";
    case "New wallet":
      return "new";
  }
}

/* --------------------------------------------------
   4. Relationship categories
-------------------------------------------------- */

export const CATEGORY = {
  persistent_loose: {
    label: "Persistent",
    description: "Repeated temporal activity over time",
    color: "#0D9488",
    background: "#CCFBF1",
  },
  one_off_loose: {
    label: "One-off",
    description: "A limited set of temporal matches",
    color: "#D97706",
    background: "#FEF3C7",
  },
  persistent_tight: {
    label: "Persistent tight",
    description: "Persistent activity with tight timing and pricing",
    color: "#DB2777",
    background: "#FCE7F3",
  },
} as const;

export const CATEGORY_ORDER = [
  "persistent_tight",
  "persistent_loose",
  "one_off_loose",
] as const;

/* --------------------------------------------------
   5. Formatting helpers
-------------------------------------------------- */

export function num(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";

  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: 0,
  }).format(value);
}

export function money(
  value: number | null | undefined,
  compact = false
): string {
  if (value == null || !Number.isFinite(value)) return "—";

  if (compact) {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      notation: "compact",
      maximumFractionDigits: 1,
    }).format(value);
  }

  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function pct(
  value: number | null | undefined,
  digits = 1
): string {
  if (value == null || !Number.isFinite(value)) return "—";

  return `${(value * 100).toFixed(digits)}%`;
}

export function fmtPrice(
  value: number | null | undefined,
  digits = 3
): string {
  if (value == null || !Number.isFinite(value)) return "—";

  return value.toFixed(digits);
}

export function cents(
  value: number | null | undefined
): string {
  if (value == null || !Number.isFinite(value)) return "—";

  return `${(value * 100).toFixed(1)}¢`;
}

export function addr(
  address: string | null | undefined,
  chars = 6
): string {
  if (!address) return "Unknown";

  if (address.length <= chars * 2 + 3) {
    return address;
  }

  return `${address.slice(0, chars + 2)}…${address.slice(-chars)}`;
}

export function fmtDate(value: string | Date): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return "—";

  return new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    month: "short",
    year: "numeric",
  }).format(date);
}

export function fmtDateTime(value: string | Date): string {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) return "—";

  return new Intl.DateTimeFormat("en-US", {
    day: "numeric",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function secondsText(
  value: number | null | undefined
): string {
  if (value == null || !Number.isFinite(value)) return "—";

  if (value < 60) return `${value.toFixed(1)}s`;

  if (value < 3600) {
    return `${(value / 60).toFixed(1)}m`;
  }

  return `${(value / 3600).toFixed(1)}h`;
}

/* --------------------------------------------------
   6. Wallet helpers
-------------------------------------------------- */

export function relationshipsForWallet(
  walletId: string
): CandidateRelationship[] {
  return pairs.filter(
    (pair) =>
      pair.wallet_a_id === walletId ||
      pair.wallet_b_id === walletId
  );
}

export function pairsForWallet(
  walletId: string
): CandidateRelationship[] {
  return relationshipsForWallet(walletId);
}

export function otherWalletId(
  pair: CandidateRelationship,
  walletId: string
): string | undefined {
  if (pair.wallet_a_id === walletId) {
    return pair.wallet_b_id;
  }

  if (pair.wallet_b_id === walletId) {
    return pair.wallet_a_id;
  }

  return undefined;
}

export function pairName(
  pair: CandidateRelationship,
  abbreviated = true
): string {
  const walletA = walletById.get(pair.wallet_a_id);
  const walletB = walletById.get(pair.wallet_b_id);

  if (!walletA || !walletB) {
    return `${pair.wallet_a_id} ↔ ${pair.wallet_b_id}`;
  }

  if (!abbreviated) {
    return `${walletA.address} ↔ ${walletB.address}`;
  }

  return `${addr(walletA.address)} ↔ ${addr(walletB.address)}`;
}

/* --------------------------------------------------
   7. Market relationship helpers
-------------------------------------------------- */

export function relationshipsForMarket(
  marketId: string
): CandidateRelationship[] {
  return pairs.filter((pair) =>
    pair.market_ids.includes(marketId)
  );
}

export function walletsForMarket(
  marketId: string
): Wallet[] {
  const walletIds = new Set(
    relationshipsForMarket(marketId).flatMap((pair) => [
      pair.wallet_a_id,
      pair.wallet_b_id,
    ])
  );

  return wallets.filter((wallet) =>
    walletIds.has(wallet.wallet_id)
  );
}

/*
  Keep source-reported market aggregates separate from
  counts derived from individual relationship records.

  This is intentional: the dataset contains aggregate and
  relationship-level figures that do not always reconcile.
*/

export const marketStats = new Map(
  markets.map((market) => {
    const relationships = relationshipsForMarket(
      market.market_id
    );

    const walletIds = new Set(
      relationships.flatMap((pair) => [
        pair.wallet_a_id,
        pair.wallet_b_id,
      ])
    );

    return [
      market.market_id,
      {
        reportedCandidatePairs: market.candidate_pairs,
        reportedWalletsInRelationships:
          market.wallets_in_relationships,

        observedRelationshipRecords: relationships.length,
        observedDistinctWallets: walletIds.size,

        matchedVolumeUsd: market.matched_pair_volume_usd,
        matchedVolumeShare: market.matched_pair_volume_share,
        totalVolumeUsd: market.total_volume_usd,
        totalTraders: market.total_traders,
      },
    ];
  })
);

/* --------------------------------------------------
   8. Overview totals
-------------------------------------------------- */

export const totals = {
  markets: overview.markets_scanned,
  trades: overview.trades_fetched,
  wallets: overview.unique_wallets,
  temporalEdges: overview.temporal_edges,
  candidatePairs: overview.candidate_pairs,
  crossMarketPairs: overview.cross_market_pairs,
  tightPairs: overview.tight_pairs,
  persistentPairs: overview.persistent_pairs,
  matchedWallets: overview.matched_wallets,
  selectedMarketTraders:
    overview.total_market_traders_selected,
};

/* --------------------------------------------------
   9. Relationship filters
-------------------------------------------------- */

export function filterRelationships(
  relationships: CandidateRelationship[],
  options: {
    marketId?: string | null;
    category?: string | null;
    search?: string;
    persistentOnly?: boolean;
    tightOnly?: boolean;
  } = {}
): CandidateRelationship[] {
  const {
    marketId,
    category,
    search,
    persistentOnly,
    tightOnly,
  } = options;

  const query = search?.trim().toLowerCase();

  return relationships.filter((pair) => {
    if (
      marketId &&
      !pair.market_ids.includes(marketId)
    ) {
      return false;
    }

    if (category && pair.category !== category) {
      return false;
    }

    if (persistentOnly && !pair.is_persistent) {
      return false;
    }

    if (tightOnly && !pair.is_tight) {
      return false;
    }

    if (query) {
      const walletA = walletById.get(pair.wallet_a_id);
      const walletB = walletById.get(pair.wallet_b_id);

      const searchable = [
        pair.pair_id,
        pair.wallet_a_id,
        pair.wallet_b_id,
        walletA?.address,
        walletB?.address,
        ...pair.market_ids,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      if (!searchable.includes(query)) {
        return false;
      }
    }

    return true;
  });
}

/* --------------------------------------------------
   10. Data integrity helpers
-------------------------------------------------- */

export const datasetStatus = {
  isMock: meta.is_mock,
  environment: meta.environment,
  warning: meta.warning,
};

export function isValidMarketId(
  marketId: string
): boolean {
  return marketById.has(marketId);
}

export function isValidWalletId(
  walletId: string
): boolean {
  return walletById.has(walletId);
}

export function getMarket(
  marketId: string
): Market | undefined {
  return marketById.get(marketId);
}

export function getWallet(
  walletId: string
): Wallet | undefined {
  return walletById.get(walletId);
}

export function getRelationship(
  pairId: string
): CandidateRelationship | undefined {
  return relationshipById.get(pairId);
}
