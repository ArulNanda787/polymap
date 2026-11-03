
export type WalletKind =
  | "whale"
  | "frequent"
  | "occasional"
  | "new";

export type RelationshipCategory =
  | "persistent_loose"
  | "one_off_loose"
  | "persistent_tight";

export type RelationshipFilter =
  | "all"
  | "persistent"
  | "one_off"
  | "tight";

export type WalletType =
  | "Whale"
  | "Frequent trader"
  | "Occasional trader"
  | "New wallet";

export type RelationshipStatus =
  | "available"
  | "coming_next"
  | "partial";

export interface PolymapMeta {
  product: string;
  dataset_id: string;
  dataset_name: string;
  environment: string;
  is_mock: boolean;
  schema_version: string;
  generated_for: string;
  warning: string;
  currency: string;
  markets_count: number;
  wallets_count: number;
  candidate_pairs_count: number;
  cross_market_pairs_count: number;

  // UI-friendly alias populated by the data adapter.
  note: string;
}

export interface Investigation {
  id: string;
  name: string;
  selected_market_id: string;
  selected_market_name: string;
  time_window: string;
  status: string;
}

export interface OverviewStats {
  markets_scanned: number;
  trades_fetched: number;
  unique_wallets: number;
  temporal_edges: number;
  candidate_pairs: number;
  cross_market_pairs: number;
  tight_pairs: number;
  persistent_pairs: number;
  matched_wallets: number;
  total_market_traders_selected: number;
  matched_wallet_share_selected_market: number;
  matched_volume_share_range: [number, number];

  category_counts: {
    persistent_loose: number;
    one_off_loose: number;
    persistent_tight: number;
  };
}

export interface PipelineStep {
  step: number;
  name: string;
  count: number;
  unit: string;
  description: string;
}

export interface PipelineDefinitions {
  temporal_edge: string;
  candidate_pair: string;
  tight: string;
  persistent: string;
  model_score: string;
}

export interface BackendReference {
  run: string;
  trades: number;
  wallets: number;
  temporal_edges: number;
  candidate_pairs: number;
  tight_pairs: number;
  persistent_pairs: number;
  note: string;
}

export interface Pipeline {
  title: string;
  subtitle: string;
  steps: PipelineStep[];
  definitions: PipelineDefinitions;
  actual_backend_reference: BackendReference;
}

export interface Market {
  market_id: string;
  name: string;
  category: string;
  total_volume_usd: number;
  total_traders: number;
  matched_pair_volume_share: number;
  matched_pair_volume_usd: number;
  wallets_in_relationships: number;
  candidate_pairs: number;
  matched_volume_note: string;
}

export interface Wallet {
  wallet_id: string;
  address: string;
  wallet_type: WalletType;
  traded_volume_30d_usd: number;
  trades_30d: number;
  markets_traded: number;
  yes_share: number;
  average_entry_price: number;
  median_trade_size_usd: number;
  relative_size_index: number;
  activity_label: string;
}

export interface CandidateRelationship {
  pair_id: string;
  wallet_a_id: string;
  wallet_b_id: string;
  market_ids: string[];
  market_count: number;
  events: number;
  median_time_gap_seconds: number;
  median_price_difference: number;
  same_side_rate: number;
  span_days: number;
  matched_volume_usd: number;
  relationship_strength: number;
  is_tight: boolean;
  is_persistent: boolean;
  category: RelationshipCategory;
  evidence: string[];
}

export interface RelationshipType {
  id: string;
  label: string;
  status: RelationshipStatus;
  description: string;
}

export interface NetworkLegendItem {
  type: WalletType;
  colour: string;
}

export interface RecommendedPageHeader {
  title: string;
  purpose: string;
  next_action: string;
}

export interface PolymapUI {
  navigation: string[];
  relationship_types: RelationshipType[];
  network_legend: NetworkLegendItem[];
  recommended_page_headers: {
    overview: RecommendedPageHeader;
    markets: RecommendedPageHeader;
    network: RecommendedPageHeader;
    pairs: RecommendedPageHeader;
    wallets: RecommendedPageHeader;
    charts: RecommendedPageHeader;
  };
}

export interface PolymapData {
  meta: PolymapMeta;
  investigation: Investigation;
  overview: OverviewStats;
  pipeline: Pipeline;
  markets: Market[];
  wallets: Wallet[];
  candidate_relationships: CandidateRelationship[];
  ui: PolymapUI;
}
