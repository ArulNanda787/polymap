// TS mirrors of polymap/api/schemas.py. Kept hand-written rather than
// generated because the API surface is small and stable, and a codegen step
// would be more moving parts than it saves.

export type Stage = "fetch" | "build" | "train" | "screen" | "all";
export type JobStatus = "queued" | "running" | "done" | "error";
export type Sampler = "random" | "hard" | "popularity";
export type Category =
    | "persistent_tight"
    | "persistent_loose"
    | "one_off_tight"
    | "one_off_loose";

export interface Market {
    market_id: string | null;
    gamma_id: string;
    slug: string | null;
    question: string | null;
    volume: number;
    resolved: boolean;
    closed: boolean;
    end_date: string | null;
}

export interface RunSummary {
    run_name: string;
    path: string;
    has_trades: boolean;
    has_features: boolean;
    has_edges: boolean;
    has_events: boolean;
    has_screening: boolean;
    has_metrics: boolean;
}

export interface JobView {
    id: string;
    stage: Stage;
    status: JobStatus;
    run_name: string;
    started_at: string | null;
    finished_at: string | null;
    logs: string[];
    error: string | null;
    result: Record<string, unknown> | null;
}

export interface JobCreated {
    job_id: string;
}

export interface JobRequest {
    config_paths?: string[];
    overrides?: Record<string, unknown>;
    refresh?: boolean;
    negative?: Sampler;
}

export interface Metrics {
    roc_auc: number;
    average_precision: number;
    mrr: number;
    mean_rank: number;
    "hits@1": number;
    "hits@5": number;
    "hits@10": number;
    below_chance?: boolean;
    note?: string;
}

export type MetricsBySampler = Record<Sampler, Metrics>;

export interface ScreeningRow {
    pair: string;
    wallet_a: string;
    wallet_b: string;
    n_events: number;
    mean_margin: number;
    std_margin: number;
    mean_positive_score: number;
    first_event: number;
    last_event: number;
    real_events: number;
    distinct_timestamps: number;
    span_seconds: number;
    span_days: number;
    pct_same_side: number;
    pct_same_outcome: number;
    pct_same_behavior: number;
    median_time_diff: number;
    median_price_diff: number;
    median_size_ratio: number;
    persistent: boolean;
    tight: boolean;
    category: Category;
}


export interface WalletFeature {
    wallet: string;
    market_id?: string;
    trade_count: number;
    buy_count: number;
    sell_count: number;
    avg_trade_size: number;
    median_trade_size: number;
    total_size: number;
    avg_price: number;
    price_std: number;
    unique_outcomes: number;
    buy_ratio: number;
    sell_ratio: number;
    first_trade: number;
    last_trade: number;
    active_duration_hours: number;
    trade_frequency: number;
    log_trade_count: number;
    log_avg_trade_size: number;
    log_median_trade_size: number;
    log_active_duration: number;
}

export interface MapNode {
    id: string;
    wallet: string;
    radius: number;
    color: string;
    trade_count: number;
    buy_ratio: number;
    avg_trade_size: number;
    active_duration_hours: number;
    inCategory: boolean;
    x?: number;
    y?: number;
    vx?: number;
    vy?: number;
    fx?: number | null;
    fy?: number | null;
}

export interface MapLink {
    id: string;
    source: string | MapNode;
    target: string | MapNode;
    category: Category;
    n_events: number;
    mean_margin: number;
    median_time_diff: number;
    same_outcome_pct: number;
}