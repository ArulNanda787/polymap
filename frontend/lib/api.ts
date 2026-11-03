import type {
    JobCreated,
    JobRequest,
    JobView,
    Market,
    MetricsBySampler,
    RunSummary,
    ScreeningRow,
} from "./types";

const BASE =
    process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8000";

class ApiError extends Error {
    constructor(message: string, public status: number) {
        super(message);
        this.name = "ApiError";
    }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await fetch(`${BASE}${path}`, {
        ...init,
        headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
    });
    if (!res.ok) {
        const body = await res.text().catch(() => "");
        throw new ApiError(
            `${res.status} ${res.statusText}${body ? `: ${body.slice(0, 300)}` : ""}`,
            res.status,
        );
    }
    return (await res.json()) as T;
}

export const api = {
    // --- read --------------------------------------------------------------
    health: () => req<{ ok: boolean }>("/health"),

    markets: (params?: { min_volume?: number; closed?: boolean; limit?: number }) => {
        const q = new URLSearchParams();
        if (params?.min_volume !== undefined) q.set("min_volume", String(params.min_volume));
        if (params?.closed !== undefined) q.set("closed", String(params.closed));
        if (params?.limit !== undefined) q.set("limit", String(params.limit));
        const s = q.toString();
        return req<Market[]>(`/markets${s ? `?${s}` : ""}`);
    },

    runs: () => req<RunSummary[]>("/runs"),

    metrics: (runName: string) =>
        req<MetricsBySampler>(`/runs/${encodeURIComponent(runName)}/metrics`),

    screening: (
        runName: string,
        params?: { category?: string; limit?: number },
    ) => {
        const q = new URLSearchParams();
        if (params?.category) q.set("category", params.category);
        if (params?.limit !== undefined) q.set("limit", String(params.limit));
        const s = q.toString();
        return req<ScreeningRow[]>(
            `/runs/${encodeURIComponent(runName)}/screening${s ? `?${s}` : ""}`,
        );
    },
    wallets: (runName: string) =>
        req<import("./types").WalletFeature[]>(
            `/runs/${encodeURIComponent(runName)}/wallets`,
        ),

    edges: (runName: string, limit = 200) =>
        req<Record<string, unknown>[]>(
            `/runs/${encodeURIComponent(runName)}/edges?limit=${limit}`,
        ),

    runConfig: (runName: string) =>
        req<{ config: Record<string, unknown>; fingerprint: string }>(
            `/runs/${encodeURIComponent(runName)}/config`,
        ),

    jobs: () => req<JobView[]>("/jobs"),
    job: (id: string) => req<JobView>(`/jobs/${id}`),

    // --- jobs --------------------------------------------------------------
    startFetch: (body: JobRequest) =>
        req<JobCreated>("/jobs/fetch", { method: "POST", body: JSON.stringify(body) }),
    startBuild: (body: JobRequest) =>
        req<JobCreated>("/jobs/build", { method: "POST", body: JSON.stringify(body) }),
    startTrain: (body: JobRequest) =>
        req<JobCreated>("/jobs/train", { method: "POST", body: JSON.stringify(body) }),
    startScreen: (body: JobRequest) =>
        req<JobCreated>("/jobs/screen", { method: "POST", body: JSON.stringify(body) }),
    startAll: (body: JobRequest) =>
        req<JobCreated>("/jobs/all", { method: "POST", body: JSON.stringify(body) }),
};

export { ApiError };