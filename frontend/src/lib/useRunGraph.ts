import { useEffect, useState } from "react";
import { api } from "./api";
import { buildGraph, type Graph } from "./adapter";
import type { Market } from "./types";

export type OpenState =
    | { phase: "idle" }
    | { phase: "running"; logs: string[] }
    | { phase: "error"; message: string; logs: string[] }
    | { phase: "ready"; run: string; graph: Graph };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const runNameFor = (m: Market) => (m.slug ?? m.gamma_id).replace(/[^a-zA-Z0-9_-]/g, "_");

/** Open a market: reuse an existing run if it has screening output, else run the pipeline. */
export function useOpenMarket(market: Market | null): OpenState {
    const [state, setState] = useState<OpenState>({ phase: "idle" });

    useEffect(() => {
        if (!market) return;
        let live = true;
        const logsRef: string[] = [];
        setState({ phase: "running", logs: [] });

        (async () => {
            const run = runNameFor(market);
            const existing = (await api.runs()).find((r) => r.run_name === run);
            if (!existing?.has_screening) {
                const { job_id } = await api.startAll({
                    config_paths: ["configs/default.yaml"],
                    overrides: { run_name: run, markets: [market.market_id] },
                });
                for (;;) {
                    await sleep(2000);
                    if (!live) return;
                    const j = await api.job(job_id);
                    logsRef.splice(0, logsRef.length, ...j.logs);
                    setState({ phase: "running", logs: [...logsRef] });
                    if (j.status === "error") throw new Error(j.error ?? "pipeline failed");
                    if (j.status === "done") break;
                }
            }
            const [wallets, rows] = await Promise.all([
                api.wallets(run),
                api.screening(run, { limit: 2000 }),
            ]);
            if (live) setState({ phase: "ready", run, graph: buildGraph(wallets, rows) });
        })().catch((e) => live && setState({ phase: "error", message: String(e?.message ?? e), logs: logsRef }));

        return () => { live = false; };
    }, [market]);

    return state;
}

export function useMarkets() {
    const [markets, setMarkets] = useState<Market[]>([]);
    const [error, setError] = useState<string | null>(null);
    useEffect(() => {
        api.markets({ min_volume: 100000, limit: 40 }).then(setMarkets).catch((e) => setError(String(e)));
    }, []);
    return { markets, error };
}