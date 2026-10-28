export function shortWallet(addr: string, chars = 6): string {
    if (addr.length <= chars * 2 + 2) return addr;
    return `${addr.slice(0, chars + 2)}…${addr.slice(-chars)}`;
}

export function fmtVolume(v: number): string {
    if (v >= 1_000_000) return `$${(v / 1_000_000).toFixed(2)}M`;
    if (v >= 1_000) return `$${(v / 1_000).toFixed(1)}K`;
    return `$${v.toFixed(0)}`;
}

export function fmtNum(v: number, digits = 3): string {
    if (Number.isNaN(v)) return "—";
    return v.toFixed(digits);
}

export function fmtPercent(v: number): string {
    if (Number.isNaN(v)) return "—";
    return `${(v * 100).toFixed(1)}%`;
}

export function fmtUnix(ts: number | null | undefined): string {
    if (!ts) return "—";
    const d = new Date(ts * 1000);
    return d.toISOString().replace("T", " ").slice(0, 16);
}

export function fmtDays(days: number): string {
    if (days >= 1) return `${days.toFixed(1)}d`;
    const hrs = days * 24;
    if (hrs >= 1) return `${hrs.toFixed(1)}h`;
    return `${(hrs * 60).toFixed(0)}m`;
}

export function fmtSeconds(s: number): string {
    if (s >= 3600) return `${(s / 3600).toFixed(1)}h`;
    if (s >= 60) return `${(s / 60).toFixed(1)}m`;
    return `${s.toFixed(1)}s`;
}

export function timeAgo(iso: string | null): string {
    if (!iso) return "—";
    const then = new Date(iso).getTime();
    const now = Date.now();
    const s = Math.max(0, Math.floor((now - then) / 1000));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.floor(s / 60)}m ago`;
    if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    return `${Math.floor(s / 86400)}d ago`;
}

export const CATEGORY_COLOR: Record<string, string> = {
    persistent_tight: "bg-red-500/15 text-red-400 border-red-500/30",
    persistent_loose: "bg-amber-500/15 text-amber-400 border-amber-500/30",
    one_off_tight: "bg-blue-500/15 text-blue-400 border-blue-500/30",
    one_off_loose: "bg-zinc-500/15 text-zinc-400 border-zinc-500/30",
};