"use client";

import { useMemo } from "react";
import MetricCard, { MetricPoint } from "@/components/kokonutui/metric-card";

const TARGET_CATEGORIES = ["Crypto", "Economy", "Politics"];

// A simple seeded random generator so the chart doesn't jump around on re-renders
function mulberry32(a: number) {
    return function () {
        let t = (a += 0x6d2b79f5);
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export default function MarketVolumeCard({ className }: { className?: string }) {
    // useMemo ensures the data only generates once per mount
    const { data, events } = useMemo(() => {
        const random = mulberry32(42);
        const points: MetricPoint[] = [];
        const today = new Date("2026-10-30");

        // Generate 120 days of data (so we have plenty for 90D comparison)
        let baseCrypto = 500000;
        let baseEcon = 400000;
        let basePol = 300000;

        for (let i = 120; i >= 0; i--) {
            const d = new Date(today);
            d.setDate(d.getDate() - i);

            // Add some random walk and a little upward trend
            baseCrypto += (random() - 0.45) * 80000 + 2000;
            baseEcon += (random() - 0.48) * 60000 + 1500;
            basePol += (random() - 0.50) * 50000 + 1000;

            // Add a fake spike in the middle
            if (i === 40) {
                baseCrypto *= 1.4;
                baseEcon *= 1.2;
            }

            points.push({
                date: d.toISOString().split("T")[0],
                parts: [
                    Math.max(10000, Math.round(baseCrypto)),
                    Math.max(10000, Math.round(baseEcon)),
                    Math.max(10000, Math.round(basePol)),
                ],
            });
        }

        // Add some events to make the chart look interesting
        const generatedEvents = [
            { date: points[points.length - 45].date, label: "Regulatory News" },
            { date: points[points.length - 15].date, label: "Fed Rate Cut" },
        ];

        return { data: points, events: generatedEvents };
    }, []);

    return (
        <MetricCard
            label="Market Volume"
            segments={TARGET_CATEGORIES}
            data={data}
            events={events} // Pass the generated events here
            defaultPeriod={30}
            format="currency"
            currency="USD"
            className={`!max-w-none w-full lg:w-[520px] !p-6 border-teal-100 shadow-teal-900/5 shadow-xl ${className}`}
        />
    );
}