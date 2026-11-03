"use client";

import React, { useRef, useState, useEffect } from "react";
import {
    Wallet,
    Network,
    GitBranch,
    ArrowLeftRight,
    Layers,
    Activity,
    Search,
    ArrowLeft,
    ArrowRight,
    Share2,
} from "lucide-react";
import Link from "next/link";
import { PointerHighlight } from "@/components/pointer-highlight";

type Feature = {
    icon: React.ElementType;
    eyebrow: string;
    title: string;
    highlight: string;
    description: string;
    bg: string;
    accent: string;
    textColor: string;
    pillBg: string;
    pillText: string;
    iconColor: string;
};

const FEATURES: Feature[] = [
    {
        icon: Wallet,
        eyebrow: "Wallet Intelligence",
        title: "Turn wallet addresses",
        highlight: "into context",
        description:
            "Explore wallet activity, inspect transaction history, and bring useful context to addresses across a connected network.",
        bg: "#fed7aa",
        accent: "#9a3412",
        textColor: "#292018",
        pillBg: "rgba(154,52,18,0.12)",
        pillText: "#7c2d12",
        iconColor: "#7c2d12",
    },
    {
        icon: Network,
        eyebrow: "Relationship Mapping",
        title: "See how wallets",
        highlight: "connect",
        description:
            "Map relationships between wallets and entities to understand how activity moves through a transaction network.",
        bg: "#99f6e4",
        accent: "#115e59",
        textColor: "#102b29",
        pillBg: "rgba(17,94,89,0.12)",
        pillText: "#134e4a",
        iconColor: "#134e4a",
    },
    {
        icon: GitBranch,
        eyebrow: "Transaction Graphs",
        title: "Follow the links",
        highlight: "between transactions",
        description:
            "Move beyond isolated transaction records. Explore connected nodes and edges to investigate the structure of a wallet network.",
        bg: "#fbcfe8",
        accent: "#9d174d",
        textColor: "#321725",
        pillBg: "rgba(157,23,77,0.11)",
        pillText: "#831843",
        iconColor: "#831843",
    },
    {
        icon: ArrowLeftRight,
        eyebrow: "Fund Flow Analysis",
        title: "Trace where",
        highlight: "value moves",
        description:
            "Follow transaction paths between wallets and examine how funds move across connected addresses.",
        bg: "#fde68a",
        accent: "#92400e",
        textColor: "#302515",
        pillBg: "rgba(146,64,14,0.12)",
        pillText: "#78350f",
        iconColor: "#78350f",
    },
    {
        icon: Layers,
        eyebrow: "Wallet Clustering",
        title: "Find groups",
        highlight: "within the network",
        description:
            "Bring related wallets into view and examine clusters of connected activity instead of investigating every address in isolation.",
        bg: "#bae6fd",
        accent: "#075985",
        textColor: "#102b3b",
        pillBg: "rgba(7,89,133,0.12)",
        pillText: "#075985",
        iconColor: "#0c4a6e",
    },
    {
        icon: Activity,
        eyebrow: "Network Activity",
        title: "Spot patterns",
        highlight: "in wallet behaviour",
        description:
            "Study transaction activity and recurring connections to build a clearer picture of how a network behaves over time.",
        bg: "#a5f3fc",
        accent: "#155e75",
        textColor: "#102d35",
        pillBg: "rgba(21,94,117,0.12)",
        pillText: "#164e63",
        iconColor: "#164e63",
    },
    {
        icon: Search,
        eyebrow: "Network Exploration",
        title: "Investigate",
        highlight: "the bigger picture",
        description:
            "Start with a wallet, expand into its connected addresses, and explore the wider transaction graph from one place.",
        bg: "#bbf7d0",
        accent: "#14532d",
        textColor: "#14291c",
        pillBg: "rgba(20,83,45,0.14)",
        pillText: "#14532d",
        iconColor: "#14532d",
    },
];

function LedPill({
    icon: Icon,
    label,
    pillBg,
    pillText,
}: {
    icon: React.ElementType;
    label: string;
    pillBg: string;
    pillText: string;
}) {
    return (
        <div
            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[10px] font-semibold tracking-[0.16em] uppercase cursor-default select-none w-fit"
            style={{
                background: pillBg,
                color: pillText,
                border: `1px solid ${pillText}33`,
            }}
        >
            <Icon size={11} strokeWidth={2.5} />
            {label}
        </div>
    );
}

function FeatureCard({ feature }: { feature: Feature }) {
    const Icon = feature.icon;

    return (
        <div
            className="group relative overflow-hidden flex flex-col gap-6 p-7 sm:p-8 rounded-2xl cursor-default select-none transition-transform duration-300 ease-out hover:-translate-y-1 shrink-0"
            style={{ background: feature.bg, width: "340px", height: "420px" }}
        >
            <Icon
                aria-hidden
                size={220}
                strokeWidth={0.8}
                color={feature.iconColor}
                className="pointer-events-none absolute -right-8 -bottom-10 transition-opacity duration-500 group-hover:opacity-[0.55]"
                style={{ opacity: 0.34 }}
            />

            <div className="relative z-10">
                <LedPill
                    icon={feature.icon}
                    label={feature.eyebrow}
                    pillBg={feature.pillBg}
                    pillText={feature.pillText}
                />
            </div>

            <div className="relative z-10 flex flex-col gap-3">
                <h3
                    className="text-[24px] sm:text-[28px] font-medium tracking-[-0.03em] leading-[1.05]"
                    style={{ color: feature.textColor }}
                >
                    {feature.title}{" "}
                    <span style={{ color: feature.accent }}>
                        {feature.highlight}
                    </span>
                </h3>
                <p
                    className="text-[13px] leading-relaxed max-w-[85%]"
                    style={{ color: feature.textColor, opacity: 0.85 }}
                >
                    {feature.description}
                </p>
            </div>
        </div>
    );
}

function LedArrowButton({
    direction,
    onClick,
    disabled,
}: {
    direction: "left" | "right";
    onClick: () => void;
    disabled: boolean;
}) {
    const Icon = direction === "left" ? ArrowLeft : ArrowRight;
    const glowColor = "#4f46e5";

    return (
        <button
            onClick={onClick}
            disabled={disabled}
            aria-label={direction === "left" ? "Scroll left" : "Scroll right"}
            className="relative flex h-12 w-12 items-center justify-center rounded-full transition-all duration-300 hover:scale-105 disabled:opacity-30 disabled:hover:scale-100"
            style={{
                background: disabled ? "#f4f4f5" : "#ffffff",
                border: disabled
                    ? "1px solid rgba(24,24,27,0.10)"
                    : `1px solid ${glowColor}66`,
                boxShadow: disabled
                    ? "none"
                    : `0 0 20px ${glowColor}18, 0 0 60px ${glowColor}08, inset 0 0 20px ${glowColor}08`,
            }}
        >
            {!disabled && (
                <>
                    <div
                        className="absolute inset-0 rounded-full animate-pulse"
                        style={{
                            boxShadow: `0 0 30px ${glowColor}22, 0 0 60px ${glowColor}10`,
                        }}
                    />
                    <div
                        className="absolute -inset-1 rounded-full opacity-50"
                        style={{
                            background: `radial-gradient(circle at center, ${glowColor}33, transparent 70%)`,
                        }}
                    />
                </>
            )}

            <Icon
                size={20}
                strokeWidth={2.5}
                className="relative z-10"
                style={{
                    color: disabled ? "rgba(24,24,27,0.3)" : "#27272a",
                }}
            />
        </button>
    );
}

export default function PolymapFeatures() {
    const carouselRef = useRef<HTMLDivElement>(null);
    const [canScrollLeft, setCanScrollLeft] = useState(false);
    const [canScrollRight, setCanScrollRight] = useState(true);

    const checkScrollability = () => {
        if (!carouselRef.current) return;
        const { scrollLeft, scrollWidth, clientWidth } = carouselRef.current;
        setCanScrollLeft(scrollLeft > 0);
        setCanScrollRight(scrollLeft < scrollWidth - clientWidth - 1);
    };

    useEffect(() => {
        checkScrollability();
        window.addEventListener("resize", checkScrollability);
        return () => window.removeEventListener("resize", checkScrollability);
    }, []);

    const scroll = (dir: "left" | "right") => {
        carouselRef.current?.scrollBy({
            left: dir === "left" ? -360 : 360,
            behavior: "smooth",
        });
    };

    return (
        <section className="relative w-full overflow-hidden bg-white">
            <div className="absolute top-0 inset-x-0 h-px z-20" />

            <div className="relative z-10 mx-auto max-w-7xl px-6 py-8 md:px-8 md:py-14">
                {/* Header */}
                <div className="mb-16 flex flex-col items-start gap-5">
                    <div
                        className="inline-flex items-center rounded-full bg-white px-5 py-1.5 text-[11px] font-semibold tracking-[0.18em] uppercase text-indigo-700 cursor-default select-none"
                        style={{
                            boxShadow:
                                "0 0 10px rgba(79,70,229,0.18), 0 0 24px rgba(79,70,229,0.10)",
                            outline: "1px solid rgba(79,70,229,0.35)",
                            outlineOffset: "-1px",
                        }}
                    >
                        POLYMAP
                    </div>

                    <div>
                        <h2 className="text-4xl sm:text-6xl lg:text-[50px] font-medium tracking-[-0.03em] leading-[1.05] text-zinc-950">
                            Understand the network.
                        </h2>
                        <div className="flex flex-wrap items-baseline gap-x-3 mt-1.5">
                            <h2 className="text-4xl sm:text-6xl lg:text-[50px] font-medium tracking-[-0.03em] leading-[1.05] text-zinc-950">
                                Follow every
                            </h2>
                            <PointerHighlight
                                rectangleClassName="bg-indigo-500/10 border-indigo-400/30 leading-loose rounded"
                                pointerClassName="text-indigo-500 h-4 w-4"
                                containerClassName="inline-block"
                            >
                                <span className="relative z-10 text-4xl sm:text-6xl lg:text-[50px] font-medium tracking-[-0.03em] leading-[1.05] text-indigo-600">
                                    connection.
                                </span>
                            </PointerHighlight>
                        </div>
                    </div>

                    <p className="text-zinc-600 text-[17px] leading-relaxed max-w-lg mt-1">
                        Explore wallet relationships, trace transaction flows, and
                        make sense of the connections that shape an on-chain network.
                    </p>
                </div>

                {/* Feature carousel */}
                <div
                    ref={carouselRef}
                    onScroll={checkScrollability}
                    className="flex gap-4 overflow-x-scroll [scrollbar-width:none] pb-6"
                >
                    {FEATURES.map((feature) => (
                        <FeatureCard
                            key={feature.eyebrow}
                            feature={feature}
                        />
                    ))}
                </div>

                {/* Arrow controls */}
                <div className="flex justify-end gap-3 mt-4">
                    <LedArrowButton
                        direction="left"
                        onClick={() => scroll("left")}
                        disabled={!canScrollLeft}
                    />
                    <LedArrowButton
                        direction="right"
                        onClick={() => scroll("right")}
                        disabled={!canScrollRight}
                    />
                </div>

                {/* CTA */}
                <div className="mt-4 rounded-2xl bg-white px-8 py-14 sm:py-16 relative overflow-hidden flex flex-col sm:flex-row sm:items-center sm:justify-between gap-8">
                    <Share2
                        aria-hidden
                        size={340}
                        strokeWidth={0.7}
                        color="#000000"
                        className="pointer-events-none absolute -right-16 -bottom-20"
                        style={{ opacity: 0.045 }}
                    />

                    <div className="relative z-10 flex flex-col gap-4 max-w-xl">
                        <div
                            className="inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-[10px] font-semibold tracking-[0.16em] uppercase cursor-default select-none w-fit"
                            style={{
                                background: "rgba(0,0,0,0.07)",
                                color: "#000000",
                                border: "1px solid rgba(0,0,0,0.18)",
                            }}
                        >
                            <Network size={11} strokeWidth={2.5} />
                            Wallet Intelligence
                        </div>

                        <p className="text-black text-[28px] sm:text-[36px] font-medium tracking-[-0.03em] leading-[1.05]">
                            Every wallet is part of a bigger picture.
                        </p>

                        <p className="text-zinc-900 text-[14px] leading-relaxed max-w-md">
                            Start with an address. Discover its connections. Explore
                            the network around it.
                        </p>
                    </div>

                    <div className="relative z-10 flex flex-wrap items-center gap-3 shrink-0">
                        <Link
                            href="/dashboard"
                            className="rounded-full bg-black text-white px-6 py-3 text-[13px] font-semibold transition-transform duration-200 hover:scale-[1.03]"
                        >
                            Explore Polymap
                        </Link>

                        <Link
                            href="/"
                            className="rounded-full border border-black/15 px-6 py-3 text-[13px] font-semibold text-black transition-colors duration-200 hover:bg-black/[0.04]"
                        >
                            Learn more
                        </Link>
                    </div>
                </div>
            </div>
        </section>
    );
}
