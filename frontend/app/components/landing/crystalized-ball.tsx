"use client"
import NavMenu from "./branched-menu-main";
import Link from "next/link";
import TechText from "@/components/TechText";
import MarketVolumeCard from "./market-volume-card";

export default function CrystalizedBallBlue() {
    return (
        <div className="relative w-full min-h-[600px] bg-white flex items-center px-6 py-12 md:px-16 lg:px-24 overflow-hidden">

            <div className="relative z-10 flex flex-col lg:flex-row items-start gap-12 lg:gap-20 w-full">

                {/* Left Section */}
                <div className="flex-1 flex flex-col items-start justify-start w-full lg:max-w-2xl">
                    <div className="max-w-md flex flex-col gap-2">
                        <TechText
                            text="Polymap"
                            fontWeight={600}
                            fontSize={150}
                            reveal="letter"
                            dashLength={4}
                            dashGap={2}
                            specks={15}
                            intro="none"
                            introDuration={2}
                            clickKick
                            guides={false}
                            fontFamily=""
                            letterSpacing={-0.05}
                            accentColor="currentColor"
                            reach={200}
                            softness={0.7}
                            lineStyle="dashed"
                            strokeWidth={1.5}
                            speed={1}
                            bounce={0.4}
                            selection
                            labels
                            color="currentColor"
                            align="left"
                            draggable
                            sweep
                        />
                        <p className="text-base text-black leading-relaxed mt-2">
                            An investigative workspace for exploring temporal trading relationships between wallets in a market, inspecting the underlying observations, and comparing candidate relationships using transparent measurements.
                        </p>

                        <div className="pt-4">
                            <Link
                                href="/dashboard"
                                className="inline-flex items-center justify-center rounded-md bg-zinc-900 px-5 py-2.5 text-sm font-medium text-white transition-colors hover:bg-zinc-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900 focus-visible:ring-offset-2"
                            >
                                Explore Dashboard
                            </Link>
                        </div>
                    </div>
                </div>

                {/* Right Section */}
                <div className="shrink-0 w-full lg:w-auto flex justify-start lg:justify-end">
                    <MarketVolumeCard />
                </div>

            </div>
        </div>
    );
}