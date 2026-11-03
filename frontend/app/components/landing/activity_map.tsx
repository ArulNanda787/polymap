"use client";

import ActivityTerrain from "@/components/kokonutui/activity-terrain";

// Assuming you have the JSON imported or can pass the data in
// import polymapData from "@/lib/polymap/data"; 

export default function PolymapActivity() {
    // Extracting the last 28 days of total volume or trades for the terrain
    // For this example, I'm using a 28-day slice of realistic mock data 
    // to populate the 3D columns. You can swap this with data from your JSON.
    const mockActivityData = [
        24, 45, 38, 52, 41, 12, 8,
        30, 48, 56, 42, 35, 15, 9,
        22, 38, 49, 44, 39, 18, 11,
        28, 42, 51, 47, 36, 14, 7
    ];

    return (
        <div className="w-full bg-white text-black py-24 px-6 md:px-12 flex flex-col justify-center">
            <div className="max-w-7xl mx-auto w-full flex flex-col lg:flex-row items-center gap-16 lg:gap-24">

                {/* Left Side: Minimalist Typography & Stats */}
                <div className="w-full lg:w-4/12 flex flex-col gap-8">
                    <div>
                        <h2 className="text-4xl md:text-5xl font-medium tracking-tight mb-4">
                            Temporal Activity
                        </h2>
                        <p className="text-lg text-zinc-500 leading-relaxed max-w-sm">
                            An investigative view of wallet interactions. Drag and tilt the terrain to explore spikes in trading volume and market relationships.
                        </p>
                    </div>

                    {/* Clean, borderless stats */}
                    <div className="flex gap-12 mt-4">
                        <div className="flex flex-col gap-1">
                            <span className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
                                Volume (30d)
                            </span>
                            <span className="text-3xl font-light tracking-tight text-black">
                                $205M
                            </span>
                        </div>
                        <div className="flex flex-col gap-1">
                            <span className="text-sm font-medium text-zinc-400 uppercase tracking-wider">
                                Traders
                            </span>
                            <span className="text-3xl font-light tracking-tight text-black">
                                181k
                            </span>
                        </div>
                    </div>
                </div>

                {/* Right Side: The Terrain (Massively scaled up) */}
                <div className="w-full lg:w-8/12 flex justify-end">
                    <div className="w-full max-w-3xl">
                        <ActivityTerrain
                            data={mockActivityData}
                            title="Interaction Volume"
                            defaultFlat={false}
                            // I removed the shadow and borders to let it float seamlessly on the white page
                            className="w-full border-none shadow-none bg-transparent dark:bg-transparent"
                        />
                    </div>
                </div>

            </div>
        </div>
    );
}