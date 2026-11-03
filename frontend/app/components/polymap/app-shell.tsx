
"use client";

import type { ComponentType } from "react";

import {
  BarChart3,
  BrainCircuit,
  Crosshair,
  FlaskConical,
  GitCompare,
  Home as HomeIcon,
  Network as NetworkIcon,
  Store,
  Wallet,
} from "lucide-react";

import Link from "next/link";


import { ArchitectureView } from "./architecture-view";
import { ChartsView } from "./charts-view";
import { Home } from "./home";
import { MarketsView } from "./markets-view";
import { ModelResultsView } from "./model-results-view";
import { Network } from "./network";
import { PredictionsView } from "./predictions-view";
import { PairsView } from "./pairs-view";
import {
  PolymapProvider,
  usePolymap,
  type View,
} from "./store";

import { cn } from "./ui";
import { WalletsView } from "./wallets-view";

/* --------------------------------------------------
   Navigation
-------------------------------------------------- */

const NAV: {
  id: View;
  label: string;
  icon: ComponentType<{ className?: string }>;
}[] = [
    { id: "home", label: "Overview", icon: HomeIcon },
    { id: "markets", label: "Markets", icon: Store },
    { id: "network", label: "Network", icon: NetworkIcon },
    { id: "pairs", label: "Pairs", icon: GitCompare },
    { id: "wallets", label: "Wallets", icon: Wallet },
    { id: "charts", label: "Charts", icon: BarChart3 },
    { id: "architecture", label: "PolymapTGN architecture", icon: BrainCircuit },
    { id: "results", label: "Model results", icon: FlaskConical },
    { id: "predictions", label: "Predictions", icon: Crosshair },
  ];

/* --------------------------------------------------
   Sidebar
-------------------------------------------------- */

function Sidebar() {
  const { view, setView } = usePolymap();

  return (
    <aside className="fixed inset-y-0 left-0 hidden w-56 flex-col border-r border-neutral-200 bg-white md:flex">
      <div className="flex h-14 items-center gap-2 border-b border-neutral-200 px-5">
        <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-neutral-950 text-sm font-bold text-white">
          P
        </div>

        <Link
          href="/"
          className="transition-opacity hover:opacity-70"
        >
          <span className="text-xl font-semibold tracking-tight text-neutral-950">
            Polymap
          </span>
        </Link>
      </div>

      <nav className="flex-1 space-y-1 p-3">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setView(id)}
            aria-current={
              view === id ? "page" : undefined
            }
            className={cn(
              "flex h-10 w-full items-center gap-3 rounded-lg px-3 text-sm font-medium transition-colors",
              view === id
                ? "bg-teal-50 text-teal-900"
                : "text-neutral-700 hover:bg-neutral-50 hover:text-neutral-950"
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </button>
        ))}
      </nav>

      <div className="border-t border-neutral-200 p-4">
        <p className="text-xs text-neutral-500">
          Polymap · Wallet intelligence
        </p>
      </div>
    </aside>
  );
}

/* --------------------------------------------------
   Mobile navigation
-------------------------------------------------- */

function MobileNav() {
  const { view, setView } = usePolymap();

  return (
    <div className="flex gap-1 overflow-x-auto border-b border-neutral-200 bg-white px-3 py-2 md:hidden">
      {NAV.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => setView(id)}
          aria-current={
            view === id ? "page" : undefined
          }
          className={cn(
            "flex h-9 shrink-0 items-center gap-2 rounded-lg px-3 text-sm font-medium transition-colors",
            view === id
              ? "bg-teal-50 text-teal-900"
              : "text-neutral-600 hover:bg-neutral-50"
          )}
        >
          <Icon className="h-4 w-4" />
          {label}
        </button>
      ))}
    </div>
  );
}

/* --------------------------------------------------
   Page content
-------------------------------------------------- */

function Content() {
  const { view } = usePolymap();

  switch (view) {
    case "home":
      return <Home />;

    case "markets":
      return <MarketsView />;

    case "network":
      return <Network />;

    case "pairs":
      return <PairsView />;

    case "wallets":
      return <WalletsView />;

    case "charts":
      return <ChartsView />;

    case "architecture":
      return <ArchitectureView />;

    case "results":
      return <ModelResultsView />;

    case "predictions":
      return <PredictionsView />;

    default:
      return <Home />;
  }
}

/* --------------------------------------------------
   Application shell
-------------------------------------------------- */

export function AppShell() {
  return (
    <PolymapProvider>
      <div className="min-h-screen bg-white font-sans text-neutral-950 antialiased">
        <Sidebar />

        <div className="md:pl-56">
          <MobileNav />

          <main className="mx-auto w-full max-w-7xl px-4 py-8 md:px-8">
            <Content />
          </main>
        </div>
      </div>
    </PolymapProvider>
  );
}
