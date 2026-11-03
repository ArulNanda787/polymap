
"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import {
  filterRelationships,
  pairs,
} from "@/lib/polymap/data";

import type {
  CandidateRelationship,
  RelationshipFilter,
} from "@/lib/polymap/types";

/* --------------------------------------------------
   Types
-------------------------------------------------- */

export type View =
  | "home"
  | "markets"
  | "network"
  | "pairs"
  | "wallets"
  | "charts"
  | "architecture"
  | "results"
  | "predictions";

export type Level = RelationshipFilter;

const VIEWS: readonly View[] = [
  "home",
  "markets",
  "network",
  "pairs",
  "wallets",
  "charts",
  "architecture",
  "results",
  "predictions",
];

function viewFromHash(): View | null {
  if (typeof window === "undefined") return null;
  const hash = window.location.hash.replace(/^#/, "");
  return (VIEWS as readonly string[]).includes(hash) ? (hash as View) : null;
}

interface PolymapContextValue {
  view: View;
  setView: (view: View) => void;

  marketId: string | null;
  setMarketId: (id: string | null) => void;

  level: Level;
  setLevel: (level: Level) => void;

  search: string;
  setSearch: (value: string) => void;

  selectedWalletId: string | null;
  selectedPairId: string | null;

  selectWallet: (id: string | null) => void;
  selectPair: (id: string | null) => void;

  filteredPairs: CandidateRelationship[];

  resetFilters: () => void;
}

/* --------------------------------------------------
   Context
-------------------------------------------------- */

const PolymapContext =
  createContext<PolymapContextValue | null>(null);

/* --------------------------------------------------
   Provider
-------------------------------------------------- */

export function PolymapProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [view, setViewState] = useState<View>("home");

  // Keep the open tab in the URL (#predictions etc.) so a reload, a shared
  // link or the browser back button returns to the same tab.
  useEffect(() => {
    const sync = () => {
      const fromHash = viewFromHash();
      if (fromHash) setViewState(fromHash);
    };
    sync();
    window.addEventListener("hashchange", sync);
    return () => window.removeEventListener("hashchange", sync);
  }, []);

  const setView = useCallback((next: View) => {
    setViewState(next);
    if (window.location.hash !== `#${next}`) {
      window.location.hash = next;
    }
  }, []);

  const [marketId, setMarketId] =
    useState<string | null>(null);

  const [level, setLevel] = useState<Level>("all");

  const [search, setSearch] = useState("");

  const [selectedWalletId, setSelectedWalletId] =
    useState<string | null>(null);

  const [selectedPairId, setSelectedPairId] =
    useState<string | null>(null);

  /* ----------------------------------------------
     Selection handlers
  ---------------------------------------------- */

  const selectWallet = useCallback(
    (id: string | null) => {
      setSelectedWalletId(id);
      setSelectedPairId(null);
    },
    []
  );

  const selectPair = useCallback(
    (id: string | null) => {
      setSelectedPairId(id);
      setSelectedWalletId(null);
    },
    []
  );

  /* ----------------------------------------------
     Relationship filtering
  ---------------------------------------------- */

  const filteredPairs = useMemo(() => {
    const options: {
      marketId: string | null;
      search: string;
      persistentOnly?: boolean;
      tightOnly?: boolean;
      category?: string | null;
    } = {
      marketId,
      search: search.trim(),
    };

    switch (level) {
      case "persistent":
        options.persistentOnly = true;
        break;

      case "one_off":
        options.category = "one_off_loose";
        break;

      case "tight":
        options.tightOnly = true;
        break;

      case "all":
      default:
        break;
    }

    return filterRelationships(pairs, options);
  }, [marketId, search, level]);

  /* ----------------------------------------------
     Reset
  ---------------------------------------------- */

  const resetFilters = useCallback(() => {
    setMarketId(null);
    setLevel("all");
    setSearch("");
    setSelectedWalletId(null);
    setSelectedPairId(null);
  }, []);

  /* ----------------------------------------------
     Context value
  ---------------------------------------------- */

  const value = useMemo<PolymapContextValue>(
    () => ({
      view,
      setView,

      marketId,
      setMarketId,

      level,
      setLevel,

      search,
      setSearch,

      selectedWalletId,
      selectedPairId,

      selectWallet,
      selectPair,

      filteredPairs,

      resetFilters,
    }),
    [
      view,
      marketId,
      level,
      search,
      selectedWalletId,
      selectedPairId,
      selectWallet,
      selectPair,
      filteredPairs,
      resetFilters,
    ]
  );

  return (
    <PolymapContext.Provider value={value}>
      {children}
    </PolymapContext.Provider>
  );
}

/* --------------------------------------------------
   Hook
-------------------------------------------------- */

export function usePolymap(): PolymapContextValue {
  const context = useContext(PolymapContext);

  if (!context) {
    throw new Error(
      "usePolymap must be used inside PolymapProvider"
    );
  }

  return context;
}
