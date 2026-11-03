
import * as React from "react";

import {
  walletById,
  CATEGORY,
} from "@/lib/polymap/data";

import type {
  WalletType,
  RelationshipCategory,
} from "@/lib/polymap/types";

/* --------------------------------------------------
   Utilities
-------------------------------------------------- */

export function cn(
  ...classes: (string | false | null | undefined)[]
) {
  return classes.filter(Boolean).join(" ");
}

/* --------------------------------------------------
   Cards
-------------------------------------------------- */

export function Card({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "rounded-xl border border-neutral-200 bg-white text-neutral-950 shadow-sm",
        className
      )}
      {...props}
    />
  );
}

export function CardHeader({
  title,
  description,
  action,
}: {
  title: string;
  description?: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 px-5 pt-5">
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-neutral-950">
          {title}
        </h3>

        {description && (
          <p className="mt-1 text-sm text-neutral-600">
            {description}
          </p>
        )}
      </div>

      {action}
    </div>
  );
}

export function CardBody({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("p-5", className)}
      {...props}
    />
  );
}

/* --------------------------------------------------
   Buttons
-------------------------------------------------- */

type ButtonProps =
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: "primary" | "outline" | "ghost";
    size?: "sm" | "md" | "icon";
  };

export function Button({
  variant = "outline",
  size = "md",
  className,
  ...props
}: ButtonProps) {
  return (
    <button
      className={cn(
        "inline-flex shrink-0 items-center justify-center gap-2 whitespace-nowrap rounded-lg text-sm font-medium transition-all",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600/30",
        "disabled:pointer-events-none disabled:opacity-50",

        variant === "primary" &&
        "bg-teal-700 text-white shadow-sm hover:bg-teal-800",

        variant === "outline" &&
        "border border-neutral-200 bg-white text-neutral-950 hover:border-teal-300 hover:bg-teal-50",

        variant === "ghost" &&
        "text-neutral-700 hover:bg-teal-50 hover:text-teal-900",

        size === "sm" && "h-8 px-3",
        size === "md" && "h-9 px-4",
        size === "icon" && "h-8 w-8",

        className
      )}
      {...props}
    />
  );
}

/* --------------------------------------------------
   Generic badge
-------------------------------------------------- */

export function Badge({
  className,
  ...props
}: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border border-neutral-200 bg-neutral-50 px-2.5 py-0.5 text-xs font-medium text-neutral-800",
        className
      )}
      {...props}
    />
  );
}

/* --------------------------------------------------
   Relationship categories
-------------------------------------------------- */

const CATEGORY_STYLE: Record<
  RelationshipCategory,
  {
    label: string;
    color: string;
    background: string;
    border: string;
  }
> = {
  persistent_tight: {
    label: "Persistent · Tight",
    color: "#6D28D9",
    background: "#EDE9FE",
    border: "#DDD6FE",
  },

  persistent_loose: {
    label: "Persistent",
    color: "#0F766E",
    background: "#CCFBF1",
    border: "#99F6E4",
  },

  one_off_loose: {
    label: "One-off",
    color: "#C2410C",
    background: "#FFEDD5",
    border: "#FED7AA",
  },
};

export function CategoryBadge({
  category,
}: {
  category: RelationshipCategory;
}) {
  const style = CATEGORY_STYLE[category];

  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-xs font-semibold"
      style={{
        color: style.color,
        background: style.background,
        borderColor: style.border,
      }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: style.color }}
      />
      {style.label}
    </span>
  );
}

/* --------------------------------------------------
   Wallet types
-------------------------------------------------- */

const WALLET_STYLE: Record<
  WalletType,
  {
    label: string;
    color: string;
    background: string;
    border: string;
  }
> = {
  Whale: {
    label: "Whale",
    color: "#6D28D9",
    background: "#EDE9FE",
    border: "#DDD6FE",
  },

  "Frequent trader": {
    label: "Frequent trader",
    color: "#1D4ED8",
    background: "#DBEAFE",
    border: "#BFDBFE",
  },

  "Occasional trader": {
    label: "Occasional trader",
    color: "#15803D",
    background: "#DCFCE7",
    border: "#BBF7D0",
  },

  "New wallet": {
    label: "New wallet",
    color: "#C2410C",
    background: "#FFEDD5",
    border: "#FED7AA",
  },
};

export function KindDot({
  kind,
  size = 10,
}: {
  kind: WalletType;
  size?: number;
}) {
  const style = WALLET_STYLE[kind];

  return (
    <span
      className="inline-block shrink-0 rounded-full ring-2 ring-white"
      style={{
        width: size,
        height: size,
        background: style.color,
      }}
    />
  );
}

export function KindBadge({
  kind,
}: {
  kind: WalletType;
}) {
  const style = WALLET_STYLE[kind];

  return (
    <span
      className="inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2.5 py-1 text-xs font-semibold"
      style={{
        color: style.color,
        background: style.background,
        borderColor: style.border,
      }}
    >
      <KindDot kind={kind} size={8} />
      {style.label}
    </span>
  );
}

/* --------------------------------------------------
   Wallet chip
-------------------------------------------------- */

export function WalletChip({
  walletId,
}: {
  walletId: string;
}) {
  const wallet = walletById.get(walletId);

  if (!wallet) return null;

  return (
    <span
      className="inline-flex items-center gap-2 text-sm"
      title={wallet.wallet_type}
    >
      <KindDot kind={wallet.wallet_type} />
      <span className="font-medium text-neutral-900">
        {wallet.address}
      </span>
    </span>
  );
}

/* --------------------------------------------------
   Input
-------------------------------------------------- */

export const Input = React.forwardRef<
  HTMLInputElement,
  React.InputHTMLAttributes<HTMLInputElement>
>(function Input({ className, ...props }, ref) {
  return (
    <input
      ref={ref}
      className={cn(
        "h-9 w-full rounded-lg border border-neutral-200 bg-white px-3 text-sm text-neutral-950 placeholder:text-neutral-400",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600/30 focus-visible:border-teal-500",
        className
      )}
      {...props}
    />
  );
});

/* --------------------------------------------------
   Select
-------------------------------------------------- */

export function Select({
  className,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return (
    <select
      className={cn(
        "h-9 rounded-lg border border-neutral-200 bg-white px-3 text-sm text-neutral-950",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-600/30 focus-visible:border-teal-500",
        className
      )}
      {...props}
    />
  );
}

/* --------------------------------------------------
   Segmented control
-------------------------------------------------- */

export function Segmented<T extends string>({
  value,
  onChange,
  options,
}: {
  value: T;
  onChange: (value: T) => void;
  options: {
    value: T;
    label: string;
  }[];
}) {
  return (
    <div className="inline-flex flex-wrap rounded-lg border border-neutral-200 bg-neutral-100 p-1">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          aria-pressed={value === option.value}
          className={cn(
            "h-8 rounded-md px-3 text-sm font-medium transition-all",
            value === option.value
              ? "bg-white text-teal-900 shadow-sm"
              : "text-neutral-600 hover:bg-white/60 hover:text-neutral-950"
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

/* --------------------------------------------------
   Statistics
-------------------------------------------------- */

export function Stat({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
}) {
  return (
    <div title={hint}>
      <div className="text-xs font-medium text-neutral-500">
        {label}
      </div>

      <div className="mt-1 text-base font-semibold tracking-tight text-neutral-950">
        {value}
      </div>

      {hint && (
        <p className="mt-1 text-xs leading-relaxed text-neutral-500">
          {hint}
        </p>
      )}
    </div>
  );
}

/* --------------------------------------------------
   Page header
-------------------------------------------------- */

export function PageHeader({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0">
        <h1 className="text-xl font-semibold tracking-tight text-neutral-950">
          {title}
        </h1>

        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-neutral-600">
          {description}
        </p>
      </div>

      {action}
    </div>
  );
}

/* --------------------------------------------------
   Empty state
-------------------------------------------------- */

export function EmptyState({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
      <p className="text-sm font-semibold text-neutral-950">
        {title}
      </p>

      <p className="mt-1 max-w-xs text-sm text-neutral-600">
        {description}
      </p>
    </div>
  );
}

/* --------------------------------------------------
   Sparkline
   Kept for any remaining legacy consumers.
-------------------------------------------------- */

export function Sparkline({
  values,
}: {
  values: number[];
}) {
  if (values.length < 2) return null;

  const width = 96;
  const height = 28;

  const max = Math.max(...values);
  const min = Math.min(...values);

  const points = values
    .map((value, index) => {
      const x =
        (index / (values.length - 1)) * width;

      const y =
        height -
        3 -
        ((value - min) / Math.max(max - min, 1)) *
        (height - 6);

      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden
    >
      <polyline
        points={points}
        fill="none"
        stroke="#0D9488"
        strokeWidth="2"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
