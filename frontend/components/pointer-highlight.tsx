"use client";
import { cn } from "@/lib/utils";
import { motion } from "motion/react";
import { useRef, useEffect, useState, CSSProperties } from "react";

export function PointerHighlight({
    children,
    rectangleClassName,
    rectangleStyle,
    pointerClassName,
    containerClassName,
}: {
    children: React.ReactNode;
    rectangleClassName?: string;
    rectangleStyle?: CSSProperties;
    pointerClassName?: string;
    containerClassName?: string;
}) {
    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 0, height: 0 });
    const [hasAnimated, setHasAnimated] = useState(false);
    const [isVisible, setIsVisible] = useState(false);

    // Gate everything behind IntersectionObserver — nothing runs until element is near viewport
    useEffect(() => {
        const el = containerRef.current;
        if (!el) return;

        const io = new IntersectionObserver(
            ([entry]) => {
                if (entry.isIntersecting) {
                    // Seed dimensions once visible
                    const { width, height } = el.getBoundingClientRect();
                    setDimensions({ width, height });
                    setIsVisible(true);
                    io.disconnect();
                }
            },
            { threshold: 0, rootMargin: "120px" }
        );

        io.observe(el);
        return () => io.disconnect();
    }, []);

    // ResizeObserver only starts after element is visible
    useEffect(() => {
        if (!isVisible || !containerRef.current) return;

        const ro = new ResizeObserver((entries) => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                setDimensions({ width, height });
            }
        });

        ro.observe(containerRef.current);
        return () => ro.disconnect();
    }, [isVisible]);

    return (
        <div
            className={cn("relative w-fit", containerClassName)}
            ref={containerRef}
            suppressHydrationWarning
        >
            {children}
            {isVisible && dimensions.width > 0 && dimensions.height > 0 && (
                <motion.div
                    className="pointer-events-none absolute inset-0 z-0"
                    initial={{ opacity: 0, scale: 0.95, originX: 0, originY: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                >
                    <motion.div
                        className={cn(
                            "absolute inset-0 border border-neutral-800 dark:border-neutral-200",
                            rectangleClassName,
                        )}
                        style={rectangleStyle}
                        initial={{ width: 0, height: 0 }}
                        animate={{
                            width: hasAnimated ? dimensions.width : 0,
                            height: hasAnimated ? dimensions.height : 0,
                        }}
                        whileInView={{
                            width: dimensions.width,
                            height: dimensions.height,
                        }}
                        onViewportEnter={() => setHasAnimated(true)}
                        transition={{ duration: 1, ease: "easeInOut" }}
                    />
                    <motion.div
                        className="pointer-events-none absolute"
                        initial={{ opacity: 0 }}
                        animate={{
                            opacity: hasAnimated ? 1 : 0,
                            x: hasAnimated ? dimensions.width + 4 : 0,
                            y: hasAnimated ? dimensions.height + 4 : 0,
                        }}
                        whileInView={{
                            opacity: 1,
                            x: dimensions.width + 4,
                            y: dimensions.height + 4,
                        }}
                        onViewportEnter={() => setHasAnimated(true)}
                        style={{ rotate: -90 }}
                        transition={{
                            opacity: { duration: 0.1, ease: "easeInOut" },
                            duration: 1,
                            ease: "easeInOut",
                        }}
                    >
                        <Pointer className={cn("h-5 w-5 text-blue-500", pointerClassName)} />
                    </motion.div>
                </motion.div>
            )}
        </div>
    );
}

const Pointer = ({ ...props }: React.SVGProps<SVGSVGElement>) => (
    <svg
        stroke="currentColor"
        fill="currentColor"
        strokeWidth="1"
        strokeLinecap="round"
        strokeLinejoin="round"
        viewBox="0 0 16 16"
        height="1em"
        width="1em"
        xmlns="http://www.w3.org/2000/svg"
        {...props}
    >
        <path d="M14.082 2.182a.5.5 0 0 1 .103.557L8.528 15.467a.5.5 0 0 1-.917-.007L5.57 10.694.803 8.652a.5.5 0 0 1-.006-.916l12.728-5.657a.5.5 0 0 1 .556.103z" />
    </svg>
);