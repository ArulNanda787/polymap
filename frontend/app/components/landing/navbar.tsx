"use client";

import Link from "next/link";
import { useState } from "react";
import { Menu, X } from "lucide-react"; // Make sure you have lucide-react installed

export default function Navbar() {
    const [isOpen, setIsOpen] = useState(false);

    const links = [
        { name: "Dashboard", href: "/dashboard" },

    ];

    return (
        <nav className="sticky top-0 z-50 w-full border-b border-zinc-200/80 bg-white/80 backdrop-blur-md">
            <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-6 md:px-12">

                {/* Logo / Brand */}
                <div className="flex items-center gap-2">
                    <Link href="/" className="flex items-center gap-2 transition-opacity hover:opacity-80">
                        {/* You can replace this div with your <TechText /> component if you want a micro version */}
                        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-600 text-white font-bold text-lg">
                            P
                        </div>
                        <span className="text-xl font-semibold tracking-tight text-zinc-900">
                            Polymap
                        </span>
                    </Link>
                </div>

                {/* Desktop Navigation */}
                <div className="hidden md:flex md:items-center md:gap-8">
                    <div className="flex items-center gap-6">
                        {links.map((link) => (
                            <Link
                                key={link.name}
                                href={link.href}
                                className="text-sm font-medium text-zinc-600 transition-colors hover:text-zinc-900"
                            >
                                {link.name}
                            </Link>
                        ))}
                    </div>

                    <div className="h-4 w-px bg-zinc-200" /> {/* Divider */}

                    <div className="flex items-center gap-4">
                       
                    </div>
                </div>

                {/* Mobile Menu Button */}
                <button
                    onClick={() => setIsOpen(!isOpen)}
                    className="flex h-10 w-10 items-center justify-center rounded-md text-zinc-600 hover:bg-zinc-100 hover:text-zinc-900 md:hidden"
                    aria-label="Toggle menu"
                >
                    {isOpen ? <X size={20} /> : <Menu size={20} />}
                </button>
            </div>

            {/* Mobile Navigation Dropdown */}
            {isOpen && (
                <div className="border-t border-zinc-200 bg-white px-6 py-4 md:hidden">
                    <div className="flex flex-col gap-4">
                        {links.map((link) => (
                            <Link
                                key={link.name}
                                href={link.href}
                                onClick={() => setIsOpen(false)}
                                className="text-base font-medium text-zinc-600 hover:text-zinc-900"
                            >
                                {link.name}
                            </Link>
                        ))}
                        <hr className="border-zinc-100" />
                        <Link
                            href="/login"
                            onClick={() => setIsOpen(false)}
                            className="text-base font-medium text-zinc-600 hover:text-zinc-900"
                        >
                            Log in
                        </Link>
                        <Link
                            href="/signup"
                            onClick={() => setIsOpen(false)}
                            className="inline-flex h-10 items-center justify-center rounded-md bg-zinc-900 px-4 text-sm font-medium text-white transition-colors hover:bg-zinc-800"
                        >
                            Get Started
                        </Link>
                    </div>
                </div>
            )}
        </nav>
    );
}