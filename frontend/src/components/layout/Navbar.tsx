"use client";

import React, { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Film, Sparkles, FilmIcon, Users, UserCheck, ChevronDown, Menu, X } from "lucide-react";
import { useUser } from "@/context/UserContext";

export const Navbar: React.FC = () => {
  const pathname = usePathname();
  const { activeUser, setActiveUser } = useUser();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const navLinks = [
    { href: "/", label: "Home", icon: Film },
    { href: "/movies", label: "Movies", icon: FilmIcon },
    { href: "/recommendations", label: "Recommendations", icon: Sparkles },
    { href: "/users", label: "Users", icon: Users },
  ];

  const isActive = (path: string) => {
    if (path === "/" && pathname !== "/") return false;
    return pathname.startsWith(path);
  };

  return (
    <nav className="sticky top-0 z-40 bg-slate-950/80 backdrop-blur-md border-b border-slate-800/80">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand Logo */}
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="p-2 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-400 group-hover:scale-105 transition-transform">
              <Film className="w-5 h-5" />
            </div>
            <div className="flex flex-col">
              <span className="font-bold text-lg text-slate-100 tracking-tight flex items-center gap-1.5">
                CineForge <span className="text-xs px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 font-medium">ML</span>
              </span>
              <span className="text-[10px] text-slate-400 -mt-0.5">Hybrid Engine</span>
            </div>
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex items-center space-x-1">
            {navLinks.map((link) => {
              const Icon = link.icon;
              const active = isActive(link.href);
              return (
                <Link
                  key={link.href}
                  href={link.href}
                  className={`flex items-center gap-2 px-3.5 py-2 rounded-lg text-sm font-medium transition-all ${
                    active
                      ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                      : "text-slate-300 hover:text-slate-100 hover:bg-slate-900"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  {link.label}
                </Link>
              );
            })}
          </div>

          {/* Active User Badge / Switcher */}
          <div className="hidden md:flex items-center gap-3">
            {activeUser ? (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800">
                <Link href={`/users/${activeUser.id}`} className="flex items-center gap-2 text-sm text-slate-200 hover:text-amber-400 transition-colors">
                  <UserCheck className="w-4 h-4 text-emerald-400" />
                  <span className="font-medium">{activeUser.username}</span>
                  <span className="text-xs text-slate-400">#{activeUser.id}</span>
                </Link>
                <button
                  onClick={() => setActiveUser(null)}
                  className="text-xs text-slate-400 hover:text-rose-400 ml-1 transition-colors"
                  title="Switch user"
                >
                  Clear
                </button>
              </div>
            ) : (
              <Link
                href="/users"
                className="flex items-center gap-2 text-xs font-medium text-amber-400 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 px-3 py-1.5 rounded-xl transition-all"
              >
                Select Profile
              </Link>
            )}
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-900"
            >
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <div className="md:hidden bg-slate-950 border-b border-slate-800 px-4 pt-2 pb-4 space-y-2">
          {navLinks.map((link) => {
            const Icon = link.icon;
            const active = isActive(link.href);
            return (
              <Link
                key={link.href}
                href={link.href}
                onClick={() => setMobileMenuOpen(false)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-base font-medium ${
                  active
                    ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                    : "text-slate-300 hover:bg-slate-900"
                }`}
              >
                <Icon className="w-5 h-5" />
                {link.label}
              </Link>
            );
          })}
          {activeUser && (
            <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
              <span className="text-sm text-slate-300">
                User: <strong>{activeUser.username}</strong>
              </span>
              <button
                onClick={() => {
                  setActiveUser(null);
                  setMobileMenuOpen(false);
                }}
                className="text-xs text-rose-400 hover:underline"
              >
                Clear User
              </button>
            </div>
          )}
        </div>
      )}
    </nav>
  );
};
