"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  Film,
  Search,
  Sparkles,
  Users,
  Sliders,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  TrendingUp,
} from "lucide-react";
import { getHealth } from "@/lib/api/health";
import { searchMovies } from "@/lib/api/movies";
import { Movie } from "@/types/movie";
import { HealthResponse } from "@/types/common";
import { MovieCard } from "@/components/movies/MovieCard";
import { SkeletonCard } from "@/components/ui/SkeletonCard";
import { useUser } from "@/context/UserContext";

export default function HomePage() {
  const router = useRouter();
  const { activeUser } = useUser();

  const [searchQuery, setSearchQuery] = useState("");
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [featuredMovies, setFeaturedMovies] = useState<Movie[]>([]);
  const [isLoadingMovies, setIsLoadingMovies] = useState(true);

  useEffect(() => {
    let isMounted = true;

    getHealth()
      .then((data) => {
        if (isMounted) setHealth(data);
      })
      .catch((err) => {
        if (isMounted) setHealthError(err.message || "Failed to connect to backend");
      });

    searchMovies("star", 6)
      .then((data) => {
        if (isMounted) setFeaturedMovies(data.movies);
      })
      .catch(() => {})
      .finally(() => {
        if (isMounted) setIsLoadingMovies(false);
      });

    return () => {
      isMounted = false;
    };
  }, []);

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery.trim()) {
      router.push(`/movies?q=${encodeURIComponent(searchQuery.trim())}`);
    }
  };

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative rounded-3xl bg-gradient-to-b from-slate-900 via-slate-900/90 to-slate-950 border border-slate-800 p-8 sm:p-12 overflow-hidden shadow-2xl">
        <div className="absolute -top-24 -right-24 w-96 h-96 bg-amber-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl space-y-6">
          <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs font-semibold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Hybrid Machine Learning Engine</span>
          </div>

          <h1 className="text-4xl sm:text-5xl font-black text-slate-100 tracking-tight leading-tight">
            Discover Movies Powered by <span className="text-transparent bg-clip-text bg-gradient-to-r from-amber-400 to-amber-200">ML Intelligence</span>
          </h1>

          <p className="text-base sm:text-lg text-slate-300 leading-relaxed">
            Combine <strong className="text-slate-100">Content-Based TF-IDF metadata similarity</strong> with <strong className="text-slate-100">Item-Based Collaborative Filtering</strong> to generate personalized movie recommendations.
          </p>

          {/* Search Form */}
          <form onSubmit={handleSearchSubmit} className="flex items-center gap-2 max-w-xl">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-3.5 w-5 h-5 text-slate-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search movies by title (e.g. Toy Story, Star Wars)..."
                className="w-full bg-slate-950/80 border border-slate-700/80 focus:border-amber-500 text-slate-100 placeholder-slate-500 pl-11 pr-4 py-3 rounded-2xl text-sm font-medium focus:outline-none focus:ring-2 focus:ring-amber-500/20 transition-all"
              />
            </div>
            <button
              type="submit"
              className="bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold px-6 py-3 rounded-2xl text-sm transition-all shadow-lg shadow-amber-500/20 hover:shadow-amber-500/30 flex items-center gap-2 shrink-0"
            >
              <span>Search</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </form>

          {/* System Status Pill */}
          <div className="pt-2 flex items-center gap-3 text-xs text-slate-400">
            <span>Backend Status:</span>
            {health ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-medium">
                <CheckCircle2 className="w-3.5 h-3.5" />
                All Models Ready (PostgreSQL Connected)
              </span>
            ) : healthError ? (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 font-medium">
                <AlertCircle className="w-3.5 h-3.5" />
                {healthError}
              </span>
            ) : (
              <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-slate-800 text-slate-400 font-medium">
                Checking models...
              </span>
            )}
          </div>
        </div>
      </section>

      {/* Recommendation Modes Discovery Grid */}
      <section className="space-y-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-amber-400" />
              Recommendation Modes
            </h2>
            <p className="text-xs text-slate-400">Explore recommendations calculated by different algorithms</p>
          </div>
          <Link
            href="/recommendations"
            className="text-xs font-semibold text-amber-400 hover:text-amber-300 flex items-center gap-1 transition-colors"
          >
            <span>Open Recommender Engine</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Card 1: Content-Based */}
          <Link
            href="/recommendations?mode=content"
            className="group bg-slate-900/80 border border-slate-800 hover:border-amber-500/40 p-6 rounded-2xl transition-all duration-300 hover:-translate-y-1 shadow-lg space-y-4 flex flex-col justify-between"
          >
            <div className="space-y-3">
              <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 w-fit">
                <Film className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-100 group-hover:text-amber-300 transition-colors">
                Content-Based
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Find movies with similar textual metadata, genre tokens, and title features using TF-IDF vectorization and Cosine Similarity.
              </p>
            </div>
            <div className="pt-2 text-xs font-semibold text-amber-400 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
              <span>Try Content Mode</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </div>
          </Link>

          {/* Card 2: Collaborative Filtering */}
          <Link
            href="/recommendations?mode=collaborative"
            className="group bg-slate-900/80 border border-slate-800 hover:border-amber-500/40 p-6 rounded-2xl transition-all duration-300 hover:-translate-y-1 shadow-lg space-y-4 flex flex-col justify-between"
          >
            <div className="space-y-3">
              <div className="p-3 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400 w-fit">
                <Users className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-100 group-hover:text-amber-300 transition-colors">
                Collaborative Filtering
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Predict user preferences by analyzing sparse user-movie interaction matrices and mean-centered rating similarity.
              </p>
            </div>
            <div className="pt-2 text-xs font-semibold text-amber-400 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
              <span>Try Collaborative Mode</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </div>
          </Link>

          {/* Card 3: Hybrid Recommender */}
          <Link
            href="/recommendations?mode=hybrid"
            className="group bg-slate-900/80 border border-slate-800 hover:border-amber-500/40 p-6 rounded-2xl transition-all duration-300 hover:-translate-y-1 shadow-lg space-y-4 flex flex-col justify-between"
          >
            <div className="space-y-3">
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-amber-400 w-fit">
                <Sliders className="w-6 h-6" />
              </div>
              <h3 className="text-lg font-bold text-slate-100 group-hover:text-amber-300 transition-colors">
                Hybrid Recommendation
              </h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Combine content and collaborative signals with customizable weight sliders (e.g. 50% Content / 50% Collaborative).
              </p>
            </div>
            <div className="pt-2 text-xs font-semibold text-amber-400 flex items-center gap-1 group-hover:translate-x-1 transition-transform">
              <span>Try Hybrid Mode</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </div>
          </Link>
        </div>
      </section>

      {/* Featured Movie Catalog Preview */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-slate-800 pb-4">
          <div>
            <h2 className="text-2xl font-bold text-slate-100 flex items-center gap-2">
              <TrendingUp className="w-6 h-6 text-amber-400" />
              Featured Catalog Movies
            </h2>
            <p className="text-xs text-slate-400">Popular movies from MovieLens dataset</p>
          </div>
          <Link
            href="/movies"
            className="text-xs font-semibold text-amber-400 hover:text-amber-300 flex items-center gap-1 transition-colors"
          >
            <span>Browse Catalog</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        {isLoadingMovies ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {featuredMovies.map((movie) => (
              <MovieCard key={movie.movie_id} movie={movie} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
