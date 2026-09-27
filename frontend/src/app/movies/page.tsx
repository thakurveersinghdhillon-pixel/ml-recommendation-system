"use client";

import React, { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { Search, Film, AlertCircle, X, Sparkles } from "lucide-react";
import { searchMovies } from "@/lib/api/movies";
import { Movie } from "@/types/movie";
import { MovieCard } from "@/components/movies/MovieCard";
import { SkeletonCard } from "@/components/ui/SkeletonCard";

function MoviesContent() {
  const searchParams = useSearchParams();
  const initialQuery = searchParams.get("q") || "toy";

  const [query, setQuery] = useState(initialQuery);
  const [debouncedQuery, setDebouncedQuery] = useState(initialQuery);
  const [movies, setMovies] = useState<Movie[]>([]);
  const [totalMatches, setTotalMatches] = useState(0);
  const [limit, setLimit] = useState(12);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 300);
    return () => clearTimeout(handler);
  }, [query]);

  const fetchMovies = useCallback(async (searchQ: string, searchLimit: number) => {
    if (!searchQ) {
      setMovies([]);
      setTotalMatches(0);
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      const res = await searchMovies(searchQ, searchLimit);
      setMovies(res.movies);
      setTotalMatches(res.total_matches);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to search movies.";
      setError(msg);
      setMovies([]);
      setTotalMatches(0);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMovies(debouncedQuery, limit);
  }, [debouncedQuery, limit, fetchMovies]);

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-slate-100 flex items-center gap-3">
            <Film className="w-8 h-8 text-amber-400" />
            Movie Catalog Discovery
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Search 9,742 catalog movies from MovieLens dataset using live FastAPI backend
          </p>
        </div>

        {/* Debounced Search Bar */}
        <div className="relative max-w-md w-full">
          <Search className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Type movie title (e.g. Star, Toy, Matrix)..."
            className="w-full bg-slate-900 border border-slate-700/80 focus:border-amber-500 text-slate-100 placeholder-slate-500 pl-10 pr-9 py-2.5 rounded-xl text-sm font-medium focus:outline-none focus:ring-2 focus:ring-amber-500/20 transition-all"
          />
          {query && (
            <button
              onClick={() => setQuery("")}
              className="absolute right-3 top-3 text-slate-400 hover:text-slate-200"
            >
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Results Header Info */}
      <div className="flex items-center justify-between text-xs text-slate-400">
        <div>
          {debouncedQuery ? (
            <span>
              Showing <strong className="text-slate-200">{movies.length}</strong> of{" "}
              <strong className="text-amber-400">{totalMatches}</strong> matches for &quot;
              <strong className="text-slate-200">{debouncedQuery}</strong>&quot;
            </span>
          ) : (
            <span>Enter a search term to discover catalog movies</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span>Results Limit:</span>
          <select
            value={limit}
            onChange={(e) => setLimit(Number(e.target.value))}
            className="bg-slate-900 border border-slate-800 text-slate-200 rounded-lg px-2 py-1 text-xs focus:outline-none"
          >
            <option value={12}>12</option>
            <option value={24}>24</option>
            <option value={48}>48</option>
          </select>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Movie Grid */}
      {isLoading ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {Array.from({ length: 8 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : movies.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
          {movies.map((movie) => (
            <MovieCard key={movie.movie_id} movie={movie} />
          ))}
        </div>
      ) : (
        <div className="py-16 text-center space-y-3 bg-slate-900/40 rounded-3xl border border-slate-800">
          <Sparkles className="w-12 h-12 text-slate-600 mx-auto" />
          <h3 className="text-lg font-bold text-slate-300">No Movies Found</h3>
          <p className="text-xs text-slate-400 max-w-sm mx-auto">
            Try searching for common titles like &quot;Toy Story&quot;, &quot;Star Wars&quot;, &quot;Jurassic&quot;, or &quot;Matrix&quot;.
          </p>
        </div>
      )}
    </div>
  );
}

export default function MoviesPage() {
  return (
    <Suspense fallback={<div className="py-12 text-center text-slate-400">Loading catalog...</div>}>
      <MoviesContent />
    </Suspense>
  );
}
