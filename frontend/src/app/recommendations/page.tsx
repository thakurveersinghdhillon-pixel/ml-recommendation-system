"use client";

import React, { useState, useEffect, useCallback, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  Sparkles,
  Film,
  Users,
  Sliders,
  AlertCircle,
  Info,
  Search,
} from "lucide-react";
import {
  getContentRecommendations,
  getCollaborativeRecommendations,
  getHybridRecommendations,
} from "@/lib/api/recommendations";
import { searchMovies } from "@/lib/api/movies";
import { Movie } from "@/types/movie";
import { RecommendationResponse } from "@/types/recommendation";
import { MovieCard } from "@/components/movies/MovieCard";
import { SkeletonCard } from "@/components/ui/SkeletonCard";
import { useUser } from "@/context/UserContext";

type Mode = "content" | "collaborative" | "hybrid";

function RecommendationsContent() {
  const searchParams = useSearchParams();
  const initialMode = (searchParams.get("mode") as Mode) || "hybrid";
  const initialMovieId = searchParams.get("movie_id")
    ? Number(searchParams.get("movie_id"))
    : 1;

  const { activeUser } = useUser();

  const [mode, setMode] = useState<Mode>(initialMode);
  const [topK, setTopK] = useState<number>(10);

  // Content Mode state
  const [contentMovieId, setContentMovieId] = useState<number>(initialMovieId);
  const [movieQuery, setMovieQuery] = useState("");
  const [movieSearchResults, setMovieSearchResults] = useState<Movie[]>([]);

  // Collaborative & Hybrid State
  const [userId, setUserId] = useState<number>(activeUser ? activeUser.id : 1);
  const [contentWeight, setContentWeight] = useState<number>(0.5);
  const [collaborativeWeight, setCollaborativeWeight] = useState<number>(0.5);

  const [response, setResponse] = useState<RecommendationResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (activeUser) {
      setUserId(activeUser.id);
    }
  }, [activeUser]);

  useEffect(() => {
    if (!movieQuery.trim()) {
      setMovieSearchResults([]);
      return;
    }
    const timer = setTimeout(() => {
      searchMovies(movieQuery, 5)
        .then((res) => setMovieSearchResults(res.movies))
        .catch(() => setMovieSearchResults([]));
    }, 300);
    return () => clearTimeout(timer);
  }, [movieQuery]);

  const fetchRecommendations = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      if (mode === "content") {
        if (!contentMovieId || contentMovieId <= 0) {
          throw new Error("Please enter a valid seed Movie ID.");
        }
        const data = await getContentRecommendations({
          movie_id: contentMovieId,
          top_k: topK,
        });
        setResponse(data);
      } else if (mode === "collaborative") {
        if (!userId || userId <= 0) {
          throw new Error("Please enter a valid User ID.");
        }
        const data = await getCollaborativeRecommendations({
          user_id: userId,
          top_k: topK,
        });
        setResponse(data);
      } else if (mode === "hybrid") {
        const data = await getHybridRecommendations({
          user_id: userId > 0 ? userId : undefined,
          movie_id: contentMovieId > 0 ? contentMovieId : undefined,
          top_k: topK,
          content_weight: contentWeight,
          collaborative_weight: collaborativeWeight,
        });
        setResponse(data);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to fetch recommendations.";
      setError(msg);
    } finally {
      setIsLoading(false);
    }
  }, [mode, contentMovieId, userId, topK, contentWeight, collaborativeWeight]);

  useEffect(() => {
    fetchRecommendations();
  }, [mode, fetchRecommendations]);

  const handleWeightChange = (newContentWeight: number) => {
    const roundedContent = Math.round(newContentWeight * 10) / 10;
    const roundedCollab = Math.round((1.0 - roundedContent) * 10) / 10;
    setContentWeight(roundedContent);
    setCollaborativeWeight(roundedCollab);
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="border-b border-slate-800 pb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-slate-100 flex items-center gap-3">
            <Sparkles className="w-8 h-8 text-amber-400" />
            Recommendation Engine
          </h1>
          <p className="text-xs text-slate-400 mt-1">
            Test Content-Based, Item-Based Collaborative Filtering, and Hybrid recommendation algorithms
          </p>
        </div>

        {/* Mode Switcher Tabs */}
        <div className="flex items-center p-1.5 bg-slate-900 border border-slate-800 rounded-2xl">
          <button
            onClick={() => setMode("content")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              mode === "content"
                ? "bg-amber-500 text-slate-950 shadow-lg shadow-amber-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Film className="w-4 h-4" />
            <span>Content-Based</span>
          </button>
          <button
            onClick={() => setMode("collaborative")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              mode === "collaborative"
                ? "bg-amber-500 text-slate-950 shadow-lg shadow-amber-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Users className="w-4 h-4" />
            <span>Collaborative</span>
          </button>
          <button
            onClick={() => setMode("hybrid")}
            className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
              mode === "hybrid"
                ? "bg-amber-500 text-slate-950 shadow-lg shadow-amber-500/20"
                : "text-slate-400 hover:text-slate-200"
            }`}
          >
            <Sliders className="w-4 h-4" />
            <span>Hybrid</span>
          </button>
        </div>
      </div>

      {/* Control Panel for Active Mode */}
      <div className="p-6 rounded-3xl bg-slate-900/90 border border-slate-800 space-y-6 shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold text-slate-100 uppercase tracking-wider">
              {mode} Recommendation Configuration
            </span>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span>Top-K Results:</span>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="bg-slate-950 border border-slate-800 text-slate-200 rounded-lg px-2.5 py-1 focus:outline-none"
            >
              <option value={5}>5</option>
              <option value={10}>10</option>
              <option value={20}>20</option>
            </select>
          </div>
        </div>

        {/* Inputs per Mode */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {(mode === "content" || mode === "hybrid") && (
            <div className="space-y-2 relative">
              <label className="text-xs font-semibold text-slate-300">
                Seed Movie {mode === "content" ? "(Required)" : "(Optional Context)"}
              </label>
              <div className="flex gap-2">
                <input
                  type="number"
                  value={contentMovieId || ""}
                  onChange={(e) => setContentMovieId(Number(e.target.value))}
                  placeholder="Movie ID (e.g. 1)"
                  className="w-28 bg-slate-950 border border-slate-800 focus:border-amber-500 text-slate-100 px-3 py-2 rounded-xl text-sm font-medium focus:outline-none"
                />
                <div className="relative flex-1">
                  <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
                  <input
                    type="text"
                    value={movieQuery}
                    onChange={(e) => setMovieQuery(e.target.value)}
                    placeholder="Or search title..."
                    className="w-full bg-slate-950 border border-slate-800 focus:border-amber-500 text-slate-100 pl-9 pr-3 py-2 rounded-xl text-xs font-medium focus:outline-none"
                  />
                </div>
              </div>

              {movieSearchResults.length > 0 && (
                <div className="absolute top-full left-0 right-0 z-20 mt-1 bg-slate-950 border border-slate-800 rounded-xl shadow-2xl p-2 space-y-1">
                  {movieSearchResults.map((m) => (
                    <button
                      key={m.movie_id}
                      type="button"
                      onClick={() => {
                        setContentMovieId(m.movie_id);
                        setMovieQuery("");
                        setMovieSearchResults([]);
                      }}
                      className="w-full text-left px-3 py-1.5 rounded-lg hover:bg-slate-900 text-xs text-slate-200 flex justify-between"
                    >
                      <span className="truncate">{m.title}</span>
                      <span className="text-amber-400 font-mono text-[10px]">#{m.movie_id}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {(mode === "collaborative" || mode === "hybrid") && (
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-300">
                Target User ID {mode === "collaborative" ? "(Required)" : "(Optional)"}
              </label>
              <div className="flex gap-2 items-center">
                <input
                  type="number"
                  value={userId || ""}
                  onChange={(e) => setUserId(Number(e.target.value))}
                  placeholder="User ID (e.g. 1)"
                  className="w-full bg-slate-950 border border-slate-800 focus:border-amber-500 text-slate-100 px-3 py-2 rounded-xl text-sm font-medium focus:outline-none"
                />
                {activeUser && (
                  <button
                    type="button"
                    onClick={() => setUserId(activeUser.id)}
                    className="text-xs bg-amber-500/10 text-amber-400 border border-amber-500/30 px-2.5 py-2 rounded-xl hover:bg-amber-500/20 shrink-0 font-medium"
                  >
                    Use Active (#{activeUser.id})
                  </button>
                )}
              </div>
            </div>
          )}

          {mode === "hybrid" && (
            <div className="space-y-2 md:col-span-2 lg:col-span-1">
              <div className="flex justify-between items-center text-xs">
                <span className="font-semibold text-slate-300">Content Weight: {(contentWeight * 100).toFixed(0)}%</span>
                <span className="font-semibold text-slate-300">Collaborative: {(collaborativeWeight * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={contentWeight}
                onChange={(e) => handleWeightChange(Number(e.target.value))}
                className="w-full accent-amber-500 bg-slate-950 rounded-lg h-2 cursor-pointer"
              />
            </div>
          )}
        </div>

        <div className="flex items-center justify-end pt-2">
          <button
            onClick={fetchRecommendations}
            disabled={isLoading}
            className="bg-amber-500 hover:bg-amber-400 text-slate-950 font-bold px-6 py-2.5 rounded-xl text-xs transition-all shadow-lg shadow-amber-500/20 flex items-center gap-2 disabled:opacity-50"
          >
            <Sparkles className="w-4 h-4" />
            <span>{isLoading ? "Generating Recommendations..." : "Generate Recommendations"}</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-xs flex items-center gap-3">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      <div className="space-y-6">
        {response && (
          <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-4 gap-2">
            <div>
              <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-amber-400" />
                Recommended Movies ({response.recommendations.length})
              </h2>
              {response.seed_movie_title && (
                <p className="text-xs text-slate-400 mt-0.5">
                  Seed Movie: <strong className="text-slate-200">{response.seed_movie_title}</strong>
                </p>
              )}
            </div>
            <span className="text-xs font-mono text-slate-400 bg-slate-900 border border-slate-800 px-3 py-1 rounded-xl w-fit">
              Model: {response.model}
            </span>
          </div>
        )}

        {isLoading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {Array.from({ length: 8 }).map((_, i) => (
              <SkeletonCard key={i} />
            ))}
          </div>
        ) : response && response.recommendations.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {response.recommendations.map((rec) => (
              <MovieCard key={rec.movie_id} movie={rec} />
            ))}
          </div>
        ) : !error ? (
          <div className="py-16 text-center space-y-3 bg-slate-900/40 rounded-3xl border border-slate-800">
            <Info className="w-12 h-12 text-slate-600 mx-auto" />
            <h3 className="text-lg font-bold text-slate-300">No Recommendations Available</h3>
            <p className="text-xs text-slate-400 max-w-sm mx-auto">
              Configure parameters above and click &quot;Generate Recommendations&quot; to calculate scores.
            </p>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function RecommendationsPage() {
  return (
    <Suspense fallback={<div className="py-12 text-center text-slate-400">Loading Recommender...</div>}>
      <RecommendationsContent />
    </Suspense>
  );
}
