"use client";

import React, { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  Film,
  Tag,
  Star,
  Sparkles,
  ArrowLeft,
  AlertCircle,
  Calendar,
  Share2,
} from "lucide-react";
import { getMovieDetails } from "@/lib/api/movies";
import { getContentRecommendations } from "@/lib/api/recommendations";
import { getUserMovieRating, upsertRating } from "@/lib/api/ratings";
import { Movie } from "@/types/movie";
import { RecommendationItem } from "@/types/recommendation";
import { StarRating } from "@/components/ratings/StarRating";
import { MovieCard } from "@/components/movies/MovieCard";
import { SkeletonCard } from "@/components/ui/SkeletonCard";
import { useUser } from "@/context/UserContext";
import { useToast } from "@/context/ToastContext";

export default function MovieDetailPage() {
  const params = useParams();
  const router = useRouter();
  const { activeUser } = useUser();
  const { showToast } = useToast();

  const movieId = Number(params.movieId);

  const [movie, setMovie] = useState<Movie | null>(null);
  const [userRating, setUserRating] = useState<number>(0);
  const [similarMovies, setSimilarMovies] = useState<RecommendationItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isLoadingSimilar, setIsLoadingSimilar] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isSubmittingRating, setIsSubmittingRating] = useState(false);

  useEffect(() => {
    if (!movieId || isNaN(movieId)) {
      setError("Invalid Movie ID.");
      setIsLoading(false);
      return;
    }

    let isMounted = true;
    setIsLoading(true);
    setError(null);

    getMovieDetails(movieId)
      .then((data) => {
        if (isMounted) setMovie(data);
      })
      .catch((err) => {
        if (isMounted) setError(err.message || "Movie not found in catalog.");
      })
      .finally(() => {
        if (isMounted) setIsLoading(false);
      });

    // Fetch similar movies via Content-Based API
    getContentRecommendations({ movie_id: movieId, top_k: 6 })
      .then((res) => {
        if (isMounted) setSimilarMovies(res.recommendations);
      })
      .catch(() => {})
      .finally(() => {
        if (isMounted) setIsLoadingSimilar(false);
      });

    // Check existing user rating if active user exists
    if (activeUser) {
      getUserMovieRating(activeUser.id, movieId)
        .then((r) => {
          if (isMounted) setUserRating(r.rating);
        })
        .catch(() => {
          if (isMounted) setUserRating(0);
        });
    }

    return () => {
      isMounted = false;
    };
  }, [movieId, activeUser]);

  const handleRatingSubmit = async (newRating: number) => {
    if (!activeUser) {
      showToast("Please select or create a user profile to rate movies.", "info");
      router.push("/users");
      return;
    }

    try {
      setIsSubmittingRating(true);
      setUserRating(newRating);
      await upsertRating({
        user_id: activeUser.id,
        movie_id: movieId,
        rating: newRating,
      });
      showToast(`Successfully rated ${newRating.toFixed(1)} stars!`, "success");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to save rating.";
      showToast(msg, "error");
    } finally {
      setIsSubmittingRating(false);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-8 max-w-4xl mx-auto">
        <div className="h-64 bg-slate-900/80 rounded-3xl animate-pulse" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  if (error || !movie) {
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-4">
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center justify-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error || "Movie not found"}</span>
        </div>
        <Link
          href="/movies"
          className="inline-flex items-center gap-2 text-xs font-semibold text-amber-400 hover:underline"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Movie Discovery</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-12 max-w-5xl mx-auto">
      {/* Back button */}
      <div>
        <Link
          href="/movies"
          className="inline-flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-slate-100 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Movies</span>
        </Link>
      </div>

      {/* Hero Banner Detail Card */}
      <div className="relative rounded-3xl bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-slate-800 p-8 sm:p-10 shadow-2xl overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-amber-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col md:flex-row gap-8 items-start">
          {/* Visual Poster Frame */}
          <div className="w-full md:w-56 h-72 rounded-2xl bg-gradient-to-tr from-slate-950 via-slate-900 to-indigo-950 border border-slate-700/60 flex flex-col items-center justify-center p-6 text-center shrink-0 shadow-xl">
            <Film className="w-16 h-16 text-amber-500/40 mb-3" />
            <span className="font-bold text-sm text-slate-200 line-clamp-3">
              {movie.title}
            </span>
            <span className="text-[10px] text-slate-400 mt-2 font-mono">
              ID #{movie.movie_id}
            </span>
          </div>

          {/* Metadata Information */}
          <div className="flex-1 space-y-6">
            <div>
              <div className="flex items-center gap-2 text-xs text-amber-400 font-semibold mb-2">
                <Sparkles className="w-4 h-4" />
                <span>Catalog Movie Details</span>
              </div>
              <h1 className="text-3xl sm:text-4xl font-black text-slate-100 tracking-tight">
                {movie.title}
              </h1>

              {movie.year && (
                <div className="flex items-center gap-2 text-xs text-slate-400 mt-2">
                  <Calendar className="w-4 h-4 text-slate-500" />
                  <span>Released in {movie.year}</span>
                </div>
              )}
            </div>

            {/* Genres */}
            <div className="space-y-2">
              <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                Genres
              </span>
              <div className="flex flex-wrap gap-2">
                {movie.genres.map((g) => (
                  <span
                    key={g}
                    className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl text-xs font-medium bg-slate-800 text-slate-200 border border-slate-700/60"
                  >
                    <Tag className="w-3 h-3 text-amber-400" />
                    {g}
                  </span>
                ))}
              </div>
            </div>

            {/* User Interactive Rating Card */}
            <div className="p-5 rounded-2xl bg-slate-950/80 border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-200 flex items-center gap-1.5">
                  <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
                  Rate this movie
                </span>
                {activeUser ? (
                  <span className="text-[11px] text-slate-400">
                    Active Profile: <strong>{activeUser.username}</strong>
                  </span>
                ) : (
                  <Link
                    href="/users"
                    className="text-[11px] text-amber-400 hover:underline"
                  >
                    Select User Profile
                  </Link>
                )}
              </div>

              <div className="flex items-center gap-4">
                <StarRating
                  value={userRating}
                  onChange={handleRatingSubmit}
                  readonly={isSubmittingRating}
                  size="lg"
                />
                {userRating > 0 && (
                  <span className="text-xs text-emerald-400 font-semibold bg-emerald-500/10 px-2.5 py-1 rounded-lg border border-emerald-500/20">
                    Rated {userRating.toFixed(1)} ★
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Related / Similar Movies (Content-Based) */}
      <section className="space-y-6">
        <div className="border-b border-slate-800 pb-4 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-slate-100 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-amber-400" />
              Similar Movies (Content-Based TF-IDF)
            </h2>
            <p className="text-xs text-slate-400">
              Calculated using TF-IDF text features and Cosine Similarity
            </p>
          </div>
          <Link
            href={`/recommendations?mode=content&movie_id=${movieId}`}
            className="text-xs font-semibold text-amber-400 hover:underline"
          >
            Explore in Recommender
          </Link>
        </div>

        {isLoadingSimilar ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </div>
        ) : similarMovies.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-6">
            {similarMovies.map((rec) => (
              <MovieCard key={rec.movie_id} movie={rec} />
            ))}
          </div>
        ) : (
          <p className="text-xs text-slate-400 text-center py-8">
            No similar movies returned by Content-Based model.
          </p>
        )}
      </section>
    </div>
  );
}
