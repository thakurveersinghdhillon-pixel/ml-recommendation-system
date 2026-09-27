"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Film, Star, Sparkles, Tag, ArrowRight } from "lucide-react";
import { Movie } from "@/types/movie";
import { RecommendationItem } from "@/types/recommendation";
import { StarRating } from "@/components/ratings/StarRating";
import { useUser } from "@/context/UserContext";
import { useToast } from "@/context/ToastContext";
import { upsertRating } from "@/lib/api/ratings";

interface MovieCardProps {
  movie: Movie | RecommendationItem;
  userRating?: number;
  onRatingUpdated?: () => void;
}

export const MovieCard: React.FC<MovieCardProps> = ({
  movie,
  userRating,
  onRatingUpdated,
}) => {
  const { activeUser } = useUser();
  const { showToast } = useToast();
  const [rating, setRating] = useState<number>(userRating || 0);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const movieId = "movie_id" in movie ? movie.movie_id : (movie as RecommendationItem).movie_id;
  const score = "score" in movie ? (movie as RecommendationItem).score : undefined;
  const explanation = "explanation" in movie ? (movie as RecommendationItem).explanation : undefined;
  const year = "year" in movie ? (movie as Movie).year : undefined;

  const handleRatingChange = async (newVal: number) => {
    if (!activeUser) {
      showToast("Please select or create a user profile to submit ratings.", "info");
      return;
    }

    try {
      setIsSubmitting(true);
      setRating(newVal);
      await upsertRating({
        user_id: activeUser.id,
        movie_id: movieId,
        rating: newVal,
      });
      showToast(`Saved ${newVal.toFixed(1)} star rating for ${movie.title}`, "success");
      if (onRatingUpdated) onRatingUpdated();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to save rating.";
      showToast(msg, "error");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Generate deterministic gradient background for local placeholder artwork
  const getGradientColor = (id: number) => {
    const gradients = [
      "from-slate-900 via-indigo-950 to-slate-950",
      "from-slate-900 via-purple-950 to-slate-950",
      "from-slate-900 via-emerald-950 to-slate-950",
      "from-slate-900 via-amber-950 to-slate-950",
      "from-slate-900 via-rose-950 to-slate-950",
    ];
    return gradients[id % gradients.length];
  };

  return (
    <div className="group relative flex flex-col bg-slate-900/90 rounded-2xl border border-slate-800/80 hover:border-amber-500/40 transition-all duration-300 overflow-hidden shadow-lg hover:shadow-2xl hover:-translate-y-1">
      {/* Placeholder Poster Artwork Container */}
      <div className={`relative h-44 w-full bg-gradient-to-br ${getGradientColor(movieId)} flex flex-col justify-between p-4 border-b border-slate-800/60`}>
        <div className="flex items-center justify-between w-full">
          <span className="text-[10px] font-mono tracking-widest text-slate-400 bg-slate-950/60 px-2 py-0.5 rounded-full border border-slate-800">
            ID #{movieId}
          </span>
          {score !== undefined && (
            <div className="flex items-center gap-1 text-xs font-bold bg-amber-500/20 text-amber-300 border border-amber-500/40 px-2.5 py-0.5 rounded-full">
              <Sparkles className="w-3 h-3" />
              <span>{(score * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>

        <div className="my-auto">
          <Film className="w-10 h-10 text-slate-700/60 group-hover:text-amber-500/40 transition-colors mx-auto mb-1" />
          <h3 className="font-bold text-sm text-center text-slate-100 line-clamp-2 px-2 group-hover:text-amber-300 transition-colors">
            {movie.title}
          </h3>
        </div>

        {year && (
          <div className="self-center">
            <span className="text-[11px] font-medium text-slate-400 bg-slate-950/40 px-2 py-0.5 rounded-md">
              {year}
            </span>
          </div>
        )}
      </div>

      {/* Card Content Body */}
      <div className="p-4 flex flex-col justify-between flex-1 gap-3">
        {/* Genres */}
        <div className="flex flex-wrap gap-1.5">
          {movie.genres && movie.genres.length > 0 ? (
            movie.genres.slice(0, 3).map((genre) => (
              <span
                key={genre}
                className="inline-flex items-center gap-1 text-[11px] font-medium text-slate-300 bg-slate-800/80 px-2 py-0.5 rounded-md"
              >
                <Tag className="w-2.5 h-2.5 text-slate-400" />
                {genre}
              </span>
            ))
          ) : (
            <span className="text-xs text-slate-500">No genres listed</span>
          )}
        </div>

        {/* Explanation Snippet if available */}
        {explanation && (
          <p className="text-[11px] text-slate-400 italic bg-slate-950/50 p-2 rounded-lg border border-slate-800/40 line-clamp-2">
            &quot;{explanation}&quot;
          </p>
        )}


        {/* Footer Actions & Rating */}
        <div className="pt-2 border-t border-slate-800/60 flex items-center justify-between gap-2">
          <div className="flex flex-col">
            <span className="text-[10px] text-slate-500 uppercase tracking-wider">Your Rating</span>
            <StarRating
              value={rating}
              onChange={handleRatingChange}
              readonly={isSubmitting}
              size="sm"
            />
          </div>

          <Link
            href={`/movies/${movieId}`}
            className="p-2 rounded-xl bg-slate-800 hover:bg-amber-500/20 text-slate-300 hover:text-amber-300 border border-slate-700/60 hover:border-amber-500/40 transition-all flex items-center gap-1 text-xs font-medium"
            title="View details"
          >
            <span>Details</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
        </div>
      </div>
    </div>
  );
};
