"use client";

import React, { useState, useEffect, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  User as UserIcon,
  Star,
  History,
  Sparkles,
  Trash2,
  ArrowLeft,
  AlertCircle,
  Calendar,
  Clock,
  Film,
} from "lucide-react";
import { getUser } from "@/lib/api/users";
import { getUserRatings, deleteUserRating, upsertRating } from "@/lib/api/ratings";
import { getUserRecommendationHistory } from "@/lib/api/users";
import { User } from "@/types/user";
import { Rating } from "@/types/rating";
import { RecommendationHistoryItem } from "@/types/recommendation";
import { StarRating } from "@/components/ratings/StarRating";
import { ConfirmationModal } from "@/components/ui/ConfirmationModal";
import { useUser } from "@/context/UserContext";
import { useToast } from "@/context/ToastContext";

export default function UserProfilePage() {
  const params = useParams();
  const router = useRouter();
  const userId = Number(params.userId);

  const { activeUser, setActiveUser } = useUser();
  const { showToast } = useToast();

  const [user, setUser] = useState<User | null>(null);
  const [activeTab, setActiveTab] = useState<"ratings" | "history">("ratings");

  // Ratings State
  const [ratings, setRatings] = useState<Rating[]>([]);
  const [ratingsTotal, setRatingsTotal] = useState(0);
  const [isLoadingRatings, setIsLoadingRatings] = useState(true);

  // History State
  const [history, setHistory] = useState<RecommendationHistoryItem[]>([]);
  const [historyTotal, setHistoryTotal] = useState(0);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);

  const [isLoadingUser, setIsLoadingUser] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Delete Rating modal state
  const [deleteRatingMovieId, setDeleteRatingMovieId] = useState<number | null>(null);
  const [isDeletingRating, setIsDeletingRating] = useState(false);

  const fetchUserData = useCallback(async () => {
    if (!userId || isNaN(userId)) {
      setError("Invalid User ID.");
      setIsLoadingUser(false);
      return;
    }

    try {
      setIsLoadingUser(true);
      setError(null);
      const userData = await getUser(userId);
      setUser(userData);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "User not found.";
      setError(msg);
    } finally {
      setIsLoadingUser(false);
    }
  }, [userId]);

  const fetchRatings = useCallback(async () => {
    if (!userId) return;
    try {
      setIsLoadingRatings(true);
      const res = await getUserRatings(userId, 50, 0);
      setRatings(res.items);
      setRatingsTotal(res.total);
    } catch {
      setRatings([]);
    } finally {
      setIsLoadingRatings(false);
    }
  }, [userId]);

  const fetchHistory = useCallback(async () => {
    if (!userId) return;
    try {
      setIsLoadingHistory(true);
      const res = await getUserRecommendationHistory(userId, 50, 0);
      setHistory(res.items);
      setHistoryTotal(res.total);
    } catch {
      setHistory([]);
    } finally {
      setIsLoadingHistory(false);
    }
  }, [userId]);

  useEffect(() => {
    fetchUserData();
    fetchRatings();
    fetchHistory();
  }, [fetchUserData, fetchRatings, fetchHistory]);

  const handleUpdateRating = async (movieId: number, newVal: number) => {
    try {
      await upsertRating({
        user_id: userId,
        movie_id: movieId,
        rating: newVal,
      });
      showToast(`Updated rating for Movie #${movieId} to ${newVal.toFixed(1)} stars`, "success");
      fetchRatings();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to update rating.";
      showToast(msg, "error");
    }
  };

  const handleDeleteRatingConfirm = async () => {
    if (deleteRatingMovieId === null) return;
    try {
      setIsDeletingRating(true);
      await deleteUserRating(userId, deleteRatingMovieId);
      showToast(`Rating for Movie #${deleteRatingMovieId} deleted.`, "success");
      setDeleteRatingMovieId(null);
      fetchRatings();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to delete rating.";
      showToast(msg, "error");
    } finally {
      setIsDeletingRating(false);
    }
  };

  if (isLoadingUser) {
    return (
      <div className="space-y-6 max-w-4xl mx-auto">
        <div className="h-44 bg-slate-900/80 rounded-3xl animate-pulse" />
        <div className="h-64 bg-slate-900/60 rounded-3xl animate-pulse" />
      </div>
    );
  }

  if (error || !user) {
    return (
      <div className="max-w-xl mx-auto py-16 text-center space-y-4">
        <div className="p-4 rounded-2xl bg-rose-500/10 border border-rose-500/20 text-rose-300 text-sm flex items-center justify-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0" />
          <span>{error || "User not found"}</span>
        </div>
        <Link
          href="/users"
          className="inline-flex items-center gap-2 text-xs font-semibold text-amber-400 hover:underline"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to User Management</span>
        </Link>
      </div>
    );
  }

  return (
    <div className="space-y-8 max-w-5xl mx-auto">
      {/* Back button */}
      <div>
        <Link
          href="/users"
          className="inline-flex items-center gap-2 text-xs font-semibold text-slate-400 hover:text-slate-100 transition-colors"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to Users</span>
        </Link>
      </div>

      {/* User Header Profile Card */}
      <div className="p-8 rounded-3xl bg-gradient-to-br from-slate-900 via-slate-900 to-slate-950 border border-slate-800 shadow-2xl flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div className="flex items-center gap-4">
          <div className="w-16 h-16 rounded-2xl bg-amber-500/10 border border-amber-500/30 flex items-center justify-center text-amber-400 shrink-0">
            <UserIcon className="w-8 h-8" />
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-black text-slate-100">{user.username}</h1>
              <span className="text-xs font-mono text-amber-400 bg-amber-500/10 border border-amber-500/20 px-2.5 py-0.5 rounded-full">
                User ID #{user.id}
              </span>
            </div>
            <p className="text-xs text-slate-400">{user.email}</p>
            <p className="text-[11px] text-slate-500 flex items-center gap-1.5 pt-1">
              <Calendar className="w-3.5 h-3.5" />
              <span>Joined {new Date(user.created_at).toLocaleDateString()}</span>
            </p>
          </div>
        </div>

        {/* Shortcuts & Actions */}
        <div className="flex flex-wrap items-center gap-3">
          {activeUser?.id === user.id ? (
            <span className="text-xs font-bold text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 px-3 py-1.5 rounded-xl">
              Currently Active Profile
            </span>
          ) : (
            <button
              onClick={() => {
                setActiveUser(user);
                showToast(`Set '${user.username}' as active profile!`, "success");
              }}
              className="text-xs font-semibold bg-amber-500 text-slate-950 hover:bg-amber-400 px-4 py-2 rounded-xl transition-all shadow-lg shadow-amber-500/20"
            >
              Set as Active Profile
            </button>
          )}

          <Link
            href={`/recommendations?mode=hybrid&user_id=${user.id}`}
            className="text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 px-4 py-2 rounded-xl transition-all flex items-center gap-1.5"
          >
            <Sparkles className="w-4 h-4 text-amber-400" />
            <span>Get Recommendations</span>
          </Link>
        </div>
      </div>

      {/* Tabs Switcher */}
      <div className="flex items-center gap-2 border-b border-slate-800 pb-2">
        <button
          onClick={() => setActiveTab("ratings")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
            activeTab === "ratings"
              ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <Star className="w-4 h-4" />
          <span>User Ratings ({ratingsTotal})</span>
        </button>

        <button
          onClick={() => setActiveTab("history")}
          className={`flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-bold transition-all ${
            activeTab === "history"
              ? "bg-amber-500/15 text-amber-400 border border-amber-500/30"
              : "text-slate-400 hover:text-slate-200"
          }`}
        >
          <History className="w-4 h-4" />
          <span>Recommendation History ({historyTotal})</span>
        </button>
      </div>

      {/* Tab 1: User Ratings List */}
      {activeTab === "ratings" && (
        <div className="space-y-4">
          {isLoadingRatings ? (
            <div className="space-y-3">
              <div className="h-16 bg-slate-900/60 rounded-2xl animate-pulse" />
              <div className="h-16 bg-slate-900/60 rounded-2xl animate-pulse" />
            </div>
          ) : ratings.length > 0 ? (
            <div className="space-y-3">
              {ratings.map((r) => (
                <div
                  key={r.id}
                  className="p-4 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-between gap-4 hover:border-slate-700 transition-colors"
                >
                  <div className="flex items-center gap-4">
                    <div className="p-2.5 rounded-xl bg-slate-950 border border-slate-800 text-amber-400">
                      <Film className="w-5 h-5" />
                    </div>
                    <div>
                      <Link
                        href={`/movies/${r.movie_id}`}
                        className="font-bold text-sm text-slate-100 hover:text-amber-400 transition-colors"
                      >
                        Movie #{r.movie_id}
                      </Link>
                      <p className="text-[11px] text-slate-500 flex items-center gap-1 mt-0.5">
                        <Clock className="w-3 h-3" />
                        <span>Rated on {new Date(r.created_at).toLocaleDateString()}</span>
                      </p>
                    </div>
                  </div>

                  {/* Rating Controls & Delete */}
                  <div className="flex items-center gap-4">
                    <StarRating
                      value={r.rating}
                      onChange={(newVal) => handleUpdateRating(r.movie_id, newVal)}
                      size="md"
                    />

                    <button
                      onClick={() => setDeleteRatingMovieId(r.movie_id)}
                      className="p-2 rounded-xl text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 transition-colors"
                      title="Delete Rating"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-12 text-center space-y-3 bg-slate-900/40 rounded-3xl border border-slate-800">
              <Star className="w-12 h-12 text-slate-600 mx-auto" />
              <h3 className="text-base font-bold text-slate-300">No Movie Ratings Found</h3>
              <p className="text-xs text-slate-400 max-w-sm mx-auto">
                This user has not rated any movies yet. Browse movies to add ratings.
              </p>
              <Link
                href="/movies"
                className="inline-block text-xs font-bold bg-amber-500 text-slate-950 px-4 py-2 rounded-xl"
              >
                Browse Movie Catalog
              </Link>
            </div>
          )}
        </div>
      )}

      {/* Tab 2: Recommendation History */}
      {activeTab === "history" && (
        <div className="space-y-4">
          {isLoadingHistory ? (
            <div className="space-y-3">
              <div className="h-24 bg-slate-900/60 rounded-2xl animate-pulse" />
            </div>
          ) : history.length > 0 ? (
            <div className="space-y-4">
              {history.map((item) => (
                <div
                  key={item.id}
                  className="p-5 rounded-2xl bg-slate-900 border border-slate-800 space-y-3"
                >
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800/80 pb-3">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-xs text-slate-100 uppercase tracking-wider bg-amber-500/10 text-amber-400 border border-amber-500/20 px-2.5 py-0.5 rounded-md">
                        {item.model}
                      </span>
                      <span className="text-xs text-slate-400">
                        Top-{item.top_k} Results
                      </span>
                    </div>
                    <span className="text-[11px] text-slate-500 flex items-center gap-1">
                      <Clock className="w-3.5 h-3.5" />
                      <span>{new Date(item.created_at).toLocaleString()}</span>
                    </span>
                  </div>

                  {/* Weights info if hybrid */}
                  {item.model === "hybrid" && (
                    <div className="text-xs text-slate-400 flex gap-4">
                      <span>Content Weight: <strong>{((item.content_weight || 0.5) * 100).toFixed(0)}%</strong></span>
                      <span>Collaborative Weight: <strong>{((item.collaborative_weight || 0.5) * 100).toFixed(0)}%</strong></span>
                    </div>
                  )}

                  {/* Recommendations list items */}
                  <div className="flex flex-wrap gap-2 pt-1">
                    {Array.isArray(item.recommendations) &&
                      item.recommendations.map((rec, idx) => (
                        <Link
                          key={idx}
                          href={`/movies/${rec.movie_id}`}
                          className="text-xs bg-slate-950 hover:bg-slate-800 border border-slate-800 text-slate-300 px-3 py-1.5 rounded-xl transition-colors flex items-center gap-1.5"
                        >
                          <span className="font-bold text-slate-100">{rec.title}</span>
                          <span className="text-[10px] text-amber-400 font-mono">
                            {(rec.score * 100).toFixed(0)}%
                          </span>
                        </Link>
                      ))}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-12 text-center space-y-3 bg-slate-900/40 rounded-3xl border border-slate-800">
              <History className="w-12 h-12 text-slate-600 mx-auto" />
              <h3 className="text-base font-bold text-slate-300">No Recommendation History</h3>
              <p className="text-xs text-slate-400 max-w-sm mx-auto">
                No logged recommendation requests found for this user.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Delete Rating Confirmation Modal */}
      <ConfirmationModal
        isOpen={deleteRatingMovieId !== null}
        title="Delete Movie Rating"
        message={`Are you sure you want to delete your rating for Movie #${deleteRatingMovieId}?`}
        confirmLabel="Delete Rating"
        onConfirm={handleDeleteRatingConfirm}
        onCancel={() => setDeleteRatingMovieId(null)}
        isConfirming={isDeletingRating}
      />
    </div>
  );
}
