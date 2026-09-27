import { describe, it, expect, vi, beforeEach } from "vitest";
import { searchMovies, getMovieDetails } from "../lib/api/movies";
import { getContentRecommendations, getHybridRecommendations } from "../lib/api/recommendations";
import { createUser, getUser } from "../lib/api/users";
import { upsertRating } from "../lib/api/ratings";
import { CustomApiError } from "../types/common";

global.fetch = vi.fn();

describe("Frontend API Client", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  it("searchMovies fetches movies correctly", async () => {
    const mockResponse = {
      query: "toy",
      total_matches: 1,
      limit: 10,
      movies: [{ movie_id: 1, title: "Toy Story (1995)", genres: ["Animation"] }],
    };

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockResponse,
    });

    const data = await searchMovies("toy", 10);
    expect(data.movies.length).toBe(1);
    expect(data.movies[0].title).toBe("Toy Story (1995)");
  });

  it("getMovieDetails returns movie info", async () => {
    const mockMovie = { movie_id: 1, title: "Toy Story (1995)", genres: ["Animation"], year: 1995 };
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockMovie,
    });

    const data = await getMovieDetails(1);
    expect(data.movie_id).toBe(1);
    expect(data.title).toBe("Toy Story (1995)");
  });

  it("getContentRecommendations sends POST request", async () => {
    const mockRecs = {
      model: "content_based",
      seed_movie_id: 1,
      seed_movie_title: "Toy Story (1995)",
      recommendations: [{ movie_id: 3114, title: "Toy Story 2 (1999)", genres: ["Animation"], score: 0.75 }],
    };

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockRecs,
    });

    const data = await getContentRecommendations({ movie_id: 1, top_k: 5 });
    expect(data.model).toBe("content_based");
    expect(data.recommendations.length).toBe(1);
  });

  it("createUser handles 409 Conflict error payload", async () => {
    const mockError = {
      error: { code: "DUPLICATE_USER", message: "User already exists." },
    };

    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 409,
      statusText: "Conflict",
      json: async () => mockError,
    });

    await expect(createUser({ username: "alice", email: "alice@example.com" })).rejects.toThrow(CustomApiError);
  });

  it("upsertRating handles rating creation", async () => {
    const mockRating = { id: 1, user_id: 1, movie_id: 10, rating: 4.5, created_at: "", updated_at: "" };
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 201,
      json: async () => mockRating,
    });

    const res = await upsertRating({ user_id: 1, movie_id: 10, rating: 4.5 });
    expect(res.rating).toBe(4.5);
  });
});
