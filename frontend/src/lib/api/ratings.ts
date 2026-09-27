import { fetchApi } from "./client";
import { Rating, RatingCreate, RatingListResponse } from "@/types/rating";

export async function upsertRating(data: RatingCreate): Promise<Rating> {
  return fetchApi<Rating>("/ratings", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function getUserRatings(
  userId: number,
  limit: number = 20,
  offset: number = 0
): Promise<RatingListResponse> {
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });
  return fetchApi<RatingListResponse>(`/users/${userId}/ratings?${params.toString()}`);
}

export async function getUserMovieRating(userId: number, movieId: number): Promise<Rating> {
  return fetchApi<Rating>(`/users/${userId}/ratings/${movieId}`);
}

export async function deleteUserRating(userId: number, movieId: number): Promise<void> {
  return fetchApi<void>(`/users/${userId}/ratings/${movieId}`, {
    method: "DELETE",
  });
}
